#!/usr/bin/env python
"""Hợp nhất NHIỀU ENCODER bằng RRF — việc 4.4(a), hướng do 4.1 chỉ ra.

    python -X utf8 thu_nghiem_hop_nhat_encoder.py               # 120 câu KIS
    python -X utf8 thu_nghiem_hop_nhat_encoder.py --pool 1000
    python -X utf8 thu_nghiem_hop_nhat_encoder.py --bien-the so400m,base

VÌ SAO THỬ CÁCH NÀY
-------------------
`chan_doan_qa_truot.py` (4.1) đo được: với các câu trượt nhóm B, keyframe ĐÚNG
nằm ở hạng **106–1.766** trên 177.321 — tức ngay ngoài top-100. Trần nếu truy hồi
sâu tới 500 rồi có reranker hoàn hảo là **90/120 câu KIS** so với 67 hiện tại.

Cách rẻ nhất để mua phần đó là hợp nhất nhiều encoder: **vector ảnh của cả ba
checkpoint đã có sẵn trên đĩa** (so400m 1152 chiều, base-224 768 chiều, CLIP BTC
512 chiều) và cả ba bộ vector truy vấn đều đã cache. Không phải encode lại gì,
không tốn API. Hai encoder độc lập hay sai ở hai chỗ khác nhau, nên RRF của chúng
thường kéo được đúng loại "hạng 300" này lên.

Đây là NGHIÊN CỨU, không phải cấu hình thi: script chỉ in delta ghép cặp ± SE và
chặn theo 2·SE. Chỉ khi vượt mới nối vào `config.example.yaml`.

CÁCH ĐO
-------
Mỗi biến thể chạy riêng (nạp dataset của nó, thả ra rồi mới nạp cái sau), lưu lại
danh sách event_id theo hạng cho từng câu. Sau đó fuse bằng RRF trên HẠNG — không
cộng điểm cosine thô, vì cosine của hai không gian khác nhau không cùng thang.
"""
from __future__ import annotations

import argparse
import copy
import gc
import itertools
import math
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026 import config as cau_hinh
from aic2026.control.query_compiler import compile_query
from aic2026.core import scoring
from aic2026.core import submission as sub
from aic2026.core.offline_eval import load_queries
from aic2026.core.providers import RuleBasedProvider

GOC = Path(__file__).resolve().parent
DEV_JSON = GOC / "data" / "dev_set_tu_lam.json"
_DATA = Path(r"D:\COMP\AIC2026\data cuộc thi")

# Ba bộ vector ảnh đã có sẵn trên đĩa + cache truy vấn tương ứng.
BIEN_THE = {
    "so400m": {"provider": "siglip2",
               "model_id": "google/siglip2-so400m-patch14-384",
               "cache_path": "dev_set/query_vectors_siglip2_so400m.npz",
               "features": _DATA / "aic2026_siglip2_so400m"},
    "base": {"provider": "siglip2",
             "model_id": "google/siglip2-base-patch16-224",
             "cache_path": "dev_set/query_vectors_siglip2.npz",
             "features": _DATA / "aic2026_siglip2"},
    "clip": {"provider": "clip",
             "model_id": "openai/clip-vit-base-patch32",
             "cache_path": "dev_set/query_vectors.npz",
             "features": None},          # dùng clip-features BTC trong bundle nhẹ
}
RRF_K = 60.0


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--task", default="kis", choices=("kis", "qa", "all"))
    p.add_argument("--pool", type=int, default=500,
                   help="độ sâu lấy từ mỗi encoder trước khi fuse")
    p.add_argument("--top-k", type=int, default=100, help="số dòng nộp bài")
    p.add_argument("--bien-the", default="so400m,base,clip")
    return p.parse_args()


def _config_bien_the(goc: dict, ten: str) -> dict:
    c = copy.deepcopy(goc)
    b = BIEN_THE[ten]
    enc = c.setdefault("encoder", {})
    enc["provider"] = b["provider"]
    enc["model_id"] = b["model_id"]
    enc["cache_path"] = str(GOC / b["cache_path"])
    enc["live"] = True
    c.setdefault("offline", {})["image_feature_root"] = (
        str(b["features"]) if b["features"] else None)
    return c


def _hang_theo_cau(config, queries, pool):
    """{qid: [event_id theo hạng]} cho MỘT biến thể, kèm bảng event_id -> row."""
    events, emb, _, rep = cau_hinh.nap_du_lieu(config)
    p = cau_hinh.dung_pipeline(config, events, emb, llm=RuleBasedProvider())
    thong_tin = {e.event_id: (e.video_id, e.keyframe_idx) for e in events}
    ra = {}
    for q, _gt in queries:
        compiled = compile_query(q.task, q.text)
        cands = p.retrieval_service.retrieve_compiled(compiled, top_k=pool)
        ra[q.qid] = [c.event.event_id for c in cands]
    del events, emb, p
    gc.collect()
    return ra, thong_tin, rep


def _fuse(cac_hang, trong_so, qid):
    """RRF CÓ TRỌNG SỐ trên HẠNG cho một câu. Trả [event_id] đã xếp.

    Cộng theo hạng chứ không theo cosine: hai không gian vector khác nhau không
    cùng thang điểm, cộng cosine thô là so hai đơn vị khác nhau. Trọng số để
    encoder mạnh hơn được nói to hơn — so400m một mình đã hơn base 0.045.
    """
    diem = {}
    for ten, w in trong_so.items():
        for hang, eid in enumerate(cac_hang[ten].get(qid, ()), start=1):
            diem[eid] = diem.get(eid, 0.0) + float(w) / (RRF_K + hang)
    return [eid for eid, _ in sorted(diem.items(), key=lambda kv: -kv[1])]


def _rows(eids, thong_tin, top_k):
    """Dựng dòng ĐÚNG như `submission.kis_rows`: cắt top_k trước, rồi rải đuôi.

    Cắt sau khi rải sẽ cho điểm khác (đo được 0.3050 thay vì 0.3400) — không phải
    vì fusion, mà vì đuôi được rải trên nhiều video hơn. Muốn so với đường thi thì
    phải dựng dòng y hệt đường thi.
    """
    rows = []
    for eid in eids[:top_k]:
        vid, kf = thong_tin.get(eid, (None, None))
        if vid is not None:
            rows.append({"video_id": vid, "frame_idx": kf})
    return sub.diversify_tail(rows, max_rows=top_k)


def _tb_se(xs):
    tb = sum(xs) / len(xs)
    if len(xs) < 2:
        return tb, float("inf")
    var = sum((x - tb) ** 2 for x in xs) / (len(xs) - 1)
    return tb, math.sqrt(var / len(xs))


def main():
    args = _args()
    if not DEV_JSON.exists():
        print(f"❌ Chưa có dev set: {DEV_JSON}")
        sys.exit(1)
    ten_bien_the = [x.strip() for x in args.bien_the.split(",") if x.strip()]
    for ten in ten_bien_the:
        if ten not in BIEN_THE:
            print(f"❌ Biến thể không biết: {ten}. Có: {', '.join(BIEN_THE)}")
            sys.exit(1)

    goc = cau_hinh.load_config(args.config)
    queries = [(q, gt) for q, gt in load_queries(DEV_JSON)
               if args.task == "all" or q.task == args.task]
    queries = [(q, gt) for q, gt in queries if q.task in ("kis", "qa")]
    print(f"Dev set: {len(queries)} câu {args.task} · pool {args.pool} · "
          f"biến thể: {', '.join(ten_bien_the)}\n")

    cac_hang, thong_tin = {}, None
    for ten in ten_bien_the:
        c = _config_bien_the(goc, ten)
        feat = (c.get("offline") or {}).get("image_feature_root")
        if feat and not Path(feat).is_dir():
            print(f"  ⚠ bỏ {ten}: không có {feat}")
            continue
        t = time.time()
        try:
            hang, tt, rep = _hang_theo_cau(c, queries, args.pool)
        except Exception as exc:                                # noqa: BLE001
            print(f"  ❌ {ten}: {type(exc).__name__}: {str(exc)[:200]}")
            continue
        cac_hang[ten] = hang
        thong_tin = thong_tin or tt
        print(f"  {ten:8s} {rep.events:,d} event · {time.time() - t:.0f}s")
    if len(cac_hang) < 1:
        print("❌ Không biến thể nào chạy được.")
        sys.exit(1)

    # Mọi tổ hợp trọng số 1, cộng vài phương án CÓ TRỌNG SỐ ưu tiên encoder mạnh.
    nen = ten_bien_the[0]
    co = [x for x in ten_bien_the if x in cac_hang]
    to_hop = []
    for r in range(1, len(co) + 1):
        for combo in itertools.combinations(co, r):
            to_hop.append({ten: 1 for ten in combo})
    khong_trong_so = list(to_hop)
    for w in (2, 3):
        for combo in khong_trong_so:
            if len(combo) > 1 and nen in combo:
                to_hop.append({**combo, nen: w})

    print(f"\n{'tổ hợp':22s} {'Final':>8s} {'R@1':>7s} {'R@100':>7s} "
          f"{'Δ vs nền':>10s} {'paired SE':>10s}")
    print("-" * 70)
    diem_nen = None
    ket = []
    for combo in to_hop:
        eids_theo_cau = {q.qid: _fuse(cac_hang, combo, q.qid) for q, _ in queries}
        diem, r1, r100 = [], [], []
        for q, gt in queries:
            rows = _rows(eids_theo_cau[q.qid], thong_tin, args.top_k)
            diem.append(scoring.final_score(rows, gt))
            r1.append(scoring.recall_at_k(rows, gt, 1))
            r100.append(scoring.recall_at_k(rows, gt, args.top_k))
        la_nen = combo == {nen: 1}
        if la_nen:
            diem_nen = diem
        tb, _ = _tb_se(diem)
        if diem_nen is not None and not la_nen:
            d, dse = _tb_se([a - b for a, b in zip(diem, diem_nen)])
        else:
            d, dse = float("nan"), float("nan")
        nhan = "+".join(f"{k}×{v}" if v != 1 else k for k, v in combo.items())
        ket.append((nhan, tb, d, dse, diem))
        print(f"{nhan:22s} {tb:8.4f} "
              f"{sum(r1) / len(r1):6.1%} {sum(r100) / len(r100):6.1%} "
              + ("" if la_nen else f" {d:+9.4f} {dse:10.4f}"))

    tot = max((x for x in ket if math.isfinite(x[3])),
              key=lambda x: x[1], default=None)
    print("\n  KẾT LUẬN")
    if tot is None:
        print("    Chỉ có một biến thể — không có gì để hợp nhất.")
        return
    ten, tb, d, dse, diem_tot = tot

    # THIÊN LỆCH CHỌN LỌC: đã thử `len(to_hop)` cấu hình rồi chọn cái cao nhất, nên
    # 2·SE trên chính tập đó là quá dễ dãi. Chia theo VIDEO (không phải theo câu:
    # hai câu cùng video không độc lập) rồi xem cấu hình thắng có thắng ở CẢ HAI nửa.
    nua_a = [i for i, (q, gt) in enumerate(queries)
             if hash(gt.video_id) % 2 == 0]
    nua_b = [i for i in range(len(queries)) if i not in set(nua_a)]
    print(f"\n  KIỂM CHỨNG CHIA THEO VIDEO (đã thử {len(to_hop)} cấu hình)")
    for ten_nua, chi_so in (("nửa A", nua_a), ("nửa B", nua_b)):
        if len(chi_so) < 2:
            continue
        dd, sse = _tb_se([diem_tot[i] - diem_nen[i] for i in chi_so])
        print(f"    {ten_nua} ({len(chi_so)} câu): Δ {dd:+.4f} ± {sse:.4f}"
              + ("  ✅ cùng dấu dương" if dd > 0 else "  ⚠ ĐỔI DẤU"))
    if not math.isfinite(dse):
        print(f"    Không tính được SE cho {ten}.")
    elif d > 2 * dse:
        print(f"    ✅ {ten} hơn {nen} **{d:+.4f} ± {dse:.4f}** — vượt 2·SE.")
        print(f"       Nối vào config (fusion nhiều encoder) rồi xác nhận lại trên")
        print(f"       split khác trước khi coi là mặc định.")
    elif d < -2 * dse:
        print(f"    ❌ {ten} KÉM hơn {nen} {d:+.4f} ± {dse:.4f} (vượt 2·SE). "
              f"Đừng hợp nhất.")
    else:
        print(f"    ⚪ {ten} chênh {d:+.4f} ± {dse:.4f} — CHƯA vượt 2·SE "
              f"({2 * dse:.4f}), chưa phân biệt được với nhiễu.")
        print(f"       Với effect size này, cần khoảng "
              f"{int(math.ceil((2 * dse * math.sqrt(len(queries)) / max(abs(d), 1e-9)) ** 2))}"
              f" câu để phân giải.")


if __name__ == "__main__":
    main()
