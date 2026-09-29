#!/usr/bin/env python
"""Đề thi ra TIẾNG VIỆT — nên đưa gì vào encoder? (việc 4.5)

    python -X utf8 thu_nghiem_dich_vi.py                    # base-224, 120 câu KIS
    python -X utf8 thu_nghiem_dich_vi.py --nllb             # thêm nllb-600M
    python -X utf8 thu_nghiem_dich_vi.py --config --encoder theo-config --hop-nhat

VÌ SAO PHẢI ĐO CÁI NÀY
----------------------
Mọi con số KIS đang khoe (base-224 0.2950 · so400m+hợp nhất 0.3750) đều đo trên
cột `text` của dev set, tức bản tiếng Anh **do người gõ tay**. Lúc thi không có
cột đó: BTC phát đề `.txt` tiếng Việt. `nop_bai.py` gặp câu chỉ có tiếng Việt thì
in cảnh báo rồi **dùng luôn bản tiếng Việt** — đo trên Kaggle 17/08 ra KIS
**0.2883** (so400m + hợp nhất) thay vì 0.3750. Tức đường thi thật hôm nay đang
mất ~0.087 mà không ai bù.

Câu hỏi của script này: **máy dịch có lấy lại được phần đó không?**

Bốn đường, cùng dev set, cùng encoder, cùng cách dựng dòng — chỉ khác chuỗi đưa
vào encoder:

    1. EN người      cột `text`        <- TRẦN. Lúc thi KHÔNG có
    2. VI thô        cột `_mo_ta_vi`   <- ĐƯỜNG THI HÔM NAY (nền để so)
    3. opus-mt       VI -> Helsinki-NLP/opus-mt-vi-en -> encoder
    4. nllb-600M     VI -> facebook/nllb-200-distilled-600M (cờ --nllb)

Đọc kết quả:
  * (3) hoặc (4) hơn (2) quá 2·SE  -> nối lớp dịch vào đường thi, ăn điểm thật.
  * (3) ≈ (2)                       -> máy dịch không mua được gì; giữ nguyên,
                                        và phải tính đường người dịch lúc thi.
  * (3) < (2)                       -> dịch làm hỏng; SigLIP2 hiểu tiếng Việt tốt
                                        hơn bản dịch máy.
Khoảng cách còn lại tới (1) là phần chỉ người dịch mới lấy được.

KHÁC GÌ `thu_nghiem_dich.py` (10/08)
------------------------------------
Script cũ chạy thời CLIP ViT-B/32: nó hỏi "máy dịch có thay được người dịch khi
gán nhãn không" và nền của nó là CLIP + mCLIP. Encoder giờ là SigLIP2 (đa ngữ),
nền phải là **VI thô vào chính SigLIP2** — đường mà hệ thống thật sự chạy lúc
thi. Hai câu hỏi khác nhau, đừng lấy kết luận cũ trả lời câu hỏi này.

CÁCH ĐO
-------
Nạp dataset MỘT lần, dựng pipeline MỘT lần, rồi lần lượt đổi chuỗi truy vấn. Dòng
nộp dựng y hệt đường thi (cắt top_k trước rồi mới rải đuôi) nên số ra so thẳng
được với 0.2950 / 0.3750. Delta là GHÉP CẶP theo qid, kèm SE; kết luận chặn theo
2·SE và kiểm chứng bằng cách chia đôi theo VIDEO.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026 import config as cau_hinh
from aic2026.control.query_compiler import compile_query
from aic2026.core import scoring
from aic2026.core import submission as sub
from aic2026.core.dich import NLLBTranslator, OpusMTTranslator, co_dau_tieng_viet
from aic2026.core.offline_eval import load_queries
from aic2026.core.retrieval import rrf_fuse
from aic2026.core.providers import RuleBasedProvider, build_llm
from aic2026.core.viet_lai_truy_van import VietLaiTruyVan

GOC = Path(__file__).resolve().parent
DEV_JSON = GOC / "data" / "dev_set_tu_lam.json"
BAN_DICH = GOC / "out" / "ban_dich_dev_set.json"
_DATA = Path(r"D:\COMP\AIC2026\data cuộc thi")

# Cùng bảng với `thu_nghiem_hop_nhat_encoder.py`: vector ảnh đã có sẵn trên đĩa.
BIEN_THE = {
    "base": {"provider": "siglip2",
             "model_id": "google/siglip2-base-patch16-224",
             "cache_path": "dev_set/query_vectors_siglip2.npz",
             "features": _DATA / "aic2026_siglip2"},
    "so400m": {"provider": "siglip2",
               "model_id": "google/siglip2-so400m-patch14-384",
               "cache_path": "dev_set/query_vectors_siglip2_so400m.npz",
               "features": _DATA / "aic2026_siglip2_so400m"},
}


def _args():
    p = argparse.ArgumentParser(
        description="Đề tiếng Việt: dịch máy có lấy lại được điểm không?")
    p.add_argument("--config", nargs="?", const="", default=None,
                   help="đường dẫn config; `--config` trống = config.yaml")
    p.add_argument("--encoder", default="base",
                   choices=("base", "so400m", "theo-config"),
                   help="mặc định base-224: chạy được trên máy không GPU")
    p.add_argument("--hop-nhat", action="store_true",
                   help="giữ `retrieval.hop_nhat` của config (mặc định TẮT để "
                        "cô lập ảnh hưởng của ngôn ngữ)")
    p.add_argument("--task", default="kis", choices=("kis", "qa", "all"))
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--so", type=int, default=None, help="chỉ đo N câu đầu")
    p.add_argument("--nllb", action="store_true",
                   help="đo thêm nllb-200-distilled-600M (~2,4 GB, chậm hơn)")
    p.add_argument("--to-hop", action="store_true",
                   help="đo thêm mẫu câu 'A photo of' và trung bình vector "
                        "VI+dịch (miễn phí, không gọi mạng)")
    p.add_argument("--llm", action="store_true",
                   help="đo thêm đường LLM VIẾT LẠI câu Việt thành chú thích "
                        "tiếng Anh (TỐN QUOTA: ~1 request/20 câu, có cache)")
    p.add_argument("--llm-so-ban", type=int, default=3,
                   help="số cách diễn đạt xin LLM cho mỗi câu (mặc định 3)")
    p.add_argument("--llm-provider", default="gemini",
                   choices=("gemini", "groq"),
                   help="AI viết lại truy vấn. `groq` (qwen3.6-27b) có free tier "
                        "rộng hơn Gemini nhiều — dùng khi Gemini cạn quota ngày")
    p.add_argument("--llm-model", default=None,
                   help="ép tên model viết lại, bỏ qua model trong config. "
                        "Nên dùng bậc LITE: `gemini-3.5-flash-lite` có RPD 500 "
                        "so với 20 của bậc flash — viết lại là việc CHỮ, không "
                        "cần model mạnh nhất")
    p.add_argument("--llm-lo", type=int, default=10,
                   help="số câu mỗi request viết lại (mặc định 10; lô 20 đã đo "
                        "là quá hạn 45s)")
    return p.parse_args()


NEN = "2. VI thô (đường thi)"
TRAN = "1. EN người (trần)"


class EncoderTraBang:
    """Trả vector ĐÃ TÍNH SẴN cho khoá tổng hợp, còn lại ủy quyền encoder gốc.

    Dùng để đo *trung bình vector* nhiều cách diễn đạt mà không phải sửa
    `MultiChannelRetriever`: tầng retrieval vẫn gọi `encode_text` như thường,
    chỉ có chuỗi truyền vào là khoá giả `@ens-<qid>`.
    """

    def __init__(self, goc, bang):
        self.goc = goc
        self.bang = bang
        self.dim = getattr(goc, "dim", None)

    def encode_text(self, text):
        v = self.bang.get(text)
        return list(v) if v is not None else self.goc.encode_text(text)


def _tim_rrf(p, task: str, cac_ban, top_k: int):
    """Tìm RIÊNG từng cách diễn đạt rồi hợp nhất BẢNG XẾP HẠNG bằng RRF.

    Khác hẳn `_trung_binh`: trung bình vector bóp N câu thành MỘT điểm ở giữa,
    nên đặc trưng riêng của từng cách diễn đạt bị nhoè — đo Kaggle 19/08 thấy nó
    THUA bản đơn ở so400m (0.3233 so với 0.3450). RRF giữ nguyên top hit của
    từng bản và chỉ cộng phiếu theo HẠNG, nên một khung chỉ cần được một bản
    xếp cao là vẫn có cửa, còn khung được nhiều bản cùng bầu thì lên đầu.
    """
    bang, xep = {}, {}
    for j, t in enumerate(cac_ban):
        cands = p.retrieval_service.retrieve_compiled(
            compile_query(task, t), top_k=top_k)
        xep[f"v{j}"] = [c.event.event_id for c in cands]
        for c in cands:
            bang.setdefault(c.event.event_id, c)
    diem = rrf_fuse(xep)
    thu_tu = sorted(diem, key=lambda e: (-diem[e], e))[:top_k]
    return [bang[e] for e in thu_tu]


def _bang_shim(p) -> dict:
    """Bảng vector tổng hợp DÙNG CHUNG cho mọi đường ensemble.

    Gán đè `p.retriever.encoder` hai lần sẽ làm bảng trước biến mất, và khoá giả
    `@llm-...` rơi xuống `encode_text` như một câu bình thường — sai mà không có
    triệu chứng. Nên chỉ bọc MỘT lần rồi cùng ghi vào một bảng.
    """
    if not isinstance(p.retriever.encoder, EncoderTraBang):
        p.retriever.encoder = EncoderTraBang(p.encoder, {})
    return p.retriever.encoder.bang


def _trung_binh(vectors):
    """Chuẩn hoá từng vector rồi cộng — cộng vector THÔ sẽ để câu dài nói to hơn."""
    tong = None
    for v in vectors:
        chuan = math.sqrt(sum(x * x for x in v)) or 1.0
        don_vi = [x / chuan for x in v]
        tong = don_vi if tong is None else [a + b for a, b in zip(tong, don_vi)]
    chuan = math.sqrt(sum(x * x for x in tong)) or 1.0
    return [x / chuan for x in tong]


def _config(args) -> dict:
    c = cau_hinh.load_config(args.config or None)
    if args.encoder != "theo-config":
        b = BIEN_THE[args.encoder]
        enc = c.setdefault("encoder", {})
        enc["provider"] = b["provider"]
        enc["model_id"] = b["model_id"]
        enc["cache_path"] = str(GOC / b["cache_path"])
        enc["live"] = True
        c.setdefault("offline", {})["image_feature_root"] = str(b["features"])
    if not args.hop_nhat:
        # Hợp nhất dùng encoder phụ có checkpoint KHÁC; nếu ép encoder chính về
        # `base` mà vẫn bật hợp nhất thì đường phụ cũng là base -> fuse với chính
        # nó. Tắt để phép so chỉ còn một biến là NGÔN NGỮ.
        hn = (c.get("retrieval") or {}).get("hop_nhat")
        if isinstance(hn, dict):
            hn["enabled"] = False
    return c


KHOA_CHU = {"gemini": ("gemini_vlm", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
            "groq": ("groq_vlm", ("GROQ_API_KEY",))}


def _provider_chu(config, ten: str = "gemini", model: Optional[str] = None):
    """Provider CHỈ-CHỮ để viết lại truy vấn: lấy đúng MỘT mục của config.

    Không dùng `FailoverVLMProvider` ở đây: lớp dự phòng chỉ có `answer_images`.
    Và nếu provider chết thì phải BIẾT, chứ không âm thầm đổi model viết lại —
    đó đúng là cái bẫy đã dính hai lần (việc 2.2 và 3.4). Muốn đổi AI thì nói rõ
    bằng `--llm-provider`.
    """
    kind, khoa = KHOA_CHU[ten]
    muc = [m for m in cau_hinh.provider_vlm_dang_bat(config)
           if m.get("provider") == kind]
    if not muc:
        print(f"❌ Config không có provider `{kind}` đang bật.")
        sys.exit(1)
    # Chỉ đòi khoá của ĐÚNG provider sẽ gọi. `thieu_khoa_api` xét cả chuỗi
    # failover, nên một provider khác trong chuỗi thiếu key sẽ chặn nhầm việc
    # này — mà đường viết lại vốn không dùng failover (xem docstring trên).
    if not any(os.environ.get(k) for k in khoa):
        print(f"❌ Thiếu {' hoặc '.join(khoa)} cho đường viết lại.")
        sys.exit(1)
    cfg = dict(muc[0])
    if model:
        # Đổi model thì PHẢI đổi cả cache_path, nếu không hai model ghi chung
        # một file (khoá cache có `self.model` nên không lẫn đáp án, nhưng file
        # trộn lẫn thì soi lại không ra ai trả câu nào).
        cfg["model"] = model
        cfg["cache_path"] = f"out/vlm_qa_cache_{kind}_{model}.json"
    prov = build_llm({"llm": cfg})
    if not hasattr(prov, "tra_loi_chu"):
        print(f"❌ {type(prov).__name__} không có `tra_loi_chu` (gọi chỉ-chữ).")
        sys.exit(1)
    # `timeout_seconds: 45` của config chỉnh cho lô ẢNH. Viết lại là việc SINH
    # chữ: đo thật 18/08 ~2,4s/câu (6 câu × 3 bản = 14,5s), nên lô 10 câu cần
    # ~25s và lô 20 câu đã đúng 45s -> quá hạn. Nới hạn giờ + cho retry 1 lần.
    prov.timeout_seconds = max(float(prov.timeout_seconds), 180.0)
    # Groq không có `max_retries` (REST một phát, không có lớp retry như SDK
    # Gemini) — `getattr` để không phải rẽ nhánh theo tên provider.
    if hasattr(prov, "max_retries"):
        prov.max_retries = max(int(prov.max_retries), 1)
    return prov


def _tb_se(xs):
    tb = sum(xs) / len(xs)
    if len(xs) < 2:
        return tb, float("inf")
    var = sum((x - tb) ** 2 for x in xs) / (len(xs) - 1)
    return tb, math.sqrt(var / len(xs))


def _rows(cands, thong_tin, top_k):
    """Dựng dòng ĐÚNG như đường thi: cắt top_k trước, rồi rải đuôi."""
    rows = []
    for c in cands[:top_k]:
        vid, kf = thong_tin.get(c.event.event_id, (None, None))
        if vid is not None:
            rows.append({"video_id": vid, "frame_idx": kf})
    return sub.diversify_tail(rows, max_rows=top_k)


def _cham(rows, gt, task):
    """KIS chấm bằng R-Score thật; QA chấm bằng TRẦN ĐỊNH VỊ.

    QA có `answer` do tầng đáp án sinh ra, không liên quan tới ngôn ngữ truy vấn.
    Trộn nó vào đây sẽ làm loãng đúng thứ đang đo, nên câu QA chỉ lấy phần định
    vị — đúng phép đo `tran_dinh_vi` trong `do_diem_qa_dau_cuoi.py`.
    """
    if task == "qa":
        diem = [scoring.r_score_kis(row, gt) for row in rows]
        return sum(max(diem[:k], default=0.0)
                   for k in scoring.DEFAULT_KS) / len(scoring.DEFAULT_KS)
    return scoring.final_score(rows, gt)


def _nap_ban_dich() -> dict:
    if BAN_DICH.is_file():
        try:
            return json.loads(BAN_DICH.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}
    return {}


def _dich(ten: str, translator, cau_vi, kho: dict):
    """Dịch có nhớ trên đĩa: chạy lại không tốn thời gian, và Kaggle dùng lại được."""
    bang = kho.setdefault(ten, {})
    thieu = [t for t in dict.fromkeys(cau_vi) if t not in bang]
    if thieu:
        print(f"  {ten}: dịch {len(thieu)} câu mới ...", flush=True)
        t0 = time.time()
        for i in range(0, len(thieu), 16):
            lo = thieu[i:i + 16]
            for goc, ra in zip(lo, translator.dich(lo)):
                bang[goc] = ra
        giay = time.time() - t0
        print(f"     xong {giay:.1f}s ({giay / len(thieu) * 1000:.0f} ms/câu)")
    else:
        print(f"  {ten}: dùng lại {len(cau_vi)} bản dịch đã lưu")
    return [bang.get(t, t) for t in cau_vi]


def main():
    args = _args()
    if not DEV_JSON.exists():
        print(f"❌ Chưa có dev set: {DEV_JSON}")
        sys.exit(1)

    raw = {q["qid"]: q for q in json.loads(
        DEV_JSON.read_text(encoding="utf-8"))["queries"]}
    cap = [(q, gt) for q, gt in load_queries(DEV_JSON)
           if q.task in ("kis", "qa")
           and (args.task == "all" or q.task == args.task)]
    # TRAKE bị loại: câu của nó là chuỗi nhiều mốc, cần tách rồi dịch từng mốc —
    # một phép đo khác, đừng trộn vào đây.
    thieu_vi = [q.qid for q, _ in cap if not (raw[q.qid].get("_mo_ta_vi") or "").strip()]
    if thieu_vi:
        print(f"❌ {len(thieu_vi)} câu không có `_mo_ta_vi`: {thieu_vi[:5]}")
        sys.exit(1)
    if args.so:
        cap = cap[:args.so]

    config = _config(args)
    print("=" * 78)
    print("  ĐỀ TIẾNG VIỆT — DỊCH MÁY CÓ LẤY LẠI ĐƯỢC ĐIỂM KHÔNG?")
    print("=" * 78)
    for dong in cau_hinh.mo_ta_cau_hinh(config):
        print("  " + dong)
    print(f"\nDev set: {len(cap)} câu ({args.task}) · top-k {args.top_k}")

    en_nguoi = [q.text for q, _ in cap]
    vi_tho = [raw[q.qid]["_mo_ta_vi"].strip() for q, _ in cap]
    # Câu `mo_ta` gõ nhầm sang tiếng Anh sẽ đi thẳng không qua dịch. Đó là hành vi
    # đúng của `DichRoiEncode`, nhưng nó làm loãng phép so nên phải đếm ra.
    khong_dau = [q.qid for (q, _), v in zip(cap, vi_tho)
                 if not co_dau_tieng_viet(v)]
    if khong_dau:
        print(f"  ⚠ {len(khong_dau)}/{len(cap)} câu `_mo_ta_vi` không có dấu tiếng "
              f"Việt (đi thẳng, không qua dịch): {khong_dau[:5]}")

    print("\nDịch:")
    kho = _nap_ban_dich()
    duong = [(TRAN, en_nguoi), (NEN, vi_tho)]
    try:
        duong.append(("3. opus-mt vi-en",
                      _dich("opus-mt", OpusMTTranslator(), vi_tho, kho)))
    except Exception as exc:                                    # noqa: BLE001
        print(f"  ❌ opus-mt không chạy được: {type(exc).__name__}: "
              f"{str(exc)[:160]}")
    if args.nllb:
        try:
            duong.append(("4. nllb-600M",
                          _dich("nllb-600M", NLLBTranslator(), vi_tho, kho)))
        except Exception as exc:                                # noqa: BLE001
            print(f"  ❌ nllb không chạy được: {type(exc).__name__}: "
                  f"{str(exc)[:160]}")
    ban_llm = None
    if args.llm:
        prov = _provider_chu(config, args.llm_provider, args.llm_model)
        bo = VietLaiTruyVan(prov, so_ban=args.llm_so_ban, batch=args.llm_lo,
                            cache_path=str(GOC / "out" / "viet_lai_truy_van.json"))
        print(f"  LLM viết lại: {type(prov).__name__}/{bo.model} · "
              f"{args.llm_so_ban} bản/câu · lô {args.llm_lo} · "
              f"hạn {prov.timeout_seconds:g}s", flush=True)
        t0 = time.time()
        try:
            bang_llm = bo.viet_lai(vi_tho)
        except RuntimeError as exc:
            # Hết quota giữa chừng là chuyện thường với free tier. Cache ghi theo
            # TỪNG LÔ nên phần đã viết lại vẫn còn: chạy lại sau khi quota reset
            # sẽ đi tiếp từ chỗ dừng, không tốn lại request nào.
            het_quota = "429" in str(exc) or "quota" in str(exc).lower()
            print(f"\n❌ Dừng đường LLM sau {bo.requests} request "
                  f"({time.time() - t0:.0f}s): {str(exc)[:200]}")
            if het_quota:
                print("   Đây là giới hạn free tier, không phải lỗi code. Phần "
                      "đã viết lại vẫn nằm trong\n"
                      "   out/viet_lai_truy_van.json — chạy lại lệnh này sau khi "
                      "quota reset để đi tiếp.")
            sys.exit(2)
        # Luôn in AI NÀO thật sự trả lời + số request: hai lần trước hệ thống bị
        # hạ cấp âm thầm mà không có dấu hiệu nào (việc 2.2 và 3.4).
        print(f"     {bo.requests} request · {bo.cache_hits} cache hit · "
              f"{time.time() - t0:.1f}s"
              + (f" · ⚠ {len(bo.thieu)} câu LLM không trả, giữ nguyên bản Việt"
                 if bo.thieu else ""))
        ban_llm = [bang_llm[t] for t in vi_tho]
        duong.append(("8. LLM viết lại", [b[0] for b in ban_llm]))
    BAN_DICH.parent.mkdir(parents=True, exist_ok=True)
    BAN_DICH.write_text(json.dumps(kho, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"  bản dịch đã lưu: {BAN_DICH}")

    print("\n  3 ví dụ:")
    for i in range(min(3, len(cap))):
        for ten, cau in duong:
            print(f"    {ten:22s} {cau[i][:72]}")
        print()

    print("Đang nạp dữ liệu ...", flush=True)
    t = time.time()
    events, emb, _metas, rep = cau_hinh.nap_du_lieu(config)
    phu = cau_hinh.nap_hop_nhat(config, events)
    p = cau_hinh.dung_pipeline(config, events, emb, llm=RuleBasedProvider(),
                               retrieval_phu=phu)
    thong_tin = {e.event_id: (e.video_id, e.keyframe_idx) for e in events}
    print(f"  {rep.events:,d} event / {rep.videos_da_nap} video · "
          f"hợp nhất {len(phu)} encoder phụ · {time.time() - t:.1f}s\n")

    if ban_llm is not None and args.llm_so_ban > 1:
        # Ensemble truy vấn (việc 4.4b) trên chính các cách diễn đạt LLM sinh ra:
        # cùng nội dung, khác từ vựng -> sai ở những chỗ khác nhau.
        bang, khoa_llm, khoa_llm_vi = _bang_shim(p), [], []
        for i, (q, _gt) in enumerate(cap):
            vecs = [p.encoder.encode_text(t) for t in ban_llm[i]]
            k1, k2 = f"@llm-{q.qid}", f"@llmvi-{q.qid}"
            bang[k1] = _trung_binh(vecs)
            bang[k2] = _trung_binh(vecs + [p.encoder.encode_text(vi_tho[i])])
            khoa_llm.append(k1)
            khoa_llm_vi.append(k2)
        # Hai cách dùng CÙNG 3 bản viết lại, khác nhau ở chỗ hợp nhất:
        #   tb(...)  gộp ở tầng VECTOR   -> nhoè, đo được là thua bản đơn
        #   RRF(...) gộp ở tầng HẠNG     -> giữ top hit riêng của từng bản
        duong.append((f"11. RRF(LLM ×{args.llm_so_ban})",
                      [list(b) for b in ban_llm]))
        duong.append((f"12. RRF(LLM ×{args.llm_so_ban}, VI)",
                      [list(b) + [vi_tho[i]] for i, b in enumerate(ban_llm)]))
        duong.append((f"9. tb(LLM ×{args.llm_so_ban})", khoa_llm))
        duong.append((f"10. tb(LLM ×{args.llm_so_ban}, VI)", khoa_llm_vi))

    if args.to_hop:
        # Hai đòn miễn phí, cùng trả lời câu hỏi "đưa gì vào encoder":
        #   * MẪU CÂU: SigLIP2/CLIP học trên alt-text, câu bắt đầu bằng "A photo
        #     of" nằm đúng phân phối huấn luyện hơn câu dịch máy trần trụi.
        #   * TRUNG BÌNH VECTOR: hai cách diễn đạt sai ở hai chỗ khác nhau; trung
        #     bình chúng là ensemble truy vấn (việc 4.4b) mà không tốn API.
        goc = dict(duong)
        may = [ten for ten, _ in duong if ten[0] in "34"]
        if may:
            cau_may = goc[may[0]]
            cau_mau = [f"A photo of {t[0].lower() + t[1:]}" if t else t
                       for t in cau_may]
            duong.append((f"5. mẫu câu + {may[0][3:]}", cau_mau))
            bang, khoa_vi_may, khoa_ba = _bang_shim(p), [], []
            for i, (q, _gt) in enumerate(cap):
                v_vi = p.encoder.encode_text(vi_tho[i])
                v_may = p.encoder.encode_text(cau_may[i])
                v_mau = p.encoder.encode_text(cau_mau[i])
                k2, k3 = f"@ens2-{q.qid}", f"@ens3-{q.qid}"
                bang[k2] = _trung_binh([v_vi, v_may])
                bang[k3] = _trung_binh([v_vi, v_may, v_mau])
                khoa_vi_may.append(k2)
                khoa_ba.append(k3)
            duong.append(("6. tb(VI, dịch)", khoa_vi_may))
            duong.append(("7. tb(VI, dịch, mẫu câu)", khoa_ba))

    ket = {}
    for ten, cau in duong:
        t = time.time()
        diem, r1, r100 = [], [], []
        for (q, gt), text in zip(cap, cau):
            if isinstance(text, str):
                compiled = compile_query(q.task, text)
                cands = p.retrieval_service.retrieve_compiled(
                    compiled, top_k=args.top_k)
            else:
                cands = _tim_rrf(p, q.task, text, args.top_k)
            rows = _rows(cands, thong_tin, args.top_k)
            diem.append(_cham(rows, gt, q.task))
            r1.append(scoring.recall_at_k(rows, gt, 1))
            r100.append(scoring.recall_at_k(rows, gt, args.top_k))
        ket[ten] = (diem, r1, r100)
        print(f"  {ten:22s} xong ({time.time() - t:.0f}s)")

    nen = ket[NEN][0]
    tran = ket[TRAN][0]
    print("\n" + "=" * 78)
    print("  KẾT QUẢ")
    print("=" * 78)
    print(f"  {'đường':24s} {'Final':>7s} {'R@1':>7s} {'R@100':>7s} "
          f"{'Δ vs VI thô':>18s}")
    print("  " + "-" * 74)
    for ten, _ in duong:
        diem, r1, r100 = ket[ten]
        tb, _se = _tb_se(diem)
        n = len(diem)
        if ten == NEN:
            cot = f"{'(nền)':>18s}"
        else:
            d, dse = _tb_se([a - b for a, b in zip(diem, nen)])
            cot = f"{d:>+10.4f}±{dse:.4f}" + ("*" if abs(d) > 2 * dse else " ")
        print(f"  {ten:24s} {tb:7.4f} {sum(r1)/n:6.1%} {sum(r100)/n:6.1%} {cot}")

    print("\n  " + "-" * 74)
    tb_nen, _ = _tb_se(nen)
    tb_tran, _ = _tb_se(tran)
    print(f"  Khoảng cách VI thô -> EN người: {tb_tran - tb_nen:+.4f} "
          f"({(tb_tran - tb_nen) / max(tb_nen, 1e-9):+.0%}) — đây là phần đang mất "
          f"lúc thi.")

    may = [(ten, ket[ten][0]) for ten, _ in duong if ten not in (NEN, TRAN)]
    if not may:
        print("\n  Không có đường máy dịch nào chạy được — chưa kết luận gì.")
        return
    ten_tot, diem_tot = max(may, key=lambda x: sum(x[1]))
    d, dse = _tb_se([a - b for a, b in zip(diem_tot, nen)])

    print("\n" + "=" * 78)
    print("  KẾT LUẬN")
    print("=" * 78)
    for ten, diem in may:
        dd, sse = _tb_se([a - b for a, b in zip(diem, nen)])
        de, sse2 = _tb_se([a - b for a, b in zip(diem, tran)])
        print(f"  {ten:22s} vs VI thô  Δ {dd:+.4f} ± {sse:.4f}  "
              f"KTC95 [{dd - 2*sse:+.4f}, {dd + 2*sse:+.4f}]")
        print(f"  {'':22s} vs EN người Δ {de:+.4f} ± {sse2:.4f}")

    # Chia đôi theo VIDEO (không phải theo câu: hai câu cùng video không độc lập).
    nua_a = [i for i, (_q, gt) in enumerate(cap) if hash(gt.video_id) % 2 == 0]
    nua_b = [i for i in range(len(cap)) if i not in set(nua_a)]
    print(f"\n  KIỂM CHỨNG CHIA THEO VIDEO — {ten_tot}")
    for ten_nua, chi_so in (("nửa A", nua_a), ("nửa B", nua_b)):
        if len(chi_so) < 2:
            continue
        dd, sse = _tb_se([diem_tot[i] - nen[i] for i in chi_so])
        print(f"    {ten_nua} ({len(chi_so)} câu): Δ {dd:+.4f} ± {sse:.4f}"
              + ("  ✅ cùng dấu dương" if dd > 0 else "  ⚠ không dương"))

    print()
    if d > 2 * dse:
        print(f"  ✅ {ten_tot} hơn đường thi hiện tại **{d:+.4f} ± {dse:.4f}** "
              f"(vượt 2·SE).")
        print(f"     -> nối lớp dịch vào `nop_bai.py` cho câu chỉ có bản tiếng "
              f"Việt, rồi")
        print(f"        xác nhận lại bằng so400m + hợp nhất trên Kaggle trước khi "
              f"chốt.")
    elif d < -2 * dse:
        print(f"  ❌ {ten_tot} KÉM hơn VI thô {d:+.4f} ± {dse:.4f} (vượt 2·SE). "
              f"SigLIP2 đọc")
        print(f"     tiếng Việt tốt hơn bản dịch máy — đừng nối lớp dịch.")
    else:
        print(f"  ⚪ {ten_tot} chênh {d:+.4f} ± {dse:.4f} — CHƯA vượt 2·SE "
              f"({2 * dse:.4f}).")
        can = int(math.ceil((2 * dse * math.sqrt(len(cap))
                             / max(abs(d), 1e-9)) ** 2))
        print(f"     Với effect size này cần ~{can} câu để phân giải. Chưa đủ cớ "
              f"đổi đường thi.")
    print("\n  ⚠ 'Chưa vượt 2·SE' KHÔNG phải 'tương đương'. Xem CẬN DƯỚI của KTC "
          "ở trên:")
    print("    nó phải loại trừ được mức mất mát đủ lớn để mình bận tâm.")


if __name__ == "__main__":
    main()
