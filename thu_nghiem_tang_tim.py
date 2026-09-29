#!/usr/bin/env python
"""So các cấu hình TẦNG TÌM trên dev set thật, bằng thống kê ghép cặp.

    python thu_nghiem_tang_tim.py

TRẢ LỜI HAI CÂU ĐANG TREO
-------------------------
B1. Kênh chữ metadata (title + keywords của video) đang TẮT:
    `dataset.load_dataset(..., meta_vao_kenh_chu=False)` là mặc định, nên
    `ocr_text` và `asr_text` rỗng cho cả 177.321 event. `retrieval.py` quảng cáo
    trộn 3 kênh nhưng thực tế chỉ trộn 1. Bật lên có hơn không?

B2/B3. Khử modality gap đang BẬT mặc định trong `do_diem_thuc_te.py` mà chưa ai
    kết luận. Đo trước đó: video-R@100 61.7% -> 68.3% khi bật, nhưng hạng trung
    vị xấu hơn (283 -> 303) và R@100 không đổi.

VÌ SAO PHẢI GHÉP CẶP
--------------------
Dev set 60 câu, điểm nền 0.1233 với SE ±0.0305. Nếu so hai trung bình rời nhau
thì chênh lệch 0.01 chìm hoàn toàn trong sai số đó. So TỪNG CÂU với chính nó
(cùng câu, hai cấu hình) thì phần lớn phương sai "câu này khó" triệt tiêu, SE
của chênh lệch tụt xuống ~±0.007 — đủ để phân biệt.

Cột `Δ` kèm `±SE`. Dấu `*` = |Δ| > 2·SE, tức đo được thật. Không có dấu thì
KHÔNG kết luận được gì, dù con số trông có vẻ hơn.

GIỮ CỐ ĐỊNH: cùng dev set, cùng encoder, cùng cache truy vấn, portfolio TẮT
(để chênh lệch phản ánh tầng TÌM chứ không lẫn cách tiêu slot).
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026.core import scoring
from aic2026.core import submission as sub
from aic2026.core.dataset import load_dataset
from aic2026.core.offline_eval import load_queries
from aic2026.core.providers import CLIPTextEncoder
from aic2026.core.retrieval import MultiChannelRetriever
from aic2026.core.stores import NumpyVectorStore

GOC = os.path.dirname(os.path.abspath(__file__))
DATA = r"D:\COMP\AIC2026\data cuộc thi\aic2026_data_nhe"
DEV_JSON = os.path.join(GOC, "data", "dev_set_tu_lam.json")
CACHE = os.path.join(GOC, "dev_set", "query_vectors.npz")

# (tên, kênh chữ metadata, khử modality gap)
CAU_HINH = [
    ("nền: visual + khử gap",      False, True),
    ("visual, KHÔNG khử gap",      False, False),
    ("+ kênh chữ, khử gap",        True,  True),
    ("+ kênh chữ, không khử gap",  True,  False),
]


def tb_va_se(v):
    n = len(v)
    tb = sum(v) / n
    if n < 2:
        return tb, float("inf")
    var = sum((x - tb) ** 2 for x in v) / (n - 1)
    return tb, math.sqrt(var / n)


def main():
    thieu = [p for p in (DATA, DEV_JSON, CACHE) if not os.path.exists(p)]
    if thieu:
        raise SystemExit("❌ Thiếu:\n" + "\n".join(f"   - {p}" for p in thieu))

    queries = load_queries(DEV_JSON)
    enc = CLIPTextEncoder(cache_path=CACHE, cho_phep_live=True)

    # Nạp dataset hai lần: một lần có metadata trong kênh chữ, một lần không.
    du_lieu = {}
    for co_chu in (False, True):
        print(f"Đang nạp dataset (kênh chữ metadata: "
              f"{'BẬT' if co_chu else 'tắt'}) ...")
        ev, emb, metas, rep = load_dataset(DATA, meta_vao_kenh_chu=co_chu)
        co = sum(1 for e in ev if e.asr_text or e.ocr_text)
        print(f"  {rep.events:,d} event, {co:,d} event có chữ")
        du_lieu[co_chu] = (ev, emb, metas)

    print(f"\nDev set: {len(queries)} câu\n")

    ket = {}
    for ten, co_chu, khu_gap in CAU_HINH:
        ev, emb, metas = du_lieu[co_chu]
        r = MultiChannelRetriever(
            enc, vector_store=NumpyVectorStore(khu_modality_gap=khu_gap))
        r.index(ev, embeddings=emb)

        t = time.time()
        diem, r1, r100, rv100, dong_gop_chu = [], [], [], [], 0
        for q, gt in queries:
            cands = r.retrieve(q.text, top_k=100, fusion="rrf")
            rows = sub.kis_rows(cands)
            diem.append(scoring.final_score(rows, gt))
            r1.append(scoring.recall_at_k(rows, gt, 1, threshold=1.0))
            r100.append(scoring.recall_at_k(rows, gt, 100, threshold=1.0))
            rv100.append(1.0 if any(x["video_id"] == gt.video_id
                                    for x in rows) else 0.0)
            dong_gop_chu += sum(
                1 for c in cands
                if c.modality_scores.get("ocr", 0.0) > 0
                or c.modality_scores.get("asr", 0.0) > 0)
        ket[ten] = dict(diem=diem, r1=r1, r100=r100, rv100=rv100,
                        chu=dong_gop_chu, giay=time.time() - t)
        print(f"  {ten:28s} xong {ket[ten]['giay']:.1f}s")

    nen_ten = CAU_HINH[0][0]
    nen = ket[nen_ten]["diem"]

    print("\n" + "=" * 78)
    print("  KẾT QUẢ  (portfolio TẮT — chênh lệch chỉ do tầng TÌM)")
    print("=" * 78)
    print(f"  {'cấu hình':28s} {'điểm':>7s} {'Δ so nền':>17s} "
          f"{'R@1':>6s} {'R@100':>7s} {'video':>7s}")
    print("  " + "-" * 76)
    for ten, _, _ in CAU_HINH:
        k = ket[ten]
        d, _ = tb_va_se(k["diem"])
        if ten == nen_ten:
            cot_d = f"{'(nền)':>17s}"
        else:
            dd, se = tb_va_se([a - b for a, b in zip(k["diem"], nen)])
            sao = "*" if abs(dd) > 2 * se else " "
            cot_d = f"{dd:>+9.4f}±{se:.4f}{sao}"
        print(f"  {ten:28s} {d:>7.4f} {cot_d} "
              f"{sum(k['r1'])/len(k['r1']):>5.1%} "
              f"{sum(k['r100'])/len(k['r100']):>6.1%} "
              f"{sum(k['rv100'])/len(k['rv100']):>6.1%}")

    print("\n  Kênh chữ có thực sự trả về gì không?")
    for ten, co_chu, _ in CAU_HINH:
        print(f"    {ten:28s} {ket[ten]['chu']:>7,d} ứng viên có điểm chữ > 0")

    print("\n  '*' = |Δ| > 2·SE. Không có dấu = KHÔNG phân biệt được với nền,")
    print("  bất kể con số trông thế nào.")


if __name__ == "__main__":
    main()
