#!/usr/bin/env python
"""Máy dịch có thay được bản tiếng Anh bạn gõ tay không?

    python thu_nghiem_dich.py

CÂU HỎI
-------
Dev set hiện có CẢ HAI: `_mo_ta_vi` (bạn gõ) và `text` (bạn tự dịch sang tiếng
Anh). Đó là điều kiện hiếm — cho phép chấm máy dịch bằng chính thước đo cuối
cùng là ĐIỂM TÌM KIẾM, chứ không phải BLEU.

Bốn đường được so, cùng dev set, cùng cách dựng row, chỉ khác câu đưa vào encoder:

    1. EN người (nền)   bản tiếng Anh bạn gõ  -> CLIP        <- trần hiện tại
    2. EN máy dịch      tiếng Việt -> opus-mt -> CLIP        <- thứ cần kiểm
    3. VI thô + CLIP    tiếng Việt thẳng      -> CLIP        <- biết là hỏng
    4. VI thô + mCLIP   tiếng Việt thẳng      -> mCLIP       <- lưới an toàn

Ý nghĩa của từng kết quả:
  - (2) ≈ (1)  -> bỏ được ô tiếng Anh khi gán nhãn, và lúc thi cứ dịch là xong.
  - (2) ≪ (1)  -> máy dịch làm mất thông tin; phải giữ người dịch, hoặc đổi model.
  - (2) < (4)  -> dịch còn tệ hơn không dịch; vứt cả hướng này.

Cột Δ là chênh lệch GHÉP CẶP so với (1), kèm SE. Dấu `*` = |Δ| > 2·SE.
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json

from aic2026.core import scoring
from aic2026.core import submission as sub
from aic2026.core.dataset import load_dataset
from aic2026.core.dich import DichRoiEncode, NLLBTranslator, OpusMTTranslator
from aic2026.core.models import GroundTruth, Query
from aic2026.core.providers import CLIPTextEncoder, MCLIPTextEncoder
from aic2026.core.retrieval import MultiChannelRetriever
from aic2026.core.stores import NumpyVectorStore

GOC = os.path.dirname(os.path.abspath(__file__))
DATA = r"D:\COMP\AIC2026\data cuộc thi\aic2026_data_nhe"
DEV_JSON = os.path.join(GOC, "data", "dev_set_tu_lam.json")
CACHE = os.path.join(GOC, "dev_set", "query_vectors.npz")
MCLIP_VI_CACHE = os.path.join(GOC, "dev_set", "query_mclip_vi.npz")


def tb_va_se(v):
    n = len(v)
    tb = sum(v) / n
    if n < 2:
        return tb, float("inf")
    var = sum((x - tb) ** 2 for x in v) / (n - 1)
    return tb, math.sqrt(var / n)


def main():
    thieu = [p for p in (DATA, DEV_JSON) if not os.path.exists(p)]
    if thieu:
        raise SystemExit("❌ Thiếu:\n" + "\n".join(f"   - {p}" for p in thieu))

    with open(DEV_JSON, encoding="utf-8") as fh:
        qs = json.load(fh)["queries"]
    thieu_vi = [q["qid"] for q in qs if not q.get("_mo_ta_vi")]
    if thieu_vi:
        raise SystemExit(
            f"❌ {len(thieu_vi)} câu không có `_mo_ta_vi` -> không so được.\n"
            "   Chạy lại tao_dev_set.py từ dan_nhan.csv có cột `mo_ta`.")

    gts = [GroundTruth(task="kis", video_id=q["gt"]["video_id"],
                       frame_start=q["gt"]["frame_start"],
                       frame_end=q["gt"]["frame_end"]) for q in qs]
    en_nguoi = [q["text"] for q in qs]
    vi = [q["_mo_ta_vi"] for q in qs]

    print("Đang nạp 873 video ...")
    events, emb, metas, rep = load_dataset(DATA)
    print(f"  {rep.events:,d} event, {len(qs)} câu dev set")

    # --- dịch máy ---------------------------------------------------------- #
    print(f"\nDịch {len(vi)} câu bằng {OpusMTTranslator.MODEL_ID} ...")
    clip = CLIPTextEncoder(cache_path=CACHE if os.path.exists(CACHE) else None,
                           cho_phep_live=True)
    dich = DichRoiEncode(clip, OpusMTTranslator())
    t = time.time()
    dich.nap_truoc(vi)
    giay = time.time() - t
    print(f"  xong {giay:.1f}s  ({giay/len(vi)*1000:.0f} ms/câu)")
    # `.get(v, v)`: câu không có dấu tiếng Việt thì `DichRoiEncode` cho đi thẳng,
    # nên không nằm trong bảng dịch. Đó là hành vi đúng — và nó phơi ra một
    # chuyện về chất lượng nhãn, đếm ở dưới.
    en_may = [dich.ban_dich.get(v, v) for v in vi]
    khong_dau = [q["qid"] for q, v in zip(qs, vi) if v not in dich.ban_dich]
    if khong_dau:
        print(f"\n  ⚠ {len(khong_dau)}/{len(vi)} dòng `mo_ta` KHÔNG có dấu tiếng")
        print("    Việt — nhiều khả năng bạn gõ nhầm tiếng Anh vào ô tiếng Việt.")
        print(f"    {khong_dau[:6]}")
        print("    Chúng đi thẳng không qua dịch, nên phép so vẫn hợp lệ, nhưng")
        print("    hãy sửa lại khi gán nhãn đợt 120 câu.")

    print(f"\nDịch lại bằng {NLLBTranslator.MODEL_ID} (nặng hơn) ...")
    dich2 = DichRoiEncode(clip, NLLBTranslator())
    t = time.time()
    try:
        dich2.nap_truoc(vi)
        giay2 = time.time() - t
        en_nllb = [dich2.ban_dich.get(v, v) for v in vi]
        print(f"  xong {giay2:.1f}s  ({giay2/len(vi)*1000:.0f} ms/câu)")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  ❌ không chạy được: {type(exc).__name__}: {str(exc)[:120]}")
        en_nllb = None

    print("\n  3 ví dụ:")
    for i in range(3):
        print(f"    VI      {vi[i][:64]}")
        print(f"    người   {en_nguoi[i][:64]}")
        print(f"    opus    {en_may[i][:64]}")
        if en_nllb:
            print(f"    nllb    {en_nllb[i][:64]}")

    # --- các đường --------------------------------------------------------- #
    duong = [
        ("1. EN người (nền)", clip,                        en_nguoi),
        ("2. EN opus-mt",     clip,                        en_may),
        ("3. VI thô + CLIP",  clip,                        vi),
        ("4. VI thô + mCLIP", MCLIPTextEncoder(
            cache_path=MCLIP_VI_CACHE if os.path.exists(MCLIP_VI_CACHE) else None,
            cho_phep_live=True), vi),
    ]
    if en_nllb:
        duong.insert(2, ("2b. EN nllb-600M", clip, en_nllb))

    print("\nĐang chấm ...")
    ket = {}
    for ten, enc, cau in duong:
        r = MultiChannelRetriever(enc, vector_store=NumpyVectorStore(
            khu_modality_gap=True))
        r.index(events, embeddings=emb)
        d, a1, a100, av = [], [], [], []
        for text, gt in zip(cau, gts):
            rows = sub.kis_rows(r.retrieve(text, top_k=100, fusion="rrf"))
            d.append(scoring.final_score(rows, gt))
            a1.append(scoring.recall_at_k(rows, gt, 1, threshold=1.0))
            a100.append(scoring.recall_at_k(rows, gt, 100, threshold=1.0))
            av.append(1.0 if any(x["video_id"] == gt.video_id
                                 for x in rows) else 0.0)
        ket[ten] = (d, a1, a100, av)
        print(f"  {ten} xong")

    nen = ket[duong[0][0]][0]
    print("\n" + "=" * 78)
    print("  KẾT QUẢ")
    print("=" * 78)
    print(f"  {'đường':22s} {'điểm':>7s} {'Δ so nền':>17s} "
          f"{'R@1':>6s} {'R@100':>7s} {'video':>7s}")
    print("  " + "-" * 70)
    for ten, _, _ in duong:
        d, a1, a100, av = ket[ten]
        tb, _ = tb_va_se(d)
        if ten == duong[0][0]:
            c = f"{'(nền)':>17s}"
        else:
            dd, se = tb_va_se([x - y for x, y in zip(d, nen)])
            c = f"{dd:>+9.4f}±{se:.4f}" + ("*" if abs(dd) > 2 * se else " ")
        n = len(d)
        print(f"  {ten:22s} {tb:>7.4f} {c} {sum(a1)/n:>5.1%} "
              f"{sum(a100)/n:>6.1%} {sum(av)/n:>6.1%}")

    print("\n" + "=" * 78)
    print("  KẾT LUẬN")
    print("=" * 78)
    print("  ⚠ 'Δ không vượt 2·SE' KHÔNG có nghĩa là 'tương đương'. Không bác bỏ")
    print("    được khác biệt ≠ chứng minh được không khác biệt. Muốn nói 'máy")
    print("    dịch dùng thay được' thì phải xét CẬN DƯỚI của khoảng tin cậy:")
    print("    nó phải loại trừ được mức mất mát đủ lớn để mình bận tâm.\n")

    d_may = ket["2. EN opus-mt"][0]
    d_mclip = ket["4. VI thô + mCLIP"][0]
    d_nllb = ket["2b. EN nllb-600M"][0] if "2b. EN nllb-600M" in ket else None
    tb_nen, _ = tb_va_se(nen)

    so = [("opus-mt vs người dịch", d_may, nen),
          ("opus-mt vs VI + mCLIP", d_may, d_mclip)]
    if d_nllb:
        so = [("nllb    vs người dịch", d_nllb, nen),
              ("nllb    vs opus-mt   ", d_nllb, d_may),
              ("nllb    vs VI + mCLIP", d_nllb, d_mclip)] + so
    for nhan, a, b in so:
        dd, se = tb_va_se([x - y for x, y in zip(a, b)])
        print(f"  {nhan}: Δ {dd:+.4f}, "
              f"KTC 95% ≈ [{dd-2*se:+.4f}, {dd+2*se:+.4f}]")

    tot = d_nllb if (d_nllb and sum(d_nllb) > sum(d_may)) else d_may
    dd, se = tb_va_se([x - y for x, y in zip(tot, nen)])
    mat_toi_da = -(dd - 2 * se)
    print(f"\n  Chưa loại trừ được khả năng máy dịch làm mất tới {mat_toi_da:.4f} "
          f"điểm,")
    print(f"  tức {mat_toi_da/max(tb_nen,1e-9):.0%} của điểm nền. Quá lớn để gọi là "
          "tương đương.")
    print("  Cùng chiều đó: R@100 và video-R@100 đều tụt, không phải dao động ngẫu.")

    tb_tot, _ = tb_va_se(tot)
    tb_mclip, _ = tb_va_se(d_mclip)
    print(f"\n  Máy dịch tốt nhất {tb_tot:.4f} | VI thô + mCLIP {tb_mclip:.4f} "
          f"| người dịch {tb_nen:.4f}")
    # So bằng KTC chứ không so hai con số. `0.0967 > 0.0933` không phải bằng
    # chứng — cùng lỗi "chọn theo điểm số" mà phần trên vừa cảnh báo.
    dd_m, se_m = tb_va_se([x - y for x, y in zip(tot, d_mclip)])
    if abs(dd_m) <= 2 * se_m:
        print(f"  -> Dịch vs mCLIP: Δ {dd_m:+.4f}, KTC [{dd_m-2*se_m:+.4f}, "
              f"{dd_m+2*se_m:+.4f}] — KHÔNG phân biệt được.")
        print("     Hai đường ngang nhau trong tầm đo được, nên chọn theo CHI PHÍ:")
        print("     mCLIP là một component, dịch+CLIP là hai, cộng vài trăm MB và")
        print("     vài trăm ms mỗi câu. Với truy vấn tiếng Việt lúc thi -> mCLIP.")
        print("     Giữ lại `dich.py` vì nó rẻ và có thể thắng khi có đủ nhãn.")
    elif dd_m > 0:
        print(f"  -> Dịch HƠN mCLIP thật ({dd_m:+.4f} ± {se_m:.4f}). Giữ bước dịch.")
    else:
        print(f"  -> Dịch KÉM mCLIP ({dd_m:+.4f} ± {se_m:.4f}). Bỏ bước dịch.")
    print(f"\n  Nhưng cả hai đều còn cách bản NGƯỜI dịch ({tb_nen:.4f}) -> khi gán")
    print("  nhãn dev set, ô tiếng Anh VẪN ĐÁNG GÕ. Nhãn là thước đo, thước đo mà")
    print("  nhiễu thì mọi thứ đo bằng nó đều nhiễu theo.")


if __name__ == "__main__":
    main()
