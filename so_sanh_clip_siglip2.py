#!/usr/bin/env python
"""VIỆC 2.5 + 2.6 — CLIP (BTC cấp) vs SigLIP2 (tự encode), ghép cặp theo từng câu.

    python so_sanh_clip_siglip2.py

VÌ SAO KHÔNG CHỈ SO HAI CON SỐ
------------------------------
`do_diem_thuc_te.py` chạy một cấu hình một lần và in `delta ± paired SE` cho
riêng cặp core/+optimize. So hai lần chạy khác nhau bằng mắt thì không biết
chênh lệch có vượt nhiễu hay không. Ở đây cả ba cấu hình chấm trên CÙNG 120 câu,
nên tính được SE của chênh lệch GHÉP CẶP — đúng thứ mà nghiệm thu của việc 2 đòi.

HAI THƯỚC ĐO, ĐỌC KHÁC NHAU
---------------------------
1. Điểm chính thức (mean of top-k R-Score) — thứ BTC chấm. Chịu ảnh hưởng của cả
   tầng xếp slot.
2. Hạng của keyframe ĐÚNG trong 177.321 — thuần chất lượng tầng TÌM, không dính
   xếp slot. Bảng trong docstring `MCLIPTextEncoder` ghi CLIP+EN = 283; đây là
   con số phải tụt xuống để nói encoder mạnh hơn thật.

Hạng tính bằng cosine thô, KHÔNG khử modality gap, để so được với bảng cũ.
"""
import importlib.util
import math
import os
import sys

import numpy as np

GOC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, GOC)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026 import evaluation as scoring          # noqa: E402
from aic2026.core.offline_eval import load_queries  # noqa: E402
from aic2026.core.pipeline import PortfolioConfig   # noqa: E402


def _nap_ddt():
    """Nạp `do_diem_thuc_te.py` như module để dùng lại đường dẫn + hàm dựng."""
    spec = importlib.util.spec_from_file_location(
        "ddt", os.path.join(GOC, "do_diem_thuc_te.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ddt = _nap_ddt()

# (tên, biến thể siglip2 hoặc "" cho CLIP, dùng truy vấn tiếng Việt?)
CAU_HINH = [
    ("CLIP+EN   (BTC cấp)", "", False),
    ("SigLIP2+EN", "base", False),
    ("SigLIP2+VI", "base", True),
]


def hang_keyframe_dung(events, emb, enc, queries):
    """Hạng (1 = tốt nhất) của keyframe ĐÚNG trong toàn bộ event, theo cosine thô.

    Trả None cho câu không tìm được event nào khớp ground truth — đừng lẫn nó với
    hạng xấu, vì đó là lỗi dữ liệu chứ không phải lỗi encoder.
    """
    vid_arr = np.array([e.video_id for e in events])
    kf_arr = np.array([e.keyframe_idx for e in events])

    ra = []
    for q, gt in queries:
        v = np.asarray(enc.encode_text(q.text), dtype=emb.dtype)
        v /= np.linalg.norm(v)
        diem = emb @ v
        dung = ((vid_arr == gt.video_id) &
                (kf_arr >= gt.frame_start) & (kf_arr <= gt.frame_end))
        if not dung.any():
            ra.append(None)
            continue
        # Hạng = số event ghi điểm CAO HƠN event đúng tốt nhất, cộng 1.
        tot_nhat = diem[dung].max()
        ra.append(int((diem > tot_nhat).sum()) + 1)
    return ra


def cham(ten, bien_the, vi):
    print(f"\n--- {ten} ---")
    ddt.kiem_tra_dieu_kien(can_dev_set=True, siglip2=bien_the, vi=vi)
    events, emb, _, enc = ddt.nap_he_thong(siglip2=bien_the, vi=vi)
    queries = load_queries(ddt.DEV_JSON,
                           truong_text=ddt.truong_text_dev_set(vi))

    p = ddt.dung_pipeline(events, emb, enc,
                          portfolio=PortfolioConfig(enabled=False))
    diem = {}
    for q, gt in queries:
        diem[q.qid] = scoring.final_score(p.answer(q, top_k=100), gt)
    tb = sum(diem.values()) / len(diem)

    hang = hang_keyframe_dung(events, emb, enc, queries)
    co = [h for h in hang if h is not None]
    print(f"  điểm {tb:.4f} · hạng trung vị {int(np.median(co)):,d}/{len(events):,d}"
          f" · {len(hang) - len(co)} câu không có event khớp GT")
    return tb, diem, co


def se_ghep_cap(a, b):
    """SE của chênh lệch ghép cặp theo qid. Ghép cặp mới trừ được nhiễu do câu dễ/khó."""
    if set(a) != set(b):
        raise ValueError("hai cấu hình không chấm cùng tập câu")
    d = [b[k] - a[k] for k in a]
    tb = sum(d) / len(d)
    if len(d) < 2:
        return tb, float("inf")
    var = sum((x - tb) ** 2 for x in d) / (len(d) - 1)
    return tb, math.sqrt(var / len(d))


def main():
    kq = []
    for ten, bt, vi in CAU_HINH:
        kq.append((ten,) + cham(ten, bt, vi))

    ten_nen, tb_nen, diem_nen, hang_nen = kq[0]
    print("\n" + "=" * 78)
    print("  VIỆC 2.5 — CLIP vs SigLIP2, ghép cặp trên cùng 120 câu")
    print("=" * 78)
    print(f"  {'cấu hình':22s} {'điểm':>8s} {'Δ so nền':>10s} {'±SE':>8s} "
          f"{'2·SE?':>7s} {'hạng trung vị':>14s}")
    for ten, tb, diem, hang in kq:
        h = f"{int(np.median(hang)):,d}"
        if ten == ten_nen:
            print(f"  {ten:22s} {tb:8.4f} {'(nền)':>10s} {'':>8s} {'':>7s} {h:>14s}")
            continue
        d, se = se_ghep_cap(diem_nen, diem)
        vuot = "CÓ" if abs(d) > 2 * se else "chưa"
        print(f"  {ten:22s} {tb:8.4f} {d:+10.4f} {se:8.4f} {vuot:>7s} {h:>14s}")

    print(f"\n  Nghiệm thu việc 2: hạng trung vị tụt rõ dưới 283 (mức CLIP+EN cũ)")
    print(f"  và Δ ghép cặp vượt 2·SE. Đọc bảng trên để kết luận.")

    # 2.6 — có bỏ được lớp dịch và ô tiếng Anh khi gán nhãn hay không?
    ten_en, tb_en, diem_en, _ = kq[1]
    ten_vi, tb_vi, diem_vi, _ = kq[2]
    d, se = se_ghep_cap(diem_en, diem_vi)
    print("\n" + "=" * 78)
    print("  VIỆC 2.6 — SigLIP2 đọc tiếng Việt thẳng, có bỏ được lớp dịch không?")
    print("=" * 78)
    print(f"  VI so với EN: {d:+.4f} ± {se:.4f}"
          f"   ({'vượt' if abs(d) > 2 * se else 'chưa vượt'} 2·SE)")
    if d < -2 * se:
        print("  -> KHÔNG bỏ được. Giữ ô tiếng Anh khi gán nhãn; bản người dịch")
        print("     vẫn là đường chính. Mất mát khi dùng tiếng Việt thẳng đo được ở trên.")
    elif abs(d) <= 2 * se:
        print("  -> Chưa phân biệt được với nhiễu. Chưa đủ cơ sở bỏ ô tiếng Anh.")
    else:
        print("  -> Tiếng Việt thẳng KHÔNG kém hơn. Bỏ được lớp dịch lẫn ô tiếng Anh,")
        print("     tiết kiệm nửa công gán nhãn các đợt sau.")


if __name__ == "__main__":
    main()
