#!/usr/bin/env python
"""Fit `KIS_TOL_FRAMES` trên dev set thật, và kiểm luôn calibrator.

    python fit_tol_frames.py

VÌ SAO PHẢI FIT
---------------
`tol_frames` quyết định KHI NÀO HAI KHUNG LÀ CÙNG MỘT GIẢ THUYẾT. Điểm chính
thức là `max` trên một tiền tố, nên một dòng đúng "cùng hoàn cảnh" với dòng
trước nó cộng thêm ĐÚNG BẰNG 0. Vì vậy tol quá lớn thì optimizer tưởng một dòng
phủ nhiều kịch bản rồi bỏ qua video đó (mất recall); tol quá nhỏ thì nó tiêu slot
vào các khung thực chất trùng nhau.

`submission_optimizer.KIS_TOL_FRAMES = 25` là số ĐOÁN (README ghi rõ
"unverified default"). Script này thay nó bằng số đo.

CHỌN PLATEAU, KHÔNG CHỌN ĐỈNH
-----------------------------
Dev set 120 câu, nên đỉnh của đường cong gần như chắc chắn là nhiễu lấy
mẫu. Cái đáng tin là ĐOẠN BẰNG PHẲNG: nếu điểm gần như không đổi trên một dải
tol rộng, lấy giữa dải đó thì đổi dev set vẫn còn đúng. Script in cả đường cong
để bạn tự nhìn, không chỉ in một con số.

GIỮ CỐ ĐỊNH MỌI THỨ KHÁC
------------------------
Ứng viên được truy hồi ĐÚNG MỘT LẦN rồi tái dùng cho mọi giá trị tol. Nếu
retrieve lại mỗi vòng thì chênh lệch đo được sẽ lẫn cả nhiễu của tầng TÌM.
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
from aic2026.core.providers import CLIPTextEncoder, SigLIP2TextEncoder
from aic2026.core.retrieval import MultiChannelRetriever
from aic2026.core.stores import NumpyVectorStore
from aic2026.optimize import submission_optimizer as opt
from aic2026.optimize.calibration import IdentityCalibrator, RankCalibrator

GOC = os.path.dirname(os.path.abspath(__file__))
DATA = r"D:\COMP\AIC2026\data cuộc thi\aic2026_data_nhe"
DEV_JSON = os.path.join(GOC, "data", "dev_set_tu_lam.json")
CACHE = os.path.join(GOC, "dev_set", "query_vectors.npz")

# Dải tol: dày ở vùng nhỏ (nơi đường cong đổi nhanh), thưa dần về sau.
DAI_TOL = [0, 1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 75, 100, 150, 200, 300, 500]


def _nap_ddt():
    """Nạp `do_diem_thuc_te.py` như module để dùng chung `BIEN_THE_SIGLIP2`.

    Khai lại đường dẫn ở đây là cách chắc chắn nhất để hai script trỏ vào hai
    thư mục khác nhau sau vài lần sửa.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ddt", os.path.join(GOC, "do_diem_thuc_te.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tb_va_se(v):
    """Trung bình và sai số chuẩn của trung bình."""
    n = len(v)
    tb = sum(v) / n
    if n < 2:
        return tb, float("inf")
    var = sum((x - tb) ** 2 for x in v) / (n - 1)
    return tb, math.sqrt(var / n)


def main():
    # `--clip` = fit lại trên encoder CŨ của BTC. Mặc định là SigLIP2, vì tol
    # fit trên encoder nào thì chỉ đúng cho encoder đó: plateau phụ thuộc mật độ
    # ứng viên trong một video, mà cái đó do encoder quyết định (việc 3.3).
    dung_clip = "--clip" in sys.argv
    ddt = _nap_ddt()

    if dung_clip:
        goc_feature, cache, ten_enc = None, CACHE, "CLIP ViT-B/32 (BTC cấp)"
        lop_enc = CLIPTextEncoder
        model_id = CLIPTextEncoder.MODEL_ID
    else:
        model_id, goc_feature, cache = ddt.duong_siglip2("base", vi=False)
        lop_enc = SigLIP2TextEncoder
        ten_enc = f"SigLIP2 {model_id}"

    thieu = [p for p in (DATA, DEV_JSON, cache) if not os.path.exists(p)]
    if goc_feature and not os.path.isdir(goc_feature):
        thieu.append(goc_feature)
    if thieu:
        raise SystemExit("❌ Thiếu:\n" + "\n".join(f"   - {p}" for p in thieu)
                         + "\n   (dùng `--clip` để fit trên encoder cũ của BTC)")

    print(f"Encoder: {ten_enc}")
    print("Đang nạp 873 video ...")
    events, emb, metas, rep = load_dataset(DATA, goc_feature=goc_feature)
    print(f"  {rep.events:,d} event · {emb.shape[1]} chiều")

    enc = lop_enc(cho_phep_live=True)
    enc.MODEL_ID = model_id          # phải đặt TRƯỚC nap_cache, xem build_encoder
    enc.nap_cache(cache)
    r = MultiChannelRetriever(enc, vector_store=NumpyVectorStore(
        khu_modality_gap=True))
    r.index(events, embeddings=emb)

    queries = load_queries(DEV_JSON)
    print(f"  {len(queries)} câu dev set")

    # --- truy hồi MỘT LẦN, tái dùng cho mọi tol ---------------------------- #
    print("\nTruy hồi ứng viên (một lần duy nhất) ...")
    t = time.time()
    goi = [(q, gt, r.retrieve(q.text, top_k=100, fusion="rrf"))
           for q, gt in queries]
    print(f"  xong {time.time()-t:.1f}s")

    # --- kiểm calibrator nhận đúng loại đầu vào ---------------------------- #
    cands0 = goi[0][2]
    rc = RankCalibrator()
    p = {round(rc.prior(float(c.score), i), 6)
         for i, c in enumerate(cands0, start=1)}
    print("\n" + "=" * 74)
    print("  KIỂM CALIBRATOR")
    print("=" * 74)
    print(f"  điểm RRF: {min(c.score for c in cands0):.5f} .. "
          f"{max(c.score for c in cands0):.5f}  (luôn < 1)")
    print(f"  RankCalibrator.prior() -> {len(p)} prior khác nhau trên 100 ứng viên")
    if len(p) == 1:
        raise SystemExit(
            "\n  ❌ Mọi ứng viên nhận CÙNG một prior -> tầng optimize mất sạch\n"
            "     thứ hạng của retrieval. Xem Calibrator.dung_hang.")
    print("  ✅ prior biến thiên theo ứng viên")

    # --- điểm TỪNG CÂU, để so được theo cặp -------------------------------- #
    def diem_tung_cau(lam_rows):
        return [scoring.final_score(lam_rows(c), gt) for _, gt, c in goi]

    nen_v = diem_tung_cau(lambda c: sub.kis_rows(c))
    nen, nen_se = tb_va_se(nen_v)
    print("\n" + "=" * 74)
    print("  QUÉT tol_frames")
    print("=" * 74)
    print(f"  nền core/ (heuristic, không phụ thuộc tol): {nen:.4f} ± {nen_se:.4f}")
    print(f"  Cột 'Δ so nền' là chênh lệch GHÉP CẶP trên cùng {len(queries)} câu, "
          "kèm sai số")
    print("  chuẩn của chênh lệch. |Δ| < 2·SE nghĩa là ĐO ĐƯỢC KHÔNG KHÁC GÌ.\n")

    cot = [("Identity", IdentityCalibrator), ("Rank", RankCalibrator)]
    print(f"  {'tol':>5s} " + " ".join(f"{t:>10s} {'Δ so nền':>16s}"
                                       for t, _ in cot))
    print("  " + "-" * 62)

    bang = {ten: [] for ten, _ in cot}
    for tol in DAI_TOL:
        dong = [f"  {tol:>5d}"]
        for ten, Lop in cot:
            v = diem_tung_cau(
                lambda c, L=Lop, tol=tol:
                opt.kis_portfolio(c, L(), tol_frames=tol).rows)
            d, _ = tb_va_se(v)
            dd, dd_se = tb_va_se([a - b for a, b in zip(v, nen_v)])
            bang[ten].append((tol, d, dd, dd_se))
            sao = "*" if abs(dd) > 2 * dd_se else " "
            dong.append(f"{d:>10.4f} {dd:>+8.4f}±{dd_se:.4f}{sao}")
        print(" ".join(dong))

    # --- plateau theo sai số, không theo đỉnh ------------------------------- #
    print("\n" + "=" * 74)
    print("  KẾT LUẬN")
    print("=" * 74)
    for ten, _ in cot:
        v = bang[ten]
        cao = max(d for _, d, _, _ in v)
        # SE điển hình của chênh lệch -> bề rộng dải "không phân biệt được"
        se_dh = sorted(s for _, _, _, s in v)[len(v) // 2]
        nguong = cao - se_dh
        # dải liên tục DÀI NHẤT mà điểm còn trong 1 SE của đỉnh
        tot, dau, dai = (0, -1), None, 0
        for i in range(len(v) + 1):
            if i < len(v) and v[i][1] >= nguong:
                dau = i if dau is None else dau
                dai += 1
            else:
                if dau is not None and dai > tot[1] - tot[0] + 1:
                    tot = (dau, dau + dai - 1)
                dau, dai = None, 0
        i, j = tot
        a, b = v[i][0], v[j][0]
        giua = v[(i + j) // 2][0]
        _, d_giua, dd_giua, se_giua = v[(i + j) // 2]
        print(f"\n  {ten}:")
        print(f"    đỉnh {cao:.4f} | SE chênh lệch điển hình ±{se_dh:.4f}")
        print(f"    plateau (trong 1 SE của đỉnh): tol {a} .. {b}  "
              f"({j - i + 1}/{len(v)} điểm quét)")
        print(f"    -> chọn GIỮA dải: tol = {giua}   điểm {d_giua:.4f}, "
              f"Δ so nền {dd_giua:+.4f} ± {se_giua:.4f}")
        if abs(dd_giua) <= 2 * se_giua:
            print("    ⚠ Δ nằm trong 2·SE -> KHÔNG kết luận được là hơn hay kém nền.")
        elif dd_giua > 0:
            print("    ✅ hơn nền, vượt 2·SE.")
        else:
            print("    ❌ kém nền, vượt 2·SE.")

    print(f"\n  Mặc định hiện tại: KIS_TOL_FRAMES = {opt.KIS_TOL_FRAMES}")
    print("  Sửa ở aic2026/optimize/submission_optimizer.py rồi chạy lại")
    print("  do_diem_thuc_te.py để có điểm --ab thật.")


if __name__ == "__main__":
    main()
