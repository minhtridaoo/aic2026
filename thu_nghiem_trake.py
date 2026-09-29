#!/usr/bin/env python
"""TRAKE: keyframe chính thức quá thưa — tiêu 100 slot thế nào cho đỡ nhất?

    python thu_nghiem_trake.py

VẤN ĐỀ CẤU TRÚC
---------------
BTC chấm TRAKE theo cửa sổ "thường dưới 10 khung" (§2.1.3). Nhưng keyframe chính
thức cách nhau TRUNG VỊ 116 KHUNG (đo trên 55.926 khoảng, 200 video). Tức nếu
chỉ nộp đúng chỉ số keyframe, xác suất rơi trúng cửa sổ chỉ ~10/116 = 8.6% mỗi
mốc — DÙ đã tìm đúng video và đúng thời điểm.

Không có thuật toán nào lách được chuyện này. Nhưng có hai điều cứu vãn:

  1. R-Score của TRAKE trả ĐIỂM TỪNG PHẦN: `(1/N)Σⱼ I(khungⱼ ∈ cửa sổⱼ)`.
     Sai 2 trong 3 mốc vẫn được 1/3, không phải 0.
  2. Final lấy MAX trên tiền tố. Nộp nhiều dòng lệch nhau thì chỉ cần MỘT dòng
     may mắn.

Nên chiến lược đúng là RẢI: nộp cùng một căn chỉnh ở nhiều độ lệch khác nhau,
phủ dần khoảng trống 116 khung. `PortfolioConfig.trake_jitters` làm đúng việc đó
nhưng mặc định là `(0,)` — tức không rải gì cả.

Script này đo phân bố khoảng cách THẬT rồi mô phỏng xem lưới jitter nào đáng
dùng, thay vì chọn bằng cảm tính.

MÔ PHỎNG MÔ HÌNH GÌ
-------------------
Giả định đã tin đúng một video + một căn chỉnh (trường hợp TỐT NHẤT — nên con số
ra là TRẦN TRÊN, không phải dự đoán điểm thi). Với mỗi mốc j, khung đúng lệch
khỏi keyframe ta chọn một khoảng uₙ, lấy ngẫu nhiên trong ±gap/2 với gap rút từ
phân bố THẬT. Dòng có jitter d ăn điểm mốc j khi |d − uⱼ| ≤ tol.
"""
import glob
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026.core.scoring import DEFAULT_KS
from aic2026.optimize.submission_optimizer import TRAKE_TOL_FRAMES

DATA = r"D:\COMP\AIC2026\data cuộc thi\aic2026_data_nhe"
N_MOC = 3           # số mốc mỗi truy vấn TRAKE
LAN = 4000          # số lần mô phỏng
MAX_ROWS = 100


def doc_khoang_cach(so_video=300):
    kc = []
    fs = sorted(glob.glob(os.path.join(DATA, "keyframe_mapping", "**", "*.csv"),
                          recursive=True))[:so_video]
    for f in fs:
        idx = []
        with open(f, encoding="utf-8") as fh:
            next(fh)
            for ln in fh:
                p = ln.strip().split(",")
                if len(p) >= 4:
                    idx.append(int(p[3]))
        kc += [b - a for a, b in zip(idx, idx[1:]) if b > a]
    return kc


def luoi(buoc, nhip):
    """Lưới jitter đối xứng, 0 trước rồi toả dần ra hai bên.

    Thứ tự quan trọng: Final lấy max trên TIỀN TỐ, và slot 1 đáng giá 5× slot 51.
    Nên độ lệch khả dĩ nhất (0) phải đứng đầu.
    """
    ra = [0]
    for i in range(1, nhip + 1):
        ra += [i * buoc, -i * buoc]
    return ra


def mo_phong(kc, jitters, tol, lan=LAN, n_moc=N_MOC, seed=7):
    rng = random.Random(seed)
    ks = DEFAULT_KS
    tong = 0.0
    for _ in range(lan):
        # khung đúng lệch bao nhiêu so với keyframe gần nhất
        u = []
        for _ in range(n_moc):
            g = kc[rng.randrange(len(kc))]
            u.append(rng.uniform(-g / 2, g / 2))
        diem_dong = []
        for d in jitters[:MAX_ROWS]:
            hits = sum(1 for x in u if abs(d - x) <= tol)
            diem_dong.append(hits / n_moc)
        tong += sum(max(diem_dong[:k], default=0.0) for k in ks) / len(ks)
    return tong / lan


def main():
    print("Đang đọc khoảng cách keyframe thật ...")
    kc = doc_khoang_cach()
    kc_s = sorted(kc)
    tv = kc_s[len(kc_s) // 2]
    print(f"  {len(kc):,d} khoảng | trung vị {tv} khung | "
          f"p25 {kc_s[len(kc_s)//4]} | p75 {kc_s[3*len(kc_s)//4]}")
    tol = TRAKE_TOL_FRAMES
    print(f"  tol hiện tại = ±{tol} khung (cửa sổ rộng {2*tol+1})")
    print(f"  -> phủ được {(2*tol+1)/tv:.1%} khoảng trống nếu chỉ nộp jitter 0\n")

    print("=" * 74)
    print("  QUÉT LƯỚI JITTER  (mô phỏng, N=3 mốc, đã tin đúng video+căn chỉnh)")
    print("=" * 74)
    print(f"  {'lưới':>22s} {'số dòng':>8s} {'Final kỳ vọng':>14s}")
    print("  " + "-" * 48)

    ung_vien = [("chỉ (0,) — mặc định", [0])]
    for buoc in (2 * tol, 2 * tol + 2, 20, 30):
        for nhip in (3, 6, 12, 24):
            g = luoi(buoc, nhip)
            if len(g) > MAX_ROWS:
                continue
            ung_vien.append((f"bước {buoc}, ±{nhip} nhịp", g))

    ket = []
    for ten, g in ung_vien:
        d = mo_phong(kc, g, tol)
        ket.append((d, ten, g))
        print(f"  {ten:>22s} {len(g):>8d} {d:>14.4f}")

    ket.sort(reverse=True)
    tot = ket[0]
    nen = [k for k in ket if k[2] == [0]][0]
    print("\n" + "=" * 74)
    print("  KẾT LUẬN")
    print("=" * 74)
    print(f"  mặc định (0,)     : {nen[0]:.4f}")
    print(f"  tốt nhất          : {tot[0]:.4f}  ({tot[1]}, {len(tot[2])} dòng)")
    print(f"  chênh lệch        : {tot[0]-nen[0]:+.4f}  "
          f"(gấp {tot[0]/max(nen[0],1e-9):.1f} lần)")
    print(f"\n  Đề nghị: PortfolioConfig(trake_jitters={tuple(tot[2][:12])}...)")
    print("\n  ⚠ Đây là TRẦN TRÊN, không phải dự đoán điểm thi: mô phỏng giả định")
    print("    đã tìm đúng video và đúng căn chỉnh. Điểm thật = trần này × tỉ lệ")
    print("    tìm đúng video. Và cách duy nhất phá được trần là có KHUNG DÀY HƠN,")
    print("    tức tự trích frame từ video gốc thay vì dùng keyframe của BTC.")


if __name__ == "__main__":
    main()
