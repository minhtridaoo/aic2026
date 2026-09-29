#!/usr/bin/env python
"""So các cách xử lý truy vấn tiếng Việt, trên 177.321 vector ảnh thật.

    python so_sanh_encoder.py

Đọc mọi file `dev_set/query_*.npz` do `kaggle_so_sanh_encoder.py` sinh ra và
chấm từng cách bằng hai thước đo KHÔNG cần đáp án:

  1. CHỒNG LẤN top-100 giữa các truy vấn KHÁC NGHĨA
     Thấp = phân biệt được nội dung. Cao = mọi câu đổ về một vùng, vô dụng.
     Đây là thước đo quyết định — xem ghi chú trong `do_diem_thuc_te.py` về
     lý do KHÔNG dùng z-score.

  2. LIFT theo nhãn yếu lấy từ tiêu đề video
     Trong top-100 có bao nhiêu phần trăm thuộc đúng nhóm chủ đề, so với tỉ lệ
     nền của cả corpus. Nhãn thô nhưng miễn phí và có thật.

Cả hai đều đo được TRƯỚC KHI có dev set gán nhãn, nên chạy được ngay hôm nay.
"""
import glob
import itertools
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from aic2026.core.dataset import load_dataset

DATA = r"D:\COMP\AIC2026\data cuộc thi\aic2026_data_nhe"
DEV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dev_set")

# Nhãn yếu: tiêu đề video chứa từ khoá nào thì xếp vào nhóm đó. Thô, nhưng đủ
# để thấy chênh lệch lớn giữa các encoder.
NHOM = {
    "bien": r"(?i)(BIỂN|ĐẢO|BÃI |VỊNH)",
    "do_an": r"(?i)(MÓN NGON|ẨM THỰC|LẨU|NƯỚNG|CANH |BÁNH|CHÁO|PHỞ|BÚN)",
    "hoc": r"(?i)(ÔN THI|MÔN |HỌC SINH|GIẢNG)",
    "tin": r"(?i)(60 Giây|Tin Tức|Thời sự)",
}
# Khái niệm -> nhóm kỳ vọng. Chỉ những câu đủ rõ mới đưa vào chấm lift.
KY_VONG = {
    "bãi biển và hàng dừa": "bien", "a beach with palm trees": "bien",
    "món ăn trên bàn trong nhà hàng": "do_an",
    "food on a table in a restaurant": "do_an",
    "phòng học có bảng trắng": "hoc", "a classroom with a whiteboard": "hoc",
    "người phát biểu trước micro": "tin",
    "a person speaking at a microphone": "tin",
}


def chuan(s):
    return " ".join(str(s).split())


def main():
    files = sorted(glob.glob(os.path.join(DEV, "query_*.npz")))
    if not files:
        raise SystemExit(
            f"Không thấy file query_*.npz nào trong {DEV}\n"
            "  Chạy kaggle_so_sanh_encoder.py trên Kaggle trước, tải kết quả về.")

    print("Đang nạp 873 video ...")
    ev, emb, metas, rep = load_dataset(DATA)
    print(f"  {rep.events:,d} keyframe\n")

    lab = {}
    for vid, m in metas.items():
        for t, p in NHOM.items():
            if re.search(p, m.title):
                lab[vid] = t
                break
    L = np.array([lab.get(e.video_id, "") for e in ev])
    nen = {t: float((L == t).mean()) for t in NHOM}

    print(f"  {'cách':14s} {'#câu':>5s} {'chồng lấn':>10s} {'lift':>7s}  nhận xét")
    print("  " + "-" * 74)
    ket = []
    for f in files:
        ten = os.path.basename(f)[len("query_"):-len(".npz")]
        d = np.load(f, allow_pickle=False)
        keys = [chuan(k) for k in d["keys"]]
        vecs = np.asarray(d["vectors"], dtype=np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        if vecs.shape[1] != emb.shape[1]:
            print(f"  {ten:14s} bỏ qua — {vecs.shape[1]} chiều, kho {emb.shape[1]}")
            continue

        top, lifts = {}, []
        for k, q in zip(keys, vecs):
            idx = np.argsort(-(emb @ q))[:100]
            top[k] = set(idx.tolist())
            nh = KY_VONG.get(k)
            if nh and nen[nh] > 0:
                lifts.append(float((L[idx] == nh).mean()) / nen[nh])

        cap = list(itertools.combinations(keys, 2))
        cl = (sum(len(top[a] & top[b]) for a, b in cap) / (100 * len(cap))
              if cap else float("nan"))
        lf = float(np.mean(lifts)) if lifts else float("nan")
        ket.append((ten, cl, lf))

        nx = ("✅ phân biệt tốt" if cl < 0.25 else
              "⚠️  yếu" if cl < 0.40 else "❌ GOM CỤM, vô dụng")
        print(f"  {ten:14s} {len(keys):5d} {cl:9.1%} {lf:7.2f}x  {nx}")

    print("  " + "-" * 74)
    hop_le = [k for k in ket if not np.isnan(k[1])]
    if hop_le:
        thang = min(hop_le, key=lambda k: k[1])
        print(f"\n  🏆 THẤP NHẤT (phân biệt tốt nhất): {thang[0]}  "
              f"chồng lấn {thang[1]:.1%}, lift {thang[2]:.2f}x")
        print("\n  Đọc bảng: 'chồng lấn' là thước đo chính — hai câu khác nghĩa mà")
        print("  trả về cùng kết quả thì encoder đó không mang thông tin nội dung.")
        print("  'lift' đo bằng nhãn yếu từ tiêu đề, chỉ dùng để đối chiếu thêm.")
        print("\n  ⚠ Đây là đo GIÁN TIẾP, chưa phải điểm thi. Sau khi có dev set gán")
        print("    nhãn, chạy do_diem_thuc_te.py để có điểm theo công thức BTC.")


if __name__ == "__main__":
    main()
