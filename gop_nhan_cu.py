#!/usr/bin/env python
"""Mang nhãn đã gõ từ file cũ sang file dan_nhan.csv mới, khỏi gõ lại.

    python gop_nhan_cu.py CU.csv MOI.csv [RA.csv]

    # thường dùng: file cũ đang nằm sẵn trong dev_set/, file mới vừa tải về
    python gop_nhan_cu.py dev_set/dan_nhan.csv ~/Downloads/dan_nhan.csv

VÌ SAO KHÔNG CHỈ CHÉP 60 DÒNG ĐẦU
---------------------------------
`kaggle_chuan_bi_dev_set.py` bốc mẫu bằng `random.Random(HAT_GIONG)`, nên VỀ LÝ
THUYẾT nâng SO_CAU từ 60 lên 120 sẽ giữ nguyên 60 lượt bốc đầu. Nhưng điều đó
chỉ đúng nếu tập video hợp lệ trên Kaggle không đổi chút nào — mà đó là giả định
không kiểm được từ máy này.

Nên script khớp theo KHOÁ THẬT `(video_id, n)`. Thứ tự có đổi, có xen thêm video
mới, hay bốc trúng keyframe khác thì vẫn đúng; và nếu số nhãn chuyển được ít hơn
mong đợi, nó nói thẳng thay vì âm thầm gán lệch dòng — gán lệch nghĩa là đáp án
sai mà không ai phát hiện.
"""
import csv
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

COT_NHAN = ("mo_ta", "mo_ta_en")


def doc(p):
    with open(p, "r", encoding="utf-8-sig", newline="") as fh:
        r = csv.DictReader(fh)
        return list(r), list(r.fieldnames or [])


def khoa(h):
    return (str(h.get("video_id", "")).upper().strip(),
            str(h.get("n", "")).strip())


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    p_cu, p_moi = sys.argv[1], sys.argv[2]
    p_ra = sys.argv[3] if len(sys.argv) > 3 else p_moi

    for p in (p_cu, p_moi):
        if not os.path.exists(p):
            raise SystemExit(f"❌ Không thấy {p}")

    cu, _ = doc(p_cu)
    moi, cot_moi = doc(p_moi)
    print(f"cũ : {len(cu):>4d} dòng  {p_cu}")
    print(f"mới: {len(moi):>4d} dòng  {p_moi}")

    co_nhan = {}
    for h in cu:
        if any((h.get(c) or "").strip() for c in COT_NHAN):
            co_nhan[khoa(h)] = h
    print(f"\nfile cũ có {len(co_nhan)} dòng ĐÃ gán nhãn")

    for c in COT_NHAN:
        if c not in cot_moi:
            cot_moi.append(c)

    chuyen, da_co, trung_khoa = 0, 0, len({khoa(h) for h in moi}) != len(moi)
    for h in moi:
        k = khoa(h)
        if k not in co_nhan:
            continue
        if any((h.get(c) or "").strip() for c in COT_NHAN):
            da_co += 1
            continue
        for c in COT_NHAN:
            h[c] = (co_nhan[k].get(c) or "").strip()
        chuyen += 1

    if trung_khoa:
        print("⚠ file mới có (video_id, n) trùng nhau — kiểm lại nguồn sinh file")

    con_trong = sum(1 for h in moi
                    if not any((h.get(c) or "").strip() for c in COT_NHAN))
    khong_khop = len(co_nhan) - chuyen - da_co

    print(f"\n  chuyển sang được : {chuyen}")
    if da_co:
        print(f"  bỏ qua (đã có nhãn): {da_co}")
    if khong_khop > 0:
        print(f"  ⚠ {khong_khop} nhãn cũ KHÔNG khớp dòng nào ở file mới")
        print("     -> lượt bốc mẫu đã đổi. Không mất gì, chỉ là phải gõ lại")
        print("        chừng đó câu. Giữ file cũ lại, đừng xoá.")
    print(f"  còn trống phải gõ: {con_trong}/{len(moi)}")

    with open(p_ra, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cot_moi)
        w.writeheader()
        for h in moi:
            w.writerow({c: h.get(c, "") for c in cot_moi})
    print(f"\n✅ Ghi {p_ra}")
    print("   Bước kế: python tao_trang_gan_nhan.py  rồi mở dev_set/gan_nhan.html")


if __name__ == "__main__":
    main()
