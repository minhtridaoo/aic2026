#!/usr/bin/env python
"""XÁC MINH `frame_row` của Stage11 CÓ TRÙNG thứ tự event của hệ thống không.

    python -X utf8 kiem_thu_tu_ocr.py --sqlite "...\\ocr_fts.sqlite"

VÌ SAO PHẢI XÁC MINH TRƯỚC KHI DÙNG
-----------------------------------
`ocr_fts.sqlite` cho mỗi dòng một `frame_row` (0..177320) chứ KHÔNG cho
`(video_id, n)`. Nghĩa là muốn gán OCR vào event thì phải tin rằng thứ tự
177.321 dòng của Stage11 trùng khít thứ tự 177.321 event mà `load_dataset`
dựng ra. Nếu lệch dù chỉ một video, mọi khung sẽ nhận OCR của khung khác —
sai hoàn toàn im lặng, và còn tệ hơn không có OCR vì nó đầu độc cả xếp hạng.

PHÉP KIỂM DÙNG ĐỒNG HỒ TRÊN MÀN HÌNH
------------------------------------
Bản tin *60 Giây* của HTV có đồng hồ chạy ở góc màn hình, và OCR đọc được nó
("htvo ad 06:30:11 giáy"). Đó là một mốc thời gian TUYỆT ĐỐI, độc lập hoàn toàn
với `pts_time` trong map-keyframes của BTC.

Nếu hai thứ tự trùng nhau thì với các khung của cùng một video, hiệu đồng hồ
phải bám sát hiệu `pts_time`. Lệch thứ tự thì tương quan vỡ ngay.

Đây là kiểm chứng CHÉO thật: hai nguồn độc lập (pixel do OCR đọc, và bảng
map-keyframes của BTC) phải kể cùng một câu chuyện.
"""
import argparse
import os
import re
import sqlite3
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026 import config as cau_hinh
from aic2026.core import dataset

DONG_HO = re.compile(r"\b([01]?\d|2[0-3])[:.\s]([0-5]\d)[:.\s]([0-5]\d)\b")


def _args():
    p = argparse.ArgumentParser(description="Xác minh thứ tự frame_row Stage11")
    p.add_argument("--sqlite", required=True)
    p.add_argument("--config", default=None)
    p.add_argument("--bang", default="OCR_ALL")
    p.add_argument("--so-video", type=int, default=6,
                   help="số video đầu đem kiểm")
    return p.parse_args()


def main():
    a = _args()
    config = cau_hinh.load_config(a.config)
    goc = (config.get("offline") or {}).get("btc_data_root")

    print("Nạp map-keyframes (không nạp vector) ...")
    paths = dataset.scan(goc)
    thu_tu = sorted(paths)                       # ĐÚNG thứ tự của load_dataset
    dem, pts = [], {}
    tong = 0
    for vid in thu_tu:
        vp = paths[vid]
        if not vp.du_toi_thieu:
            continue
        rows = dataset.load_map_keyframes(vp.mapping)
        pts[vid] = [float(r.pts_time) for r in rows]
        dem.append((vid, tong, len(rows)))
        tong += len(rows)
    print(f"  {len(dem)} video · {tong:,d} keyframe\n")

    con = sqlite3.connect(a.sqlite)
    n_ocr = con.execute(f"SELECT COUNT(*) FROM {a.bang}").fetchone()[0]
    print(f"OCR `{a.bang}`: {n_ocr:,d} dòng")
    print(f"  tổng số khớp: "
          + ("✅" if n_ocr == tong else f"❌ {n_ocr} vs {tong} — DỪNG LẠI"))
    if n_ocr != tong:
        sys.exit(1)

    print(f"\nKiểm đồng hồ trên {a.so_video} video đầu:")
    print(f"  {'video':12s} {'kf':>5s} {'khung có đồng hồ':>17s} "
          f"{'lệch trung vị':>14s}  kết luận")
    print("  " + "-" * 74)

    tot = xau = 0
    for vid, base, n in dem[:a.so_video]:
        cur = con.execute(
            f"SELECT frame_row, text_norm FROM {a.bang} "
            f"WHERE frame_row >= ? AND frame_row < ? ORDER BY frame_row",
            (base, base + n))
        cap = []
        for fr, txt in cur:
            m = DONG_HO.search(str(txt or ""))
            if not m:
                continue
            giay = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            cap.append((fr - base, giay))
        if len(cap) < 8:
            print(f"  {vid:12s} {n:5d} {len(cap):17d} {'—':>14s}  "
                  f"(ít đồng hồ, bỏ qua)")
            continue

        # So HIỆU: đồng hồ tuyệt đối lệch pts một hằng số (giờ phát sóng), nên
        # chỉ hiệu mới so được. Khớp thứ tự => hiệu đồng hồ ≈ hiệu pts.
        lech = []
        for k in range(1, len(cap)):
            i0, t0 = cap[k - 1]
            i1, t1 = cap[k]
            d_ocr = t1 - t0
            d_pts = pts[vid][i1] - pts[vid][i0]
            if 0 < d_pts < 600:
                lech.append(abs(d_ocr - d_pts))
        if not lech:
            print(f"  {vid:12s} {n:5d} {len(cap):17d} {'—':>14s}  (không so được)")
            continue
        med = statistics.median(lech)
        ok = med <= 2.0
        tot += ok
        xau += not ok
        print(f"  {vid:12s} {n:5d} {len(cap):17d} {med:13.2f}s  "
              + ("✅ khớp" if ok else "❌ LỆCH"))

    print()
    if xau == 0 and tot >= 2:
        print("  ✅ THỨ TỰ TRÙNG KHỚP — gán OCR theo `frame_row` là an toàn.")
    elif tot == 0:
        print("  ⚠ KHÔNG KẾT LUẬN ĐƯỢC (không đủ khung có đồng hồ). "
              "Đừng bật OCR khi chưa có bằng chứng khác.")
        sys.exit(2)
    else:
        print(f"  ❌ {xau} video LỆCH — KHÔNG được gán OCR theo `frame_row`.")
        sys.exit(1)


if __name__ == "__main__":
    main()
