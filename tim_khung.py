#!/usr/bin/env python
"""(video, giây) -> keyframe + frame_idx, và HẠNG của nó trong bài nộp.

    python -X utf8 tim_khung.py L21_V015 839 --zip out/NOP/aic2026thunghiemL8.zip \
        --cau query-p1-1-kis

Dùng khi ASR/OCR đã chỉ ra ĐÚNG GIÂY nhưng cần biết nộp khung số mấy, và bài
nộp hiện tại đang để khung đó ở đâu. Cửa sổ chấm KIS chỉ ~107 khung (~4,3 giây)
nên "đúng video, lệch 10 giây" vẫn là 0 điểm.
"""
import argparse
import csv
import io
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026 import config as cau_hinh
from aic2026.core import dataset

GOC = os.path.dirname(os.path.abspath(__file__))


def _args():
    p = argparse.ArgumentParser(description="giây -> keyframe -> hạng trong bài nộp")
    p.add_argument("video")
    p.add_argument("giay", type=float)
    p.add_argument("--quanh", type=float, default=12.0,
                   help="in các keyframe trong ±N giây (mặc định 12)")
    p.add_argument("--zip", default="")
    p.add_argument("--cau", default="")
    p.add_argument("--config", default=os.path.join(GOC, "config_ocr_asr.yaml"))
    return p.parse_args()


def main():
    a = _args()
    config = cau_hinh.load_config(a.config)
    paths = dataset.scan(config["offline"]["btc_data_root"])
    vid = a.video.upper()
    if vid not in paths:
        raise SystemExit(f"không có video {vid}")
    rows = dataset.load_map_keyframes(paths[vid].mapping)

    gan = [r for r in rows if abs(float(r.pts_time) - a.giay) <= a.quanh]
    if not gan:
        gan = [min(rows, key=lambda r: abs(float(r.pts_time) - a.giay))]

    hang = {}
    if a.zip and a.cau:
        z = zipfile.ZipFile(a.zip)
        ten = f"submission/{a.cau}.csv"
        csv_rows = list(csv.reader(io.StringIO(z.read(ten).decode("utf-8"))))
        for i, r in enumerate(csv_rows, 1):
            if r[0].upper() == vid:
                hang.setdefault(int(r[1]), i)
        co = sum(1 for r in csv_rows if r[0].upper() == vid)
        print(f"Bài nộp `{a.cau}`: {co}/100 dòng thuộc {vid}\n")

    print(f"{vid} · {len(rows)} keyframe · quanh {a.giay:.0f}s ±{a.quanh:.0f}s")
    print(f"  {'kf':>5s} {'pts':>9s} {'frame_idx':>10s}   hạng trong bài nộp")
    print("  " + "-" * 56)
    for r in gan:
        h = hang.get(int(r.frame_idx))
        print(f"  {r.n:5d} {float(r.pts_time):9.2f} {int(r.frame_idx):10d}   "
              + (f"#{h}" if h else "— không có trong 100 dòng"))

    khung = ", ".join(f'"{vid}|{int(r.frame_idx)}"' for r in gan)
    print(f"\n  Dán vào file ghim:\n    \"khung\": [{khung}]")


if __name__ == "__main__":
    main()
