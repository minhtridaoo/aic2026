"""Lọc `out/bang_chung_dot2.txt` xuống các video ứng viên mạnh, kèm nguyên văn OCR."""
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

QUAN_TAM = {"L25_V060", "L25_V012", "L26_V450", "L26_V192", "L30_V026",
            "L26_V461", "L26_V471", "L30_V043", "L30_V031", "L26_V342"}
t = open("out/bang_chung_dot2.txt", encoding="utf-8").read().splitlines()
cau = "?"
thay = set()
for i, l in enumerate(t):
    if l.startswith("▸"):
        cau = l.strip("▸ ").strip()
    m = re.search(r"(OCR|ASR) ([\d.]+) (L\d\d_V\d+) (kf=\d+|\d+s)", l)
    if m and m.group(3) in QUAN_TAM and float(m.group(2)) >= 0.93:
        khoa = (cau, m.group(3), m.group(4))
        if khoa in thay:
            continue
        thay.add(khoa)
        txt = t[i + 1].strip() if i + 1 < len(t) else ""
        print(f"{cau:20s} {m.group(3)} {m.group(4):9s} {m.group(1)} {m.group(2)}")
        print(f"     {txt[:140]}")
