"""Tìm cụm từ trong ASR của CẢ KHO (18.660 khoảng) — khác `_dao_video.py` vốn
chỉ soi một video. Dùng khi chưa biết video nào.

    python -X utf8 _tim_asr.py "tieu xanh" "la chanh"      # phải khớp CẢ HAI
"""
import csv
import sys
import unicodedata
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, __file__.rsplit("\\", 1)[0])

from aic2026 import config as C

TU = [x.lower() for x in sys.argv[1:]]
if not TU:
    raise SystemExit("cần ít nhất một cụm từ")

cfg = C.load_config("config_xem.yaml")
kenh = ((cfg.get("retrieval") or {}).get("kenh_chu") or [])
duong = next(k["duong_dan"] for k in kenh if k.get("kieu") == "khoang")


def bo_dau(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s).lower())
                   if unicodedata.category(c) != "Mn")


TU_KD = [bo_dau(t) for t in TU]
hit = defaultdict(list)
with open(duong, encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        txt = r.get("asr_text_compact") or ""
        if not txt:
            continue
        b = bo_dau(txt)
        if all(t in b for t in TU_KD):
            hit[r.get("video_id", "?")].append(
                (float(r.get("start_pts_time_technical") or 0), txt))

print(f"khớp CẢ {TU}: {len(hit)} video")
for vid, ds in sorted(hit.items(), key=lambda kv: -len(kv[1]))[:10]:
    print(f"\n▸ {vid}  ({len(ds)} khoảng)")
    giay, txt = sorted(ds)[0]
    # in quanh cụm từ đầu tiên
    b = bo_dau(txt)
    i = b.find(TU_KD[0])
    print(f"   @{giay:.0f}s  …{txt[max(0, i - 120):i + 220]}…")
