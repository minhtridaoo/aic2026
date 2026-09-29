"""Đổ toàn bộ OCR (Stage11 FTS) + ASR (Stage20) của MỘT video ra, lọc theo từ khoá.

    python -X utf8 _dao_video.py L30_V043 duong,quan tro
    python -X utf8 _dao_video.py L26_V461 ca,nguyen lieu

Dùng khi đã CHẮC video nhưng còn thiếu ĐÁP ÁN (câu QA hỏi tên riêng/con số).
"""
import csv
import sqlite3
import sys
import unicodedata

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, __file__.rsplit("\\", 1)[0])

from aic2026 import config as C

VIDEO = sys.argv[1].upper()
TU = [x.strip().lower() for x in (sys.argv[2] if len(sys.argv) > 2 else "").split(",") if x.strip()]

cfg = C.load_config("config_xem.yaml")
kenh = ((cfg.get("retrieval") or {}).get("kenh_chu") or [])
duong_ocr = next((k["duong_dan"] for k in kenh if k.get("kieu") == "fts"), None)
duong_asr = next((k["duong_dan"] for k in kenh if k.get("kieu") == "khoang"), None)


def bo_dau(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s).lower())
                   if unicodedata.category(c) != "Mn")


def khop(s):
    if not TU:
        return True
    b = bo_dau(s)
    return any(bo_dau(t) in b for t in TU)


print(f"=== OCR {VIDEO} ===")
con = sqlite3.connect(duong_ocr)
n = 0
for row in con.execute("SELECT frame_ref_id, text_norm FROM OCR_ALL "
                       "WHERE frame_ref_id LIKE ?", (f"%{VIDEO}%",)):
    if row[1] and khop(row[1]):
        print(f"  {row[0]}  {row[1][:190]}")
        n += 1
        if n >= 40:
            print("  … (cắt ở 40 dòng)")
            break
if not n:
    print("  (không khớp từ khoá nào)")

print(f"\n=== ASR {VIDEO} ===")
m = 0
with open(duong_asr, encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        if r.get("video_id", "").upper() != VIDEO:
            continue
        txt = r.get("asr_text_compact") or ""
        if txt and khop(txt):
            print(f"  @{float(r.get('start_pts_time_technical') or 0):7.1f}s  {txt[:900]}")
            m += 1
            if m >= 25:
                print("  … (cắt ở 25 dòng)")
                break
if not m:
    print("  (không khớp từ khoá nào)")
