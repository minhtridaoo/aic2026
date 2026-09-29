"""Tra nhiều bộ cụm từ trong ASR CẢ KHO trong MỘT lần nạp (khác `_tim_asr.py`
vốn nạp lại parquet mỗi lần gọi). Đọc {nhan: [cụm, ...]} từ JSON trên argv[1].
Mỗi bộ đòi khớp TẤT CẢ các cụm, giống `_tim_asr.py`."""
import json
import sys
import unicodedata
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from aic2026 import config as C

cfg = C.load_config("config_xem.yaml")
kenh = ((cfg.get("retrieval") or {}).get("kenh_chu") or [])
duong = next(k["duong_dan"] for k in kenh if k.get("kieu") == "khoang")


def bo_dau(s):
    s = unicodedata.normalize("NFD", str(s or "").lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn").replace("đ", "d")


import csv
csv.field_size_limit(1 << 30)
with open(duong, encoding="utf-8", newline="") as fh:
    rows = list(csv.DictReader(fh))

kho = []
for r in rows:
    txt = bo_dau(str(r.get("asr_text_compact") or "") + " || " + str(r.get("ocr_text_compact") or ""))
    kho.append((r.get("video_id"), r.get("start_pts_time_technical"), txt,
                str(r.get("asr_text_compact") or ""), str(r.get("ocr_text_compact") or "")))

bo = json.loads(open(sys.argv[1], encoding="utf-8").read())
for nhan, cum in bo.items():
    if nhan.startswith("_"):
        continue
    tu = [bo_dau(c) for c in cum]
    hit = defaultdict(list)
    for vid, t, txt, asr, ocr in kho:
        if all(c in txt for c in tu):
            hit[vid].append((t, asr, ocr))
    print(f"\n{'='*70}\n{nhan}   {cum}   -> {len(hit)} video")
    for vid, ds in sorted(hit.items(), key=lambda kv: -len(kv[1]))[:4]:
        t, asr, ocr = ds[0]
        print(f"  ▸ {vid} ({len(ds)} khoảng)  @{int(float(t or 0))}s")
        if asr.strip():
            print(f"      ASR: …{asr[:260]}…")
        if ocr.strip():
            print(f"      OCR: …{ocr[:180]}…")
