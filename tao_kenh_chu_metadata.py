#!/usr/bin/env python
"""Dựng file kênh chữ dạng Stage20 TỪ METADATA BTC — phương án dự phòng.

    python -X utf8 tao_kenh_chu_metadata.py --ra data/kenh_chu_metadata.jsonl

VÌ SAO CÓ SCRIPT NÀY
--------------------
Kênh chữ thật (OCR + ASR) nằm ở dataset `nguyenminhtric/20-note` của đội bạn.
Nếu vì lý do gì không add được dataset đó vào Kaggle thì cả tính năng kênh chữ
nằm im. Script này dựng một nguồn text THAY THẾ từ thứ chắc chắn có trong tay:
`media-info/<VIDEO>.json` mà BTC cấp (title 100%, keywords 100%).

Nó KHÔNG tương đương ASR. Phải nói rõ giới hạn, vì đã kiểm tận mắt ngày 20/08:

  * Với chương trình chuyên đề (Món ngon mỗi ngày, Việt Nam đi là ghiền, Lân Sư
    Rồng, Lan tỏa năng lượng tích cực) thì TIÊU ĐỀ nói thẳng nội dung — tra
    "panna cotta" ra đúng 1/873 video, "nguyen trung truc" ra đúng 1/873.
  * Với BẢN TIN *60 Giây* thì `description` chỉ là boilerplate quảng cáo kênh,
    không liệt kê từng mẩu tin. Kênh này VÔ DỤNG với đúng 6 câu khó nhất
    (1, 2, 8, 14, 21, 23) — mà đó lại là nhóm ASR thật sẽ cứu được.

Nên: dùng khi chưa có 20-note, và bỏ ngay khi có.

MỘT KHOẢNG = MỘT VIDEO
----------------------
Metadata không có mốc thời gian, nên mỗi video thành đúng một khoảng phủ toàn
bộ trục thời gian. Hệ quả trực tiếp: kênh này chỉ định vị được tới mức VIDEO,
không tới mức khung. Đó cũng chính là lý do `toi_da_event_moi_khoang` quan
trọng — không giới hạn thì một video trúng sẽ nuốt sạch pool.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026 import config as cau_hinh
from aic2026.core import dataset

GOC = os.path.dirname(os.path.abspath(__file__))


def _args():
    p = argparse.ArgumentParser(
        description="Metadata BTC -> file kênh chữ dạng Stage20")
    p.add_argument("--config", default=None)
    p.add_argument("--ra", default=os.path.join(GOC, "data",
                                                "kenh_chu_metadata.jsonl"))
    p.add_argument("--kem-mo-ta", action="store_true",
                   help="nhét cả `description` (đầy link/hashtag — mặc định BỎ)")
    # `keywords` của YouTube là thẻ đặt cho CẢ KÊNH, không phải cho tập phim:
    # 480+ video L26 dùng chung đúng một danh sách ('VIVU TV', 'món ngon mỗi
    # ngày', 'ẩm thực', ...). IDF dập chúng về gần 0 nên chúng không thêm tín
    # hiệu, nhưng chúng kéo dài văn bản gấp ~10 lần tiêu đề, mà BM25 chuẩn hoá
    # theo độ dài — nên tác dụng ròng là DÌM đúng cái tiêu đề đang muốn khớp.
    p.add_argument("--chi-tieu-de", action="store_true",
                   help="chỉ dùng `title`, bỏ `keywords` (nên bật với L26)")
    return p.parse_args()


def main():
    a = _args()
    config = cau_hinh.load_config(a.config)
    goc = (config.get("offline") or {}).get("btc_data_root")
    paths = dataset.scan(goc)

    rows, thieu, co_kw = [], 0, 0
    for vid in sorted(paths):
        vp = paths[vid]
        if vp.meta is None:
            thieu += 1
            continue
        try:
            m = dataset.load_media_info(vp.meta, vid)
        except Exception:                                     # noqa: BLE001
            thieu += 1
            continue
        # `title` + `keywords` vào kênh OCR (chữ hiện trên màn hình gần với tiêu
        # đề chương trình hơn), `author` + `description` vào ASR nếu bật.
        ocr = (m.title if a.chi_tieu_de
               else " ".join([m.title, " ".join(m.keywords)])).strip()
        asr = m.author or ""
        if a.kem_mo_ta:
            asr = (asr + " " + m.description).strip()
        if m.keywords:
            co_kw += 1
        rows.append({
            "video_id": vid,
            "start_pts_time_technical": 0.0,
            # Phủ trọn trục thời gian: metadata không có mốc, nên mọi keyframe
            # của video đều thuộc khoảng này.
            "end_pts_time_technical": 1e9,
            "ocr_text_compact": ocr,
            "asr_text_compact": asr,
        })

    os.makedirs(os.path.dirname(os.path.abspath(a.ra)), exist_ok=True)
    with open(a.ra, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  {len(rows)} video -> {a.ra}")
    print(f"  có keywords: {co_kw}  ·  thiếu metadata: {thieu}")
    print(f"\n  Bật trong config.yaml:\n"
          f"    retrieval:\n      kenh_chu:\n        bat: true\n"
          f"        duong_dan: '{os.path.abspath(a.ra)}'")
    print(f"\n  ⚠ Đây là NGUỒN THAY THẾ, không phải ASR. Vô dụng với các câu rơi"
          f" vào bản tin\n     60 Giây (description chỉ là quảng cáo kênh). Có "
          f"20-note thì bỏ nó đi.")


if __name__ == "__main__":
    main()
