#!/usr/bin/env python
"""TRA TIÊU ĐỀ/MÔ TẢ VIDEO — kênh chữ cấp video, 0 request, ~10 giây.

    python -X utf8 tra_metadata.py "nguyen trung truc"
    python -X utf8 tra_metadata.py "mang tay" "chien" --mo-ta

VÌ SAO ĐÁNG CÓ RIÊNG MỘT SCRIPT
-------------------------------
Bộ dữ liệu nhẹ của BTC có `media-info/<VIDEO>.json` (title 100%, keywords 100%).
Kênh chữ này KHÔNG được `Pipeline` dùng: `dataset.load_dataset(meta_vao_kenh_chu)`
mặc định TẮT vì đo ngày 08/08 cho thấy nó làm HỎNG điểm — nhưng phép đo đó chạy
trên dev set TIẾNG ANH, còn tiêu đề của BTC là TIẾNG VIỆT, nên hai bên gần như
không có token chung và RRF chỉ chèn nhiễu. Đề thi thật ra TIẾNG VIỆT, tức tiền
đề của phép đo đó không còn đúng.

Chưa đo lại được bằng dev set, nên KHÔNG bật nó trong đường nộp bài. Nhưng dùng
làm công cụ TRA CỨU cho người thì không cần đo: nó chỉ trả về danh sách video để
mắt người kiểm. Với đề dạng chương trình chuyên đề ("múa lân", "panna cotta",
"đình Nguyễn Trung Trực") tiêu đề nói thẳng nội dung.

Giới hạn phải biết: video BẢN TIN (60 Giây - HTV) có `description` là boilerplate
kênh, không liệt kê từng mẩu tin. Tra metadata VÔ DỤNG với mọi câu hỏi rơi vào
bản tin — đã kiểm tận mắt 20/08.
"""
import argparse
import os
import re
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026 import config as cau_hinh
from aic2026.core import dataset


def khong_dau(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s.replace("đ", "d"))


def _args():
    p = argparse.ArgumentParser(description="Tra tiêu đề video trong metadata BTC")
    p.add_argument("tu", nargs="+", help="các cụm từ (bỏ dấu cũng được); "
                                         "phải khớp HẾT")
    p.add_argument("--mo-ta", action="store_true", help="tìm cả trong description")
    p.add_argument("--config", default=None)
    p.add_argument("--sau", type=int, default=40)
    return p.parse_args()


def main():
    args = _args()
    config = cau_hinh.load_config(args.config)
    goc = (config.get("offline") or {}).get("btc_data_root")
    paths = dataset.scan(goc)
    can = [khong_dau(t) for t in args.tu]

    hit = []
    for vid in sorted(paths):
        vp = paths[vid]
        if vp.meta is None:
            continue
        try:
            m = dataset.load_media_info(vp.meta, vid)
        except Exception:                                     # noqa: BLE001
            continue
        kho = khong_dau(m.title + " " + " ".join(m.keywords))
        if args.mo_ta:
            kho += " " + khong_dau(m.description)
        if all(t in kho for t in can):
            hit.append((vid, m.title))

    print(f"{len(hit)} video khớp {args.tu} "
          f"({'title+keywords+description' if args.mo_ta else 'title+keywords'})\n")
    for vid, tieu in hit[:args.sau]:
        print(f"  {vid}  {tieu[:96]}")
    if len(hit) > args.sau:
        print(f"  ... còn {len(hit) - args.sau} video nữa")


if __name__ == "__main__":
    main()
