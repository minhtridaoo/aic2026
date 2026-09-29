#!/usr/bin/env python
"""Đo kênh chữ TIÊU ĐỀ trên nguyên văn câu đề tiếng Việt.

    python -X utf8 do_kenh_tieu_de.py --de data/de_dot1.json

VÌ SAO
------
Sơ tuyển đợt 1 được 3,8/12 trong khi vòng thử nghiệm được 8,8/12 trên CÙNG kho
873 video. Soi bài đã nộp thì 17/25 câu có top-20 rải ra 15–19 video khác nhau —
tầng ảnh không chọn nổi video nào. Mà tra tay thì ASR chỉ thẳng ra đáp án:
`query-p1-5-kis` ("đậu hà lan bỏ vào với mực đang xào") -> `L26_V035`, tiêu đề
đúng chữ **"MỰC XÀO ĐẬU HÀ LAN"**.

Nguyên nhân: L26 là ~480 tập *Món ngon mỗi ngày*, mọi tập đều là bếp, thớt,
chảo — tầng ảnh không có cách nào tách chúng ra. Thứ tách được là TÊN MÓN, và
tên món nằm nguyên trong `metadata/*.json` mà BTC cấp, chưa dùng đến dòng nào.

Script này KHÔNG sửa gì, chỉ trả lời một câu: nguyên văn câu đề tiếng Việt (dài,
tả cảnh, không nói tên món) có đủ để BM25 lôi đúng tiêu đề lên không — hay bắt
buộc phải có một tầng rút từ khoá đứng trước.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026.retrieval.kenh_chu import CauHinhKenhChu, KhoChuKhoang

GOC = os.path.dirname(os.path.abspath(__file__))


def _args():
    p = argparse.ArgumentParser(description="Đo kênh tiêu đề trên câu đề")
    p.add_argument("--de", default=os.path.join(GOC, "data", "de_dot1.json"))
    p.add_argument("--kenh", default=os.path.join(GOC, "data",
                                                  "kenh_chu_tieu_de.jsonl"))
    p.add_argument("--sau", type=int, default=5)
    p.add_argument("--cau", default="")
    p.add_argument("--chuan-hoa", default="chong_lan",
                   choices=["truy_van", "chong_lan"])
    p.add_argument("--nguong", type=float, default=0.30)
    return p.parse_args()


def main():
    a = _args()
    kho = KhoChuKhoang.tu_file(a.kenh, CauHinhKenhChu(
        dung_ocr=True, dung_asr=False,
        chuan_hoa_diem=a.chuan_hoa, nguong_diem=a.nguong))
    tieu = {k["video_id"]: str(k.get("ocr_text_compact") or "")
            for k in kho.khoang}
    print(f"{len(kho.khoang)} tiêu đề\n")

    with open(a.de, encoding="utf-8") as fh:
        de = json.load(fh)["queries"]
    chi = {x.strip() for x in a.cau.split(",") if x.strip()}

    for q in de:
        if chi and q["qid"] not in chi:
            continue
        text = q.get("text", "")
        print("=" * 92)
        print(f"▸ {q['qid']}  [{q['task']}]")
        print(f"  {text[:140]}")
        kq = kho.khoi_idf_khop(text)[:a.sau]
        if not kq:
            print("   (không khớp tiêu đề nào)")
            continue
        for i, khoi, ty in kq:
            vid = kho.khoang[i]["video_id"]
            print(f"   idf={khoi:6.2f}  tỉ lệ={ty:.3f}  {vid:10s}  "
                  f"{tieu.get(vid, '')[:64]}")


if __name__ == "__main__":
    main()
