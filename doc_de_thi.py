#!/usr/bin/env python
"""NGÀY THI, BƯỚC 1 — thư mục `.txt` của BTC -> file JSON cho `nop_bai.py`.

    python -X utf8 doc_de_thi.py /kaggle/input/de-thi --ra data/de_thi.json
    python -X utf8 doc_de_thi.py de_thi/ --ra data/de_thi.json --xem 5

Rồi:

    python -X utf8 nop_bai.py --config config_kaggle.yaml \
        --truy-van data/de_thi.json --ra out/nop_bai.zip

VÌ SAO PHẢI CÓ SCRIPT NÀY
-------------------------
`qid` lấy NGUYÊN VĂN tên file BTC, nên tên file CSV trong zip tự khớp. Tự gõ tay
JSON cho ~170 câu lúc đang chạy đua là cách chắc chắn nhất để sai `qid` — mà sai
`qid` thì Codabench không ghép được, câu đó **0 điểm và vẫn trừ một lần nộp**,
trong khi zip vẫn hợp lệ và mọi phép kiểm vẫn xanh.

ĐỌC KỸ PHẦN IN RA TRƯỚC KHI CHẠY `nop_bai.py`
---------------------------------------------
Chưa ai thấy file đề thật nên loại câu, câu hỏi QA và cách tách sự kiện TRAKE
đều là SUY ĐOÁN. Script in ra nó hiểu thế nào và cảnh báo mọi chỗ phải đoán.
Sửa tay file JSON là hoàn toàn hợp lệ — nó chỉ là điểm khởi đầu, không phải
phán quyết.
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026.core import de_bai, doc_de_txt

GOC = os.path.dirname(os.path.abspath(__file__))


def _args():
    p = argparse.ArgumentParser(
        description="Đọc thư mục đề .txt của BTC thành JSON cho nop_bai.py")
    p.add_argument("thu_muc", help="thư mục chứa các file đề .txt")
    p.add_argument("--ra", default=os.path.join(GOC, "data", "de_thi.json"))
    p.add_argument("--mau", default="*.txt", help="mẫu tên file (mặc định *.txt)")
    p.add_argument("--xem", type=int, default=3,
                   help="in nguyên văn N câu đầu để mắt người soi (mặc định 3)")
    return p.parse_args()


def main():
    args = _args()
    print("=" * 74)
    print("  ĐỌC ĐỀ THI — .txt của BTC -> JSON")
    print("=" * 74)
    print(f"  thư mục: {args.thu_muc}")

    muc, ban_ghi = doc_de_txt.doc_thu_muc(args.thu_muc, args.mau)
    dem = Counter(m["task"] for m in muc)
    print(f"  đọc được: {len(muc)} câu  ·  "
          + " · ".join(f"{k} {v}" for k, v in sorted(dem.items())))
    print()

    for bg in ban_ghi:
        print("  " + bg.mo_ta())

    canh_bao = [(bg.qid, c) for bg in ban_ghi for c in bg.canh_bao]
    if canh_bao:
        print(f"\n  ⚠ {len(canh_bao)} cảnh báo — ĐỌC HẾT trước khi chạy nộp bài:")
        for _, c in canh_bao:
            print(f"     {c}")

    # Tên file nộp là chỗ mất điểm câm; kiểm lại ngay tại đây.
    ten = {m["qid"]: de_bai.ten_file_nop(m["qid"], m.get("ten_file", ""))
           for m in muc}
    lech = [c for c in (de_bai.canh_bao_ten_file(t) for t in ten.values()) if c]
    print(f"\n  tên file nộp: "
          + ("✅ cả %d file khớp quy ước BTC" % len(ten) if not lech
             else f"⚠ {len(lech)}/{len(ten)} LỆCH"))
    for c in lech[:5]:
        print(f"     {c}")

    if args.xem > 0:
        print(f"\n  {min(args.xem, len(muc))} câu đầu, nguyên văn — soi bằng mắt:")
        for m in muc[:args.xem]:
            print(f"\n    [{m['qid']}] task={m['task']}")
            if m.get("sub_events"):
                for i, se in enumerate(m["sub_events"], 1):
                    print(f"       {i}. {se[:100]}")
            else:
                print(f"       mô tả : {str(m.get('text',''))[:150]}")
            if m.get("_cau_hoi_vi"):
                print(f"       hỏi   : {m['_cau_hoi_vi'][:120]}")

    p = doc_de_txt.ghi_json(muc, args.ra)
    print(f"\n  -> {p}")
    print(f"\n  Chạy tiếp:\n"
          f"    python -X utf8 nop_bai.py --config config_kaggle.yaml \\\n"
          f"        --truy-van {p} --ra out/nop_bai.zip")
    if canh_bao or lech:
        print("\n  ⚠ CÓ CẢNH BÁO Ở TRÊN. Mở file JSON sửa tay nếu cần — nó chỉ "
              "là điểm khởi đầu.")


if __name__ == "__main__":
    main()
