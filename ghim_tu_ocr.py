#!/usr/bin/env python
"""SINH FILE GHIM TỪ CHÍNH BẰNG CHỨNG OCR/ASR — tự động hoá vòng lặp người.

    python -X utf8 ghim_tu_ocr.py --config config_ocr_asr.yaml --ra data/ghim_tu_ocr.json

VÌ SAO
------
Ghim theo VIDEO mới giải một nửa bài toán. Cửa sổ chấm KIS chỉ ~107 khung
(~4,3 giây ≈ 2 keyframe), nên rải 60 dòng đều khắp một video 290 keyframe vẫn
chỉ ~38% cơ hội trúng.

Nhưng khi OCR đọc được đúng dòng chữ đề hỏi, ta biết luôn KHUNG NÀO — mẩu tin
đó nằm ngay quanh khung ấy. Ghim `khung` (đúng frame_idx) mạnh hơn hẳn ghim
`videos` (rải đều).

Script này: với mỗi câu + từ khoá người cấp, lấy top khung OCR, NỞ ra vài
keyframe hai bên (mẩu tin dài vài giây), rồi xuất file ghim kèm nguyên văn chữ
OCR vào `ghi_chu` để kiểm lại được.

⚠ Nó KHÔNG tự quyết đúng/sai. Người phải đọc `ghi_chu` và bỏ mục nào vô lý.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026 import config as cau_hinh
from aic2026.core import dataset

GOC = os.path.dirname(os.path.abspath(__file__))

# 🔴 BÀI HỌC TRẢ GIÁ 0,4 ĐIỂM (nộp lượt #7 ngày 21/08: 7,6 -> 7,2).
#
# Bản đầu của script này ghim KHUNG cho mọi câu OCR khớp. Sai, vì OCR bắt được
# HAI LOẠI CHỮ hoàn toàn khác nhau:
#
#   DÒNG CHẠY BẢN TIN — có đồng hồ, liệt kê nhiều tin cùng lúc:
#       "ftvo 06:44:12 TIẾP THEO lễ hội điêu khắc trên cát tại st. petersburg"
#       "ftvo 06:36:31 đồng nai: 2 hổ mẹ bengal... | bạc liêu: 10 | tp.hcm:..."
#     Chữ này chỉ nói ĐÚNG VIDEO, KHÔNG nói đúng khoảnh khắc. Câu 14 ghi thẳng
#     "TIẾP THEO" — phóng sự còn ở phía sau, có khi vài phút. Ghim khung ở đây
#     là ghim nhầm thời điểm, lại còn đẩy 12-16 khung của máy xuống dưới.
#
#   CHÚ THÍCH CỦA CHÍNH CẢNH ĐÓ — không đồng hồ:
#       "fana clb nắng ấm yêu thương ... xã giang ly, huyện khánh vĩnh"
#       "nguyên liệu · tôm 200g · thịt nạc dăm xay 200g"
#     Chữ này gắn với khung đang chiếu. Ghim khung ở đây MỚI đúng.
#
# Nên: có đồng hồ -> chỉ ghim VIDEO. Không đồng hồ -> ghim cả KHUNG.
_DONG_HO = re.compile(r"\b\d{1,2}\s*[:.]\s*\d{2}\s*[:.]\s*\d{2}\b")
_SAP_TOI = re.compile(r"(?i)\bti[êế]p\s*theo\b")


def la_dong_chay_ban_tin(text: str) -> str:
    """'' nếu là chú thích cảnh; ngược lại trả LÝ DO không được ghim khung."""
    t = str(text or "")
    if _SAP_TOI.search(t):
        return "chữ ghi 'TIẾP THEO' — phóng sự còn ở phía sau"
    if _DONG_HO.search(t):
        return "chữ có đồng hồ -> dòng chạy bản tin, không gắn với cảnh đang chiếu"
    return ""

# (qid, từ khoá tra OCR, số keyframe nở về trước, nở về sau)
# Từ khoá do NGƯỜI cấp — đây là chỗ tri thức ngoài đi vào ("phim Spielberg
# 1975" -> "cá mập"), thứ mà encoder ảnh lẫn OCR đều không tự suy ra.
SAN = [
    ("query-p1-2-kis",  "hổ mẹ Bengal quý hiếm sinh con khu du lịch", 3, 8),
    ("query-p1-8-kis",  "lễ hội ẩm thực Nhật Bản lớn nhất thế giới", 3, 8),
    ("query-p1-14-kis", "lễ hội điêu khắc trên cát St. Petersburg", 3, 8),
    ("query-p1-23-kis", "cá mập trở lại bờ biển thu hút du khách hiếu kỳ", 3, 8),
    ("query-p1-15-qa",  "FANA câu lạc bộ nắng ấm yêu thương đến trường", 3, 8),
    ("query-p1-22-qa",  "nguyên liệu thịt nạc dăm xay 200g", 3, 10),
    ("query-p1-19-qa",  "khu di tích lịch sử văn hóa mộ đình Nguyễn Trung Trực",
     3, 10),
    ("query-p1-17-kis", "đoàn viên mừng xuân 2024 bệnh viện", 6, 10),
]

# Câu mà ASR (không phải OCR) mới là nguồn — cấp thẳng (video, giây).
# ASR khác OCR-dòng-chạy ở chỗ nó là LỜI NÓI TRÊN CHÍNH ĐOẠN HÌNH ĐÓ, nên gắn
# đúng thời điểm. Và ở câu này bất đối xứng rất có lợi: khung đúng đang nằm ở
# hạng 52/61/79 tức chỉ 0,2 điểm, nên ghim sai cũng gần như không mất gì
# (8 khung đẩy phần còn lại xuống 8 bậc, vẫn trong 100), còn ghim đúng thì 1,0.
SAN_ASR = [
    ("query-p1-1-kis", "L21_V015", 839.0, 25.0,
     "ASR: 'hãng SpaceX cho biết họ đang hướng đến phá vỡ giới hạn khám phá "
     "CÁC CỰC TRÁI ĐẤT' — khớp 'tàu vũ trụ tư nhân' + 'nghiên cứu ánh sáng "
     "cực quang ở vùng cực' trong đề. OCR không bắt được câu này."),
]


def _args():
    p = argparse.ArgumentParser(description="Sinh file ghim từ bằng chứng OCR/ASR")
    p.add_argument("--config", default=os.path.join(GOC, "config_ocr_asr.yaml"))
    p.add_argument("--ra", default=os.path.join(GOC, "data", "ghim_tu_ocr.json"))
    p.add_argument("--so-dong-video", type=int, default=40,
                   help="ghim thêm bao nhiêu dòng cho VIDEO làm lưới an toàn")
    return p.parse_args()


def main():
    a = _args()
    config = cau_hinh.load_config(a.config)
    kenh = cau_hinh.nap_kenh_chu(config)
    kho_ocr = next((k for k, _ in kenh if type(k).__name__ == "KhoOcrFts"), None)
    if kho_ocr is None:
        raise SystemExit("Cần kênh OCR (kieu: fts) trong config")

    paths = dataset.scan(config["offline"]["btc_data_root"])
    moc, tong, tieu = [], 0, {}
    for vid in sorted(paths):
        vp = paths[vid]
        if not vp.du_toi_thieu:
            continue
        rows = dataset.load_map_keyframes(vp.mapping)
        moc.append((tong, tong + len(rows), vid, rows))
        tong += len(rows)
        if vp.meta is not None:
            try:
                tieu[vid] = dataset.load_media_info(vp.meta, vid).title
            except Exception:                                  # noqa: BLE001
                pass
    theo_vid = {vid: rows for _, _, vid, rows in moc}

    def tra_row(fr):
        for x, y, vid, rows in moc:
            if x <= fr < y:
                return vid, rows[fr - x]
        return None, None

    ghim = {"_ghi_chu": ("Sinh bằng `ghim_tu_ocr.py` ngày 21/08. Mỗi mục kèm "
                         "NGUYÊN VĂN chữ OCR/ASR đọc được — đọc `ghi_chu` "
                         "trước khi tin.")}

    for qid, tu_khoa, truoc, sau in SAN:
        kq = kho_ocr.cham(tu_khoa)[:1]
        if not kq:
            print(f"  {qid:22s} ⚠ OCR không khớp {tu_khoa!r} — BỎ")
            continue
        fr, diem = kq[0]
        vid, r0 = tra_row(fr)
        van_ban = kho_ocr.van_ban(fr)
        vi_sao_khong = la_dong_chay_ban_tin(van_ban)

        muc = {"videos": [vid],
               "so_dong": a.so_dong_video if vi_sao_khong else 40,
               "ghi_chu": (f"OCR({diem:.2f}) tại {vid} kf={r0.n}: "
                           f"\"{van_ban[:120]}\" · video: "
                           f"{tieu.get(vid, '?')[:60]}")}
        if vi_sao_khong:
            muc["so_dong"] = max(a.so_dong_video, 60)
            muc["ghi_chu"] += (f" · CHỈ ghim VIDEO, không ghim khung: "
                               f"{vi_sao_khong}.")
            print(f"  {qid:22s} {vid} kf={r0.n} -> chỉ VIDEO  "
                  f"({vi_sao_khong[:44]})")
        else:
            rows = theo_vid[vid]
            i = next(i for i, r in enumerate(rows) if r.n == r0.n)
            cum = rows[max(0, i - truoc): i + sau + 1]
            muc["khung"] = [f"{vid}|{int(r.frame_idx)}" for r in cum]
            muc["ghi_chu"] += (f" · chữ là CHÚ THÍCH CẢNH (không đồng hồ) nên "
                               f"ghim khung, nở {truoc}+{sau} keyframe hai bên.")
            print(f"  {qid:22s} {vid} kf={r0.n} -> {len(cum)} KHUNG  "
                  f"[{tieu.get(vid, '?')[:40]}]")
        ghim[qid] = muc

    for qid, vid, giay, quanh, vi_sao in SAN_ASR:
        rows = theo_vid.get(vid)
        if not rows:
            continue
        cum = [r for r in rows if abs(float(r.pts_time) - giay) <= quanh]
        ghim[qid] = {
            "khung": [f"{vid}|{int(r.frame_idx)}" for r in cum],
            "videos": [vid],
            "so_dong": a.so_dong_video,
            "ghi_chu": f"{vi_sao} Khung quanh {giay:.0f}s ±{quanh:.0f}s.",
        }
        print(f"  {qid:22s} {vid} @{giay:.0f}s -> {len(cum)} khung  (ASR)")

    # Giữ nguyên các ghim TRAKE / đáp án QA từ file cũ.
    cu_path = os.path.join(GOC, "data", "ghim_thunghiem_ocr.json")
    if os.path.exists(cu_path):
        with open(cu_path, encoding="utf-8") as fh:
            cu = json.load(fh)
        for qid, m in cu.items():
            if qid.startswith("_"):
                continue
            if qid in ghim:
                # Giữ đáp án QA đã xác minh; khung mới thắng video cũ.
                if m.get("answer"):
                    ghim[qid]["answer"] = m["answer"]
                    ghim[qid]["ghi_chu"] += f" ĐÁP ÁN giữ từ bản cũ."
            else:
                ghim[qid] = m
                print(f"  {qid:22s} (giữ nguyên từ bản cũ)")

    with open(a.ra, "w", encoding="utf-8") as fh:
        json.dump(ghim, fh, ensure_ascii=False, indent=2)
    n = len([k for k in ghim if not k.startswith("_")])
    print(f"\n  -> {a.ra}  ({n} truy vấn)")
    print("  ⚠ MỞ FILE ĐỌC `ghi_chu` trước khi nộp. Bỏ mục nào chữ OCR không "
          "ăn nhập với đề.")


if __name__ == "__main__":
    main()
