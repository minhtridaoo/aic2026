#!/usr/bin/env python
"""DÒ TRỌNG SỐ kênh OCR trên chính bộ đề, dùng đích đã XÁC MINH BẰNG OCR.

    python -X utf8 do_trong_so_ocr.py --config config_ocr.yaml

VÌ SAO ĐO ĐƯỢC MÀ KHÔNG CẦN GROUND TRUTH CỦA BTC
------------------------------------------------
Với 7 câu dưới đây, OCR đọc được NGUYÊN VĂN nội dung đề hỏi ngay trên khung
hình — "cá mập trở lại bờ biển Cape Cod", "lễ hội điêu khắc trên cát tại
St. Petersburg", "xã Giang Ly, huyện Khánh Vĩnh, tỉnh Khánh Hòa". Đó không
phải phỏng đoán mà là chữ đọc được từ pixel, nên VIDEO đích coi như đã biết.

Chưa biết KHUNG chính xác trong video, nên thước đo ở đây là **hạng của khung
ĐẦU TIÊN thuộc video đích** trong 100 dòng nộp. Quy về điểm theo đúng thang của
BTC (hạng 1 -> 1.0 · 2-5 -> 0.8 · 6-20 -> 0.6 · 21-50 -> 0.4 · 51-100 -> 0.2).

Đây là CẬN TRÊN của điểm: đúng video vẫn có thể sai khung. Nhưng nó so được
giữa các trọng số, và đó là việc cần làm.

Chạy KHÔNG có ghim — mục đích là xem tầng tìm tự làm được tới đâu.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026 import config as cau_hinh
from aic2026.core import viet_lai_truy_van
from aic2026.core.de_bai import chon_cau_hoi, chon_van_ban
from aic2026.core.models import Query

GOC = os.path.dirname(os.path.abspath(__file__))

# Video đích, xác minh bằng chữ OCR đọc được trên khung hình (21/08).
DICH = {
    "query-p1-2-kis":  ["L21_V029"],
    "query-p1-8-kis":  ["L22_V030", "L22_V027"],
    "query-p1-14-kis": ["L21_V027"],
    "query-p1-15-qa":  ["L30_V072"],
    "query-p1-19-qa":  ["L27_V010", "L24_V011"],
    "query-p1-22-qa":  ["L26_V178"],
    "query-p1-23-kis": ["L22_V022"],
}
# NHÓM HAI — những câu tầng ảnh VỐN ĐÃ chọn đúng video (soi tiêu đề 20/08).
# Nhóm này đo MẶT HẠI: trọng số OCR quá to sẽ đẩy chúng xuống. Đo một mình
# nhóm trên là tự lừa, vì đó đúng 7 câu OCR mạnh nhất.
GIU = {
    "query-p1-5-kis":  ["L27_V014"],   # Một ngày làm nông dân chăn dê
    "query-p1-6-kis":  ["L26_V056"],   # CUỐN CHAY ĐẬU HỦ RAU RỪNG
    "query-p1-9-kis":  ["L27_V013"],   # Trăm Năm làng Khóm
    "query-p1-10-kis": ["L30_V017"],   # trống Hang-drum
    "query-p1-11-kis": ["L30_V057"],   # mảnh bìa đổ bóng
    "query-p1-12-kis": ["L26_V279", "L26_V200"],   # bánh rán (2 ứng viên)
    "query-p1-20-kis": ["L26_V004"],   # PANNA COTTA KEM MUỐI
}
MOC = ((1, 1.0), (5, 0.8), (20, 0.6), (50, 0.4), (100, 0.2))


def diem_tu_hang(h):
    if h is None:
        return 0.0
    return next((d for k, d in MOC if h <= k), 0.0)


def _args():
    p = argparse.ArgumentParser(description="Dò trọng số kênh OCR")
    p.add_argument("--config", default=os.path.join(GOC, "config_ocr.yaml"))
    p.add_argument("--de", default=os.path.join(GOC, "data", "de_thunghiem.json"))
    p.add_argument("--trong-so", default="0,0.5,1.0,1.5,2.0",
                   help="HỆ SỐ NHÂN lên trọng số trong config, "
                        "không phải trọng số tuyệt đối")
    return p.parse_args()


def main():
    a = _args()
    config = cau_hinh.load_config(a.config)

    print("Nạp dữ liệu ...")
    t0 = time.time()
    events, emb, metas, rep = cau_hinh.nap_du_lieu(config)
    print(f"  {rep.events:,d} event ({time.time() - t0:.0f}s)")

    with open(a.de, encoding="utf-8") as fh:
        de = {q["qid"]: q for q in json.load(fh)["queries"]}

    chinh_sach = cau_hinh.ngon_ngu_truy_van(config)
    TAT_CA = {**DICH, **GIU}
    truy_van, ban_ghi = [], []
    for qid in TAT_CA:
        q = de[qid]
        text, bg = chon_van_ban(q, chinh_sach, qid)
        truy_van.append(Query(qid=qid, task=q["task"], text=text,
                              question_vi=chon_cau_hoi(q)))
        ban_ghi.append(bg)

    bo, vi_sao = viet_lai_truy_van.tu_config(config)
    if bo is not None:
        print(f"Viết lại truy vấn bằng {vi_sao} ...")
        for d in viet_lai_truy_van.ap_dung(bo, truy_van, ban_ghi):
            print(f"  {d}")

    p = cau_hinh.dung_pipeline(config, events, emb, qa_cheap=True)
    # 🔴 PHẢI lấy instance mà CHÍNH pipeline đã dựng. `dung_pipeline` gọi
    # `nap_kenh_chu` lần nữa và `p.index()` chỉ gọi `gan_event` trên instance
    # đó. Gán đè bằng instance riêng ở trên là gán một kho CHƯA gắn event —
    # `xep_hang_event` trả rỗng và mọi trọng số cho ra cùng một số, trông y hệt
    # "kênh OCR vô dụng". Đã dẫm phải đúng bẫy này lúc 11:00 ngày 21/08.
    kenh = list(p.kenh_chu or ())
    if not kenh:
        raise SystemExit("Không kênh chữ nào bật — xem `retrieval.kenh_chu`.")
    for _k, _w in kenh:
        n = len(getattr(_k, "_row2event", None) or {}) or sum(
            len(v) for v in (getattr(_k, "_event_theo_khoang", None)
                             or {}).values())
        print(f"  {type(_k).__name__:14s} w={_w:<4g} {n:>8,d} event đã gắn")
        if not n:
            raise SystemExit("Kênh chữ chưa gắn được event — đừng đo mù.")
    print()
    top_k = cau_hinh.so_dong_nop(config)
    ws = [float(x) for x in a.trong_so.split(",")]

    def _so(q):
        return q.split("-")[2]

    print(f"\n{'w':>4s} | " + " ".join(f"{_so(q):>3s}" for q in DICH)
          + " ‖ " + " ".join(f"{_so(q):>3s}" for q in GIU)
          + " |  OCR-mạnh   đã-đúng      TỔNG")
    print("-" * 94)
    bang = {}
    for w in ws:
        # `w` là HỆ SỐ NHÂN lên trọng số trong config, không phải trọng số
        # tuyệt đối — nhờ vậy tỷ lệ giữa OCR và ASR được giữ nguyên khi dò.
        # w = 0 -> tắt hết; w > 0 -> NHÂN trọng số gốc của từng kênh, giữ
        # nguyên TỶ LỆ giữa OCR và ASR (chúng có trọng số riêng trong config).
        p.retrieval_service.kenh_chu = (
            [] if w == 0 else [(k, w0 * w) for k, w0 in kenh])
        hang = {}
        for q in truy_van:
            rows = p.answer(q, top_k=top_k)
            muon = {v.upper() for v in TAT_CA[q.qid]}
            hang[q.qid] = next((i for i, r in enumerate(rows, 1)
                                if str(r["video_id"]).upper() in muon), None)
        d1 = sum(diem_tu_hang(hang[q]) for q in DICH)
        d2 = sum(diem_tu_hang(hang[q]) for q in GIU)
        bang[w] = (hang, d1, d2)
        print(f"{w:4g} | " + " ".join(f"{(hang[q] or 0):>3d}" for q in DICH)
              + " ‖ " + " ".join(f"{(hang[q] or 0):>3d}" for q in GIU)
              + f" | {d1:5.1f}/{len(DICH)}  {d2:5.1f}/{len(GIU)}"
              f"  {d1 + d2:6.1f}/{len(TAT_CA)}")

    print("\n  (ô = HẠNG khung đầu tiên thuộc video đích; 0 = ngoài 100 dòng)")
    print("  trái ‖ = 7 câu OCR mạnh nhất   ·   phải = 7 câu tầng ảnh vốn đã đúng")
    tot = max(bang, key=lambda k: bang[k][1] + bang[k][2])
    n1, n2 = bang.get(0.0, ({}, 0.0, 0.0))[1], bang.get(0.0, ({}, 0.0, 0.0))[2]
    print(f"\n  nền  w=0     : OCR-mạnh {n1:.1f} · đã-đúng {n2:.1f} · "
          f"tổng {n1 + n2:.1f}")
    print(f"  tốt nhất w={tot:<4g} : OCR-mạnh {bang[tot][1]:.1f} · đã-đúng "
          f"{bang[tot][2]:.1f} · tổng {bang[tot][1] + bang[tot][2]:.1f}"
          f"   (+{bang[tot][1] + bang[tot][2] - n1 - n2:.1f})")
    print("\n  ⚠ Đây là CẬN TRÊN: đúng video vẫn có thể sai khung. Và 7 câu này "
          "là 7 câu\n     OCR vốn mạnh nhất — đừng suy ra cho cả đề.")


if __name__ == "__main__":
    main()
