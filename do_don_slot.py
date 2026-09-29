#!/usr/bin/env python
"""ĐO LỚP DỒN SLOT trên DEV SET — nơi có ground truth CẤP KHUNG thật.

    python -X utf8 do_don_slot.py --config config_ocr_asr.yaml

VÌ SAO PHẢI ĐO Ở ĐÂY, KHÔNG ĐO TRÊN BỘ ĐỀ
------------------------------------------
Mọi phép đo trước đó (`do_trong_so_ocr.py`) dùng "hạng của khung ĐẦU TIÊN thuộc
video đích" — vì trên bộ đề ta chỉ biết VIDEO, không biết KHUNG. Với kênh
OCR/ASR thì thước đó đủ dùng, vì việc của chúng là tìm đúng video.

Lớp dồn slot thì khác hẳn: nó KHÔNG đổi video hạng 1: nó đổi cách tiêu 99 slot
còn lại. Đo bằng hạng-video sẽ cho ra "không đổi gì" ở mọi cấu hình — đúng loại
phép đo mù đã hai lần suýt làm vứt nhầm một hướng đúng trong ngày hôm nay.

Dev set 120 câu KIS có `gt.frame_start/frame_end` THẬT, nên chấm được bằng đúng
công thức của BTC: `Final = (1/5)·Σ_{k∈{1,5,20,50,100}} R@k`.

⚠ Giới hạn: truy vấn dev set là MÔ TẢ CẢNH người gõ, không phải đề BTC. Nó đo
đúng thứ cần đo ở đây (phân bổ slot) nhưng độ chính xác cấp video của nó thấp
hơn đề thật — nên lợi ích thật ngoài kia có thể CAO HƠN con số dưới đây.
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026 import config as cau_hinh
from aic2026.core.models import Query
from aic2026.optimize.don_slot import CauHinhDonSlot

GOC = os.path.dirname(os.path.abspath(__file__))
KS = (1, 5, 20, 50, 100)


def _args():
    p = argparse.ArgumentParser(description="Đo lớp dồn slot trên dev set")
    p.add_argument("--config", default=os.path.join(GOC, "config_ocr_asr.yaml"))
    p.add_argument("--so-cau", type=int, default=120)
    p.add_argument("--vi", action="store_true", default=True,
                   help="dùng bản mô tả tiếng Việt (đúng điều kiện thi)")
    return p.parse_args()


def cham(rows, gt):
    """Final của MỘT truy vấn theo đúng công thức BTC."""
    dung = [i for i, r in enumerate(rows, 1)
            if str(r["video_id"]).upper() == gt["video_id"].upper()
            and gt["frame_start"] <= int(r["frame_idx"]) <= gt["frame_end"]]
    if not dung:
        return 0.0
    dau = min(dung)
    return sum(1.0 for k in KS if dau <= k) / len(KS)


def main():
    a = _args()
    config = cau_hinh.load_config(a.config)
    print("Nạp dữ liệu ...")
    events, emb, metas, rep = cau_hinh.nap_du_lieu(config)
    print(f"  {rep.events:,d} event")

    with open(os.path.join(GOC, "data", "dev_set_tu_lam.json"),
              encoding="utf-8") as fh:
        qs = [q for q in json.load(fh)["queries"] if q.get("task") == "kis"]
    qs = qs[:a.so_cau]
    print(f"  {len(qs)} câu KIS dev (ground truth CẤP KHUNG)\n")

    p = cau_hinh.dung_pipeline(config, events, emb, qa_cheap=True)
    for kho, w in (p.kenh_chu or ()):
        n = len(getattr(kho, "_row2event", None) or {}) or sum(
            len(v) for v in (getattr(kho, "_event_theo_khoang", None)
                             or {}).values())
        print(f"  {type(kho).__name__:14s} w={w:<4g} {n:>8,d} event đã gắn")

    truy_van = [Query(qid=q["qid"], task="kis",
                      text=q.get("_mo_ta_vi") if a.vi else q["text"],
                      text_goc=q.get("_mo_ta_vi", ""))
                for q in qs]
    gts = [q["gt"] for q in qs]
    top_k = cau_hinh.so_dong_nop(config)

    # (nhãn, cấu-hình-dồn-slot, có-bật-kênh-chữ-không)
    kenh_goc = list(p.kenh_chu or ())
    CAU_HINH = [
        ("KHÔNG kênh chữ", CauHinhDonSlot(bat=False), False),
        ("tắt (nền)", CauHinhDonSlot(bat=False), True),
        # NHẸ TAY: tổng hạn ngạch < 100 nên phần còn lại vẫn là danh sách ứng
        # viên gốc — lưới an toàn ở đuôi được giữ. Bản 55/20/12/8/5 tiêu hết
        # 100 slot và làm R@100 sập 45% -> 29%.
        ("20/10/6 (36 slot)", CauHinhDonSlot(
            bat=True, han_ngach=(20, 10, 6), cach_chon="tron"), True),
        ("30/12/8 (50 slot)", CauHinhDonSlot(
            bat=True, han_ngach=(30, 12, 8), cach_chon="tron"), True),
        ("12/6 (18 slot)", CauHinhDonSlot(
            bat=True, han_ngach=(12, 6), cach_chon="tron"), True),
        ("20 (chỉ video 1)", CauHinhDonSlot(
            bat=True, han_ngach=(20,), cach_chon="tron"), True),
        ("30/12/8 · điểm", CauHinhDonSlot(
            bat=True, han_ngach=(30, 12, 8), cach_chon="diem"), True),
    ]

    print(f"\n{'cấu hình':>24s} | {'Final':>7s} {'±SE':>6s} | "
          f"{'R@1':>5s} {'R@100':>6s} |  Δ so nền")
    print("-" * 78)
    nen_diem = None
    for ten, cfg, co_kenh in CAU_HINH:
        p.don_slot_cfg = cfg
        p.retrieval_service.kenh_chu = kenh_goc if co_kenh else []
        t0 = time.time()
        diem = []
        r1 = r100 = 0
        for q, gt in zip(truy_van, gts):
            rows = p.answer(q, top_k=top_k)
            d = cham(rows, gt)
            diem.append(d)
            r1 += d == 1.0
            r100 += d > 0.0
        tb = statistics.fmean(diem)
        se = (statistics.stdev(diem) / len(diem) ** 0.5) if len(diem) > 1 else 0.0
        if nen_diem is None:
            nen_diem = diem
            dstr = "—"
        else:
            hieu = [x - y for x, y in zip(diem, nen_diem)]
            dtb = statistics.fmean(hieu)
            dse = (statistics.stdev(hieu) / len(hieu) ** 0.5) if len(hieu) > 1 else 0.0
            dau = "✅" if dtb > 2 * dse else ("❌" if dtb < -2 * dse else "≈")
            dstr = f"{dtb:+.4f} ± {dse:.4f} {dau}"
        print(f"{ten:>24s} | {tb:7.4f} {se:6.4f} | {r1 / len(diem):5.1%} "
              f"{r100 / len(diem):6.1%} |  {dstr}   ({time.time() - t0:.0f}s)")

    print("\n  Δ dùng SE GHÉP CẶP (cùng truy vấn, cùng hạt giống) nên nhạy hơn "
          "nhiều\n  so với trừ hai trung bình. ✅ = vượt 2·SE.")


if __name__ == "__main__":
    main()
