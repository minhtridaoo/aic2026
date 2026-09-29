#!/usr/bin/env python
"""Sáu thí nghiệm nhỏ, chạy trên chính code thật, để thấy hệ thống làm gì.

    python giai_thich.py          # chạy hết
    python giai_thich.py 3        # chỉ chạy thí nghiệm số 3

Mỗi thí nghiệm in ra: làm gì -> con số -> rút ra điều gì. Không có gì bịa;
tất cả gọi thẳng vào aic2026/, xem HUONG_DAN.md để đọc kèm.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026.core import scoring, submission as sub
from aic2026.core.ingestion import sliding_windows
from aic2026.core.models import Frame, GroundTruth
from aic2026.core.offline_eval import load_benchmark
from aic2026.core.pipeline import Pipeline, PortfolioConfig
from aic2026.core.providers import HashEmbedder, RuleBasedProvider
from aic2026.optimize.calibration import RankCalibrator

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "data", "benchmark_queries.sample.json")


def tieu_de(so, ten):
    print("\n" + "=" * 72)
    print(f"  THÍ NGHIỆM {so}: {ten}")
    print("=" * 72)


def ket_luan(*dong):
    print("\n  >> RÚT RA:")
    for d in dong:
        print(f"     {d}")


# --------------------------------------------------------------------------- #
def tn1_vi_tri_quyet_dinh_diem():
    tieu_de(1, "Cùng một đáp án đúng, đặt ở vị trí khác nhau -> điểm khác nhau")
    print("""
  Ta nộp 100 dòng. Trong đó đúng 1 dòng là đáp án đúng. Ta thử đặt dòng
  đúng đó ở các vị trí khác nhau rồi xem BTC chấm bao nhiêu điểm.
""")
    gt = GroundTruth(task="kis", video_id="L01", frame_start=100, frame_end=200)
    dung = {"video_id": "L01", "frame_idx": 150}
    sai = {"video_id": "LX", "frame_idx": 0}

    print(f"  {'đặt ở slot':>12s} | {'R@1':>5s} {'R@5':>5s} {'R@20':>5s} "
          f"{'R@50':>5s} {'R@100':>6s} | {'ĐIỂM':>6s}")
    print("  " + "-" * 62)
    for vi_tri in (1, 3, 10, 30, 80, 101):
        rows = [sai] * (vi_tri - 1) + [dung] + [sai] * 120
        rows = rows[:100] if vi_tri <= 100 else [sai] * 100
        tk = scoring.top_k_r_scores(rows, gt)
        diem = scoring.final_score(rows, gt)
        print(f"  {vi_tri:12d} | " + " ".join(f"{tk[k]:5.0f}" for k in
              (1, 5, 20, 50, 100)) + f"  | {diem:6.2f}")

    ket_luan(
        "Slot 1 ăn trọn 1.00 điểm. Slot 30 chỉ còn 0.40. Slot 101 = 0.",
        "Vì R@k lấy MAX trên k dòng đầu: dòng ở slot 30 chỉ cứu được k=50,100.",
        "=> Đặt đúng thứ tự quan trọng ngang với tìm đúng.")


# --------------------------------------------------------------------------- #
def tn2_trung_lap_khong_them_diem():
    tieu_de(2, "Nộp 100 dòng na ná nhau = phí 99 dòng")
    print("""
  Hai chiến lược, cùng 100 dòng:
    A) 100 khung hình của CÙNG một cảnh trong CÙNG một video
    B) 1 khung của 100 video khác nhau
  Giả sử sự thật là video L50. Xem ai ăn điểm.
""")
    gt = GroundTruth(task="kis", video_id="L50", frame_start=100, frame_end=200)

    a = [{"video_id": "L01", "frame_idx": 100 + i} for i in range(100)]
    b = [{"video_id": f"L{i:02d}", "frame_idx": 150} for i in range(1, 101)]

    print(f"  A) 100 khung của video L01  -> điểm = {scoring.final_score(a, gt):.2f}")
    print(f"  B) 1 khung của 100 video    -> điểm = {scoring.final_score(b, gt):.2f}")
    print(f"     (trong B, L50 nằm ở slot 50)")

    ket_luan(
        "A đặt cược tất cả vào một cửa. Sai cửa là mất trắng.",
        "B phủ 100 cửa. Điểm thấp hơn slot-1 nhưng KHÔNG bao giờ về 0.",
        "=> 100 slot là DANH MỤC ĐẦU TƯ, không phải bảng xếp hạng.")


# --------------------------------------------------------------------------- #
def tn3_event_la_gi():
    tieu_de(3, "'Event' là gì — đơn vị mà hệ thống đi tìm")
    print("""
  Hệ thống KHÔNG tìm từng khung hình lẻ. Nó cắt video thành các cửa sổ
  10 giây, chồng lên nhau 2 giây, gọi mỗi cửa sổ là một 'event'.
  Lý do: một khung hình tĩnh không kể được 'ai làm gì', một đoạn 10s thì có.
""")
    frames = [Frame(video_id="L01_V001", frame_idx=i * 25, pts_time=float(i))
              for i in range(31)]  # 31 giây, 25 fps
    events = sliding_windows(frames, window=10.0, overlap=2.0)

    print(f"  Đầu vào : {len(frames)} khung hình (31 giây, 25 fps)")
    print(f"  Đầu ra  : {len(events)} event\n")
    print(f"  {'event_id':<18s} {'thời gian':>14s} {'khung':>13s} {'keyframe':>9s}")
    print("  " + "-" * 58)
    for e in events:
        print(f"  {e.event_id:<18s} {e.t_start:5.1f}s -> {e.t_end:5.1f}s "
              f"{e.start_frame:5d}-{e.end_frame:<5d} {e.keyframe_idx:9d}")

    ket_luan(
        "Cửa sổ nhích 8 giây một lần (10 trừ 2 chồng lấn).",
        "Chồng lấn để một hành động nằm vắt giữa hai cửa sổ vẫn được bắt trọn.",
        "'keyframe' = khung gần giữa cửa sổ nhất, chính là số ta nộp lên BTC.")


# --------------------------------------------------------------------------- #
def tn4_ba_kenh_va_tron():
    tieu_de(4, "Ba kênh tìm kiếm và cách trộn kết quả (RRF)")
    print("""
  Mỗi event mang 3 loại bằng chứng: hình ảnh, chữ trên màn hình (OCR),
  lời nói (ASR). Mỗi kênh tự xếp hạng riêng, rồi trộn ba bảng lại.
""")
    events, queries = load_benchmark(DATA)
    pipe = Pipeline(HashEmbedder(), RuleBasedProvider())
    pipe.index(events)
    cands = pipe.retriever.retrieve("container Maersk kiện hàng", top_k=5)

    print(f"  {'event':<16s} {'điểm trộn':>10s} | {'visual':>8s} {'ocr':>8s} {'asr':>8s}")
    print("  " + "-" * 56)
    for c in cands:
        m = c.modality_scores
        print(f"  {c.event.event_id:<16s} {c.score:10.4f} | "
              f"{m['visual']:8.3f} {m['ocr']:8.3f} {m['asr']:8.3f}")

    ket_luan(
        "RRF cộng 1/(60+hạng) của từng kênh — chỉ dùng THỨ HẠNG, không dùng điểm thô.",
        "Nhờ vậy kênh cho điểm 0-1 và kênh cho điểm 0-1000 vẫn trộn công bằng.",
        "LƯU Ý: bộ data BTC không có OCR/ASR nên 2 kênh này hiện chưa có nguồn thật.")


# --------------------------------------------------------------------------- #
def tn5_hai_cach_xep_slot():
    tieu_de(5, "Hai cách xếp 100 slot: heuristic vs portfolio")
    print("""
  Cùng một danh sách ứng viên, hai cách xếp thứ tự dòng nộp bài.
  Đây chính là cờ PortfolioConfig.enabled.
""")
    events, queries = load_benchmark(DATA)
    q = [qq for qq, _ in queries if qq.qid == "q1-kis"][0]

    ket_qua = {}
    for ten, cfg in (("TẮT", PortfolioConfig(enabled=False)),
                     ("BẬT", PortfolioConfig(enabled=True,
                                             calibrator=RankCalibrator()))):
        p = Pipeline(HashEmbedder(), RuleBasedProvider(), portfolio=cfg)
        p.index(events)
        ket_qua[ten] = p.answer(q)

    print(f"  {'slot':>4s} | {'TẮT (core/submission)':<26s} | {'BẬT (optimize/)':<26s}")
    print("  " + "-" * 62)
    for i in range(len(ket_qua["TẮT"])):
        a, b = ket_qua["TẮT"][i], ket_qua["BẬT"][i]
        print(f"  {i+1:4d} | {a['video_id']} frame {a['frame_idx']:<12d} | "
              f"{b['video_id']} frame {b['frame_idx']:<12d}")

    ket_luan(
        "TẮT: đan xen các video cho đều — chiến thuật an toàn.",
        "BẬT: dồn slot đắt vào video được cho là chắc ăn nhất.",
        "Chưa biết bên nào đúng. Phải đo trên dữ liệu thật: run_benchmark.py --ab")


# --------------------------------------------------------------------------- #
def tn6_trake_diem_tung_phan():
    tieu_de(6, "TRAKE có điểm từng phần — vì sao không được bỏ task này")
    print("""
  TRAKE hỏi một CHUỖI khoảnh khắc, ví dụ 3 bước. Luật chấm:
     sai video            -> 0 điểm, hết
     đúng video           -> (số bước trúng) / (tổng số bước)
""")
    gt = GroundTruth(task="trake", video_id="L02",
                     event_windows=[[0, 90], [90, 180], [180, 270]])
    cac_truong_hop = [
        ("đúng cả 3 bước",        {"video_id": "L02", "frames": [40, 130, 220]}),
        ("đúng 2/3 bước",         {"video_id": "L02", "frames": [40, 130, 999]}),
        ("đúng 1/3 bước",         {"video_id": "L02", "frames": [40, 999, 999]}),
        ("đúng video, sai hết",   {"video_id": "L02", "frames": [999, 999, 999]}),
        ("SAI VIDEO (dù trúng)",  {"video_id": "L09", "frames": [40, 130, 220]}),
    ]
    for ten, row in cac_truong_hop:
        print(f"  {ten:<24s} -> R-Score = {scoring.r_score_trake(row, gt):.3f}")

    ket_luan(
        "Đúng video mà chỉ trúng 1 bước vẫn được 0.333 — không phải ăn cả ngã không.",
        "Sai video thì trúng bao nhiêu bước cũng bằng 0 => tìm đúng VIDEO là ưu tiên 1.",
        "Đúng video rồi thì mọi slot còn lại vẫn có giá trị => nộp nhiều cách ghép",
        "chuỗi khác nhau của cùng video đó. Đó là việc của align_k_best.")


# --------------------------------------------------------------------------- #
THI_NGHIEM = [tn1_vi_tri_quyet_dinh_diem, tn2_trung_lap_khong_them_diem,
              tn3_event_la_gi, tn4_ba_kenh_va_tron,
              tn5_hai_cach_xep_slot, tn6_trake_diem_tung_phan]


def main():
    chon = sys.argv[1:]
    if chon:
        for c in chon:
            THI_NGHIEM[int(c) - 1]()
    else:
        for tn in THI_NGHIEM:
            tn()
        print("\n" + "=" * 72)
        print("  Xong 6 thí nghiệm. Đọc HUONG_DAN.md để hiểu bức tranh tổng thể.")
        print("=" * 72)


if __name__ == "__main__":
    main()
