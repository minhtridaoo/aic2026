#!/usr/bin/env python
"""VIỆC 4.3 — Chấm QA và TRAKE trên dữ liệu thật, lần đầu tiên.

    python do_diem_qa_trake.py
    python do_diem_qa_trake.py --clip      # đường encoder cũ của BTC
    python do_diem_qa_trake.py --so400m   # encoder chính 1152 chiều
    python do_diem_qa_trake.py --qa-en      # mốc A: giữ đầu ra zero-shot tiếng Anh
    python do_diem_qa_trake.py --opus-en-vi # dịch lớp mở bằng Opus; màu/số dùng bảng tay

`scoring.r_score_qa` và `r_score_trake` chưa từng chạy trên nhãn thật. Script này
chạy chúng, và quan trọng hơn: TÁCH điểm ra thành các phần để biết mất điểm ở đâu.

VÌ SAO PHẢI TÁCH
----------------
QA đòi ĐÚNG CẢ BA: video, khung, và chuỗi trả lời. Một con số 0.0000 không cho
biết hệ thống tìm sai chỗ, hay tìm đúng chỗ mà trả lời sai. Hai bệnh đó chữa bằng
hai thứ hoàn toàn khác nhau (retrieval vs VLM), nên đo gộp là tự bịt mắt.

Vì vậy QA được chấm hai lần:
  điểm thật    — `r_score_qa`, đòi đúng cả đáp án
  TRẦN định vị — cũng vậy nhưng BỎ QUA đáp án; tức "điểm sẽ là bao nhiêu nếu có
                 một VLM trả lời đúng mọi lúc". Khoảng cách giữa hai số này là
                 phần mà một VLM có thể ăn được.

Hệ thống hiện tại dùng `RuleBasedProvider.answer()` — nó trả về token viết hoa
đầu tiên trong metadata, tức KHÔNG phải một mô hình trả lời câu hỏi. Điểm QA thật
gần 0 là điều DỰ KIẾN, không phải lỗi. Trần định vị mới là con số có tin tức.

TRAKE cũng tách: điểm chuỗi, và tỉ lệ trúng theo TỪNG MỐC (mốc 1, 2, 3 riêng) —
vì `r_score_trake` cho điểm từng phần, mốc đầu thường dễ hơn mốc sau.

⚠ `answers_match` là proxy offline (so chuỗi chính xác sau chuẩn hoá). BTC chấm
"semantic exact match" nên rộng hơn. Điểm QA ở đây là GIỚI HẠN DƯỚI.
"""
import importlib.util
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Lấy từ `core.scoring` chứ không qua `aic2026.evaluation`: gói đó chỉ tái xuất
# `final_score`/`recall_at_k`/..., không có `DEFAULT_KS`, `r_score_kis` hay
# `answers_match` — mà script này cần cả ba để tách điểm QA.
from aic2026.core import scoring                     # noqa: E402
from aic2026.core.offline_eval import load_queries   # noqa: E402
from aic2026.core.pipeline import PortfolioConfig    # noqa: E402

GOC = os.path.dirname(os.path.abspath(__file__))
KS = scoring.DEFAULT_KS


def _nap_ddt():
    spec = importlib.util.spec_from_file_location(
        "ddt", os.path.join(GOC, "do_diem_thuc_te.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tb_se(v):
    n = len(v)
    if not n:
        return float("nan"), float("nan")
    tb = sum(v) / n
    if n < 2:
        return tb, float("inf")
    var = sum((x - tb) ** 2 for x in v) / (n - 1)
    return tb, math.sqrt(var / n)


def diem_bo_dap_an(rows, gt):
    """`final_score` nhưng QA KHÔNG đòi đúng đáp án — trần của tầng định vị.

    Dùng lại đúng `r_score_kis` (video + khung) rồi lấy max trên tiền tố y như
    `top_k_r_scores`, để con số so được trực tiếp với điểm thật.
    """
    per = [scoring.r_score_kis(r, gt) for r in rows]
    return sum((max(per[:k]) if per[:k] else 0.0) for k in KS) / len(KS)


def main():
    ddt = _nap_ddt()
    siglip2 = ("" if "--clip" in sys.argv
               else "so400m" if "--so400m" in sys.argv else "base")
    ddt.kiem_tra_dieu_kien(can_dev_set=True, siglip2=siglip2)
    events, emb, _, enc = ddt.nap_he_thong(siglip2=siglip2)

    queries = load_queries(ddt.DEV_JSON)
    theo = {}
    for q, gt in queries:
        theo.setdefault(q.task, []).append((q, gt))
    print("\nDev set: " + " · ".join(f"{t}: {len(v)}" for t, v in sorted(theo.items())))

    if "qa" not in theo and "trake" not in theo:
        raise SystemExit(
            "Dev set chưa có câu QA/TRAKE nào.\n"
            "  -> gán nhãn bằng tao_trang_gan_nhan_qa.py / _trake.py,\n"
            "     chép CSV vào dev_set\\, rồi chạy tao_dev_set.py")

    qa_language = "en" if "--qa-en" in sys.argv else "vi"
    answer_translator = None
    if "--opus-en-vi" in sys.argv:
        from aic2026.core.dich import OpusMTEnViTranslator
        answer_translator = OpusMTEnViTranslator()
    p = ddt.dung_pipeline(
        events, emb, enc, portfolio=PortfolioConfig(enabled=False),
        qa_output_language=qa_language,
        qa_answer_translator=answer_translator)

    # ----------------------------------------------------------------------- #
    # QA
    # ----------------------------------------------------------------------- #
    if "qa" in theo:
        that, strict_scores, tran, dung_khung, dung_dap = [], [], [], 0, 0
        vi_du = []
        predictions = []
        latencies = []
        for q, gt in theo["qa"]:
            started = time.perf_counter()
            rows = p.answer(q, top_k=100)
            latencies.append(time.perf_counter() - started)
            that.append(scoring.final_score(rows, gt, KS))
            tran.append(diem_bo_dap_an(rows, gt))
            # Có dòng nào chạm đúng video+khung? Có dòng nào trả lời đúng?
            khung = any(scoring.r_score_kis(r, gt) > 0 for r in rows)
            accepted = [gt.answer] + list(gt.answer_alt)
            dap = any(scoring.answers_match(r.get("answer", ""), answer)
                      for r in rows for answer in accepted)
            dung_khung += bool(khung)
            dung_dap += bool(dap)
            predicted = rows[0].get("answer", "") if rows else ""
            strict_match = scoring.answers_match(predicted, gt.answer)
            # ClipZeroShotQA dùng một đáp án cho toàn bộ rows, nên nếu chuỗi
            # khớp nhãn chính thì điểm strict bằng trần định vị, ngược lại 0.
            strict_scores.append(tran[-1] if strict_match else 0.0)
            predictions.append({
                "qid": q.qid,
                "question": q.question_vi or q.text,
                "ground_truth_vi": gt.answer,
                "answer_alt": ";".join(gt.answer_alt),
                "prediction": predicted,
                "strict_vi_match": int(strict_match),
                "answer_matches": int(any(scoring.answers_match(predicted, a)
                                          for a in accepted)),
                "hits_gt_frame": int(khung),
                "final_score": f"{that[-1]:.6f}",
                "localization_ceiling": f"{tran[-1]:.6f}",
                "latency_seconds": f"{latencies[-1]:.6f}",
            })
            if len(vi_du) < 5:
                vi_du.append((q.qid, gt.answer, predicted, khung))

        out_dir = os.path.join(GOC, "out")
        os.makedirs(out_dir, exist_ok=True)
        prediction_path = os.path.join(
            out_dir, f"qa_predictions_{qa_language}.csv")
        with open(prediction_path, "w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)

        tb_that, se_that = tb_se(that)
        tb_strict, se_strict = tb_se(strict_scores)
        tb_tran, se_tran = tb_se(tran)
        n = len(that)
        print("\n" + "=" * 78)
        print(f"  QA — {n} câu")
        print("=" * 78)
        print(f"  điểm THẬT (đòi đúng đáp án)   {tb_that:.4f} ± {se_that:.4f}")
        print(f"  điểm STRICT nhãn VI chính     {tb_strict:.4f} ± {se_strict:.4f}")
        print(f"  TRẦN định vị (bỏ qua đáp án)  {tb_tran:.4f} ± {se_tran:.4f}")
        print(f"  -> khoảng cách {tb_tran - tb_that:+.4f} = phần một VLM có thể ăn")
        print(f"\n  có dòng chạm đúng video+khung : {dung_khung}/{n} ({dung_khung/n:.1%})")
        print(f"  có dòng trả lời đúng đáp án   : {dung_dap}/{n} ({dung_dap/n:.1%})")
        print(f"  độ trễ QA end-to-end/câu      : {sum(latencies)/n:.3f}s trung bình"
              f" · {max(latencies):.3f}s lớn nhất")
        print(f"  đủ 30 dự đoán đã lưu          : {prediction_path}")
        print("\n  5 ví dụ (đáp án đúng | hệ thống trả lời | có chạm đúng khung):")
        for qid, dung, doan, kh in vi_du:
            print(f"    {qid}  {dung!r:22s} | {doan!r:22s} | {'có' if kh else 'không'}")
        # --- đối chứng: cùng keyframe, câu HỎI vs câu MÔ TẢ ----------------- #
        # QA đặt trên đúng những keyframe đã có nhãn KIS. Nếu định vị bằng câu hỏi
        # tệ hơn rõ so với định vị bằng câu mô tả CÙNG ảnh đó, thì nút cổ chai của
        # QA là DẠNG CÂU (nghi vấn), không phải tầng tìm — và cách chữa là viết lại
        # câu hỏi thành câu tả trước khi truy hồi, chứ không phải encoder mạnh hơn.
        theo_khung = {}
        for q, gt in theo.get("kis", []):
            theo_khung[(gt.video_id, gt.frame_start)] = q
        cap = [(qa_q, kis_q, gt) for qa_q, gt in theo["qa"]
               if (kis_q := theo_khung.get((gt.video_id, gt.frame_start)))]
        if cap:
            d_hoi, d_ta = [], []
            for qa_q, kis_q, gt in cap:
                d_hoi.append(diem_bo_dap_an(p.answer(qa_q, top_k=100), gt))
                d_ta.append(diem_bo_dap_an(p.answer(kis_q, top_k=100), gt))
            hieu = [b - a for a, b in zip(d_hoi, d_ta)]
            tb_h, _ = tb_se(d_hoi)
            tb_t, _ = tb_se(d_ta)
            d, se = tb_se(hieu)
            print(f"\n  --- đối chứng trên {len(cap)} keyframe có CẢ hai loại nhãn ---")
            print(f"  định vị bằng câu HỎI (QA)     {tb_h:.4f}")
            print(f"  định vị bằng câu MÔ TẢ (KIS)  {tb_t:.4f}")
            print(f"  chênh lệch ghép cặp           {d:+.4f} ± {se:.4f}"
                  f"   ({'vượt' if abs(d) > 2 * se else 'chưa vượt'} 2·SE)")
            if d > 2 * se:
                print("  -> Nút cổ chai của QA là DẠNG CÂU NGHI VẤN, không phải tầng tìm.")
                print("     Chữa bằng viết lại câu hỏi thành câu tả trước khi truy hồi")
                print("     (`control/query_compiler.py`), không phải bằng encoder mạnh hơn.")

        # --- ngôn ngữ đáp án ------------------------------------------------- #
        # Tách "trả lời SAI" khỏi "trả lời ĐÚNG NGHĨA nhưng khác ngôn ngữ". Không
        # tách thì hai bệnh khác nhau hoàn toàn cùng hiện ra là 0.0, và người đọc
        # sẽ đi sửa sai chỗ.
        def co_dau(s):
            return any(ord(c) > 127 for c in s)

        vi_gt = sum(1 for _, gt in theo["qa"] if co_dau(gt.answer))
        so_gt = sum(1 for _, gt in theo["qa"] if gt.answer.strip().isdigit())
        co_alt = sum(1 for _, gt in theo["qa"] if gt.answer_alt)
        print(f"\n  --- ngôn ngữ đáp án chuẩn ---")
        print(f"  tiếng Việt (có dấu) : {vi_gt}/{n}")
        print(f"  chữ số              : {so_gt}/{n}")
        print(f"  có khai biến thể    : {co_alt}/{n}  (cột `dap_an_alt`)")
        if vi_gt and co_alt < vi_gt:
            print("\n  ⚠ Đáp án chuẩn là TIẾNG VIỆT, và hệ thống đang trả lời bằng")
            print("    tiếng Anh ('xanh lá' vs 'green' — nghĩa trùng, chuỗi khác).")
            print("    `answers_match` so chuỗi nên vẫn cho 0. Hai cách xử lý:")
            print("      1. Bắt tầng trả lời xuất TIẾNG VIỆT (đúng với đề thi VN), hoặc")
            print("      2. Điền cột `dap_an_alt` trong dan_nhan_qa.csv (ngăn bằng `;`)")
            print("         để bản dịch cũng được tính — `GroundTruth.answer_alt`.")
            print("    Chưa làm một trong hai thì điểm QA còn bị chặn cứng ở 0.")

        if dung_dap == 0:
            print("\n  ⚠ KHÔNG câu nào trả lời đúng. Đây là điều DỰ KIẾN:")
            print("    `RuleBasedProvider.answer()` trả token viết hoa đầu tiên trong")
            print("    metadata — nó không phải mô hình trả lời câu hỏi. Muốn QA có")
            print("    điểm thì phải cắm một VLM vào `LLMProvider.answer`.")
            print("    Con số đáng theo dõi bây giờ là TRẦN ĐỊNH VỊ ở trên.")

    # ----------------------------------------------------------------------- #
    # TRAKE
    # ----------------------------------------------------------------------- #
    if "trake" in theo:
        diem, trung_moc, du_moc, dung_video = [], None, 0, 0
        for q, gt in theo["trake"]:
            rows = p.answer(q, top_k=100)
            diem.append(scoring.final_score(rows, gt, KS))
            n_moc = len(gt.event_windows)
            if trung_moc is None:
                trung_moc = [0] * n_moc
            # Dòng tốt nhất trong 100 dòng: mốc nào của nó trúng?
            tot, tot_hit = -1.0, [False] * n_moc
            for r in rows:
                s = scoring.r_score_trake(r, gt)
                if s > tot:
                    fr = r.get("frames", [])
                    tot, tot_hit = s, [
                        j < len(fr) and w[0] <= int(fr[j]) <= w[1]
                        for j, w in enumerate(gt.event_windows)]
            for j, h in enumerate(tot_hit):
                trung_moc[j] += bool(h)
            du_moc += any(len(r.get("frames", [])) == n_moc for r in rows)
            dung_video += any(r["video_id"] == gt.video_id for r in rows)

        tb, se = tb_se(diem)
        n = len(diem)
        print("\n" + "=" * 78)
        print(f"  TRAKE — {n} chuỗi, {len(trung_moc)} mốc mỗi chuỗi")
        print("=" * 78)
        print(f"  điểm {tb:.4f} ± {se:.4f}")
        print(f"  có dòng đúng video          : {dung_video}/{n} ({dung_video/n:.1%})")
        print(f"  có dòng nộp đủ {len(trung_moc)} mốc       : {du_moc}/{n} ({du_moc/n:.1%})")
        print("\n  tỉ lệ trúng theo từng mốc (dòng tốt nhất của mỗi câu):")
        for j, c in enumerate(trung_moc, start=1):
            print(f"    mốc {j}: {c}/{n} ({c/n:.1%})")
        if trung_moc and trung_moc[0] and trung_moc[-1] / max(trung_moc[0], 1) < 0.5:
            print("\n  -> Mốc sau tụt rõ so với mốc đầu. Đó là dấu hiệu DP căn chuỗi")
            print("     đang neo vào mốc đầu rồi trôi, chứ không phải encoder yếu.")
        print("\n  Nhắc: r_score_trake cho ĐIỂM TỪNG PHẦN — trúng 2/3 mốc được 0.667,")
        print("  nên điểm chuỗi không phải all-or-nothing.")

    print("\n" + "=" * 78)
    print("  Điểm QA ở đây là GIỚI HẠN DƯỚI: `answers_match` so chuỗi chính xác,")
    print("  còn BTC chấm 'semantic exact match' nên rộng hơn.")


if __name__ == "__main__":
    main()
