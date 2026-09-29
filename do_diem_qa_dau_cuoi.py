#!/usr/bin/env python
"""Đo QA ĐẦU-CUỐI: pipeline tự gọi VLM trên đúng khung NÓ tìm được.

    # Đo GEMINI thật: không cho rơi âm thầm sang Cloudflare
    python -X utf8 do_diem_qa_dau_cuoi.py --config config_kaggle.yaml --gemini-only --so 1
    python -X utf8 do_diem_qa_dau_cuoi.py --config config_kaggle.yaml --gemini-only

Không có ``--gemini-only`` thì script giữ đúng chuỗi production
Gemini -> Cloudflare để diễn tập khả năng sống sót, không phải để đo riêng Gemini.

VÌ SAO CẦN SCRIPT NÀY (2026-08-17)
----------------------------------
`do_diem_vlm_qa.py` và phép đo Kaggle 17/08 đều gửi cho VLM **ảnh ground-truth**
(dựng event_id từ `_ghi_chu.keyframe_n`), rồi mới áp đáp án đó lên 100 dòng
retrieval. Con số ra là **0.2267 ± 0.0627**. Đó là phép đo tốt cho TẦNG ĐÁP ÁN,
nhưng nó KHÔNG phải điểm của đường thi, vì đường thi không có ground truth.

`Pipeline._answer_qa` gửi **top-10 khung đã truy hồi**. Với 18/30 câu dev, khung
đúng không nằm trong top-100 — nói gì top-10. Chỗ lệch lớn nhất là 8 câu có
`score = 0.20`: điểm 0.20 nghĩa là chỉ `R@100` đúng, tức dòng đúng ở hạng 51–100,
**chắc chắn ngoài top-10 gửi VLM**. Đáp án cho những dòng đó, theo
`qa.tail_answer: vlm`, là đáp án phổ biến nhất suy từ 10 ảnh SAI.

Nếu cả 8 câu đó mất, QA rơi từ 0.2267 xuống ~0.1733. Script này trả lời câu hỏi
"thật ra là bao nhiêu" bằng cách KHÔNG can thiệp gì: dựng pipeline đúng config
(`llm` mặc định = provider VLM thật) rồi gọi `pipeline.answer()`.

Chi phí bằng đúng phép đo cũ: pipeline gửi 1 batch/câu (`batch_max_images: 10`
với `vlm_top_n: 10`), nên 30 câu QA = ~30 request.

ĐỌC SỐ RA THẾ NÀO
-----------------
  * `QA đầu-cuối` < `0.2267` là BÌNH THƯỜNG, không phải hồi quy: hai phép đo trả
    lời hai câu hỏi khác nhau. Đây mới là con số mang đi thi.
  * `trần định vị` giống nhau ở cả hai phép đo (nó chỉ phụ thuộc retrieval).
    Khoảng cách trần − thật là phần TẦNG ĐÁP ÁN làm mất.
  * `provider` PHẢI là Gemini. Ra Cloudflare là đang bị hạ cấp âm thầm — số đo
    lúc đó thuộc về Cloudflare (0.1200), không phải Gemini.
"""
import argparse
import copy
import csv
import math
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026 import config as CFG
from aic2026.core import scoring
from aic2026.core.offline_eval import load_queries

GOC = os.path.dirname(os.path.abspath(__file__))
DEV_JSON = os.path.join(GOC, "data", "dev_set_tu_lam.json")


def trung_binh_va_se(gia_tri):
    n = len(gia_tri)
    tb = sum(gia_tri) / n
    if n < 2:
        return tb, float("inf")
    var = sum((x - tb) ** 2 for x in gia_tri) / (n - 1)
    return tb, math.sqrt(var / n)


def tran_dinh_vi(rows, gt):
    """Điểm nếu MỌI đáp án đều đúng — chỉ còn phụ thuộc định vị."""
    diem = [scoring.r_score_kis(row, gt) for row in rows]
    return sum(max(diem[:k], default=0.0)
               for k in scoring.DEFAULT_KS) / len(scoring.DEFAULT_KS)


def _args():
    p = argparse.ArgumentParser(
        description="Đo QA đầu-cuối, pipeline tự gọi VLM")
    p.add_argument("--config", default=None)
    p.add_argument("--so", type=int, default=None,
                   help="chỉ đo N câu đầu (tiết kiệm quota khi thử)")
    p.add_argument("--ra", default=os.path.join(GOC, "out",
                                                "qa_dau_cuoi.csv"))
    p.add_argument(
        "--gemini-only", action="store_true",
        help=("chỉ dùng Gemini, tắt Cloudflare và dừng ngay nếu Gemini lỗi; "
              "dùng khi đo điểm Gemini thật"))
    p.add_argument("--gemini-timeout", type=float, default=120.0,
                   help="timeout mỗi request khi --gemini-only (mặc định 120s)")
    p.add_argument("--gemini-retries", type=int, default=2,
                   help="số retry lỗi 429/timeout/5xx khi --gemini-only")
    p.add_argument(
        "--loai-cau-vlm", default=None,
        help=("ghi đè `qa.vlm_question_types`, ngăn bằng dấu phẩy. Mặc định "
              "config là `color,open` nên câu ĐẾM rơi xuống CLIP zero-shot. "
              "Thử `color,count,open` để Gemini trả lời cả câu đếm — đo 19/08 "
              "cho thấy qa-021 (trần 1.00) mất trắng vì đường zero-shot."))
    return p.parse_args()


def cau_hinh_gemini_only(config, timeout_seconds=120.0, max_retries=2):
    """Tạo config benchmark Gemini không có đường hạ cấp âm thầm.

    Config production giữ nguyên chuỗi Gemini -> Cloudflare. Chỉ phép đo có cờ
    ``--gemini-only`` mới lấy mục primary ra làm provider trực tiếp và tắt luôn
    zero-shot fallback, để mọi lỗi quota/model/timeout hiện nguyên văn.
    """
    if timeout_seconds <= 0:
        raise ValueError("gemini timeout phải > 0")
    if max_retries < 0:
        raise ValueError("gemini retries phải >= 0")
    result = copy.deepcopy(config)
    llm = result.get("llm") or {}
    if str(llm.get("provider", "")).lower() == "failover_vlm":
        gemini = copy.deepcopy(llm.get("primary") or {})
    else:
        gemini = copy.deepcopy(llm)
    if str(gemini.get("provider", "")).lower() != "gemini_vlm":
        raise ValueError("Không tìm thấy Gemini primary trong `llm` của config")
    gemini["enabled"] = True
    gemini["timeout_seconds"] = float(timeout_seconds)
    gemini["max_retries"] = int(max_retries)
    result["llm"] = gemini
    result.setdefault("qa", {})["vlm_fallback_to_zero_shot"] = False
    return result


def main():
    args = _args()
    config = CFG.load_config(args.config)
    if args.gemini_only:
        config = cau_hinh_gemini_only(
            config, timeout_seconds=args.gemini_timeout,
            max_retries=args.gemini_retries)

    if args.loai_cau_vlm is not None:
        loai = [x.strip().lower() for x in args.loai_cau_vlm.split(",")
                if x.strip()]
        hop_le = {"color", "count", "open"}
        la = sorted(set(loai) - hop_le)
        if la:
            print(f"❌ Loại câu không biết: {la} (có: {sorted(hop_le)})")
            sys.exit(1)
        config.setdefault("qa", {})["vlm_question_types"] = loai

    print("=" * 74)
    print("  QA ĐẦU-CUỐI — pipeline tự gọi VLM trên khung nó tìm được")
    print("=" * 74)
    if args.loai_cau_vlm is not None:
        print(f"  ⚙ GHI ĐÈ loại câu gọi VLM: {loai}  (config: "
              f"{sorted((CFG.load_config(args.config).get('qa') or {}).get('vlm_question_types', []))})")
    if args.gemini_only:
        print(f"  CHẾ ĐỘ GEMINI-ONLY · timeout {args.gemini_timeout:g}s · "
              f"retry {args.gemini_retries} · KHÔNG failover")
    for dong in CFG.mo_ta_cau_hinh(config):
        print("  " + dong)
    for dong in CFG.canh_bao_cau_hinh(config):
        print(f"\n  ⚠ {dong}")

    thieu = CFG.thieu_khoa_api(config)
    if thieu:
        print("\n❌ Provider VLM đang bật nhưng thiếu khoá môi trường:")
        for d in thieu:
            print(f"   - {d}")
        sys.exit(1)

    t = time.perf_counter()
    print("\nĐang nạp dữ liệu ...")
    events, emb, metas, rep = CFG.nap_du_lieu(config)
    phu = CFG.nap_hop_nhat(config, events)
    # KHÔNG truyền `llm=`: để `dung_pipeline` lấy đúng provider VLM của config.
    # Truyền RuleBasedProvider vào đây là quay lại đúng phép đo cũ.
    p = CFG.dung_pipeline(config, events, emb, retrieval_phu=phu)
    print(f"  {rep.events:,d} event / {rep.videos_da_nap} video · "
          f"hợp nhất {len(phu)} encoder phụ · {time.perf_counter() - t:.1f}s")
    print(f"  QA: {type(p.llm).__name__} · vlm_top_n {p.qa_vlm_top_n} · "
          f"tail_answer {p.qa_tail_answer} · loại câu {sorted(p.qa_vlm_question_types)}")
    if p.image_resolver is None:
        print("\n❌ Không có image_resolver — `qa.image_roots` rỗng. "
              "Không thể đo đường VLM.")
        sys.exit(1)

    cap = [(q, gt) for q, gt in load_queries(DEV_JSON) if q.task == "qa"]
    if args.so:
        cap = cap[:args.so]
    print(f"\nĐo {len(cap)} câu QA ...\n")

    diem, tran, chi_tiet = [], [], []
    provider_thay = set()
    loi_vlm = []
    for q, gt in cap:
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            rows = p.answer(q, top_k=100)
        giay = time.perf_counter() - t0

        fs = scoring.final_score(rows, gt)
        tr = tran_dinh_vi(rows, gt)
        dap_an = rows[0].get("answer", "") if rows else ""
        chap_nhan = [gt.answer, *gt.answer_alt]
        khop = any(scoring.answers_match(dap_an, x) for x in chap_nhan)

        ai = getattr(p.llm, "last_provider", None) or type(p.llm).__name__
        provider_thay.add(str(ai))
        if getattr(p.last_trace, "vlm_error", None):
            loi_vlm.append((q.qid, str(p.last_trace.vlm_error)[:120]))

        diem.append(fs)
        tran.append(tr)
        chi_tiet.append({
            "qid": q.qid, "cau_hoi": q.question_vi or q.text,
            "ground_truth": gt.answer,
            "answer_alt": ";".join(gt.answer_alt),
            "prediction": dap_an, "answer_matches": int(khop),
            "final_score": f"{fs:.6f}", "localization_ceiling": f"{tr:.6f}",
            "provider": ai, "seconds": f"{giay:.3f}",
        })
        print(f"  {q.qid}: score={fs:.2f} · trần={tr:.2f} · "
              f"đáp án={dap_an!r} · {giay:.1f}s · {ai}")

    tb, se = trung_binh_va_se(diem)
    tb_tran, se_tran = trung_binh_va_se(tran)
    tb_gap, se_gap = trung_binh_va_se([a - b for a, b in zip(tran, diem)])
    khop_tong = sum(int(x["answer_matches"]) for x in chi_tiet)

    os.makedirs(os.path.dirname(os.path.abspath(args.ra)), exist_ok=True)
    with open(args.ra, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(chi_tiet[0]))
        w.writeheader()
        w.writerows(chi_tiet)

    print("\n" + "=" * 74)
    print("  KẾT QUẢ — QA ĐẦU-CUỐI (đây là con số mang đi thi)")
    print("=" * 74)
    print(f"  QA đầu-cuối        : {tb:.4f} ± {se:.4f}")
    print(f"  Trần định vị       : {tb_tran:.4f} ± {se_tran:.4f}")
    print(f"  Trần − thật        : {tb_gap:+.4f} ± {se_gap:.4f}  "
          f"(phần tầng ĐÁP ÁN làm mất)")
    print(f"  Đáp án khớp nhãn   : {khop_tong}/{len(cap)}")
    print(f"  Provider đã trả lời: {', '.join(sorted(provider_thay))}")
    for ten in ("cache_hits", "network_calls", "backend_requests"):
        if hasattr(p.llm, ten):
            print(f"  {ten:19s}: {getattr(p.llm, ten)}")
    if loi_vlm:
        print(f"\n  ⚠ {len(loi_vlm)} câu có lỗi VLM (đã hạ xuống đường rẻ):")
        for qid, msg in loi_vlm[:5]:
            print(f"     {qid}: {msg}")
    failures = list(getattr(p.llm, "failures", ()) or ())
    if failures:
        print("\n  Lỗi provider chính (để chẩn đoán failover):")
        for message in failures[:5]:
            print(f"     {message}")
    if not any("Gemini" in x for x in provider_thay):
        print("\n  ⚠ KHÔNG có câu nào do Gemini trả lời — số đo này KHÔNG phải "
              "điểm của Gemini. Kiểm khoá API và `timeout_seconds`.")
    print(f"\n  CSV: {args.ra}")
    print("\n  So với phép đo cũ (ảnh ground-truth, 0.2267): thấp hơn là BÌNH "
          "THƯỜNG.\n  Hai phép đo trả lời hai câu hỏi khác nhau — xem docstring.")


if __name__ == "__main__":
    main()
