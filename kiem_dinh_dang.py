#!/usr/bin/env python
"""Kiểm ĐỊNH DẠNG file nộp bài trên dữ liệu thật — KHÔNG phải đường nộp bài.

    python kiem_dinh_dang.py

Trả lời đúng một câu hẹp: mỗi dạng truy vấn có sinh ra đủ 100 dòng ĐÚNG ĐỊNH DẠNG
mà validator chấp nhận không (một file sai định dạng vẫn đốt một lượt nộp). Nó
chạy trên 177.321 vector thật chứ không phải dữ liệu bịa như `demo.py`, nhưng nó
KHÔNG chấm điểm và KHÔNG phản ánh cấu hình thi.

⚠ ĐỌC TRƯỚC KHI DÙNG (lịch sử: đây từng là `nop_bai_thu.py`)
------------------------------------------------------------
Ngày 15/08 khi diễn tập, script này xuất ra zip có đáp án QA **tiếng Anh**
(`Blue`/`Red`/`White`) trong khi cấu hình thi đặt `qa.output_language: vi`. Vì nó
tự dựng `Pipeline` riêng, nó đã trôi khỏi cấu hình thật ở ba chỗ: base CLIP thay
vì SigLIP2 so400m, không dịch đáp án, và portfolio bật. Zip vẫn hợp lệ, validator
vẫn `ok=True` — sai lệch loại này không có triệu chứng.

Nên nó được đổi tên và **giữ nguyên cấu hình rẻ đó một cách CÓ Ý THỨC**: CLIP
512 chiều nạp nhanh, portfolio bật để phép kiểm phủ luôn nhánh `optimize/`. Đây
là bộ kiểm định dạng, và chỉ vậy.

    Nộp bài thật -> python nop_bai.py   (đọc config.yaml / config.example.yaml)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026.core import submission as sub
from aic2026.core.dataset import load_dataset
from aic2026.core.models import Query, SubEvent
from aic2026.core.pipeline import Pipeline, PortfolioConfig
from aic2026.core.providers import CLIPTextEncoder, RuleBasedProvider
from aic2026.core.qa import ClipZeroShotQA, dung_sinh_ung_vien
from aic2026.core.stores import NumpyVectorStore
from aic2026.optimize.calibration import RankCalibrator

GOC = os.path.dirname(os.path.abspath(__file__))
DATA = r"D:\COMP\AIC2026\data cuộc thi\aic2026_data_nhe"
CACHE = os.path.join(GOC, "dev_set", "query_vectors.npz")
RA = os.path.join(GOC, "out", "kiem_dinh_dang.zip")


def main():
    print(__doc__.splitlines()[0])
    print("  (cấu hình RẺ, cố ý khác cấu hình thi — nộp bài thật: nop_bai.py)\n")
    print("Đang nạp 873 video ...")
    events, emb, metas, rep = load_dataset(DATA)
    enc = CLIPTextEncoder(cache_path=CACHE if os.path.exists(CACHE) else None,
                          cho_phep_live=True)
    kho = NumpyVectorStore(khu_modality_gap=True)
    p = Pipeline(enc, RuleBasedProvider(), fusion="rrf",
                 visual_qa=ClipZeroShotQA(enc, dung_sinh_ung_vien()),
                 portfolio=PortfolioConfig(enabled=True,
                                           calibrator=RankCalibrator()))
    p.retriever.visual = kho
    p.index(events, embeddings=emb)
    print(f"  {rep.events:,d} event\n")

    q_kis = Query(qid="kis-1", task="kis",
                  text="a man in a suit speaking at a microphone")
    q_qa = Query(qid="qa-1", task="qa",
                 text="What color is the shirt of the person speaking?")
    q_trake = Query(qid="trake-1", task="trake", text="")
    q_trake.sub_events = [SubEvent(text="a chef pouring oil into a pan"),
                          SubEvent(text="a chef stirring food in a pan"),
                          SubEvent(text="a finished dish on a plate")]

    files, n_events = {}, {}
    for q in (q_kis, q_qa, q_trake):
        rows = p.answer(q, top_k=100)
        ten = f"query-{q.qid}.csv"
        files[ten] = (q.task, rows)
        if q.task == "trake":
            n_events[ten] = q.n_events
        print(f"  {q.task.upper():6s} {len(rows):>3d} dòng | ví dụ: {rows[0]}")
        if q.task == "trake":
            kh = {tuple(r["frames"]) for r in rows}
            print(f"         {len(kh)} bộ khung phân biệt "
                  f"(jitter đang rải {len(p.portfolio.trake_jitters)} mức)")
        if q.task == "qa":
            da = []
            for r in rows:
                if r.get("answer") not in da:
                    da.append(r["answer"])
            print(f"         đáp án phân biệt: {da[:5]}")
            print("         (tiếng Anh là ĐÚNG ở đây — script này không dịch)")

    ok, loi = sub.build_submission_zip(RA, files, n_events=n_events)
    print(f"\n  zip ok={ok} lỗi={loi}")
    print(f"  -> {RA}")
    if ok:
        print("\n  ✅ Cả ba dạng đều sinh được file ĐÚNG ĐỊNH DẠNG trên dữ liệu thật.")
        print("     (Hợp lệ ≠ đúng, và ≠ cấu hình thi. Nộp bài thật: nop_bai.py)")


if __name__ == "__main__":
    main()
