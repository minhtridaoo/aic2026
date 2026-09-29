"""Offline evaluation — the #1 development metric.

Loads a benchmark JSON (events + queries + ground truth), runs the pipeline, and
reports the OFFICIAL Final Score (Mean of Top-k R-Scores) per task and overall,
plus cheap diagnostics (Recall@1/@100, MRR). This is what every tuning decision
should be judged by, because it is exactly the score the organiser computes.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from .models import Event, GroundTruth, Query, SubEvent
from .pipeline import Pipeline
from . import de_bai, scoring


def load_benchmark(path: str) -> Tuple[List[Event], List[Tuple[Query, GroundTruth]]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    events = [Event(**e) for e in data["events"]]
    queries: List[Tuple[Query, GroundTruth]] = []
    for q in data["queries"]:
        task = q["task"]
        if task == "trake":
            query = Query(qid=q["qid"], task=task,
                          sub_events=[SubEvent(**d) for d in
                                      de_bai.tach_sub_events(q["sub_events"],
                                                             q["qid"])])
        else:
            query = Query(qid=q["qid"], task=task, text=q.get("text", ""),
                          question_vi=q.get("_cau_hoi_vi", ""))
        g = q["gt"]
        gt = GroundTruth(
            task=task,
            video_id=g["video_id"],
            frame_start=g.get("frame_start", 0),
            frame_end=g.get("frame_end", 0),
            answer=g.get("answer", ""),
            answer_alt=[str(x) for x in g.get("answer_alt", [])],
            event_windows=g.get("event_windows", []),
        )
        queries.append((query, gt))
    return events, queries


def load_queries(path: str,
                 truong_text: str = "text") -> List[Tuple[Query, GroundTruth]]:
    """Đọc CHỈ phần query + đáp án, không kèm events.

    `load_benchmark` nhét cả events vào một file JSON — ổn với bộ 8 sự kiện bịa,
    nhưng dữ liệu thật có 177.321 event, không thể nhồi vào JSON. Với dev set
    thật, events nạp từ `core.dataset` còn query nạp từ đây.

    `truong_text` chọn cột nào làm câu truy vấn. Mặc định `"text"` (bản tiếng Anh
    người dịch). Truyền `"_mo_ta_vi"` để đo **tiếng Việt thẳng** — cần cho việc
    2.6: nếu SigLIP2 đọc tiếng Việt đủ tốt thì bỏ được cả lớp dịch lẫn ô tiếng
    Anh khi gán nhãn các đợt sau.

    Thiếu trường đó ở BẤT KỲ câu nào là NÉM LỖI, không lặng lẽ tụt về `"text"`.
    Tụt về được thì một phần dev set chạy tiếng Việt và phần còn lại chạy tiếng
    Anh, và con số đo ra không nói về ngôn ngữ nào cả.

    Từ 2026-08-17 quy tắc đó áp cho CẢ TRAKE: chuỗi sự kiện tiếng Việt nằm ở
    `_mo_ta_vi` dạng chuỗi ngăn bằng "→", còn `sub_events` là bản tiếng Anh.
    Nhánh TRAKE trước đây bỏ qua `truong_text` nên mọi số TRAKE "tiếng Việt"
    thật ra là số tiếng Anh — xem `core/de_bai.py`.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    queries: List[Tuple[Query, GroundTruth]] = []
    for q in data["queries"]:
        task = q["task"]
        if task == "trake":
            # `truong_text` phải áp cho CẢ TRAKE. Trước 2026-08-17 nhánh này bỏ
            # qua nó, nên `--vi` đo TRAKE bằng `sub_events` TIẾNG ANH mà không
            # nói gì — hai lần chạy en/vi ra TRAKE giống hệt nhau, và con số đó
            # bị đọc nhầm thành "tiếng Việt không ảnh hưởng TRAKE".
            if truong_text == "text":
                nguon = q["sub_events"]
            else:
                nguon = q.get(truong_text)
                if not (isinstance(nguon, (list, tuple))
                        or (isinstance(nguon, str) and nguon.strip())):
                    raise KeyError(
                        f"{q['qid']}: không có `{truong_text}` (hoặc để rỗng).\n"
                        f"  Với TRAKE, trường này là chuỗi sự kiện ngăn bằng "
                        f"'→' hoặc một danh sách.")
            query = Query(qid=q["qid"], task=task,
                          sub_events=[SubEvent(**d) for d in
                                      de_bai.tach_sub_events(nguon, q["qid"])])
        else:
            if truong_text != "text" and not (q.get(truong_text) or "").strip():
                raise KeyError(
                    f"{q['qid']}: không có `{truong_text}` (hoặc để rỗng).\n"
                    f"  Chạy lại tao_dev_set.py sau khi điền đủ cột tương ứng "
                    f"trong dev_set/dan_nhan.csv.")
            query = Query(qid=q["qid"], task=task,
                          text=q.get(truong_text) or q.get("text", ""),
                          question_vi=q.get("_cau_hoi_vi", ""))
        g = q["gt"]
        queries.append((query, GroundTruth(
            task=task,
            video_id=g["video_id"],
            frame_start=g.get("frame_start", 0),
            frame_end=g.get("frame_end", 0),
            answer=g.get("answer", ""),
            answer_alt=[str(x) for x in g.get("answer_alt", [])],
            event_windows=g.get("event_windows", []),
        )))
    return queries


def evaluate(pipeline: Pipeline, events: List[Event],
             queries: Sequence[Tuple[Query, GroundTruth]],
             ks=scoring.DEFAULT_KS) -> Dict:
    pipeline.index(events)
    per_task: Dict[str, List[float]] = defaultdict(list)
    detail = []
    for query, gt in queries:
        rows = pipeline.answer(query)
        fs = scoring.final_score(rows, gt, ks)
        per_task[query.task].append(fs)
        detail.append({
            "qid": query.qid,
            "task": query.task,
            "final_score": round(fs, 4),
            "recall@1": scoring.recall_at_k(rows, gt, 1, threshold=1.0),
            "recall@100": scoring.recall_at_k(rows, gt, 100, threshold=1.0),
            "mrr": round(scoring.mrr(rows, gt, threshold=1.0), 4),
        })
    summary = {t: round(sum(v) / len(v), 4) for t, v in per_task.items()}
    all_scores = [d["final_score"] for d in detail]
    summary["OVERALL"] = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0
    return {"summary": summary, "detail": detail}


def _print_report(report: Dict) -> None:
    print("\n=== Per-task Final Score (Mean of Top-k R-Scores) ===")
    for task, score in report["summary"].items():
        print(f"  {task:10s} {score:.4f}")
    print("\n=== Per-query detail ===")
    print(f"  {'qid':6s} {'task':6s} {'final':>7s} {'R@1':>5s} {'R@100':>6s} {'MRR':>6s}")
    for d in report["detail"]:
        print(f"  {d['qid']:6s} {d['task']:6s} {d['final_score']:7.4f} "
              f"{d['recall@1']:5.0f} {d['recall@100']:6.0f} {d['mrr']:6.3f}")


if __name__ == "__main__":  # pragma: no cover
    import sys
    from .providers import HashEmbedder, RuleBasedProvider
    path = sys.argv[1] if len(sys.argv) > 1 else "data/benchmark_queries.sample.json"
    events, queries = load_benchmark(path)
    pipe = Pipeline(HashEmbedder(), RuleBasedProvider())
    _print_report(evaluate(pipe, events, queries))
