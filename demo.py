#!/usr/bin/env python
"""End-to-end demo on synthetic data (no GPU, no network).

Shows: ingestion (frames -> event windows) -> retrieval -> submission rows ->
CSV validation -> submission.zip. This is a skeleton walkthrough, NOT a
competition result.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026.core.ingestion import sliding_windows
from aic2026.core.models import Frame, Query, SubEvent
from aic2026.core.pipeline import Pipeline
from aic2026.core.providers import HashEmbedder, RuleBasedProvider
from aic2026.core.offline_eval import load_benchmark
from aic2026.core import submission as sub


def demo_ingestion() -> None:
    print("=== Ingestion: 10s window / 2s overlap ===")
    frames = [Frame("L09_V001", i, float(i)) for i in range(0, 30, 2)]
    events = sliding_windows(frames, window=10.0, overlap=2.0)
    for e in events:
        print(f"  {e.event_id}  t[{e.t_start:.0f},{e.t_end:.0f})  kf={e.keyframe_idx}")


def demo_query_and_submission() -> None:
    print("\n=== Retrieval + submission ===")
    data_path = os.path.join(os.path.dirname(__file__), "data",
                             "benchmark_queries.sample.json")
    events, _ = load_benchmark(data_path)
    pipe = Pipeline(HashEmbedder(), RuleBasedProvider())
    pipe.index(events)

    kis = Query(qid="q1", task="kis",
                text="người đàn ông mở cửa container Maersk nhiều kiện hàng")
    kis_rows = pipe.answer(kis)
    print(f"  KIS top-3: {[(r['video_id'], r['frame_idx']) for r in kis_rows[:3]]}")

    trake = Query(qid="q4", task="trake", sub_events=[
        SubEvent("người đàn ông mở cửa phòng bước vào"),
        SubEvent("dòng chữ thông báo khẩn xuất hiện trên bảng"),
        SubEvent("người phát biểu tuyên bố khai mạc sự kiện"),
    ])
    trake_rows = pipe.answer(trake)
    print(f"  TRAKE top row (N frames guaranteed): "
          f"{trake_rows[0]['video_id']} {trake_rows[0]['frames']}")

    out_dir = os.path.join(os.path.dirname(__file__), "out")
    zip_path = os.path.join(out_dir, "team_demo_round1.zip")
    ok, errs = sub.build_submission_zip(
        zip_path,
        files={
            "query-1-kis.csv": ("kis", kis_rows),
            "query-4-trake.csv": ("trake", trake_rows),
        },
        n_events={"query-4-trake.csv": trake.n_events},
    )
    print(f"  submission zip ok={ok} errors={errs} -> "
          f"{os.path.relpath(zip_path, os.path.dirname(__file__))}")


if __name__ == "__main__":
    demo_ingestion()
    demo_query_and_submission()
