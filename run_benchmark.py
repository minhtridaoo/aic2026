#!/usr/bin/env python
"""Run the offline benchmark and print the official Final Score report.

Usage:
    python run_benchmark.py [path/to/benchmark.json]         # core/ only
    python run_benchmark.py --ab [path/to/benchmark.json]    # core/ vs core+optimize/

`--ab` chạy đúng một retriever qua hai đường xếp dòng nộp bài -- heuristic gốc
trong `core.submission` và portfolio trong `optimize.submission_optimizer` --
rồi in hai điểm cạnh nhau. Cùng dữ liệu, cùng ứng viên, khác mỗi cách tiêu 100
slot; nên chênh lệch đọc được chính là phần tầng `optimize` kiếm thêm (hoặc làm
mất). Đó là cách duy nhất biết nó có đáng giữ hay không.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aic2026.evaluation import evaluate, load_benchmark, _print_report
from aic2026.core.providers import HashEmbedder, RuleBasedProvider
from aic2026.core.pipeline import Pipeline, PortfolioConfig
from aic2026.optimize.calibration import RankCalibrator


def _run_ab(events, queries) -> None:
    configs = [
        ("core/     (heuristic)", PortfolioConfig(enabled=False)),
        ("+optimize (portfolio)", PortfolioConfig(enabled=True,
                                                  calibrator=RankCalibrator())),
    ]
    reports = []
    for label, cfg in configs:
        pipe = Pipeline(HashEmbedder(), RuleBasedProvider(), portfolio=cfg)
        reports.append((label, evaluate(pipe, events, queries)))

    tasks = list(reports[0][1]["summary"].keys())
    print("\n=== A/B: Final Score (Mean of Top-k R-Scores) ===")
    print(f"  {'':22s} " + " ".join(f"{t:>8s}" for t in tasks))
    for label, rep in reports:
        print(f"  {label:22s} "
              + " ".join(f"{rep['summary'][t]:8.4f}" for t in tasks))
    base, opt = reports[0][1]["summary"], reports[1][1]["summary"]
    print(f"  {'delta':22s} "
          + " ".join(f"{opt[t] - base[t]:+8.4f}" for t in tasks))


def main() -> None:
    argv = [a for a in sys.argv[1:] if a != "--ab"]
    ab = "--ab" in sys.argv
    path = argv[0] if argv else os.path.join(
        os.path.dirname(__file__), "data", "benchmark_queries.sample.json")
    events, queries = load_benchmark(path)

    if ab:
        _run_ab(events, queries)
    else:
        pipe = Pipeline(HashEmbedder(), RuleBasedProvider())
        _print_report(evaluate(pipe, events, queries))
    print("\n(Reminder: synthetic data — these numbers mean nothing about real "
          "performance. Metric = official Mean of Top-k R-Scores.)")


if __name__ == "__main__":
    main()
