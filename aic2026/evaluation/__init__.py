"""Verified official scoring and shared offline evaluation entry points."""

from aic2026.core.offline_eval import (
    _print_report,
    evaluate,
    load_benchmark,
    load_queries,
)
from aic2026.core.scoring import (
    final_score,
    first_correct_rank,
    recall_at_k,
    r_score,
    top_k_r_scores,
)

__all__ = ["_print_report", "evaluate", "final_score",
           "first_correct_rank", "load_benchmark", "load_queries",
           "recall_at_k", "r_score", "top_k_r_scores"]
