"""OFFICIAL scoring metric (development metric #1).

Implements the AIC preliminary rule (verified against the 2025 regulation; the
2026 submission guide is byte-identical in format, but the exact k-thresholds
live on the Codabench 'Evaluation' tab and MUST be re-confirmed when the 2026
package opens — hence `ks` is parameterised, never hard-coded downstream):

    Final Score(query) = (1/|K|) * sum_{k in K} max_{1<=i<=k} R-Score(r_i)
    K = {1, 5, 20, 50, 100}  (default)

Per-task R-Score:
    Textual KIS : I(video == GT_v  and  frame in [s, e])
    Q&A         : I(video == GT_v  and  frame in [s, e]  and  answer == GT_a)
    TRAKE       : 0 if video != GT_v
                  else (1/N) * sum_j I(frame_j in [s_j, e_j])   (per-frame partial credit)

An "answer" r_i is one submitted row (see models / submission for the row shape).
Rows here are plain tuples/dicts so scoring has no dependency on retrieval.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Sequence, Tuple

from .models import GroundTruth

DEFAULT_KS: Tuple[int, ...] = (1, 5, 20, 50, 100)


# --------------------------------------------------------------------------- #
# Answer comparison (QA).  Offline proxy for the judge's "semantic exact match".
# Real judge is stricter/semantic; keep the proxy explicit and swappable.
# --------------------------------------------------------------------------- #
def normalize_answer(a: str) -> str:
    a = unicodedata.normalize("NFC", str(a)).strip().lower()
    a = a.strip('"').strip()
    a = re.sub(r"\s+", " ", a)
    return a


def answers_match(pred: str, gt: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gt)


# --------------------------------------------------------------------------- #
# Per-answer R-Score
# --------------------------------------------------------------------------- #
def r_score_kis(row: Dict, gt: GroundTruth) -> float:
    if row["video_id"] != gt.video_id:
        return 0.0
    return 1.0 if gt.frame_start <= int(row["frame_idx"]) <= gt.frame_end else 0.0


def r_score_qa(row: Dict, gt: GroundTruth) -> float:
    """Đúng cả video, khung, VÀ chuỗi trả lời.

    Đáp án khớp `gt.answer` hoặc BẤT KỲ biến thể trong `gt.answer_alt` — xem
    docstring `GroundTruth.answer_alt`. Danh sách biến thể rỗng thì hành vi y như
    trước, nên đây là nới lỏng CÓ KIỂM SOÁT: chỉ những cách diễn đạt do người gán
    nhãn khai ra mới được tính, không phải khớp mờ tuỳ tiện.
    """
    if row["video_id"] != gt.video_id:
        return 0.0
    if not (gt.frame_start <= int(row["frame_idx"]) <= gt.frame_end):
        return 0.0
    pred = row.get("answer", "")
    if answers_match(pred, gt.answer):
        return 1.0
    return 1.0 if any(answers_match(pred, a) for a in gt.answer_alt) else 0.0


def r_score_trake(row: Dict, gt: GroundTruth) -> float:
    """Full-chain partial credit — the prior offline_eval scored only event #1."""
    if row["video_id"] != gt.video_id:
        return 0.0
    frames: Sequence[int] = row["frames"]
    windows = gt.event_windows
    n = len(windows)
    if n == 0:
        return 0.0
    # A row must present exactly N frames; a short row can only lose credit.
    hits = 0
    for j, (s, e) in enumerate(windows):
        if j < len(frames) and s <= int(frames[j]) <= e:
            hits += 1
    return hits / n


def r_score(row: Dict, gt: GroundTruth) -> float:
    return {
        "kis": r_score_kis,
        "qa": r_score_qa,
        "trake": r_score_trake,
    }[gt.task](row, gt)


# --------------------------------------------------------------------------- #
# Final score = Mean of Top-k R-Scores
# --------------------------------------------------------------------------- #
def top_k_r_scores(rows: List[Dict], gt: GroundTruth,
                   ks: Sequence[int] = DEFAULT_KS) -> Dict[int, float]:
    """R@k = max R-Score over the first k rows, for each k in ks."""
    per_row = [r_score(r, gt) for r in rows]
    out: Dict[int, float] = {}
    for k in ks:
        prefix = per_row[:k]
        out[k] = max(prefix) if prefix else 0.0
    return out


def final_score(rows: List[Dict], gt: GroundTruth,
                ks: Sequence[int] = DEFAULT_KS) -> float:
    rk = top_k_r_scores(rows, gt, ks)
    return sum(rk.values()) / len(ks)


# --------------------------------------------------------------------------- #
# Diagnostics (cheap, for debugging during dev — NOT the ranking metric)
# --------------------------------------------------------------------------- #
def first_correct_rank(rows: List[Dict], gt: GroundTruth,
                       threshold: float = 1.0) -> int:
    """1-indexed rank of the first fully-correct row, or 0 if none.

    For TRAKE, threshold<1.0 lets a partially-correct row count.
    """
    for i, r in enumerate(rows, start=1):
        if r_score(r, gt) >= threshold:
            return i
    return 0


def recall_at_k(rows: List[Dict], gt: GroundTruth, k: int,
                threshold: float = 1.0) -> float:
    rank = first_correct_rank(rows[:k], gt, threshold)
    return 1.0 if rank else 0.0


def mrr(rows: List[Dict], gt: GroundTruth, threshold: float = 1.0) -> float:
    rank = first_correct_rank(rows, gt, threshold)
    return 1.0 / rank if rank else 0.0
