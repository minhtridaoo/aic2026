"""Rerank a deep retrieval pool using local temporal agreement.

This module is deliberately independent from the production path.  The signal
has not yet earned the right to be enabled by default: first benchmark it on a
video-level holdout with ``thu_nghiem_rerank_thoi_gian.py``.

Only ranks and timestamps are used.  Raw cosine/RRF scores are intentionally
ignored because scores from different encoders are not on a shared scale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from aic2026.core.models import Candidate


@dataclass(frozen=True)
class TemporalRerankConfig:
    """Parameters for rank-based temporal-neighbour reranking."""

    window_seconds: float = 5.0
    alpha: float = 1.0
    rank_k: float = 60.0
    max_neighbours: int = 4

    def __post_init__(self) -> None:
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")
        if self.rank_k < 0:
            raise ValueError("rank_k must be >= 0")
        if self.max_neighbours < 1:
            raise ValueError("max_neighbours must be >= 1")


def temporal_scores(
    candidates: Sequence[Candidate],
    config: TemporalRerankConfig | None = None,
) -> Dict[str, Tuple[float, float]]:
    """Return ``event_id -> (rerank_score, neighbour_support)``.

    The base signal is reciprocal rank.  Support comes only from other hits in
    the same video and decays linearly to zero at ``window_seconds``.  Capping
    neighbours avoids rewarding videos merely because they were sampled more
    densely than the rest of the corpus.
    """
    cfg = config or TemporalRerankConfig()
    if not candidates:
        return {}

    ranks = {c.event.event_id: rank
             for rank, c in enumerate(candidates, start=1)}
    by_video: Dict[str, List[Candidate]] = {}
    for candidate in candidates:
        by_video.setdefault(candidate.video_id, []).append(candidate)

    output: Dict[str, Tuple[float, float]] = {}
    for rows in by_video.values():
        rows.sort(key=lambda c: (c.t_start, c.event.event_id))
        for candidate in rows:
            event_id = candidate.event.event_id
            rank = ranks[event_id]
            base = 1.0 / (cfg.rank_k + rank)
            neighbours: List[float] = []
            for other in rows:
                if other is candidate:
                    continue
                distance = abs(float(other.t_start) - float(candidate.t_start))
                if distance > cfg.window_seconds:
                    continue
                time_weight = 1.0 - distance / cfg.window_seconds
                neighbours.append(
                    time_weight / (cfg.rank_k + ranks[other.event.event_id]))
            neighbours.sort(reverse=True)
            support = sum(neighbours[:cfg.max_neighbours])
            output[event_id] = (base + cfg.alpha * support, support)
    return output


def rerank_temporal(
    candidates: Sequence[Candidate],
    config: TemporalRerankConfig | None = None,
    top_k: int | None = None,
) -> List[Candidate]:
    """Return candidates reordered by temporal support, deterministically."""
    cfg = config or TemporalRerankConfig()
    scores = temporal_scores(candidates, cfg)
    original_rank = {c.event.event_id: rank
                     for rank, c in enumerate(candidates, start=1)}
    ranked = sorted(
        candidates,
        key=lambda c: (
            -scores[c.event.event_id][0],
            original_rank[c.event.event_id],
            c.event.event_id,
        ),
    )
    return ranked if top_k is None else ranked[:max(0, int(top_k))]
