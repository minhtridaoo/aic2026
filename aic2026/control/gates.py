"""Scale-free confidence gate; raw cosine/RRF values are not probabilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

from aic2026.core.models import Candidate


@dataclass(frozen=True)
class GateDecision:
    route: str
    confidence: float
    reasons: List[str] = field(default_factory=list)


def decide(candidates: Iterable[Candidate], *, min_margin: float = 0.08,
           min_channel_agreement: int = 2, min_neighbor_support: float = 0.25,
           neighbor_support: float = 0.0) -> GateDecision:
    rows = list(candidates)
    if not rows:
        return GateDecision("escalate", 0.0, ["no_candidates"])
    top = rows[0]
    active_channels = {
        channel for row in rows for channel, value in row.modality_scores.items()
        if abs(value) > 1e-12
    }
    second_row = rows[1] if len(rows) > 1 else None
    second = second_row.score if second_row is not None else 0.0
    margins = [
        max(0.0, (top.score - second) / max(abs(top.score), 1e-12))
    ]
    for channel in active_channels:
        top_score = top.modality_scores.get(channel, 0.0)
        second_score = (second_row.modality_scores.get(channel, 0.0)
                        if second_row is not None else 0.0)
        if top_score > 0:
            margins.append(
                max(0.0, (top_score - second_score) / abs(top_score)))
    # RRF deliberately compresses rank gaps (rank 1 vs 2 is about 1.6% with
    # k=60), so a gate that only reads fused RRF can never become confident.
    margin = min(1.0, max(margins))
    positive_channels = sum(
        top.modality_scores.get(channel, 0.0) > 0
        for channel in active_channels)
    # A visual-only BTC index cannot be required to agree with OCR/ASR indexes
    # that do not exist. Agreement becomes informative only among active
    # channels.
    required_agreement = min(min_channel_agreement, len(active_channels))
    reasons: List[str] = []
    positive_evidence = (
        top.score > 0
        and bool(active_channels)
        and any(top.modality_scores.get(channel, 0.0) > 0
                for channel in active_channels)
    )
    if not active_channels:
        reasons.append("no_active_channels")
    elif not positive_evidence:
        reasons.append("nonpositive_top_evidence")
    if margin < min_margin:
        reasons.append("small_relative_margin")
    if required_agreement and positive_channels < required_agreement:
        reasons.append("weak_channel_agreement")
    if neighbor_support < min_neighbor_support:
        reasons.append("weak_temporal_support")
    agreement = (0.0 if not required_agreement else
                 min(1.0, positive_channels / required_agreement))
    confidence = max(0.0, min(1.0, 0.5 * margin + 0.25 * agreement
                              + 0.25 * neighbor_support))
    if not positive_evidence:
        confidence = 0.0
    return GateDecision("escalate" if reasons else "fast", confidence, reasons)
