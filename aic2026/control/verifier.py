"""Evidence contract shared by fast and escalated paths."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class VerificationResult:
    accepted: bool
    score: float
    reasons: List[str] = field(default_factory=list)
    evidence: Dict[str, str] = field(default_factory=dict)


def verify_required_evidence(evidence: Dict[str, str],
                             required: List[str]) -> VerificationResult:
    missing = [name for name in required if not evidence.get(name)]
    if missing:
        return VerificationResult(False, 0.0,
                                  [f"missing:{name}" for name in missing],
                                  evidence)
    return VerificationResult(True, 1.0, [], evidence)
