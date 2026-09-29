"""Metric-aware submission construction: the ranking metric IS the objective.

THE METRIC (official, AIC 2026 preliminary, section 2.2 of the BTC document)

    Final(query) = (1/|K|) * sum_{k in K} max_{1<=i<=k} R-Score(r_i),
    K = {1, 5, 20, 50, 100}
                                          ^^^ a MAX over a prefix, not a sum.

Two consequences that "sort candidates by score, emit 100 rows" throws away.

1.  POSITION BUYS TIERS, NOTHING ELSE.  A row at position i can only ever help
    the tiers k >= i.  Define

        w(i) = #{k in K : k >= i}

    giving w(1)=5, w(2..5)=4, w(6..20)=3, w(21..50)=2, w(51..100)=1, w(>100)=0.
    Slot 1 is worth five times slot 51.  There are only five tiers, so there are
    only five distinct "prices" for the 100 slots.

2.  BECAUSE IT IS A MAX, REDUNDANCY IS FREE FOR THE JUDGE AND EXPENSIVE FOR YOU.
    A second row that is correct under exactly the same circumstances as an
    earlier row adds *nothing* to any tier.  Sorting by score alone spends the
    budget on near-duplicates: 100 frames of one shot (KIS), one answer string
    repeated 100 times (QA), one alignment per video (TRAKE).

So a submission is a PORTFOLIO, not a sorted list.

DERIVATION
----------
Introduce a latent truth state theta (which video, where the window is, what the
answer is).  Given theta, R is deterministic.  Then

    E[Final] = (1/|K|) * sum_k  E_theta[ max_{i<=k} R(r_i; theta) ]

For a fixed theta, `max over prefix k` is a running maximum, and a running
maximum telescopes into the increments that produced it.  If row r placed at
position pos raised the running max under theta by delta(r, theta), that
increment is counted by every tier k >= pos, i.e. exactly w(pos) times.  Hence

    E[Final] = (1/|K|) * sum over placements  w(pos) * sum_theta p_theta * delta

which is a sum of independent (increment x slot-price) terms.  Two corollaries:

  * greedy is the right algorithm -- at each slot take the row with the largest
    probability-weighted increment.  Under this model greedy is *optimal* (the
    available increments are fixed and w is non-increasing, so the rearrangement
    inequality applies); in the general submodular case it is (1 - 1/e);
  * a row whose increment is 0 under every theta is worth exactly 0 -- but it
    also COSTS nothing, because the metric never penalises a wrong answer.  So
    we still fill to 100 ("free recall"), we just stop pretending it is ranked.

The binary special case is worth stating plainly, because it is the whole KIS
strategy in one line:  partition hypotheses into mutually exclusive groups (same
group == correct under the same theta), keep ONE row per group, and order groups
by descending calibrated probability.

TUNING NOTE
-----------
`tol_frames` is the only genuinely data-dependent knob here: it encodes how wide
the judge's accepted window [s, e] is, which is what decides whether two frames
are the same hypothesis or two.  It must be fitted on the real dev set, not
guessed -- see the defaults below for the reasoning behind the starting values
and the measurement that replaces them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .calibration import Calibrator, IdentityCalibrator
from ..core.scoring import DEFAULT_KS, normalize_answer

MAX_ROWS = 100

# KIS/QA: the judge accepts any frame inside the event span [s, e].  Corpus fps
# is 25 for 781/873 videos, and the official keyframe spacing has median 2.16s,
# so +-25 frames (1.0s) is a conservative half-width: too small wastes slots on
# frames that are really the same hypothesis, too large merges genuinely
# different moments and hides recall.
#
# ĐÃ FIT ngày 2026-08-08 (`fit_tol_frames.py`, dev set 60 câu KIS thật, quét 17
# giá trị tol từ 0 tới 500, ứng viên truy hồi một lần rồi tái dùng nên chỉ có tol
# thay đổi).  Kết quả với RankCalibrator:
#
#     tol 0 .. 100   -> điểm phẳng 0.1300  (13/17 điểm quét, plateau thật)
#     tol 150        -> 0.0900             (sụt rõ: tol quá lớn gộp nhầm các
#                                           thời điểm khác nhau -> mất recall)
#
# 25 nằm giữa plateau nên GIỮ NGUYÊN.  Đỉnh của đường cong là tol=18 (0.1333)
# nhưng chênh lệch 0.0033 nhỏ hơn sai số ghép cặp (±0.0067) — lấy đỉnh là chạy
# theo nhiễu lấy mẫu của 60 câu.  Đây là lý do README nói "chọn plateau, không
# chọn đỉnh".
#
# Cần fit lại khi: dev set đổi, hoặc encoder đổi (plateau phụ thuộc mật độ ứng
# viên trong một video, mà cái đó do tầng TÌM quyết định).
KIS_TOL_FRAMES = 25

# TRAKE: the BTC document states the per-moment window is "thuong la duoi 10
# frame", i.e. < 0.4s at 25 fps.  +-5 frames is that window's half-width.  This
# is ~5x tighter than KIS and is the reason official keyframes alone cannot
# score TRAKE: a keyframe lands inside a 10-frame window roughly 10/54 ~ 18% of
# the time at median spacing.
TRAKE_TOL_FRAMES = 5


# --------------------------------------------------------------------------- #
# Slot pricing
# --------------------------------------------------------------------------- #
def slot_weight(pos: int, ks: Sequence[int] = DEFAULT_KS) -> int:
    """w(pos) = number of tiers a row at 1-indexed `pos` can still influence."""
    if pos < 1:
        raise ValueError(f"position must be 1-indexed, got {pos}")
    return sum(1 for k in ks if k >= pos)


def slot_weights(ks: Sequence[int] = DEFAULT_KS,
                 max_rows: int = MAX_ROWS) -> List[int]:
    return [slot_weight(i, ks) for i in range(1, max_rows + 1)]


# --------------------------------------------------------------------------- #
# Model objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Scenario:
    """One weighted guess about the unknown ground truth (a theta)."""
    key: str
    weight: float


@dataclass
class Hypothesis:
    """One candidate submission row, plus its payoff under each scenario.

    `payoff` is sparse: scenarios absent from the dict score 0.  That is what
    keeps the greedy loop cheap -- a hypothesis only ever touches the handful of
    scenarios it could possibly satisfy.
    """
    row: Dict
    key: str
    payoff: Dict[str, float] = field(default_factory=dict)
    prior: float = 0.0          # tie-break + ordering for the free-recall tail


@dataclass
class Placement:
    pos: int
    weight: int
    gain: float
    hypothesis_key: str
    free_recall: bool


@dataclass
class PortfolioResult:
    rows: List[Dict]
    expected_final: float
    placements: List[Placement]
    n_scenarios: int
    covered_mass: float

    def as_dict(self) -> dict:
        return {
            "n_rows": len(self.rows),
            "expected_final": self.expected_final,
            "n_scenarios": self.n_scenarios,
            "covered_mass": self.covered_mass,
            "n_scoring_rows": sum(1 for p in self.placements if not p.free_recall),
        }


# --------------------------------------------------------------------------- #
# The optimizer
# --------------------------------------------------------------------------- #
def marginal_gain(hypothesis: Hypothesis, running_best: Dict[str, float],
                  scenario_weight: Dict[str, float]) -> float:
    """Probability-weighted increase in the running max, summed over scenarios."""
    total = 0.0
    for skey, payoff in hypothesis.payoff.items():
        w = scenario_weight.get(skey)
        if not w:
            continue
        delta = payoff - running_best.get(skey, 0.0)
        if delta > 0.0:
            total += w * delta
    return total


def optimize_slots(hypotheses: Sequence[Hypothesis],
                   scenarios: Sequence[Scenario],
                   ks: Sequence[int] = DEFAULT_KS,
                   max_rows: int = MAX_ROWS,
                   fill_free_recall: bool = True) -> PortfolioResult:
    """Greedy portfolio construction; see the module docstring for the proof.

    Deterministic: ties break on (gain desc, prior desc, key asc).

    `fill_free_recall=True` keeps emitting rows after every marginal gain has
    hit zero.  Those rows cannot raise the score, but they cannot lower it
    either (the metric has no penalty term), and they are the cheap hedge
    against the scenario set being wrong -- which it will be.
    """
    scenario_weight = {s.key: float(s.weight) for s in scenarios if s.weight > 0.0}
    running_best: Dict[str, float] = {}
    rows: List[Dict] = []
    placements: List[Placement] = []
    expected_units = 0.0
    n_ks = len(list(ks))
    if n_ks == 0:
        raise ValueError("ks must be non-empty")

    # ADMISSIBLE BOUND for pruning: gain(h) <= potential(h), because
    # running_best only ever grows, so every delta can only shrink.  Keeping
    # `remaining` sorted by potential descending lets each slot stop scanning as
    # soon as the bound can no longer beat the incumbent.  Popping preserves the
    # order, and the bound stays valid for the whole run.
    def potential_of(h: Hypothesis) -> float:
        return sum(scenario_weight.get(k, 0.0) * v for k, v in h.payoff.items())

    # (potential, hypothesis), potential cached because scenario_weight is fixed.
    remaining: List[Tuple[float, Hypothesis]] = sorted(
        ((potential_of(h), h) for h in hypotheses),
        key=lambda t: (-t[0], -t[1].prior, t[1].key),
    )

    limit = min(max_rows, len(remaining))
    for pos in range(1, limit + 1):
        weight = slot_weight(pos, ks)
        best: Optional[Tuple[float, float, str, int]] = None
        for idx, (pot, h) in enumerate(remaining):
            # `<` not `<=`: a hypothesis whose bound merely TIES the incumbent
            # gain must still be evaluated, because it may win the prior
            # tie-break. Only a strictly smaller bound is safe to prune.
            if best is not None and pot < best[0]:
                break
            gain = marginal_gain(h, running_best, scenario_weight)
            if gain <= 0.0:
                continue
            # Order on (-gain, -prior, key): max gain, then max prior, then
            # lexicographically smallest key -> fully deterministic.
            if best is None or (-gain, -h.prior, h.key) < (-best[0], -best[1], best[2]):
                best = (gain, h.prior, h.key, idx)

        if best is None:
            if not fill_free_recall:
                break
            # Nothing improves any scenario. Emit the best remaining hypothesis
            # anyway (free recall) and mark the row as non-scoring.
            best_idx = min(range(len(remaining)),
                           key=lambda i: (-remaining[i][1].prior, remaining[i][1].key))
            best_gain = 0.0
        else:
            best_gain, _, _, best_idx = best

        chosen = remaining.pop(best_idx)[1]
        rows.append(chosen.row)
        placements.append(Placement(pos=pos, weight=weight, gain=best_gain,
                                    hypothesis_key=chosen.key,
                                    free_recall=(best_gain <= 0.0)))
        expected_units += weight * best_gain
        for skey, payoff in chosen.payoff.items():
            if payoff > running_best.get(skey, 0.0):
                running_best[skey] = payoff

    return PortfolioResult(
        rows=rows,
        expected_final=expected_units / n_ks,
        placements=placements,
        n_scenarios=len(scenario_weight),
        covered_mass=sum(scenario_weight.get(k, 0.0) * v
                         for k, v in running_best.items()),
    )


def expected_final_of(ordered: Sequence[Hypothesis],
                      scenarios: Sequence[Scenario],
                      ks: Sequence[int] = DEFAULT_KS,
                      max_rows: int = MAX_ROWS) -> float:
    """E[Final] of an ARBITRARY ordering -- the A/B baseline for the optimizer.

    Use this to score what `submission.diversify_tail` (or a plain score sort)
    produces, then compare against `optimize_slots(...).expected_final` on the
    same hypotheses and scenarios.  Same inputs, one number, no argument.
    """
    scenario_weight = {s.key: float(s.weight) for s in scenarios if s.weight > 0.0}
    running_best: Dict[str, float] = {}
    n_ks = len(list(ks))
    units = 0.0
    for pos, h in enumerate(ordered[:max_rows], start=1):
        weight = slot_weight(pos, ks)
        units += weight * marginal_gain(h, running_best, scenario_weight)
        for skey, payoff in h.payoff.items():
            if payoff > running_best.get(skey, 0.0):
                running_best[skey] = payoff
    return units / n_ks if n_ks else 0.0


# --------------------------------------------------------------------------- #
# Scenario weighting
# --------------------------------------------------------------------------- #
def normalise_scenarios(raw: Sequence[Tuple[str, float]],
                        mass: Optional[float] = None) -> List[Scenario]:
    """Turn calibrated per-candidate probabilities into a scenario distribution.

    Exactly one theta is true, so the weights must sum to <= 1.  If the
    calibrated values already sum to <= 1 they are KEPT AS THEY ARE and the
    residual (1 - sum) is left unassigned -- that residual is the honest
    statement "the ground truth may not be in my candidate set at all", and it
    is what stops `expected_final` from being inflated.

    Only when the values sum to > 1 (an over-confident calibrator) are they
    rescaled down.  Pass `mass` explicitly to override, e.g. with a measured
    recall@100 from the dev set.
    """
    total = sum(max(0.0, w) for _, w in raw)
    if total <= 0.0:
        return []
    if mass is None:
        scale = 1.0 / total if total > 1.0 else 1.0
    else:
        scale = float(mass) / total
    return [Scenario(key=k, weight=max(0.0, w) * scale) for k, w in raw]


# --------------------------------------------------------------------------- #
# Task-specific portfolio builders
# --------------------------------------------------------------------------- #
def _within(a: int, b: int, tol: int) -> bool:
    return abs(int(a) - int(b)) <= tol


def kis_portfolio(candidates: Sequence,
                  calibrator: Optional[Calibrator] = None,
                  *,
                  tol_frames: int = KIS_TOL_FRAMES,
                  mass: Optional[float] = None,
                  frame_offsets: Sequence[int] = (0,),
                  ks: Sequence[int] = DEFAULT_KS,
                  max_rows: int = MAX_ROWS,
                  max_scenarios: int = 400) -> PortfolioResult:
    """KIS: hypotheses are (video, frame); scenarios are (video, true moment).

    `frame_offsets` is the hedge against the unresolved frame-index convention.
    The audit found MIXED_CONVENTION (no safe global offset; within-video winner
    consistency only 33%), with offset 0 winning ~79% of discriminative samples.
    Passing e.g. (0, -1, 1, -2, 2) emits the same moment at several offsets; the
    optimizer will only spend a slot on an offset if it actually covers scenario
    mass that offset 0 misses, so the hedge is self-limiting.
    """
    calibrator = calibrator or IdentityCalibrator()

    seen: set = set()
    raw: List[Tuple[str, float]] = []
    moments: List[Tuple[str, str, int, float]] = []   # (skey, video, frame, p)
    for c in candidates:
        video, frame = str(c.video_id), int(c.frame_idx)
        skey = f"{video}@{frame}"
        if skey in seen:
            continue
        seen.add(skey)
        # `prior()` chứ không phải `predict()`: một số calibrator (RankCalibrator)
        # chờ HẠNG chứ không chờ điểm. `candidates` đã được `retrieve()` sắp giảm
        # dần, nên số thứ tự trong danh sách đã lọc trùng chính là hạng.
        p = calibrator.prior(float(c.score), len(raw) + 1)
        raw.append((skey, p))
        moments.append((skey, video, frame, p))
        if len(raw) >= max_scenarios:
            break
    scenarios = normalise_scenarios(raw, mass)

    hyps: List[Hypothesis] = []
    for skey, video, frame, p in moments:
        for off in frame_offsets:
            emitted = frame + int(off)
            if emitted < 0:
                continue
            payoff = {
                s_key: 1.0
                for s_key, s_video, s_frame, _ in moments
                if s_video == video and _within(emitted, s_frame, tol_frames)
            }
            if not payoff:
                continue
            hyps.append(Hypothesis(
                row={"video_id": video, "frame_idx": emitted},
                key=f"kis:{video}:{emitted}",
                payoff=payoff,
                # Offset-0 rows outrank their own hedges at equal gain.
                prior=p - 1e-6 * abs(int(off)),
            ))
    return optimize_slots(hyps, scenarios, ks=ks, max_rows=max_rows)


def qa_portfolio(candidates: Sequence,
                 answers: Sequence[Tuple[str, float]],
                 calibrator: Optional[Calibrator] = None,
                 *,
                 tol_frames: int = KIS_TOL_FRAMES,
                 mass: Optional[float] = None,
                 ks: Sequence[int] = DEFAULT_KS,
                 max_rows: int = MAX_ROWS,
                 max_moments: int = 60) -> PortfolioResult:
    """QA: hypotheses are (video, frame, answer).

    `answers` is [(answer_text, probability), ...] from the VLM/LLM -- a
    DISTRIBUTION, not a single string.  This is the cheapest unclaimed win in
    the whole task: R-Score needs video AND frame AND answer to match, answers
    are mutually exclusive, so each distinct answer is a distinct scenario.  A
    submission that repeats one answer 100 times has exactly one answer
    hypothesis; spending a few slots on the runner-up answer at the same
    (video, frame) costs nothing and covers real probability mass.

    Answers are compared with `scoring.normalize_answer`, the same normaliser
    the offline scorer uses, so two spellings of one answer stay one scenario.
    """
    calibrator = calibrator or IdentityCalibrator()

    ans: List[Tuple[str, str, float]] = []       # (norm, raw, p)
    seen_ans: set = set()
    for text, prob in answers:
        norm = normalize_answer(text)
        if norm in seen_ans:
            continue
        seen_ans.add(norm)
        ans.append((norm, str(text), max(0.0, float(prob))))
    if not ans:
        return PortfolioResult([], 0.0, [], 0, 0.0)
    a_total = sum(p for _, _, p in ans) or 1.0
    ans = [(n, r, p / a_total) for n, r, p in ans]

    moments: List[Tuple[str, int, float]] = []   # (video, frame, p_moment)
    seen_m: set = set()
    for c in candidates:
        video, frame = str(c.video_id), int(c.frame_idx)
        if (video, frame) in seen_m:
            continue
        seen_m.add((video, frame))
        moments.append((video, frame,
                        calibrator.prior(float(c.score), len(moments) + 1)))
        if len(moments) >= max_moments:
            break

    raw: List[Tuple[str, float]] = []
    grid: List[Tuple[str, str, int, str, str, float]] = []
    for video, frame, pm in moments:
        for norm, rawtext, pa in ans:
            skey = f"{video}@{frame}|{norm}"
            raw.append((skey, pm * pa))
            grid.append((skey, video, frame, norm, rawtext, pm * pa))
    scenarios = normalise_scenarios(raw, mass)

    hyps: List[Hypothesis] = []
    for skey, video, frame, norm, rawtext, p in grid:
        payoff = {
            s_key: 1.0
            for s_key, s_video, s_frame, s_norm, _, _ in grid
            if s_video == video and s_norm == norm
            and _within(frame, s_frame, tol_frames)
        }
        hyps.append(Hypothesis(
            row={"video_id": video, "frame_idx": frame, "answer": rawtext},
            key=f"qa:{video}:{frame}:{norm}",
            payoff=payoff,
            prior=p,
        ))
    return optimize_slots(hyps, scenarios, ks=ks, max_rows=max_rows)


def jitter_alignment(frames: Sequence[int], offset: int) -> List[int]:
    """Shift a whole alignment by `offset` frames, clamped at 0."""
    return [max(0, int(f) + int(offset)) for f in frames]


def trake_portfolio(align_results: Sequence,
                    calibrator: Optional[Calibrator] = None,
                    *,
                    tol_frames: int = TRAKE_TOL_FRAMES,
                    mass: Optional[float] = None,
                    jitters: Sequence[int] = (0,),
                    ks: Sequence[int] = DEFAULT_KS,
                    max_rows: int = MAX_ROWS,
                    max_scenarios: int = 200) -> PortfolioResult:
    """TRAKE: hypotheses are (video, alignment); payoff is FRACTIONAL.

    Two facts drive this:

      * wrong video => 0 for the whole row, so scenario mass is per-video;
      * right video => all 100 slots stay useful, because R-Score is the
        fraction of moments whose frame lands in its window.  So the correct
        play once you trust a video is to spend many slots on DIFFERENT
        alignments of that same video -- which is what `temporal_dp.align_k_best`
        supplies and what a one-path-per-video ranking throws away.

    `jitters` adds shifted copies. Because the accepted window is ~10 frames,
    a shift of +-10 or more produces a genuinely different hypothesis rather
    than a duplicate; a shift smaller than `tol_frames` would be redundant and
    the optimizer would (correctly) assign it zero gain and never spend a slot.
    """
    calibrator = calibrator or IdentityCalibrator()

    raw: List[Tuple[str, float]] = []
    truths: List[Tuple[str, str, Tuple[int, ...], float]] = []
    seen: set = set()
    for r in align_results:
        video = str(r.video_id)
        frames = tuple(int(f) for f in r.frames)
        skey = f"{video}#{'-'.join(map(str, frames))}"
        if skey in seen:
            continue
        seen.add(skey)
        p = calibrator.prior(float(r.score), len(raw) + 1)
        raw.append((skey, p))
        truths.append((skey, video, frames, p))
        if len(raw) >= max_scenarios:
            break
    scenarios = normalise_scenarios(raw, mass)

    hyps: List[Hypothesis] = []
    emitted: set = set()
    for skey, video, frames, p in truths:
        for off in jitters:
            cand_frames = tuple(jitter_alignment(frames, off))
            hkey = f"trake:{video}:{'-'.join(map(str, cand_frames))}"
            if hkey in emitted:
                continue
            emitted.add(hkey)
            payoff: Dict[str, float] = {}
            for s_key, s_video, s_frames, _ in truths:
                if s_video != video or len(s_frames) != len(cand_frames):
                    continue
                hits = sum(1 for a, b in zip(cand_frames, s_frames)
                           if _within(a, b, tol_frames))
                if hits:
                    payoff[s_key] = hits / len(s_frames)
            if not payoff:
                continue
            hyps.append(Hypothesis(
                row={"video_id": video, "frames": list(cand_frames)},
                key=hkey,
                payoff=payoff,
                prior=p - 1e-6 * abs(int(off)),
            ))
    return optimize_slots(hyps, scenarios, ks=ks, max_rows=max_rows)
