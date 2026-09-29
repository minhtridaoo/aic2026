"""Temporal alignment for TRAKE via Dynamic Programming.

Features / fixes baked in:
  * MISS branch: a step may be skipped when no good candidate exists. The DP
    backpointer uses `None` as the "no predecessor" sentinel (NOT -1, which in
    Python indexes the last element and could silently mis-link a path).
  * cross-modal-per-event: each sub-event may prefer a different modality via an
    LLM hint. A WRONG hint applies a SOFT penalty (xmiss_factor, default 0.5) —
    never a hard zero — so one bad hint cannot annihilate a good candidate.
  * N-FRAME GUARANTEE (submission fix): the reconstructed path always yields
    exactly N frame indices. MISS steps are INTERPOLATED from neighbouring real
    steps instead of emitting None, because TRAKE is scored per-frame with
    partial credit and there is no penalty for a wrong guess — a blank frame can
    only score 0, an interpolated guess can score > 0.

All tunables are passed via `DPParams` (config-driven) and must be tuned on real
past-year queries with a video-level split, choosing a stable plateau rather
than a sharp peak (overfitting guard, critique #5).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models import Candidate

MISS = None  # explicit sentinel for a skipped step


@dataclass
class DPParams:
    lam: float = 0.05                     # time-gap penalty per second
    gap_max: float = 120.0                # max allowed gap between consecutive events (s)
    penalty_miss: float = 0.35            # cost of skipping a step
    modality_confidence_threshold: float = 0.6  # on normalised modality score [0,1]
    miss_factor: float = 0.5              # soft penalty when a modality hint is unmet
    # order_penalty_per_violation: previously a dead knob. Ordering is enforced as
    # a hard transition constraint below, so this is intentionally NOT a free
    # parameter. Keep it out of the tuning grid.


@dataclass
class AlignResult:
    video_id: str
    frames: List[int]                     # length N, guaranteed
    steps: List[Optional[Candidate]]      # None = MISS (before interpolation)
    score: float                          # normalised path score
    n_miss: int
    confidence: float


def cross_modal_event_score(cand: Candidate, hint: Optional[str],
                            norm_score: float, norm_modality: Dict[str, float],
                            params: DPParams) -> float:
    """Reward for placing `cand` at a step wanting modality `hint`.

    Base is the (normalised) fused score. If a hint is given and the hinted
    channel does NOT confidently support this event, apply a soft multiplicative
    penalty instead of discarding the candidate.
    """
    if not hint:
        return norm_score
    m = norm_modality.get(hint, 0.0)
    if m >= params.modality_confidence_threshold:
        return norm_score
    return norm_score * params.miss_factor


def _normalize(cand_lists: List[List[Candidate]]):
    """Min-max normalise scores PER STEP (across all videos of that sub-event).

    Per-step (not global) keeps the reward of each step's best candidate near 1.0
    regardless of raw fusion scale, so it stays commensurate with the time-gap
    and miss penalties in the DP recurrence — a global normalisation shrinks
    later-step candidates and lets the gap penalty wipe out perfect matches
    (the lam-too-large pathology). Cross-video ranking is still preserved because
    a video that tops every step accumulates ~1.0 per step.
    """
    norm_score: Dict[int, float] = {}
    mod_norm: Dict[int, Dict[str, float]] = {}
    for lst in cand_lists:
        if not lst:
            continue
        # Max-normalise (not min-max): the step's best candidate -> 1.0, an absent
        # signal -> 0.0. Min-max would collapse a lone/all-equal candidate to 0.0,
        # which made the DP prefer opening with a MISS to dodge a gap penalty.
        s_max = max((c.score for c in lst), default=0.0)
        for c in lst:
            norm_score[id(c)] = (c.score / s_max) if s_max > 0 else 0.0
        for ch in ("visual", "ocr", "asr"):
            m_max = max((c.modality_scores.get(ch, 0.0) for c in lst), default=0.0)
            for c in lst:
                v = c.modality_scores.get(ch, 0.0)
                mod_norm.setdefault(id(c), {})[ch] = (v / m_max) if m_max > 0 else 0.0
    return norm_score, mod_norm


def _interpolate_misses(frames_opt: List[Optional[int]]) -> List[int]:
    """Fill None frames by linear interpolation between known neighbours.

    Boundary Nones are extrapolated from the nearest known frame (flat/step).
    If nothing is known, everything becomes 0.
    """
    n = len(frames_opt)
    known = [(i, f) for i, f in enumerate(frames_opt) if f is not None]
    if not known:
        return [0] * n
    out: List[int] = [f if f is not None else 0 for f in frames_opt]
    for i in range(n):
        if frames_opt[i] is not None:
            continue
        left = [(j, f) for j, f in known if j < i]
        right = [(j, f) for j, f in known if j > i]
        if left and right:
            (lj, lf), (rj, rf) = left[-1], right[0]
            frac = (i - lj) / (rj - lj)
            out[i] = int(round(lf + frac * (rf - lf)))
        elif left:
            out[i] = left[-1][1]
        else:
            out[i] = right[0][1]
    return out


def align(cand_lists: List[List[Candidate]],
          hints: Optional[List[Optional[str]]] = None,
          params: Optional[DPParams] = None) -> Optional[AlignResult]:
    """Align N sub-events to a single best video.

    cand_lists[j] = candidates retrieved for sub-event j (may span videos).
    Returns the best AlignResult, or None if there are no candidates at all.
    """
    params = params or DPParams()
    n = len(cand_lists)
    if n == 0:
        return None
    hints = hints or [None] * n

    videos = {c.video_id for lst in cand_lists for c in lst}
    if not videos:
        return None

    norm_score, mod_norm = _normalize(cand_lists)
    results = [r for v in videos
               if (r := _align_single_video(v, cand_lists, hints, params,
                                            norm_score, mod_norm))]
    if not results:
        return None
    return max(results, key=lambda r: r.score)


def align_all(cand_lists: List[List[Candidate]],
              hints: Optional[List[Optional[str]]] = None,
              params: Optional[DPParams] = None,
              top: int = 100) -> List[AlignResult]:
    """Rank candidate videos by alignment score -> up to `top` TRAKE rows.

    Each result already carries exactly N frames, so every row is submittable.
    """
    params = params or DPParams()
    n = len(cand_lists)
    if n == 0:
        return []
    hints = hints or [None] * n
    videos = {c.video_id for lst in cand_lists for c in lst}
    norm_score, mod_norm = _normalize(cand_lists)
    results = [r for v in videos
               if (r := _align_single_video(v, cand_lists, hints, params,
                                            norm_score, mod_norm))]
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top]


def _align_single_video(video_id: str, cand_lists, hints, params: DPParams,
                        norm_score: Dict[int, float],
                        mod_norm: Dict[int, Dict[str, float]]
                        ) -> Optional[AlignResult]:
    n = len(cand_lists)
    per_step = [[c for c in cand_lists[j] if c.video_id == video_id] for j in range(n)]

    flat = [c for lst in per_step for c in lst]
    if not flat:
        return None

    # Options per step: candidates + MISS (index == len(candidates)).
    NEG = float("-inf")
    dp: List[List[float]] = []
    back: List[List[Optional[int]]] = []
    last_t: List[List[Optional[float]]] = []

    def options(j):
        return per_step[j] + [MISS]

    def reward(j, opt) -> float:
        if opt is MISS:
            return -params.penalty_miss
        return cross_modal_event_score(
            opt, hints[j], norm_score[id(opt)], mod_norm[id(opt)], params)

    # init step 0
    last_eid: List[List[Optional[str]]] = []  # event_id of the last REAL step
    opts0 = options(0)
    dp.append([reward(0, o) for o in opts0])
    back.append([None] * len(opts0))          # None sentinel: no predecessor
    last_t.append([None if o is MISS else o.t_start for o in opts0])
    last_eid.append([None if o is MISS else o.event.event_id for o in opts0])

    for j in range(1, n):
        opts = options(j)
        row_dp = [NEG] * len(opts)
        row_bk: List[Optional[int]] = [None] * len(opts)
        row_lt: List[Optional[float]] = [None] * len(opts)
        row_eid: List[Optional[str]] = [None] * len(opts)
        prev_opts = options(j - 1)
        for oi, o in enumerate(opts):
            for pi, p in enumerate(prev_opts):
                prev_score = dp[j - 1][pi]
                if prev_score == NEG:
                    continue
                lt_prev = last_t[j - 1][pi]
                eid_prev = last_eid[j - 1][pi]
                trans_pen = 0.0
                new_lt = lt_prev
                new_eid = eid_prev
                if o is not MISS:
                    if lt_prev is not None:
                        if o.t_start < lt_prev:
                            continue                    # ordering violation
                        # Non-decreasing time is allowed (two distinct events may
                        # share a window), but the SAME event cannot be reused —
                        # that degenerate "repeat one strong event" path pays no
                        # gap penalty and would otherwise beat the correct chain.
                        if o.t_start == lt_prev and o.event.event_id == eid_prev:
                            continue
                        gap = o.t_start - lt_prev
                        if gap > params.gap_max:
                            continue                    # gap too large
                        trans_pen = params.lam * gap
                    new_lt = o.t_start
                    new_eid = o.event.event_id
                cand_val = prev_score + reward(j, o) - trans_pen
                if cand_val > row_dp[oi]:
                    row_dp[oi] = cand_val
                    row_bk[oi] = pi
                    row_lt[oi] = new_lt
                    row_eid[oi] = new_eid
        dp.append(row_dp)
        back.append(row_bk)
        last_t.append(row_lt)
        last_eid.append(row_eid)

    # pick best terminal option
    last = dp[n - 1]
    best_oi = max(range(len(last)), key=lambda i: last[i])
    if last[best_oi] == NEG:
        return None

    # reconstruct
    path_opt: List[Optional[int]] = [None] * n
    oi = best_oi
    for j in range(n - 1, -1, -1):
        path_opt[j] = oi
        oi = back[j][oi] if oi is not None else None
        if oi is None and j > 0:
            oi = 0  # defensive; should not trigger given valid backpointers

    steps: List[Optional[Candidate]] = []
    frames_opt: List[Optional[int]] = []
    n_miss = 0
    for j in range(n):
        opts = options(j)
        chosen = opts[path_opt[j]]
        if chosen is MISS:
            steps.append(None)
            frames_opt.append(None)
            n_miss += 1
        else:
            steps.append(chosen)
            frames_opt.append(chosen.frame_idx)

    frames = _interpolate_misses(frames_opt)  # N-frame guarantee
    raw = last[best_oi]
    score = raw / n
    confidence = max(0.0, min(1.0, score)) * (1.0 - 0.5 * n_miss / n)
    return AlignResult(video_id=video_id, frames=frames, steps=steps,
                       score=score, n_miss=n_miss, confidence=confidence)


# --------------------------------------------------------------------------- #
# Từ đây xuống hết file phục vụ tầng `optimize`; phần trên (align / align_all)
# phục vụ đường heuristic. Hai đường dùng chung DP bên dưới, khác ở chỗ giữ lại
# một đường đi tốt nhất hay k đường.
#
# k-BEST alignment (list-Viterbi)
#
# WHY: `align_all` returns the single argmax path per video, so a TRAKE query
# gets at most one row per video -- and then the 100-row budget is spent on
# videos, most of which are wrong.  But the official R-Score for TRAKE is
#
#     0                                  if the video is wrong
#     (1/N) * sum_j I(frame_j in [s_j, e_j])   if the video is right
#
# so once a video is right, EVERY remaining slot is still worth something: a
# different alignment of the same video can hit moments the argmax path missed.
# The per-moment window is under ~10 frames (0.4s at 25 fps) while the official
# keyframes sit ~2.16s apart at the median, so the argmax path being off by one
# keyframe on one moment is the normal case, not the exception.
#
# List-Viterbi gives the k best DISTINCT paths, which is exactly the hypothesis
# set `submission_optimizer.trake_portfolio` needs.  Each state keeps its own
# top-k partial paths, and every partial path carries its OWN (last_t, last_eid)
# because the ordering and same-event-reuse constraints depend on the path, not
# just on the state.
# --------------------------------------------------------------------------- #
@dataclass
class _KEntry:
    """One partial path arriving at a (step, option) state."""
    score: float
    last_t: Optional[float]
    last_eid: Optional[str]
    prev_oi: Optional[int]
    prev_rank: Optional[int]


def align_k_best(cand_lists: List[List[Candidate]],
                 hints: Optional[List[Optional[str]]] = None,
                 params: Optional[DPParams] = None,
                 per_video: int = 8,
                 top: int = 100) -> List[AlignResult]:
    """Up to `per_video` distinct alignments per video, globally ranked.

    Every returned AlignResult still carries exactly N frames, so every one is
    directly submittable. Videos are iterated in sorted order so the output is
    deterministic for a fixed input.
    """
    params = params or DPParams()
    n = len(cand_lists)
    if n == 0 or per_video < 1:
        return []
    hints = hints or [None] * n
    videos = {c.video_id for lst in cand_lists for c in lst}
    if not videos:
        return []

    norm_score, mod_norm = _normalize(cand_lists)
    out: List[AlignResult] = []
    for video_id in sorted(videos):
        out.extend(_align_single_video_kbest(
            video_id, cand_lists, hints, params, norm_score, mod_norm, per_video))
    out.sort(key=lambda r: -r.score)
    return out[:top]


def _align_single_video_kbest(video_id: str, cand_lists, hints, params: DPParams,
                              norm_score: Dict[int, float],
                              mod_norm: Dict[int, Dict[str, float]],
                              k: int) -> List[AlignResult]:
    n = len(cand_lists)
    per_step = [[c for c in cand_lists[j] if c.video_id == video_id] for j in range(n)]
    if not any(per_step):
        return []

    def options(j):
        return per_step[j] + [MISS]

    def reward(j, opt) -> float:
        if opt is MISS:
            return -params.penalty_miss
        return cross_modal_event_score(
            opt, hints[j], norm_score[id(opt)], mod_norm[id(opt)], params)

    # dp[j][oi] = list of up to k _KEntry, sorted by score descending.
    dp: List[List[List[_KEntry]]] = [[
        [_KEntry(score=reward(0, o),
                 last_t=None if o is MISS else o.t_start,
                 last_eid=None if o is MISS else o.event.event_id,
                 prev_oi=None, prev_rank=None)]
        for o in options(0)
    ]]

    for j in range(1, n):
        opts = options(j)
        prev_opts = options(j - 1)
        row: List[List[_KEntry]] = []
        for o in opts:
            bucket: List[_KEntry] = []
            for pi, _prev in enumerate(prev_opts):
                for rank, entry in enumerate(dp[j - 1][pi]):
                    new_lt, new_eid = entry.last_t, entry.last_eid
                    trans_pen = 0.0
                    if o is not MISS:
                        if entry.last_t is not None:
                            if o.t_start < entry.last_t:
                                continue                      # ordering violation
                            if (o.t_start == entry.last_t
                                    and o.event.event_id == entry.last_eid):
                                continue                      # same-event reuse
                            gap = o.t_start - entry.last_t
                            if gap > params.gap_max:
                                continue                      # gap too large
                            trans_pen = params.lam * gap
                        new_lt = o.t_start
                        new_eid = o.event.event_id
                    bucket.append(_KEntry(
                        score=entry.score + reward(j, o) - trans_pen,
                        last_t=new_lt, last_eid=new_eid,
                        prev_oi=pi, prev_rank=rank))
            bucket.sort(key=lambda e: -e.score)
            row.append(bucket[:k])
        dp.append(row)

    terminal = [
        (entry.score, oi, rank)
        for oi, bucket in enumerate(dp[n - 1])
        for rank, entry in enumerate(bucket)
    ]
    terminal.sort(key=lambda t: (-t[0], t[1], t[2]))

    results: List[AlignResult] = []
    seen: set = set()
    for raw, oi, rank in terminal:
        path: List[Optional[int]] = [None] * n
        cur_oi, cur_rank = oi, rank
        for j in range(n - 1, -1, -1):
            if cur_oi is None or cur_rank is None:
                break
            path[j] = cur_oi
            entry = dp[j][cur_oi][cur_rank]
            cur_oi, cur_rank = entry.prev_oi, entry.prev_rank
        if any(p is None for p in path):
            continue                     # defensive: broken backpointer chain

        steps: List[Optional[Candidate]] = []
        frames_opt: List[Optional[int]] = []
        n_miss = 0
        for j in range(n):
            chosen = options(j)[path[j]]  # type: ignore[index]
            if chosen is MISS:
                steps.append(None)
                frames_opt.append(None)
                n_miss += 1
            else:
                steps.append(chosen)
                frames_opt.append(chosen.frame_idx)

        frames = _interpolate_misses(frames_opt)   # N-frame guarantee
        fingerprint = tuple(frames)
        if fingerprint in seen:
            # Distinct DP paths can interpolate to the same frames (e.g. two
            # different MISS placements). Those are the SAME submission row, so
            # they must not consume a slot twice.
            continue
        seen.add(fingerprint)
        score = raw / n
        results.append(AlignResult(
            video_id=video_id, frames=frames, steps=steps, score=score,
            n_miss=n_miss,
            confidence=max(0.0, min(1.0, score)) * (1.0 - 0.5 * n_miss / n)))
        if len(results) >= k:
            break
    return results
