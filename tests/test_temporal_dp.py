from aic2026.core.models import Candidate, Event
from aic2026.core.temporal_dp import (DPParams, align, align_all,
                                 cross_modal_event_score)


def ev(vid, eid, t, kf):
    return Event(video_id=vid, event_id=eid, t_start=t, t_end=t + 10,
                 start_frame=kf - 5, end_frame=kf + 5, keyframe_idx=kf)


def cand(vid, eid, t, kf, score, mod=None):
    return Candidate(event=ev(vid, eid, t, kf), score=score,
                     modality_scores=mod or {"visual": score, "ocr": 0.0, "asr": 0.0})


def test_align_picks_single_video_in_order():
    cl = [
        [cand("L", "L#0", 0.0, 10, 0.9), cand("M", "M#0", 0.0, 100, 0.3)],
        [cand("L", "L#1", 10.0, 20, 0.9), cand("M", "M#1", 10.0, 110, 0.3)],
    ]
    res = align(cl)
    assert res is not None
    assert res.video_id == "L"
    assert res.frames == [10, 20]
    assert res.n_miss == 0


def test_miss_step_is_interpolated_never_none():
    # sub-event 1 has NO candidate at all -> forced MISS, but must still emit a frame
    cl = [[cand("L", "L#0", 0.0, 10, 0.9)], []]
    res = align(cl)
    assert res is not None
    assert len(res.frames) == 2          # N-frame guarantee
    assert None not in res.frames
    assert all(isinstance(f, int) for f in res.frames)
    assert res.n_miss == 1


def test_cross_modal_hint_soft_penalty_not_zero():
    p = DPParams(modality_confidence_threshold=0.6, miss_factor=0.5)
    # hint satisfied -> full score
    assert cross_modal_event_score(None, "ocr", 0.8, {"ocr": 0.9}, p) == 0.8
    # hint unmet -> soft penalty x0.5, NOT zero
    assert cross_modal_event_score(None, "ocr", 0.8, {"ocr": 0.2}, p) == 0.8 * 0.5
    # no hint -> untouched
    assert cross_modal_event_score(None, None, 0.8, {"ocr": 0.0}, p) == 0.8


def test_gap_max_blocks_far_transition_but_still_returns_n_frames():
    p = DPParams(gap_max=5.0)  # events are 10s apart -> transition invalid
    cl = [
        [cand("L", "L#0", 0.0, 10, 0.9)],
        [cand("L", "L#1", 100.0, 20, 0.9)],
    ]
    res = align(cl, params=p)
    assert res is not None
    assert len(res.frames) == 2          # still submittable despite forced MISS
    assert None not in res.frames


def test_no_degenerate_single_event_repeat():
    """Regression: a strong event present at t=0 in every step must NOT be
    reused for all N steps (gap-free repeat) instead of the correct progression.
    """
    cl = [
        [cand("L", "L#0", 0.0, 10, 1.0)],
        [cand("L", "L#1", 10.0, 20, 1.0), cand("L", "L#0", 0.0, 10, 0.9)],
        [cand("L", "L#2", 20.0, 30, 1.0), cand("L", "L#0", 0.0, 10, 0.9)],
    ]
    res = align(cl)
    assert res is not None
    assert res.frames == [10, 20, 30]           # distinct, time-ordered
    assert len(set(res.frames)) == 3
    assert res.n_miss == 0


def test_align_all_ranks_videos():
    cl = [
        [cand("L", "L#0", 0.0, 10, 0.9), cand("M", "M#0", 0.0, 100, 0.5)],
        [cand("L", "L#1", 10.0, 20, 0.9), cand("M", "M#1", 10.0, 110, 0.5)],
    ]
    results = align_all(cl, top=10)
    assert [r.video_id for r in results][:2] == ["L", "M"]
    assert all(len(r.frames) == 2 for r in results)
