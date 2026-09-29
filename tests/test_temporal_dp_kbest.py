"""k-best alignment: many submittable TRAKE rows per video, constraints intact.

`align_all` gives one row per video, which wastes the 100-row budget on wrong
videos. TRAKE pays partial credit per moment once the video is right, so extra
alignments of the RIGHT video are worth real score. These tests lock that the
extra paths are genuinely distinct and still obey every DP constraint.
"""
from aic2026.core.models import Candidate, Event
from aic2026.core.temporal_dp import DPParams, align, align_all, align_k_best


def ev(vid, eid, t, kf):
    return Event(video_id=vid, event_id=eid, t_start=t, t_end=t + 10,
                 start_frame=kf - 5, end_frame=kf + 5, keyframe_idx=kf)


def cand(vid, eid, t, kf, score):
    return Candidate(event=ev(vid, eid, t, kf), score=score,
                     modality_scores={"visual": score, "ocr": 0.0, "asr": 0.0})


def two_step_two_options():
    return [
        [cand("L", "L#0", 0.0, 10, 0.9), cand("L", "L#0b", 1.0, 15, 0.7)],
        [cand("L", "L#1", 10.0, 20, 0.9), cand("L", "L#1b", 11.0, 25, 0.7)],
    ]


# --------------------------------------------------------------------------- #
# Diversity: the whole point
# --------------------------------------------------------------------------- #
def test_returns_multiple_distinct_alignments_for_one_video():
    res = align_k_best(two_step_two_options(), per_video=4)
    assert len(res) == 4
    frames = [tuple(r.frames) for r in res]
    assert len(set(frames)) == len(frames)      # no duplicate submission rows


def test_align_all_gives_only_one_row_per_video_but_k_best_gives_many():
    """The regression this module exists to fix."""
    cl = two_step_two_options()
    assert len(align_all(cl, top=100)) == 1     # one video -> one row
    assert len(align_k_best(cl, per_video=6, top=100)) > 1


def test_top_k_best_path_agrees_with_the_argmax_aligner():
    cl = two_step_two_options()
    best = align(cl)
    top = align_k_best(cl, per_video=5)[0]
    assert best is not None
    assert top.video_id == best.video_id
    assert top.frames == best.frames
    assert abs(top.score - best.score) < 1e-12


def test_results_are_sorted_by_score_descending():
    res = align_k_best(two_step_two_options(), per_video=6)
    assert [r.score for r in res] == sorted((r.score for r in res), reverse=True)


# --------------------------------------------------------------------------- #
# Every row must stay submittable
# --------------------------------------------------------------------------- #
def test_every_path_keeps_the_n_frame_guarantee():
    res = align_k_best(two_step_two_options(), per_video=9)
    assert res
    for r in res:
        assert len(r.frames) == 2
        assert None not in r.frames
        assert all(isinstance(f, int) for f in r.frames)


def test_miss_paths_are_interpolated_here_too():
    cl = [[cand("L", "L#0", 0.0, 10, 0.9)], []]     # step 1 has no candidate
    res = align_k_best(cl, per_video=3)
    assert res
    assert all(len(r.frames) == 2 and None not in r.frames for r in res)
    assert res[0].n_miss == 1


# --------------------------------------------------------------------------- #
# Constraints must hold on EVERY path, not just the argmax
# --------------------------------------------------------------------------- #
def test_all_paths_are_time_ordered():
    res = align_k_best(two_step_two_options(), per_video=9)
    for r in res:
        times = [s.t_start for s in r.steps if s is not None]
        assert times == sorted(times)


def test_no_path_reuses_the_same_event_back_to_back():
    """The degenerate 'repeat one strong event' path pays no gap penalty."""
    cl = [
        [cand("L", "L#0", 0.0, 10, 1.0)],
        [cand("L", "L#1", 10.0, 20, 1.0), cand("L", "L#0", 0.0, 10, 0.9)],
        [cand("L", "L#2", 20.0, 30, 1.0), cand("L", "L#0", 0.0, 10, 0.9)],
    ]
    res = align_k_best(cl, per_video=8)
    assert res
    assert res[0].frames == [10, 20, 30]
    for r in res:
        real = [s for s in r.steps if s is not None]
        for a, b in zip(real, real[1:]):
            assert not (a.t_start == b.t_start
                        and a.event.event_id == b.event.event_id)


def test_gap_max_is_enforced_on_every_path():
    p = DPParams(gap_max=5.0)               # options are ~10s apart
    cl = two_step_two_options()
    for r in align_k_best(cl, params=p, per_video=9):
        times = [s.t_start for s in r.steps if s is not None]
        for a, b in zip(times, times[1:]):
            assert b - a <= 5.0


# --------------------------------------------------------------------------- #
# Multi-video behaviour + determinism
# --------------------------------------------------------------------------- #
def test_spans_videos_and_ranks_the_stronger_one_first():
    cl = [
        [cand("L", "L#0", 0.0, 10, 0.9), cand("M", "M#0", 0.0, 100, 0.4)],
        [cand("L", "L#1", 10.0, 20, 0.9), cand("M", "M#1", 10.0, 110, 0.4)],
    ]
    res = align_k_best(cl, per_video=2, top=100)
    assert res[0].video_id == "L"
    assert {r.video_id for r in res} == {"L", "M"}
    assert len(res) <= 4                    # per_video cap honoured per video


def test_top_caps_the_global_result_count():
    cl = two_step_two_options()
    assert len(align_k_best(cl, per_video=9, top=3)) == 3


def test_deterministic_across_runs():
    cl = two_step_two_options()
    a = [(r.video_id, tuple(r.frames)) for r in align_k_best(cl, per_video=9)]
    b = [(r.video_id, tuple(r.frames)) for r in align_k_best(cl, per_video=9)]
    assert a == b


def test_empty_and_degenerate_inputs():
    assert align_k_best([]) == []
    assert align_k_best([[]]) == []
    assert align_k_best(two_step_two_options(), per_video=0) == []
