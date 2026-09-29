"""Locks the metric-aware slot allocation against the official scoring rule."""
from aic2026.core.models import Candidate, Event
from aic2026.optimize.submission_optimizer import (Hypothesis, Scenario, expected_final_of,
                                          kis_portfolio, normalise_scenarios,
                                          optimize_slots, qa_portfolio,
                                          slot_weight, trake_portfolio)
from aic2026.core.temporal_dp import AlignResult


def cand(vid, frame, score, t=0.0):
    ev = Event(video_id=vid, event_id=f"{vid}#{frame}", t_start=t, t_end=t + 10,
               start_frame=frame - 5, end_frame=frame + 5, keyframe_idx=frame)
    return Candidate(event=ev, score=score,
                     modality_scores={"visual": score, "ocr": 0.0, "asr": 0.0})


def hyp(key, payoff, prior=0.0, row=None):
    return Hypothesis(row=row or {"k": key}, key=key, payoff=payoff, prior=prior)


# --------------------------------------------------------------------------- #
# Slot pricing must match K = {1, 5, 20, 50, 100}
# --------------------------------------------------------------------------- #
def test_slot_weight_matches_official_tiers():
    assert slot_weight(1) == 5          # helps R@1,5,20,50,100
    assert slot_weight(2) == 4
    assert slot_weight(5) == 4          # last slot of the R@5 tier
    assert slot_weight(6) == 3
    assert slot_weight(20) == 3
    assert slot_weight(21) == 2
    assert slot_weight(50) == 2
    assert slot_weight(51) == 1
    assert slot_weight(100) == 1
    assert slot_weight(101) == 0        # beyond the 100-row cap, worthless


def test_expected_final_reproduces_official_examples():
    """A single certain answer at rank 1 scores 1.0; at rank 2 it scores 0.8."""
    scen = [Scenario("s", 1.0)]
    good = hyp("good", {"s": 1.0}, prior=1.0)
    dud = hyp("dud", {}, prior=0.5)

    assert expected_final_of([good], scen) == 1.0
    # R@1=0, R@5=R@20=R@50=R@100=1  ->  (0+1+1+1+1)/5 = 0.8
    assert abs(expected_final_of([dud, good], scen) - 0.8) < 1e-12


# --------------------------------------------------------------------------- #
# The core claim: redundancy is what a plain score sort wastes
# --------------------------------------------------------------------------- #
def test_optimizer_beats_score_sort_when_top_hypotheses_are_redundant():
    """20 near-duplicates of the best guess push the other guesses past rank 20.

    Score-sort:  A at 1 (w=5), B at 21 (w=2), C at 22 (w=2) -> 0.70
    Optimizer:   A at 1 (w=5), B at 2  (w=4), C at 3  (w=4) -> 0.90
    """
    scen = [Scenario("A", 0.5), Scenario("B", 0.3), Scenario("C", 0.2)]
    dupes = [hyp(f"a{i:02d}", {"A": 1.0}, prior=0.9 - i * 1e-3) for i in range(20)]
    others = [hyp("b", {"B": 1.0}, prior=0.30), hyp("c", {"C": 1.0}, prior=0.20)]

    by_score = sorted(dupes + others, key=lambda h: -h.prior)
    baseline = expected_final_of(by_score, scen)
    optimized = optimize_slots(dupes + others, scen).expected_final

    assert abs(baseline - 0.70) < 1e-12
    assert abs(optimized - 0.90) < 1e-12
    assert optimized > baseline


def test_redundant_rows_are_marked_free_recall_not_scoring():
    scen = [Scenario("A", 0.6), Scenario("B", 0.4)]
    hyps = [hyp("a1", {"A": 1.0}, prior=0.9),
            hyp("a2", {"A": 1.0}, prior=0.8),   # same circumstances as a1
            hyp("b1", {"B": 1.0}, prior=0.5)]
    res = optimize_slots(hyps, scen)

    scoring = [p for p in res.placements if not p.free_recall]
    assert len(scoring) == 2                       # a1 and b1 only
    assert [p.hypothesis_key for p in scoring] == ["a1", "b1"]
    # a2 is still emitted -- a wrong row is never penalised (free recall).
    assert len(res.rows) == 3


def test_higher_probability_takes_the_earlier_slot():
    scen = [Scenario("A", 0.2), Scenario("B", 0.7)]
    hyps = [hyp("a", {"A": 1.0}, prior=0.2), hyp("b", {"B": 1.0}, prior=0.7)]
    res = optimize_slots(hyps, scen)
    assert [p.hypothesis_key for p in res.placements] == ["b", "a"]


def test_deterministic_across_runs_and_input_order():
    scen = [Scenario("A", 0.4), Scenario("B", 0.4), Scenario("C", 0.2)]
    hyps = [hyp("a", {"A": 1.0}, prior=0.4), hyp("b", {"B": 1.0}, prior=0.4),
            hyp("c", {"C": 1.0}, prior=0.2)]
    first = [p.hypothesis_key for p in optimize_slots(hyps, scen).placements]
    again = [p.hypothesis_key for p in optimize_slots(hyps, scen).placements]
    reversed_input = [p.hypothesis_key
                      for p in optimize_slots(list(reversed(hyps)), scen).placements]
    assert first == again == reversed_input


def test_max_rows_is_respected():
    scen = [Scenario(f"s{i}", 0.01) for i in range(200)]
    hyps = [hyp(f"h{i:03d}", {f"s{i}": 1.0}, prior=1.0 - i * 1e-4) for i in range(200)]
    assert len(optimize_slots(hyps, scen, max_rows=100).rows) == 100


# --------------------------------------------------------------------------- #
# Scenario mass must not be inflated
# --------------------------------------------------------------------------- #
def test_scenario_mass_below_one_is_left_as_residual():
    """Under-confident input keeps its residual: GT may not be in the list."""
    scen = normalise_scenarios([("a", 0.3), ("b", 0.2)])
    assert abs(sum(s.weight for s in scen) - 0.5) < 1e-12


def test_scenario_mass_above_one_is_rescaled():
    scen = normalise_scenarios([("a", 0.9), ("b", 0.8), ("c", 0.7)])
    assert abs(sum(s.weight for s in scen) - 1.0) < 1e-12


# --------------------------------------------------------------------------- #
# KIS
# --------------------------------------------------------------------------- #
def test_kis_treats_nearby_frames_as_one_hypothesis():
    """Frames 10 apart are inside one accepted window -> one scoring row."""
    res = kis_portfolio([cand("L", 100, 0.9), cand("L", 110, 0.8),
                         cand("M", 500, 0.7)], tol_frames=25)
    scoring = [p for p in res.placements if not p.free_recall]
    assert len(scoring) == 2
    assert res.rows[0] == {"video_id": "L", "frame_idx": 100}
    assert res.rows[1]["video_id"] == "M"


def test_kis_distant_frames_in_one_video_are_separate_hypotheses():
    res = kis_portfolio([cand("L", 100, 0.9), cand("L", 900, 0.8)], tol_frames=25)
    scoring = [p for p in res.placements if not p.free_recall]
    assert len(scoring) == 2
    assert {r["frame_idx"] for r in res.rows[:2]} == {100, 900}


def test_kis_offset_hedge_never_outranks_offset_zero():
    """The frame-convention hedge is self-limiting: offset 0 keeps slot 1."""
    res = kis_portfolio([cand("L", 100, 0.9)], tol_frames=2,
                        frame_offsets=(0, -1, 1, -2, 2))
    assert res.rows[0]["frame_idx"] == 100
    # A hedge inside the accepted window adds nothing and must not score.
    assert sum(1 for p in res.placements if not p.free_recall) == 1


# --------------------------------------------------------------------------- #
# QA -- answer diversification
# --------------------------------------------------------------------------- #
def test_qa_spends_slots_on_distinct_answers():
    res = qa_portfolio([cand("L", 100, 0.9)], [("5", 0.6), ("4", 0.4)])
    assert [r["answer"] for r in res.rows[:2]] == ["5", "4"]
    assert all(r["video_id"] == "L" and r["frame_idx"] == 100 for r in res.rows[:2])
    assert sum(1 for p in res.placements if not p.free_recall) == 2


def test_qa_answer_spellings_collapse_to_one_hypothesis():
    """Normalisation is shared with the offline scorer, so ' 5 ' is not new."""
    res = qa_portfolio([cand("L", 100, 0.9)], [("5", 0.5), (" 5 ", 0.5)])
    assert sum(1 for p in res.placements if not p.free_recall) == 1


# --------------------------------------------------------------------------- #
# TRAKE -- fractional payoff, many alignments of one video
# --------------------------------------------------------------------------- #
def align(vid, frames, score):
    return AlignResult(video_id=vid, frames=list(frames), steps=[],
                       score=score, n_miss=0, confidence=score)


def test_trake_second_alignment_of_same_video_earns_a_slot():
    """Right video + partial credit means extra alignments still pay."""
    res = trake_portfolio([align("L", [100, 200], 0.9),
                           align("L", [100, 260], 0.6)], tol_frames=5)
    scoring = [p for p in res.placements if not p.free_recall]
    assert len(scoring) == 2
    assert res.rows[0] == {"video_id": "L", "frames": [100, 200]}
    assert res.rows[1] == {"video_id": "L", "frames": [100, 260]}
    # (5*0.8 + 4*0.2)/5 -- see the partial-credit arithmetic in the module doc.
    assert abs(res.expected_final - 0.96) < 1e-12


def test_trake_jitter_inside_the_window_is_not_a_new_hypothesis():
    res = trake_portfolio([align("L", [100, 200], 0.9)], tol_frames=5,
                          jitters=(0, 3))
    assert sum(1 for p in res.placements if not p.free_recall) == 1


def test_trake_jitter_beyond_every_window_is_dropped():
    """A shift that can satisfy no scenario is not worth building a row for."""
    res = trake_portfolio([align("L", [100, 200], 0.9)], tol_frames=5,
                          jitters=(0, 500))
    assert all(r["frames"] == [100, 200] for r in res.rows)


def test_trake_wrong_video_scenarios_stay_separate():
    res = trake_portfolio([align("L", [100, 200], 0.9),
                           align("M", [100, 200], 0.5)], tol_frames=5)
    assert [r["video_id"] for r in res.rows[:2]] == ["L", "M"]
    assert sum(1 for p in res.placements if not p.free_recall) == 2
