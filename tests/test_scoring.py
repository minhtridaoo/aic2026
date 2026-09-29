from aic2026.core import scoring
from aic2026.core.models import GroundTruth


def kis(v, f):
    return {"video_id": v, "frame_idx": f}


def test_kis_binary():
    gt = GroundTruth(task="kis", video_id="L01", frame_start=100, frame_end=200)
    assert scoring.r_score_kis(kis("L01", 150), gt) == 1.0
    assert scoring.r_score_kis(kis("L01", 99), gt) == 0.0
    assert scoring.r_score_kis(kis("L02", 150), gt) == 0.0


def test_qa_requires_answer_match():
    gt = GroundTruth(task="qa", video_id="L01", frame_start=0, frame_end=100,
                     answer="Maersk")
    good = {"video_id": "L01", "frame_idx": 50, "answer": "maersk"}
    bad = {"video_id": "L01", "frame_idx": 50, "answer": "Cosco"}
    assert scoring.r_score_qa(good, gt) == 1.0   # case-insensitive match
    assert scoring.r_score_qa(bad, gt) == 0.0


def test_trake_full_chain_partial_credit():
    gt = GroundTruth(task="trake", video_id="L02",
                     event_windows=[[0, 90], [90, 180], [180, 270]])
    all_ok = {"video_id": "L02", "frames": [40, 130, 220]}
    two_of_three = {"video_id": "L02", "frames": [40, 130, 999]}
    wrong_video = {"video_id": "L09", "frames": [40, 130, 220]}
    assert scoring.r_score_trake(all_ok, gt) == 1.0
    assert abs(scoring.r_score_trake(two_of_three, gt) - 2 / 3) < 1e-9
    assert scoring.r_score_trake(wrong_video, gt) == 0.0


def test_final_score_top_k_positions():
    gt = GroundTruth(task="kis", video_id="L01", frame_start=100, frame_end=200)
    # correct answer only at rank 7 -> R@1=R@5=0, R@20=R@50=R@100=1
    rows = [kis("LX", 0)] * 6 + [kis("L01", 150)] + [kis("LX", 0)] * 50
    assert scoring.top_k_r_scores(rows, gt) == {1: 0.0, 5: 0.0, 20: 1.0, 50: 1.0, 100: 1.0}
    assert abs(scoring.final_score(rows, gt) - 0.6) < 1e-9


def test_final_score_rank1_is_perfect():
    gt = GroundTruth(task="kis", video_id="L01", frame_start=100, frame_end=200)
    rows = [kis("L01", 150)] + [kis("LX", 0)] * 99
    assert scoring.final_score(rows, gt) == 1.0
