"""Calibration must be monotone, non-saturating, and honestly measured."""
import math

from aic2026.optimize.submission_optimizer import kis_portfolio
from aic2026.optimize.calibration import (IdentityCalibrator, IsotonicCalibrator,
                                 PlattCalibrator, RankCalibrator,
                                 brier_score, calibration_report, ece,
                                 reliability_table, risk_coverage_curve)


def _separable(n=200):
    """Scores where high => correct, with a soft overlap in the middle."""
    scores, labels = [], []
    for i in range(n):
        s = i / (n - 1)
        scores.append(s)
        labels.append(1.0 if s > 0.5 else 0.0)
    return scores, labels


# --------------------------------------------------------------------------- #
# Identity / rank stand-ins
# --------------------------------------------------------------------------- #
def test_identity_clamps_and_declares_itself_uncalibrated():
    c = IdentityCalibrator()
    assert c.predict(-1.0) == 0.0
    assert c.predict(2.0) == 1.0
    assert c.predict(0.3) == 0.3
    assert c.calibrated is False


# --------------------------------------------------------------------------- #
# REGRESSION — sai khớp điểm/hạng, 2026-08-08
# --------------------------------------------------------------------------- #
class _Cand:
    """Ứng viên tối giản: portfolio chỉ cần video_id / frame_idx / score."""

    def __init__(self, video_id, frame_idx, score):
        self.video_id = video_id
        self.frame_idx = frame_idx
        self.score = score


def test_rank_calibrator_khai_bao_can_hang():
    assert RankCalibrator().dung_hang is True
    assert IdentityCalibrator().dung_hang is False


def test_prior_dua_dung_loai_dau_vao():
    """`prior()` phải đưa HẠNG cho RankCalibrator, ĐIỂM cho Identity."""
    rc, ic = RankCalibrator(), IdentityCalibrator()
    # điểm 0.01 / hạng 4 -> Rank phải phản ứng theo hạng
    assert rc.prior(0.01, 4) == rc.predict(4)
    assert rc.prior(0.01, 4) != rc.predict(0.01)
    assert ic.prior(0.3, 7) == ic.predict(0.3)


def test_portfolio_khong_lam_bet_prior_khi_diem_be_hon_1():
    """Bug cũ: điểm RRF luôn ≤ 1/61, `RankCalibrator.predict` có max(1.0, s),
    nên mọi ứng viên nhận cùng prior 0.5 -> optimizer mất sạch thứ hạng.

    Kiểm bằng hành vi quan sát được: ứng viên hạng 1 phải được xếp trước ứng
    viên hạng cuối, dù CHÊNH LỆCH ĐIỂM nhỏ tới mức nào. Các frame cách nhau xa
    hơn tol nên chúng là những giả thuyết riêng biệt.
    """
    cands = [_Cand("L01", 100 + 1000 * i, 0.0164 - 0.0001 * i) for i in range(6)]
    rows = kis_portfolio(cands, RankCalibrator(), tol_frames=25).rows
    assert rows, "portfolio không trả về dòng nào"
    assert rows[0]["frame_idx"] == 100, (
        "ứng viên hạng 1 phải chiếm slot 1; nếu mọi prior bằng nhau thì thứ tự "
        "trở thành tuỳ ý — đó chính là bug cũ")


def test_rank_calibrator_is_decreasing_in_rank():
    c = RankCalibrator(p1=0.5, alpha=0.7)
    ps = [c.predict(r) for r in (1, 2, 5, 20, 100)]
    assert ps == sorted(ps, reverse=True)
    assert all(0.0 <= p <= 1.0 for p in ps)


# --------------------------------------------------------------------------- #
# Platt
# --------------------------------------------------------------------------- #
def test_platt_is_monotone_increasing():
    scores, labels = _separable()
    c = PlattCalibrator().fit(scores, labels)
    ps = [c.predict(s) for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert ps == sorted(ps)
    assert all(0.0 < p < 1.0 for p in ps)


def test_platt_separates_the_two_classes():
    scores, labels = _separable()
    c = PlattCalibrator().fit(scores, labels)
    assert c.predict(0.9) > 0.5 > c.predict(0.1)


def test_platt_target_smoothing_prevents_saturation():
    """A saturated calibrator would collapse the whole portfolio onto one row."""
    c = PlattCalibrator().fit([0.0, 1.0], [0.0, 1.0])
    assert 0.0 < c.predict(-10.0)
    assert c.predict(10.0) < 1.0


def test_platt_handles_a_constant_score_column():
    """Zero variance must not divide by zero."""
    c = PlattCalibrator().fit([0.5] * 10, [1.0] * 5 + [0.0] * 5)
    p = c.predict(0.5)
    assert 0.0 < p < 1.0 and math.isfinite(p)


# --------------------------------------------------------------------------- #
# Isotonic
# --------------------------------------------------------------------------- #
def test_isotonic_is_non_decreasing():
    scores, labels = _separable()
    c = IsotonicCalibrator().fit(scores, labels)
    ps = [c.predict(s / 20) for s in range(21)]
    assert ps == sorted(ps)


def test_isotonic_pools_a_violating_pair():
    """A high score labelled wrong must be pooled, not fitted exactly."""
    c = IsotonicCalibrator().fit([0.1, 0.2, 0.3, 0.4], [0.0, 1.0, 0.0, 1.0])
    assert c.predict(0.2) == c.predict(0.3)     # the violation is pooled to 0.5
    assert abs(c.predict(0.2) - 0.5) < 1e-9


def test_isotonic_never_extrapolates_below_the_training_range():
    c = IsotonicCalibrator().fit([0.4, 0.6], [0.0, 1.0])
    assert c.predict(-5.0) == c.predict(0.4)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def test_ece_is_zero_for_a_perfectly_calibrated_input():
    probs = [0.05] * 100 + [0.95] * 100
    labels = [0.0] * 95 + [1.0] * 5 + [0.0] * 5 + [1.0] * 95
    assert ece(probs, labels, n_bins=10) < 1e-9


def test_ece_is_large_for_a_confidently_wrong_input():
    assert ece([0.95] * 100, [0.0] * 100, n_bins=10) > 0.9


def test_brier_matches_the_definition():
    assert abs(brier_score([1.0, 0.0], [1.0, 0.0])) < 1e-12
    assert abs(brier_score([0.5, 0.5], [1.0, 0.0]) - 0.25) < 1e-12


def test_reliability_table_drops_empty_bins_and_keeps_all_samples():
    bins = reliability_table([0.05, 0.06, 0.95], [0.0, 0.0, 1.0], n_bins=10)
    assert len(bins) == 2
    assert sum(b.n for b in bins) == 3


def test_risk_coverage_is_monotone_when_confidence_is_informative():
    probs = [0.9, 0.8, 0.7, 0.2, 0.1]
    labels = [1.0, 1.0, 1.0, 0.0, 0.0]
    curve = risk_coverage_curve(probs, labels)
    assert curve[0].risk == 0.0                 # most confident answer is right
    assert curve[-1].coverage == 1.0
    assert curve[-1].risk > curve[0].risk       # risk grows as we accept more


def test_calibration_report_round_trips_to_dict():
    scores, labels = _separable(50)
    c = PlattCalibrator().fit(scores, labels)
    rep = calibration_report(c.predict_many(scores), labels).as_dict()
    assert rep["n"] == 50
    assert 0.0 <= rep["ece"] <= 1.0
    assert 0.0 <= rep["brier"] <= 1.0
    assert rep["bins"]
