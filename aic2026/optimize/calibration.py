"""Score -> probability calibration, plus the calibration diagnostics.

WHY this module exists
----------------------
`submission_optimizer` allocates the 100 answer slots by *expected marginal
gain*, and an expected gain is only meaningful if the per-hypothesis numbers are
probabilities.  A retriever score is monotone in quality but is NOT a
probability: RRF scores live in ~[0, 0.05], cosine in ~[0.1, 0.4], a DP path
score in ~[0, 1] with a different shape again.  Feeding raw scores to the
optimizer keeps the *ordering* right but makes every expected-score estimate
wrong, and breaks the one comparison that actually matters:

    "is this hypothesis worth a slot at all, versus a hypothesis from another
     channel / another video / another answer string?"

This is guardrail #2 of the QUEST/DANTE design ("never use raw model
confidence; measure ECE, Brier, risk-coverage") made executable.

Everything is pure stdlib:
  * isotonic regression via Pool-Adjacent-Violators (PAV) - non-parametric,
    the safe default once you have >= ~200 labelled (score, correct) pairs;
  * Platt scaling (1-D logistic) - 2 parameters, the safe default when the
    dev set is small, which is where you start.

Fit these on the dev set with a **video-level split** (never let the same video
appear in both the calibration fold and the evaluation fold) - otherwise the
calibrator memorises the videos and the optimizer inherits the leak.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


# --------------------------------------------------------------------------- #
# Calibrators
# --------------------------------------------------------------------------- #
class Calibrator:
    """Maps a raw score to P(hypothesis is correct).

    `dung_hang` nói cho phía GỌI biết `predict()` chờ cái gì:

        False (mặc định)  -> truyền ĐIỂM của ứng viên
        True              -> truyền HẠNG 1-indexed của ứng viên

    Cần cờ này vì `RankCalibrator` cố ý bỏ qua giá trị điểm và chỉ dùng hạng,
    nhưng phía gọi không thể tự đoán ra điều đó. Bug đã trả giá ngày 2026-08-08:
    `kis_portfolio` truyền điểm RRF (luôn ≤ 1/61) vào `RankCalibrator`, mà hàm
    đó có `max(1.0, score)`, nên MỌI ứng viên nhận cùng một prior 0.5 — tầng
    optimize mất sạch thứ hạng của retrieval. Đo được: điểm tụt từ 0.1233 xuống
    0.1033, đúng bằng phần "optimize làm mất 0.0200" từng báo.
    """

    dung_hang = False

    def fit(self, scores: Sequence[float], labels: Sequence[float]) -> "Calibrator":
        raise NotImplementedError

    def predict(self, score: float) -> float:
        raise NotImplementedError

    def predict_many(self, scores: Sequence[float]) -> List[float]:
        return [self.predict(s) for s in scores]

    def prior(self, score: float, hang: int) -> float:
        """Gọi `predict()` với đúng loại đầu vào mà calibrator này chờ.

        Phía gọi dùng hàm NÀY, không gọi `predict()` trực tiếp — như vậy việc
        thêm một calibrator kiểu mới không đòi sửa cả ba builder trong
        `submission_optimizer`.
        """
        return self.predict(float(hang) if self.dung_hang else float(score))


class IdentityCalibrator(Calibrator):
    """Clamp to [0, 1] and call it a probability.

    HONEST LABEL: this preserves ordering but carries no probabilistic meaning.
    It exists so the optimizer runs before you have any labels - the slot
    allocation is still better than a plain sort (it still removes redundant
    hypotheses), but `expected_final` from it is meaningless.  Replace with
    Platt/Isotonic as soon as the dev set is wired.
    """

    calibrated = False

    def fit(self, scores, labels):  # noqa: D102 - nothing to fit
        return self

    def predict(self, score: float) -> float:
        return 0.0 if score < 0.0 else (1.0 if score > 1.0 else float(score))


class RankCalibrator(Calibrator):
    """P from rank alone: p(rank) = p1 / rank**alpha, ignoring score values.

    Useful as a *baseline* calibrator and when scores are incomparable across
    channels but ranks are not (the same reason RRF works).  Still uncalibrated
    in the strict sense unless p1/alpha are fitted.

    ⚠ `predict()` chờ HẠNG, không phải điểm — nên `dung_hang = True`. Truyền
    điểm RRF vào đây thì `max(1.0, ...)` biến mọi ứng viên thành cùng một prior.
    """

    calibrated = False
    dung_hang = True

    def __init__(self, p1: float = 0.5, alpha: float = 0.7):
        self.p1 = p1
        self.alpha = alpha

    def fit(self, scores, labels):
        return self

    def predict(self, score: float) -> float:
        """`score` is interpreted as a 1-indexed rank here."""
        rank = max(1.0, float(score))
        return max(0.0, min(1.0, self.p1 / (rank ** self.alpha)))


class PlattCalibrator(Calibrator):
    """p = sigmoid(a * z + b) where z is the standardised score.

    Standardising the score first makes a single fixed learning rate work across
    wildly different score scales (RRF vs cosine vs DP path score), so the fit
    is deterministic and needs no per-channel tuning.

    Uses Platt's target smoothing -- y+ = (N+ + 1)/(N+ + 2), y- = 1/(N- + 2) --
    which stops the logistic from saturating to exactly 0/1 on a small dev set.
    A saturated calibrator is the worst input to the optimizer: it claims
    certainty it does not have and collapses the portfolio onto one hypothesis.
    """

    calibrated = True

    def __init__(self, iters: int = 3000, lr: float = 0.5):
        self.iters = iters
        self.lr = lr
        self.a = 1.0
        self.b = 0.0
        self.mu = 0.0
        self.sigma = 1.0
        self.n_fit = 0

    def fit(self, scores: Sequence[float], labels: Sequence[float]) -> "PlattCalibrator":
        xs = [float(s) for s in scores]
        ys = [float(y) for y in labels]
        if not xs:
            raise ValueError("PlattCalibrator.fit needs at least one sample")
        if len(xs) != len(ys):
            raise ValueError("scores and labels must have the same length")

        n = len(xs)
        self.n_fit = n
        self.mu = sum(xs) / n
        var = sum((x - self.mu) ** 2 for x in xs) / n
        self.sigma = math.sqrt(var) if var > 1e-12 else 1.0
        zs = [(x - self.mu) / self.sigma for x in xs]

        n_pos = sum(1 for y in ys if y > 0.5)
        n_neg = n - n_pos
        hi = (n_pos + 1.0) / (n_pos + 2.0)
        lo = 1.0 / (n_neg + 2.0)
        ts = [hi if y > 0.5 else lo for y in ys]

        a, b = 1.0, 0.0
        for _ in range(self.iters):
            ga = gb = 0.0
            for z, t in zip(zs, ts):
                p = _sigmoid(a * z + b)
                d = p - t
                ga += d * z
                gb += d
            a -= self.lr * ga / n
            b -= self.lr * gb / n
        self.a, self.b = a, b
        return self

    def predict(self, score: float) -> float:
        z = (float(score) - self.mu) / self.sigma
        return _sigmoid(self.a * z + self.b)


class IsotonicCalibrator(Calibrator):
    """Non-parametric monotone fit via Pool-Adjacent-Violators.

    Makes no shape assumption beyond "higher score is not worse", which is the
    only assumption a retriever score actually justifies.  Needs more labels
    than Platt; below ~200 pairs it will overfit steps into the curve.

    `predict` is a step function over the fitted blocks (not interpolated), so
    a score is always mapped to a mean actually observed in the training fold -
    it can never extrapolate a probability it never saw.
    """

    calibrated = True

    def __init__(self, clip: Tuple[float, float] = (1e-4, 1 - 1e-4)):
        self.clip = clip
        self._lo: List[float] = []      # block lower bounds (ascending)
        self._mean: List[float] = []    # block means (non-decreasing)
        self.n_fit = 0

    def fit(self, scores: Sequence[float], labels: Sequence[float]) -> "IsotonicCalibrator":
        pairs = sorted(zip((float(s) for s in scores), (float(y) for y in labels)))
        if not pairs:
            raise ValueError("IsotonicCalibrator.fit needs at least one sample")
        self.n_fit = len(pairs)

        # blocks: [x_lo, sum_y, n]
        blocks: List[List[float]] = []
        for x, y in pairs:
            blocks.append([x, y, 1.0])
            while len(blocks) >= 2:
                prev, cur = blocks[-2], blocks[-1]
                if prev[1] / prev[2] <= cur[1] / cur[2]:
                    break
                cur = blocks.pop()
                prev[1] += cur[1]
                prev[2] += cur[2]
        lo, hi = self.clip
        self._lo = [b[0] for b in blocks]
        self._mean = [min(hi, max(lo, b[1] / b[2])) for b in blocks]
        return self

    def predict(self, score: float) -> float:
        if not self._mean:
            raise RuntimeError("IsotonicCalibrator used before fit()")
        i = bisect_right(self._lo, float(score)) - 1
        if i < 0:
            i = 0
        return self._mean[i]


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


# --------------------------------------------------------------------------- #
# Diagnostics -- report these next to every Final Score, per guardrail #2
# --------------------------------------------------------------------------- #
@dataclass
class ReliabilityBin:
    lo: float
    hi: float
    n: int
    mean_p: float
    mean_y: float

    @property
    def gap(self) -> float:
        return abs(self.mean_p - self.mean_y)


def reliability_table(probs: Sequence[float], labels: Sequence[float],
                      n_bins: int = 10) -> List[ReliabilityBin]:
    """Equal-width bins over [0, 1]; empty bins are dropped."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    buckets: List[List[Tuple[float, float]]] = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        p = min(1.0, max(0.0, float(p)))
        idx = min(n_bins - 1, int(p * n_bins))
        buckets[idx].append((p, float(y)))
    out: List[ReliabilityBin] = []
    for i, rows in enumerate(buckets):
        if not rows:
            continue
        out.append(ReliabilityBin(
            lo=i / n_bins, hi=(i + 1) / n_bins, n=len(rows),
            mean_p=sum(p for p, _ in rows) / len(rows),
            mean_y=sum(y for _, y in rows) / len(rows),
        ))
    return out


def ece(probs: Sequence[float], labels: Sequence[float], n_bins: int = 10) -> float:
    """Expected Calibration Error: sum_b (n_b/N) * |acc_b - conf_b|."""
    total = len(list(probs)) if not hasattr(probs, "__len__") else len(probs)
    if total == 0:
        return 0.0
    bins = reliability_table(probs, labels, n_bins)
    return sum(b.n / total * b.gap for b in bins)


def brier_score(probs: Sequence[float], labels: Sequence[float]) -> float:
    """Mean squared error of the probability. Lower is better; 0.25 = chance."""
    ps = [float(p) for p in probs]
    if not ps:
        return 0.0
    ys = [float(y) for y in labels]
    return sum((p - y) ** 2 for p, y in zip(ps, ys)) / len(ps)


@dataclass
class RiskCoveragePoint:
    coverage: float
    risk: float
    n: int


def risk_coverage_curve(probs: Sequence[float], labels: Sequence[float]
                        ) -> List[RiskCoveragePoint]:
    """Sort by confidence desc; risk = error rate over the accepted prefix.

    Reads as: "if I only answer the top X% most confident queries, how often am
    I wrong?"  This is the curve that tells you whether an abstain threshold is
    worth having at all -- if risk is flat, confidence carries no information
    and abstaining buys nothing.
    """
    pairs = sorted(zip((float(p) for p in probs), (float(y) for y in labels)),
                   key=lambda t: -t[0])
    n = len(pairs)
    out: List[RiskCoveragePoint] = []
    wrong = 0.0
    for i, (_, y) in enumerate(pairs, start=1):
        wrong += (1.0 - y)
        out.append(RiskCoveragePoint(coverage=i / n, risk=wrong / i, n=i))
    return out


@dataclass
class CalibrationReport:
    n: int
    ece: float
    brier: float
    bins: List[ReliabilityBin] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "ece": self.ece,
            "brier": self.brier,
            "bins": [
                {"lo": b.lo, "hi": b.hi, "n": b.n,
                 "mean_p": b.mean_p, "mean_y": b.mean_y, "gap": b.gap}
                for b in self.bins
            ],
        }


def calibration_report(probs: Sequence[float], labels: Sequence[float],
                       n_bins: int = 10) -> CalibrationReport:
    ps = [float(p) for p in probs]
    return CalibrationReport(
        n=len(ps),
        ece=ece(ps, labels, n_bins),
        brier=brier_score(ps, labels),
        bins=reliability_table(ps, labels, n_bins),
    )
