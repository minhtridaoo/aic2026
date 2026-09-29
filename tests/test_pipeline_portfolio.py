"""Portfolio mode must stay wired end-to-end AND keep every submission invariant.

The optimizer only earns its complexity if it can be A/B'd against the old
heuristic on the same retriever, and if the rows it emits are still legal. Both
are asserted here.
"""
import os

from aic2026.optimize.calibration import RankCalibrator
from aic2026.core.models import Event, Query, SubEvent
from aic2026.core.pipeline import Pipeline, PortfolioConfig
from aic2026.core.providers import HashEmbedder, RuleBasedProvider
from aic2026.core import submission as sub


def make_events():
    events = []
    for v, vid in enumerate(("L01_V001", "L02_V001", "L03_V001")):
        for i in range(6):
            t = i * 8.0
            events.append(Event(
                video_id=vid, event_id=f"{vid}#e{i}",
                t_start=t, t_end=t + 10.0,
                start_frame=int(t * 25), end_frame=int((t + 10) * 25),
                keyframe_idx=int(t * 25) + 40 * v + 3,
                visual_text=f"nguoi dan ong ao do phat bieu canh {i} kenh {v}",
                ocr_text="YAMAHA" if i % 3 == 0 else "",
                asr_text="xin chao quy vi" if i % 2 == 0 else "",
            ))
    return events


class MultiAnswerProvider(RuleBasedProvider):
    """Stands in for a VLM that reports an answer distribution, not one string."""

    def answer_distribution(self, question, context, n=3):
        return [("5", 0.55), ("4", 0.30), ("6", 0.15)][:n]


def build(portfolio: PortfolioConfig, llm=None) -> Pipeline:
    p = Pipeline(HashEmbedder(dim=128), llm or RuleBasedProvider(),
                 portfolio=portfolio)
    p.index(make_events())
    return p


# --------------------------------------------------------------------------- #
# Both strategies must run and stay comparable
# --------------------------------------------------------------------------- #
def test_portfolio_off_is_the_unchanged_heuristic_path():
    p = build(PortfolioConfig(enabled=False))
    rows = p.answer(Query(qid="q1", task="kis", text="nguoi dan ong ao do"))
    assert rows
    assert all({"video_id", "frame_idx"} <= set(r) for r in rows)
    assert p.last_trace.expected_final is None      # no portfolio diagnostics


def test_portfolio_on_reports_diagnostics():
    p = build(PortfolioConfig(enabled=True, calibrator=RankCalibrator()))
    rows = p.answer(Query(qid="q1", task="kis", text="nguoi dan ong ao do"))
    assert rows
    t = p.last_trace
    assert t.expected_final is not None and t.expected_final >= 0.0
    assert t.n_scenarios > 0
    assert t.n_scoring_rows is not None and t.n_scoring_rows >= 1


def test_both_strategies_produce_rows_for_the_same_query():
    q = Query(qid="q1", task="kis", text="nguoi dan ong ao do phat bieu")
    off = build(PortfolioConfig(enabled=False)).answer(q)
    on = build(PortfolioConfig(enabled=True, calibrator=RankCalibrator())).answer(q)
    assert off and on
    assert len(off) <= sub.MAX_ROWS and len(on) <= sub.MAX_ROWS


# --------------------------------------------------------------------------- #
# Submission invariants survive the new row builder
# --------------------------------------------------------------------------- #
def test_kis_portfolio_rows_pass_the_csv_validator(tmp_path):
    p = build(PortfolioConfig(enabled=True, calibrator=RankCalibrator(),
                              frame_offsets=(0, -1, 1)))
    rows = p.answer(Query(qid="q1", task="kis", text="nguoi dan ong ao do"))
    path = os.path.join(str(tmp_path), "query-1-kis.csv")
    sub.write_csv(path, "kis", rows)
    ok, errors = sub.validate_csv(path, "kis")
    assert ok, errors


def test_trake_portfolio_keeps_exactly_n_frames_per_row(tmp_path):
    p = build(PortfolioConfig(enabled=True, calibrator=RankCalibrator(),
                              trake_jitters=(0, 25, -25),
                              trake_paths_per_video=4))
    q = Query(qid="q4", task="trake", text="",
              sub_events=[SubEvent("nguoi dan ong ao do"),
                          SubEvent("canh 3"),
                          SubEvent("xin chao quy vi")])
    rows = p.answer(q)
    assert rows
    assert all(len(r["frames"]) == 3 for r in rows)
    path = os.path.join(str(tmp_path), "query-4-trake.csv")
    sub.write_csv(path, "trake", rows)
    ok, errors = sub.validate_csv(path, "trake", n_events=3)
    assert ok, errors


def test_trake_frames_are_non_negative_after_negative_jitter():
    """Jitter must never push a frame id below 0 -- that row would be invalid."""
    p = build(PortfolioConfig(enabled=True, calibrator=RankCalibrator(),
                              trake_jitters=(0, -100000)))
    q = Query(qid="q4", task="trake", text="",
              sub_events=[SubEvent("canh 0"), SubEvent("canh 3")])
    rows = p.answer(q)
    assert all(f >= 0 for r in rows for f in r["frames"])


# --------------------------------------------------------------------------- #
# QA answer diversification actually reaches the submission
# --------------------------------------------------------------------------- #
def test_qa_portfolio_emits_more_than_one_distinct_answer():
    p = build(PortfolioConfig(enabled=True, calibrator=RankCalibrator(), n_answers=3),
              llm=MultiAnswerProvider())
    rows = p.answer(Query(qid="q3", task="qa", text="co bao nhieu nguoi len san khau"))
    assert rows
    assert len({r["answer"] for r in rows}) >= 2
    assert rows[0]["answer"] == "5"                 # highest-probability answer first
    assert p.last_trace.answers[0] == ("5", 0.55)


def test_qa_default_provider_still_emits_a_single_answer():
    """The default `answer_distribution` degrades to one answer, not a crash."""
    p = build(PortfolioConfig(enabled=True, calibrator=RankCalibrator()))
    rows = p.answer(Query(qid="q3", task="qa", text="mau gi"))
    assert rows
    assert len({r["answer"] for r in rows}) == 1
