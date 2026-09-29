import os

from aic2026.core.offline_eval import evaluate, load_benchmark
from aic2026.core.pipeline import Pipeline
from aic2026.core.providers import HashEmbedder, RuleBasedProvider

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "benchmark_queries.sample.json")


def _run():
    events, queries = load_benchmark(DATA)
    pipe = Pipeline(HashEmbedder(), RuleBasedProvider())
    return evaluate(pipe, events, queries)


def test_end_to_end_runs_and_scores():
    report = _run()
    assert "OVERALL" in report["summary"]
    assert report["summary"]["OVERALL"] > 0.0
    # every query should surface its correct answer somewhere in top-100
    by_qid = {d["qid"]: d for d in report["detail"]}
    assert by_qid["q1-kis"]["recall@100"] == 1.0
    assert by_qid["q2-kis"]["recall@100"] == 1.0


def test_trake_full_chain_scored():
    report = _run()
    by_qid = {d["qid"]: d for d in report["detail"]}
    # TRAKE uses full-chain partial credit; correct chain should score > 0
    assert by_qid["q4-trake"]["final_score"] > 0.0
