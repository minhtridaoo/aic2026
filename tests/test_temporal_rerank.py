import pytest

from aic2026.core.models import Candidate, Event
from aic2026.retrieval.temporal_rerank import (
    TemporalRerankConfig,
    rerank_temporal,
    temporal_scores,
)


def _candidate(video, number, time):
    event_id = f"{video}#{number:05d}"
    return Candidate(Event(
        video_id=video, event_id=event_id,
        t_start=float(time), t_end=float(time) + 1.0,
        start_frame=number * 25, end_frame=number * 25 + 24,
        keyframe_idx=number,
    ))


def test_cluster_sau_co_the_vuot_hit_don_le():
    candidates = [
        _candidate("solo", 1, 0),
        _candidate("noise", 1, 0),
        _candidate("target", 1, 10),
        _candidate("target", 2, 11),
        _candidate("target", 3, 12),
    ]
    reranked = rerank_temporal(
        candidates,
        TemporalRerankConfig(window_seconds=3, alpha=2, max_neighbours=2),
    )
    assert reranked[0].video_id == "target"


def test_khong_co_lan_can_thi_giu_nguyen_thu_tu():
    candidates = [_candidate(f"v{i}", 1, 0) for i in range(5)]
    assert rerank_temporal(candidates) == candidates


def test_alpha_0_giu_nguyen_thu_tu_du_co_cluster():
    candidates = [
        _candidate("solo", 1, 0),
        _candidate("target", 1, 10),
        _candidate("target", 2, 11),
    ]
    cfg = TemporalRerankConfig(alpha=0)
    assert rerank_temporal(candidates, cfg) == candidates


def test_neighbour_ngoai_cua_so_khong_duoc_tinh():
    candidates = [_candidate("v", 1, 0), _candidate("v", 2, 20)]
    scores = temporal_scores(
        candidates, TemporalRerankConfig(window_seconds=5))
    assert all(support == 0 for _, support in scores.values())


@pytest.mark.parametrize("kwargs", [
    {"window_seconds": 0}, {"alpha": -1}, {"rank_k": -1},
    {"max_neighbours": 0},
])
def test_tu_choi_tham_so_vo_nghia(kwargs):
    with pytest.raises(ValueError):
        TemporalRerankConfig(**kwargs)
