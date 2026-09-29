"""TRAKE coarse-to-fine solver using full candidate-video timelines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from aic2026.control.query_compiler import CompiledQuery
from aic2026.core.models import Candidate
from aic2026.core.temporal_dp import AlignResult, DPParams, align_k_best
from aic2026.retrieval.service import RetrievalService


@dataclass(frozen=True)
class TrakeConfig:
    # Full timeline is scored first; only the strongest local hypotheses enter
    # the O(K^2) k-best DP. 64 is an engineering default, not a tuned value.
    timeline_top_per_event: int = 64
    paths_per_video: int = 8
    output_top: int = 100


class FullTimelineTrakeSolver:
    def __init__(self, service: RetrievalService,
                 config: TrakeConfig | None = None,
                 dp_params: DPParams | None = None):
        self.service = service
        self.config = config or TrakeConfig()
        self.dp_params = dp_params or DPParams()
        self.last_stats = {}

    def solve(self, query: CompiledQuery) -> List[AlignResult]:
        if query.task != "trake" or not query.events:
            raise ValueError("compiled TRAKE query with events required")
        encoder = self.service.retriever.encoder
        store = self.service.retriever.visual
        per_event: List[List[Candidate]] = [[] for _ in query.events]
        query_vectors = [
            encoder.encode_text(event_query.text)
            for event_query in query.events
        ]
        coarse = self.service.event_pools(query, query_vectors)
        videos = self.service.candidate_videos(coarse)
        timeline_samples = 0

        # Critical invariant: rescore every event against every indexed sample
        # of each shortlisted video before temporal alignment.
        for video_id in videos:
            timeline = self.service.full_timeline(video_id)
            timeline_samples += len(timeline)
            keys = [event.event_id for event in timeline]
            for index, qvec in enumerate(query_vectors):
                scores = store.score_many(keys, qvec)
                local: List[Candidate] = []
                for event in timeline:
                    if event.event_id not in scores:
                        continue
                    score = scores[event.event_id]
                    local.append(Candidate(
                        event=event, score=score,
                        modality_scores={"visual": score}))
                local.sort(key=lambda x: (-x.score, x.t_start,
                                          x.event.event_id))
                per_event[index].extend(
                    local[:self.config.timeline_top_per_event])

        results = align_k_best(
            per_event, hints=[x.modality_hint for x in query.events],
            params=self.dp_params, per_video=self.config.paths_per_video,
            top=self.config.output_top)
        self.last_stats = {
            "coarse_candidates": sum(len(pool) for pool in coarse),
            "candidate_videos": len(videos),
            "timeline_samples": timeline_samples,
            "query_encodes": len(query.events),
            "alignments": len(results),
        }
        return results
