"""3-channel retrieval (visual / OCR / ASR) + rank fusion.

Default fusion is REAL Reciprocal Rank Fusion. This fixes a prior bug where the
"_rrf_fallback_score" was actually an arithmetic mean of modality scores (label
said RRF, code did mean). Borda is offered as an alternative; weighted-sum is
kept ONLY as a manual debug knob (Goldenball reached 73pts with weighted-sum, so
we do not claim RRF is strictly better until ablation on real data says so).

Soft filtering is the default: OCR/ASR contribute additive rank evidence. Hard
filtering (drop non-matching events) is opt-in for explicit queries only.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .models import Candidate, Event
from .providers import VisualEncoder
from .stores import InMemoryTextStore, InMemoryVectorStore

CHANNELS = ("visual", "ocr", "asr")


# --------------------------------------------------------------------------- #
# Fusion primitives (operate on ranked lists of keys)
# --------------------------------------------------------------------------- #
def rrf_fuse(ranked_lists: Dict[str, List[str]], k: int = 60) -> Dict[str, float]:
    """Reciprocal Rank Fusion. rank is 1-indexed; score = sum 1/(k+rank)."""
    fused: Dict[str, float] = {}
    for keys in ranked_lists.values():
        for rank, key in enumerate(keys, start=1):
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
    return fused


def borda_fuse(ranked_lists: Dict[str, List[str]]) -> Dict[str, float]:
    """Borda count: a list of length n gives (n-rank+1) points to each item."""
    fused: Dict[str, float] = {}
    for keys in ranked_lists.values():
        n = len(keys)
        for rank, key in enumerate(keys, start=1):
            fused[key] = fused.get(key, 0.0) + (n - rank + 1)
    return fused


def weighted_sum_fuse(scored_lists: Dict[str, List[Tuple[str, float]]],
                      weights: Dict[str, float]) -> Dict[str, float]:
    """DEBUG ONLY. Min-max normalise each channel, then weighted sum."""
    fused: Dict[str, float] = {}
    for ch, scored in scored_lists.items():
        if not scored:
            continue
        vals = [s for _, s in scored]
        lo, hi = min(vals), max(vals)
        rng = (hi - lo) or 1.0
        w = weights.get(ch, 1.0)
        for key, s in scored:
            fused[key] = fused.get(key, 0.0) + w * (s - lo) / rng
    return fused


# --------------------------------------------------------------------------- #
# Retriever
# --------------------------------------------------------------------------- #
class MultiChannelRetriever:
    def __init__(self, encoder: VisualEncoder, vector_store=None):
        self.encoder = encoder
        # Mặc định là bản stdlib để test chạy không cần numpy. Với dữ liệu thật
        # hãy truyền `NumpyVectorStore()` vào — xem `core/dataset.py`.
        self.visual = vector_store if vector_store is not None else InMemoryVectorStore()
        self.ocr = InMemoryTextStore()
        self.asr = InMemoryTextStore()
        self.events: Dict[str, Event] = {}

    def reset(self) -> None:
        """Xoá sạch cả ba kênh + bảng event.

        Cặp đôi với `index()`, vì `index()` CỘNG DỒN. Muốn thay chỉ mục bằng một
        tập khác thì phải `reset()` trước — xem docstring `VectorStore.reset`
        để biết chuyện gì xảy ra khi quên (lớp ghim, 20/08).
        """
        self.events = {}
        self.visual.reset()
        self.ocr.reset()
        self.asr.reset()

    def index(self, events: List[Event], embeddings=None) -> None:
        """Nạp event vào chỉ mục. CỘNG DỒN — gọi `reset()` nếu muốn thay thế.

        `embeddings`: ma trận (N, D) CÙNG THỨ TỰ với `events`. Đường nạp hàng
        loạt cho dữ liệu thật — 177k vector mà gắn vào từng Event dưới dạng
        list Python sẽ ngốn vài GB. Bỏ trống thì giữ nguyên hành vi cũ: lấy
        `ev.embedding`, không có thì encode `ev.visual_text`.
        """
        if embeddings is not None and len(events) != len(embeddings):
            raise ValueError(
                f"lệch số lượng: {len(events)} event, {len(embeddings)} embedding")

        for ev in events:
            self.events[ev.event_id] = ev
            self.ocr.add(ev.event_id, ev.ocr_text)
            self.asr.add(ev.event_id, ev.asr_text)

        if embeddings is not None:
            self.visual.add_many([ev.event_id for ev in events], embeddings)
        else:
            for ev in events:
                vec = ev.embedding or self.encoder.encode_text(ev.visual_text)
                self.visual.add(ev.event_id, vec)

    def _channel_scores(self, query: str, top_k: int, query_vector=None
                        ) -> Dict[str, List[Tuple[str, float]]]:
        qvec = (query_vector if query_vector is not None
                else self.encoder.encode_text(query))
        return {
            "visual": self.visual.search(qvec, top_k),
            "ocr": self.ocr.search(query, top_k),
            "asr": self.asr.search(query, top_k),
        }

    def retrieve(self, query: str, top_k: int = 100, fusion: str = "rrf",
                 rrf_k: int = 60, weights: Dict[str, float] | None = None,
                 query_vector=None
                 ) -> List[Candidate]:
        scored = self._channel_scores(query, top_k, query_vector=query_vector)
        return self._fuse_scored(
            scored, top_k, fusion, rrf_k, weights)

    def retrieve_channels(self, visual_query: str, *,
                          ocr_query: str = "", asr_query: str = "",
                          top_k: int = 100, fusion: str = "rrf",
                          rrf_k: int = 60,
                          weights: Dict[str, float] | None = None
                          ) -> List[Candidate]:
        """Retrieve only compiler-enabled channels with channel-specific text."""
        scored: Dict[str, List[Tuple[str, float]]] = {}
        if visual_query:
            qvec = self.encoder.encode_text(visual_query)
            scored["visual"] = self.visual.search(qvec, top_k)
        if ocr_query:
            scored["ocr"] = self.ocr.search(ocr_query, top_k)
        if asr_query:
            scored["asr"] = self.asr.search(asr_query, top_k)
        return self._fuse_scored(
            scored, top_k, fusion, rrf_k, weights)

    def _fuse_scored(self, scored, top_k, fusion, rrf_k, weights):
        ranked = {ch: [key for key, _ in lst] for ch, lst in scored.items()}

        if fusion == "rrf":
            fused = rrf_fuse(ranked, k=rrf_k)
        elif fusion == "borda":
            fused = borda_fuse(ranked)
        elif fusion == "weighted_sum":
            fused = weighted_sum_fuse(scored, weights or {c: 1.0 for c in CHANNELS})
        else:
            raise ValueError(f"unknown fusion: {fusion!r}")

        raw = {ch: dict(lst) for ch, lst in scored.items()}
        cands = []
        for key, fscore in fused.items():
            ev = self.events[key]
            cands.append(Candidate(
                event=ev,
                score=fscore,
                modality_scores={
                    ch: raw.get(ch, {}).get(key, 0.0) for ch in CHANNELS},
            ))
        cands.sort(key=lambda c: c.score, reverse=True)
        return cands[:top_k]
