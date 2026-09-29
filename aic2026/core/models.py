"""Core data models shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class Frame:
    """A sampled frame with its timestamp inside a video."""
    video_id: str
    frame_idx: int
    pts_time: float  # seconds


@dataclass
class Event:
    """A sliding-window 'event' — the retrieval unit (WESp idea).

    An event bundles the evidence from all modalities inside a time window,
    plus a representative keyframe index used when emitting a submission.
    """
    video_id: str
    event_id: str
    t_start: float
    t_end: float
    start_frame: int
    end_frame: int
    keyframe_idx: int
    visual_text: str = ""   # caption / object tags (stands in for visual embedding text)
    ocr_text: str = ""
    asr_text: str = ""
    embedding: Optional[List[float]] = None

    def text_for(self, modality: str) -> str:
        return {
            "visual": self.visual_text,
            "ocr": self.ocr_text,
            "asr": self.asr_text,
        }.get(modality, "")


@dataclass
class Candidate:
    """A retrieved event with fused + per-modality scores."""
    event: Event
    score: float = 0.0
    modality_scores: Dict[str, float] = field(default_factory=dict)

    @property
    def video_id(self) -> str:
        return self.event.video_id

    @property
    def frame_idx(self) -> int:
        return self.event.keyframe_idx

    @property
    def t_start(self) -> float:
        return self.event.t_start


@dataclass
class SubEvent:
    """One step of a TRAKE query (a described moment in the sequence)."""
    text: str
    modality_hint: Optional[str] = None  # 'visual'|'ocr'|'asr' from LLM, may be wrong


@dataclass
class Query:
    qid: str
    task: str                      # 'kis' | 'qa' | 'trake'
    text: str = ""
    # Câu hỏi tiếng Việt tách riêng cho VLM. `text` có thể là bản tiếng Anh
    # dùng để truy hồi; không được ép VLM trả lời tiếng Việt bằng một câu hỏi
    # tiếng Anh khi bản gốc đã có sẵn trong dev set / request của BTC.
    question_vi: str = ""
    # NGUYÊN VĂN tiếng Việt của đề, giữ lại TRƯỚC khi lớp `viet_lai_truy_van`
    # ghi đè `text` bằng chú thích tiếng Anh.
    #
    # Không thừa: hai tầng cần hai bản KHÁC NHAU của cùng một câu hỏi. Encoder
    # ảnh (SigLIP2, học trên alt-text) thưởng lối viết chú thích tiếng Anh —
    # đó là lý do có lớp viết lại (+0.0567 ± 0.0209). Nhưng kênh OCR so chữ với
    # chữ TIẾNG VIỆT đọc từ khung hình, nên đưa bản tiếng Anh cho nó là gần như
    # chắc chắn không khớp gì. Đã quan sát thật 21/08: kênh OCR tìm đúng khung
    # khi hỏi bằng tiếng Việt, và câm hoàn toàn khi hỏi bằng bản đã viết lại.
    text_goc: str = ""
    sub_events: List[SubEvent] = field(default_factory=list)  # TRAKE only

    @property
    def n_events(self) -> int:
        return len(self.sub_events)


@dataclass
class GroundTruth:
    """Ground truth for offline scoring (mirrors official R-Score inputs)."""
    task: str
    video_id: str
    # KIS/QA: correct frame range [s, e]; QA also has answer.
    frame_start: int = 0
    frame_end: int = 0
    answer: str = ""
    # QA: cách diễn đạt KHÁC cũng được tính đúng. Cần vì `answers_match` so chuỗi
    # chính xác, còn BTC chấm "semantic exact match" — rộng hơn. Cụ thể: đáp án
    # chuẩn của ta là TIẾNG VIỆT ("xanh lá") mà một VLM tiếng Anh trả "green" thì
    # về nghĩa là đúng; không có chỗ ghi biến thể thì hai lỗi rất khác nhau —
    # "trả lời sai" và "trả lời đúng nhưng khác ngôn ngữ" — bị gộp thành cùng 0.0.
    answer_alt: List[str] = field(default_factory=list)
    # TRAKE: per-event windows [(s_j, e_j), ...] in the same order as the query.
    event_windows: List[Sequence[int]] = field(default_factory=list)
