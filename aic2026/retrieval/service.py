"""Task-aware retrieval policy and authoritative per-video timelines."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from aic2026.control.query_compiler import CompiledQuery
from aic2026.core.models import Candidate, Event
from aic2026.core.retrieval import MultiChannelRetriever


@dataclass(frozen=True)
class RetrievalPolicy:
    pool_per_query: int = 300
    candidate_videos: int = 20
    output_top_k: int = 100
    # Hợp nhất nhiều ENCODER (việc 4.4a). Mỗi retriever phụ là một cặp
    # (encoder, kho vector) độc lập; hợp nhất bằng RRF CÓ TRỌNG SỐ trên HẠNG.
    fuse_pool: int = 500
    fuse_rrf_k: float = 60.0
    fuse_trong_so_chinh: float = 2.0


class RetrievalService:
    """Tầng chia sẻ ứng viên + timeline, và là chỗ HỢP NHẤT nhiều encoder.

    Vì sao hợp nhất ở đây chứ không trong `MultiChannelRetriever`: mỗi encoder có
    số chiều riêng nên không dùng chung một kho vector được, và `search()` chỉ
    nhận MỘT vector truy vấn. Ở tầng này ta còn nguyên *câu chữ*, nên mỗi
    (encoder, kho) tự encode rồi trả danh sách hạng của nó — thứ duy nhất so sánh
    được giữa hai không gian khác nhau là HẠNG, không phải cosine.

    Đo trên dev set 120 câu KIS (2026-08-16), dựng dòng y hệt đường thi:

        so400m một mình              0.3400   R@1  9,2%
        so400m×2 + base-224 (RRF)    0.3750   R@1 15,8%   Δ +0.0350 ± 0.0127
        kiểm chứng chia theo video:  nửa A +0.0328 ± 0.0178 · nửa B +0.0377 ± 0.0179

    Kho của retriever CHÍNH vẫn là kho duy nhất dùng cho QA zero-shot
    (`lay_vector`) và cho TRAKE, nên bật hợp nhất không đổi hình học của hai
    đường đó — chỉ đổi thứ tự ứng viên KIS/QA.
    """

    def __init__(self, retriever: MultiChannelRetriever,
                 events: Iterable[Event],
                 policy: RetrievalPolicy | None = None,
                 phu: Sequence[Tuple[MultiChannelRetriever, float]] = (),
                 kenh_chu=None, trong_so_kenh_chu: float = 0.6):
        # `kenh_chu`: None · MỘT kho · hoặc DANH SÁCH [(kho, trọng số)].
        #
        # Danh sách vì OCR và ASR là hai loại bằng chứng khác hẳn nhau và xứng
        # đáng hai trọng số khác nhau: OCR là chữ IN TRÊN MÀN HÌNH (chính xác,
        # thường là tên riêng/tiêu đề tin), ASR là LỜI NÓI (giàu hơn nhiều
        # nhưng nhiễu hơn và trải rộng cả khoảng shot). Ép chung một trọng số
        # là bỏ mất phần lớn giá trị của cái mạnh hơn.
        # Nó góp MỘT danh sách hạng nữa vào cùng phép RRF đang dùng cho hợp
        # nhất encoder — không phải một đường truy hồi song song. Nhờ vậy trọng
        # số của nó là một con số đọc được, thay vì bị RRF ngầm cấp 1/(60+rank)
        # ngang với kênh visual (đúng cách phép đo 08/08 hỏng, xem
        # `retrieval/kenh_chu.py`).
        if kenh_chu is None:
            self.kenh_chu = []
        elif isinstance(kenh_chu, (list, tuple)):
            self.kenh_chu = [(k, float(w)) for k, w in kenh_chu if k is not None]
        else:
            self.kenh_chu = [(kenh_chu, float(trong_so_kenh_chu))]
        self.trong_so_kenh_chu = float(trong_so_kenh_chu)
        self.retriever = retriever
        self.policy = policy or RetrievalPolicy()
        self.phu = [(r, float(w)) for r, w in (phu or ()) if w]
        timelines: Dict[str, List[Event]] = defaultdict(list)
        self._su_kien: Dict[str, Event] = {}
        for event in events:
            timelines[event.video_id].append(event)
            self._su_kien[event.event_id] = event
        self._timelines = {key: sorted(rows,
                                      key=lambda x: (x.t_start, x.event_id))
                           for key, rows in timelines.items()}

    def retrieve_text(self, text: str,
                      top_k: int | None = None) -> List[Candidate]:
        return self.retriever.retrieve(
            text, top_k=top_k or self.policy.output_top_k, fusion="rrf")

    def retrieve_compiled(self, query: CompiledQuery,
                          top_k: int | None = None) -> List[Candidate]:
        if query.task == "trake":
            raise ValueError("use event_pools for TRAKE")
        can = top_k or self.policy.output_top_k
        text = query.retrieval_query or query.visual_queries[0]
        ocr = " ".join(query.ocr_queries)
        asr = " ".join(query.asr_queries)
        if not self.phu and not self.kenh_chu:
            return self.retriever.retrieve_channels(
                text, ocr_query=ocr, asr_query=asr, top_k=can, fusion="rrf")

        pool = max(can, self.policy.fuse_pool)
        xep = [(self.retriever.retrieve_channels(
                    text, ocr_query=ocr, asr_query=asr, top_k=pool,
                    fusion="rrf"),
                self.policy.fuse_trong_so_chinh)]
        # Kênh chữ nhận NGUYÊN VĂN TIẾNG VIỆT, không phải bản đã viết lại
        # sang tiếng Anh cho encoder ảnh. Xem `Query.text_goc`.
        xep.extend(self._xep_kenh_chu(
            getattr(query, "text_goc", "") or text, pool))
        for retriever, trong_so in self.phu:
            try:
                xep.append((retriever.retrieve_channels(
                    text, ocr_query=ocr, asr_query=asr, top_k=pool,
                    fusion="rrf"), trong_so))
            except LookupError:
                # Encoder phụ không encode được câu này (thiếu cache, không có
                # model) — bỏ NÓ, không bỏ cả truy vấn. Đường chính vẫn đủ chạy.
                continue
        return self._hop_nhat(xep, can)

    def _xep_kenh_chu(self, text: str, pool: int):
        """Danh sách hạng của kênh chữ, hoặc rỗng. KHÔNG BAO GIỜ ném ra ngoài.

        Kênh chữ là bằng chứng bổ sung; nó hỏng thì bài nộp phải vẫn chạy bằng
        encoder ảnh như trước. Ném lỗi ở đây là biến một tính năng thêm vào
        thành một điểm chết mới ngay giữa buổi thi.
        """
        ra = []
        for kho, w in self.kenh_chu:
            try:
                xep_hang = kho.xep_hang_event(text, pool)
            except Exception:                                 # noqa: BLE001
                continue
            ds = [Candidate(event=self._su_kien[eid], score=float(sc),
                            modality_scores={"text": float(sc)})
                  for eid, sc in xep_hang if eid in self._su_kien]
            if ds:
                ra.append((ds, w))
        return ra

    def _hop_nhat(self, xep_theo_encoder, top_k: int) -> List[Candidate]:
        """RRF có trọng số trên HẠNG, rồi dựng lại Candidate từ event thật."""
        k = float(self.policy.fuse_rrf_k)
        diem: Dict[str, float] = {}
        goc: Dict[str, Candidate] = {}
        for danh_sach, trong_so in xep_theo_encoder:
            for hang, cand in enumerate(danh_sach, start=1):
                eid = cand.event.event_id
                diem[eid] = diem.get(eid, 0.0) + float(trong_so) / (k + hang)
                goc.setdefault(eid, cand)
        thu_tu = sorted(diem.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        ra: List[Candidate] = []
        for eid, score in thu_tu:
            cand = goc.get(eid)
            event = self._su_kien.get(eid) or (cand.event if cand else None)
            if event is None:
                continue
            ra.append(Candidate(
                event=event, score=score,
                modality_scores=dict(cand.modality_scores) if cand else {}))
        return ra

    def event_pools(self, query: CompiledQuery,
                    query_vectors=None) -> List[List[Candidate]]:
        if query.task != "trake":
            raise ValueError("event_pools is only valid for TRAKE")
        if query_vectors is None:
            return [self.retrieve_text(event.text, self.policy.pool_per_query)
                    for event in query.events]
        return [
            self.retriever.retrieve(
                event.text, top_k=self.policy.pool_per_query, fusion="rrf",
                query_vector=vector)
            for event, vector in zip(query.events, query_vectors)
        ]

    def candidate_videos(self,
                         pools: Sequence[Sequence[Candidate]]) -> List[str]:
        # Aggregate ranks, not incompatible raw scores from different queries.
        scores: Dict[str, float] = defaultdict(float)
        for pool in pools:
            seen: set[str] = set()
            for rank, candidate in enumerate(pool, start=1):
                if candidate.video_id not in seen:
                    scores[candidate.video_id] += 1.0 / (60 + rank)
                    seen.add(candidate.video_id)
        return [video_id for video_id, _ in
                sorted(scores.items(), key=lambda item: (-item[1], item[0]))
                [:self.policy.candidate_videos]]

    def full_timeline(self, video_id: str) -> List[Event]:
        """Return all indexed samples, never only the video's global ANN hits."""
        return list(self._timelines.get(video_id, []))
