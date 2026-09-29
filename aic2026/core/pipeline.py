"""End-to-end query answering that produces submission-ready rows.

Ties ingestion/retrieval/DP/providers together so both the demo and the offline
benchmark share one code path.

Two row-building strategies are available, selected by `PortfolioConfig.enabled`:

  * OFF (default, unchanged behaviour) -- `submission.*_rows` heuristics:
    sort by score, keep a fixed head, round-robin videos in the tail.
  * ON -- `submission_optimizer`: treat the 100 slots as a portfolio and fill
    them by expected marginal gain under the official metric. Requires a
    calibrator to be meaningful; see `calibration`.

The switch is a config flag rather than a rewrite so the two can be A/B'd on the
same dev set with the same retriever -- which is the only way to know whether
the optimizer is actually earning its complexity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Sequence
import warnings

from .models import Event, Query
from .providers import (CLIPTextEncoder, LLMProvider, MCLIPTextEncoder,
                        VisualEncoder)
from .retrieval import MultiChannelRetriever
from . import submission as sub
from .temporal_dp import DPParams
from .dich import (ChonEncoderTheoNgonNgu, DichRoiEncode,
                   OpusMTEnViTranslator, Translator)
from ..control.gates import GateDecision, decide
from ..control.query_compiler import CompiledQuery, compile_query
from ..control.verifier import VerificationResult, verify_required_evidence
from ..reasoning.trake import FullTimelineTrakeSolver, TrakeConfig
from ..retrieval.service import RetrievalPolicy, RetrievalService
# Chỗ duy nhất trong core/ gọi sang tầng optimize. Giữ đúng một điểm nối như vậy
# để tắt/bật portfolio là chuyện của một cờ, không phải sửa nhiều nơi.
from ..optimize.calibration import Calibrator
from ..optimize import submission_optimizer as opt
from ..optimize.don_slot import CauHinhDonSlot, don_slot


@dataclass
class PortfolioConfig:
    """Knobs for metric-aware row building. All defaults are UNVERIFIED.

    `kis_tol_frames` / `trake_tol_frames` decide when two frames are the same
    hypothesis, so they set how the budget is spent. Fit them on the labelled
    dev set (median half-width of the accepted window per task) before trusting
    any number this produces -- and pick a stable plateau, not a sharp peak.
    """
    enabled: bool = False
    calibrator: Optional[Calibrator] = None
    kis_tol_frames: int = opt.KIS_TOL_FRAMES
    trake_tol_frames: int = opt.TRAKE_TOL_FRAMES
    # Hedge against the unresolved frame-index convention (audit: MIXED_
    # CONVENTION, offset 0 wins ~79% of discriminative samples). (0,) = no hedge.
    frame_offsets: Sequence[int] = (0,)
    # Shifted TRAKE alignments. Must exceed trake_tol_frames to be a new
    # hypothesis at all, or the optimizer will (correctly) never spend a slot.
    #
    # ĐÃ FIT 2026-08-08 (`thu_nghiem_trake.py`). Vì sao cần rải:
    #
    #   cửa sổ chấm TRAKE của BTC : dưới 10 khung
    #   khoảng cách keyframe THẬT : trung vị 96 khung (74.773 khoảng, 300 video)
    #   -> nộp đúng chỉ số keyframe chỉ phủ 11.5% khoảng trống
    #
    # Mô phỏng với phân bố khoảng cách thật, N=3 mốc, giả định đã tin đúng
    # video+căn chỉnh (tức TRẦN TRÊN):
    #
    #   (0,) — mặc định cũ          1 dòng    Final 0.2521
    #   bước 10, ±6 nhịp           13 dòng    Final 0.3952   <- chọn
    #   bước 10, ±12 nhịp          25 dòng    Final 0.3958   (bão hoà)
    #   bước 20, ±6 nhịp           13 dòng    Final 0.3458   (bước quá thưa, hở)
    #
    # Chọn ±6 chứ không ±12: thêm 12 dòng nữa chỉ được +0.0006, mà 12 slot đó
    # đáng giá hơn khi dành cho căn chỉnh/video khác. Bước 10 = 2×tol, đủ để mỗi
    # dòng là một giả thuyết mới chứ không trùng dòng trước.
    #
    # Thứ tự 0 trước rồi toả ra hai bên là CÓ CHỦ Ý: Final lấy max trên tiền tố
    # và slot 1 đáng 5× slot 51, nên độ lệch khả dĩ nhất phải đứng đầu.
    trake_jitters: Sequence[int] = (0, 10, -10, 20, -20, 30, -30,
                                    40, -40, 50, -50, 60, -60)
    trake_paths_per_video: int = 8
    n_answers: int = 3
    # P(ground truth is inside the candidate set). None = infer from the
    # calibrated weights and leave the residual unassigned. Set it to a measured
    # recall@100 from the dev set to stop expected_final being optimistic.
    mass: Optional[float] = None


@dataclass
class AnswerTrace:
    """Diagnostics for one query -- kept separate from the submittable rows."""
    task: str
    n_candidates: int = 0
    expected_final: Optional[float] = None
    n_scoring_rows: Optional[int] = None
    n_scenarios: Optional[int] = None
    covered_mass: Optional[float] = None
    answers: List = field(default_factory=list)
    compiled_query: Optional[CompiledQuery] = None
    gate: Optional[GateDecision] = None
    verification: Optional[VerificationResult] = None
    escalation_available: bool = False
    vlm_error: Optional[str] = None


class Pipeline:
    def __init__(self, encoder: VisualEncoder, llm: LLMProvider,
                 dp_params: Optional[DPParams] = None, fusion: str = "rrf",
                 portfolio: Optional[PortfolioConfig] = None,
                 visual_qa=None, translator: Optional[Translator] = None,
                 vietnamese_encoder: Optional[VisualEncoder] = None,
                 auto_route_vietnamese: bool = True,
                 auto_translate_clip: Optional[bool] = None,
                 retrieval_policy: Optional[RetrievalPolicy] = None,
                 retrieval_phu: Sequence = (),
                 kenh_chu=None,
                 trong_so_kenh_chu: float = 0.6,
                 don_slot_cfg=None,
                 qa_output_language: str = "vi",
                 qa_answer_translator: Optional[Translator] = None,
                 image_resolver=None,
                 vlm_fallback_to_zero_shot: bool = True,
                 qa_vlm_top_n: Optional[int] = 10,
                 qa_vlm_question_types: Optional[Sequence[str]] =
                 ("color", "open"),
                 qa_tail_answer: str = "vlm"):
        self.base_encoder = encoder
        # Compatibility for callers that used the old opt-out flag.  It now
        # controls automatic Vietnamese routing; explicit translator= always
        # selects the translation A/B path below.
        if auto_translate_clip is not None:
            auto_route_vietnamese = bool(auto_translate_clip)
        if translator is not None and not isinstance(encoder, DichRoiEncode):
            encoder = DichRoiEncode(encoder, translator)
        elif (auto_route_vietnamese
              and isinstance(encoder, CLIPTextEncoder)
              and not isinstance(encoder, MCLIPTextEncoder)):
            # Default runtime: English stays on the exact BTC CLIP text tower;
            # Vietnamese uses a multilingual text tower aligned to the same
            # 512-D BTC image vectors.  Never send raw Vietnamese to CLIP.
            vietnamese_encoder = vietnamese_encoder or MCLIPTextEncoder(
                cho_phep_live=True)
            encoder = ChonEncoderTheoNgonNgu(encoder, vietnamese_encoder)
        self.encoder = encoder
        self.llm = llm
        # QA zero-shot bằng CLIP. Khi có, nó THAY `llm.answer*` cho task qa.
        # Lý do tồn tại: `RuleBasedProvider.answer()` đọc visual_text/ocr_text/
        # asr_text, mà bộ data BTC để cả ba rỗng -> đáp án rỗng -> QA luôn 0 điểm
        # (đo 2026-08-08, 100/100 dòng rỗng). Xem `core/qa.py`.
        self.image_resolver = image_resolver
        has_vlm = bool(getattr(llm, "supports_images", False))
        self.vlm_fallback_to_zero_shot = bool(vlm_fallback_to_zero_shot)
        self.qa_vlm_top_n = (None if qa_vlm_top_n is None
                             else max(0, int(qa_vlm_top_n)))
        allowed_types = {"color", "count", "open"}
        self.qa_vlm_question_types = (
            allowed_types if qa_vlm_question_types is None
            else {str(item).lower() for item in qa_vlm_question_types})
        unknown_types = self.qa_vlm_question_types - allowed_types
        if unknown_types:
            raise ValueError(f"Loại câu QA VLM không hợp lệ: {unknown_types}")
        # Đáp án cho 90 dòng ĐUÔI (ngoài top-N gửi VLM). Đo ghép cặp 16/08 trên 30
        # câu dev, 0 request thêm:
        #
        #   "cheap" (đuôi dùng ClipZeroShotQA)   0.1933 ± 0.0625
        #   "vlm"   (lan đáp án VLM ra cả 100)   0.2067 ± 0.0618
        #   delta +0.0133 ± 0.0093 (1,4·SE — chưa vượt 2·SE)
        #
        # Chọn "vlm" làm mặc định dù chưa vượt 2·SE, vì đây KHÔNG phải một núm
        # được fit: cơ chế là đáp án VLM đúng 26/30 còn đáp án rẻ ~13/30, và
        # R-Score chỉ cộng điểm cho dòng nằm trong cửa sổ đúng — cửa sổ đó thuộc
        # đúng MỘT video, nên dùng cùng một đáp án tốt nhất cho mọi dòng là tối ưu.
        # Chi phí đổi: 0 request. Muốn về hành vi cũ thì đặt "cheap".
        if qa_tail_answer not in {"vlm", "cheap"}:
            raise ValueError("qa_tail_answer phải là 'vlm' hoặc 'cheap'")
        self.qa_tail_answer = qa_tail_answer
        if (visual_qa is None
                and (not has_vlm or self.vlm_fallback_to_zero_shot)
                and isinstance(self.base_encoder, CLIPTextEncoder)):
            from .qa import ClipZeroShotQA, DichDapAnVi, dung_sinh_ung_vien
            if qa_output_language not in {"en", "vi"}:
                raise ValueError("qa_output_language phải là 'en' hoặc 'vi'")
            answer_translator = qa_answer_translator or OpusMTEnViTranslator()
            localizer = (DichDapAnVi(answer_translator)
                         if qa_output_language == "vi" else None)
            visual_qa = ClipZeroShotQA(
                self.base_encoder, dung_sinh_ung_vien(),
                dich_dap_an=localizer)
        self.visual_qa = visual_qa
        self.retriever = MultiChannelRetriever(encoder)
        self.dp_params = dp_params or DPParams()
        self.fusion = fusion
        self.portfolio = portfolio or PortfolioConfig()
        self.retrieval_policy = retrieval_policy or RetrievalPolicy()
        # Retriever PHỤ để hợp nhất nhiều encoder: [(MultiChannelRetriever, trọng
        # số)]. Rỗng = hành vi cũ, một encoder. Xem `RetrievalService._hop_nhat`.
        self.retrieval_phu = list(retrieval_phu or ())
        # Kênh chữ OCR/ASR theo khoảng shot (Stage20). None = tắt hoàn toàn,
        # và tắt phải là mặc định cho tới khi có phép đo trên dev set.
        # Luôn là DANH SÁCH [(kho, trọng số)] — xem `RetrievalService`.
        if kenh_chu is None:
            self.kenh_chu = []
        elif isinstance(kenh_chu, (list, tuple)):
            self.kenh_chu = [(k, float(w)) for k, w in kenh_chu if k is not None]
        else:
            self.kenh_chu = [(kenh_chu, float(trong_so_kenh_chu))]
        self.trong_so_kenh_chu = float(trong_so_kenh_chu)
        # Phân bổ 100 slot theo VIDEO thay vì rải đều — xem `optimize/don_slot`.
        self.don_slot_cfg = don_slot_cfg or CauHinhDonSlot()
        self.retrieval_service: Optional[RetrievalService] = None
        self.trake_solver: Optional[FullTimelineTrakeSolver] = None
        self.last_trace: Optional[AnswerTrace] = None

    def index(self, events: List[Event], embeddings=None) -> None:
        self.retriever.index(events, embeddings=embeddings)
        for _kho, _ in (self.kenh_chu or ()):
            # Gắn lại mỗi lần index: lớp ghim thu chỉ mục xuống một video rồi
            # nạp lại đầy đủ, nên bản đồ khoảng->event phải theo kịp. Quên chỗ
            # này thì kênh chữ trả event không còn trong chỉ mục và bị lọc câm.
            _kho.gan_event(events)
        self.retrieval_service = RetrievalService(
            self.retriever, events, self.retrieval_policy,
            phu=self.retrieval_phu, kenh_chu=self.kenh_chu)
        self.trake_solver = FullTimelineTrakeSolver(
            self.retrieval_service,
            TrakeConfig(
                paths_per_video=self.portfolio.trake_paths_per_video,
                output_top=self.retrieval_policy.output_top_k),
            self.dp_params)

    # ----------------------------------------------------------------------- #
    def answer(self, query: Query, top_k: int = 100) -> List[Dict]:
        """Return ranked submission rows (dicts) for one query."""
        compiled = compile_query(
            query.task, query.text,
            [event.text for event in query.sub_events]
            if query.task == "trake" else None,
            text_goc=getattr(query, "text_goc", "") or query.question_vi)
        if query.task == "kis":
            return self._answer_kis(query, compiled, top_k)
        if query.task == "qa":
            return self._answer_qa(query, compiled, top_k)
        if query.task == "trake":
            return self._answer_trake(query, compiled, top_k)
        raise ValueError(f"unknown task: {query.task!r}")



    @staticmethod
    def _so_ung_vien(top_k: int) -> int:
        """Số ứng viên cần xin để sau khử trùng vẫn đủ `top_k` dòng."""
        return int(top_k) + max(10, int(top_k) // 5)

    def _don_slot(self, candidates, compiled):
        """Dồn 100 slot vào vài video đầu. Tắt hoặc lỗi thì trả nguyên xi.

        Encode câu truy vấn MỘT lần ở đây rồi truyền xuống, để `score_many`
        chấm lại timeline bằng ĐÚNG phép biến đổi của `search` (có khử modality
        gap). Chấm bằng vector thô là đổi hình học giữa hai bước — đúng lỗi đã
        mắc ở `reasoning/trake` hồi đợt 1 rebuild.
        """
        if not getattr(self.don_slot_cfg, "bat", False):
            return candidates
        try:
            text = compiled.retrieval_query or compiled.visual_queries[0]
            qvec = self.retrieval_service.retriever.encoder.encode_text(text)
            ra, bc = don_slot(candidates, self.retrieval_service, qvec,
                              self.don_slot_cfg)
            self.last_don_slot = bc
            return ra
        except Exception as exc:                              # noqa: BLE001
            # Lớp phân bổ hỏng thì bài nộp vẫn phải ra. Ghi lại để in.
            self.last_don_slot = f"LỖI: {str(exc)[:200]}"
            return candidates

    # ----------------------------------------------------------------------- #
    def _answer_kis(self, query: Query, compiled: CompiledQuery,
                    top_k: int) -> List[Dict]:
        self._ensure_retrieval_service()
        # Lấy DƯ BIÊN: `khu_trung` bỏ những dòng render giống hệt nhau (614
        # `frame_idx` trùng trong map-keyframes BTC), nên xin đúng 100 ứng viên
        # thì nộp ra 99 dòng. Dư biên để chỗ trống được lấp bằng giả thuyết
        # mới chứ không bỏ không.
        candidates = self.retrieval_service.retrieve_compiled(
            compiled, top_k=self._so_ung_vien(top_k))
        candidates = self._don_slot(candidates, compiled)
        trace = AnswerTrace(task="kis", n_candidates=len(candidates),
                            compiled_query=compiled)
        if not self.portfolio.enabled:
            # `max_rows=top_k`, KHÔNG để mặc định 100: từ khi `_so_ung_vien`
            # xin dư biên ứng viên, gọi `answer(top_k=72)` mà không truyền
            # xuống đây sẽ trả 86 dòng. Lớp ghim dựa vào con số đó để chia
            # slot, nên lệch ở đây là cấp sai hạn ngạch mà không ai thấy.
            rows = sub.kis_rows(candidates, max_rows=top_k)
            return self._finish(trace, candidates, rows)

        p = self.portfolio
        out = opt.kis_portfolio(candidates, p.calibrator,
                                tol_frames=p.kis_tol_frames, mass=p.mass,
                                frame_offsets=p.frame_offsets)
        return self._finish(self._with_portfolio(trace, out),
                            candidates, out.rows)

    def _answer_qa(self, query: Query, compiled: CompiledQuery,
                   top_k: int) -> List[Dict]:
        self._ensure_retrieval_service()
        candidates = self.retrieval_service.retrieve_compiled(
            compiled, top_k=self._so_ung_vien(top_k))
        candidates = self._don_slot(candidates, compiled)
        context = " ".join(
            c.event.visual_text + " " + c.event.ocr_text + " " + c.event.asr_text
            for c in candidates[:5])
        trace = AnswerTrace(task="qa", n_candidates=len(candidates),
                            compiled_query=compiled)
        # `question_vi` ĐỨNG TRƯỚC, giống hệt đường VLM ngay dưới. Bản cũ lấy
        # `compiled.question or query.text` nên câu hỏi thật bị bỏ qua: nó chỉ
        # chạy đúng khi MÔ TẢ cảnh tình cờ chứa "màu gì"/"bao nhiêu" để trình
        # biên dịch bóc ra. Bắt được 19/08 khi bật lớp viết lại truy vấn —
        # `query.text` thành tiếng Anh, luật sinh ứng viên khớp theo từ khoá
        # tiếng Việt nên trượt sạch, và MỌI câu QA rơi về nhóm nơi chốn
        # ('nhà bếp' 18/30 câu). Sai này có sẵn từ trước, chỉ là chưa lộ.
        question = query.question_vi or compiled.question or query.text

        # VLM thật: mỗi dòng phải được trả lời từ chính ảnh của event đó. Dùng
        # một đáp án của khung top-1 cho cả 100 dòng sẽ biến mọi khung còn lại
        # thành giả thuyết video/frame đúng nhưng answer sai. Cache provider bảo
        # đảm chạy lại cùng (event_id, câu hỏi) không tốn quota lần hai.
        question_type = self._qa_question_type(query.question_vi or question)
        use_vlm = (getattr(self.llm, "supports_images", False)
                   and question_type in self.qa_vlm_question_types)
        if use_vlm:
            if self.image_resolver is None:
                raise RuntimeError(
                    "Đã chọn VLM nhưng chưa cấu hình image_resolver. Bundle nhẹ "
                    "chỉ có vector, không có ảnh để trả lời QA.")
            prompt_question = query.question_vi or question
            # VLM chỉ trả lời phần đầu đắt giá; đuôi vẫn giữ đủ 100 dòng bằng
            # đáp án zero-shot. `None` là chế độ benchmark gọi VLM toàn bộ.
            n_vlm = (top_k if self.qa_vlm_top_n is None
                     else min(top_k, self.qa_vlm_top_n))
            selected = list(candidates[:n_vlm])
            # `resolve()` PHẢI nằm trong `try`. Nó ném ImageResolutionError, mà
            # lớp đó kế thừa FileNotFoundError -> OSError, KHÔNG phải
            # RuntimeError. Để nó ngoài `try` (hoặc chỉ bắt RuntimeError) thì
            # thiếu đúng MỘT file ảnh trong top-N của MỘT câu là cả đường QA ném
            # lỗi và điểm QA về 0 — dù `vlm_fallback_to_zero_shot` đang bật.
            try:
                requests = []
                for candidate in selected:
                    event = candidate.event
                    image_path = self.image_resolver.resolve(event.event_id)
                    event_context = " ".join(filter(None, (
                        event.visual_text, event.ocr_text, event.asr_text)))
                    requests.append((prompt_question, str(image_path),
                                     event.event_id, event_context))
                answers = self.llm.answer_images(requests) if requests else []
            except (RuntimeError, OSError) as exc:
                message = str(exc)
                # Thiếu key là LỖI CẤU HÌNH, không phải sự cố runtime: vẫn ném to
                # để không âm thầm chạy cheap suốt cả buổi thi mà tưởng đang chạy VLM.
                misconfigured_key = ("Thiếu GEMINI_API_KEY" in message
                                     or "Thiếu ANTHROPIC_API_KEY" in message)
                if (misconfigured_key or not self.vlm_fallback_to_zero_shot
                        or self.visual_qa is None):
                    raise
                trace.vlm_error = message
                nguyen_nhan = ("THIẾU ẢNH" if isinstance(exc, OSError)
                               else "VLM lỗi")
                warnings.warn(
                    f"{nguyen_nhan}; đang hạ xuống ClipZeroShotQA: {message}",
                    RuntimeWarning, stacklevel=2)
            else:
                answered = list(zip(selected, answers))
                tail = list(candidates[len(selected):top_k])
                if tail:
                    dap_an_duoi = self._dap_an_duoi(question, candidates, answers)
                    if dap_an_duoi is not None:
                        answered.extend((candidate, dap_an_duoi)
                                        for candidate in tail)
                trace.answers = [(c.event.event_id, a) for c, a in answered]
                rows = sub.qa_rows_per_candidate(answered, max_rows=top_k)
                return self._finish(trace, candidates, rows)

        def phan_phoi(n: int):
            """[(đáp án, xác suất)] — ưu tiên QA thị giác nếu có."""
            if self.visual_qa is not None:
                ra = self.visual_qa.tra_loi(
                    question, candidates,
                    lay_vector=self.retriever.visual.lay_vector)
                return ra[:n]
            return self.llm.answer_distribution(question, context, n=n)

        if not self.portfolio.enabled:
            trace.answers = [phan_phoi(1)[0][0]] if self.visual_qa is not None \
                else [self.llm.answer(question, context)]
            rows = sub.qa_rows(candidates, trace.answers[0],
                               max_rows=top_k)
            return self._finish(trace, candidates, rows)

        p = self.portfolio
        answers = phan_phoi(p.n_answers)
        trace.answers = list(answers)
        out = opt.qa_portfolio(candidates, answers, p.calibrator,
                               tol_frames=p.kis_tol_frames, mass=p.mass)
        return self._finish(self._with_portfolio(trace, out),
                            candidates, out.rows)

    def _dap_an_duoi(self, question: str, candidates,
                     answers: Sequence[str]) -> Optional[str]:
        """Đáp án cho các dòng NGOÀI top-N gửi VLM. Xem `qa_tail_answer`.

        `vlm` lấy đáp án PHỔ BIẾN NHẤT trong các đáp án VLM, hoà thì lấy đáp án
        của ứng viên hạng cao nhất — hạng cao nhất là ứng viên dễ thuộc đúng video
        nhất, còn lấy đa số thì đỡ được một khung bị đọc sai.
        """
        if self.qa_tail_answer == "vlm":
            sach = [str(x) for x in answers if str(x).strip()]
            if sach:
                dem: Dict[str, int] = {}
                for x in sach:
                    dem[x] = dem.get(x, 0) + 1
                nhieu_nhat = max(dem.values())
                for x in sach:                       # giữ thứ tự hạng
                    if dem[x] == nhieu_nhat:
                        return x
        if self.visual_qa is None:
            return None
        return self.visual_qa.tra_loi(
            question, candidates,
            lay_vector=self.retriever.visual.lay_vector)[0][0]

    @staticmethod
    def _qa_question_type(question: str) -> str:
        low = " ".join(str(question).lower().split())
        if re.search(r"\b(?:bao nhiêu|mấy|số lượng)\b", low):
            return "count"
        if "màu" in low:
            return "color"
        return "open"

    def _answer_trake(self, query: Query, compiled: CompiledQuery,
                      top_k: int) -> List[Dict]:
        self._ensure_retrieval_service()
        aligned = self.trake_solver.solve(compiled)[:top_k]
        trace = AnswerTrace(
            task="trake",
            n_candidates=self.trake_solver.last_stats.get(
                "coarse_candidates", 0),
            compiled_query=compiled)

        if not self.portfolio.enabled:
            rows = sub.trake_rows(aligned)
            return self._finish_trake(trace, aligned, rows)

        p = self.portfolio
        out = opt.trake_portfolio(
            aligned, p.calibrator, tol_frames=p.trake_tol_frames, mass=p.mass,
            jitters=p.trake_jitters)
        return self._finish_trake(
            self._with_portfolio(trace, out), aligned, out.rows)

    # ----------------------------------------------------------------------- #
    def _ensure_retrieval_service(self) -> None:
        if self.trake_solver is not None:
            return
        events = list(self.retriever.events.values())
        if not events:
            raise RuntimeError("Pipeline has no indexed events")
        self.retrieval_service = RetrievalService(
            self.retriever, events, self.retrieval_policy,
            phu=self.retrieval_phu, kenh_chu=self.kenh_chu)
        self.trake_solver = FullTimelineTrakeSolver(
            self.retrieval_service,
            TrakeConfig(
                paths_per_video=self.portfolio.trake_paths_per_video,
                output_top=self.retrieval_policy.output_top_k),
            self.dp_params)

    def _finish(self, trace: AnswerTrace, candidates,
                rows: List[Dict]) -> List[Dict]:
        trace.gate = decide(
            candidates, neighbor_support=self._temporal_support(candidates))
        evidence: Dict[str, str] = {}
        if rows:
            evidence = {
                "video_id": str(rows[0].get("video_id", "")),
                "frame_id": str(rows[0].get("frame_idx", "")),
            }
            if trace.task == "qa":
                evidence["answer"] = str(rows[0].get("answer", ""))
        required = ["video_id", "frame_id"]
        if trace.task == "qa":
            required.append("answer")
        trace.verification = verify_required_evidence(evidence, required)
        if not trace.verification.accepted and trace.gate.route == "fast":
            trace.gate = GateDecision(
                "escalate", trace.gate.confidence,
                trace.gate.reasons + ["evidence_rejected"])
        self.last_trace = trace
        return rows

    def _finish_trake(self, trace: AnswerTrace, alignments,
                      rows: List[Dict]) -> List[Dict]:
        if not alignments:
            trace.gate = GateDecision("escalate", 0.0, ["no_alignment"])
        else:
            top = alignments[0]
            reasons = []
            if top.n_miss:
                reasons.append("trake_missing_events")
            if top.confidence < 0.55:
                reasons.append("weak_temporal_alignment")
            trace.gate = GateDecision(
                "escalate" if reasons else "fast", top.confidence, reasons)
        evidence: Dict[str, str] = {}
        if rows and rows[0].get("frames"):
            evidence = {
                "video_id": str(rows[0].get("video_id", "")),
                "frame_id": str(rows[0]["frames"]),
            }
        trace.verification = verify_required_evidence(
            evidence, ["video_id", "frame_id"])
        if not trace.verification.accepted and trace.gate.route == "fast":
            trace.gate = GateDecision(
                "escalate", trace.gate.confidence,
                trace.gate.reasons + ["evidence_rejected"])
        self.last_trace = trace
        return rows

    @staticmethod
    def _temporal_support(candidates) -> float:
        if not candidates:
            return 0.0
        top = candidates[0]
        neighbours = sum(
            row.video_id == top.video_id
            and abs(row.t_start - top.t_start) <= 5.0
            for row in candidates[1:11])
        return min(1.0, neighbours / 3.0)

    # ----------------------------------------------------------------------- #
    @staticmethod
    def _with_portfolio(trace: AnswerTrace,
                        out: "opt.PortfolioResult") -> AnswerTrace:
        trace.expected_final = out.expected_final
        trace.n_scenarios = out.n_scenarios
        trace.covered_mass = out.covered_mass
        trace.n_scoring_rows = sum(1 for pl in out.placements if not pl.free_recall)
        return trace
