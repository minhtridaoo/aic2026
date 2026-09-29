"""LLM + visual-encoder interfaces, plus runnable pure-stdlib fallbacks.

PLUG POINTS
  LLMProvider   -> OpenAIProvider / GeminiProvider (config-selectable, single
                   interface so we don't fork cloud/local codebases).
  VisualEncoder -> SigLIP2 / BEiT3 text+image encoder. HashEmbedder below is a
                   deterministic stand-in so the whole pipeline runs & tests
                   without a GPU; it is NOT semantically meaningful.

Vietnamese caveat (critique #14): RuleBasedProvider does whitespace tokenisation
only. Real deployment should route parsing/answering through a VN-aware model and
OCR/ASR through VietOCR/PaddleOCR + Whisper.
"""
from __future__ import annotations

import hashlib
import base64
import json
import math
import mimetypes
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .models import Query, SubEvent


# --------------------------------------------------------------------------- #
# Prompt ảnh dùng CHUNG cho mọi provider VLM
# --------------------------------------------------------------------------- #
def vlm_image_prompt(question: str, context: str = "") -> str:
    """Prompt QA ảnh tiếng Việt, một bản duy nhất cho mọi provider VLM.

    Phải dùng chung, nếu không thì điểm giữa các provider không so được với
    nhau. Ba luật cuối sinh ra từ phép đo Haiku 4.5 ngày 2026-08-14, nơi 10/30
    câu mất điểm KHÔNG vì model nhìn sai mà vì dạng đáp án:
      - "Áo trắng" thay vì "trắng"  -> luật chỉ-nêu-thuộc-tính
      - "Hai người" thay vì "2"     -> luật chữ số
      - "Đen, nâu" / "Trắng đỏ"     -> luật một-đáp-án-duy-nhất
    """
    prompt = (
        "Hãy nhìn kỹ ảnh và trả lời câu hỏi bằng tiếng Việt. "
        "Chỉ xuất đúng một đáp án ngắn, thường 1–3 từ; không giải thích, "
        "không mở đầu và không có dấu câu cuối. Nếu hỏi số lượng, trả lời "
        "bằng chữ số. Nếu hỏi thuộc tính như màu sắc, chỉ nêu giá trị thuộc "
        "tính nhưng phải giữ trọn tên ghép: ví dụ \"xanh dương\", \"xanh lá\", "
        "không trả \"áo trắng\" hay chỉ \"xanh\". Nếu hỏi tên vật, người hoặc "
        "hành động, giữ trọn cụm từ cần thiết, ví dụ \"cây đánh trứng\". "
        "Nếu do dự giữa nhiều đáp án, chọn MỘT đáp án khả năng cao nhất; "
        "không liệt kê hai đáp án.\n"
        f"Câu hỏi: {question}"
    )
    if context.strip():
        prompt += f"\nNgữ cảnh phụ (chỉ dùng nếu phù hợp): {context.strip()}"
    return prompt


# Fingerprint lấy trực tiếp từ prompt mẫu: sửa prompt sẽ tự tạo namespace cache
# mới, không còn đọc nhầm đáp án của prompt cũ.
VLM_PROMPT_FINGERPRINT = hashlib.sha256(
    vlm_image_prompt("__QUESTION__", "__CONTEXT__").encode("utf-8")
).hexdigest()[:16]
VLM_NORMALIZER_VERSION = "vi-short-answer-v2"


# --------------------------------------------------------------------------- #
# LLM layer
# --------------------------------------------------------------------------- #
class LLMProvider(ABC):
    supports_images = False

    @abstractmethod
    def parse_query(self, text: str, task: str) -> Query: ...

    @abstractmethod
    def answer(self, question: str, context: str) -> str: ...

    # Dùng bởi `optimize.submission_optimizer.qa_portfolio`.
    def answer_distribution(self, question: str, context: str,
                            n: int = 3) -> List[Tuple[str, float]]:
        """Top-n candidate answers with probabilities, best first.

        QA R-Score needs video AND frame AND answer to match, and answers are
        mutually exclusive -- so each distinct answer string is a separate
        hypothesis that the 100-row budget can cover independently. A provider
        that returns only its argmax answer throws that away: 100 rows carrying
        one answer cover one answer.

        Default implementation degrades to the single `answer()` with mass 1.0,
        so existing providers keep working; override it in a real VLM provider
        (logprobs, self-consistency samples, or an explicit ranked list).
        """
        return [(self.answer(question, context), 1.0)]

    def answer_image(self, question: str, image_path: str,
                     event_id: str, context: str = "") -> str:
        """Answer from one real frame without changing the legacy signature.

        Existing text-only providers and ``answer_distribution`` callers keep
        using ``answer(question, context)``.  Only image-aware providers
        override this method and set ``supports_images = True``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} không hỗ trợ ảnh; dùng provider VLM")

    def suggest_modality(self, sub_event_text: str) -> Optional[str]:
        """Optional per-event modality hint for TRAKE cross-modal routing.

        Returning None means 'no hint' -> retrieval uses fused score. A WRONG
        hint is expected sometimes; temporal_dp applies a soft penalty rather
        than trusting it blindly.
        """
        return None


_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    # NOTE: naive; replace with pyvi/underthesea for Vietnamese word segmentation.
    return _TOKEN.findall(text.lower())


class RuleBasedProvider(LLMProvider):
    """Zero-dependency fallback. Deterministic, no network."""

    # crude modality cues
    _OCR_CUES = {"chữ", "text", "dòng chữ", "biển", "logo", "thương hiệu", "tên"}
    _ASR_CUES = {"nói", "phát biểu", "âm thanh", "lời", "giọng", "tuyên bố"}

    def parse_query(self, text: str, task: str) -> Query:
        qid = "q"
        if task == "trake":
            # Split a multi-event description on common separators.
            parts = re.split(r"\s*(?:->|→|;|\.\s|,\s*(?:sau đó|rồi|tiếp theo)\s+)\s*",
                             text.strip())
            parts = [p.strip() for p in parts if p.strip()]
            subs = [SubEvent(text=p, modality_hint=self.suggest_modality(p)) for p in parts]
            return Query(qid=qid, task=task, text=text, sub_events=subs)
        return Query(qid=qid, task=task, text=text)

    def answer(self, question: str, context: str) -> str:
        # Fallback: echo the most frequent capitalised token in context, else "".
        caps = re.findall(r"\b[A-ZĐ][\wÀ-ỹ]+\b", context)
        return caps[0] if caps else ""

    def suggest_modality(self, sub_event_text: str) -> Optional[str]:
        low = sub_event_text.lower()
        if any(c in low for c in self._OCR_CUES):
            return "ocr"
        if any(c in low for c in self._ASR_CUES):
            return "asr"
        return None  # default to fused visual-led retrieval


class CachedImageLLMProvider(LLMProvider):
    """Base class for a real VLM with a persistent, versioned answer cache."""

    supports_images = True

    def __init__(self, cache_path: str | os.PathLike, model: str,
                 parser: Optional[LLMProvider] = None,
                 prompt_fingerprint: str = VLM_PROMPT_FINGERPRINT,
                 normalizer_version: str = VLM_NORMALIZER_VERSION):
        self.cache_path = Path(cache_path)
        self.model = str(model)
        self.parser = parser or RuleBasedProvider()
        self.prompt_fingerprint = str(prompt_fingerprint)
        self.normalizer_version = str(normalizer_version)
        self._entries: Dict[str, Dict] = {}
        self.cache_hits = 0
        self.network_calls = 0
        self.backend_requests = 0
        self._load_cache()

    def parse_query(self, text: str, task: str) -> Query:
        return self.parser.parse_query(text, task)

    def answer(self, question: str, context: str) -> str:
        raise ValueError(
            "Provider VLM cần ảnh. Gọi answer_image(), hoặc cấu hình "
            "Pipeline(image_resolver=...).")

    def _cache_key(self, event_id: str, question: str) -> str:
        raw = json.dumps([
            self.model, self.prompt_fingerprint, self.normalizer_version,
            str(event_id), " ".join(str(question).split()),
        ],
                         ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _short_answer(text: str, question: str = "") -> str:
        """Dọn định dạng mà không cắt mất cụm danh từ có nghĩa.

        Hai chuẩn hóa có điều kiện theo câu hỏi là định dạng, không phải
        LLM-judge: số đếm luôn về chữ số; câu hỏi màu lấy trọn tên màu ghép.
        """
        lines = [part.strip() for part in str(text).splitlines()
                 if part.strip() and not part.strip().startswith("```")]
        if not lines:
            return ""
        line = re.sub(
            r"^(?:đáp\s*án|trả\s*lời|answer)\s*[:：-]\s*", "", lines[0],
            flags=re.IGNORECASE)
        line = re.split(r"[.!?;](?:\s|$)", line, maxsplit=1)[0]
        line = re.sub(
            r"^[\s*#>`'\"“”‘’-]+|[\s.,;:!?*`'\"“”‘’-]+$", "", line)
        line = " ".join(line.split())
        if not line:
            return ""

        question_low = " ".join(str(question).lower().split())
        line_low = line.lower()
        if re.search(r"\b(?:bao nhiêu|mấy|số lượng)\b", question_low):
            digit = re.search(r"\b\d+\b", line_low)
            if digit:
                return digit.group(0)
            number_words = {
                "không": "0", "một": "1", "mốt": "1", "hai": "2",
                "ba": "3", "bốn": "4", "tư": "4", "năm": "5",
                "lăm": "5", "sáu": "6", "bảy": "7", "tám": "8",
                "chín": "9", "mười": "10",
            }
            for word in re.findall(r"\w+", line_low, flags=re.UNICODE):
                if word in number_words:
                    return number_words[word]

        if "màu" in question_low:
            # Chọn màu xuất hiện sớm nhất; nếu cùng vị trí thì ưu tiên tên dài
            # để `xanh dương` không bị co thành `xanh`.
            colors = {
                "xanh lá cây": "xanh lá", "xanh da trời": "xanh dương",
                "xanh dương": "xanh dương", "xanh lá": "xanh lá",
                "vàng nhạt": "vàng nhạt", "xanh nhạt": "xanh nhạt",
                "xanh đậm": "xanh đậm", "đỏ": "đỏ", "cam": "cam",
                "vàng": "vàng", "lục": "xanh lá", "lam": "xanh dương",
                "tím": "tím", "hồng": "hồng", "nâu": "nâu",
                "đen": "đen", "trắng": "trắng", "xám": "xám",
                "xanh": "xanh",
            }
            matches = []
            for phrase, canonical in colors.items():
                found = re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", line_low)
                if found:
                    matches.append((found.start(), -len(phrase), canonical))
            if matches:
                return min(matches)[2]
        return line

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            entries = payload.get("entries", {})
            if not isinstance(entries, dict):
                raise ValueError("`entries` không phải object")
            self._entries = entries
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Cache VLM hỏng: {self.cache_path}: {exc}") from exc

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "model": self.model,
            "prompt_fingerprint": self.prompt_fingerprint,
            "normalizer_version": self.normalizer_version,
            "entries": self._entries,
        }
        temp = self.cache_path.with_name(
            f".{self.cache_path.name}.{os.getpid()}.tmp")
        temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        os.replace(temp, self.cache_path)

    def peek_cached_answer(self, question: str, event_id: str) -> Optional[str]:
        """Đọc cache không đổi bộ đếm; dùng để cứu kết quả sau failover lỗi."""
        cached = self._entries.get(self._cache_key(event_id, question))
        return str(cached["answer"]) if cached is not None else None

    def answer_image(self, question: str, image_path: str,
                     event_id: str, context: str = "") -> str:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Ảnh VLM không tồn tại: {path}")
        key = self._cache_key(event_id, question)
        cached = self._entries.get(key)
        if cached is not None:
            self.cache_hits += 1
            return str(cached["answer"])

        self.backend_requests += 1
        raw = self._request_image(question, path, context=context)
        answer = self._short_answer(raw, question)
        if not answer:
            raise ValueError(f"VLM trả đáp án rỗng cho {event_id}: {question!r}")
        self.network_calls += 1
        self._entries[key] = {
            "event_id": str(event_id), "question": str(question),
            "raw_answer": str(raw), "answer": answer, "model": self.model,
            "prompt_fingerprint": self.prompt_fingerprint,
            "normalizer_version": self.normalizer_version,
        }
        self._save_cache()
        return answer

    def answer_images(self, requests: Sequence[Tuple[str, str, str, str]]) \
            -> List[str]:
        """Batch-compatible interface; text/image/event/context per item.

        Providers without native batching retain exact legacy behaviour by
        calling ``answer_image`` item-by-item. Gemini overrides this to fit
        free-tier RPM limits while preserving one cache entry per item.
        """
        return [self.answer_image(question, image_path, event_id, context)
                for question, image_path, event_id, context in requests]

    @abstractmethod
    def _request_image(self, question: str, image_path: Path,
                       context: str = "") -> str: ...


class AnthropicVLMProvider(CachedImageLLMProvider):
    """Claude vision through the existing ``ANTHROPIC_API_KEY`` path."""

    DEFAULT_MODEL = "claude-sonnet-5"

    def __init__(self, cache_path="out/vlm_qa_cache.json",
                 model: Optional[str] = None, max_tokens: int = 64,
                 timeout_seconds: float = 30.0, max_retries: int = 0):
        super().__init__(cache_path, model or self.DEFAULT_MODEL)
        self.max_tokens = int(max_tokens)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self._client = None

    def _request_image(self, question: str, image_path: Path,
                       context: str = "") -> str:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise RuntimeError(
                    "Thiếu gói `anthropic`. Cài gói này trên máy/Kaggle chạy VLM."
                ) from exc
            self._client = anthropic.Anthropic(
                timeout=self.timeout_seconds, max_retries=self.max_retries)
        mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        if mime not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
            raise ValueError(f"Định dạng ảnh chưa hỗ trợ bởi VLM: {mime}")
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        prompt = vlm_image_prompt(question, context)
        try:
            response = self._client.messages.create(
                model=self.model, max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": mime, "data": encoded}},
                    {"type": "text", "text": prompt},
                ]}],
            )
        except Exception as exc:  # SDK có nhiều lớp lỗi không kế thừa RuntimeError
            raise RuntimeError(f"Anthropic SDK lỗi: {exc}") from exc
        return response.content[0].text


GENAI_TOI_THIEU = (2, 0, 0)


def _chan_sdk_cu() -> None:
    """Chặn SDK google-genai quá cũ NGAY khi dựng client, kèm cách sửa.

    Bắt được trên Kaggle 19/08: máy Kaggle cài sẵn bản < 2.0.0, và server trả
    `400 The legacy Interactions API schema is no longer supported`. Triệu chứng
    rất dễ đọc nhầm thành lỗi quota/khoá, mà nó lại làm hỏng MỌI lời gọi Gemini
    — cả QA lẫn viết lại truy vấn. Kiểm ở đây để hỏng thì hỏng to và sớm, chứ
    không phải sau khi đã nạp 177.321 vector.
    """
    import importlib.metadata as md

    try:
        raw = md.version("google-genai")
    except md.PackageNotFoundError:
        return                     # không đọc được phiên bản thì để SDK tự báo
    so = tuple(int(x) for x in re.findall(r"\d+", raw)[:3])
    if so and so < GENAI_TOI_THIEU:
        can = ".".join(map(str, GENAI_TOI_THIEU))
        raise RuntimeError(
            f"google-genai {raw} quá cũ (cần >= {can}). Server sẽ trả 400 "
            f"'legacy Interactions API schema is no longer supported' cho MỌI "
            f"lời gọi Gemini.\n"
            f"  Sửa: pip install -U 'google-genai>={can}'\n"
            f"  Trên Kaggle phải chạy TRƯỚC khi import, rồi RESTART kernel.")


class GeminiVLMProvider(CachedImageLLMProvider):
    """Gemini vision qua Google Gen AI SDK native + Interactions API.

    Không gọi REST thủ công và không đi qua ``models.generate_content``. Khóa
    chỉ đọc từ môi trường; ``client`` chỉ là điểm tiêm dependency cho unit test.
    """

    DEFAULT_MODEL = "gemini-3.6-flash"

    def __init__(self, cache_path="out/vlm_qa_cache_gemini.json",
                 model: Optional[str] = None, max_tokens: int = 64,
                 timeout_seconds: float = 60.0, max_retries: int = 2,
                 api_key: Optional[str] = None,
                 min_interval_seconds: float = 0.0,
                 batch_max_images: int = 1,
                 batch_max_bytes: int = 15 * 1024 * 1024, client=None):
        super().__init__(cache_path, model or self.DEFAULT_MODEL)
        self.max_tokens = int(max_tokens)
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.min_interval_seconds = float(min_interval_seconds)
        self.batch_max_images = max(1, int(batch_max_images))
        self.batch_max_bytes = int(batch_max_bytes)
        self._api_key = api_key
        self._client = client
        self.last_usage: Dict = {}
        self._last_request_at = 0.0

    def _key(self) -> str:
        key = self._api_key or os.environ.get("GEMINI_API_KEY") \
            or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "Thiếu GEMINI_API_KEY (hoặc GOOGLE_API_KEY). Không ghi key vào "
                "config/source; đặt biến môi trường ở tiến trình chạy.")
        return key

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:
                raise RuntimeError(
                    "Thiếu gói `google-genai>=2.3.0` cho Gemini native SDK."
                ) from exc
            _chan_sdk_cu()
            # Tắt retry ẩn của SDK (attempts=1); retry/backoff nằm ở provider để
            # circuit breaker có thể failover sớm thay vì treo nhiều phút.
            self._client = genai.Client(
                api_key=self._key(),
                http_options=types.HttpOptions(
                    timeout=max(1, int(self.timeout_seconds * 1000)),
                    retry_options=types.HttpRetryOptions(attempts=1)))
        return self._client

    _prompt = staticmethod(vlm_image_prompt)

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        if isinstance(exc, TimeoutError):
            return True
        code = getattr(exc, "code", None)
        if code in {429, 500, 502, 503, 504}:
            return True
        low = str(exc).lower()
        return any(word in low for word in (
            "timeout", "timed out", "connection", "high demand",
            "unavailable", "temporarily", "quota exceeded",
            "rate_limit_exceeded", "resource_exhausted",
            "too_many_requests", "error code: 408", "error code: 429"))

    @staticmethod
    def _retry_delay(exc: Exception, attempt: int) -> float:
        match = re.search(
            r"(?:retry\s+(?:in|after)|retrydelay[^0-9]*)\s*[:=]?\s*"
            r"['\"]?([0-9.]+)s",
            str(exc), flags=re.IGNORECASE)
        if match:
            # Request 429 bị từ chối vẫn có thể nằm trong cửa sổ rolling; sàn
            # 65s tránh tự đẩy cửa sổ về trước bằng retry quá sớm.
            return min(max(float(match.group(1)) + 1.0, 65.0), 180.0)
        return min(2 ** attempt, 20.0)

    def _sdk_create(self, **kwargs):
        """Chặn cứng timeout khi SDK sync bị treo trong transport.

        Thread là daemon để request mạng quá hạn không giữ tiến trình sống. Mỗi
        batch chỉ có tối đa một thread rơi rớt vì FailoverVLMProvider mở circuit
        ngay sau lỗi đầu tiên.
        """
        result = queue.Queue(maxsize=1)

        def invoke():
            try:
                value = self._get_client().interactions.create(**kwargs)
                result.put((True, value))
            except BaseException as exc:  # chuyển exception về thread gọi
                result.put((False, exc))

        worker = threading.Thread(target=invoke, daemon=True,
                                  name="gemini-native-sdk")
        worker.start()
        try:
            ok, value = result.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            raise TimeoutError(
                f"Gemini native SDK quá hạn {self.timeout_seconds:g}s") from exc
        if not ok:
            raise value
        return value

    def _interact(self, **kwargs):
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self.min_interval_seconds:
                time.sleep(self.min_interval_seconds - elapsed)
            try:
                self._last_request_at = time.monotonic()
                interaction = self._sdk_create(model=self.model, **kwargs)
                usage = getattr(interaction, "usage", None)
                self.last_usage = (usage.model_dump(exclude_none=True)
                                   if hasattr(usage, "model_dump") else {})
                return interaction
            except Exception as exc:  # SDK dùng APIError/httpx theo từng backend
                last_error = exc
                if attempt >= self.max_retries or not self._retryable(exc):
                    raise RuntimeError(f"Gemini native SDK lỗi: {exc}") from exc
                time.sleep(self._retry_delay(exc, attempt))
        raise RuntimeError(f"Gemini native SDK thất bại: {last_error}")

    def tra_loi_chu(self, system_instruction: str, prompt: str,
                    schema: Optional[Dict] = None) -> str:
        """Một lượt hỏi CHỈ-CHỮ, không ảnh. Trả `output_text` thô.

        Tách khỏi `answer_image*` vì đây không phải QA: nó phục vụ việc xử lý
        NGÔN NGỮ của truy vấn (viết lại câu tiếng Việt thành chú thích tiếng Anh —
        xem `core/viet_lai_truy_van.py`). Dùng chung `_interact` nên vẫn có
        timeout cứng, retry, `min_interval_seconds` và đếm `backend_requests`.

        Không cache ở đây: bên gọi tự quyết cách khoá, vì khoá phụ thuộc prompt
        của từng việc chứ không phải (event_id, câu hỏi) như QA.
        """
        if not str(prompt).strip():
            raise ValueError("prompt rỗng")
        kwargs: Dict = {
            "system_instruction": system_instruction,
            "input": [{"type": "text", "text": str(prompt)}],
        }
        if schema is not None:
            kwargs["response_format"] = {"type": "text",
                                         "mime_type": "application/json",
                                         "schema": schema}
        self.backend_requests += 1
        self.network_calls += 1
        interaction = self._interact(**kwargs)
        text = str(getattr(interaction, "output_text", "") or "")
        if not text.strip():
            raise RuntimeError("Gemini native SDK không trả output_text")
        return text

    # Điểm ảnh không khớp gì cả. Đặt tên để chỗ khác khỏi gõ số trần.
    DIEM_TOI_THIEU = 0.0
    DIEM_TOI_DA = 100.0

    def cham_anh(self, mo_ta: str, anh: Sequence[Path]) -> List[float]:
        """N ảnh + MỘT mô tả cảnh -> N điểm khớp 0..100. Một request duy nhất.

        VÌ SAO KHÔNG DÙNG `answer_images`
        ---------------------------------
        `answer_images` là N câu hỏi cho N ảnh ("người này mặc áo màu gì"), mỗi
        ảnh một đáp án độc lập. Ở đây ngược lại: MỘT câu hỏi trải trên N ảnh
        ("ảnh nào khớp mô tả này"), và model nhìn cả N ảnh cùng lúc nên nó SO
        SÁNH được — đúng thứ mắt người làm khi lướt trang kết quả.

        VÌ SAO CHẤM ĐIỂM CHỨ KHÔNG BẮT CHỌN MỘT
        ---------------------------------------
        Bắt chọn một ảnh ép model phải quyết ngay cả khi cả lô đều sai, và với
        100 ảnh chia 10 lô thì ta nhận về 10 "người thắng" giả. Chấm điểm cho
        phép cả lô cùng thấp — và điểm hoà nhau còn là TÍN HIỆU: model trả y
        một điểm cho mọi ảnh nghĩa là nó không nhìn. Bẫy đó đã bắt được thật
        ngày 20/08 với gemini-3.6-flash (cùng một đáp án cho 6 video khác nhau).

        Không cache ở đây: khoá phụ thuộc (mô tả, TẬP ảnh) chứ không phải
        (event_id, câu hỏi) như QA, nên bên gọi tự lo.
        """
        anh = [Path(p) for p in anh]
        if not anh:
            raise ValueError("cham_anh: danh sách ảnh rỗng")
        if not str(mo_ta).strip():
            raise ValueError("cham_anh: mô tả rỗng")

        noi_dung: List[Dict] = []
        for i, p in enumerate(anh, start=1):
            mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
            if mime not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
                raise ValueError(f"Định dạng ảnh chưa hỗ trợ: {mime} ({p.name})")
            # NHÃN ĐỨNG TRƯỚC ẢNH. Bản đầu đặt nhãn SAU ảnh và đo ra 50% top-1,
            # trong đó CẢ BA lô sai đều lệch đúng một vị trí về trước (đúng ở
            # #7 model trả #6, #9 -> #8, #9 -> #8). Đó không phải không hiểu
            # ảnh — đó là model gán nhãn cho ảnh KẾ TIẾP. Với "[Ảnh i]" đứng
            # trước thì không còn chỗ cho hiểu nhầm.
            noi_dung.append({"type": "text", "text": f"[Ảnh {i}]"})
            noi_dung.append({"type": "image",
                             "data": base64.b64encode(p.read_bytes()).decode("ascii"),
                             "mime_type": mime})
        noi_dung.append({"type": "text", "text": (
            f"Mô tả cảnh cần tìm:\n{mo_ta}\n\n"
            f"Chấm từng ảnh trong {len(anh)} ảnh trên theo mức khớp với mô tả, "
            f"thang 0-100. 0 = hoàn toàn không liên quan, 100 = đúng chính xác "
            f"cảnh được mô tả. Chấm ĐỘC LẬP từng ảnh; nhiều ảnh cùng thấp là "
            f"bình thường và được phép. Chỉ trả JSON.")})

        schema = {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"id": {"type": "integer"},
                                     "diem": {"type": "number"}},
                      "required": ["id", "diem"]},
        }
        self.backend_requests += 1
        self.network_calls += 1
        interaction = self._interact(
            system_instruction=(
                "Bạn chấm mức khớp giữa ảnh và mô tả cảnh. Chỉ trả JSON array, "
                "không giải thích."),
            input=noi_dung,
            response_format={"type": "text", "mime_type": "application/json",
                             "schema": schema})
        return self._doc_diem(str(getattr(interaction, "output_text", "") or ""),
                              len(anh))

    @classmethod
    def _doc_diem(cls, text: str, n: int) -> List[float]:
        """JSON -> đúng `n` điểm. Thiếu id nào thì cho 0, không đoán bù."""
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text).strip(),
                       flags=re.IGNORECASE)
        try:
            rows = json.loads(clean)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"cham_anh không trả JSON hợp lệ: {clean[:300]!r}") from exc
        if not isinstance(rows, list):
            raise RuntimeError("cham_anh phải trả JSON array")
        theo_id: Dict[int, float] = {}
        for row in rows:
            if not isinstance(row, dict) or row.get("id") is None:
                continue
            try:
                theo_id[int(row["id"])] = max(
                    cls.DIEM_TOI_THIEU,
                    min(cls.DIEM_TOI_DA, float(row.get("diem", 0))))
            except (TypeError, ValueError):
                continue
        if not theo_id:
            raise RuntimeError(f"cham_anh không đọc được điểm nào: {clean[:200]!r}")
        return [theo_id.get(i, cls.DIEM_TOI_THIEU) for i in range(1, n + 1)]

    def _request_image(self, question: str, image_path: Path,
                       context: str = "") -> str:
        mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        if mime not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
            raise ValueError(f"Định dạng ảnh chưa hỗ trợ bởi Gemini: {mime}")
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        interaction = self._interact(
            system_instruction=(
                "Tuân thủ chính xác định dạng đầu ra. Không thêm giải thích "
                "hoặc lời mở đầu."),
            input=[
                {"type": "image", "data": encoded, "mime_type": mime},
                {"type": "text", "text": self._prompt(question, context)},
            ])
        text = str(getattr(interaction, "output_text", "") or "")
        if not text.strip():
            raise RuntimeError("Gemini native SDK không trả output_text")
        return text

    @staticmethod
    def _json_answers(text: str) -> Dict[str, str]:
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text).strip(),
                       flags=re.IGNORECASE)
        try:
            rows = json.loads(clean)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Gemini batch không trả JSON hợp lệ: {clean[:300]!r}") from exc
        if not isinstance(rows, list):
            raise RuntimeError("Gemini batch phải trả JSON array")
        return {str(row["id"]): str(row.get("answer", "")) for row in rows
                if isinstance(row, dict) and row.get("id") is not None}

    def _groups(self, items):
        group, size = [], 0
        for item in items:
            item_size = item[4].stat().st_size
            if group and (len(group) >= self.batch_max_images
                          or size + item_size > self.batch_max_bytes):
                yield group
                group, size = [], 0
            group.append(item)
            size += item_size
        if group:
            yield group

    def answer_images(self, requests: Sequence[Tuple[str, str, str, str]]) \
            -> List[str]:
        if self.batch_max_images <= 1:
            return super().answer_images(requests)
        answers: List[Optional[str]] = [None] * len(requests)
        missing = []
        for index, (question, image_path, event_id, context) in enumerate(requests):
            path = Path(image_path)
            if not path.is_file():
                raise FileNotFoundError(f"Ảnh VLM không tồn tại: {path}")
            key = self._cache_key(event_id, question)
            cached = self._entries.get(key)
            if cached is not None:
                self.cache_hits += 1
                answers[index] = str(cached["answer"])
            else:
                missing.append((index, question, event_id, context, path, key))

        for group in self._groups(missing):
            input_parts, expected = [], []
            for local_index, (_, question, _, context, path, _) in enumerate(group):
                item_id = f"q{local_index:03d}"
                expected.append(item_id)
                mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                input_parts.extend([
                    {"type": "text", "text":
                     f"ID {item_id}\n{self._prompt(question, context)}"},
                    {"type": "image", "data": encoded, "mime_type": mime},
                ])
            input_parts.append({"type": "text", "text":
                "Trả đúng một JSON array, mỗi ID đúng một lần, theo thứ tự: "
                '[{"id":"q000","answer":"đáp án"}]. Không thêm chữ khác.'})
            self.backend_requests += 1
            interaction = self._interact(
                system_instruction=(
                    "Trả lời độc lập từng câu theo ảnh ngay sau câu đó. Đáp án "
                    "tiếng Việt 1–2 từ; số lượng dùng chữ số."),
                input=input_parts,
                response_format={
                    "type": "text", "mime_type": "application/json",
                    "schema": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"id": {"type": "string"},
                                       "answer": {"type": "string"}},
                        "required": ["id", "answer"]}}})
            returned = self._json_answers(
                str(getattr(interaction, "output_text", "") or ""))
            absent = [item_id for item_id in expected if item_id not in returned]
            if absent:
                raise RuntimeError(f"Gemini batch thiếu ID: {absent}")
            for item_id, item in zip(expected, group):
                index, question, event_id, _, _, key = item
                answer = self._short_answer(returned[item_id], question)
                if not answer:
                    raise ValueError(f"Gemini trả rỗng cho {event_id}")
                answers[index] = answer
                self.network_calls += 1
                self._entries[key] = {
                    "event_id": str(event_id), "question": str(question),
                    "raw_answer": str(returned[item_id]),
                    "answer": answer, "model": self.model,
                    "prompt_fingerprint": self.prompt_fingerprint,
                    "normalizer_version": self.normalizer_version}
            self._save_cache()
        return [str(answer) for answer in answers]


class GroqVLMProvider(CachedImageLLMProvider):
    """GroqCloud vision qua Chat Completions REST, không cần SDK riêng."""

    DEFAULT_MODEL = "qwen/qwen3.6-27b"
    ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
    _prompt = staticmethod(vlm_image_prompt)

    def __init__(self, cache_path="out/vlm_qa_cache_groq.json",
                 model: Optional[str] = None, max_tokens: int = 64,
                 timeout_seconds: float = 45.0,
                 api_key: Optional[str] = None, requester=None):
        super().__init__(cache_path, model or self.DEFAULT_MODEL)
        self.max_tokens = int(max_tokens)
        self.timeout_seconds = float(timeout_seconds)
        self._api_key = api_key
        # Điểm tiêm test: (url, headers, payload, timeout) -> dict.
        self._requester = requester

    def _key(self) -> str:
        key = self._api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "Thiếu GROQ_API_KEY. Đặt key trong biến môi trường/Kaggle "
                "Secrets; không ghi key vào config hoặc source.")
        return key

    def _post(self, payload: Dict) -> Dict:
        headers = {"Authorization": f"Bearer {self._key()}",
                   "Content-Type": "application/json",
                   "Accept": "application/json",
                   # Cloudflare Browser Integrity Check có thể chặn chữ ký mặc
                   # định `Python-urllib/*` bằng error 1010 trước khi request
                   # tới Groq. Khai báo client API ổn định, không giả browser.
                   "User-Agent": "aic2026-groq-api/1.0"}
        if self._requester is not None:
            return self._requester(
                self.ENDPOINT, headers, payload, self.timeout_seconds)
        request = urllib.request.Request(
            self.ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers, method="POST")
        try:
            with urllib.request.urlopen(
                    request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 403 and "1010" in body:
                raise RuntimeError(
                    "Groq gateway HTTP 403/error 1010 chặn chữ ký HTTP của "
                    "client. Đã gửi User-Agent API; nếu vẫn gặp lỗi này thì "
                    "egress IP Kaggle đang bị Groq/Cloudflare chặn — thử Groq "
                    "SDK chính thức hoặc liên hệ Groq kèm Ray ID.") from exc
            raise RuntimeError(f"Groq HTTP {exc.code}: {body[:1000]}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Groq network lỗi: {exc}") from exc

    def tra_loi_chu(self, system_instruction: str, prompt: str,
                    schema: Optional[Dict] = None,
                    max_tokens: int = 2048) -> str:
        """Một lượt hỏi CHỈ-CHỮ, không ảnh. Cùng hợp đồng với `GeminiVLMProvider`.

        `max_tokens` mặc định 2048 chứ KHÔNG dùng `self.max_tokens`: giá trị đó
        là 64, chỉnh cho đáp án QA 1–3 từ. Việc sinh chữ (viết lại cả lô truy vấn)
        cần nhiều token hơn hẳn, và bị cắt giữa câu thì JSON hỏng.

        `schema` chỉ dùng để BÁO Ý ĐỊNH: Groq json_object mode đòi JSON top-level
        là object, còn ở đây cần một array, nên không bật response_format mà dựa
        vào prompt + bộ đọc JSON chịu được chữ thừa.
        """
        if not str(prompt).strip():
            raise ValueError("prompt rỗng")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": str(system_instruction)},
                {"role": "user", "content": str(prompt)},
            ],
            "temperature": 0,
            "max_completion_tokens": int(max_tokens),
        }
        self.backend_requests += 1
        self.network_calls += 1
        response = self._post(payload)
        try:
            text = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Groq response thiếu choices[0].message.content: "
                f"{str(response)[:500]}") from exc
        if not str(text).strip():
            raise RuntimeError("Groq không trả nội dung")
        return str(text)

    def _request_image(self, question: str, image_path: Path,
                       context: str = "") -> str:
        mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        if mime not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
            raise ValueError(f"Định dạng ảnh chưa hỗ trợ bởi Groq: {mime}")
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": self._prompt(question, context)},
                {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64,{encoded}"}},
            ]}],
            "temperature": 0,
            "max_completion_tokens": self.max_tokens,
        }
        response = self._post(payload)
        try:
            text = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Groq response thiếu choices[0].message.content: "
                f"{str(response)[:500]}") from exc
        if not str(text).strip():
            raise RuntimeError("Groq không trả nội dung")
        return str(text)


class CloudflareVLMProvider(CachedImageLLMProvider):
    """Workers AI Vision qua REST API chính thức của Cloudflare.

    Không cần SDK hay key trong source. Token và Account ID chỉ đọc từ
    ``CLOUDFLARE_API_TOKEN`` / ``CLOUDFLARE_ACCOUNT_ID``. Cloudflare hiện yêu
    cầu chấp nhận Meta license một lần cho model Vision; việc đó là lựa chọn có
    chủ ý của người vận hành, không tự động làm trong provider.
    """

    DEFAULT_MODEL = "@cf/meta/llama-3.2-11b-vision-instruct"

    def __init__(self, cache_path="out/vlm_qa_cache_cloudflare.json",
                 model: Optional[str] = None, max_tokens: int = 64,
                 timeout_seconds: float = 45.0, api_token: Optional[str] = None,
                 account_id: Optional[str] = None, requester=None):
        super().__init__(cache_path, model or self.DEFAULT_MODEL)
        self.max_tokens = int(max_tokens)
        self.timeout_seconds = float(timeout_seconds)
        self._api_token = api_token
        self._account_id = account_id
        # Điểm tiêm cho test: (url, headers, json_body, timeout) -> dict.
        self._requester = requester

    def _credentials(self) -> Tuple[str, str]:
        token = self._api_token or os.environ.get("CLOUDFLARE_API_TOKEN")
        account = self._account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        if not token or not account:
            raise RuntimeError(
                "Thiếu CLOUDFLARE_API_TOKEN hoặc CLOUDFLARE_ACCOUNT_ID. "
                "Đặt hai biến môi trường, không ghi token vào config/source.")
        return token, account

    def _post(self, payload: Dict) -> Dict:
        token, account = self._credentials()
        url = ("https://api.cloudflare.com/client/v4/accounts/"
               f"{account}/ai/run/{self.model}")
        headers = {"Authorization": f"Bearer {token}",
                   "Content-Type": "application/json"}
        if self._requester is not None:
            return self._requester(url, headers, payload, self.timeout_seconds)
        request = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"), headers=headers,
            method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(f"Cloudflare HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Cloudflare Workers AI lỗi kết nối: {exc}") from exc

    def agree_to_model_license(self) -> str:
        """Chấp nhận license Meta, chỉ gọi khi CLI/user yêu cầu rõ ràng."""
        self.backend_requests += 1
        try:
            response = self._post({"prompt": "agree"})
        except RuntimeError as exc:
            # Workers AI hiện lưu agreement nhưng trả HTTP 403/code 5016 cho
            # chính request "agree". Nội dung chính thức nói rõ "may now use
            # the model", nên đây là acknowledgement thành công, không phải
            # lỗi token/quyền. Các HTTP 403 khác vẫn được ném nguyên vẹn.
            if "code\":5016" in str(exc) and "may now use the model" in str(exc):
                return "Cloudflare đã ghi nhận Meta license; model sẵn sàng."
            raise
        if not response.get("success", False):
            raise RuntimeError(f"Cloudflare không chấp nhận license: {response}")
        return str((response.get("result") or {}).get("response", ""))

    def _request_image(self, question: str, image_path: Path,
                       context: str = "") -> str:
        mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            raise ValueError(f"Cloudflare Vision chỉ nhận JPEG/PNG/WebP, đang là {mime}")
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        # Workers AI Vision nhận data URI theo hướng dẫn Cloudflare.
        payload = {"prompt": vlm_image_prompt(question, context),
                   "image": f"data:{mime};base64,{encoded}",
                   "max_tokens": self.max_tokens, "temperature": 0}
        response = self._post(payload)
        if not response.get("success", False):
            errors = response.get("errors") or response.get("messages") or response
            raise RuntimeError(f"Cloudflare Workers AI từ chối request: {errors}")
        result = response.get("result") or {}
        text = str(result.get("response", "") if isinstance(result, dict) else result)
        if not text.strip():
            raise RuntimeError("Cloudflare Workers AI không trả result.response")
        return text


class FailoverVLMProvider(LLMProvider):
    """Failover theo request, có circuit breaker trong mỗi batch.

    Provider lỗi một lần sẽ bị bỏ qua cho phần còn lại của batch. Cache vẫn do
    từng provider quản lý, nên lần chạy sau không tốn lại request đã thành công.
    """

    supports_images = True

    def __init__(self, providers: Sequence[LLMProvider]):
        self.providers = list(providers)
        if not self.providers:
            raise ValueError("FailoverVLMProvider cần ít nhất một provider")
        self.model = " -> ".join(getattr(p, "model", type(p).__name__)
                                 for p in self.providers)
        self.failures: List[str] = []
        self.last_provider: Optional[str] = None

    @property
    def cache_hits(self):
        return sum(int(getattr(p, "cache_hits", 0)) for p in self.providers)

    @property
    def network_calls(self):
        return sum(int(getattr(p, "network_calls", 0)) for p in self.providers)

    @property
    def backend_requests(self):
        return sum(int(getattr(p, "backend_requests", 0)) for p in self.providers)

    @property
    def total_cost_usd(self):
        return sum(float(getattr(p, "total_cost_usd", 0.0))
                   for p in self.providers)

    def parse_query(self, text: str, task: str) -> Query:
        return self.providers[0].parse_query(text, task)

    def answer(self, question: str, context: str) -> str:
        raise ValueError("Failover VLM cần ảnh; gọi answer_image().")

    def peek_cached_answer(self, question: str, event_id: str) -> Optional[str]:
        for provider in self.providers:
            peek = getattr(provider, "peek_cached_answer", None)
            answer = peek(question, event_id) if peek is not None else None
            if answer is not None:
                return answer
        return None

    def answer_image(self, question: str, image_path: str,
                     event_id: str, context: str = "") -> str:
        return self.answer_images([(question, image_path, event_id, context)])[0]

    def answer_images(self, requests: Sequence[Tuple[str, str, str, str]]) \
            -> List[str]:
        errors = []
        for provider in self.providers:
            try:
                answers = provider.answer_images(requests)
                self.last_provider = type(provider).__name__
                return answers
            except RuntimeError as exc:
                detail = f"{type(provider).__name__}: {exc}"
                errors.append(detail)
                self.failures.append(detail)
        raise RuntimeError("VLM failover cạn provider: " + " | ".join(errors))


# --------------------------------------------------------------------------- #
# Visual encoder
# --------------------------------------------------------------------------- #
class VisualEncoder(ABC):
    @abstractmethod
    def encode_text(self, text: str) -> List[float]: ...


class HashEmbedder(VisualEncoder):
    """Deterministic hashed bag-of-words -> unit vector. Reproducible stub.

    Replace with SigLIP2/BEiT3 text encoder (and an image encoder for frames).
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode_text(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


class CLIPTextEncoder(VisualEncoder):
    """Encode CÂU TRUY VẤN vào đúng không gian 512 chiều mà BTC đã dùng cho ảnh.

    BTC cấp sẵn vector ảnh (`clip-features-32`, chính là `clip-ViT-B-32`), nên ta
    KHÔNG phải encode 177.321 ảnh. Nhưng muốn tìm bằng chữ thì vẫn cần nửa còn
    lại của CLIP — phần text tower — để câu truy vấn nằm cùng không gian.

    Hai chế độ, thử theo thứ tự:

      1. CACHE  — đọc file vector đã encode sẵn (encode trên Kaggle rồi tải về).
                  Không cần thư viện gì, chạy offline. Dùng khi mạng chậm hoặc
                  khi muốn kết quả tái lập chính xác.
      2. LIVE   — encode tại chỗ bằng `transformers` hoặc `open_clip` nếu máy có.

    KHÔNG có chế độ thứ ba. Nếu một câu truy vấn không có trong cache và cũng
    không encode được, hàm sẽ NÉM LỖI chứ không trả vector ngẫu nhiên. Đó chính
    là cái bẫy của `HashEmbedder`: nó luôn trả về một vector trông hợp lệ, nên
    hệ thống vẫn in ra điểm số đẹp đẽ trong khi không hề tìm kiếm gì cả.
    """

    # DIM = None nghĩa là "suy ra từ dữ liệu": lớp con dùng model mà số chiều phụ
    # thuộc checkpoint (SigLIP2 base 768 vs so400m 1152) không thể chốt cứng ở
    # đây. Với CLIP ViT-B/32 thì 512 là hằng số thật của model nên giữ nguyên —
    # chốt được thì chốt, để cache lệch chiều bị chặn ngay lúc nạp.
    DIM: Optional[int] = 512
    MODEL_ID = "openai/clip-vit-base-patch32"
    PIP = "transformers"          # gói cần cài để chạy được chế độ LIVE

    def __init__(self, cache_path=None, cho_phep_live: bool = True):
        self.dim = self.DIM
        self._cache: Dict[str, List[float]] = {}
        self._live = None
        self._cho_phep_live = cho_phep_live
        self._ly_do_khong_live = ""
        self.so_lan_cache_trung = 0
        self.so_lan_encode_live = 0
        if cache_path:
            self.nap_cache(cache_path)

    # ---- cache -------------------------------------------------------------
    def nap_cache(self, path) -> int:
        """Đọc file .npz chứa `keys` (chuỗi) và `vectors` (N, D).

        Nếu file có ghi kèm khoá `model`, nó PHẢI khớp `MODEL_ID` của lớp này.
        Hai text tower khác nhau (CLIP gốc vs bản đa ngữ) cùng cho vector 512
        chiều và cùng ghép được với vector ảnh, nên trộn nhầm sẽ không có triệu
        chứng nào — chỉ có điểm số tụt mà không rõ vì sao. File cũ chưa ghi
        `model` thì bỏ qua kiểm tra (giữ tương thích ngược).
        """
        import numpy as np  # chỉ cần khi dùng cache

        d = np.load(str(path), allow_pickle=False)
        keys, vecs = d["keys"], d["vectors"]
        model = str(d["model"][()]) if "model" in d.files else ""
        if model and model != self.MODEL_ID:
            raise ValueError(
                f"cache encode bằng {model!r}, nhưng encoder này là "
                f"{self.MODEL_ID!r}.\n"
                "  Hai text tower khác nhau => vector không so sánh được.\n"
                "  Dùng đúng lớp encoder, hoặc encode lại cache.")
        self._chot_dim(int(vecs.shape[1]), "cache")
        if len(keys) != vecs.shape[0]:
            raise ValueError(f"cache lệch: {len(keys)} key, {vecs.shape[0]} vector")
        for k, v in zip(keys, vecs):
            self._cache[self._khoa(str(k))] = [float(x) for x in v]
        return len(keys)

    @classmethod
    def luu_cache(cls, path, texts, vectors, model_id=None) -> None:
        """Ghi file cache, kèm tên model đã encode để `nap_cache` kiểm chéo."""
        import numpy as np

        np.savez_compressed(str(path),
                            keys=np.array(list(texts), dtype=object).astype(str),
                            vectors=np.asarray(vectors, dtype=np.float32),
                            model=np.array(model_id or cls.MODEL_ID))

    @staticmethod
    def _khoa(text: str) -> str:
        """Chuẩn hoá khoá cache: bỏ khoảng trắng thừa, không đụng dấu tiếng Việt."""
        return " ".join(text.split())

    def _chot_dim(self, d: int, nguon: str) -> None:
        """Chốt số chiều lần đầu gặp, rồi bắt mọi nguồn sau phải khớp.

        Với `DIM` cố định (CLIP) đây là phép kiểm như cũ. Với `DIM = None`
        (SigLIP2) nó học số chiều từ nguồn đầu tiên — nhưng vẫn chặn việc trộn
        hai checkpoint khác chiều trong cùng một encoder, vì trộn xong thì
        `NumpyVectorStore` chỉ báo lỗi shape ở tận lúc matmul, hoặc tệ hơn là
        không báo gì nếu tình cờ cùng chiều.
        """
        if self.dim is None:
            self.dim = d
            return
        if d != self.dim:
            raise ValueError(
                f"{nguon} {d} chiều, nhưng encoder này đang ở {self.dim} chiều"
                + (f" (DIM cố định của {type(self).__name__})"
                   if self.DIM is not None else " (đã chốt từ nguồn trước)"))

    # ---- live --------------------------------------------------------------
    def _lay_live(self):
        if self._live is not None:
            return self._live
        if not self._cho_phep_live:
            return None
        try:
            import torch
            from transformers import (CLIPTextModelWithProjection,
                                      CLIPTokenizerFast)

            # ⚠ KHÔNG dùng `CLIPModel.get_text_features()`.
            #
            # Bug đã trả giá ngày 2026-08-08: ở transformers 5.x hàm đó trả về
            # `BaseModelOutputWithPooling` mà `pooler_output` ĐÃ nhân
            # `text_projection` sẵn (đo được: cosine 1.000 với text_embeds của
            # lớp này). Code cũ thấy không có `text_embeds` nên nhân
            # `text_projection` LẦN NỮA -> project hai lần -> cosine chỉ 0.096
            # với vector đúng, tức gần như vuông góc = nhiễu.
            #
            # Vì ViT-B/32 có hidden size = projection dim = 512, vector sai vẫn
            # đúng 512 chiều, vẫn chuẩn hoá được, vẫn chạy trơn tru. Không một
            # phép kiểm shape nào bắt được. Hậu quả: `do_diem_thuc_te.py` in
            # 0.0000 cho cả 60 câu, trông y như "encoder yếu".
            #
            # `CLIPTextModelWithProjection` có ở cả transformers 4.x và 5.x, và
            # project ĐÚNG MỘT LẦN theo thiết kế — nên nó xoá luôn nhánh rẽ theo
            # phiên bản chứ không phải đoán cho đúng.
            tok = CLIPTokenizerFast.from_pretrained(self.MODEL_ID)
            mod = CLIPTextModelWithProjection.from_pretrained(
                self.MODEL_ID).eval()

            def encode(text: str) -> List[float]:
                with torch.no_grad():
                    x = tok([text], return_tensors="pt",
                            padding=True, truncation=True, max_length=77)
                    v = mod(**x).text_embeds[0]
                    v = v / v.norm()
                self._chot_dim(int(v.shape[-1]), "encode live")
                return [float(z) for z in v]

            self._live = encode
            return encode
        except Exception as exc:                       # noqa: BLE001
            self._live = None
            self._cho_phep_live = False
            self._ly_do_khong_live = f"{type(exc).__name__}: {exc}"
            return None

    # ---- giao diện chung ---------------------------------------------------
    def encode_text(self, text: str) -> List[float]:
        khoa = self._khoa(text)
        if khoa in self._cache:
            self.so_lan_cache_trung += 1
            return self._cache[khoa]

        encode = self._lay_live()
        if encode is not None:
            v = encode(text)
            self._cache[khoa] = v
            self.so_lan_encode_live += 1
            return v

        raise LookupError(
            f"Không encode được truy vấn: {text[:60]!r}\n"
            "  - Không có trong cache, và không dựng được encoder tại chỗ.\n"
            f"  - Lý do: {self._ly_do_khong_live or 'chế độ live bị tắt'}\n"
            f"  - Cách 1: pip install {self.PIP}  (tải ~600MB)\n"
            "  - Cách 2: encode câu truy vấn trên Kaggle rồi nạp qua nap_cache().\n"
            "  Cố tình NÉM LỖI thay vì trả vector bừa — xem docstring của lớp.")


class MCLIPTextEncoder(CLIPTextEncoder):
    """Text tower ĐA NGỮ, căn sẵn vào đúng không gian ảnh của CLIP ViT-B/32.

    Dùng mặc định khi truy vấn là TIẾNG VIỆT. Xem số đo bên dưới để phân biệt
    chất lượng của text tower với chất lượng bản dịch người.

    Đo 2026-08-08 trên dev set 60 câu có nhãn, đối chiếu toàn bộ 177.321 keyframe
    thật. Cột "hạng" = vị trí của keyframe ĐÚNG trong 177.321 (thấp là tốt;
    ~88.660 = xếp ngẫu nhiên):

        cách        hạng trung vị   R@100    R@100 (đúng video)
        clip_vi            72.620    0.0%           3.3%    ❌ ~ngẫu nhiên
        clip_en               283   28.3%          61.7%    ✅ tốt nhất
        mclip_vi            1.183   18.3%          56.7%
        mclip_en              468   23.3%          55.0%

    Đọc bảng:
      - CLIP gốc + tiếng Việt = xếp ngẫu nhiên. OpenAI CLIP huấn luyện trên
        alt-text tiếng Anh, nó KHÔNG biết tiếng Việt. Kết luận này đứng vững.
      - Bản tiếng Anh NGƯỜI viết + CLIP (283) vẫn tốt hơn mCLIP tiếng Việt
        (1.183). Đây là trần tham chiếu cho chất lượng dịch, không phải bằng
        chứng rằng máy dịch Opus đạt cùng chất lượng.
      - mCLIP cứu R@100 từ 0.0% lên 18.3% mà không phụ thuộc chất lượng máy
        dịch, nên là đường runtime mặc định cho câu tiếng Việt của BTC.

    ⚠ Bảng cũ trong file này từng ghi mclip_vi thắng. Số đó sinh ra từ đường
    CLIP bị project HAI LẦN (xem ghi chú trong `CLIPTextEncoder._lay_live`), nên
    cột clip_* lúc ấy là nhiễu. Đã đo lại sau khi sửa.

    Vẫn dùng nguyên 173 MB vector ảnh BTC cấp sẵn — chỉ thay nửa chữ của CLIP.
    """

    MODEL_ID = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    PIP = "sentence-transformers"

    def _lay_live(self):
        if self._live is not None:
            return self._live
        if not self._cho_phep_live:
            return None
        try:
            from sentence_transformers import SentenceTransformer

            mod = SentenceTransformer(self.MODEL_ID)

            def encode(text: str) -> List[float]:
                v = mod.encode([text], convert_to_numpy=True,
                               normalize_embeddings=True)[0]
                self._chot_dim(int(v.shape[-1]), "encode live")
                return [float(z) for z in v]

            self._live = encode
            return encode
        except Exception as exc:                       # noqa: BLE001
            self._live = None
            self._cho_phep_live = False
            self._ly_do_khong_live = f"{type(exc).__name__}: {exc}"
            return None


class SigLIP2TextEncoder(CLIPTextEncoder):
    """Text tower của SigLIP2 — dùng CÙNG CẶP với vector ảnh SigLIP2 tự encode.

    Khác mọi encoder phía trên ở một điểm quyết định: nó KHÔNG dùng được vector
    ảnh BTC cấp sẵn. CLIP/mCLIP ghép được vào `clip-features-32` vì cùng không
    gian 512 chiều của ViT-B/32; SigLIP2 là không gian khác hẳn, nên phải chạy
    `kaggle_encode_siglip2.py` để có vector ảnh trước, rồi trỏ
    `load_dataset(goc_feature=...)` vào đó.

    ĐO 2026-08-12 — dev set 120 câu có nhãn, 177.321 keyframe thật
    --------------------------------------------------------------
    Ghép cặp theo từng câu (`so_sanh_clip_siglip2.py`). "hạng" = vị trí của
    keyframe ĐÚNG trong 177.321, cosine thô, không khử modality gap:

        cấu hình              điểm     Δ so nền      hạng trung vị
        CLIP+EN (BTC cấp)   0.1867       (nền)                 246
        SigLIP2+EN          0.2950   +0.1083 ± 0.0313          119   ✅ vượt 2·SE
        SigLIP2+VI          0.1750   -0.0117 ± 0.0293          285   chưa vượt

    Đọc bảng:
      - SigLIP2 base-224 ăn **+0.1083**, tức 3.5·SE — cú nhảy lớn nhất từ đầu dự
        án, và lớn hơn mọi thứ tầng `optimize` từng đo được (≤ +0.0067).
      - Hạng trung vị 246 -> 119. R@1 2.5% -> 10.8%, R@100 38.3% -> 50.8%.
      - **Tiếng Việt thẳng KÉM hơn tiếng Anh người dịch −0.1200 ± 0.0225** (vượt
        2·SE). Nên KHÔNG bỏ được ô tiếng Anh khi gán nhãn — việc 2.6 trả lời PHỦ
        ĐỊNH. SigLIP2 đọc được tiếng Việt (285 so với 72.620 của CLIP+VI) nhưng
        chỉ ngang đường CLIP+EN cũ, không thay được nó.

    Con số 283 trong bảng của `MCLIPTextEncoder` đo trên dev set 60 câu; bảng này
    đo trên 120 câu nên CLIP+EN ra 246. Hai bảng KHÔNG so trực tiếp với nhau
    được — chỉ so trong cùng một bảng.

    SỐ CHIỀU KHÔNG CHỐT ĐƯỢC Ở ĐÂY
    ------------------------------
    `base-patch16-224` cho 768, `so400m-patch14-384` cho 1152. Vì vậy
    `DIM = None` và số chiều được học từ cache (hoặc từ lần encode live đầu
    tiên), rồi khoá lại. Khoá `model` trong .npz vẫn là hàng rào chính chống
    trộn hai checkpoint — hai bộ cùng 768 chiều mà khác model thì phép kiểm
    chiều không bắt được, chỉ tên model mới bắt được.

    ⚠ ĐỔI CHECKPOINT thì phải đổi CẢ hai phía: `MODEL_ID` ở đây và `MODEL_ID`
    trong `kaggle_encode_siglip2.py`. Lệch một bên là text và ảnh nằm ở hai
    không gian, điểm sẽ tụt về mức ngẫu nhiên mà không có lỗi nào ném ra.
    """

    MODEL_ID = "google/siglip2-base-patch16-224"
    DIM = None                    # suy ra từ cache / lần encode đầu
    PIP = "transformers"

    # SigLIP huấn luyện với chuỗi text ĐỆM CỨNG tới 64 token. Dùng
    # `padding=True` (đệm tới câu dài nhất trong batch) là sai quy ước huấn
    # luyện và cho vector khác — đây là lỗi thường gặp nhất khi chuyển từ CLIP
    # sang SigLIP, và nó KHÔNG ném lỗi, chỉ làm điểm kém đi.
    MAX_LEN = 64

    def _lay_live(self):
        if self._live is not None:
            return self._live
        if not self._cho_phep_live:
            return None
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.MODEL_ID)
            mod = AutoModel.from_pretrained(self.MODEL_ID).eval()

            def encode(text: str) -> List[float]:
                with torch.no_grad():
                    x = tok([text], return_tensors="pt",
                            padding="max_length", truncation=True,
                            max_length=self.MAX_LEN)
                    v = _go_vector(mod.get_text_features(**x))[0]
                    v = v / v.norm()
                self._chot_dim(int(v.shape[-1]), "encode live")
                return [float(z) for z in v]

            self._live = encode
            return encode
        except Exception as exc:                       # noqa: BLE001
            self._live = None
            self._cho_phep_live = False
            self._ly_do_khong_live = f"{type(exc).__name__}: {exc}"
            return None


def _go_vector(out):
    """`get_*_features` -> tensor (B, D), chịu được cả transformers 4.x và 5.x.

    4.x trả thẳng tensor; 5.x trả `BaseModelOutputWithPooling` mà `pooler_output`
    ĐÃ được project sẵn. Cùng họ bug đã trả giá 2026-08-08 (xem ghi chú trong
    `CLIPTextEncoder._lay_live`), chỉ khác là ở đây nó ném AttributeError ngay
    nên không âm thầm.

    KHÔNG rơi về `last_hidden_state`: nó là (B, số token, D), lấy nhầm thì mọi
    phép kiểm chiều vẫn qua mà vector thì vô nghĩa.
    """
    if hasattr(out, "shape"):
        return out
    for ten in ("text_embeds", "image_embeds", "pooler_output"):
        v = getattr(out, ten, None)
        if v is not None:
            if v.dim() != 2:
                raise ValueError(f"`{ten}` phải là (B, D), đang là {tuple(v.shape)}")
            return v
    raise TypeError(
        f"{type(out).__name__} không có text_embeds/image_embeds/pooler_output; "
        f"các field có: {list(getattr(out, 'keys', lambda: [])())}")


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_llm(config: Dict) -> LLMProvider:
    cfg = (config or {}).get("llm", {})
    return _build_llm_cfg(cfg)


def _build_llm_cfg(cfg: Dict) -> LLMProvider:
    if cfg.get("enabled", True) is False:
        raise ValueError(
            f"Provider {cfg.get('provider', '<không rõ>')} đang enabled=false")
    kind = cfg.get("provider", "rule_based")
    if kind == "rule_based":
        return RuleBasedProvider()
    if kind == "anthropic_vlm":
        return AnthropicVLMProvider(
            cache_path=cfg.get("cache_path", "out/vlm_qa_cache.json"),
            model=cfg.get("model"),
            max_tokens=int(cfg.get("max_tokens", 64)),
            timeout_seconds=float(cfg.get("timeout_seconds", 30)),
            max_retries=int(cfg.get("max_retries", 0)),
        )
    if kind == "gemini_vlm":
        return GeminiVLMProvider(
            cache_path=cfg.get("cache_path", "out/vlm_qa_cache_gemini.json"),
            model=cfg.get("model"),
            max_tokens=int(cfg.get("max_tokens", 64)),
            timeout_seconds=float(cfg.get("timeout_seconds", 60)),
            max_retries=int(cfg.get("max_retries", 2)),
            min_interval_seconds=float(cfg.get("min_interval_seconds", 0)),
            batch_max_images=int(cfg.get("batch_max_images", 1)),
            batch_max_bytes=int(cfg.get("batch_max_bytes", 15 * 1024 * 1024)),
        )
    if kind == "cloudflare_vlm":
        return CloudflareVLMProvider(
            cache_path=cfg.get("cache_path", "out/vlm_qa_cache_cloudflare.json"),
            model=cfg.get("model"),
            max_tokens=int(cfg.get("max_tokens", 64)),
            timeout_seconds=float(cfg.get("timeout_seconds", 45)),
        )
    if kind == "groq_vlm":
        return GroqVLMProvider(
            cache_path=cfg.get("cache_path", "out/vlm_qa_cache_groq.json"),
            model=cfg.get("model"),
            max_tokens=int(cfg.get("max_tokens", 64)),
            timeout_seconds=float(cfg.get("timeout_seconds", 45)),
        )
    if kind == "failover_vlm":
        primary = cfg.get("primary") or {
            "provider": "gemini_vlm",
            "model": GeminiVLMProvider.DEFAULT_MODEL,
        }
        fallbacks = cfg.get("fallbacks") or []
        enabled = [item for item in [primary, *fallbacks]
                   if item.get("enabled", True)]
        if not enabled:
            raise ValueError("Failover VLM không còn provider enabled=true")
        return FailoverVLMProvider([_build_llm_cfg(item) for item in enabled])
    # PLUG: elif kind == "openai": return OpenAIProvider(...)
    #       elif kind == "gemini": return GeminiProvider(...)
    raise ValueError(f"unknown/unimplemented llm provider: {kind!r}")


def build_encoder(config: Dict) -> VisualEncoder:
    enc = (config or {}).get("encoder", {})
    kind = enc.get("provider", "hash")
    if kind == "hash":
        return HashEmbedder(dim=int(enc.get("dim", 256)))
    if kind in ("clip", "mclip", "siglip2"):
        lop = {"clip": CLIPTextEncoder,
               "mclip": MCLIPTextEncoder,
               "siglip2": SigLIP2TextEncoder}[kind]
        e = lop(cho_phep_live=bool(enc.get("live", True)))
        # Ghi đè checkpoint qua config (vd so400m-patch14-384) PHẢI xong TRƯỚC khi
        # nạp cache — `nap_cache` đối chiếu khoá `model` trong .npz với MODEL_ID,
        # gán sau thì phép kiểm đó chạy với giá trị cũ và mất tác dụng.
        # Giá trị này phải khớp `MODEL_ID` đã dùng trong `kaggle_encode_siglip2.py`,
        # nếu không text và ảnh nằm ở hai không gian khác nhau.
        if enc.get("model_id"):
            e.MODEL_ID = str(enc["model_id"])
        if enc.get("cache_path"):
            e.nap_cache(enc["cache_path"])
        return e
    raise ValueError(f"unknown/unimplemented encoder: {kind!r}")
