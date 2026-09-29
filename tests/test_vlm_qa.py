from pathlib import Path

import pytest

from aic2026.core.providers import CachedImageLLMProvider
from aic2026.core.models import Event, Query
from aic2026.core.pipeline import Pipeline
from aic2026.core.providers import HashEmbedder
from aic2026.core.providers import (CloudflareVLMProvider,
                                    FailoverVLMProvider, GeminiVLMProvider,
                                    GroqVLMProvider,
                                    build_llm)
from aic2026.data_plane.image_resolver import (
    EventImageResolver, ImageResolutionError, parse_event_id)


class FakeVLM(CachedImageLLMProvider):
    def __init__(self, cache_path, raw="xanh dương.\nKhông giải thích",
                 prompt_fingerprint=None, normalizer_version=None):
        self.raw = raw
        self.requests = 0
        kwargs = {}
        if prompt_fingerprint is not None:
            kwargs["prompt_fingerprint"] = prompt_fingerprint
        if normalizer_version is not None:
            kwargs["normalizer_version"] = normalizer_version
        super().__init__(cache_path, model="fake-vlm", **kwargs)

    def _request_image(self, question: str, image_path: Path,
                       context: str = "") -> str:
        self.requests += 1
        return self.raw


class PathVLM(FakeVLM):
    def _request_image(self, question: str, image_path: Path,
                       context: str = "") -> str:
        self.requests += 1
        return image_path.stem


class BrokenVLM(FakeVLM):
    def _request_image(self, question: str, image_path: Path,
                       context: str = "") -> str:
        raise RuntimeError("Gemini API HTTP 503: high demand")


class StaticVisualQA:
    def tra_loi(self, question, candidates, lay_vector):
        return [("dự phòng", 1.0)]


def test_parse_event_id_giu_ordinal():
    assert parse_event_id("L22_V006#00209") == ("L22_V006", 209)
    assert parse_event_id("L22_V006#e00209") == ("L22_V006", 209)
    with pytest.raises(ValueError):
        parse_event_id("L22_V006")


def test_resolver_doc_dung_ten_anh_dev(tmp_path):
    image = tmp_path / "001_L22_V006_n209.jpg"
    image.write_bytes(b"jpg")
    resolver = EventImageResolver(dev_image_dir=tmp_path)
    assert resolver.resolve("L22_V006#00209") == image.resolve()


def test_resolver_doc_dung_quy_uoc_that_cua_corpus(tmp_path):
    """Quy ước THẬT của keyframe BTC, đo 2026-08-16 (việc 2.3).

    Corpus đặt `<root>/<VIDEO>/001.jpg` — số thứ tự 3 chữ số, KHÔNG phải 5 như
    `event_id`. Đã đối chiếu byte 4 ảnh dev set với ảnh corpus cùng `(video, n)`:
    giống hệt, nên `n` trong `VIDEO#00007` đúng là số thứ tự file ảnh.
    Xem `kiem_resolver_anh.py`.
    """
    image = tmp_path / "L21_V001" / "007.jpg"
    image.parent.mkdir()
    image.write_bytes(b"jpg")
    resolver = EventImageResolver(roots=[tmp_path])
    assert resolver.resolve("L21_V001#00007") == image.resolve()


def test_resolver_doc_layout_corpus_va_thieu_thi_no(tmp_path):
    image = tmp_path / "L22_V006" / "00209.jpg"
    image.parent.mkdir()
    image.write_bytes(b"jpg")
    resolver = EventImageResolver(roots=[tmp_path])
    assert resolver.resolve("L22_V006#00209") == image.resolve()
    with pytest.raises(ImageResolutionError, match="Không tìm thấy ảnh"):
        resolver.resolve("L22_V999#00001")


def test_vlm_cache_theo_event_va_cau_hoi_khong_goi_lai(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"not-a-real-image-needed-by-fake")
    cache = tmp_path / "cache.json"

    first = FakeVLM(cache)
    answer = first.answer_image("Ngọn lửa màu gì?", str(image), "E#1")
    assert answer == "xanh dương"
    assert first.requests == 1 and first.network_calls == 1

    # Provider mới mô phỏng lần chạy thứ hai: đọc cache đĩa, không gọi backend.
    second = FakeVLM(cache, raw="sai")
    assert second.answer_image("Ngọn lửa màu gì?", str(image), "E#1") == answer
    assert second.requests == 0 and second.cache_hits == 1

    # Cùng event nhưng câu hỏi khác phải là một entry khác.
    second.answer_image("Có bao nhiêu người?", str(image), "E#1")
    assert second.requests == 1
    assert second.peek_cached_answer("Ngọn lửa màu gì?", "E#1") == answer
    assert second.peek_cached_answer("Câu chưa có?", "E#1") is None


def test_vlm_cache_tu_vo_hieu_khi_prompt_doi(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpg")
    cache = tmp_path / "cache.json"
    old = FakeVLM(cache, raw="đỏ", prompt_fingerprint="prompt-v1")
    assert old.answer_image("Màu gì?", str(image), "E#1") == "đỏ"

    new = FakeVLM(cache, raw="xanh dương", prompt_fingerprint="prompt-v2")
    assert new.answer_image("Màu gì?", str(image), "E#1") == "xanh dương"
    assert new.requests == 1 and new.cache_hits == 0


def test_short_answer_giu_cum_tu_va_chuan_hoa_theo_cau_hoi():
    clean = CachedImageLLMProvider._short_answer
    assert clean("Cây đánh trứng.", "Dụng cụ là gì?") == "Cây đánh trứng"
    assert clean("Hai người", "Có bao nhiêu người?") == "2"
    assert clean("Áo màu xanh dương.", "Áo màu gì?") == "xanh dương"
    assert clean("Xám và", "Áo màu gì?") == "xám"


def test_vlm_khong_co_anh_thi_khong_goi_backend(tmp_path):
    provider = FakeVLM(tmp_path / "cache.json")
    with pytest.raises(FileNotFoundError):
        provider.answer_image("Màu gì?", str(tmp_path / "missing.jpg"), "E#1")
    assert provider.requests == 0


def test_pipeline_vlm_tra_loi_rieng_tung_candidate(tmp_path):
    images = {}
    events = []
    for i, name in enumerate(("do", "xanh"), start=1):
        event_id = f"L01_V001#{i:05d}"
        image = tmp_path / f"{name}.jpg"
        image.write_bytes(b"fake")
        images[event_id] = image
        events.append(Event(
            video_id="L01_V001", event_id=event_id,
            t_start=float(i), t_end=float(i + 1),
            start_frame=i * 10, end_frame=i * 10 + 9,
            keyframe_idx=i * 10, visual_text=f"canh {name}"))
    resolver = EventImageResolver(explicit=images)
    provider = PathVLM(tmp_path / "pipeline-cache.json")
    pipeline = Pipeline(HashEmbedder(), provider, image_resolver=resolver)
    pipeline.index(events)

    rows = pipeline.answer(Query(qid="qa", task="qa", text="canh, màu gì?",
                                 question_vi="Màu gì?"), top_k=2)
    assert {row["answer"] for row in rows} == {"do", "xanh"}
    assert provider.requests == 2


def test_pipeline_vlm_top_n_va_zero_shot_cho_duoi(tmp_path):
    images, events = {}, []
    for i in range(1, 13):
        event_id = f"L01_V001#{i:05d}"
        image = tmp_path / f"vlm-{i}.jpg"
        image.write_bytes(b"fake")
        images[event_id] = image
        events.append(Event(
            video_id="L01_V001", event_id=event_id,
            t_start=float(i), t_end=float(i + 1),
            start_frame=i, end_frame=i, keyframe_idx=i,
            visual_text=f"cảnh {i}"))
    provider = PathVLM(tmp_path / "pipeline-cache.json")
    pipeline = Pipeline(
        HashEmbedder(), provider, visual_qa=StaticVisualQA(),
        image_resolver=EventImageResolver(explicit=images), qa_vlm_top_n=3,
        qa_tail_answer="cheap")
    pipeline.index(events)

    rows = pipeline.answer(Query(qid="qa", task="qa", text="cảnh, vật gì?"),
                           top_k=12)
    assert provider.requests == 3
    assert len(rows) == 12
    assert sum(row["answer"] == "dự phòng" for row in rows) == 9


def test_duoi_mac_dinh_lan_dap_an_vlm_khong_ton_request(tmp_path):
    """Mặc định `qa_tail_answer='vlm'` (chốt 16/08): đuôi dùng đáp án VLM phổ biến
    nhất thay cho đáp án rẻ, và KHÔNG gọi thêm request nào.

    Đo ghép cặp 30 câu dev: 0.1933 (đuôi rẻ) -> 0.2067 (lan đáp án VLM),
    +0.0133 ± 0.0093. Cơ chế: đáp án VLM đúng 26/30 còn đáp án rẻ ~13/30.
    """
    images, events = {}, []
    for i in range(1, 13):
        event_id = f"L01_V001#{i:05d}"
        image = tmp_path / f"{'do' if i <= 2 else 'xanh'}-{i}.jpg"
        image.write_bytes(b"fake")
        images[event_id] = image
        events.append(Event(
            video_id="L01_V001", event_id=event_id,
            t_start=float(i), t_end=float(i + 1),
            start_frame=i, end_frame=i, keyframe_idx=i,
            visual_text=f"cảnh {i}"))

    class MotDapAn(FakeVLM):
        """Trả 'đỏ' cho hai ảnh đầu, 'xanh' cho phần còn lại -> đa số là 'xanh'."""

        def _request_image(self, question, image_path, context=""):
            self.requests += 1
            return "đỏ" if image_path.name.startswith("do-") else "xanh"

    provider = MotDapAn(tmp_path / "cache.json")
    pipeline = Pipeline(
        HashEmbedder(), provider, visual_qa=StaticVisualQA(),
        image_resolver=EventImageResolver(explicit=images), qa_vlm_top_n=3)
    pipeline.index(events)
    rows = pipeline.answer(Query(qid="qa", task="qa", text="cảnh, vật gì?"),
                           top_k=12)
    assert provider.requests == 3, "đuôi không được gọi thêm request"
    assert len(rows) == 12
    assert all(row["answer"] != "dự phòng" for row in rows), \
        "đuôi phải dùng đáp án VLM, không phải đáp án rẻ"
    duoi = {row["answer"] for row in rows[3:]}
    assert len(duoi) == 1 and duoi <= {"đỏ", "xanh"}


def test_qa_tail_answer_la_hoac_vlm_hoac_cheap():
    with pytest.raises(ValueError, match="qa_tail_answer"):
        Pipeline(HashEmbedder(), FakeVLM("/tmp/khong-dung.json"),
                 qa_tail_answer="bậy")


def test_pipeline_cau_dem_khong_goi_vlm_mac_dinh(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake")
    event_id = "L01_V001#00001"
    provider = PathVLM(tmp_path / "pipeline-cache.json")
    pipeline = Pipeline(
        HashEmbedder(), provider, visual_qa=StaticVisualQA(),
        image_resolver=EventImageResolver(explicit={event_id: image}))
    pipeline.index([Event(
        video_id="L01_V001", event_id=event_id, t_start=0, t_end=1,
        start_frame=1, end_frame=1, keyframe_idx=1, visual_text="hai người")])

    rows = pipeline.answer(Query(
        qid="qa", task="qa", text="people", question_vi="Có bao nhiêu người?"),
        top_k=1)
    assert provider.requests == 0
    assert rows[0]["answer"] == "dự phòng"


def test_pipeline_vlm_bat_buoc_co_resolver(tmp_path):
    provider = FakeVLM(tmp_path / "cache.json")
    pipeline = Pipeline(HashEmbedder(), provider)
    pipeline.index([Event(
        video_id="L01", event_id="L01#00001", t_start=0, t_end=1,
        start_frame=0, end_frame=1, keyframe_idx=0, visual_text="canh")])
    with pytest.raises(RuntimeError, match="image_resolver"):
        pipeline.answer(Query(qid="qa", task="qa", text="màu gì?"), top_k=1)


def test_pipeline_vlm_loi_mang_ha_xuong_zero_shot(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpg")

    class Resolver:
        def resolve(self, event_id):
            return image

    provider = BrokenVLM(tmp_path / "cache.json")
    pipeline = Pipeline(HashEmbedder(), provider, visual_qa=StaticVisualQA(),
                        image_resolver=Resolver())
    pipeline.index([Event(
        video_id="L01", event_id="L01#00001", t_start=0, t_end=1,
        start_frame=0, end_frame=1, keyframe_idx=0, visual_text="cảnh")])
    with pytest.warns(RuntimeWarning, match="ClipZeroShotQA"):
        rows = pipeline.answer(Query(qid="qa", task="qa", text="màu gì?"),
                               top_k=1)
    assert rows[0]["answer"] == "dự phòng"
    assert "503" in pipeline.last_trace.vlm_error


# --------------------------------------------------------------------------- #
# Thiếu ảnh corpus — kịch bản duy nhất còn lại có thể làm QA = 0 vào ngày thi.
# --------------------------------------------------------------------------- #
class ResolverThieuAnh:
    """Corpus thật lúc thiếu file: ném ImageResolutionError (là OSError)."""

    def __init__(self, co_san=None):
        self.co_san = dict(co_san or {})

    def resolve(self, event_id):
        if event_id in self.co_san:
            return self.co_san[event_id]
        raise ImageResolutionError(f"Không tìm thấy ảnh cho {event_id}")


class ThieuKeyVLM(FakeVLM):
    def _request_image(self, question: str, image_path: Path,
                       context: str = "") -> str:
        raise RuntimeError("Thiếu GEMINI_API_KEY (hoặc GOOGLE_API_KEY).")


def _event_qa(i: int = 1) -> Event:
    return Event(video_id="L01_V001", event_id=f"L01_V001#{i:05d}",
                 t_start=float(i), t_end=float(i + 1),
                 start_frame=i, end_frame=i, keyframe_idx=i,
                 visual_text=f"cảnh {i}")


def test_pipeline_thieu_anh_ha_xuong_zero_shot_chu_khong_ne_loi(tmp_path):
    """HỒI QUY: ImageResolutionError kế thừa FileNotFoundError -> OSError.

    Trước khi sửa, `resolve()` nằm NGOÀI khối `try` và `except` chỉ bắt
    `RuntimeError`, nên thiếu đúng một file ảnh là cả đường QA ném lỗi và điểm
    QA về 0 — dù `vlm_fallback_to_zero_shot` đang bật. 222 test cũ không bắt
    được vì chỉ test resolver đứng một mình, không test qua pipeline.
    """
    provider = PathVLM(tmp_path / "cache.json")
    pipeline = Pipeline(HashEmbedder(), provider, visual_qa=StaticVisualQA(),
                        image_resolver=ResolverThieuAnh())
    pipeline.index([_event_qa()])

    with pytest.warns(RuntimeWarning, match="THIẾU ẢNH"):
        rows = pipeline.answer(Query(qid="qa", task="qa", text="màu gì?"),
                               top_k=1)
    assert rows[0]["answer"] == "dự phòng"
    assert provider.requests == 0          # không được đốt quota khi thiếu ảnh
    assert "Không tìm thấy ảnh" in pipeline.last_trace.vlm_error


def test_pipeline_thieu_mot_anh_giua_top_n_van_du_dong(tmp_path):
    """Kịch bản thật: corpus có đủ ảnh trừ vài file. Vẫn phải đủ số dòng nộp."""
    anh = tmp_path / "co-anh.jpg"
    anh.write_bytes(b"jpg")
    events = [_event_qa(i) for i in range(1, 4)]
    resolver = ResolverThieuAnh({"L01_V001#00001": anh})  # chỉ 1/3 có ảnh
    provider = PathVLM(tmp_path / "cache.json")
    pipeline = Pipeline(HashEmbedder(), provider, visual_qa=StaticVisualQA(),
                        image_resolver=resolver)
    pipeline.index(events)

    with pytest.warns(RuntimeWarning, match="THIẾU ẢNH"):
        rows = pipeline.answer(Query(qid="qa", task="qa", text="màu gì?"),
                               top_k=3)
    assert len(rows) == 3
    assert all(row["answer"] == "dự phòng" for row in rows)


def test_pipeline_thieu_anh_van_ne_loi_khi_tat_fallback(tmp_path):
    """Vẫn giữ được đường 'lộ ngay lúc chuẩn bị' bằng cách tắt cờ."""
    provider = PathVLM(tmp_path / "cache.json")
    pipeline = Pipeline(HashEmbedder(), provider, visual_qa=StaticVisualQA(),
                        image_resolver=ResolverThieuAnh(),
                        vlm_fallback_to_zero_shot=False)
    pipeline.index([_event_qa()])

    with pytest.raises(ImageResolutionError, match="Không tìm thấy ảnh"):
        pipeline.answer(Query(qid="qa", task="qa", text="màu gì?"), top_k=1)


def test_pipeline_thieu_api_key_van_ne_loi_du_bat_fallback(tmp_path):
    """Thiếu key là lỗi CẤU HÌNH, không phải sự cố runtime.

    Nếu nó âm thầm hạ xuống cheap thì cả buổi thi sẽ chạy QA cheap mà tưởng
    đang chạy VLM — sai lệch này không nhìn thấy được từ bảng điểm.
    """
    anh = tmp_path / "frame.jpg"
    anh.write_bytes(b"jpg")
    provider = ThieuKeyVLM(tmp_path / "cache.json")
    pipeline = Pipeline(HashEmbedder(), provider, visual_qa=StaticVisualQA(),
                        image_resolver=ResolverThieuAnh(
                            {"L01_V001#00001": anh}))
    pipeline.index([_event_qa()])

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        pipeline.answer(Query(qid="qa", task="qa", text="màu gì?"), top_k=1)


def test_build_gemini_vlm_native_sdk(tmp_path):
    provider = build_llm({"llm": {
        "provider": "gemini_vlm", "model": "gemini-test",
        "cache_path": str(tmp_path / "cache.json"), "max_retries": 0,
    }})
    assert isinstance(provider, GeminiVLMProvider)
    assert provider.model == "gemini-test"
    assert provider.max_tokens == 64


def test_gemini_vlm_native_interactions_khong_goi_mang_that(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake-jpeg")
    seen = {}

    class Usage:
        def model_dump(self, exclude_none=True):
            return {"total_input_tokens": 10}

    class Interaction:
        output_text = "xanh lá."
        usage = Usage()

    class Interactions:
        def create(self, **kwargs):
            seen.update(kwargs)
            return Interaction()

    class Client:
        interactions = Interactions()

    provider = GeminiVLMProvider(tmp_path / "cache.json", model="gemini-test",
                                 api_key="dummy-test-key", max_retries=0,
                                 client=Client())
    answer = provider.answer_image("Áo màu gì?", str(image), "L01#00001")
    assert answer == "xanh lá"
    assert seen["model"] == "gemini-test"
    assert seen["input"][0]["type"] == "image"
    assert seen["input"][0]["data"]
    assert "Áo màu gì?" in seen["input"][1]["text"]
    assert provider.last_usage["total_input_tokens"] == 10


def test_gemini_native_cache_hai_anh_khong_goi_lai(tmp_path):
    images = []
    for index in range(2):
        image = tmp_path / f"frame-{index}.jpg"
        image.write_bytes(f"fake-jpeg-{index}".encode())
        images.append(image)
    calls, outputs = [], iter(("đỏ", "2"))

    class Interaction:
        usage = None

        def __init__(self, text):
            self.output_text = text

    class Interactions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return Interaction(next(outputs))

    class Client:
        interactions = Interactions()

    cache = tmp_path / "batch-cache.json"
    requests = [
        ("Màu gì?", str(images[0]), "L01#00001", ""),
        ("Có bao nhiêu?", str(images[1]), "L01#00002", ""),
    ]
    provider = GeminiVLMProvider(cache, model="gemini-test",
                                 api_key="dummy-test-key", max_retries=0,
                                 client=Client())
    assert provider.answer_images(requests) == ["đỏ", "2"]
    assert len(calls) == 2
    assert provider.backend_requests == 2
    assert provider.network_calls == 2

    cached = GeminiVLMProvider(cache, model="gemini-test",
                               api_key="dummy-test-key", max_retries=0,
                               client=Client())
    assert cached.answer_images(requests) == ["đỏ", "2"]
    assert len(calls) == 2
    assert cached.cache_hits == 2
    assert cached.backend_requests == 0


def test_gemini_native_batch_cache_tung_anh_mot_request(tmp_path):
    import json

    paths = []
    for index in range(2):
        path = tmp_path / f"frame-{index}.jpg"
        path.write_bytes(b"jpg")
        paths.append(path)
    calls = []

    class Interaction:
        usage = None
        output_text = json.dumps([
            {"id": "q000", "answer": "đỏ"},
            {"id": "q001", "answer": "2"},
        ], ensure_ascii=False)

    class Interactions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return Interaction()

    class Client:
        interactions = Interactions()

    cache = tmp_path / "native-batch.json"
    requests = [
        ("Màu gì?", str(paths[0]), "E#1", ""),
        ("Bao nhiêu?", str(paths[1]), "E#2", ""),
    ]
    provider = GeminiVLMProvider(
        cache, client=Client(), api_key="dummy", batch_max_images=2,
        max_retries=0)
    assert provider.answer_images(requests) == ["đỏ", "2"]
    assert len(calls) == 1
    assert provider.backend_requests == 1
    assert provider.network_calls == 2

    cached = GeminiVLMProvider(
        cache, client=Client(), api_key="dummy", batch_max_images=2,
        max_retries=0)
    assert cached.answer_images(requests) == ["đỏ", "2"]
    assert len(calls) == 1
    assert cached.cache_hits == 2


def test_gemini_native_watchdog_khong_treo_pipeline(tmp_path):
    import time

    class Interactions:
        def create(self, **kwargs):
            time.sleep(0.2)

    class Client:
        interactions = Interactions()

    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpg")
    provider = GeminiVLMProvider(
        tmp_path / "cache.json", client=Client(), api_key="dummy",
        timeout_seconds=0.02, max_retries=0)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="quá hạn"):
        provider.answer_image("Màu gì?", str(image), "E#1")
    assert time.monotonic() - started < 0.15


def test_gemini_doc_retry_after_tu_loi_sdk():
    error = RuntimeError("quota exceeded; Please retry in 48.62187224s.")
    assert GeminiVLMProvider._retry_delay(error, 0) == pytest.approx(65.0)


def test_gemini_retry_nhan_ma_loi_moi_cua_interactions_api():
    assert GeminiVLMProvider._retryable(
        RuntimeError("429 rate_limit_exceeded"))
    assert GeminiVLMProvider._retryable(
        RuntimeError('RESOURCE_EXHAUSTED retryDelay: "48.6s"'))
    error = RuntimeError('RESOURCE_EXHAUSTED retryDelay: "48.6s"')
    assert GeminiVLMProvider._retry_delay(error, 0) == pytest.approx(65.0)


def test_cloudflare_vlm_gui_anh_data_uri_va_cache(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpg-bytes")
    seen = {}

    def requester(url, headers, payload, timeout):
        seen.update(url=url, headers=headers, payload=payload, timeout=timeout)
        return {"success": True, "result": {"response": "xanh dương."}}

    provider = CloudflareVLMProvider(
        tmp_path / "cloudflare.json", api_token="dummy-token", account_id="abc",
        requester=requester, max_tokens=64)
    assert provider.answer_image("Màu gì?", str(image), "E#1") == "xanh dương"
    assert "/accounts/abc/ai/run/@cf/meta/llama-3.2-11b-vision-instruct" in seen["url"]
    assert seen["headers"]["Authorization"] == "Bearer dummy-token"
    assert seen["payload"]["image"].startswith("data:image/jpeg;base64,")
    assert seen["payload"]["max_tokens"] == 64
    assert provider.answer_image("Màu gì?", str(image), "E#1") == "xanh dương"
    assert provider.backend_requests == 1


def test_groq_vlm_gui_anh_data_uri_va_cache(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpg-bytes")
    seen = {}

    def requester(url, headers, payload, timeout):
        seen.update(url=url, headers=headers, payload=payload, timeout=timeout)
        return {"choices": [{"message": {"content": "xanh lá."}}]}

    provider = GroqVLMProvider(
        tmp_path / "groq.json", api_key="dummy-groq", requester=requester)
    assert provider.answer_image("Màu gì?", str(image), "E#1") == "xanh lá"
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["headers"]["Authorization"] == "Bearer dummy-groq"
    assert seen["headers"]["Accept"] == "application/json"
    assert seen["headers"]["User-Agent"] == "aic2026-groq-api/1.0"
    content = seen["payload"]["messages"][0]["content"]
    assert content[1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,")
    assert seen["payload"]["model"] == "qwen/qwen3.6-27b"
    assert provider.answer_image("Màu gì?", str(image), "E#1") == "xanh lá"
    assert provider.backend_requests == 1


def test_groq_vlm_thieu_key_bao_ro(monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    provider = GroqVLMProvider(tmp_path / "groq.json")
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        provider._key()


def test_build_failover_gemini_groq_cloudflare(tmp_path):
    provider = build_llm({"llm": {
        "provider": "failover_vlm",
        "primary": {"provider": "gemini_vlm",
                    "cache_path": str(tmp_path / "gemini.json")},
        "fallbacks": [
            {"provider": "groq_vlm",
             "cache_path": str(tmp_path / "groq.json")},
            {"provider": "cloudflare_vlm",
             "cache_path": str(tmp_path / "cloudflare.json")},
        ],
    }})
    assert [type(item) for item in provider.providers] == [
        GeminiVLMProvider, GroqVLMProvider, CloudflareVLMProvider]


def test_cloudflare_vlm_thieu_bien_moi_truong_bao_ro(monkeypatch, tmp_path):
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    provider = CloudflareVLMProvider(tmp_path / "cache.json")
    with pytest.raises(RuntimeError, match="CLOUDFLARE_API_TOKEN"):
        provider._credentials()


def test_cloudflare_agree_5016_duoc_xem_la_da_thanh_cong(tmp_path, monkeypatch):
    provider = CloudflareVLMProvider(tmp_path / "cache.json",
                                     api_token="dummy", account_id="abc")

    def acknowledged(_payload):
        raise RuntimeError('Cloudflare HTTP 403: {"code":5016, '
                           '"message":"You may now use the model"}')

    monkeypatch.setattr(provider, "_post", acknowledged)
    assert "sẵn sàng" in provider.agree_to_model_license()


def test_failover_bo_provider_loi_cho_phan_con_lai_cua_batch(tmp_path):
    images = []
    requests = []
    for index, name in enumerate(("đỏ", "xanh"), start=1):
        path = tmp_path / f"{name}.jpg"
        path.write_bytes(b"jpg")
        images.append(path)
        requests.append(("Màu gì?", str(path), f"E#{index}", ""))
    broken = BrokenVLM(tmp_path / "broken.json")
    backup = PathVLM(tmp_path / "backup.json")
    provider = FailoverVLMProvider([broken, backup])
    assert provider.answer_images(requests) == ["đỏ", "xanh"]
    assert len(provider.failures) == 1
    assert backup.requests == 2


def test_build_failover_gemini_roi_cloudflare(tmp_path):
    """Chuỗi đã chốt 16/08: Gemini chính → Cloudflare dự phòng. Grok đã bỏ hẳn."""
    provider = build_llm({"llm": {
        "provider": "failover_vlm",
        "primary": {"provider": "gemini_vlm",
                    "cache_path": str(tmp_path / "gemini.json")},
        "fallbacks": [{"provider": "cloudflare_vlm",
                       "cache_path": str(tmp_path / "cloudflare.json")}],
    }})
    assert isinstance(provider, FailoverVLMProvider)
    assert isinstance(provider.providers[0], GeminiVLMProvider)
    assert isinstance(provider.providers[1], CloudflareVLMProvider)


def test_grok_da_bo_khoi_he_thong(tmp_path):
    """Bỏ Grok 16/08 (quyết định của người dùng): không còn provider, không còn
    nhánh factory. Test này chặn việc thêm lại một cách im lặng."""
    import aic2026.core.providers as providers_module
    assert not hasattr(providers_module, "GrokVLMProvider")
    with pytest.raises(ValueError, match="grok_vlm"):
        build_llm({"llm": {"provider": "grok_vlm",
                           "cache_path": str(tmp_path / "x.json")}})


def test_build_failover_bo_qua_provider_disabled(tmp_path):
    provider = build_llm({"llm": {
        "provider": "failover_vlm",
        "primary": {"enabled": True, "provider": "gemini_vlm",
                    "cache_path": str(tmp_path / "gemini.json")},
        "fallbacks": [
            {"enabled": False, "provider": "anthropic_vlm",
             "cache_path": str(tmp_path / "anthropic.json")},
            {"enabled": False, "provider": "cloudflare_vlm",
             "cache_path": str(tmp_path / "cloudflare.json")},
        ],
    }})
    assert len(provider.providers) == 1
    assert isinstance(provider.providers[0], GeminiVLMProvider)


def test_gemini_vlm_thieu_key_bao_ro(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    provider = GeminiVLMProvider(tmp_path / "cache.json")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        provider._key()


# --------------------------------------------------------------------------- #
# Chốt phiên bản SDK (bắt được trên Kaggle 19/08)
# --------------------------------------------------------------------------- #
def test_sdk_google_genai_qua_cu_thi_bao_ro(monkeypatch):
    """Kaggle cài sẵn bản < 2.0.0 -> server trả 400 'legacy Interactions API'.
    Lỗi đó đọc như lỗi quota; phải chặn sớm và nói thẳng cách sửa."""
    from aic2026.core import providers

    monkeypatch.setattr(providers.md if hasattr(providers, "md") else
                        __import__("importlib.metadata", fromlist=["version"]),
                        "version", lambda _: "1.44.0", raising=False)
    import importlib.metadata as md
    monkeypatch.setattr(md, "version", lambda _: "1.44.0")
    with pytest.raises(RuntimeError, match="quá cũ"):
        providers._chan_sdk_cu()


def test_sdk_google_genai_du_moi_thi_qua(monkeypatch):
    from aic2026.core import providers
    import importlib.metadata as md
    monkeypatch.setattr(md, "version", lambda _: "2.18.0")
    providers._chan_sdk_cu()          # không được ném


def test_qa_zero_shot_uu_tien_question_vi_khong_phai_query_text():
    """Bắt được trên Kaggle 19/08: bật lớp viết lại truy vấn -> `query.text`
    thành tiếng Anh -> luật sinh ứng viên (khớp từ khoá TIẾNG VIỆT) trượt sạch
    -> 18/30 câu QA trả 'nhà bếp'. Câu hỏi thật nằm ở `question_vi`, và đường
    VLM đã ưu tiên nó từ trước; đường zero-shot phải giống hệt."""
    from aic2026.core.qa import SinhTheoLuat

    cau_hoi_vi = "Người đàn ông đang mặc áo màu gì?"
    text_da_viet_lai = "A chef and a woman converse inside a modern kitchen"

    s = SinhTheoLuat()
    assert s.sinh(cau_hoi_vi), "câu hỏi tiếng Việt phải sinh được ứng viên màu"
    assert "red" in s.sinh(cau_hoi_vi)
    # Chính là chỗ hỏng: mô tả cảnh tiếng Anh không sinh được gì.
    assert not s.sinh(text_da_viet_lai)


def test_pipeline_lay_cau_hoi_theo_dung_thu_tu_uu_tien():
    """Khoá thứ tự `question_vi > compiled.question > text` trong source."""
    import inspect

    from aic2026.core import pipeline as pl

    src = inspect.getsource(pl.Pipeline)
    assert "query.question_vi or compiled.question or query.text" in src, (
        "đường QA zero-shot phải ưu tiên `question_vi`; xem lỗi 'nhà bếp' 19/08")
