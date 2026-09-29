import json

import numpy as np
import pytest

from aic2026.control import compile_query, decide
from aic2026.core.models import Candidate, Event
from aic2026.core.models import Query
from aic2026.core.pipeline import Pipeline
from aic2026.core.providers import (
    CLIPTextEncoder,
    RuleBasedProvider,
    VisualEncoder,
)
from aic2026.core.retrieval import MultiChannelRetriever
from aic2026.core.dich import Translator
from aic2026.core.stores import NumpyVectorStore
from aic2026.data_plane import (
    CanonicalManifest,
    FrameRecord,
    VideoRecord,
    build_btc_manifest,
)
from aic2026.reasoning import FullTimelineTrakeSolver, TrakeConfig
from aic2026.retrieval import (
    PersistentVectorStore,
    RetrievalPolicy,
    RetrievalService,
    write_bundle_manifest,
)
from do_diem_thuc_te import (
    optimizer_recommendation,
    paired_delta_se,
)


def event(video, key, time, frame, vector):
    return Event(video_id=video, event_id=key, t_start=time,
                 t_end=time + 1, start_frame=frame, end_frame=frame,
                 keyframe_idx=frame, embedding=vector)


def test_manifest_roundtrip_and_duplicate_frame_is_explicit_warning(tmp_path):
    manifest = CanonicalManifest(
        videos=[VideoRecord("V1", time_base_num=1, time_base_den=1000)],
        frames=[
            FrameRecord("V1", 10, 0.0, "btc", "s0", pts=0,
                        embedding_ref="e", embedding_row=0),
            FrameRecord("V1", 10, 1000.0, "btc", "s1", pts=1000,
                        embedding_ref="e", embedding_row=1),
        ],
    )
    report = manifest.validate()
    assert not report.errors
    assert [x.code for x in report.warnings] == ["DUP_FRAME_ID"]
    manifest.save(tmp_path)
    loaded = CanonicalManifest.load(tmp_path)
    assert loaded.frames == manifest.frames
    assert loaded.frames[0].frame_uid == loaded.frames[1].frame_uid
    assert loaded.frames[0].sample_id != loaded.frames[1].sample_id


def test_manifest_does_not_invent_pts():
    manifest = CanonicalManifest(
        videos=[VideoRecord("V1", fps=25.0)],
        frames=[FrameRecord("V1", 25, 1000.0, "btc", "s0")])
    report = manifest.validate()
    assert [x.code for x in report.warnings] == ["PTS_UNKNOWN"]
    assert manifest.frames[0].pts is None


def test_btc_manifest_rejects_mapping_embedding_row_mismatch(tmp_path):
    maps = tmp_path / "map-keyframes"
    clips = tmp_path / "clip-features-32"
    maps.mkdir()
    clips.mkdir()
    (maps / "L01_V001.csv").write_text(
        "n,pts_time,fps,frame_idx\n"
        "1,0.0,25,0\n2,1.0,25,25\n",
        encoding="utf-8")
    np.save(clips / "L01_V001.npy",
            np.zeros((1, 512), dtype=np.float32))
    with pytest.raises(ValueError, match="mapping rows 2"):
        build_btc_manifest(tmp_path)


def test_compiler_types_trake_events_and_qa_answer():
    trake = compile_query(
        "trake", "",
        ["đầu tiên người chạy", "tay chạm quả bóng",
         "quả bóng ở điểm cao nhất"])
    assert [x.semantic_type for x in trake.events] == [
        "FIRST_TRANSITION", "FIRST_CONTACT", "APEX"]
    qa = compile_query("qa", "Có bao nhiêu người đứng cạnh biển chữ AIC?")
    assert qa.answer_type == "number"
    assert qa.ocr_queries


def test_qa_split_only_cuts_real_question_and_keeps_visual_anchor():
    good = compile_query(
        "qa", "Trong cảnh phỏng vấn, người mặc áo màu gì?")
    assert good.retrieval_query == "Trong cảnh phỏng vấn người mặc áo"
    assert good.question == "người mặc áo màu gì"

    description = (
        "Người phụ nữ đeo kính, mặc áo sơ mi, "
        "và đeo cà vạt đang nói trong nhà")
    not_question = compile_query("qa", description)
    assert not_question.retrieval_query == description
    assert not_question.question == description

    reversed_question = (
        "Màu áo của người đang nói là gì, trong cảnh phỏng vấn?")
    reversed_result = compile_query("qa", reversed_question)
    expected = reversed_question.rstrip(" ?")
    assert reversed_result.retrieval_query == expected
    assert reversed_result.question == expected


def test_confidence_gate_uses_relative_evidence_not_raw_score():
    one = event("V", "a", 0, 0, [1, 0])
    rows = [
        Candidate(one, 0.02, {"visual": 0.1, "ocr": 0.1}),
        Candidate(one, 0.01, {"visual": 0.09}),
    ]
    assert decide(rows, neighbor_support=0.5).route == "fast"
    assert decide(rows, neighbor_support=0.0).route == "escalate"


def test_gate_can_be_fast_with_visual_only_index():
    one = event("V", "a", 0, 0, [1, 0])
    two = event("V", "b", 1, 1, [0.5, 0.5])
    rows = [
        Candidate(one, 1 / 61, {"visual": 1.0, "ocr": 0.0, "asr": 0.0}),
        Candidate(two, 1 / 62, {"visual": 0.7, "ocr": 0.0, "asr": 0.0}),
    ]
    decision = decide(rows, neighbor_support=1.0)
    assert decision.route == "fast"
    assert "weak_channel_agreement" not in decision.reasons


def test_gate_rejects_negative_or_missing_evidence_without_margin_explosion():
    one = event("V", "a", 0, 0, [1, 0])
    negative = [
        Candidate(one, -0.01, {"visual": -0.01}),
        Candidate(one, -0.05, {"visual": -0.05}),
    ]
    decision = decide(negative, neighbor_support=1.0)
    assert decision.route == "escalate"
    assert decision.confidence == 0.0
    assert "nonpositive_top_evidence" in decision.reasons

    missing = [
        Candidate(one, 0.2, {}),
        Candidate(one, 0.1, {}),
    ]
    decision = decide(missing, neighbor_support=1.0)
    assert decision.route == "escalate"
    assert decision.confidence == 0.0
    assert "no_active_channels" in decision.reasons


def test_optimizer_advice_is_gated_by_paired_standard_error():
    a = [("q1", "", 0.0), ("q2", "", 0.2), ("q3", "", 0.4)]
    b = [("q1", "", 0.1), ("q2", "", 0.1), ("q3", "", 0.5)]
    delta, se = paired_delta_se(a, b)
    assert delta == pytest.approx(1 / 30)
    assert se > 0
    assert "Giữ PortfolioConfig.enabled=False" in (
        optimizer_recommendation(0.0067, 0.025))
    assert "validation split" in optimizer_recommendation(0.06, 0.02)


def test_persistent_bundle_is_fail_closed(tmp_path):
    np.save(tmp_path / "embeddings.npy",
            np.asarray([[1, 0], [0, 1]], dtype=np.float32))
    metadata = tmp_path / "frame_metadata.jsonl"
    metadata.write_text(
        json.dumps({"event_id": "a"}) + "\n"
        + json.dumps({"event_id": "b"}) + "\n",
        encoding="utf-8")
    write_bundle_manifest(tmp_path, encoder_fingerprint="encoder:v1")
    store = PersistentVectorStore(
        tmp_path, expected_encoder_fingerprint="encoder:v1")
    assert store.search([1, 0], 1)[0][0] == "a"
    metadata.write_text('{"event_id":"changed"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        PersistentVectorStore(tmp_path)


class TableEncoder(VisualEncoder):
    def encode_text(self, text):
        return {"a": [1.0, 0.0], "b": [0.0, 1.0]}[text]


def test_trake_hydrates_full_timeline_before_alignment():
    events = [
        event("V1", "v1a", 0, 0, [1.0, 0.0]),
        event("V1", "v1b", 10, 10, [0.1, 0.9]),
        event("V2", "v2a", 0, 0, [0.7, 0.3]),
        event("V2", "v2b", 10, 10, [0.0, 1.0]),
    ]
    retriever = MultiChannelRetriever(TableEncoder())
    retriever.index(events)
    service = RetrievalService(
        retriever, events,
        RetrievalPolicy(pool_per_query=1, candidate_videos=2))
    query = compile_query("trake", "", ["a", "b"])
    solver = FullTimelineTrakeSolver(
        service, TrakeConfig(timeline_top_per_event=10,
                             paths_per_video=2, output_top=10))
    results = solver.solve(query)
    v1 = next(row for row in results if row.video_id == "V1"
              and row.frames == [0, 10])
    # v1b was not in the coarse top-1 pool for query b; it appears only because
    # V1's complete timeline was hydrated and rescored.
    assert v1.steps[1].event.event_id == "v1b"


class CountingEncoder(TableEncoder):
    def __init__(self):
        self.calls = 0

    def encode_text(self, text):
        self.calls += 1
        return super().encode_text(text)


def test_trake_encodes_each_event_once_total():
    encoder = CountingEncoder()
    events = [
        event("V1", "v1a", 0, 0, [1.0, 0.0]),
        event("V1", "v1b", 10, 10, [0.0, 1.0]),
    ]
    retriever = MultiChannelRetriever(encoder)
    retriever.index(events)
    service = RetrievalService(retriever, events)
    query = compile_query("trake", "", ["a", "b"])
    FullTimelineTrakeSolver(service).solve(query)
    assert encoder.calls == 2


def test_numpy_fine_scores_use_same_modality_gap_geometry_as_search():
    matrix = np.asarray([
        [1.0, 0.0, 0.0],
        [0.8, 0.6, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    store = NumpyVectorStore(khu_modality_gap=True)
    store.add_many(["a", "b", "c"], matrix)
    query = [1.0, 0.1, 0.0]
    coarse = store.search(query, 3)
    fine = store.score_many(["a", "b", "c"], query)
    assert [key for key, _ in coarse] == [
        key for key, _ in sorted(
            fine.items(), key=lambda item: -item[1])]


class FakeTranslator(Translator):
    def __init__(self):
        self.calls = []

    def dich(self, texts):
        self.calls.extend(texts)
        return ["red shirt" if text == "áo đỏ" else text for text in texts]


class DummyClip(CLIPTextEncoder):
    def __init__(self):
        self.seen = []

    def encode_text(self, text):
        self.seen.append(text)
        return {
            "red shirt": [1.0, 0.0],
            "other": [0.0, 1.0],
        }.get(text, [0.5, 0.5])


class DummyVietnameseEncoder(VisualEncoder):
    dim = 512

    def __init__(self):
        self.seen = []

    def encode_text(self, text):
        self.seen.append(text)
        return [1.0, 0.0]


def test_clip_pipeline_routes_vietnamese_to_mclip_and_english_to_clip():
    english = DummyClip()
    vietnamese = DummyVietnameseEncoder()
    pipeline = Pipeline(
        english, RuleBasedProvider(), vietnamese_encoder=vietnamese)
    pipeline.index([
        event("V1", "top", 0, 0, [1.0, 0.0]),
        event("V2", "other", 1, 1, [0.0, 1.0]),
    ])

    pipeline.answer(Query("vi", "kis", text="áo đỏ"))
    pipeline.answer(Query("en", "kis", text="red shirt"))

    assert vietnamese.seen == ["áo đỏ"]
    assert "áo đỏ" not in english.seen
    assert "red shirt" in english.seen
    assert pipeline.encoder.route_counts == {"english": 1, "vietnamese": 1}


def test_real_pipeline_uses_compiler_translation_gate_and_verifier():
    translator = FakeTranslator()
    pipeline = Pipeline(
        DummyClip(), RuleBasedProvider(), translator=translator)
    events = [
        event("V1", "top", 0, 0, [1.0, 0.0]),
        event("V1", "near1", 1, 1, [0.5, 0.5]),
        event("V1", "near2", 2, 2, [0.5, 0.5]),
        event("V1", "near3", 3, 3, [0.5, 0.5]),
        event("V2", "other", 0, 0, [0.0, 1.0]),
    ]
    pipeline.index(events)
    rows = pipeline.answer(Query("q", "kis", text="áo đỏ"))
    assert rows[0]["video_id"] == "V1"
    assert translator.calls == ["áo đỏ"]
    assert pipeline.last_trace.compiled_query.retrieval_query == "áo đỏ"
    assert pipeline.last_trace.gate.route == "fast"
    assert pipeline.last_trace.verification.accepted
    assert pipeline.retrieval_service is not None


def test_clip_pipeline_cannot_silently_return_empty_qa_answer():
    pipeline = Pipeline(
        DummyClip(), RuleBasedProvider(), translator=FakeTranslator())
    pipeline.index([
        event("V1", "top", 0, 0, [1.0, 0.0]),
        event("V2", "other", 1, 1, [0.0, 1.0]),
    ])
    rows = pipeline.answer(
        Query("q", "qa", text="What color is the shirt?"))
    assert rows
    assert all(str(row["answer"]).strip() for row in rows)
