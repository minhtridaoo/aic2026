"""Canonical, PTS-aware identities for every video sample.

Unknown timing facts stay None; this module never fabricates a PTS from a
frame_id/fps calculation.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

SOURCES = {"btc", "adaptive", "dense_decode", "raw_probe"}


@dataclass(frozen=True)
class VideoRecord:
    video_id: str
    video_path: str = ""
    fps: Optional[float] = None
    duration_ms: Optional[float] = None
    frame_count: Optional[int] = None
    time_base_num: Optional[int] = None
    time_base_den: Optional[int] = None
    timing_mode: str = "unknown"


@dataclass(frozen=True)
class FrameRecord:
    video_id: str
    frame_id: int
    timestamp_ms: float
    source: str
    sample_id: str
    keyframe_ordinal: Optional[int] = None
    pts: Optional[int] = None
    shot_id: Optional[str] = None
    clip_id: Optional[str] = None
    image_path: str = ""
    embedding_ref: str = ""
    embedding_row: Optional[int] = None
    evidence_status: str = "available"

    @property
    def frame_uid(self) -> str:
        """Canonical output identity. Duplicate samples may share this UID."""
        return f"{self.video_id}:{self.frame_id}"


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    sample_id: str = ""


@dataclass
class ValidationReport:
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [x for x in self.issues if x.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [x for x in self.issues if x.severity == "warning"]

    def raise_for_errors(self) -> None:
        if self.errors:
            detail = "; ".join(f"{x.code}: {x.message}" for x in self.errors[:8])
            raise ValueError(f"canonical manifest invalid: {detail}")


@dataclass
class CanonicalManifest:
    videos: List[VideoRecord]
    frames: List[FrameRecord]
    schema_version: str = "aic2026.canonical.v1"

    def validate(self) -> ValidationReport:
        report = ValidationReport()
        vids = {v.video_id: v for v in self.videos}
        if len(vids) != len(self.videos):
            report.issues.append(ValidationIssue("error", "DUP_VIDEO", "video_id is not unique"))
        samples: set[str] = set()
        embedding_rows: set[tuple[str, int]] = set()
        missing_pts_videos: set[str] = set()
        previous: Dict[str, FrameRecord] = {}
        for row in self.frames:
            if row.video_id not in vids:
                report.issues.append(ValidationIssue("error", "UNKNOWN_VIDEO", row.video_id, row.sample_id))
            if row.source not in SOURCES:
                report.issues.append(ValidationIssue("error", "BAD_SOURCE", row.source, row.sample_id))
            if row.sample_id in samples:
                report.issues.append(ValidationIssue("error", "DUP_SAMPLE", row.sample_id, row.sample_id))
            samples.add(row.sample_id)
            if row.frame_id < 0 or row.timestamp_ms < 0:
                report.issues.append(ValidationIssue("error", "NEGATIVE_POSITION", "frame/time must be >= 0", row.sample_id))
            if row.embedding_row is not None and row.embedding_ref:
                key = (row.embedding_ref, row.embedding_row)
                if key in embedding_rows:
                    report.issues.append(ValidationIssue("error", "DUP_EMBEDDING_ROW", str(key), row.sample_id))
                embedding_rows.add(key)
            prev = previous.get(row.video_id)
            if prev:
                if row.timestamp_ms < prev.timestamp_ms or row.frame_id < prev.frame_id:
                    report.issues.append(ValidationIssue("error", "TIME_REGRESSION", f"after {prev.sample_id}", row.sample_id))
                elif row.frame_id == prev.frame_id:
                    report.issues.append(ValidationIssue("warning", "DUP_FRAME_ID", row.frame_uid, row.sample_id))
            previous[row.video_id] = row
            video = vids.get(row.video_id)
            if row.pts is None and row.video_id not in missing_pts_videos:
                report.issues.append(ValidationIssue("warning", "PTS_UNKNOWN", "timestamp is source-provided; raw PTS unavailable", row.sample_id))
                missing_pts_videos.add(row.video_id)
            elif video and video.time_base_num and video.time_base_den:
                expected = row.pts * video.time_base_num / video.time_base_den * 1000.0
                if abs(expected - row.timestamp_ms) > 2.0:
                    report.issues.append(ValidationIssue("error", "PTS_MISMATCH", f"expected {expected:.3f} ms", row.sample_id))
            if video and video.duration_ms is not None and row.timestamp_ms > video.duration_ms + 2.0:
                report.issues.append(ValidationIssue("error", "AFTER_DURATION", str(video.duration_ms), row.sample_id))
        return report

    def iter_video(self, video_id: str) -> Iterator[FrameRecord]:
        return iter(sorted((x for x in self.frames if x.video_id == video_id),
                           key=lambda x: (x.timestamp_ms, x.sample_id)))

    def save(self, directory: str | Path) -> None:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        header = {"schema_version": self.schema_version, "kind": "header"}
        for filename, values in (("videos.jsonl", self.videos), ("frames.jsonl", self.frames)):
            with (root / filename).open("w", encoding="utf-8") as fh:
                fh.write(json.dumps(header, ensure_ascii=False) + "\n")
                for row in values:
                    fh.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: str | Path) -> "CanonicalManifest":
        root = Path(directory)
        def rows(name: str):
            with (root / name).open("r", encoding="utf-8") as fh:
                head = json.loads(next(fh))
                if head.get("schema_version") != "aic2026.canonical.v1":
                    raise ValueError(f"unsupported manifest schema: {head}")
                yield from (json.loads(line) for line in fh if line.strip())
        return cls(videos=[VideoRecord(**x) for x in rows("videos.jsonl")],
                   frames=[FrameRecord(**x) for x in rows("frames.jsonl")])


def build_btc_manifest(dataset_root: str | Path) -> CanonicalManifest:
    """Build BTC manifest and fail if mapping/embedding row counts diverge."""
    from aic2026.core.dataset import load_map_keyframes, scan
    import numpy as np
    videos: List[VideoRecord] = []
    frames: List[FrameRecord] = []
    for video_id, paths in sorted(scan(dataset_root).items()):
        if paths.mapping is None:
            continue
        mapping = load_map_keyframes(paths.mapping)
        if paths.clip is not None:
            matrix = np.load(paths.clip, mmap_mode="r")
            if matrix.ndim != 2:
                raise ValueError(
                    f"{paths.clip}: expected rank-2 embeddings, got {matrix.shape}")
            if matrix.shape[0] != len(mapping):
                raise ValueError(
                    f"{video_id}: mapping rows {len(mapping)} != "
                    f"embedding rows {matrix.shape[0]}")
        fps_values = [x.fps for x in mapping if x.fps > 0]
        fps = fps_values[0] if fps_values and max(fps_values) - min(fps_values) < 1e-6 else None
        videos.append(VideoRecord(video_id=video_id, fps=fps,
                                  timing_mode="cfr" if fps is not None else "unknown"))
        for row_index, row in enumerate(mapping):
            frames.append(FrameRecord(
                video_id=video_id, frame_id=row.frame_idx,
                timestamp_ms=row.pts_time * 1000.0, source="btc",
                sample_id=f"btc:{video_id}:{row.n}:{row_index}",
                keyframe_ordinal=row.n, embedding_ref=str(paths.clip or ""),
                embedding_row=row_index if paths.clip is not None else None))
    return CanonicalManifest(videos=videos, frames=frames)
