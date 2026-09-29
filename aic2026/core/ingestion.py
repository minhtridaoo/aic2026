"""Ingestion: cut a video's sampled frames into overlapping event windows.

Core WESp idea: the retrieval unit is a ~10s window ("event"), not an isolated
static keyframe. Real ingestion would attach a SigLIP2 embedding + OCR + ASR to
each window; here we carry whatever text payloads the caller provides.

Cost note (design critique #13): fixed 10s/2s over 300h -> ~108k windows. In
practice reduce with shot-boundary detection and reserve MLLM captioning for the
lazy top-k at query time. `window`/`overlap` are config-driven for that reason.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .models import Event, Frame


def sliding_windows(
    frames: List[Frame],
    window: float = 10.0,
    overlap: float = 2.0,
    payloads: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[Event]:
    """Group time-ordered frames of ONE video into overlapping events.

    payloads: optional {event_id: {"visual":..., "ocr":..., "asr":...}} to attach
    modality text (e.g. from captioner/OCR/ASR passes).
    """
    if not frames:
        return []
    if window <= 0 or overlap < 0 or overlap >= window:
        raise ValueError("require window > 0 and 0 <= overlap < window")

    frames = sorted(frames, key=lambda f: f.pts_time)
    video_id = frames[0].video_id
    if any(f.video_id != video_id for f in frames):
        raise ValueError("sliding_windows expects frames from a single video")

    step = window - overlap
    t0, t1 = frames[0].pts_time, frames[-1].pts_time
    payloads = payloads or {}

    events: List[Event] = []
    idx = 0
    start = t0
    while start <= t1:
        end = start + window
        span = [f for f in frames if start <= f.pts_time < end]
        if span:
            event_id = f"{video_id}#e{idx:05d}"
            # Keyframe = frame nearest the window centre.
            centre = (start + end) / 2.0
            kf = min(span, key=lambda f: abs(f.pts_time - centre))
            p = payloads.get(event_id, {})
            events.append(Event(
                video_id=video_id,
                event_id=event_id,
                t_start=start,
                t_end=end,
                start_frame=span[0].frame_idx,
                end_frame=span[-1].frame_idx,
                keyframe_idx=kf.frame_idx,
                visual_text=p.get("visual", ""),
                ocr_text=p.get("ocr", ""),
                asr_text=p.get("asr", ""),
            ))
            idx += 1
        start += step
    return events
