"""Resolve ``event_id`` to a real keyframe image, or fail loudly.

The lightweight competition bundle contains embeddings but no images.  QA with
a VLM therefore needs an explicit image root.  This resolver supports the dev
set's labelled-image naming convention and common corpus layouts without ever
returning an empty string for a missing frame.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


class ImageResolutionError(FileNotFoundError):
    """The event is valid but none of the configured image roots contains it."""


_EVENT = re.compile(r"^(?P<video>[A-Za-z0-9_-]+)#(?:e)?(?P<n>\d+)$")
_DEV_IMAGE = re.compile(
    r"^\d+_(?P<video>[A-Za-z0-9_-]+)_n(?P<n>\d+)\.(?:jpe?g|png|webp)$",
    re.IGNORECASE,
)


def parse_event_id(event_id: str) -> Tuple[str, int]:
    match = _EVENT.fullmatch(str(event_id).strip())
    if not match:
        raise ValueError(
            f"event_id không đúng dạng VIDEO#00001: {event_id!r}")
    return match.group("video").upper(), int(match.group("n"))


class EventImageResolver:
    """Resolve images from one or more configurable roots.

    ``dev_image_dir`` is indexed non-recursively because files there include a
    label prefix (``001_L22_V006_n209.jpg``).  ``roots`` are corpus/keyframe
    roots and use deterministic candidate paths, avoiding a recursive scan of
    a 28.66 GiB corpus at startup.
    """

    EXTENSIONS = ("jpg", "jpeg", "png", "webp")

    def __init__(self, roots: Iterable[str | os.PathLike] = (),
                 dev_image_dir: Optional[str | os.PathLike] = None,
                 explicit: Optional[Dict[str, str | os.PathLike]] = None):
        self.roots = [Path(p).expanduser().resolve() for p in roots if str(p)]
        self.dev_image_dir = (Path(dev_image_dir).expanduser().resolve()
                              if dev_image_dir else None)
        self._explicit: Dict[str, Path] = {
            str(key): Path(value).expanduser().resolve()
            for key, value in (explicit or {}).items()
        }
        self._dev: Dict[Tuple[str, int], Path] = {}
        if self.dev_image_dir is not None:
            if not self.dev_image_dir.is_dir():
                raise NotADirectoryError(
                    f"Thư mục ảnh dev không tồn tại: {self.dev_image_dir}")
            for path in self.dev_image_dir.iterdir():
                if not path.is_file():
                    continue
                match = _DEV_IMAGE.fullmatch(path.name)
                if match:
                    key = (match.group("video").upper(), int(match.group("n")))
                    self._dev[key] = path.resolve()

    @classmethod
    def from_config(cls, config: dict) -> "EventImageResolver":
        qa = (config or {}).get("qa", {})
        roots = qa.get("image_roots", [])
        if isinstance(roots, (str, os.PathLike)):
            roots = [roots]
        return cls(roots=roots, dev_image_dir=qa.get("dev_image_dir"))

    def register(self, event_id: str, path: str | os.PathLike) -> None:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Ảnh đăng ký không tồn tại: {resolved}")
        self._explicit[str(event_id)] = resolved

    def resolve(self, event_id: str) -> Path:
        explicit = self._explicit.get(str(event_id))
        if explicit is not None:
            if explicit.is_file():
                return explicit
            raise ImageResolutionError(
                f"Ảnh đã đăng ký cho {event_id} nhưng không còn tồn tại: {explicit}")

        video, ordinal = parse_event_id(event_id)
        dev = self._dev.get((video, ordinal))
        if dev is not None and dev.is_file():
            return dev

        checked = []
        stems = (
            str(ordinal), f"{ordinal:03d}", f"{ordinal:05d}", f"{ordinal:06d}",
            f"frame_{ordinal}", f"frame_{ordinal:05d}",
            f"{video}_{ordinal}", f"{video}_{ordinal:05d}",
            f"{video}_n{ordinal}",
        )
        for root in self.roots:
            for directory in (root / video, root / video[:3] / video, root):
                for stem in stems:
                    for ext in self.EXTENSIONS:
                        path = directory / f"{stem}.{ext}"
                        checked.append(path)
                        if path.is_file():
                            return path.resolve()

        roots = ", ".join(str(x) for x in self.roots) or "(chưa cấu hình)"
        sample = ", ".join(str(x) for x in checked[:4])
        raise ImageResolutionError(
            f"Không tìm thấy ảnh cho {event_id} (video={video}, n={ordinal}).\n"
            f"  image_roots: {roots}\n"
            f"  dev_image_dir: {self.dev_image_dir or '(chưa cấu hình)'}\n"
            + (f"  ví dụ đường đã kiểm: {sample}\n" if sample else "")
            + "  QA VLM không thể chạy trên bundle vector-only; hãy gắn corpus "
              "ảnh đầy đủ hoặc chạy tầng trả lời trên Kaggle.")

