"""Fail-closed, memory-mapped exact vector bundle.

The loader refuses a stale or row-shifted bundle before serving any query.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from aic2026.core.stores import VectorStore

BUNDLE_SCHEMA = "aic2026.vector-bundle.v1"


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


class PersistentVectorStore(VectorStore):
    """Read-only exact cosine store backed by a validated numpy memmap."""

    def __init__(self, directory: str | Path,
                 expected_encoder_fingerprint: str | None = None):
        root = Path(directory)
        manifest_path = root / "bundle.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing bundle manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != BUNDLE_SCHEMA:
            raise ValueError("unsupported vector bundle schema")
        required = {"embeddings", "frames", "rows", "dim",
                    "sha256", "encoder_fingerprint"}
        missing = sorted(required - manifest.keys())
        if missing:
            raise ValueError(f"bundle manifest missing: {missing}")
        if (expected_encoder_fingerprint is not None
                and manifest["encoder_fingerprint"]
                != expected_encoder_fingerprint):
            raise ValueError("encoder fingerprint does not match bundle")

        embedding_path = _inside(root, manifest["embeddings"])
        frames_path = _inside(root, manifest["frames"])
        hashes: Dict[str, str] = manifest["sha256"]
        for path in (embedding_path, frames_path):
            expected = hashes.get(path.name)
            if not expected or sha256_file(path) != expected:
                raise ValueError(f"bundle checksum mismatch: {path.name}")

        matrix = np.load(embedding_path, mmap_mode="r")
        if matrix.ndim != 2:
            raise ValueError(f"embeddings must be rank-2, got {matrix.shape}")
        expected_shape = (int(manifest["rows"]), int(manifest["dim"]))
        if matrix.shape != expected_shape:
            raise ValueError(
                f"embedding shape {matrix.shape} != {expected_shape}")
        if matrix.dtype not in (np.float16, np.float32, np.float64):
            raise ValueError(f"unsupported embedding dtype: {matrix.dtype}")

        keys: List[str] = []
        with frames_path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row.get("event_id") or row.get("sample_id") or "")
                if not key:
                    raise ValueError(f"frame metadata line {line_no} has no id")
                keys.append(key)
        if len(keys) != matrix.shape[0]:
            raise ValueError(
                f"metadata rows {len(keys)} != embeddings {matrix.shape[0]}")
        if len(set(keys)) != len(keys):
            raise ValueError("frame metadata ids are not unique")
        # Full finite scan is deliberately explicit: serving corrupt values is
        # worse than a slower startup. On Kaggle this check is vectorized.
        if not bool(np.isfinite(matrix).all()):
            raise ValueError("embeddings contain NaN or infinity")
        norms = np.linalg.norm(matrix.astype(np.float32), axis=1)
        if np.any(norms <= 0):
            raise ValueError("embeddings contain zero-norm rows")

        self._matrix = matrix
        self._norms = norms
        self._keys = keys
        self._row = {key: index for index, key in enumerate(keys)}
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self._keys)

    def add(self, key: str, vector: List[float]) -> None:
        raise TypeError("PersistentVectorStore is read-only")

    def add_many(self, keys: Sequence[str], vectors) -> None:
        raise TypeError("PersistentVectorStore is read-only")

    def lay_vector(self, key: str):
        index = self._row.get(key)
        return None if index is None else self._matrix[index]

    def search(self, vector: List[float],
               top_k: int) -> List[Tuple[str, float]]:
        if top_k <= 0 or not self._keys:
            return []
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                f"query dim {query.shape[0]} != {self._matrix.shape[1]}")
        qnorm = float(np.linalg.norm(query))
        if qnorm == 0 or not np.isfinite(qnorm):
            raise ValueError("query vector must be finite and non-zero")
        scores = (self._matrix @ query) / (self._norms * qnorm)
        k = min(int(top_k), len(scores))
        indices = np.argpartition(-scores, k - 1)[:k]
        # Stable deterministic tie break by row index.
        indices = sorted(indices.tolist(), key=lambda i: (-float(scores[i]), i))
        return [(self._keys[i], float(scores[i])) for i in indices]

    def score_many(self, keys: Sequence[str], vector) -> Dict[str, float]:
        pairs = [(key, self._row[key]) for key in keys if key in self._row]
        if not pairs:
            return {}
        query = np.asarray(vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                f"query dim {query.shape[0]} != {self._matrix.shape[1]}")
        qnorm = float(np.linalg.norm(query))
        if qnorm == 0 or not np.isfinite(qnorm):
            raise ValueError("query vector must be finite and non-zero")
        indices = np.asarray([index for _, index in pairs], dtype=np.int64)
        scores = ((self._matrix[indices] @ query)
                  / (self._norms[indices] * qnorm))
        return {key: float(score)
                for (key, _), score in zip(pairs, scores)}


def write_bundle_manifest(directory: str | Path, *,
                          embeddings: str = "embeddings.npy",
                          frames: str = "frame_metadata.jsonl",
                          encoder_fingerprint: str) -> Path:
    """Finalize an already-written bundle by hashing and recording its shape."""
    root = Path(directory)
    embedding_path = _inside(root, embeddings)
    frames_path = _inside(root, frames)
    matrix = np.load(embedding_path, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError("embeddings must be rank-2")
    manifest = {
        "schema_version": BUNDLE_SCHEMA,
        "embeddings": embeddings,
        "frames": frames,
        "rows": int(matrix.shape[0]),
        "dim": int(matrix.shape[1]),
        "encoder_fingerprint": encoder_fingerprint,
        "sha256": {
            embedding_path.name: sha256_file(embedding_path),
            frames_path.name: sha256_file(frames_path),
        },
    }
    path = root / "bundle.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    resolved_root = root.resolve()
    if resolved_root != path and resolved_root not in path.parents:
        raise ValueError(f"bundle path escapes root: {relative}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
