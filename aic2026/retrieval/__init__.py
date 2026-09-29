"""Shared retrieval plane."""

from aic2026.core.retrieval import (
    MultiChannelRetriever,
    borda_fuse,
    rrf_fuse,
    weighted_sum_fuse,
)
from .service import RetrievalPolicy, RetrievalService
from .persistent import PersistentVectorStore, write_bundle_manifest

__all__ = ["MultiChannelRetriever", "PersistentVectorStore",
           "RetrievalPolicy", "RetrievalService", "borda_fuse", "rrf_fuse",
           "weighted_sum_fuse", "write_bundle_manifest"]
