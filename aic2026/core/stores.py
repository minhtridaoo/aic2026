"""Vector + text stores with runnable in-memory fallbacks.

PLUG POINTS
  VectorStore -> FAISS / Milvus (ANN over SigLIP2 embeddings).
  TextStore   -> Elasticsearch with a Vietnamese analyzer (OCR/ASR full-text).

Both fallbacks are exact/brute-force and single-process — fine for dev &
tests, not for 300h of video.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from .providers import tokenize


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #
def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class VectorStore(ABC):
    @abstractmethod
    def add(self, key: str, vector: List[float]) -> None: ...

    @abstractmethod
    def search(self, vector: List[float], top_k: int) -> List[Tuple[str, float]]: ...

    def add_many(self, keys: Sequence[str], vectors) -> None:
        """Nạp hàng loạt. Bản mặc định chỉ lặp `add`; store nào nạp cả khối
        được nhanh hơn thì ghi đè (xem `NumpyVectorStore`)."""
        for k, v in zip(keys, vectors):
            self.add(k, v)

    def reset(self) -> None:
        """Xoá sạch kho.

        Thêm 2026-08-20. `MultiChannelRetriever.index()` CỘNG DỒN chứ không thay
        thế — đọc tên hàm thì không ai đoán ra. Hệ quả bắt được tận tay: lớp ghim
        gọi `pipeline.index(events_cua_một_video)` để thu chỉ mục xuống 204
        keyframe, nhưng 177.321 keyframe cũ vẫn nằm nguyên trong kho, nên kết quả
        "ghim" mở đầu bằng một video hoàn toàn khác. Không có ngoại lệ nào được
        ném, không có cảnh báo nào.

        Không đổi `index()` thành thay-thế vì nạp nhiều lô là cách dùng hợp lệ và
        đang có (`load_dataset` nạp theo video). Thay vào đó tách hẳn động tác
        xoá ra thành một lời gọi phải viết ra — ai muốn thay thế thì nói rõ.
        """
        raise NotImplementedError

    def lay_vector(self, key: str):
        """Vector ẢNH của một event, hoặc None nếu không có.

        `search()` đi một chiều truy-vấn -> cả kho. QA zero-shot cần chiều ngược
        lại: đã biết khung nào rồi, giờ cần chính vector của nó để chấm từng đáp
        án ứng viên. Không có hàm này thì phải giữ thêm một bản 177.321 × 512
        song song với kho — 363 MB thừa.
        """
        raise NotImplementedError

    def score_many(self, keys: Sequence[str], vector) -> Dict[str, float]:
        """Score known keys in the same geometry as search.

        Stores with transforms (for example modality-gap removal) must
        override this method. The fallback is exact raw cosine.
        """
        out: Dict[str, float] = {}
        q = list(vector)
        for key in keys:
            value = self.lay_vector(key)
            if value is not None:
                out[key] = cosine(q, list(value))
        return out


class InMemoryVectorStore(VectorStore):
    def __init__(self) -> None:
        self._vecs: Dict[str, List[float]] = {}

    def add(self, key: str, vector: List[float]) -> None:
        self._vecs[key] = vector

    def reset(self) -> None:
        self._vecs = {}

    def search(self, vector: List[float], top_k: int) -> List[Tuple[str, float]]:
        scored = [(k, cosine(vector, v)) for k, v in self._vecs.items()]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:top_k]

    def lay_vector(self, key: str):
        return self._vecs.get(key)


class NumpyVectorStore(VectorStore):
    """Bản dùng cho dữ liệu thật: một ma trận numpy thay vì 177k list Python.

    VÌ SAO CẦN: `InMemoryVectorStore` giữ mỗi vector là một `list` Python. Ở quy
    mô 177.321 x 512, riêng phần overhead của object float đã là vài GB, và mỗi
    truy vấn phải chạy 177.321 vòng lặp Python. Ma trận numpy thì tốn 363 MB và
    một truy vấn là một phép nhân ma trận duy nhất.

    Vector của BTC đã chuẩn hoá L2 (`norm = 1.0`) nên cosine đúng bằng tích vô
    hướng. Nếu nguồn khác chưa chuẩn hoá, để `normalized=False` để store tự
    chuẩn hoá lúc nạp.
    """

    def __init__(self, normalized: bool = True,
                 khu_modality_gap: bool = False) -> None:
        """`khu_modality_gap` — bỏ hướng trung bình của kho trước khi so sánh.

        CLIP đặt vector ảnh và vector chữ ở HAI NÓN TÁCH RỜI nhau (gọi là
        modality gap). Hệ quả: cosine chữ-ảnh bị chi phối bởi hướng trung bình
        chung, chứ không phải bởi nội dung. Chiếu bỏ hướng đó đi thì phần còn
        lại mới là khác biệt về nội dung.

        Đo được trên chính corpus này (nhãn yếu lấy từ tiêu đề video, lift =
        tỉ lệ đúng nhóm trong top-100 chia cho tỉ lệ nền):

            THÔ      lift trung bình 2.02x
            KHỬ GAP  lift trung bình 2.25x

        ĐO LẠI 2026-08-08 trên dev set 60 câu KIS CÓ NHÃN, ghép cặp
        (`thu_nghiem_tang_tim.py`) — đây mới là số theo thước đo của BTC:

                        điểm     R@100   video-R@100
            KHỬ GAP    0.1233    28.3%      68.3%
            THÔ        0.1167    28.3%      61.7%
                       Δ = -0.0067 ± 0.0067 khi TẮT

        Đọc cho đúng: chênh lệch điểm KHÔNG vượt 2·SE nên chưa thể nói khử gap
        làm tăng điểm. Lý do vẫn để BẬT mặc định là cột cuối: nó kéo tỉ lệ "đúng
        video lọt top-100" từ 61.7% lên 68.3% mà không mất gì đo được. Đó chính
        là phần mà một bước rerank theo thời gian có thể biến thành điểm — tìm
        đúng video là điều kiện cần, còn chọn đúng khung trong video là việc của
        rerank. (Con số 61.7/68.3 trùng khớp với phép đo thô bằng numpy tách rời
        khỏi pipeline, nên không phải hiệu ứng của cách dựng row.)

        Bật/tắt được để còn A/B. Tốn thêm một bản ma trận trong RAM (346 MiB với
        corpus batch 1).
        """
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "NumpyVectorStore cần numpy. Cài bằng: pip install numpy"
            ) from exc
        self._np = np
        self._normalized = normalized
        self._khu_gap = khu_modality_gap
        self._keys: List[str] = []
        self._chi_so = None              # khoá -> hàng, dựng lười cho lay_vector
        self._mat = None                 # (N, D) float32
        self._mat_khu = None             # bản đã chiếu bỏ hướng trung bình
        self._mu = None                  # hướng trung bình, đã chuẩn hoá
        self._cho_nap: List[Tuple[str, List[float]]] = []

    # ---- nạp ---------------------------------------------------------------
    def add(self, key: str, vector: List[float]) -> None:
        """Nạp lẻ. Gom vào bộ đệm, chỉ ghép ma trận khi có ai đó tìm kiếm —
        ghép lại sau mỗi lần add sẽ là O(N²)."""
        self._cho_nap.append((key, vector))

    def add_many(self, keys: Sequence[str], vectors) -> None:
        """Nạp cả khối. `vectors` là ma trận (N, D) đã cùng thứ tự với `keys`."""
        np = self._np
        mat = np.ascontiguousarray(vectors, dtype=np.float32)
        if mat.ndim != 2:
            raise ValueError(f"mong ma trận (N, D), nhận {mat.shape}")
        if len(keys) != mat.shape[0]:
            raise ValueError(
                f"lệch số lượng: {len(keys)} key nhưng {mat.shape[0]} vector")
        if not self._normalized:
            mat = self._chuan_hoa(mat)
        self._keys.extend(keys)
        self._mat = mat if self._mat is None else np.vstack([self._mat, mat])
        self._mat_khu = self._mu = None      # thêm dữ liệu -> hướng trung bình đổi

    def reset(self) -> None:
        """Xoá sạch — kể cả bộ đệm `_cho_nap` và mọi thứ dẫn xuất.

        Bỏ sót `_mat_khu`/`_mu` ở đây là bug âm thầm khó nhất trong file này:
        hướng trung bình của kho cũ sẽ tiếp tục được chiếu bỏ khỏi kho mới.
        """
        self._keys = []
        self._chi_so = None
        self._mat = None
        self._mat_khu = None
        self._mu = None
        self._cho_nap = []

    def _xa_bo_dem(self) -> None:
        if not self._cho_nap:
            return
        keys = [k for k, _ in self._cho_nap]
        vecs = [v for _, v in self._cho_nap]
        self._cho_nap = []
        self.add_many(keys, vecs)

    def _chuan_hoa(self, mat):
        np = self._np
        n = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.where(n == 0, 1.0, n)

    def _chieu_bo_mu(self, x):
        """Bỏ thành phần theo hướng trung bình rồi chuẩn hoá lại."""
        np = self._np
        if x.ndim == 1:
            y = x - (x @ self._mu) * self._mu
            n = float(np.linalg.norm(y))
            return y / n if n else y
        y = x - (x @ self._mu)[:, None] * self._mu
        return self._chuan_hoa(y)

    def _chuan_bi_khu_gap(self):
        """Tính hướng trung bình của kho và bản ma trận đã chiếu bỏ nó."""
        np = self._np
        if self._mat_khu is not None:
            return
        mu = self._mat.mean(axis=0)
        n = float(np.linalg.norm(mu))
        if n == 0:                                   # kho suy biến, khỏi khử
            self._mu = np.zeros(self._mat.shape[1], dtype=np.float32)
            self._mat_khu = self._mat
            return
        self._mu = (mu / n).astype(np.float32)
        self._mat_khu = self._chieu_bo_mu(self._mat)

    # ---- tìm ---------------------------------------------------------------
    def __len__(self) -> int:
        self._xa_bo_dem()
        return 0 if self._mat is None else int(self._mat.shape[0])

    def lay_vector(self, key: str):
        """Vector ảnh THÔ của một event (chưa khử modality gap).

        Cố ý trả bản thô: QA zero-shot so ĐÁP ÁN NÀY với ĐÁP ÁN KIA trên cùng
        một khung. Khử gap là phép chiếu bỏ hướng trung bình của KHO ẢNH — nó
        giúp khi so ảnh với ảnh khác, nhưng ở đây khung là cố định và chỉ có
        phần chữ thay đổi, nên trừ đi một hằng số chung không đổi thứ tự mà chỉ
        thêm một bước.
        """
        self._xa_bo_dem()
        if self._mat is None:
            return None
        if self._chi_so is None or len(self._chi_so) != len(self._keys):
            self._chi_so = {k: i for i, k in enumerate(self._keys)}
        i = self._chi_so.get(key)
        return None if i is None else self._mat[i]

    def score_many(self, keys: Sequence[str], vector) -> Dict[str, float]:
        """Batch known-key scoring using the exact search-time transform."""
        np = self._np
        self._xa_bo_dem()
        if self._mat is None or not keys:
            return {}
        if self._chi_so is None or len(self._chi_so) != len(self._keys):
            self._chi_so = {key: i for i, key in enumerate(self._keys)}
        pairs = [(key, self._chi_so[key]) for key in keys
                 if key in self._chi_so]
        if not pairs:
            return {}
        q = np.asarray(vector, dtype=np.float32).ravel()
        if q.shape[0] != self._mat.shape[1]:
            raise ValueError(
                f"số chiều lệch: truy vấn {q.shape[0]}, kho {self._mat.shape[1]}")
        norm = float(np.linalg.norm(q))
        if norm:
            q = q / norm
        indices = np.asarray([index for _, index in pairs], dtype=np.int64)
        if self._khu_gap:
            self._chuan_bi_khu_gap()
            scores = self._mat_khu[indices] @ self._chieu_bo_mu(q)
        else:
            scores = self._mat[indices] @ q
        return {key: float(score)
                for (key, _), score in zip(pairs, scores)}

    def search(self, vector, top_k: int) -> List[Tuple[str, float]]:
        np = self._np
        self._xa_bo_dem()
        if self._mat is None or self._mat.shape[0] == 0 or top_k <= 0:
            return []

        q = np.asarray(vector, dtype=np.float32).ravel()
        if q.shape[0] != self._mat.shape[1]:
            raise ValueError(
                f"số chiều lệch: truy vấn {q.shape[0]}, kho {self._mat.shape[1]}")
        nq = float(np.linalg.norm(q))
        if nq:
            q = q / nq

        if self._khu_gap:
            self._chuan_bi_khu_gap()
            diem = self._mat_khu @ self._chieu_bo_mu(q)
        else:
            diem = self._mat @ q                   # cosine, vì cả hai đã chuẩn hoá
        k = min(top_k, diem.shape[0])
        # argpartition lấy k phần tử lớn nhất trong O(N), rẻ hơn sort O(N log N)
        # rất nhiều khi N = 177k mà k chỉ 100.
        idx = np.argpartition(-diem, k - 1)[:k]
        idx = idx[np.argsort(-diem[idx])]
        return [(self._keys[i], float(diem[i])) for i in idx]


# --------------------------------------------------------------------------- #
# Text store (BM25-lite)
# --------------------------------------------------------------------------- #
class TextStore(ABC):
    @abstractmethod
    def add(self, key: str, text: str) -> None: ...

    @abstractmethod
    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]: ...


class InMemoryTextStore(TextStore):
    """Tiny TF-IDF cosine. NOT Vietnamese-aware (whitespace tokens only)."""

    def __init__(self) -> None:
        self._docs: Dict[str, List[str]] = {}
        self._df: Dict[str, int] = defaultdict(int)

    def add(self, key: str, text: str) -> None:
        toks = tokenize(text)
        self._docs[key] = toks
        for t in set(toks):
            self._df[t] += 1

    def reset(self) -> None:
        self._docs = {}
        self._df = defaultdict(int)

    def _idf(self, term: str) -> float:
        n = len(self._docs)
        return math.log((n + 1) / (self._df.get(term, 0) + 1)) + 1.0

    def _tfidf(self, toks: List[str]) -> Dict[str, float]:
        if not toks:
            return {}
        tf: Dict[str, float] = defaultdict(float)
        for t in toks:
            tf[t] += 1.0 / len(toks)
        return {t: v * self._idf(t) for t, v in tf.items()}

    @staticmethod
    def _cos(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        dot = sum(a[t] * b[t] for t in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        qv = self._tfidf(tokenize(query))
        scored = [(k, self._cos(qv, self._tfidf(toks))) for k, toks in self._docs.items()]
        scored = [s for s in scored if s[1] > 0]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        return scored[:top_k]
