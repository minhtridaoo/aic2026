"""XẾP LẠI TOP-N BẰNG VLM — thay cho việc bắt người ngồi nhìn 100 tấm ảnh.

VẤN ĐỀ NÓ GIẢI
--------------
Số đo của hệ thống này, đúng cấu hình thi:

    R@100 = 57,5%      khung đúng ĐÃ nằm trong 100 dòng máy trả về
    điểm  = 46,7%      (public leaderboard thật 20/08) nhưng chỉ ăn chừng này

Toàn bộ phần chênh nằm ở chỗ khung đúng không đứng hạng 1, và thang điểm
`Final = (1/5)·Σ_{k∈{1,5,20,50,100}} R@k` phạt rất nặng: cùng một khung, hạng
100 được 0,2 điểm, hạng 1 được 1,0.

Encoder ảnh không sửa được vì nó đã dùng hết thông tin nó có. Thứ sửa được là
một con mắt NHÌN được ảnh. Con mắt đó không nhất thiết là của người.

CÓ CƠ SỞ, KHÔNG PHẢI HY VỌNG
----------------------------
`thu_vlm_cham_anh.py` đo trên 120 ảnh dev có ground truth thật, 21/08:

    gemini-3.5-flash-lite, mô tả TIẾNG ANH   top-1 10/10
    gemini-3.5-flash-lite, mô tả TIẾNG VIỆT  top-1  9/10   (đoán mò = 1/10)

⚠ Phép thử đó là chế độ DỄ: 10 ảnh ngẫu nhiên từ 10 video khác nhau, đúng một
ảnh khớp. Ở đây khó hơn hẳn — 100 ứng viên đều là top của encoder nên trông na
ná nhau, và đích chỉ nằm trong pool 57,5% số câu. Vì vậy module này được thiết
kế để KHÔNG BAO GIỜ tệ hơn bản gốc quá một mức đã biết trước:

  * chỉ động vào `sau` dòng đầu (mặc định 100); phần còn lại giữ nguyên;
  * điểm hoà thì phá hoà bằng HẠNG GỐC, nên VLM chấm bằng nhau = không đổi gì;
  * cả lô cùng một điểm = coi như không có thông tin, bỏ qua lô đó;
  * lỗi mạng/quota giữa chừng = giữ nguyên phần chưa chấm, không ném ra ngoài.

HAI CÁCH TRỘN
-------------
`che_do="diem"`   xếp thẳng theo điểm VLM, hoà thì theo hạng gốc.
                  Mạnh nhất khi VLM đúng, và cũng rủi ro nhất.
`che_do="rrf"`    hợp nhất hạng VLM với hạng gốc bằng RRF có trọng số.
                  Bảo thủ hơn: VLM sai một câu cũng không đẩy đích xuống đáy.

Chưa có phép đo end-to-end để chọn giữa hai cái, nên mặc định là `rrf` — cái
không thể thua đậm. Đo xong trên dev set thì đổi.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple


@dataclass
class CauHinhRerank:
    sau: int = 100            # xếp lại bao nhiêu dòng đầu
    lo: int = 10              # ảnh mỗi request
    che_do: str = "rrf"       # "rrf" | "diem"
    rrf_k: float = 60.0
    trong_so_vlm: float = 1.0
    trong_so_goc: float = 1.0
    cache_path: Optional[str] = None


@dataclass
class BaoCaoRerank:
    qid: str = ""
    so_dong: int = 0
    so_lo: int = 0
    lo_hoa: int = 0           # lô mà mọi ảnh cùng điểm -> bỏ qua
    thieu_anh: int = 0
    loi: List[str] = field(default_factory=list)
    doi_hang_1: bool = False
    hang_moi_cua_top1_cu: Optional[int] = None
    cache_hit: int = 0

    def mo_ta(self) -> str:
        d = (f"{self.qid}: {self.so_dong} dòng · {self.so_lo} lô"
             f" · {self.cache_hit} cache")
        if self.thieu_anh:
            d += f" · ⚠ {self.thieu_anh} thiếu ảnh"
        if self.lo_hoa:
            d += f" · ⚠ {self.lo_hoa} lô hoà điểm (bỏ qua)"
        d += ("  -> ĐỔI hạng 1" if self.doi_hang_1 else "  -> giữ hạng 1")
        if self.loi:
            d += f" · ❌ {len(self.loi)} lỗi: {self.loi[0][:80]}"
        return d


class _Cache:
    """Cache đĩa cho (model, mô tả, tập event) -> điểm.

    Khoá gồm CẢ danh sách event của lô: cùng một mô tả nhưng lô khác thì điểm
    khác, vì model chấm bằng cách so sánh các ảnh trong lô với nhau.
    """

    def __init__(self, path: Optional[str], model: str):
        self.path = Path(path) if path else None
        self.model = model
        self.data: Dict[str, List[float]] = {}
        self.hit = 0
        if self.path and self.path.is_file():
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                if d.get("model") == model:
                    self.data = d.get("entries", {})
            except (OSError, json.JSONDecodeError):
                self.data = {}

    @staticmethod
    def khoa(mo_ta: str, events: Sequence[str]) -> str:
        raw = "\\n".join([mo_ta, *events])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, k):
        v = self.data.get(k)
        if v is not None:
            self.hit += 1
        return v

    def put(self, k, v):
        self.data[k] = list(v)

    def luu(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"model": self.model, "entries": self.data},
                       ensure_ascii=False),
            encoding="utf-8")


def _thu_tu_rrf(diem: Dict[int, float], n: int, cfg: CauHinhRerank) -> List[int]:
    """Hợp nhất hạng VLM với hạng gốc. Trả danh sách chỉ số theo thứ tự mới."""
    co_diem = sorted(diem, key=lambda i: (-diem[i], i))
    hang_vlm = {i: r for r, i in enumerate(co_diem, start=1)}
    def diem_tong(i):
        s = cfg.trong_so_goc / (cfg.rrf_k + (i + 1))
        if i in hang_vlm:
            s += cfg.trong_so_vlm / (cfg.rrf_k + hang_vlm[i])
        return s
    return sorted(range(n), key=lambda i: (-diem_tong(i), i))


def _thu_tu_diem(diem: Dict[int, float], n: int) -> List[int]:
    """Xếp thẳng theo điểm; KHÔNG có điểm hoặc hoà nhau thì theo hạng gốc.

    Phá hoà bằng `i` (hạng gốc) là điều kiện an toàn quan trọng nhất của cả
    module: VLM chấm mọi ảnh 100 điểm thì thứ tự ra y hệt bản gốc.
    """
    return sorted(range(n), key=lambda i: (-diem.get(i, -1.0), i))


def rerank_mot_truy_van(rows: Sequence[Dict], mo_ta: str, provider,
                        duong_anh: Callable[[Dict], Optional[Path]],
                        cfg: Optional[CauHinhRerank] = None,
                        cache: Optional[_Cache] = None,
                        qid: str = "") -> Tuple[List[Dict], BaoCaoRerank]:
    """Xếp lại `rows` bằng điểm VLM. Trả `(rows mới, báo cáo)`.

    `duong_anh(row) -> Path | None` do bên gọi cấp, vì cách suy từ
    `(video, frame_idx)` ra file ảnh phụ thuộc map-keyframes chứ không phải
    thứ module này nên biết.
    """
    cfg = cfg or CauHinhRerank()
    bc = BaoCaoRerank(qid=qid)
    dau = list(rows[:cfg.sau])
    duoi = list(rows[cfg.sau:])
    bc.so_dong = len(dau)
    if not dau:
        return list(rows), bc

    # Chỉ những dòng CÓ ảnh mới đem đi chấm; dòng thiếu ảnh giữ nguyên hạng gốc
    # (điểm khuyết -> `_thu_tu_diem` xếp chúng theo `i`).
    co_anh: List[Tuple[int, Path]] = []
    for i, r in enumerate(dau):
        p = duong_anh(r)
        if p is None:
            bc.thieu_anh += 1
        else:
            co_anh.append((i, p))

    diem: Dict[int, float] = {}
    for b in range(0, len(co_anh), cfg.lo):
        cum = co_anh[b:b + cfg.lo]
        khoa = _Cache.khoa(mo_ta, [str(p) for _, p in cum]) if cache else None
        d = cache.get(khoa) if cache else None
        if d is None:
            try:
                d = provider.cham_anh(mo_ta, [p for _, p in cum])
            except Exception as exc:                          # noqa: BLE001
                # Quota/mạng chết giữa chừng: giữ nguyên phần chưa chấm thay vì
                # ném ra ngoài. Nửa bài được xếp lại vẫn hơn không có bài nào.
                bc.loi.append(str(exc)[:300])
                break
            if cache:
                cache.put(khoa, d)
        bc.so_lo += 1
        if len(set(d)) <= 1:
            bc.lo_hoa += 1          # cả lô cùng điểm = không có thông tin
            continue
        for (i, _), x in zip(cum, d):
            diem[i] = float(x)

    if not diem:
        return list(rows), bc

    thu_tu = (_thu_tu_diem(diem, len(dau)) if cfg.che_do == "diem"
              else _thu_tu_rrf(diem, len(dau), cfg))
    moi = [dau[i] for i in thu_tu]
    bc.doi_hang_1 = thu_tu[0] != 0
    bc.hang_moi_cua_top1_cu = thu_tu.index(0) + 1
    if cache:
        bc.cache_hit = cache.hit
    return moi + duoi, bc
