"""KÊNH OCR THEO TỪNG KEYFRAME, dựa trên FTS5/BM25 của Stage11.

KHÁC GÌ `kenh_chu.py`
---------------------
`kenh_chu.py` xử lý text theo KHOẢNG SHOT (Stage20) vì ở đó một đoạn text trải
nhiều keyframe. Ở đây thì khác hẳn và tốt hơn: Stage11 cho **đúng 177.321 dòng
OCR, mỗi dòng một keyframe**, khớp 1:1 với `Event` của hệ thống. Không nhân
bản text, không cần ánh xạ khoảng, không méo IDF — nên chấm ở mức keyframe ở
đây là ĐÚNG, trong khi ở kia là sai.

THỨ TỰ ĐÃ ĐƯỢC XÁC MINH, KHÔNG PHẢI GIẢ ĐỊNH
--------------------------------------------
`ocr_fts.sqlite` chỉ cho `frame_row` (0..177320), không cho `(video_id, n)`.
Toàn bộ lớp này đứng trên một mệnh đề: thứ tự 177.321 dòng của Stage11 trùng
khít thứ tự event mà `load_dataset` dựng. Lệch một video là mọi khung nhận OCR
của khung khác — sai hoàn toàn im lặng.

Đã kiểm chứng chéo ngày 21/08 bằng `kiem_thu_tu_ocr.py`: bản tin *60 Giây* có
đồng hồ chạy trên màn hình, OCR đọc được nó, và đó là mốc thời gian tuyệt đối
ĐỘC LẬP với `pts_time` của BTC. Trên 8 video đầu / ~2.000 khung có đồng hồ,
hiệu đồng hồ khớp hiệu `pts_time` với **lệch trung vị 0,20 giây**.

`gan_event` vẫn kiểm lại số lượng mỗi lần và từ chối gán khi không khớp.

VÌ SAO DÙNG THẲNG FTS5 THAY VÌ TỰ CHẤM
--------------------------------------
Bảng đã là `fts5(... tokenize='unicode61 remove_diacritics 2')`, tức đã có chỉ
mục ngược và BM25 dựng sẵn, lại BỎ DẤU ở cả hai phía. Bỏ dấu rất hợp với OCR:
OCR tiếng Việt sai dấu liên tục ("giáy" thay vì "giây"), nên so không dấu bắt
được nhiều hơn hẳn. Tự viết lại bằng Python vừa chậm hơn vừa kém hơn.

BM25 cũng tự lo phần mà `kenh_chu.py` phải tự cài IDF: từ có mặt khắp nơi
("hd", "tin", "giây" trong logo đài) nhận trọng số gần 0.
"""
from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

BANG_MAC_DINH = "OCR_ALL"


@dataclass
class CauHinhOcrFts:
    bat: bool = False
    duong_dan: Optional[str] = None
    bang: str = BANG_MAC_DINH        # OCR_ALL | OCR_S05 | OCR_S08
    trong_so: float = 0.6
    pool: int = 300                  # số keyframe lấy từ FTS trước khi hợp nhất
    # Bỏ token quá ngắn và token CHỈ là số 1-2 chữ số: OCR của bản tin đầy đồng
    # hồ ("06:30:11" -> 06, 30, 11) và số hiệu kênh, chúng khớp bừa với mọi
    # truy vấn có chữ số mà không mang nghĩa gì.
    do_dai_toi_thieu: int = 2
    bo_so_ngan: bool = True


@dataclass
class BaoCaoOcr:
    so_dong: int = 0
    so_dong_co_chu: int = 0
    so_event_gan: int = 0
    bang: str = ""
    canh_bao: List[str] = field(default_factory=list)

    def mo_ta(self) -> str:
        pc = self.so_dong_co_chu / max(1, self.so_dong)
        # "gắn 0 event" ở thời điểm NẠP không phải lỗi — `gan_event` chỉ chạy
        # khi `Pipeline.index()` được gọi sau đó. Nói rõ để khỏi báo động giả.
        gan = (f"gắn {self.so_event_gan:,d} event" if self.so_event_gan
               else "chưa gắn event (chờ Pipeline.index)")
        return (f"{self.so_dong:,d} dòng OCR (`{self.bang}`) · có chữ "
                f"{self.so_dong_co_chu:,d} ({pc:.0%}) · {gan}")


def _bo_dau(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").replace("đ", "d")


_KHONG_CHU = re.compile(r"[^0-9a-z]+")


class KhoOcrFts:
    def __init__(self, duong_dan: str | Path,
                 cfg: Optional[CauHinhOcrFts] = None):
        self.cfg = cfg or CauHinhOcrFts()
        self.bang = self.cfg.bang or BANG_MAC_DINH
        if not re.fullmatch(r"[A-Za-z0-9_]+", self.bang):
            # Tên bảng không tham số hoá được trong SQL nên phải chặn ở đây.
            raise ValueError(f"tên bảng không hợp lệ: {self.bang!r}")
        p = Path(duong_dan)
        if not p.is_file():
            raise FileNotFoundError(
                f"Không thấy OCR FTS: {p}\n"
                f"  Đây là `ocr_fts.sqlite` trong dataset Stage11 "
                f"(`selaninguyen/11ntoe`), thư mục "
                f"`aic2026_kis_slice3/stage11/artifacts/`.")
        # `check_same_thread=False`: chỉ đọc, và pipeline có thể gọi từ thread
        # khác trong lúc đo.
        self.con = sqlite3.connect(str(p), check_same_thread=False)
        self.con.execute("PRAGMA query_only = ON")

        co = {r[0] for r in self.con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if self.bang not in co:
            raise RuntimeError(
                f"`{p.name}` không có bảng {self.bang!r}. Có: "
                f"{sorted(x for x in co if x.startswith('OCR') and '_' not in x[4:])}")

        self.bao_cao = BaoCaoOcr(bang=self.bang)
        self.bao_cao.so_dong = self.con.execute(
            f"SELECT COUNT(*) FROM {self.bang}").fetchone()[0]
        self.bao_cao.so_dong_co_chu = self.con.execute(
            f"SELECT COUNT(*) FROM {self.bang} "
            f"WHERE text_norm IS NOT NULL AND TRIM(text_norm) <> ''").fetchone()[0]
        self._row2event: Dict[int, str] = {}

    # ---- gắn event -------------------------------------------------------- #
    def gan_event(self, events: Sequence) -> int:
        """`frame_row` -> `event_id` theo VỊ TRÍ trong danh sách event.

        Chỉ dựng bản đồ khi số event khớp đúng số dòng OCR. Gọi với một tập CON
        (lớp ghim thu chỉ mục xuống một video) thì GIỮ NGUYÊN bản đồ cũ: các
        event ngoài tập con sẽ tự bị `RetrievalService` lọc bỏ vì không có
        trong `_su_kien`. Dựng lại bản đồ theo tập con là gán lệch toàn bộ.
        """
        n = len(events)
        if n == self.bao_cao.so_dong:
            self._row2event = {i: e.event_id for i, e in enumerate(events)}
            self.bao_cao.so_event_gan = n
            return n
        if not self._row2event:
            self.bao_cao.canh_bao.append(
                f"số event ({n:,d}) khác số dòng OCR "
                f"({self.bao_cao.so_dong:,d}) và chưa có bản đồ nào -> "
                f"KÊNH OCR TẮT. Chỉ nạp một phần corpus?")
        return 0

    # ---- truy vấn --------------------------------------------------------- #
    def _cau_fts(self, truy_van: str) -> str:
        """Câu MATCH của FTS5. Mọi token được bọc nháy kép nên không token nào
        bị hiểu thành toán tử (`NEAR`, `OR`, `-`, `*`, `:`, `^`)."""
        tho = _KHONG_CHU.sub(" ", _bo_dau(str(truy_van).lower())).split()
        tok = []
        for t in tho:
            if len(t) < self.cfg.do_dai_toi_thieu:
                continue
            if self.cfg.bo_so_ngan and t.isdigit() and len(t) <= 2:
                continue
            if t not in tok:
                tok.append(t)
        return " OR ".join(f'"{t}"' for t in tok)

    def cham(self, truy_van: str, top_k: Optional[int] = None
             ) -> List[Tuple[int, float]]:
        """[(frame_row, điểm 0..1)] — BM25 đã đổi dấu và chuẩn hoá theo lô."""
        cau = self._cau_fts(truy_van)
        if not cau:
            return []
        k = int(top_k or self.cfg.pool)
        try:
            rows = self.con.execute(
                f"SELECT frame_row, bm25({self.bang}) AS r FROM {self.bang} "
                f"WHERE {self.bang} MATCH ? ORDER BY r LIMIT ?",
                (cau, k)).fetchall()
        except sqlite3.Error:
            return []
        if not rows:
            return []
        # bm25() âm, càng âm càng khớp. Đổi dấu rồi chia cho lớn nhất để ra
        # thang 0..1 — hợp nhất bằng RRF nên chỉ THỨ HẠNG mới quan trọng, con
        # số chỉ để nhìn và để phá hoà.
        diem = [(int(fr), -float(r)) for fr, r in rows]
        cao = max(s for _, s in diem) or 1.0
        return [(fr, s / cao) for fr, s in diem]

    def xep_hang_event(self, truy_van: str,
                       top_k: int = 300) -> List[Tuple[str, float]]:
        if not self._row2event:
            return []
        ra = []
        for fr, s in self.cham(truy_van, top_k):
            eid = self._row2event.get(fr)
            if eid is not None:
                ra.append((eid, s))
                if len(ra) >= top_k:
                    break
        return ra

    def van_ban(self, frame_row: int) -> str:
        r = self.con.execute(
            f"SELECT text_norm FROM {self.bang} WHERE frame_row = ?",
            (int(frame_row),)).fetchone()
        return str(r[0]) if r else ""
