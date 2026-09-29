"""`may_xem.py` — cái duy nhất nằm giữa MẮT NGƯỜI và bài nộp, nên phải có test.

Trang xem là chỗ đắt nhất còn lại của hệ thống (R@100 = 57,5% mà chỉ ăn 34,5%
điểm — chênh 0,23 điểm/câu), và nó chạy vào đúng 3 giờ làm bài, không có lần
thứ hai. Ba loại lỗi được khoá ở đây, cả ba đều KHÔNG có triệu chứng nếu hỏng:

  1. **Ghi mất công của người vận hành.** File ghim còn chứa `videos` / `nhom` /
     `answer` / `so_dong` / `ghi_chu` tra bằng OCR-ASR-tiêu đề. Nếu mỗi cú bấm
     ảnh ghi đè cả mục thì những thứ đó biến mất im lặng.
  2. **File ghim dở dang.** Bấm ảnh là ghi file; Ctrl+C đúng lúc ghi mà không
     ghi qua file tạm thì mất sạch lựa chọn cả buổi.
  3. **Nhầm `frame_idx` với `n`.** Bài nộp ghi `frame_idx` (số khung video gốc)
     còn tên file ảnh theo `n` (số thứ tự keyframe). `L26_V004` khung 4803 là
     ảnh `061.jpg`. Nhầm thì trang hiện toàn ảnh SAI mà không báo lỗi gì.

Không test nào ở đây gọi mạng, nạp model, hay cần ảnh thật.
"""
import json
from pathlib import Path

import pytest

from aic2026.core import ghim as ghim_mod
from aic2026.core.dataset import KeyframeRow
from aic2026.core.models import Query, SubEvent

import may_xem

GOC = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# 1. Ghi file ghim
# --------------------------------------------------------------------------- #
def test_ghi_ghim_giu_nguyen_khoa_nguoi_van_hanh_viet_tay(tmp_path):
    """Nguyên văn một mục của `data/ghim_dot1.json` — bấm ảnh không được xoá nó."""
    p = tmp_path / "ghim.json"
    tho = {
        "query-p1-12-kis": {
            "videos": ["L22_V029"],
            "so_dong": 60,
            "ghi_chu": "OCR L22_V029 kf=52: 'dâu mazut còn 15.562 dông/kg'",
        }
    }
    p.write_text(json.dumps(tho, ensure_ascii=False), encoding="utf-8")

    may_xem.ghi_ghim(p, tho, "query-p1-12-kis", [["L22_V029", 1560]], ai="tuấn")

    lai = json.loads(p.read_text(encoding="utf-8"))["query-p1-12-kis"]
    assert lai["khung"] == [["L22_V029", 1560]]
    assert lai["videos"] == ["L22_V029"]        # tra bằng OCR — phải còn
    assert lai["so_dong"] == 60                 # núm đặt cược — phải còn
    assert "mazut" in lai["ghi_chu"]            # LÝ DO — phải còn
    assert lai["_nguoi"] == "tuấn"


def test_ghi_ghim_bo_chon_het_thi_bo_khoa_khung_chu_khong_de_rong(tmp_path):
    p = tmp_path / "ghim.json"
    tho = {"query-p1-1-kis": {"khung": [["L21_V001", 10]], "so_dong": 50}}
    p.write_text(json.dumps(tho), encoding="utf-8")

    may_xem.ghi_ghim(p, tho, "query-p1-1-kis", [])

    muc = json.loads(p.read_text(encoding="utf-8"))["query-p1-1-kis"]
    assert "khung" not in muc
    assert muc["so_dong"] == 50


def test_ghi_ghim_khong_de_lai_file_do_dang(tmp_path, monkeypatch):
    """Ghi hỏng giữa đường thì file CŨ phải còn nguyên, không thành JSON gãy."""
    p = tmp_path / "ghim.json"
    goc = {"query-p1-4-kis": {"videos": ["L21_V014"], "so_dong": 72}}
    p.write_text(json.dumps(goc, ensure_ascii=False), encoding="utf-8")
    tho = json.loads(p.read_text(encoding="utf-8"))

    def no(*a, **k):
        raise KeyboardInterrupt("người vận hành bấm Ctrl+C đúng lúc đang ghi")

    monkeypatch.setattr(may_xem.os, "replace", no)
    with pytest.raises(KeyboardInterrupt):
        may_xem.ghi_ghim(p, tho, "query-p1-4-kis", [["L21_V014", 228]])

    # File thật vẫn đọc được và vẫn là bản cũ; bản mới nằm ở `.tam`.
    assert json.loads(p.read_text(encoding="utf-8")) == goc


def test_ghim_do_may_xem_ghi_ra_thi_nop_bai_doc_duoc(tmp_path):
    """Vòng tròn khép kín: bấm ảnh -> file -> `doc_ghim` -> đường nộp bài."""
    p = tmp_path / "ghim.json"
    tho = {}
    may_xem.ghi_ghim(p, tho, "query-p1-4-kis", [["l21_v014", 228]])
    may_xem.ghi_ghim(p, tho, "query-p1-16-trake",
                     [["L24_V007", 100, 250, 400, 610]])

    g = ghim_mod.doc_ghim(p)
    assert g["query-p1-4-kis"].khung == [["L21_V014", 228]]      # đã hoa hoá
    assert g["query-p1-16-trake"].khung == [["L24_V007", 100, 250, 400, 610]]


# --------------------------------------------------------------------------- #
# 2. frame_idx  vs  n
# --------------------------------------------------------------------------- #
class KhoGia:
    """Chỉ đủ thứ mà các hàm sinh HTML dùng tới — không nạp gì."""

    def __init__(self, truy_van, kf=None, chon=None):
        self.truy_van = list(truy_van)
        self._kf = kf or {}
        self.chon = dict(chon or {})
        self.nhan = {}
        self.a = type("A", (), {"sau": 300, "rong": 210,
                                "ghim": "data/ghim_dot2.json"})()

    # cùng chữ ký với Kho thật
    def q(self, qid):
        return next((q for q in self.truy_van if q.qid == qid), None)

    def de_text(self, q):
        return may_xem.Kho.de_text(self, q)

    def n_events(self, q):
        return may_xem.Kho.n_events(self, q)

    def da_co(self, qid):
        return True

    def keyframes(self, vid):
        return self._kf.get(vid.upper(), [])

    def n_cua(self, vid, frame_idx):
        for r in self.keyframes(vid):
            if int(r.frame_idx) == int(frame_idx):
                return int(r.n)
        return None

    def rows(self, qid):
        q = self.q(qid)
        if q.task == "trake":
            return [{"video_id": "L24_V007", "frames": [100, 250, 400, 610]}]
        return [{"video_id": "L26_V004", "frame_idx": 4803}]


# Nguyên văn ví dụ đã ghi trong `xem_anh.py`: khung 4803 là ảnh 061.jpg.
KF_V004 = [KeyframeRow(n=60, pts_time=190.0, fps=25.0, frame_idx=4750),
           KeyframeRow(n=61, pts_time=192.1, fps=25.0, frame_idx=4803),
           KeyframeRow(n=62, pts_time=194.3, fps=25.0, frame_idx=4860)]


def test_trang_cau_mo_anh_theo_n_chu_khong_theo_frame_idx():
    q = Query(qid="query-p1-8-kis", task="kis", text="người đầu bếp")
    kho = KhoGia([q], kf={"L26_V004": KF_V004})
    trang = may_xem.trang_cau(kho, "query-p1-8-kis")

    assert "n=61" in trang            # tên file ảnh
    assert "n=4803" not in trang      # nếu lọt cái này thì hiện ảnh sai/thiếu
    assert 'data-k="L26_V004|4803"'in trang   # còn dòng NỘP vẫn là frame_idx
    assert "192s" in trang            # có mốc giây để người định vị


def test_trang_cau_thieu_map_keyframes_thi_noi_ra_chu_khong_hien_o_trong():
    q = Query(qid="query-p1-8-kis", task="kis", text="x")
    trang = may_xem.trang_cau(KhoGia([q], kf={}), "query-p1-8-kis")
    assert "không có map" in trang


# --------------------------------------------------------------------------- #
# 3. TRAKE
# --------------------------------------------------------------------------- #
def _q_trake(n=4):
    return Query(qid="query-p1-16-trake", task="trake", text="",
                 sub_events=[SubEvent(text=f"sự kiện {i}")
                             for i in range(1, n + 1)])


def test_trake_trang_cau_mot_o_la_ca_bo_khung():
    q = _q_trake(4)
    kf = [KeyframeRow(n=i, pts_time=i * 2.0, fps=25.0, frame_idx=f)
          for i, f in enumerate([100, 250, 400, 610], start=1)]
    trang = may_xem.trang_cau(KhoGia([q], kf={"L24_V007": kf}),
                              "query-p1-16-trake")
    assert 'data-k="L24_V007|100|250|400|610"' in trang
    assert "một bộ 4 khung" in trang


def test_trake_trang_video_biet_can_gom_bao_nhieu_moc():
    """Trang video bấm từng khung; TRAKE phải gom đúng `n_events` thành một dòng.

    Sai số này là lỗi đã xảy ra thật: đợt 1 nộp `query-p1-16-trake` với 3 khung
    trong khi đề có 4 sự kiện (dòng dẫn không nhãn cũng là một sự kiện) và BTC
    BÁC thẳng cả câu.
    """
    q = _q_trake(4)
    kf = [KeyframeRow(n=1, pts_time=2.0, fps=25.0, frame_idx=100)]
    trang = may_xem.trang_video(KhoGia([q], kf={"L24_V007": kf}),
                                "L24_V007", "query-p1-16-trake")
    assert "const QID" in trang and "NE = 4" in trang
    assert "E1→E4" in trang


def test_trang_video_khong_co_map_thi_bao_chu_khong_no_500():
    q = Query(qid="query-p1-1-kis", task="kis", text="x")
    trang = may_xem.trang_video(KhoGia([q], kf={}), "L21_V999",
                                "query-p1-1-kis")
    assert "Không có map-keyframes" in trang


# --------------------------------------------------------------------------- #
# 4. Trang chủ
# --------------------------------------------------------------------------- #
def test_trang_chu_hien_so_khung_da_chon_va_ai_dang_lam():
    qs = [Query(qid="query-p1-1-kis", task="kis", text="a"),
          Query(qid="query-p1-3-qa", task="qa", text="b")]
    kho = KhoGia(qs, chon={"query-p1-1-kis": [["L21_V001", 10],
                                              ["L21_V001", 20]]})
    kho.nhan["query-p1-1-kis"] = "tuấn"
    trang = may_xem.trang_chu(kho)
    assert "2 khung" in trang and "tuấn" in trang
    assert "chưa chọn" in trang          # câu còn lại
