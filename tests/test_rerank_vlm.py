"""Rerank bằng VLM — khoá các điều kiện AN TOÀN, không phải chất lượng.

Chất lượng đo bằng `do_diem_thuc_te.py` trên Kaggle (cần ảnh corpus). Ở đây chỉ
khoá đúng một điều: **module này không được làm bài nộp tệ đi một cách bất ngờ**.
Mỗi test tương ứng một cách nó có thể phá bài nộp trong đêm thi.
"""
from pathlib import Path

import pytest

from aic2026.retrieval.rerank_vlm import (BaoCaoRerank, CauHinhRerank, _Cache,
                                          rerank_mot_truy_van)


def rows(n):
    return [{"video_id": f"V{i:03d}", "frame_idx": i * 10} for i in range(n)]


def anh_luon_co(r):
    return Path(f"/gia/{r['video_id']}_{r['frame_idx']}.jpg")


class BoCham:
    """Provider giả: trả điểm theo bảng dựng sẵn, đếm số lần gọi."""

    def __init__(self, bang=None, no=None):
        self.bang = bang or {}
        self.no = no
        self.goi = 0
        self.lo_da_cham = []

    def cham_anh(self, mo_ta, anh):
        self.goi += 1
        self.lo_da_cham.append([p.name for p in anh])
        if self.no and self.goi >= self.no:
            raise RuntimeError("429 quota hết")
        return [self.bang.get(p.name, 0.0) for p in anh]


# --------------------------------------------------------------------------- #
# An toàn: không bao giờ tệ đi bất ngờ
# --------------------------------------------------------------------------- #
def test_vlm_cham_moi_anh_bang_nhau_thi_giu_nguyen_thu_tu():
    """Hoà điểm = không có thông tin. Phải ra ĐÚNG bản gốc, không xáo."""
    r = rows(20)
    bo = BoCham()                       # mọi ảnh 0.0
    ra, bc = rerank_mot_truy_van(r, "mô tả", bo, anh_luon_co,
                                 CauHinhRerank(sau=20, lo=10))
    assert ra == r
    assert bc.lo_hoa == 2 and not bc.doi_hang_1


def test_chi_dung_toi_sau_dong_dau_phan_con_lai_bat_bien():
    """`sau=10` mà xáo cả 100 dòng là phá mất lưới an toàn ở đuôi."""
    r = rows(100)
    bo = BoCham({f"V{i:03d}_{i * 10}.jpg": float(i) for i in range(100)})
    ra, _ = rerank_mot_truy_van(r, "m", bo, anh_luon_co,
                                CauHinhRerank(sau=10, lo=10, che_do="diem"))
    assert ra[10:] == r[10:]
    assert {x["video_id"] for x in ra[:10]} == {x["video_id"] for x in r[:10]}
    # điểm tăng dần theo i -> đảo ngược đúng 10 dòng đầu
    assert ra[0] == r[9] and ra[9] == r[0]


def test_loi_giua_chung_thi_giu_phan_chua_cham_chu_khong_nem():
    """Quota chết ở lô 2: nửa bài được xếp lại vẫn hơn không có bài nào."""
    r = rows(30)
    bang = {f"V{i:03d}_{i * 10}.jpg": float(30 - i) for i in range(30)}
    bo = BoCham(bang, no=2)             # lô 1 ok, lô 2 nổ
    ra, bc = rerank_mot_truy_van(r, "m", bo, anh_luon_co,
                                 CauHinhRerank(sau=30, lo=10, che_do="diem"))
    assert len(ra) == 30
    assert {x["video_id"] for x in ra} == {x["video_id"] for x in r}
    assert bc.loi and "429" in bc.loi[0]
    # 10 dòng đầu đã chấm -> lên trước; 20 dòng chưa chấm giữ thứ tự gốc
    assert [x["video_id"] for x in ra[10:]] == [x["video_id"] for x in r[10:]]


def test_thieu_anh_thi_dong_do_giu_hang_goc_chu_khong_bien_mat():
    r = rows(10)

    def thieu_mot(row):
        return None if row["video_id"] == "V003" else anh_luon_co(row)

    bo = BoCham({f"V{i:03d}_{i * 10}.jpg": 50.0 + i for i in range(10)})
    ra, bc = rerank_mot_truy_van(r, "m", bo, thieu_mot,
                                 CauHinhRerank(sau=10, lo=10, che_do="diem"))
    assert len(ra) == 10
    assert {x["video_id"] for x in ra} == {x["video_id"] for x in r}
    assert bc.thieu_anh == 1
    assert bo.lo_da_cham and not any("V003" in n for n in bo.lo_da_cham[0])


def test_khong_lam_mat_hay_nhan_ban_dong_nao():
    """Sai sót ở đây = CSV thiếu dòng hoặc trùng dòng, đốt slot vô ích."""
    r = rows(100)
    bo = BoCham({f"V{i:03d}_{i * 10}.jpg": float((i * 37) % 100)
                 for i in range(100)})
    for che_do in ("diem", "rrf"):
        ra, _ = rerank_mot_truy_van(r, "m", bo, anh_luon_co,
                                    CauHinhRerank(sau=100, lo=10, che_do=che_do))
        assert len(ra) == 100
        assert [x["video_id"] for x in sorted(ra, key=lambda y: y["frame_idx"])] \
            == [x["video_id"] for x in r]


def test_rerank_rong_hoac_it_dong_khong_no():
    for n in (0, 1, 3):
        ra, bc = rerank_mot_truy_van(rows(n), "m", BoCham(), anh_luon_co,
                                     CauHinhRerank(sau=100, lo=10))
        assert len(ra) == n


# --------------------------------------------------------------------------- #
# Hai chế độ trộn
# --------------------------------------------------------------------------- #
def test_che_do_diem_cho_vlm_toan_quyen():
    r = rows(3)
    bo = BoCham({"V000_0.jpg": 1.0, "V001_10.jpg": 99.0, "V002_20.jpg": 50.0})
    ra, bc = rerank_mot_truy_van(r, "m", bo, anh_luon_co,
                                 CauHinhRerank(sau=3, lo=10, che_do="diem"))
    assert [x["video_id"] for x in ra] == ["V001", "V002", "V000"]
    assert bc.doi_hang_1 and bc.hang_moi_cua_top1_cu == 3


def test_che_do_rrf_bao_thu_hon_khi_vlm_va_encoder_bat_dong():
    """RRF không cho VLM đẩy hạng-1-gốc xuống tận đáy chỉ vì chấm thấp."""
    r = rows(10)
    bang = {f"V{i:03d}_{i * 10}.jpg": 10.0 for i in range(10)}
    bang["V000_0.jpg"] = 0.0            # VLM ghét đúng hạng 1 của encoder
    bang["V009_90.jpg"] = 99.0
    bo = BoCham(bang)
    diem, _ = rerank_mot_truy_van(r, "m", bo, anh_luon_co,
                                  CauHinhRerank(sau=10, lo=10, che_do="diem"))
    bo2 = BoCham(bang)
    rrf, _ = rerank_mot_truy_van(r, "m", bo2, anh_luon_co,
                                 CauHinhRerank(sau=10, lo=10, che_do="rrf"))
    h_diem = [x["video_id"] for x in diem].index("V000")
    h_rrf = [x["video_id"] for x in rrf].index("V000")
    assert h_diem == 9, "chế độ điểm để nó rơi xuống đáy"
    assert h_rrf < h_diem, "chế độ rrf phải giữ nó lại gần đầu"


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def test_cache_khoa_theo_CA_tap_anh_khong_chi_mo_ta(tmp_path):
    """Cùng mô tả nhưng khác lô thì điểm khác — model chấm bằng cách SO SÁNH."""
    c = _Cache(str(tmp_path / "c.json"), "m1")
    k1 = c.khoa("cùng mô tả", ["a", "b"])
    k2 = c.khoa("cùng mô tả", ["a", "c"])
    assert k1 != k2


def test_cache_doi_model_thi_bo_cache_cu(tmp_path):
    p = str(tmp_path / "c.json")
    c = _Cache(p, "model-cu")
    c.put(c.khoa("m", ["a"]), [1.0])
    c.luu()
    assert _Cache(p, "model-cu").data
    assert _Cache(p, "model-moi").data == {}, "điểm của model khác không dùng lại được"


def test_cache_tranh_goi_lai_lo_da_cham(tmp_path):
    r = rows(10)
    bang = {f"V{i:03d}_{i * 10}.jpg": float(i) for i in range(10)}
    cfg = CauHinhRerank(sau=10, lo=10, che_do="diem")
    c = _Cache(str(tmp_path / "c.json"), "m")
    bo = BoCham(bang)
    a, _ = rerank_mot_truy_van(r, "m", bo, anh_luon_co, cfg, cache=c, qid="q")
    assert bo.goi == 1
    b, bc = rerank_mot_truy_van(r, "m", bo, anh_luon_co, cfg, cache=c, qid="q")
    assert bo.goi == 1, "lần hai phải ăn cache, không tốn request"
    assert a == b and bc.cache_hit >= 1
