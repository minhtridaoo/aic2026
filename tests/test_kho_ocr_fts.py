"""Kênh OCR per-keyframe từ `ocr_fts.sqlite` (Stage11).

Cả lớp này đứng trên MỘT mệnh đề: thứ tự `frame_row` của Stage11 trùng khít
thứ tự event của `load_dataset`. Mệnh đề đó đã kiểm chứng chéo bằng đồng hồ
trên màn hình (`kiem_thu_tu_ocr.py`, lệch trung vị 0,20s). Bộ test dưới đây
khoá phần CÒN LẠI: gán đúng, không gán bừa khi số lượng lệch, và không bao giờ
làm sập bài nộp.
"""
import sqlite3

import pytest

from aic2026.core.models import Event
from aic2026.retrieval.kho_ocr_fts import CauHinhOcrFts, KhoOcrFts, _bo_dau


def E(video, n, t=0.0):
    return Event(video_id=video, event_id=f"{video}#{n:05d}",
                 t_start=float(t), t_end=float(t) + 2, start_frame=0,
                 end_frame=1, keyframe_idx=0)


def lam_db(tmp_path, texts, bang="OCR_ALL"):
    p = tmp_path / "ocr.sqlite"
    con = sqlite3.connect(str(p))
    con.execute(
        f"CREATE VIRTUAL TABLE {bang} USING fts5("
        f"frame_ref_id UNINDEXED, frame_row UNINDEXED, text_norm, "
        f"tokenize='unicode61 remove_diacritics 2')")
    con.executemany(f"INSERT INTO {bang} VALUES (?,?,?)",
                    [(f"fr_{i}", i, t) for i, t in enumerate(texts)])
    con.commit()
    con.close()
    return str(p)


# --------------------------------------------------------------------------- #
# Nạp
# --------------------------------------------------------------------------- #
def test_thieu_file_thi_bao_loi_CHI_DUONG(tmp_path):
    with pytest.raises(FileNotFoundError, match="11ntoe"):
        KhoOcrFts(tmp_path / "khong-co.sqlite")


def test_sai_ten_bang_thi_liet_ke_bang_co_that(tmp_path):
    p = lam_db(tmp_path, ["a", "b"])
    with pytest.raises(RuntimeError, match="OCR_ALL"):
        KhoOcrFts(p, CauHinhOcrFts(bang="OCR_KHONG_CO"))


def test_ten_bang_bay_ba_bi_chan(tmp_path):
    """Tên bảng không tham số hoá được trong SQL — phải chặn ở cổng vào."""
    p = lam_db(tmp_path, ["a"])
    with pytest.raises(ValueError, match="không hợp lệ"):
        KhoOcrFts(p, CauHinhOcrFts(bang="OCR_ALL; DROP TABLE x"))


def test_bao_cao_dem_dung_dong_co_chu(tmp_path):
    kho = KhoOcrFts(lam_db(tmp_path, ["cá mập", "", "   ", "đình thần"]))
    assert kho.bao_cao.so_dong == 4 and kho.bao_cao.so_dong_co_chu == 2
    assert "OCR_ALL" in kho.bao_cao.mo_ta()


# --------------------------------------------------------------------------- #
# Gán event — chỗ nguy hiểm nhất
# --------------------------------------------------------------------------- #
def test_gan_event_theo_vi_tri_khi_so_luong_khop(tmp_path):
    kho = KhoOcrFts(lam_db(tmp_path, ["cá mập", "đình thần", "panna cotta"]))
    evs = [E("V1", 0), E("V1", 1), E("V2", 0)]
    assert kho.gan_event(evs) == 3
    assert kho.xep_hang_event("cá mập")[0][0] == "V1#00000"
    assert kho.xep_hang_event("panna cotta")[0][0] == "V2#00000"


def test_so_luong_LECH_thi_KHONG_gan_bua_va_canh_bao(tmp_path):
    """Gán theo vị trí khi số lượng lệch = mọi khung nhận OCR của khung khác."""
    kho = KhoOcrFts(lam_db(tmp_path, ["a", "b", "c"]))
    assert kho.gan_event([E("V1", 0), E("V1", 1)]) == 0
    assert kho.xep_hang_event("a") == []
    assert any("khác số dòng OCR" in c for c in kho.bao_cao.canh_bao)


def test_tap_con_KHONG_dung_lai_ban_do(tmp_path):
    """Lớp ghim thu chỉ mục xuống một video rồi gọi lại `index`.

    Dựng lại bản đồ theo tập con là gán lệch toàn bộ; phải GIỮ bản đồ đầy đủ và
    để `RetrievalService` tự lọc event ngoài chỉ mục.
    """
    kho = KhoOcrFts(lam_db(tmp_path, ["cá mập", "đình thần", "panna cotta"]))
    full = [E("V1", 0), E("V1", 1), E("V2", 0)]
    kho.gan_event(full)
    kho.gan_event([E("V2", 0)])                 # tập con
    assert kho.xep_hang_event("cá mập")[0][0] == "V1#00000"


def test_chua_gan_thi_tra_rong_chu_khong_no(tmp_path):
    kho = KhoOcrFts(lam_db(tmp_path, ["cá mập"]))
    assert kho.xep_hang_event("cá mập") == []


# --------------------------------------------------------------------------- #
# Truy vấn
# --------------------------------------------------------------------------- #
def test_bo_dau_hai_phia_nen_ocr_sai_dau_van_khop(tmp_path):
    """OCR tiếng Việt sai dấu liên tục ('giáy' thay vì 'giây')."""
    kho = KhoOcrFts(lam_db(tmp_path, ["ca mâp trò lai bò biên cape cod"]))
    kho.gan_event([E("V1", 0)])
    assert kho.xep_hang_event("cá mập Cape Cod")
    assert _bo_dau("đình thần") == "dinh than"


def test_token_dac_biet_cua_fts5_khong_lam_no_truy_van(tmp_path):
    """`NEAR`, `OR`, `-`, `*`, `:`, `^` phải được bọc nháy, không thành toán tử."""
    kho = KhoOcrFts(lam_db(tmp_path, ["cá mập"]))
    kho.gan_event([E("V1", 0)])
    for q in ['NEAR OR AND NOT', 'a* ^b -c "d"', ':: ((( )))', 'covid-19']:
        kho.xep_hang_event(q)                   # không được ném


def test_bo_so_ngan_loai_dong_ho_ban_tin(tmp_path):
    """OCR bản tin đầy '06:30:11' -> token 06/30/11, khớp bừa mọi câu có số."""
    kho = KhoOcrFts(lam_db(tmp_path, ["htv 06:30:11 giay", "cá mập"]),
                    CauHinhOcrFts(bo_so_ngan=True))
    kho.gan_event([E("V1", 0), E("V1", 1)])
    # '30' bị bỏ (số ≤ 2 chữ số); 'con' và 'ho' (bỏ dấu của 'hổ') được giữ.
    assert kho._cau_fts("30 con hổ") == '"con" OR "ho"'
    assert "06" not in kho._cau_fts("lúc 06 giờ")
    # Số DÀI vẫn giữ — "200g" hay "2024" là bằng chứng thật, không phải đồng hồ.
    assert '"2024"' in kho._cau_fts("lễ hội năm 2024")


def test_truy_van_rong_hoac_toan_tu_ngan_tra_rong(tmp_path):
    kho = KhoOcrFts(lam_db(tmp_path, ["cá mập"]))
    kho.gan_event([E("V1", 0)])
    assert kho.xep_hang_event("") == []
    assert kho.xep_hang_event("a 1 ?") == []


def test_diem_chuan_hoa_ve_0_1_va_giam_dan(tmp_path):
    kho = KhoOcrFts(lam_db(tmp_path, [
        "cá mập trắng ven biển cape cod", "cá mập", "thời tiết"]))
    kho.gan_event([E("V1", i) for i in range(3)])
    ds = kho.xep_hang_event("cá mập trắng cape cod")
    assert ds and ds[0][1] == pytest.approx(1.0)
    assert all(0.0 <= s <= 1.0 for _, s in ds)
    assert [s for _, s in ds] == sorted((s for _, s in ds), reverse=True)


def test_van_ban_tra_dung_dong(tmp_path):
    kho = KhoOcrFts(lam_db(tmp_path, ["dòng không", "dòng một"]))
    assert kho.van_ban(1) == "dòng một"
    assert kho.van_ban(999) == ""
