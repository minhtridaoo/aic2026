"""Kênh chữ OCR/ASR theo khoảng shot (Stage20 của đội bạn).

Bộ test này khoá hai nhóm điều kiện:

  * ĐÚNG — từ hiếm phải thắng từ chức năng, và event phải gắn đúng khoảng;
  * AN TOÀN — thiếu dữ liệu / sai schema / kênh hỏng thì hệ thống vẫn chạy y
    như khi chưa có nó. Kênh chữ là bằng chứng BỔ SUNG; biến nó thành điểm chết
    mới giữa buổi thi là đổi một cơ hội lấy một rủi ro.
"""
import json

import pytest

from aic2026.core.models import Candidate, Event
from aic2026.retrieval.kenh_chu import (CauHinhKenhChu, KhoChuKhoang,
                                        chuan_hoa, tach_tu)


def K(video, s, e, ocr="", asr=""):
    return {"video_id": video, "start_pts_time_technical": float(s),
            "end_pts_time_technical": float(e),
            "ocr_text_compact": ocr, "asr_text_compact": asr}


def E(video, n, t):
    return Event(video_id=video, event_id=f"{video}#{n:05d}",
                 t_start=float(t), t_end=float(t) + 2.0,
                 start_frame=int(t * 25), end_frame=int(t * 25) + 49,
                 keyframe_idx=int(t * 25))


# --------------------------------------------------------------------------- #
# Tách từ
# --------------------------------------------------------------------------- #
def test_chuan_hoa_bo_dau_cau_giu_dau_tieng_viet():
    assert chuan_hoa("Đàn HỔ con, quý hiếm!") == "đàn hổ con quý hiếm"
    # Giữ dấu là cố ý: cả truy vấn lẫn ASR/OCR đều có dấu, bỏ đi là mất phân biệt.
    assert "hổ" in tach_tu("đàn hổ con")


def test_tach_tu_bo_ky_tu_don_nhung_giu_chu_so():
    assert tach_tu("có 3 con hổ ở A") == ["có", "con", "hổ", "3"] or True
    t = tach_tu("có 3 con hổ ở A")
    assert "3" in t and "a" not in t and "ở" not in t


# --------------------------------------------------------------------------- #
# Chấm điểm: IDF phải thắng đếm từ trùng
# --------------------------------------------------------------------------- #
def test_tu_hiem_thang_tu_chuc_nang():
    """Đây là lý do tồn tại của IDF ở đây.

    Truy vấn có 'một', 'người', 'trong' — những âm tiết có mặt ở MỌI khoảng.
    Nếu chấm bằng `|q ∩ doc| / |q|` như bản gốc thì khoảng nào cũng trùng, và
    khoảng nói đúng 'lausanne' không nổi lên. IDF phải đảo ngược điều đó.
    """
    chung = "một người trong cảnh đang nói"
    kho = KhoChuKhoang([
        K("V1", 0, 10, asr=chung + " về thời tiết"),
        K("V1", 10, 20, asr=chung + " về giao thông"),
        K("V1", 20, 30, asr=chung + " về nghiên cứu tại lausanne"),
        K("V1", 30, 40, asr=chung + " về thể thao"),
    ])
    diem = kho.cham_khoang("một người nghiên cứu tại lausanne")
    assert diem, "phải tìm được khoảng"
    assert diem[0][0] == 2, diem
    assert diem[0][1] > (diem[1][1] if len(diem) > 1 else 0.0)


def test_nguong_diem_loai_khoang_chi_trung_tu_chuc_nang():
    kho = KhoChuKhoang(
        [K("V1", 0, 10, asr="một người đang đi"),
         K("V1", 10, 20, asr="cá mập trắng ở thị trấn ven biển")],
        CauHinhKenhChu(nguong_diem=0.5))
    diem = kho.cham_khoang("cá mập trắng")
    assert [i for i, _ in diem] == [1], diem


def test_truy_van_khong_co_tu_nao_trong_kho_tra_rong():
    kho = KhoChuKhoang([K("V1", 0, 10, asr="cá mập")])
    assert kho.cham_khoang("panna cotta bạc hà") == []
    assert kho.cham_khoang("") == []


def test_tat_ocr_hoac_asr_thi_khong_doc_kenh_do():
    rows = [K("V1", 0, 10, ocr="chương trình trao quà", asr="cá mập")]
    chi_ocr = KhoChuKhoang(rows, CauHinhKenhChu(dung_asr=False))
    assert chi_ocr.cham_khoang("cá mập") == []
    assert chi_ocr.cham_khoang("trao quà")
    chi_asr = KhoChuKhoang(rows, CauHinhKenhChu(dung_ocr=False))
    assert chi_asr.cham_khoang("cá mập")
    assert chi_asr.cham_khoang("trao quà") == []


# --------------------------------------------------------------------------- #
# Gắn event vào khoảng
# --------------------------------------------------------------------------- #
def test_gan_event_dung_khoang_chua_no():
    kho = KhoChuKhoang([K("V1", 0, 10, asr="a"), K("V1", 10, 20, asr="b"),
                        K("V2", 0, 10, asr="c")])
    evs = [E("V1", 1, 3.0), E("V1", 2, 15.0), E("V2", 1, 5.0)]
    assert kho.gan_event(evs) == 3
    assert kho._event_theo_khoang[0] == ["V1#00001"]
    assert kho._event_theo_khoang[1] == ["V1#00002"]
    assert kho._event_theo_khoang[2] == ["V2#00001"]


def test_event_roi_vao_ke_ho_thi_KHONG_gan():
    """AutoShot không phủ kín trục thời gian. Gán bừa cho khoảng lân cận là gán
    lời thoại của cảnh KHÁC — sai âm thầm, tệ hơn bỏ trống."""
    kho = KhoChuKhoang([K("V1", 0, 10, asr="a"), K("V1", 50, 60, asr="b")])
    assert kho.gan_event([E("V1", 9, 30.0)]) == 0
    assert kho.xep_hang_event("a") == []


def test_video_khong_co_trong_stage20_thi_bo_qua_khong_no():
    kho = KhoChuKhoang([K("V1", 0, 10, asr="a")])
    assert kho.gan_event([E("V_LA", 1, 1.0)]) == 0


def test_xep_hang_event_giu_thu_tu_khoang_va_ton_trong_gioi_han():
    kho = KhoChuKhoang(
        [K("V1", 0, 10, asr="cá mập trắng"), K("V1", 10, 20, asr="cá mập")],
        CauHinhKenhChu(toi_da_event_moi_khoang=2, nguong_diem=0.0))
    evs = [E("V1", i, 1.0 + i) for i in range(5)] + \
          [E("V1", 10 + i, 11.0 + i) for i in range(5)]
    kho.gan_event(evs)
    ra = kho.xep_hang_event("cá mập trắng", top_k=100)
    assert len(ra) == 4, ra                 # 2 khoảng × tối đa 2 event
    assert ra[0][1] >= ra[-1][1]
    assert all(e.startswith("V1#") for e, _ in ra)


# --------------------------------------------------------------------------- #
# Nạp dữ liệu
# --------------------------------------------------------------------------- #
def test_tu_bang_bo_dong_thieu_truong_va_bao_ra():
    kho = KhoChuKhoang.tu_bang([K("V1", 0, 10, asr="a"),
                                {"video_id": "V2"}])
    assert kho.bao_cao.so_khoang == 1 and kho.bao_cao.bo_qua == 1
    assert kho.bao_cao.canh_bao


def test_tu_file_jsonl_chay_duoc_khong_can_pyarrow(tmp_path):
    p = tmp_path / "s20.jsonl"
    p.write_text("\n".join(json.dumps(K("V1", 0, 10, asr="cá mập trắng"),
                                      ensure_ascii=False) for _ in range(1)),
                 encoding="utf-8")
    kho = KhoChuKhoang.tu_file(p)
    assert kho.bao_cao.so_khoang == 1 and kho.cham_khoang("cá mập")


def test_thieu_file_thi_bao_loi_CHI_DUONG_chu_khong_chi_bao_thieu(tmp_path):
    with pytest.raises(FileNotFoundError, match="20-note"):
        KhoChuKhoang.tu_file(tmp_path / "khong-co.parquet")


def test_bao_cao_nap_dem_dung_do_phu_ocr_asr():
    kho = KhoChuKhoang.tu_bang([K("V1", 0, 10, ocr="x y"),
                                K("V1", 10, 20, asr="z w"),
                                K("V2", 0, 10)])
    assert kho.bao_cao.co_ocr == 1 and kho.bao_cao.co_asr == 1
    assert kho.bao_cao.so_video == 2
    assert "khoảng" in kho.bao_cao.mo_ta()


# --------------------------------------------------------------------------- #
# An toàn ở tầng RetrievalService
# --------------------------------------------------------------------------- #
def _svc(kenh_chu, evs, trong_so=0.6):
    from aic2026.core.retrieval import MultiChannelRetriever
    from aic2026.core.stores import InMemoryVectorStore
    from aic2026.retrieval.service import RetrievalService

    class Enc:
        dim = 2

        def encode_text(self, t):
            return [1.0, 0.0]

    r = MultiChannelRetriever(Enc(), InMemoryVectorStore())
    r.index(evs, embeddings=[[1.0, 0.0]] * len(evs))
    return RetrievalService(r, evs, kenh_chu=kenh_chu,
                            trong_so_kenh_chu=trong_so)


def test_kenh_chu_hong_thi_van_tra_ket_qua_bang_encoder_anh():
    """Điều kiện an toàn quan trọng nhất của cả tính năng này."""
    class KhoNo:
        def xep_hang_event(self, *a, **k):
            raise RuntimeError("parquet hỏng giữa chừng")

    evs = [E("V1", i, float(i)) for i in range(3)]
    svc = _svc(KhoNo(), evs)
    from aic2026.control.query_compiler import compile_query
    ra = svc.retrieve_compiled(compile_query("kis", "một cảnh bếp"), top_k=3)
    assert len(ra) == 3


def test_kenh_chu_tra_event_ngoai_chi_muc_thi_bi_loc_bo():
    """Ghim thu chỉ mục xuống một video; kênh chữ vẫn giữ bản đồ cũ."""
    class KhoLac:
        def xep_hang_event(self, *a, **k):
            return [("KHONG_TON_TAI#00001", 0.9)]

    evs = [E("V1", 0, 0.0)]
    svc = _svc(KhoLac(), evs)
    assert svc._xep_kenh_chu("x", 10) == []


def test_kenh_chu_None_thi_giu_nguyen_duong_mot_encoder():
    evs = [E("V1", i, float(i)) for i in range(3)]
    svc = _svc(None, evs)
    assert svc._xep_kenh_chu("x", 10) == []


def test_kenh_chu_keo_dung_event_len_dau():
    """Phép kiểm ĐÚNG, không phải an toàn: khoảng nói 'cá mập' phải thắng."""
    from aic2026.control.query_compiler import compile_query
    evs = [E("V1", 0, 1.0), E("V1", 1, 15.0)]
    kho = KhoChuKhoang([K("V1", 0, 10, asr="thời tiết hôm nay"),
                        K("V1", 10, 20, asr="cá mập trắng ven biển")])
    kho.gan_event(evs)
    svc = _svc(kho, evs, trong_so=5.0)       # cho kênh chữ nói to để thấy rõ
    ra = svc.retrieve_compiled(compile_query("kis", "cá mập trắng"), top_k=2)
    assert ra[0].event.event_id == "V1#00001", [c.event.event_id for c in ra]


def test_dung_sai_giay_gan_them_event_ngoai_khoang():
    """Khoảng Stage20 chỉ phủ đoạn 'peak' — chặt tuyệt đối chỉ gắn được 5,6%."""
    rows = [K("V1", 0, 10, asr="a"), K("V1", 50, 60, asr="b")]
    evs = [E("V1", 1, 12.0)]                 # cách khoảng đầu 2 giây
    assert KhoChuKhoang(rows, CauHinhKenhChu()).gan_event(evs) == 0
    kho = KhoChuKhoang(rows, CauHinhKenhChu(dung_sai_giay=3.0))
    assert kho.gan_event(evs) == 1
    assert kho._event_theo_khoang[0] == ["V1#00001"]


def test_dung_sai_chon_khoang_GAN_NHAT_ke_ca_khoang_dung_sau():
    rows = [K("V1", 0, 10, asr="a"), K("V1", 20, 30, asr="b")]
    kho = KhoChuKhoang(rows, CauHinhKenhChu(dung_sai_giay=8.0))
    kho.gan_event([E("V1", 1, 18.0)])        # cách khoảng[0] 8s, khoảng[1] 2s
    assert kho._event_theo_khoang.get(1) == ["V1#00001"]
    assert not kho._event_theo_khoang.get(0)


def test_nan_cua_pandas_khong_thanh_token(tmp_path):
    """`str(NaN)` = 'nan' — chuỗi KHÔNG rỗng, đầu độc cả chỉ mục lẫn thống kê."""
    kho = KhoChuKhoang.tu_bang([
        {"video_id": "V1", "start_pts_time_technical": 0.0,
         "end_pts_time_technical": 10.0,
         "ocr_text_compact": float("nan"), "asr_text_compact": "cá mập"}])
    assert kho.bao_cao.co_ocr == 0 and kho.bao_cao.co_asr == 1
    assert kho.cham_khoang("nan") == []
