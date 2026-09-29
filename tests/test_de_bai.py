"""Bộ đọc ĐỀ THI — khoá lại hai lỗi CÂM, phát hiện 2026-08-17.

Cả hai đều thuộc loại tệ nhất: chạy trơn tru, không ném lỗi, chỉ có điểm là sai.

  1. NGÔN NGỮ. Bộ đọc cũ (`nop_bai.doc_truy_van`) lấy
     `q.get("text") or q.get("_mo_ta_vi")` — không chính sách, không log. Đề của
     BTC có cả hai bản; nếu bản tiếng Việt nằm dưới khoá tên `text` thì hệ thống
     truy hồi bằng tiếng Việt. Đo trên 120 câu KIS, SigLIP2 base-224: tiếng Việt
     0.1750 so với tiếng Anh 0.2950 — mất 41%, không triệu chứng.

  2. CHUỖI SỰ KIỆN TRAKE. Bộ đọc cũ lặp thẳng `q["sub_events"]`. Một CHUỖI
     "a → b" bị Python lặp theo KÝ TỰ thành 5 sub-event. `n_events` sai kéo theo
     mọi dòng TRAKE sai số cột, mà luật BTC cho R-Score = 0 khi số khoảnh khắc
     không khớp.
"""
import json
from pathlib import Path

import pytest

from aic2026 import config as cau_hinh
from aic2026.core import de_bai
from aic2026.core.offline_eval import load_queries


# --------------------------------------------------------------------------- #
# Lỗi 2 — chuỗi sự kiện TRAKE
# --------------------------------------------------------------------------- #
def test_chuoi_mui_ten_thanh_dung_so_su_kien():
    """Lỗi cũ trả 5 sub-event (mỗi ký tự một cái). Đúng phải là 2."""
    ra = de_bai.tach_sub_events("a → b", "trake-x")
    assert [d["text"] for d in ra] == ["a", "b"]


def test_chuoi_thuc_te_cua_dev_set():
    chuoi = ("người đàn ông ngồi đối diện cán bộ → người đàn ông cúi xuống ký "
             "giấy tờ → người đàn ông ngồi lại trong khi cán bộ xem giấy tờ")
    ra = de_bai.tach_sub_events(chuoi, "trake-001")
    assert len(ra) == 3
    assert ra[0]["text"].startswith("người đàn ông ngồi đối diện")


@pytest.mark.parametrize("dau", ["->", "=>", ";", "|", "\n"])
def test_cac_dau_tach_khac(dau):
    assert len(de_bai.tach_sub_events(f"một{dau}hai{dau}ba", "t")) == 3


def test_dau_phay_KHONG_phai_dau_tach():
    """Một sự kiện đơn lẻ hoàn toàn có thể chứa dấu phẩy — tách theo phẩy là
    băm nhỏ đúng thứ cần giữ nguyên."""
    with pytest.raises(ValueError):
        de_bai.tach_sub_events("người đàn ông, mặc áo đỏ, bước vào", "t")


def test_chuoi_khong_tach_duoc_thi_nem_loi():
    with pytest.raises(ValueError, match="không tách được"):
        de_bai.tach_sub_events("chỉ có một sự kiện duy nhất", "trake-x")


def test_bat_chu_ky_lap_theo_ky_tu():
    """Phòng khi rác đến từ nơi khác: list('a → b') phải bị chặn."""
    with pytest.raises(ValueError, match="lặp theo ký tự"):
        de_bai.tach_sub_events(list("a → b"), "trake-x")


def test_giu_modality_hint_cua_dang_dict():
    ra = de_bai.tach_sub_events(
        [{"text": "x", "modality_hint": "visual"}, {"text": "y"}], "t")
    assert ra[0]["modality_hint"] == "visual"
    assert "modality_hint" not in ra[1]


def test_sub_events_rong_thi_nem_loi():
    with pytest.raises(ValueError):
        de_bai.tach_sub_events([], "trake-x")
    with pytest.raises(ValueError):
        de_bai.tach_sub_events(None, "trake-x")


# --------------------------------------------------------------------------- #
# Lỗi 1 — chính sách ngôn ngữ
# --------------------------------------------------------------------------- #
def test_bay_that_text_la_tieng_viet_con_ban_anh_o_khoa_khac():
    """Kịch bản hỏng cụ thể: BTC đặt bản tiếng Việt dưới đúng khoá `text`.

    Bộ đọc cũ lấy `text` -> truy hồi bằng tiếng Việt -> mất 41% điểm KIS.
    """
    q = {"qid": "kis-01", "task": "kis",
         "text": "Một đầu bếp và một người phụ nữ đang trò chuyện",
         "text_en": "A chef and a woman are talking"}
    text, bg = de_bai.chon_van_ban(q, "en", "kis-01")
    assert text == "A chef and a woman are talking"
    assert bg.ngon_ngu == "en" and bg.theo_chinh_sach
    assert bg.khoa == "text_en"


def test_chinh_sach_vi_lay_ban_tieng_viet():
    q = {"qid": "k", "text": "A chef is talking",
         "text_vi": "Một đầu bếp đang nói chuyện"}
    text, bg = de_bai.chon_van_ban(q, "vi", "k")
    assert text == "Một đầu bếp đang nói chuyện"
    assert bg.theo_chinh_sach


def test_chi_co_tieng_viet_thi_canh_bao_chu_khong_im_lang():
    q = {"qid": "k", "text": "Một người đang đổ nguyên liệu vào nồi"}
    text, bg = de_bai.chon_van_ban(q, "en", "k")
    assert text.startswith("Một người")
    assert bg.theo_chinh_sach is False
    # Cảnh báo phải kèm SỐ ĐO của cấu hình thi (so400m + hợp nhất) và chỉ sang
    # lớp vá, chứ không chỉ nói suông "đang dùng bản tiếng Việt".
    assert bg.canh_bao and "0.2883" in bg.canh_bao
    assert "viet_lai" in bg.canh_bao


def test_ten_khoa_thang_phep_doan_theo_dau_khi_viet_KHONG_DAU():
    """Phát hiện lúc chạy thử 17/08: tiếng Việt không dấu ở `text` bị xếp là
    'en', nên `text` (đứng trước) thắng `text_en` — chọn đúng bản KHÔNG nên dùng,
    và còn được đánh dấu là hợp chính sách."""
    q = {"qid": "k", "text": "Mot nguoi nong dan dang lai may keo",
         "text_en": "A farmer is driving a tractor"}
    text, bg = de_bai.chon_van_ban(q, "en", "k")
    assert text == "A farmer is driving a tractor"
    assert bg.khoa == "text_en"


def test_ten_khoa_thang_khi_tieng_anh_co_danh_tu_rieng_viet():
    """Chiều sai còn lại: 'Nhổn' có dấu, nên câu TIẾNG ANH bị đoán thành 'vi'."""
    q = {"qid": "k", "text_en": "a metro train at Nhổn station",
         "text_vi": "tàu metro tại ga Nhổn"}
    assert de_bai.ngon_ngu_cua(q["text_en"]) == "vi"      # phép đoán sai
    text, bg = de_bai.chon_van_ban(q, "en", "k")          # tên khoá cứu
    assert text == "a metro train at Nhổn station"
    assert bg.khoa == "text_en"


def test_khoa_la_van_tim_duoc():
    """Chưa ai thấy file đề thật — bộ đọc phải chịu được tên khoá lạ."""
    q = {"qid": "k", "cau_hoi_truy_van_cua_btc": "A man is riding a bicycle"}
    text, bg = de_bai.chon_van_ban(q, "en", "k")
    assert text == "A man is riding a bicycle"
    assert bg.khoa == "cau_hoi_truy_van_cua_btc"


def test_khong_co_van_ban_nao_thi_nem_loi():
    with pytest.raises(ValueError, match="không tìm thấy trường văn bản"):
        de_bai.chon_van_ban({"qid": "k", "task": "kis"}, "en", "k")


def test_cau_hoi_qa_luon_uu_tien_tieng_viet():
    """Câu hỏi không tham gia chấm cosine, nên nó KHÔNG theo `chinh_sach`."""
    q = {"_cau_hoi": "What color is the shirt?",
         "_cau_hoi_vi": "Áo màu gì?"}
    assert de_bai.chon_cau_hoi(q) == "Áo màu gì?"


def test_chon_sub_events_theo_chinh_sach():
    q = {"qid": "trake-1",
         "sub_events": ["a man sits down", "the man signs papers"],
         "_mo_ta_vi": "người đàn ông ngồi xuống → người đàn ông ký giấy"}
    se_en, bg_en = de_bai.chon_sub_events(q, "en", "trake-1")
    assert [d["text"] for d in se_en] == ["a man sits down",
                                          "the man signs papers"]
    assert bg_en.ngon_ngu == "en"

    se_vi, bg_vi = de_bai.chon_sub_events(q, "vi", "trake-1")
    assert len(se_vi) == 2 and se_vi[0]["text"].startswith("người đàn ông ngồi")
    assert bg_vi.ngon_ngu == "vi" and bg_vi.khoa == "_mo_ta_vi"


# --------------------------------------------------------------------------- #
# Nối vào config và vào đường nộp bài
# --------------------------------------------------------------------------- #
def test_ngon_ngu_truy_van_mac_dinh_la_en():
    assert cau_hinh.ngon_ngu_truy_van({}) == "en"
    assert cau_hinh.ngon_ngu_truy_van({"truy_van": {"ngon_ngu": "vi"}}) == "vi"


def test_ngon_ngu_la_khong_duoc_im_lang_bo_qua():
    with pytest.raises(cau_hinh.CauHinhSai):
        cau_hinh.ngon_ngu_truy_van({"truy_van": {"ngon_ngu": "english"}})


def test_config_example_chot_en_va_answer_candidates_luat():
    """Hai khoá này là quyết định vận hành, không phải mặc định tình cờ."""
    goc = Path(__file__).resolve().parent.parent
    c = cau_hinh.load_config(goc / "config.example.yaml")
    assert cau_hinh.ngon_ngu_truy_van(c) == "en"
    assert str((c.get("qa") or {}).get("answer_candidates")).lower() == "luat"


def test_doc_truy_van_di_qua_ca_hai_bay(tmp_path):
    """Đầu-cuối: một file đề đặt đúng cả hai bẫy cùng lúc."""
    import nop_bai

    de = tmp_path / "de.json"
    de.write_text(json.dumps({"queries": [
        {"qid": "kis-1", "task": "kis",
         "text": "Một người nông dân đang lái máy kéo",
         "text_en": "A farmer is driving a tractor"},
        {"qid": "qa-1", "task": "qa",
         "text_en": "A chef is cooking", "text_vi": "Một đầu bếp đang nấu ăn",
         "_cau_hoi_vi": "Áo màu gì?"},
        {"qid": "trake-1", "task": "trake",
         "sub_events": "a man sits → the man signs → the man stands"},
    ]}, ensure_ascii=False), encoding="utf-8")

    truy_van, ban_ghi, ten_files = nop_bai.doc_truy_van(str(de), "en")
    assert [q.qid for q in truy_van] == ["kis-1", "qa-1", "trake-1"]
    assert truy_van[0].text == "A farmer is driving a tractor"
    assert truy_van[1].text == "A chef is cooking"
    assert truy_van[1].question_vi == "Áo màu gì?"
    # Bẫy chuỗi: phải là 3 sự kiện, KHÔNG phải một sự kiện cho mỗi ký tự.
    assert truy_van[2].n_events == 3
    assert all(bg.theo_chinh_sach for bg in ban_ghi)


def test_doc_truy_van_bao_khi_de_chi_co_tieng_viet(tmp_path):
    import nop_bai

    de = tmp_path / "de.json"
    de.write_text(json.dumps({"queries": [
        {"qid": "kis-1", "task": "kis", "text": "Một người đang mở laptop"},
    ]}, ensure_ascii=False), encoding="utf-8")
    _, ban_ghi, _ = nop_bai.doc_truy_van(str(de), "en")
    assert ban_ghi[0].theo_chinh_sach is False
    assert ban_ghi[0].canh_bao


# --------------------------------------------------------------------------- #
# Tên file nộp — Codabench ghép file với truy vấn THEO TÊN
# --------------------------------------------------------------------------- #
def test_qid_dung_dang_btc_thi_khong_them_tien_to():
    """Đề `query-3-qa.txt` -> phải nộp `query-3-qa.csv`, không phải
    `query-query-3-qa.csv`."""
    assert de_bai.ten_file_nop("query-3-qa") == "query-3-qa.csv"
    assert de_bai.canh_bao_ten_file("query-3-qa.csv") == ""


def test_qid_kieu_dev_set_bi_canh_bao():
    """`qa-1` cho ra `query-qa-1.csv` trong khi BTC đợi `query-1-qa.csv` —
    zip vẫn hợp lệ, mọi phép kiểm định dạng vẫn xanh, truy vấn vẫn 0 điểm."""
    ten = de_bai.ten_file_nop("qa-1")
    assert ten == "query-qa-1.csv"
    assert "0 điểm" in de_bai.canh_bao_ten_file(ten)


def test_ten_file_ghi_tay_duoc_ton_trong():
    assert de_bai.ten_file_nop("bất kỳ", "query-12-trake") == "query-12-trake.csv"
    assert de_bai.ten_file_nop("bất kỳ", "query-12-trake.csv") == \
        "query-12-trake.csv"


def test_doc_truy_van_lay_ten_file_tu_de(tmp_path):
    import nop_bai

    de = tmp_path / "de.json"
    de.write_text(json.dumps({"queries": [
        {"qid": "query-1-kis", "task": "kis", "text_en": "A farmer"},
        {"qid": "lung-tung", "task": "qa", "text_en": "A chef",
         "ten_file": "query-7-qa"},
    ]}, ensure_ascii=False), encoding="utf-8")
    _, _, ten_files = nop_bai.doc_truy_van(str(de), "en")
    assert ten_files["query-1-kis"] == "query-1-kis.csv"
    assert ten_files["lung-tung"] == "query-7-qa.csv"


# --------------------------------------------------------------------------- #
# `--vi` phải thật sự chạm tới TRAKE
# --------------------------------------------------------------------------- #
def test_load_queries_vi_doi_ca_sub_events_cua_trake(tmp_path):
    """Trước 17/08 nhánh TRAKE bỏ qua `truong_text`, nên mọi số TRAKE "tiếng
    Việt" thật ra là số tiếng Anh — hai lần chạy ra kết quả giống hệt nhau."""
    p = tmp_path / "dev.json"
    p.write_text(json.dumps({"queries": [
        {"qid": "trake-1", "task": "trake",
         "sub_events": ["a man sits down", "the man signs papers"],
         "_mo_ta_vi": "người đàn ông ngồi xuống → người đàn ông ký giấy",
         "gt": {"video_id": "L01_V001", "event_windows": [[1, 2], [3, 4]]}},
    ]}, ensure_ascii=False), encoding="utf-8")

    (q_en, _), = load_queries(str(p), truong_text="text")
    (q_vi, _), = load_queries(str(p), truong_text="_mo_ta_vi")
    assert [s.text for s in q_en.sub_events] == ["a man sits down",
                                                 "the man signs papers"]
    assert q_vi.sub_events[0].text.startswith("người đàn ông ngồi")
    assert q_en.n_events == q_vi.n_events == 2


def test_load_queries_vi_thieu_truong_thi_nem_loi(tmp_path):
    p = tmp_path / "dev.json"
    p.write_text(json.dumps({"queries": [
        {"qid": "trake-1", "task": "trake",
         "sub_events": ["a", "b"],
         "gt": {"video_id": "L01_V001", "event_windows": [[1, 2], [3, 4]]}},
    ]}, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(KeyError):
        load_queries(str(p), truong_text="_mo_ta_vi")


# --------------------------------------------------------------------------- #
# LỖI CÂM SỐ 3 — phát hiện 2026-08-20 trên bộ đề THẬT của vòng thử nghiệm.
#
# `doc_de_txt` ghi kèm khoá `ten_file` vào mỗi câu để tên CSV nộp luôn khớp tên
# file đề. Tên năm nay là `query-p1-1-kis`: 14 ký tự (ngưỡng quét dự phòng là
# 12), không dấu tiếng Việt nên bị xếp 'en', và chính sách cũng là 'en' -> nó
# THẮNG bản mô tả tiếng Việt thật. Cả 21 câu KIS/QA đã chạy một lượt đầy đủ trên
# 177.321 khung bằng đúng chuỗi "query-p1-1-kis". Zip hợp lệ, nghiệm thu xanh.
#
# Tên năm ngoái `query-1-kis` dài 11 ký tự nên nằm dưới ngưỡng — chưa lộ ra là
# MAY, không phải vì thiết kế đúng.
# --------------------------------------------------------------------------- #

def test_ten_file_khong_bao_gio_bi_lay_lam_van_ban_truy_van():
    q = {"qid": "query-p1-1-kis", "task": "kis", "ten_file": "query-p1-1-kis",
         "text": "Đây là phần giới thiệu việc phóng tàu vũ trụ tư nhân."}
    t, bg = de_bai.chon_van_ban(q, "en", "query-p1-1-kis")
    assert t.startswith("Đây là phần giới thiệu")
    assert bg.khoa == "text" and bg.ngon_ngu == "vi"
    # Đề tiếng Việt mà chính sách 'en' -> PHẢI lệch, để lớp viết lại vào cuộc.
    assert not bg.theo_chinh_sach


def test_khoa_la_phai_la_MOT_CAU_moi_duoc_lam_van_ban():
    """Chặn theo HÌNH DẠNG, không chỉ theo danh sách tên khoá đoán trước."""
    q = {"qid": "q1", "task": "kis", "ma_noi_bo": "abcdefghijklmnop",
         "duong_dan": "/kaggle/input/de-thi/query-9.txt",
         "text": "Một người nông dân đang thu hoạch lúa"}
    t, bg = de_bai.chon_van_ban(q, "vi", "q1")
    assert bg.khoa == "text" and t.startswith("Một người nông dân")
    khoa_ung_vien = [k for k, _, _ in de_bai.ung_vien_van_ban(q)]
    assert "ma_noi_bo" not in khoa_ung_vien and "duong_dan" not in khoa_ung_vien


def test_khoa_la_that_su_la_cau_thi_van_duoc_nhat():
    """Không được siết đến mức mù: khoá lạ chứa MỘT CÂU vẫn phải nhặt được."""
    q = {"qid": "q1", "task": "kis",
         "noi_dung_truy_van_cua_btc": "A chef is plating spring rolls"}
    t, bg = de_bai.chon_van_ban(q, "en", "q1")
    assert t == "A chef is plating spring rolls"
    assert bg.khoa == "noi_dung_truy_van_cua_btc" and bg.theo_chinh_sach


def test_ban_ghi_in_ra_NOI_DUNG_chu_khong_chi_ten_khoa():
    """Thứ duy nhất biến lỗi trên thành lỗi nhìn thấy được."""
    q = {"qid": "q1", "task": "kis", "text": "Một đàn hổ con quý hiếm ở miền Nam"}
    _, bg = de_bai.chon_van_ban(q, "vi", "q1")
    assert "đàn hổ con" in bg.xem_truoc
    assert "đàn hổ con" in bg.mo_ta()
