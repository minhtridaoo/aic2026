"""Đọc file đề `.txt` của BTC (việc ngày thi).

Chỗ nguy hiểm nhất là `qid`: sai `qid` -> sai tên file CSV -> Codabench không
ghép được -> câu đó 0 điểm mà zip vẫn hợp lệ. Vì vậy phần lớn test ở đây khoá
đúng một điều: **`qid` phải là tên file BTC, nguyên văn**.
"""
import pytest

from aic2026.core import de_bai
from aic2026.core.doc_de_txt import doc_mot_file, doc_thu_muc, ghi_json


def _de(tmp_path, ten, noi_dung):
    p = tmp_path / ten
    p.write_text(noi_dung, encoding="utf-8")
    return p


def test_qid_lay_nguyen_van_ten_file_btc(tmp_path):
    """Điều kiện sống còn: `query-3-qa.txt` -> qid `query-3-qa` -> CSV cùng tên."""
    p = _de(tmp_path, "query-3-qa.txt", "Một đầu bếp nấu ăn.\nÁo màu gì?")
    muc, bg = doc_mot_file(p)
    assert muc["qid"] == "query-3-qa"
    assert bg.qid == "query-3-qa"
    assert de_bai.ten_file_nop(muc["qid"], muc["ten_file"]) == "query-3-qa.csv"
    assert de_bai.canh_bao_ten_file("query-3-qa.csv") == ""


def test_loai_cau_suy_tu_ten_file(tmp_path):
    # Nội dung phải hợp lệ với TỪNG loại, nếu không cảnh báo nổi lên là đúng.
    for ten, task, noi_dung in [
            ("query-1-kis.txt", "kis", "một người đang nấu ăn"),
            ("query-2-qa.txt", "qa", "một cảnh bếp\nÁo màu gì?"),
            ("query-9-trake.txt", "trake", "rót dầu\nđảo thức ăn")]:
        muc, bg = doc_mot_file(_de(tmp_path, ten, noi_dung))
        assert muc["task"] == task and not bg.canh_bao


def test_ten_la_thi_doan_nhung_phai_canh_bao(tmp_path):
    muc, bg = doc_mot_file(_de(tmp_path, "cau_5_qa_them.txt", "Màu gì?"))
    assert muc["task"] == "qa"
    assert any("không đúng mẫu" in c for c in bg.canh_bao)


def test_ten_khong_ro_loai_thi_tam_kis_va_canh_bao_to(tmp_path):
    muc, bg = doc_mot_file(_de(tmp_path, "cau1.txt", "một người đi bộ"))
    assert muc["task"] == "kis"
    assert any("không cho biết loại" in c for c in bg.canh_bao)


def test_qa_tach_cau_hoi_o_dong_cuoi(tmp_path):
    muc, bg = doc_mot_file(_de(
        tmp_path, "query-3-qa.txt",
        "Một đầu bếp và một người phụ nữ trong bếp.\n"
        "Người phụ nữ mặc áo màu gì?"))
    assert muc["_cau_hoi_vi"] == "Người phụ nữ mặc áo màu gì?"
    assert "đầu bếp" in muc["text"] and "màu gì" not in muc["text"]
    assert bg.co_cau_hoi and not bg.canh_bao


def test_qa_khong_co_dau_hoi_thi_canh_bao(tmp_path):
    """Không có câu hỏi thì tầng đáp án đoán mò — phải nói ra."""
    muc, bg = doc_mot_file(_de(tmp_path, "query-3-qa.txt", "Một cảnh trong bếp."))
    assert "_cau_hoi_vi" not in muc
    assert any("không thấy dòng nào kết thúc" in c for c in bg.canh_bao)


def test_trake_tach_theo_dong_va_bo_danh_so(tmp_path):
    muc, bg = doc_mot_file(_de(
        tmp_path, "query-4-trake.txt",
        "1. Người rót dầu vào chảo\n2) Người đảo thức ăn\n- Món ăn hoàn thành"))
    assert muc["sub_events"] == ["Người rót dầu vào chảo",
                                 "Người đảo thức ăn",
                                 "Món ăn hoàn thành"]
    assert bg.so_su_kien == 3 and not bg.canh_bao


def test_trake_mot_dong_thi_tach_theo_mui_ten(tmp_path):
    """Lỗi câm cũ: một CHUỖI bị lặp theo KÝ TỰ thành nhiều sự kiện."""
    muc, _ = doc_mot_file(_de(tmp_path, "query-4-trake.txt", "a → b -> c; d"))
    assert muc["sub_events"] == ["a", "b", "c", "d"]


def test_trake_it_hon_hai_su_kien_thi_canh_bao(tmp_path):
    _, bg = doc_mot_file(_de(tmp_path, "query-4-trake.txt", "chỉ một sự kiện"))
    assert any("sự kiện" in c for c in bg.canh_bao)


def test_file_rong_thi_canh_bao_chu_khong_no(tmp_path):
    muc, bg = doc_mot_file(_de(tmp_path, "query-1-kis.txt", "   \n\n"))
    assert muc["text"] == ""
    assert any("RỖNG" in c for c in bg.canh_bao)


def test_bom_utf8_khong_lot_vao_truy_van(tmp_path):
    p = tmp_path / "query-1-kis.txt"
    p.write_bytes("﻿Một người đang nấu ăn".encode("utf-8"))
    muc, _ = doc_mot_file(p)
    assert muc["text"].startswith("Một người")


def test_doc_thu_muc_sap_theo_SO_khong_phai_chuoi(tmp_path):
    """`query-10` phải sau `query-2`; sắp theo chuỗi thì ngược."""
    for i in (1, 2, 10, 11):
        _de(tmp_path, f"query-{i}-kis.txt", f"câu {i}")
    muc, _ = doc_thu_muc(tmp_path)
    assert [m["qid"] for m in muc] == ["query-1-kis", "query-2-kis",
                                       "query-10-kis", "query-11-kis"]


def test_qid_trung_thi_no_ngay(tmp_path):
    """Hai câu cùng qid = ghi đè lên cùng một CSV = mất câu, phải chặn."""
    (tmp_path / "a").mkdir()
    _de(tmp_path, "query-1-kis.txt", "x")
    p2 = tmp_path / "a" / "query-1-kis.txt"
    p2.write_text("y", encoding="utf-8")
    with pytest.raises(ValueError, match="trùng"):
        doc_thu_muc(tmp_path, "**/*.txt")


def test_thu_muc_rong_thi_bao_loi_ro(tmp_path):
    with pytest.raises(FileNotFoundError, match="không có file"):
        doc_thu_muc(tmp_path)


# --------------------------------------------------------------------------- #
# BỘ ĐỀ THẬT — vòng thử nghiệm AIC2026, phát 2026-08-20.
# Ba test dưới đây dán NGUYÊN VĂN nội dung BTC. Cả ba đều FAIL với code trước
# 2026-08-20, và mỗi cái tương ứng với một câu bị 0 điểm mà zip vẫn hợp lệ.
# --------------------------------------------------------------------------- #

def test_ten_file_co_ma_goi_p1_van_khop_quy_uoc(tmp_path):
    """`query-p1-1-kis` là tên THẬT của BTC. Báo nó "lệch" là cảnh báo SAI."""
    muc, bg = doc_mot_file(_de(tmp_path, "query-p1-1-kis.txt", "phóng tàu vũ trụ"))
    assert muc["task"] == "kis" and muc["qid"] == "query-p1-1-kis"
    assert bg.canh_bao == []
    assert de_bai.ten_file_nop(muc["qid"], muc["ten_file"]) == "query-p1-1-kis.csv"
    assert de_bai.canh_bao_ten_file("query-p1-1-kis.csv") == ""
    # Dạng cũ không được vỡ.
    assert de_bai.canh_bao_ten_file("query-3-qa.csv") == ""


def test_trake_cau_dan_khong_bi_dem_thanh_su_kien(tmp_path):
    """query-p1-16-trake THẬT: 1 câu dẫn + 4 sự kiện. Bản cũ đọc ra 5 -> 0 điểm."""
    muc, bg = doc_mot_file(_de(tmp_path, "query-p1-16-trake.txt", (
        "Đoạn video múa lân một con lân màu vàng đen trắng, tìm các sự kiện sau:\n"
        "E1: Lân quay vòng trên cột số 4 bằng 2 chân trước rồi tiếp đất. "
        "Khoảnh khắc đầu tiên mà lân bắt đầu xoay vòng.\n"
        "E2: Khoảnh khắc 4 chân hoàn toàn chạm đất đầu tiên.\n"
        "E3: Khoảnh khắc đầu tiên 2 người biểu diễn lân cuối chào ban giám khảo.\n"
        "E4: Sau đó lân tiến lại chào một con rồng. "
        "Khoảnh khắc đầu tiên con rồng cử động đầu.")))
    assert len(muc["sub_events"]) == 4
    assert bg.so_su_kien == 4
    # Nhãn "E1:" bị gỡ — nó là token vô nghĩa với encoder ảnh.
    assert not any(s.startswith("E") and s[1:2].isdigit()
                   for s in muc["sub_events"])
    # Câu dẫn thành ngữ cảnh cấp video ở ĐẦU mỗi sự kiện, và giữ lại nguyên văn.
    assert muc["_mo_dau"].startswith("Đoạn video múa lân")
    assert all(s.startswith("Đoạn video múa lân") for s in muc["sub_events"])
    assert "4 chân hoàn toàn chạm đất" in muc["sub_events"][1]
    assert any("CÂU DẪN" in c for c in bg.canh_bao)


def test_trake_nhan_su_kien_danh_trung_van_ra_du_so(tmp_path):
    """query-p1-18-trake THẬT: BTC đánh nhầm E1/E2/E2/E4. Vẫn phải ra 4."""
    muc, _ = doc_mot_file(_de(tmp_path, "query-p1-18-trake.txt", (
        "Trong đoạn video nấu ăn một món ăn về nấm, gồm các khoảnh khắc sơ chế:\n"
        "E1: Khoảnh khắc đầu tiên thấy cắt nấm.\n"
        "E2: Khoảnh khắc đầu tiên cắt củ năng.\n"
        "E2: Khoảnh khắc đầu tiên cắt đậu hủ.\n"
        "E4: Khoảnh khắc chảo đặt lên bếp, đầu bếp mở lửa và thấy lửa bắt đầu "
        "xuất hiện")))
    assert len(muc["sub_events"]) == 4
    # Thứ tự XUẤT HIỆN, không phải số trong nhãn.
    assert "cắt củ năng" in muc["sub_events"][1]
    assert "cắt đậu hủ" in muc["sub_events"][2]


def test_trake_khong_co_cau_dan_thi_khong_ghep_gi(tmp_path):
    """query-p1-4-trake THẬT: 4 dòng E, không câu dẫn -> không đổi hành vi."""
    muc, bg = doc_mot_file(_de(tmp_path, "query-p1-4-trake.txt", (
        "E1: Khoảnh khắc đầu tiên bột được bỏ vào tô măng tây.\n"
        "E2: Khoảnh khắc đầu tiên thấy miến măng tây đầu tiên tiếp xúc với dầu "
        "trong chảo.\n"
        "E3: Khoảnh khắc miếng măng tây đầu tiên rời khỏi chảo dầu.\n"
        "E4: Khoảng khắc miếng măng tây cuối cùng rời chảo dầu và nằm hoàn toàn "
        "trên dĩa.")))
    assert len(muc["sub_events"]) == 4
    assert "_mo_dau" not in muc
    assert muc["sub_events"][0].startswith("Khoảnh khắc đầu tiên bột")
    assert bg.canh_bao == []


def test_qa_dau_hoi_giua_dong_van_bat_duoc_cau_hoi(tmp_path):
    """query-p1-15-qa THẬT: `... là gì? (tại thời điểm đó)` — `?` không ở cuối."""
    muc, bg = doc_mot_file(_de(tmp_path, "query-p1-15-qa.txt", (
        "Đoạn video về một chương trình từ thiện của một câu lạc bộ tên là FANA. "
        "Trong đoạn video có thể thấy câu lạc bộ này đang đi trao quà tại một xã "
        "thuộc tỉnh Khánh Hòa. Hỏi xã này có tên là gì? (tại thời điểm đó)")))
    assert bg.co_cau_hoi
    assert "tại thời điểm đó" in muc["_cau_hoi_vi"]
    assert not any("không thấy dòng nào" in c for c in bg.canh_bao)


def test_ghi_json_doc_lai_duoc_bang_nop_bai(tmp_path):
    """Vòng tròn khép kín: đọc .txt -> JSON -> `nop_bai.doc_truy_van`."""
    import nop_bai

    _de(tmp_path, "query-1-kis.txt", "Một người đang nấu ăn")
    _de(tmp_path, "query-2-qa.txt", "Một cảnh bếp.\nÁo màu gì?")
    _de(tmp_path, "query-3-trake.txt", "rót dầu\nđảo thức ăn")
    muc, _ = doc_thu_muc(tmp_path)
    ra = ghi_json(muc, tmp_path / "de.json")

    qs, ban_ghi, ten_files = nop_bai.doc_truy_van(str(ra), "en")
    assert [q.qid for q in qs] == ["query-1-kis", "query-2-qa", "query-3-trake"]
    assert ten_files["query-2-qa"] == "query-2-qa.csv"
    assert qs[1].question_vi == "Áo màu gì?"
    assert len(qs[2].sub_events) == 2
    # Đề tiếng Việt -> phải bị đánh dấu lệch chính sách để lớp viết lại vào cuộc.
    assert not ban_ghi[0].theo_chinh_sach


# --------------------------------------------------------------------------- #
# BỘ ĐỀ THẬT — SƠ TUYỂN ĐỢT 1, phát 19:30 ngày 2026-08-21.
# --------------------------------------------------------------------------- #

DE_TRAKE_DOT1 = 'Đoạn video bắt đầu bằng ảnh cận đầu một con lân trắng, mũi đỏ, bên cạnh lá cờ trắng viền đỏ.\nE1 Khoảnh khắc đầu tiên xuất hiện đầy đủ hai con rồng vàng đang xoay vòng.\nE2 Khoảnh khắc đầu tiên con lân hoàn tất cú xoay người trên các thanh trụ (thời điểm đâu tiên các chân của lân đặt trên trụ sau khi xoay).\nE3 Khoảnh khắc đầu tiên dùi chạm vào kẻng đồng múa lân.'


def test_trake_nhan_KHONG_co_dau_hai_cham_van_dung_so_su_kien(tmp_path):
    """`E1 ` KHÔNG có dấu hai chấm — định dạng MỚI của đợt 1.

    Vòng thử nghiệm BTC ghi `E1: ...`; đợt 1 ghi `E1 ...`. Mẫu cũ đòi dấu phân
    cách nên không khớp dòng nào, rơi về nhánh "mỗi dòng là một sự kiện" và
    đếm CẢ CÂU DẪN -> 4 sự kiện cho câu BTC hỏi 3. Luật ghi rõ số Frame ID
    phải khớp chính xác số events, nên đó là R-Score 0 cho cả truy vấn — mà
    zip vẫn hợp lệ và mọi phép kiểm định dạng vẫn xanh.
    """
    muc, bg = doc_mot_file(_de(tmp_path, "query-p1-16-trake.txt", DE_TRAKE_DOT1))
    assert bg.so_su_kien == 4
    assert "hai con rồng vàng" in muc["sub_events"][1]
    assert "kẻng đồng" in muc["sub_events"][3]
    for i, se in enumerate(muc["sub_events"][1:], 1):
        assert "E{} Khoảnh".format(i) not in se


def test_trake_dot1_dong_khong_nhan_LA_su_kien_dau_tien(tmp_path):
    """PHÁN QUYẾT CỦA BTC, không phải suy đoán.

    Nộp 3 khung cho `query-p1-16-trake` đợt 1, log chấm bài trả về 100 dòng:

        Row 1: Expected 4 frame IDs, got 3
        ...
        Row 100: Expected 4 frame IDs, got 3

    Đề chỉ có 4 dòng (1 không nhãn + E1/E2/E3), nên dòng KHÔNG NHÃN được BTC
    đánh số là sự kiện thứ nhất. Khác câu dẫn vòng thử nghiệm ở chỗ nó kết bằng
    dấu chấm và tả một khoảnh khắc cụ thể, không kết bằng ':' và không tuyên bố
    sắp liệt kê. Đây là test giữ ranh giới đó.
    """
    muc, bg = doc_mot_file(_de(tmp_path, "query-p1-16-trake.txt", DE_TRAKE_DOT1))
    assert len(muc["sub_events"]) == 4, muc["sub_events"]
    assert muc["sub_events"][0].startswith("Đoạn video bắt đầu bằng ảnh cận đầu")
    assert "con lân trắng" in muc["sub_events"][0]
    # Không có câu dẫn -> không ghép tiền tố vào các sự kiện sau.
    assert "_mo_dau" not in muc
    assert not muc["sub_events"][1].startswith("Đoạn video")
    assert any("KHÔNG mang nhãn" in c for c in bg.canh_bao), bg.canh_bao


def test_trake_cau_dan_ket_bang_hai_cham_van_KHONG_la_su_kien(tmp_path):
    """Chiều ngược lại: `query-p1-16-trake` vòng THỬ NGHIỆM vẫn phải ra 4.

    Nới luật để bắt đợt 1 mà làm hỏng vòng thử nghiệm thì chỉ là dời chỗ lỗi.
    """
    muc, bg = doc_mot_file(_de(tmp_path, "query-p1-16-trake.txt", (
        "Đoạn video múa lân một con lân màu vàng đen trắng, tìm các sự kiện sau:\n"
        "E1: Lân quay vòng trên cột số 4 bằng 2 chân trước rồi tiếp đất.\n"
        "E2: Khoảnh khắc 4 chân hoàn toàn chạm đất đầu tiên.\n"
        "E3: Khoảnh khắc đầu tiên 2 người biểu diễn lân cuối chào ban giám khảo.\n"
        "E4: Khoảnh khắc đầu tiên con rồng cử động đầu.")))
    assert len(muc["sub_events"]) == 4, muc["sub_events"]
    assert muc["_mo_dau"].endswith("tìm các sự kiện sau:")
    assert all(x.startswith("Đoạn video múa lân") for x in muc["sub_events"])
    assert not any("KHÔNG mang nhãn" in c for c in bg.canh_bao), bg.canh_bao


def test_trake_cau_dan_ket_bang_hai_cham_kieu_nau_an(tmp_path):
    """`query-p1-18-trake` vòng thử nghiệm: câu dẫn 'gồm các khoảnh khắc sơ chế:'."""
    muc, _ = doc_mot_file(_de(tmp_path, "query-p1-18-trake.txt", (
        "Trong đoạn video nấu ăn một món ăn vị nấm, gồm các khoảnh khắc sơ chế:\n"
        "E1: Khoảnh khắc đầu tiên thấy cắt nấm.\n"
        "E2: Khoảnh khắc đầu tiên cắt củ năng.\n"
        "E2: Khoảnh khắc đầu tiên cắt đậu hũ.\n"
        "E4: Khoảnh khắc chảo đặt lên bếp, đầu bếp mở lửa.")))
    assert len(muc["sub_events"]) == 4, muc["sub_events"]
    assert muc["_mo_dau"].endswith("sơ chế:")


def test_nhan_su_kien_phai_co_khoang_trang_khi_thieu_dau_phan_cach():
    """`E1Khoảnh` KHÔNG phải nhãn — nới lỏng quá tay là hiểu nhầm chiều kia."""
    from aic2026.core.doc_de_txt import MAU_NHAN_SU_KIEN
    assert MAU_NHAN_SU_KIEN.match("E1 Khoảnh khắc")
    assert MAU_NHAN_SU_KIEN.match("E1: Khoảnh khắc")
    assert MAU_NHAN_SU_KIEN.match("E 2 . Khoảnh khắc")
    assert not MAU_NHAN_SU_KIEN.match("E1Khoảnh khắc")
    assert not MAU_NHAN_SU_KIEN.match("Em bé đang chơi")
