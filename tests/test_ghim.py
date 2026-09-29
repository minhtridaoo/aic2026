"""Lớp GHIM VIDEO — tri thức người vận hành đã xác minh, ghép vào bài nộp.

Ca thật đứng sau bộ test này (đề vòng thử nghiệm 20/08): `query-p1-19-qa` hỏi
hai câu thơ trong đình thần Nguyễn Trung Trực. `media-info` của BTC có ĐÚNG MỘT
video khớp cụm đó, còn tầng truy hồi ảnh không đưa nó vào 100 dòng nào. Ghim là
đường duy nhất để câu đó không bằng 0.
"""
import json

import pytest

from aic2026.core import ghim as G
from aic2026.core.models import Event


def _ev(video, n):
    return Event(video_id=video, event_id=f"{video}#{n:05d}",
                 t_start=float(n), t_end=float(n + 1),
                 start_frame=n * 25, end_frame=n * 25 + 24,
                 keyframe_idx=n * 25)


def _ghi(tmp_path, data):
    p = tmp_path / "ghim.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------------- #
# Đọc file ghim
# --------------------------------------------------------------------------- #
def test_doc_ghim_chap_ca_dang_so_it_va_so_nhieu(tmp_path):
    p = _ghi(tmp_path, {
        "query-p1-19-qa": {"videos": ["l24_v011"], "answer": "hai câu thơ"},
        "query-p1-4-trake": {"video": "L26_V194"},
        "query-p1-2-kis": "L21_V029",
    })
    g = G.doc_ghim(p)
    assert g["query-p1-19-qa"].videos == ["L24_V011"]      # đã chuẩn hoá hoa
    assert g["query-p1-19-qa"].answer == "hai câu thơ"
    assert g["query-p1-4-trake"].videos == ["L26_V194"]
    assert g["query-p1-2-kis"].videos == ["L21_V029"]


def test_doc_ghim_chan_dap_an_qua_100_ky_tu(tmp_path):
    """Luật BTC: answer tối đa 100 ký tự. Chặn ở đây, đừng để zip hỏng."""
    p = _ghi(tmp_path, {"q": {"videos": ["L1_V1"], "answer": "x" * 101}})
    with pytest.raises(ValueError, match="100"):
        G.doc_ghim(p)


# --------------------------------------------------------------------------- #
# Cắt chỉ mục
# --------------------------------------------------------------------------- #
def test_loc_video_giu_dung_cap_event_embedding():
    """Lệch một hàng = gán nhầm vector cho keyframe, và sai hoàn toàn im lặng."""
    np = pytest.importorskip("numpy")
    events = [_ev("L21_V001", 0), _ev("L24_V011", 0),
              _ev("L21_V001", 1), _ev("L24_V011", 1)]
    emb = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
    ev, em = G.loc_video(events, emb, ["L24_V011"])
    assert [e.event_id for e in ev] == ["L24_V011#00000", "L24_V011#00001"]
    assert em.tolist() == [[3, 4, 5], [9, 10, 11]]


def test_loc_video_khong_khop_thi_no_ngay_chu_khong_tra_rong():
    events = [_ev("L21_V001", 0)]
    with pytest.raises(LookupError, match="L99_V999"):
        G.loc_video(events, None, ["L99_V999"])


# --------------------------------------------------------------------------- #
# Ghép
# --------------------------------------------------------------------------- #
def test_gop_dat_dong_ghim_len_dau_va_giu_lai_bai_cua_may():
    ghim = [{"video_id": "L24_V011", "frame_idx": 100}]
    goc = [{"video_id": "L28_V007", "frame_idx": 5},
           {"video_id": "L28_V007", "frame_idx": 9}]
    ra = G.gop(ghim, goc, "kis")
    assert [r["video_id"] for r in ra] == ["L24_V011", "L28_V007", "L28_V007"]


def test_gop_khu_trung_de_khong_dot_slot():
    """`R@k` lấy max trên tiền tố -> dòng trùng là một slot vứt đi."""
    r = {"video_id": "L24_V011", "frame_idx": 100}
    ra = G.gop([r], [dict(r), {"video_id": "L24_V011", "frame_idx": 200}], "kis")
    assert len(ra) == 2


def test_gop_cat_dung_100_dong():
    ghim = [{"video_id": "A", "frame_idx": i} for i in range(60)]
    goc = [{"video_id": "B", "frame_idx": i} for i in range(80)]
    assert len(G.gop(ghim, goc, "kis")) == 100


def test_gop_de_dap_an_len_moi_dong_qa():
    ghim = [{"video_id": "L24_V011", "frame_idx": 1, "answer": "sai"}]
    goc = [{"video_id": "L28_V007", "frame_idx": 2, "answer": "cũng sai"}]
    ra = G.gop(ghim, goc, "qa", answer="Hỏa hồng Nhật Tảo")
    assert {r["answer"] for r in ra} == {"Hỏa hồng Nhật Tảo"}


def test_gop_khong_de_dap_an_len_kis():
    """Chỉ QA mới có cột answer; đè nhầm sang KIS là thêm cột thứ ba, hỏng CSV."""
    ra = G.gop([], [{"video_id": "A", "frame_idx": 1}], "kis", answer="x")
    assert "answer" not in ra[0]


def test_gop_trake_khu_trung_theo_CA_chuoi_khung():
    """Hai alignment cùng video nhưng khác khung là hai giả thuyết KHÁC nhau."""
    a = {"video_id": "V", "frames": [1, 2, 3, 4]}
    b = {"video_id": "V", "frames": [1, 2, 3, 5]}
    assert len(G.gop([a], [dict(a), b], "trake")) == 2


def test_hang_dau_tien_bao_dung_vi_tri_truoc_khi_ghim():
    goc = [{"video_id": "X", "frame_idx": 1}, {"video_id": "L24_V011",
                                               "frame_idx": 2}]
    assert G._hang_dau_tien(goc, "kis", ["L24_V011"]) == 2
    assert G._hang_dau_tien(goc, "kis", ["L99_V999"]) is None


def test_so_dong_mac_dinh_50_va_chan_gia_tri_vo_ly(tmp_path):
    """Núm đặt cược: ghim đúng ăn 4/5 số hạng, ghim sai vẫn còn k=100."""
    p = _ghi(tmp_path, {"q": {"videos": ["L1_V1"]}})
    assert G.doc_ghim(p)["q"].so_dong == 50
    p2 = _ghi(tmp_path, {"q": {"videos": ["L1_V1"], "so_dong": 100}})
    assert G.doc_ghim(p2)["q"].so_dong == 100
    for xau in (0, 101, -5):
        with pytest.raises(ValueError, match="so_dong"):
            G.doc_ghim(_ghi(tmp_path, {"q": {"videos": ["L1_V1"],
                                             "so_dong": xau}}))


def test_doc_ghim_bo_qua_khoa_chu_thich(tmp_path):
    """File ghim phải chứa được LÝ DO; chú thích không được thành truy vấn."""
    p = _ghi(tmp_path, {"_ghi_chu": "vì sao ghim", "q": {"videos": ["L1_V1"]}})
    assert list(G.doc_ghim(p)) == ["q"]


# --------------------------------------------------------------------------- #
# `index()` CỘNG DỒN — cái bẫy làm lượt ghim đầu tiên ngày 20/08 vô hiệu.
# --------------------------------------------------------------------------- #
def test_reset_roi_index_moi_thuc_su_thay_chi_muc():
    """Không reset thì kho cũ còn nguyên và 'ghim' không ghim gì cả."""
    from aic2026.core.retrieval import MultiChannelRetriever
    from aic2026.core.stores import InMemoryVectorStore

    class Enc:
        dim = 2

        def encode_text(self, t):
            return [1.0, 0.0]

    r = MultiChannelRetriever(Enc(), InMemoryVectorStore())
    cu = [_ev("L28_V007", i) for i in range(3)]
    r.index(cu, embeddings=[[1.0, 0.0]] * 3)

    moi = [_ev("L24_V011", i) for i in range(2)]
    r.index(moi, embeddings=[[1.0, 0.0]] * 2)
    assert len(r.events) == 5, "index() phải CỘNG DỒN — hợp đồng đang có"

    r.reset()
    r.index(moi, embeddings=[[1.0, 0.0]] * 2)
    assert len(r.events) == 2
    assert {c.event.video_id for c in r.retrieve("x", top_k=10)} == {"L24_V011"}


def test_numpy_store_reset_xoa_ca_huong_trung_binh():
    """Quên `_mu` là kho mới bị chiếu bỏ hướng trung bình của kho CŨ."""
    np = pytest.importorskip("numpy")
    from aic2026.core.stores import NumpyVectorStore

    s = NumpyVectorStore(khu_modality_gap=True)
    s.add_many(["a", "b"], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    s.search([1.0, 0.0], 2)                     # ép dựng _mu / _mat_khu
    assert s._mu is not None
    s.reset()
    assert s._keys == [] and s._mat is None and s._mu is None
    assert s._mat_khu is None and s._cho_nap == []
    assert s.search([1.0, 0.0], 2) == []


# --------------------------------------------------------------------------- #
# GHIM THEO KHUNG — lựa chọn của mắt người từ `xem_anh.py`.
# --------------------------------------------------------------------------- #
def test_doc_khung_chap_ca_list_va_chuoi_gach_dung(tmp_path):
    """Trang xem ảnh sinh ra dạng "VIDEO|frame"; gõ tay hay ra dạng list."""
    p = _ghi(tmp_path, {"q1": {"khung": [["l26_v004", "4803"], "L26_V004|361"]},
                        "q2": {"khung": [["V", 1, 2, 3, 4]]}})
    g = G.doc_ghim(p)
    assert g["q1"].khung == [["L26_V004", 4803], ["L26_V004", 361]]
    assert g["q2"].khung == [["V", 1, 2, 3, 4]]


def test_khung_sai_dinh_dang_thi_no_ngay(tmp_path):
    with pytest.raises(ValueError, match="khung"):
        G.doc_ghim(_ghi(tmp_path, {"q": {"khung": [["L1_V1"]]}}))


def test_row_tu_khung_dung_dang_tung_task():
    assert G._row_tu_khung("kis", ["V", 7]) == {"video_id": "V", "frame_idx": 7}
    assert G._row_tu_khung("qa", ["V", 7], "đỏ") == {
        "video_id": "V", "frame_idx": 7, "answer": "đỏ"}
    assert G._row_tu_khung("trake", ["V", 1, 2, 3, 4]) == {
        "video_id": "V", "frames": [1, 2, 3, 4]}


def test_row_tu_khung_chan_nhieu_frame_cho_kis():
    """Nhiều frame trên một dòng KIS = CSV thừa cột = cả file hỏng."""
    with pytest.raises(ValueError, match="MỘT frame"):
        G._row_tu_khung("kis", ["V", 1, 2])


def test_khoa_la_trong_file_ghim_thi_no_ngay(tmp_path):
    """Khoá gõ sai từng làm ghim mất tác dụng mà KHÔNG báo gì (28/08).

    `answers` số nhiều bị bỏ qua im lặng vì bộ đọc chỉ nhận `answer` — file ghim
    ghi 2 ứng viên đáp án cho `query-p2-23-qa` mà bài nộp chạy như không có.
    """
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"q": {"videos": ["V1"], "dap_an": "15"}}),
                 encoding="utf-8")
    with pytest.raises(ValueError, match="khoá không hiểu"):
        G.doc_ghim(str(p))

    # `_` mở đầu = ghi chú tự do của người vận hành, phải được phép.
    p.write_text(json.dumps({"_ghi_chu": "x",
                             "q": {"videos": ["V1"], "_vi_sao": "y"}}),
                 encoding="utf-8")
    assert "q" in G.doc_ghim(str(p))


def test_answers_so_nhieu_la_alias_cua_answer(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps({"q": {"videos": ["V1"],
                                   "answers": ["15", "25"]}}), encoding="utf-8")
    g = G.doc_ghim(str(p))["q"]
    assert g.answers == ["15", "25"] and g.answer == ""


def test_va_dap_an_lap_cho_khung_qa_khong_de_dong_rong():
    """Hồi quy sơ tuyển đợt 2 (28/08), `query-p2-23-qa`.

    Ghim `khung` trên câu QA mà không đè `answer` -> `_row_tu_khung` trả
    `answer=''`, và 4 dòng ĐẦU của bài nộp ra `L25_V012,15268,` = 0 điểm chắc
    chắn. `validate_csv` không coi trường rỗng là lỗi định dạng nên zip vẫn báo
    'hợp lệ' — đúng chủng loại lỗi câm. Máy vốn đã để đúng video ở hạng 1 với
    đáp án '15', nên lỗi này đổi 1,0 điểm thành 0.
    """
    g = G.GhimTruyVan(qid="q", videos=["L25_V012"],
                      khung=[["L25_V012", 15268], ["L25_V012", 15118]])
    rows_nguoi = [G._row_tu_khung("qa", k, g.answer) for k in g.khung]
    assert [r["answer"] for r in rows_nguoi] == ["", ""]      # trạng thái lỗi

    G._va_dap_an("qa", rows_nguoi, g,
                 [{"video_id": "L25_V012", "frame_idx": 18892, "answer": "25"}],
                 [{"video_id": "L25_V012", "frame_idx": 14518, "answer": "15"}])
    assert [r["answer"] for r in rows_nguoi] == ["25", "25"]


def test_va_dap_an_lui_ve_bai_goc_va_khong_bia_khi_khong_co_gi():
    g = G.GhimTruyVan(qid="q", khung=[["V", 1]])
    r = [G._row_tu_khung("qa", ["V", 1], "")]
    G._va_dap_an("qa", r, g, (), [{"video_id": "V", "frame_idx": 2,
                                  "answer": "15"}])
    assert r[0]["answer"] == "15"          # lùi về đáp án của bài gốc

    r2 = [G._row_tu_khung("qa", ["V", 1], "")]
    G._va_dap_an("qa", r2, g, (), [{"video_id": "V", "frame_idx": 2,
                                   "answer": ""}])
    assert r2[0]["answer"] == ""           # không nguồn nào -> KHÔNG bịa

    # KIS không có cột đáp án, phải để yên.
    r3 = [G._row_tu_khung("kis", ["V", 1])]
    G._va_dap_an("kis", r3, g, (), [{"video_id": "V", "frame_idx": 2,
                                     "answer": "15"}])
    assert "answer" not in r3[0]


def test_khung_dung_TRUOC_ca_video_ghim_va_bai_cua_may():
    """Người đã NHÌN THẤY ảnh — đó là bằng chứng mạnh nhất đang có."""
    g = G.GhimTruyVan(qid="q", khung=[["L26_V004", 999]])
    goc = [{"video_id": "L26_V004", "frame_idx": 4803}]
    rows, bao = G.ap_dung(None, _Q("kis"), g, goc, [], None)
    assert rows[0] == {"video_id": "L26_V004", "frame_idx": 999}
    assert rows[1]["frame_idx"] == 4803          # bài của máy vẫn còn phía sau
    assert "1 khung người chọn" in bao


class _Q:
    def __init__(self, task):
        self.task = task
        self.qid = "q"


# --------------------------------------------------------------------------- #
# GHIM THEO NHÓM CHƯƠNG TRÌNH — nấc giữa, thêm sau sơ tuyển đợt 1 (3,8/12).
#
# Kho BTC là 10 chương trình chứ không phải 873 phim rời: L24 chỉ có 43 tập lân
# sư rồng, L23 có 25 chặng đua xe đạp. Nhận ra "đây là múa lân" thì chắc chắn,
# còn đoán đúng tập nào trong 43 tập thì không — nên `nhom` gánh phần chắc chắn
# và để encoder làm phần còn lại.
# --------------------------------------------------------------------------- #
def test_doc_ghim_nhan_nhom_dang_chuoi_va_danh_sach(tmp_path):
    g = G.doc_ghim(_ghi(tmp_path, {"q1": {"nhom": "l24"},
                                   "q2": {"nhom": ["L23", "l24"]}}))
    assert g["q1"].nhom == ["L24"] and g["q1"].videos == []
    assert g["q2"].nhom == ["L23", "L24"]


def test_doc_ghim_chan_ten_video_dat_nham_vao_nhom(tmp_path):
    """`L24_V011` trong `nhom` vẫn khớp đúng một video — hỏng mà không ai thấy."""
    with pytest.raises(ValueError, match="TIỀN TỐ"):
        G.doc_ghim(_ghi(tmp_path, {"q": {"nhom": ["L24_V011"]}}))


def test_doc_ghim_chan_dat_ca_videos_lan_nhom(tmp_path):
    with pytest.raises(ValueError, match="không có tác dụng"):
        G.doc_ghim(_ghi(tmp_path, {"q": {"videos": ["L24_V011"],
                                         "nhom": ["L24"]}}))


def test_nhom_no_ra_dung_cac_video_co_tien_to_do():
    """`L24` phải khớp `L24_V*` và KHÔNG khớp `L240_V*` — nên so cả dấu gạch."""
    ev = ([_ev("L24_V011", 1), _ev("L24_V012", 2)]
          + [_ev("L240_V001", 3), _ev("L26_V004", 4)])

    thay = {}

    class _Pipe:
        class retriever:
            @staticmethod
            def reset():
                pass

        @staticmethod
        def index(e, em):
            thay["ev"] = list(e)

        @staticmethod
        def answer(q, top_k):
            return [{"video_id": e.video_id, "frame_idx": e.keyframe_idx}
                    for e in thay["ev"]][:top_k]

    g = G.GhimTruyVan(qid="q", nhom=["L24"], so_dong=50)
    rows, bao = G.ap_dung(_Pipe(), _Q("kis"), g,
                          [{"video_id": "L26_V004", "frame_idx": 100}],
                          ev, None)
    assert {e.video_id for e in thay["ev"]} == {"L24_V011", "L24_V012"}
    assert {r["video_id"] for r in rows[:2]} == {"L24_V011", "L24_V012"}
    assert rows[-1]["video_id"] == "L26_V004", "bài của máy vẫn là lưới an toàn"
    assert "nhóm L24 (2 video)" in bao


def test_answer_dang_danh_sach_rai_luan_phien_xuong_cac_dong(tmp_path):
    """Câu ĐẾM SỐ: một chuỗi đáp án là đặt cược đúng một lần.

    Đợt 1 nộp "trắng" cho câu "có bao nhiêu vị trí động đất cấp độ 4?" — sai
    chắc chắn. Các keyframe liền nhau trong cùng một shot là những dòng hợp lệ
    khác nhau và cùng nằm trong cửa sổ chấm, nên mỗi dòng là một lần đoán riêng.
    """
    g = G.doc_ghim(_ghi(tmp_path, {"q": {"answer": ["3", "4", "5"]}}))["q"]
    assert g.answers == ["3", "4", "5"] and g.answer == ""

    goc = [{"video_id": "V", "frame_idx": i} for i in range(7)]
    rows, _ = G.ap_dung(None, _Q("qa"), g, goc, [], None)
    assert [r["answer"] for r in rows] == ["3", "4", "5", "3", "4", "5", "3"]
    assert len({(r["video_id"], r["frame_idx"]) for r in rows}) == 7


def test_answer_danh_sach_van_bi_chan_boi_gioi_han_100_ky_tu(tmp_path):
    with pytest.raises(ValueError, match="100"):
        G.doc_ghim(_ghi(tmp_path, {"q": {"answer": ["ok", "x" * 101]}}))


def test_answer_danh_sach_khong_dung_cho_kis():
    """KIS không có cột đáp án — rải vào đó là đẻ ra cột thứ ba, sai định dạng."""
    g = G.GhimTruyVan(qid="q", answers=["3", "4"])
    rows, _ = G.ap_dung(None, _Q("kis"), g,
                        [{"video_id": "V", "frame_idx": 1}], [], None)
    assert "answer" not in rows[0]


def test_nhom_khong_khop_video_nao_thi_no_ngay():
    with pytest.raises(LookupError, match="không khớp video nào"):
        G.ap_dung(None, _Q("kis"), G.GhimTruyVan(qid="q", nhom=["L99"]),
                  [], [_ev("L24_V011", 1)], None)
