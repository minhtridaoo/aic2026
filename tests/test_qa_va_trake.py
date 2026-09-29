"""QA zero-shot bằng CLIP + lưới jitter TRAKE.

Cả hai đều là phần vừa được đưa từ 'stub' sang 'chạy được', nên test ở đây khoá
đúng những tính chất khiến chúng KHÔNG còn là stub.
"""
import pytest

from aic2026.core.models import Candidate, Event
from aic2026.core.pipeline import PortfolioConfig
from aic2026.core.providers import HashEmbedder
from aic2026.core import qa
from aic2026.core.qa import (NHOM_MAU, NHOM_SO, ClipZeroShotQA, DichDapAnVi,
                             SinhTheoLuat, dung_sinh_ung_vien)
from aic2026.core.stores import InMemoryVectorStore
from aic2026.optimize.submission_optimizer import TRAKE_TOL_FRAMES


def ev(vid, eid, kf=0):
    return Event(video_id=vid, event_id=eid, t_start=0.0, t_end=1.0,
                 start_frame=kf, end_frame=kf + 10, keyframe_idx=kf)


# --------------------------------------------------------------------------- #
# Sinh ứng viên
# --------------------------------------------------------------------------- #
def test_nhan_dang_cau_hoi_mau():
    assert SinhTheoLuat().sinh("What color is the car?") == NHOM_MAU
    assert SinhTheoLuat().sinh("Áo của người đó màu gì?") == NHOM_MAU


def test_cau_hoi_dem_so_thang_chu_mau_trong_ta_canh():
    """Hồi quy sơ tuyển đợt 1 (21/08): hai câu ĐẾM SỐ nộp ra MÀU = 0 điểm.

    Nguyên văn đề BTC, không rút gọn — chỗ chết người chính là phần tả cảnh có
    chữ "màu" đứng TRƯỚC câu hỏi "bao nhiêu".
    """
    p1_9 = ("Đoạn phim ghi lại cảnh những chiếc xe ô tô lội nước, chiếc xe "
            "màu vàng, màu đỏ và màu đen lần lượt chuẩn bị đi qua cầu. Con số "
            "được ghi trên biển báo bên trái của cây cầu là bao nhiêu?")
    p1_15 = ("Bản đồ phân bố động đất tại một vùng trên thế giới, với một bảng "
             "chú giải phía bên trái gồm các kí hiệu nhiều màu sắc tương ứng "
             "với các vị trí tâm chấn có cấp độ động đất khác nhau. Không tính "
             "bảng chú giải, có bao nhiêu vị trí ghi nhận động đất cấp độ 4?")
    for de in (p1_9, p1_15):
        ra = SinhTheoLuat().sinh(de)
        assert ra == NHOM_SO, f"câu đếm số nhận nhóm sai: {ra[:4]}"
        assert not set(ra) & set(NHOM_MAU)

    # Không được đổi chiều còn lại: câu hỏi MÀU vẫn ra nhóm màu dù cảnh có số.
    assert SinhTheoLuat().sinh(
        "Có 3 người đứng trước cửa, người ở giữa mặc áo màu gì?") == NHOM_MAU


def test_cau_hoi_mo_tra_ve_rong_de_ben_goi_tu_lui():
    """Câu hỏi mở không khớp luật nào -> rỗng. `ClipZeroShotQA` phải tự lùi,
    chứ bộ sinh KHÔNG được bịa ra một nhóm bừa."""
    assert SinhTheoLuat().sinh("Describe what happens in this scene") == []


# --------------------------------------------------------------------------- #
# Phát hiện VLM bịa đáp án (soi_dap_an_vlm)
# --------------------------------------------------------------------------- #
def test_soi_dap_an_vlm_bat_duoc_ca_bia_that_28_08():
    """Số THẬT, chạy trên đề đợt 1 chiều 28/08 bằng `gemini-3.6-flash`."""
    tren_can = [(f"{v}#{i:05d}", "300") for i, v in enumerate(
        ("L29_V008", "L21_V013", "L26_V270", "L21_V001", "L27_V015",
         "L21_V015", "L26_V161", "L26_V499"))]
    nghi = qa.soi_dap_an_vlm(tren_can)
    assert nghi and "8 video" in nghi and "300" in nghi
    assert "ĐỪNG TIN" in nghi


def test_soi_dap_an_vlm_im_lang_voi_model_doc_that():
    """`gemini-3.5-flash-lite`, cùng câu, cùng 10 khung: đáp án đổi theo video."""
    doc_that = [("L21_V006#00025", "10"), ("L21_V010#00191", "5"),
                ("L21_V010#00192", "5"), ("L21_V010#00189", "5"),
                ("L21_V006#00015", "4"), ("L21_V006#00007", "4"),
                ("L21_V006#00016", "4"), ("L21_V010#00190", "5"),
                ("L21_V007#00162", "2"), ("L25_V076#00100", "10")]
    assert qa.soi_dap_an_vlm(doc_that) is None


def test_soi_dap_an_vlm_khong_bao_dong_gia_khi_top_roi_mot_video():
    """Một đáp án là HỢP LÝ khi cả top-10 nằm trong cùng một video — phải im."""
    mot_video = [(f"L26_V178#{i:05d}", "Bánh ít trần") for i in range(10)]
    assert qa.soi_dap_an_vlm(mot_video) is None
    assert qa.soi_dap_an_vlm([]) is None


def test_dung_sinh_ung_vien_khong_no_khi_thieu_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert isinstance(dung_sinh_ung_vien(), SinhTheoLuat)


# --------------------------------------------------------------------------- #
# ClipZeroShotQA
# --------------------------------------------------------------------------- #
def _qa_va_kho():
    enc = HashEmbedder(dim=256)
    kho = InMemoryVectorStore()
    return ClipZeroShotQA(enc, SinhTheoLuat()), kho, enc


def test_khong_bao_gio_tra_dap_an_rong():
    """Đây là lỗi cũ: `RuleBasedProvider.answer` trả '' trên 100/100 dòng vì
    visual/ocr/asr_text đều rỗng, nên QA được 0 điểm tuyệt đối."""
    qa, kho, enc = _qa_va_kho()
    c = Candidate(event=ev("L01", "L01#1"), score=0.9)
    kho.add("L01#1", enc.encode_text("a photo of a red object"))
    ra = qa.tra_loi("What color is it?", [c], kho.lay_vector)
    assert ra and all(a.strip() for a, _ in ra)


def test_xac_suat_cong_lai_bang_1_va_sap_giam_dan():
    qa, kho, enc = _qa_va_kho()
    c = Candidate(event=ev("L01", "L01#1"), score=0.9)
    kho.add("L01#1", enc.encode_text("a photo of a blue object"))
    ra = qa.tra_loi("What color is it?", [c], kho.lay_vector)
    assert sum(p for _, p in ra) == pytest.approx(1.0, abs=1e-6)
    assert [p for _, p in ra] == sorted((p for _, p in ra), reverse=True)


def test_dap_an_DOI_theo_khung_hinh():
    """Phép quyết định: nếu đổi vector ảnh mà đáp án không đổi thì lớp này chỉ
    là bộ sinh chuỗi, không phải QA thị giác.

    Dùng `HashEmbedder` nên 'đúng' ở đây vô nghĩa; điều được khẳng định là ẢNH
    CÓ ẢNH HƯỞNG, không phải ảnh cho ra đáp án đúng.
    """
    qa, kho, enc = _qa_va_kho()
    kho.add("A", enc.encode_text("a photo of a red object"))
    kho.add("B", enc.encode_text("a photo of a green object"))
    a = qa.tra_loi("What color is it?",
                   [Candidate(event=ev("L01", "A"), score=0.9)], kho.lay_vector)
    b = qa.tra_loi("What color is it?",
                   [Candidate(event=ev("L01", "B"), score=0.9)], kho.lay_vector)
    assert a != b, "đáp án không đổi khi đổi khung -> vector ảnh không tham gia"


def test_khung_khong_co_vector_thi_van_tra_loi_duoc():
    qa, kho, _ = _qa_va_kho()
    c = Candidate(event=ev("L01", "thiếu"), score=0.9)
    ra = qa.tra_loi("What color is it?", [c], kho.lay_vector)
    assert ra and ra[0][0]


def test_dich_dap_an_chi_sau_khi_cham_va_phan_biet_hai_mau_xanh():
    loc = DichDapAnVi()
    assert loc.dich_mot("green") == "xanh lá"
    assert loc.dich_mot("blue") == "xanh dương"
    assert loc.dich_mot("two") == "2"


def test_clip_zero_shot_xuat_tieng_viet_nhung_van_cham_ung_vien_tieng_anh():
    enc = HashEmbedder(dim=256)
    kho = InMemoryVectorStore()
    qa = ClipZeroShotQA(enc, SinhTheoLuat(), dich_dap_an=DichDapAnVi())
    kho.add("A", enc.encode_text("a photo of a green object"))
    ra = qa.tra_loi("What color is it?",
                   [Candidate(event=ev("L01", "A"), score=0.9)], kho.lay_vector)
    assert ra[0][0] == "xanh lá"


# --------------------------------------------------------------------------- #
# Kho vector: lay_vector
# --------------------------------------------------------------------------- #
def test_lay_vector_tra_dung_vector_da_nap():
    kho = InMemoryVectorStore()
    kho.add("x", [1.0, 0.0])
    assert list(kho.lay_vector("x")) == [1.0, 0.0]
    assert kho.lay_vector("chưa có") is None


def test_numpy_store_lay_vector():
    np = pytest.importorskip("numpy")
    from aic2026.core.stores import NumpyVectorStore
    s = NumpyVectorStore()
    v = np.zeros((2, 512), dtype=np.float32)
    v[0, 0] = 1.0
    v[1, 1] = 1.0
    s.add_many(["a", "b"], v)
    assert float(s.lay_vector("b")[1]) == pytest.approx(1.0)
    assert s.lay_vector("không có") is None


# --------------------------------------------------------------------------- #
# Lưới jitter TRAKE
# --------------------------------------------------------------------------- #
def test_jitter_mac_dinh_khong_con_la_mot_diem():
    """`(0,)` nghĩa là không rải gì — mà keyframe cách nhau trung vị 96 khung
    trong khi cửa sổ chấm dưới 10, nên nộp một điểm là bỏ trắng 88% khoảng."""
    j = PortfolioConfig().trake_jitters
    assert len(j) > 1


def test_jitter_dau_tien_la_0():
    """Final lấy max trên tiền tố, slot 1 đáng 5× slot 51 -> độ lệch khả dĩ nhất
    phải đứng đầu."""
    assert PortfolioConfig().trake_jitters[0] == 0


def test_buoc_jitter_khong_nho_hon_be_rong_cua_so():
    """Bước nhỏ hơn 2·tol thì hai dòng liền nhau phủ trùng nhau -> optimizer
    tính lợi ích biên bằng 0 và không bao giờ tiêu slot cho dòng sau."""
    j = sorted(PortfolioConfig().trake_jitters)
    buoc = min(b - a for a, b in zip(j, j[1:]))
    assert buoc >= TRAKE_TOL_FRAMES, f"bước {buoc} < tol {TRAKE_TOL_FRAMES}"


def test_jitter_doi_xung_quanh_0():
    """Keyframe gần nhất có thể nằm trước HOẶC sau thời điểm đúng."""
    j = set(PortfolioConfig().trake_jitters)
    assert {-x for x in j} == j
