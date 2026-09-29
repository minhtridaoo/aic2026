"""Test kho vector nền numpy và bộ encode truy vấn CLIP.

Cả hai chỉ dùng khi chạy dữ liệu thật, nên đều bọc `importorskip` để bộ test
vẫn chạy trên máy chỉ có stdlib.
"""
import pytest

np = pytest.importorskip("numpy")

from aic2026.core.providers import (CLIPTextEncoder, MCLIPTextEncoder,
                                    build_encoder)
from aic2026.core.retrieval import MultiChannelRetriever
from aic2026.core.models import Event
from aic2026.core.stores import InMemoryVectorStore, NumpyVectorStore


def vec(*x):
    v = np.zeros(512, dtype=np.float32)
    for i, gt in enumerate(x):
        v[i] = gt
    return v


def ev(vid, eid, kf=0):
    return Event(video_id=vid, event_id=eid, t_start=0.0, t_end=1.0,
                 start_frame=kf, end_frame=kf + 10, keyframe_idx=kf)


# --------------------------------------------------------------------------- #
# NumpyVectorStore
# --------------------------------------------------------------------------- #
def test_add_many_roi_tim_dung_thu_tu():
    s = NumpyVectorStore()
    s.add_many(["a", "b", "c"], np.stack([vec(1), vec(0, 1), vec(0.7, 0.7)]))
    kq = s.search(vec(1), top_k=3)
    assert [k for k, _ in kq] == ["a", "c", "b"]
    assert kq[0][1] == pytest.approx(1.0, abs=1e-6)
    assert len(s) == 3


def test_add_le_van_chay_va_gom_bo_dem():
    """add() lẻ chỉ gom vào bộ đệm; ghép ma trận sau mỗi add sẽ là O(N²)."""
    s = NumpyVectorStore()
    for i, k in enumerate("abcd"):
        s.add(k, vec(*([0] * i + [1])))
    assert len(s) == 4
    assert s.search(vec(0, 0, 1), top_k=1)[0][0] == "c"


def test_tron_add_le_va_add_many():
    s = NumpyVectorStore()
    s.add_many(["a"], np.stack([vec(1)]))
    s.add("b", vec(0, 1))
    kq = dict(s.search(vec(0, 1), top_k=2))
    assert kq["b"] == pytest.approx(1.0, abs=1e-6)
    assert len(s) == 2


def test_bao_loi_khi_lech_so_luong():
    s = NumpyVectorStore()
    with pytest.raises(ValueError, match="lệch số lượng"):
        s.add_many(["a", "b"], np.stack([vec(1)]))


def test_bao_loi_khi_truy_van_sai_so_chieu():
    s = NumpyVectorStore()
    s.add_many(["a"], np.stack([vec(1)]))
    with pytest.raises(ValueError, match="số chiều lệch"):
        s.search(np.ones(8, dtype=np.float32), top_k=1)


def test_kho_rong_tra_ve_rong():
    assert NumpyVectorStore().search(vec(1), top_k=5) == []


def test_top_k_lon_hon_so_vector_khong_no():
    s = NumpyVectorStore()
    s.add_many(["a", "b"], np.stack([vec(1), vec(0, 1)]))
    assert len(s.search(vec(1), top_k=99)) == 2


def test_tu_chuan_hoa_khi_nguon_chua_chuan():
    s = NumpyVectorStore(normalized=False)
    s.add_many(["a", "b"], np.stack([vec(5), vec(0, 0.1)]))   # độ dài rất khác nhau
    kq = dict(s.search(vec(0, 1), top_k=2))
    assert kq["b"] == pytest.approx(1.0, abs=1e-6)   # hướng mới quyết định, không phải độ dài


# --------------------------------------------------------------------------- #
# Khử modality gap
# --------------------------------------------------------------------------- #
def test_khu_gap_loai_bo_huong_chung():
    """Khi mọi vector có chung một hướng lớn, hướng đó át hết nội dung.

    Đây chính là cái xảy ra với CLIP: vector ảnh và vector chữ nằm ở hai nón
    tách rời, nên cosine bị chi phối bởi hướng trung bình chứ không phải nội
    dung. Chiếu bỏ hướng chung thì phần khác biệt mới lộ ra.
    """
    chung = vec(0, 0, 0, 10)                 # thành phần chung, rất lớn
    a = chung + vec(1)                       # khác nhau ở chiều 0
    b = chung + vec(0, 1)                    # khác nhau ở chiều 1

    q = chung + vec(0, 3)                    # truy vấn nghiêng hẳn về b

    tho = NumpyVectorStore(normalized=False)
    tho.add_many(["a", "b"], np.stack([a, b]))
    d_tho = dict(tho.search(q, top_k=2))
    tach_tho = abs(d_tho["a"] - d_tho["b"])

    khu = NumpyVectorStore(normalized=False, khu_modality_gap=True)
    khu.add_many(["a", "b"], np.stack([a, b]))
    kq = khu.search(q, top_k=2)
    tach_khu = kq[0][1] - kq[1][1]

    # Điều cần khẳng định là SO SÁNH, không phải một ngưỡng tuyệt đối tuỳ tiện:
    # hướng chung át nội dung, bỏ nó đi thì khoảng cách giữa hai ứng viên giãn ra.
    assert kq[0][0] == "b"                   # chọn đúng ứng viên
    assert tach_khu > 10 * tach_tho, f"thô={tach_tho:.4f} khử={tach_khu:.4f}"


def test_khu_gap_van_giu_dung_so_luong_va_khoa():
    s = NumpyVectorStore(khu_modality_gap=True)
    s.add_many(list("abcd"), np.stack([vec(1), vec(0, 1), vec(0, 0, 1), vec(1, 1)]))
    kq = s.search(vec(1), top_k=4)
    assert len(kq) == 4 and {k for k, _ in kq} == set("abcd")


def test_them_du_lieu_thi_tinh_lai_huong_trung_binh():
    """Hướng trung bình phụ thuộc toàn bộ kho -> thêm vector phải làm mới cache,
    nếu không sẽ chiếu bỏ nhầm hướng cũ."""
    s = NumpyVectorStore(khu_modality_gap=True)
    s.add_many(["a", "b"], np.stack([vec(1), vec(0, 1)]))
    s.search(vec(1), top_k=1)                         # ép tính cache lần đầu
    mu_cu = s._mu.copy()
    s.add_many(["c"], np.stack([vec(0, 0, 1)]))
    s.search(vec(1), top_k=1)
    assert not np.allclose(mu_cu, s._mu)


def test_kho_suy_bien_khong_no_khi_khu_gap():
    """Mọi vector giống hệt nhau -> hướng trung bình là chính nó, chiếu bỏ xong
    còn vector 0. Không được chia cho 0."""
    s = NumpyVectorStore(khu_modality_gap=True)
    s.add_many(["a", "b"], np.stack([vec(1), vec(1)]))
    assert len(s.search(vec(1), top_k=2)) == 2


def test_cung_ket_qua_voi_ban_stdlib():
    """Bản numpy và bản stdlib phải xếp hạng giống nhau, nếu không thì đổi store
    sẽ âm thầm đổi kết quả thi."""
    khoa = ["a", "b", "c", "d"]
    mat = np.stack([vec(1), vec(0.9, 0.4), vec(0, 1), vec(0.5, 0.5, 0.7)])
    nps = NumpyVectorStore()
    nps.add_many(khoa, mat)
    std = InMemoryVectorStore()
    for k, v in zip(khoa, mat):
        std.add(k, [float(x) for x in v])
    q = vec(0.8, 0.3)
    assert [k for k, _ in nps.search(q, 4)] == [k for k, _ in std.search(list(q), 4)]


# --------------------------------------------------------------------------- #
# Đường nạp hàng loạt của retriever
# --------------------------------------------------------------------------- #
def test_retriever_nap_hang_loat_bang_ma_tran():
    r = MultiChannelRetriever(encoder=None, vector_store=NumpyVectorStore())
    events = [ev("L01", "L01#1", 10), ev("L01", "L01#2", 20), ev("L02", "L02#1", 30)]
    r.index(events, embeddings=np.stack([vec(1), vec(0, 1), vec(0.6, 0.8)]))
    assert len(r.visual) == 3
    assert r.events["L01#2"].keyframe_idx == 20
    assert r.visual.search(vec(0, 1), top_k=1)[0][0] == "L01#2"


def test_retriever_bat_loi_lech_so_luong_embedding():
    r = MultiChannelRetriever(encoder=None, vector_store=NumpyVectorStore())
    with pytest.raises(ValueError, match="lệch số lượng"):
        r.index([ev("L01", "L01#1")], embeddings=np.stack([vec(1), vec(0, 1)]))


# --------------------------------------------------------------------------- #
# CLIPTextEncoder
# --------------------------------------------------------------------------- #
def test_cache_ghi_roi_doc_lai(tmp_path):
    p = tmp_path / "q.npz"
    texts = ["người đàn ông mở container", "cảng bến xe tải"]
    vecs = np.stack([vec(1), vec(0, 1)])
    CLIPTextEncoder.luu_cache(p, texts, vecs)

    enc = CLIPTextEncoder(cache_path=p, cho_phep_live=False)
    ra = enc.encode_text("người đàn ông mở container")
    assert len(ra) == 512 and ra[0] == pytest.approx(1.0)
    assert enc.so_lan_cache_trung == 1


def test_cache_bo_qua_khoang_trang_thua(tmp_path):
    p = tmp_path / "q.npz"
    CLIPTextEncoder.luu_cache(p, ["cảng bến xe tải"], np.stack([vec(1)]))
    enc = CLIPTextEncoder(cache_path=p, cho_phep_live=False)
    assert enc.encode_text("  cảng   bến  xe tải  ")[0] == pytest.approx(1.0)


def test_khong_co_trong_cache_thi_NEM_LOI_chu_khong_doan_bua(tmp_path):
    """Đây là điểm khác cốt lõi so với HashEmbedder.

    HashEmbedder luôn trả về một vector trông hợp lệ, nên hệ thống vẫn in ra
    điểm số đẹp trong khi thực chất không tìm kiếm gì. Encoder này phải gãy to.
    """
    p = tmp_path / "q.npz"
    CLIPTextEncoder.luu_cache(p, ["đã có"], np.stack([vec(1)]))
    enc = CLIPTextEncoder(cache_path=p, cho_phep_live=False)
    with pytest.raises(LookupError, match="Không encode được truy vấn"):
        enc.encode_text("câu chưa từng encode")


def test_cache_sai_so_chieu_bi_tu_choi(tmp_path):
    p = tmp_path / "q.npz"
    np.savez_compressed(str(p), keys=np.array(["a"]),
                        vectors=np.zeros((1, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="chiều"):
        CLIPTextEncoder(cache_path=p, cho_phep_live=False)


def test_cache_lech_so_luong_bi_tu_choi(tmp_path):
    p = tmp_path / "q.npz"
    np.savez_compressed(str(p), keys=np.array(["a", "b"]),
                        vectors=np.zeros((1, 512), dtype=np.float32))
    with pytest.raises(ValueError, match="cache lệch"):
        CLIPTextEncoder(cache_path=p, cho_phep_live=False)


def test_so_chieu_khop_voi_vector_cua_BTC():
    """512 — cùng không gian với clip-features-32 mà BTC cấp. Lệch là vô nghĩa."""
    assert CLIPTextEncoder.DIM == 512


# --------------------------------------------------------------------------- #
# MCLIPTextEncoder + khoá chống trộn hai text tower
# --------------------------------------------------------------------------- #
def test_mclip_cung_so_chieu_nhung_khac_model():
    """Cùng 512 chiều, cùng ghép được với vector ảnh BTC — nên KHÔNG có triệu
    chứng nào nếu trộn nhầm. Đó chính là lý do phải khoá bằng tên model."""
    assert MCLIPTextEncoder.DIM == CLIPTextEncoder.DIM == 512
    assert MCLIPTextEncoder.MODEL_ID != CLIPTextEncoder.MODEL_ID


def test_cache_cua_clip_khong_nap_duoc_vao_mclip(tmp_path):
    p = tmp_path / "q.npz"
    CLIPTextEncoder.luu_cache(p, ["bãi biển"], np.stack([vec(1)]))
    with pytest.raises(ValueError, match="không so sánh được"):
        MCLIPTextEncoder(cache_path=p, cho_phep_live=False)


def test_cache_cua_mclip_nap_lai_duoc_vao_chinh_no(tmp_path):
    p = tmp_path / "q.npz"
    MCLIPTextEncoder.luu_cache(p, ["bãi biển"], np.stack([vec(1)]))
    enc = MCLIPTextEncoder(cache_path=p, cho_phep_live=False)
    assert enc.encode_text("bãi biển")[0] == pytest.approx(1.0)


def test_cache_cu_khong_ghi_ten_model_van_nap_duoc(tmp_path):
    """File encode từ trước khi có khoá này không được biến thành lỗi cứng."""
    p = tmp_path / "q.npz"
    np.savez_compressed(str(p), keys=np.array(["bãi biển"]),
                        vectors=np.stack([vec(1)]))
    enc = CLIPTextEncoder(cache_path=p, cho_phep_live=False)
    assert enc.encode_text("bãi biển")[0] == pytest.approx(1.0)


def test_loi_thieu_vector_noi_ro_goi_can_cai(tmp_path):
    """Thông báo lỗi phải chỉ đúng gói của TỪNG encoder, không chỉ chung chung."""
    enc = MCLIPTextEncoder(cho_phep_live=False)
    with pytest.raises(LookupError, match="sentence-transformers"):
        enc.encode_text("câu chưa từng encode")


def _encoder_that(lop):
    """Dựng encoder LIVE thật, hoặc skip nếu máy không tải được model.

    Bộ test phải chạy được trên máy chỉ có stdlib, nên không được biến việc
    thiếu model thành lỗi đỏ.
    """
    enc = lop(cho_phep_live=True)
    if enc._lay_live() is None:
        pytest.skip(f"không dựng được {lop.__name__}: {enc._ly_do_khong_live}")
    return enc


def test_hai_text_tower_phai_nam_cung_khong_gian():
    """REGRESSION — bug project hai lần, 2026-08-08.

    `clip-ViT-B-32-multilingual-v1` được distill để KHỚP text tower của CLIP
    ViT-B/32, nên cùng một câu tiếng Anh phải cho hai vector gần song song.

    Đây là phép kiểm duy nhất bắt được bug đã làm điểm về 0: code cũ nhân
    `text_projection` lên `pooler_output` (ở transformers 5.x nó đã được project
    sẵn), ra vector project HAI LẦN. Vector đó vẫn 512 chiều nên mọi phép kiểm
    shape đều xanh, nhưng cosine với vector đúng chỉ còn 0.096.

    Đo được sau khi sửa: 0.958. Ngưỡng 0.80 để chừa chỗ cho sai khác phiên bản,
    vẫn cách xa mức 0.096 của bản hỏng.
    """
    cau = "a beach with palm trees"
    a = np.asarray(_encoder_that(CLIPTextEncoder).encode_text(cau))
    b = np.asarray(_encoder_that(MCLIPTextEncoder).encode_text(cau))
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos > 0.80, (
        f"cosine CLIP vs mCLIP = {cos:.3f}, quá thấp.\n"
        "Gần như chắc chắn một trong hai tower bị project sai số lần — "
        "xem ghi chú trong CLIPTextEncoder._lay_live.")


def test_vector_live_da_chuan_hoa():
    v = np.asarray(_encoder_that(CLIPTextEncoder).encode_text("a red car"))
    assert len(v) == 512
    assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-5)


def test_build_encoder_dung_ca_hai_ten():
    assert isinstance(build_encoder({"encoder": {"provider": "clip",
                                                 "live": False}}),
                      CLIPTextEncoder)
    enc = build_encoder({"encoder": {"provider": "mclip", "live": False}})
    assert isinstance(enc, MCLIPTextEncoder)
