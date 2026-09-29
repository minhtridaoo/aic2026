"""Bộ feature thay thế (SigLIP2) và encoder không chốt cứng 512 chiều.

Hai thứ được canh ở đây, cả hai đều là loại lỗi KHÔNG có triệu chứng nếu sai:

  1. Trộn hai không gian vector. Vector ảnh CLIP 512 chiều và SigLIP2 768 chiều
     cùng là "vector ảnh", cùng chuẩn L2, cùng ghép được vào `NumpyVectorStore`.
     Nếu một video dùng bộ này và video khác dùng bộ kia thì điểm chỉ tụt, không
     có lỗi nào ném ra.
  2. Cache truy vấn của checkpoint A bị nạp vào encoder của checkpoint B. Nếu hai
     bên tình cờ cùng số chiều thì phép kiểm chiều không bắt được — chỉ khoá
     `model` trong .npz bắt được.
"""
import numpy as np
import pytest

from aic2026.core.dataset import load_dataset, scan
from aic2026.core.providers import (CLIPTextEncoder, MCLIPTextEncoder,
                                    SigLIP2TextEncoder)


# --------------------------------------------------------------------------- #
# Dựng một bộ dữ liệu BTC tí hon trên đĩa
# --------------------------------------------------------------------------- #
def _dung_bo_btc(goc, vids=("L21_V001", "L21_V002"), n_frame=3, dim=512):
    clip = goc / "clip_features" / "clip-features-32"
    mapd = goc / "keyframe_mapping" / "map-keyframes"
    clip.mkdir(parents=True)
    mapd.mkdir(parents=True)
    for k, vid in enumerate(vids):
        m = np.full((n_frame, dim), 0.1 + k, dtype=np.float16)
        np.save(clip / f"{vid}.npy", m)
        dong = ["n,pts_time,fps,frame_idx"]
        for i in range(n_frame):
            dong.append(f"{i + 1},{i * 2.0},25.0,{i * 50}")
        (mapd / f"{vid}.csv").write_text("\n".join(dong), encoding="utf-8")
    return goc


def _dung_bo_feature(goc, vids, n_frame=3, dim=768):
    goc.mkdir(parents=True, exist_ok=True)
    for k, vid in enumerate(vids):
        np.save(goc / f"{vid}.npy",
                np.full((n_frame, dim), 0.2 + k, dtype=np.float16))
    (goc / "_meta.json").write_text('{"dim": %d}' % dim, encoding="utf-8")
    return goc


# --------------------------------------------------------------------------- #
# goc_feature
# --------------------------------------------------------------------------- #
def test_goc_feature_thay_duoc_va_doi_so_chieu(tmp_path):
    btc = _dung_bo_btc(tmp_path / "btc")
    sig = _dung_bo_feature(tmp_path / "sig", ("L21_V001", "L21_V002"))

    _, emb_cu, _, _ = load_dataset(btc)
    assert emb_cu.shape == (6, 512)

    ev, emb, _, rep = load_dataset(btc, goc_feature=sig)
    assert emb.shape == (6, 768), "phải lấy số chiều từ bộ feature mới"
    assert len(ev) == emb.shape[0]
    assert rep.videos_da_nap == 2


def test_video_thieu_feature_bi_bao_thieu_chu_khong_tut_ve_clip_cu(tmp_path):
    """Đây là ca quan trọng nhất của cả file.

    Lặng lẽ dùng lại vector CLIP 512 chiều cho video thiếu feature mới sẽ tạo ra
    ma trận trộn hai không gian. Phải báo THIẾU và bỏ video đó.
    """
    btc = _dung_bo_btc(tmp_path / "btc")
    sig = _dung_bo_feature(tmp_path / "sig", ("L21_V001",))   # thiếu L21_V002

    ev, emb, _, rep = load_dataset(btc, goc_feature=sig)
    assert emb.shape == (3, 768)
    assert rep.videos_da_nap == 1
    assert rep.thieu_file == ["L21_V002"]
    assert len(ev) == emb.shape[0]


def test_hai_npy_cho_cung_video_thi_nem_loi(tmp_path):
    btc = _dung_bo_btc(tmp_path / "btc")
    sig = _dung_bo_feature(tmp_path / "sig", ("L21_V001", "L21_V002"))
    (sig / "ban_du").mkdir()
    np.save(sig / "ban_du" / "L21_V001.npy", np.zeros((3, 768), dtype=np.float16))

    with pytest.raises(ValueError, match="HAI file"):
        scan(btc, goc_feature=sig)


def test_goc_feature_rong_thi_nem_loi(tmp_path):
    btc = _dung_bo_btc(tmp_path / "btc")
    trong = tmp_path / "trong"
    trong.mkdir()
    with pytest.raises(ValueError, match="không thấy file .npy"):
        scan(btc, goc_feature=trong)


def test_goc_feature_khong_ton_tai_thi_nem_loi(tmp_path):
    btc = _dung_bo_btc(tmp_path / "btc")
    with pytest.raises(FileNotFoundError):
        scan(btc, goc_feature=tmp_path / "khong-co")


def test_feature_lech_so_hang_so_voi_csv_thi_bo_video(tmp_path):
    btc = _dung_bo_btc(tmp_path / "btc", n_frame=3)
    sig = tmp_path / "sig"
    sig.mkdir()
    np.save(sig / "L21_V001.npy", np.zeros((3, 768), dtype=np.float16))
    np.save(sig / "L21_V002.npy", np.zeros((5, 768), dtype=np.float16))  # lệch

    _, emb, _, rep = load_dataset(btc, goc_feature=sig)
    assert emb.shape == (3, 768)
    assert rep.lech_so_hang == [("L21_V002", 3, 5)]


# --------------------------------------------------------------------------- #
# Encoder: số chiều và khoá model
# --------------------------------------------------------------------------- #
def _ghi_cache(path, texts, dim, model_id):
    v = np.random.default_rng(0).normal(size=(len(texts), dim)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    np.savez_compressed(str(path),
                        keys=np.array(list(texts), dtype=object).astype(str),
                        vectors=v, model=np.array(model_id))
    return path


def test_siglip2_hoc_so_chieu_tu_cache(tmp_path):
    p = _ghi_cache(tmp_path / "q.npz", ["a", "b"], 768,
                   SigLIP2TextEncoder.MODEL_ID)
    e = SigLIP2TextEncoder(cache_path=p, cho_phep_live=False)
    assert e.dim == 768
    assert len(e.encode_text("a")) == 768


def test_siglip2_so400m_1152_chieu_cung_chay(tmp_path):
    """DIM = None phải chấp nhận checkpoint khác, không chỉ base-224."""
    p = _ghi_cache(tmp_path / "q.npz", ["a"], 1152,
                   "google/siglip2-so400m-patch14-384")
    e = SigLIP2TextEncoder(cho_phep_live=False)
    e.MODEL_ID = "google/siglip2-so400m-patch14-384"
    e.nap_cache(p)
    assert e.dim == 1152


def test_siglip2_chan_tron_hai_checkpoint_khac_chieu(tmp_path):
    a = _ghi_cache(tmp_path / "a.npz", ["x"], 768, SigLIP2TextEncoder.MODEL_ID)
    e = SigLIP2TextEncoder(cache_path=a, cho_phep_live=False)
    b = _ghi_cache(tmp_path / "b.npz", ["y"], 1152, SigLIP2TextEncoder.MODEL_ID)
    with pytest.raises(ValueError, match="đã chốt từ nguồn trước"):
        e.nap_cache(b)


def test_siglip2_chan_cache_cua_model_khac_du_cung_so_chieu(tmp_path):
    """Cùng 768 chiều nhưng khác checkpoint -> chỉ khoá `model` bắt được."""
    p = _ghi_cache(tmp_path / "q.npz", ["a"], 768, "google/siglip2-base-patch32-256")
    e = SigLIP2TextEncoder(cho_phep_live=False)
    with pytest.raises(ValueError, match="text tower"):
        e.nap_cache(p)


def test_clip_van_chot_cung_512(tmp_path):
    """Bỏ hardcode cho SigLIP2 KHÔNG được làm CLIP lỏng tay theo."""
    p = _ghi_cache(tmp_path / "q.npz", ["a"], 768, CLIPTextEncoder.MODEL_ID)
    e = CLIPTextEncoder(cho_phep_live=False)
    with pytest.raises(ValueError, match="DIM cố định"):
        e.nap_cache(p)
    assert CLIPTextEncoder.DIM == 512
    assert MCLIPTextEncoder.DIM == 512


def test_siglip2_khong_encode_duoc_thi_nem_loi_chu_khong_tra_vector_bua(tmp_path):
    p = _ghi_cache(tmp_path / "q.npz", ["co trong cache"], 768,
                   SigLIP2TextEncoder.MODEL_ID)
    e = SigLIP2TextEncoder(cache_path=p, cho_phep_live=False)
    with pytest.raises(LookupError):
        e.encode_text("khong co trong cache")


def test_padding_max_length_duoc_ghi_ro():
    """Quy ước đệm của SigLIP là 64 token cứng — sai thì im lặng kém đi."""
    assert SigLIP2TextEncoder.MAX_LEN == 64


# --------------------------------------------------------------------------- #
# Toàn chuỗi ở 768 chiều
# --------------------------------------------------------------------------- #
def test_toan_chuoi_chay_duoc_o_768_chieu(tmp_path):
    """Chứng minh đường ống KHÔNG chốt cứng 512 ở bất cứ tầng nào.

    Chạy thật: load_dataset(goc_feature=) -> NumpyVectorStore -> Pipeline.answer,
    với vector ảnh 768 chiều và cache truy vấn 768 chiều. Không có phép này thì
    lỗi kích thước chỉ lộ ra sau khi bạn đã đốt 30-60 phút GPU Kaggle.
    """
    from aic2026.core.models import Query
    from aic2026.core.pipeline import Pipeline
    from aic2026.core.providers import RuleBasedProvider
    from aic2026.core.stores import NumpyVectorStore

    btc = _dung_bo_btc(tmp_path / "btc", n_frame=4)
    sig = tmp_path / "sig"
    sig.mkdir()
    rng = np.random.default_rng(7)
    for vid in ("L21_V001", "L21_V002"):
        v = rng.normal(size=(4, 768)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        np.save(sig / f"{vid}.npy", v.astype(np.float16))

    events, emb, _, rep = load_dataset(btc, goc_feature=sig)
    assert emb.shape == (8, 768) and rep.videos_da_nap == 2

    cau = "nguoi dan ong mo cua container"
    enc = SigLIP2TextEncoder(
        cache_path=_ghi_cache(tmp_path / "q.npz", [cau], 768,
                              SigLIP2TextEncoder.MODEL_ID),
        cho_phep_live=False)

    p = Pipeline(enc, RuleBasedProvider(), fusion="rrf", vietnamese_encoder=enc)
    p.retriever.visual = NumpyVectorStore()
    p.index(events, embeddings=emb)
    rows = p.answer(Query(qid="q1", task="kis", text=cau), top_k=5)
    assert rows, "phải trả về ứng viên, không được rỗng"
    # `answer` trả về dòng nộp bài dạng dict {video_id, frame_idx}, không phải Candidate.
    assert all(r["video_id"] in {"L21_V001", "L21_V002"} for r in rows)
    assert len({(r["video_id"], r["frame_idx"]) for r in rows}) == len(rows), \
        "nộp trùng dòng là phí slot — xem luật chấm trong HUONG_DAN.md"


# --------------------------------------------------------------------------- #
# Dev set tiếng Việt (việc 2.6)
# --------------------------------------------------------------------------- #
def _ghi_dev_set(path, muc):
    import json
    path.write_text(json.dumps({"_note": "test", "queries": muc},
                               ensure_ascii=False), encoding="utf-8")
    return str(path)


def test_load_queries_doc_duoc_ban_tieng_viet(tmp_path):
    from aic2026.core.offline_eval import load_queries

    p = _ghi_dev_set(tmp_path / "dev.json", [{
        "qid": "kis-001", "task": "kis",
        "text": "a chef and a woman in a kitchen",
        "_mo_ta_vi": "một đầu bếp và một người phụ nữ trong nhà bếp",
        "gt": {"video_id": "L21_V001", "frame_start": 0, "frame_end": 50},
    }])
    en = load_queries(p)
    vi = load_queries(p, truong_text="_mo_ta_vi")
    assert en[0][0].text.startswith("a chef")
    assert vi[0][0].text.startswith("một đầu bếp")
    # Đáp án phải giữ nguyên, chỉ câu hỏi đổi ngôn ngữ.
    assert en[0][1].video_id == vi[0][1].video_id


def test_thieu_mo_ta_vi_thi_nem_loi_chu_khong_tut_ve_tieng_anh(tmp_path):
    """Tụt về tiếng Anh cho một phần dev set = con số đo ra không nói về ngôn ngữ nào."""
    from aic2026.core.offline_eval import load_queries

    p = _ghi_dev_set(tmp_path / "dev.json", [
        {"qid": "kis-001", "task": "kis", "text": "en one",
         "_mo_ta_vi": "câu một",
         "gt": {"video_id": "L21_V001", "frame_start": 0, "frame_end": 5}},
        {"qid": "kis-002", "task": "kis", "text": "en two", "_mo_ta_vi": "   ",
         "gt": {"video_id": "L21_V002", "frame_start": 0, "frame_end": 5}},
    ])
    assert len(load_queries(p)) == 2          # đường tiếng Anh vẫn chạy
    with pytest.raises(KeyError, match="kis-002"):
        load_queries(p, truong_text="_mo_ta_vi")


def test_moi_bien_the_siglip2_co_thu_muc_va_cache_rieng():
    """Dùng chung đường dẫn giữa base 768 và so400m 1152 là trộn hai không gian."""
    import importlib.util

    goc = __import__("pathlib").Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("ddt", goc / "do_diem_thuc_te.py")
    ddt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ddt)

    bt = ddt.BIEN_THE_SIGLIP2
    assert set(bt) == {"base", "so400m"}
    for khoa in ("model_id", "features", "cache", "cache_vi"):
        gia_tri = [c[khoa] for c in bt.values()]
        assert len(set(gia_tri)) == len(gia_tri), f"`{khoa}` bị dùng chung"
    assert bt["base"]["model_id"] == ddt.SigLIP2TextEncoder.MODEL_ID

    # cache tiếng Việt và tiếng Anh cũng phải khác nhau trong cùng biến thể
    for ten, c in bt.items():
        assert c["cache"] != c["cache_vi"], ten


def test_ten_file_cache_khop_giua_script_encode_va_script_do():
    """Hợp đồng tên file giữa hai script — lệch là chạy xong mới biết.

    `kaggle_encode_truy_van_siglip2.py` sinh tên theo (CHECKPOINT, BAN_TIENG_VIET);
    `do_diem_thuc_te.BIEN_THE_SIGLIP2` khai tên nó sẽ đi tìm. Hai bên lệch nhau thì
    lỗi chỉ lộ ra sau khi đã encode xong, dưới dạng "còn thiếu file" khó hiểu.
    """
    import importlib.util
    import itertools
    import pathlib
    import re

    goc = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("ddt", goc / "do_diem_thuc_te.py")
    ddt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ddt)

    mong = set()
    for c in ddt.BIEN_THE_SIGLIP2.values():
        mong |= {c["cache"], c["cache_vi"]}

    # Lấy hậu tố thật từ chính script encode, không gõ lại bằng tay.
    src = (goc / "kaggle_encode_truy_van_siglip2.py").read_text(encoding="utf-8")
    assert '_HAU_TO = "" if CHECKPOINT == "base" else f"_{CHECKPOINT}"' in src, \
        "cách sinh hậu tố đã đổi, cập nhật lại test này"
    thuc = {
        f"query_vectors_siglip2{'' if ck == 'base' else '_' + ck}"
        f"{'_vi' if vi else ''}.npz"
        for ck, vi in itertools.product(("base", "so400m"), (False, True))
    }
    assert mong == thuc, f"lệch: chỉ có ở do_diem={mong - thuc}, chỉ có ở encode={thuc - mong}"

    # Thư mục feature của hai biến thể cũng không được trùng nhau.
    ck_src = (goc / "kaggle_encode_siglip2.py").read_text(encoding="utf-8")
    assert re.search(r'CHECKPOINT\s*=\s*"(base|so400m)"', ck_src)
    assert '_HAU_TO' in ck_src, "output của kaggle_encode_siglip2 phải đổi theo checkpoint"
