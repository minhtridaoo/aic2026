"""Config phải là NGUỒN SỰ THẬT của đường nộp bài — test hồi quy cho việc 1.5.

Lỗi được khoá lại ở đây (phát hiện khi diễn tập 15/08): script nộp bài tự dựng
`Pipeline` nên đã trôi khỏi cấu hình đã chốt — đáp án QA ra **tiếng Anh** trong
khi `qa.output_language: vi`, chạy base CLIP thay vì SigLIP2 so400m, và bật
portfolio. Zip vẫn hợp lệ nên không phép kiểm định dạng nào bắt được.

Nên các test dưới đây kiểm hai lớp:
  1. `aic2026.config` đọc YAML và dựng đúng thứ config nói (kể cả đáp án tiếng Việt)
  2. `nop_bai.py` KHÔNG được tự dựng `Pipeline` — nó phải đi qua lớp trên
"""
import math
import os
from pathlib import Path

import numpy as np
import pytest

from aic2026 import config as cau_hinh
from aic2026.control.query_compiler import compile_query
from aic2026.core.models import Event, Query
from aic2026.core.providers import CLIPTextEncoder, RuleBasedProvider
from aic2026.core.qa import NHOM_MAU, con_la_tieng_anh

GOC = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Đọc file
# --------------------------------------------------------------------------- #
def _ghi(path: Path, noi_dung: str) -> Path:
    path.write_text(noi_dung, encoding="utf-8")
    return path


def test_extends_gop_sau_va_lop_phu_thang(tmp_path):
    _ghi(tmp_path / "goc.yaml",
         "encoder:\n  provider: siglip2\n  live: true\nqa:\n  vlm_top_n: 10\n")
    con = _ghi(tmp_path / "phu.yaml",
               "extends: goc.yaml\nencoder:\n  provider: clip\n")
    c = cau_hinh.load_config(con)
    assert c["encoder"]["provider"] == "clip"       # lớp phủ thắng
    assert c["encoder"]["live"] is True             # khoá không nhắc thì giữ
    assert c["qa"]["vlm_top_n"] == 10               # cả mục khác cũng giữ
    assert c["_nguon_config"] == str(con)


def test_extends_vong_tron_bao_loi(tmp_path):
    _ghi(tmp_path / "a.yaml", "extends: b.yaml\n")
    _ghi(tmp_path / "b.yaml", "extends: a.yaml\n")
    with pytest.raises(cau_hinh.CauHinhSai, match="vòng tròn"):
        cau_hinh.load_config(tmp_path / "a.yaml")


def test_duong_dan_tuong_doi_quy_ve_goc_repo(tmp_path):
    con = _ghi(tmp_path / "c.yaml",
               "encoder:\n  cache_path: dev_set/x.npz\n"
               "qa:\n  image_roots: [anh/1]\n  dev_image_dir: dev_set/anh\n")
    c = cau_hinh.load_config(con)
    for gia_tri in (c["encoder"]["cache_path"], c["qa"]["dev_image_dir"],
                    c["qa"]["image_roots"][0]):
        assert Path(gia_tri).is_absolute()
        assert str(cau_hinh.goc_repo()) in gia_tri


def test_thieu_file_config_bao_ro(tmp_path):
    with pytest.raises(FileNotFoundError):
        cau_hinh.load_config(tmp_path / "khong-co.yaml")


# --------------------------------------------------------------------------- #
# Dữ liệu và thành phần
# --------------------------------------------------------------------------- #
def test_thieu_btc_data_root_noi_ro_thieu_gi():
    with pytest.raises(cau_hinh.CauHinhSai, match="btc_data_root"):
        cau_hinh.duong_du_lieu({"encoder": {"provider": "clip"}})


def test_siglip2_bat_buoc_co_thu_muc_vector_anh_rieng(tmp_path):
    config = {"offline": {"btc_data_root": str(tmp_path)},
              "encoder": {"provider": "siglip2"}}
    with pytest.raises(cau_hinh.CauHinhSai, match="image_feature_root"):
        cau_hinh.duong_du_lieu(config)


def test_portfolio_tu_config(tmp_path):
    tat = cau_hinh.dung_portfolio({})
    assert tat.enabled is False and tat.calibrator is None
    bat = cau_hinh.dung_portfolio(
        {"portfolio": {"enabled": True, "calibrator": "rank",
                       "kis_tol_frames": 7}})
    assert bat.enabled is True and bat.calibrator is not None
    assert bat.kis_tol_frames == 7
    with pytest.raises(cau_hinh.CauHinhSai, match="calibrator"):
        cau_hinh.dung_portfolio({"portfolio": {"enabled": True,
                                               "calibrator": "bậy"}})


_MOI_KHOA = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY",
             "CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID")


def test_thieu_khoa_api_liet_ke_dung_provider_dang_bat(monkeypatch):
    for bien in _MOI_KHOA:
        monkeypatch.delenv(bien, raising=False)
    config = {"llm": {"provider": "failover_vlm",
                      "primary": {"enabled": True, "provider": "gemini_vlm"},
                      "fallbacks": [{"enabled": False,
                                     "provider": "anthropic_vlm"}]}}
    thieu = cau_hinh.thieu_khoa_api(config)
    assert len(thieu) == 1 and "gemini_vlm" in thieu[0]
    assert "GEMINI_API_KEY" in thieu[0]

    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    assert cau_hinh.thieu_khoa_api(config) == []
    # provider tắt thì không đòi khoá, dù thiếu key
    assert cau_hinh.thieu_khoa_api({"llm": {"provider": "rule_based"}}) == []


def test_cloudflare_doi_CA_HAI_bien_moi_truong(monkeypatch):
    """Token và account id là hai thứ khác nhau — có một nửa vẫn là THIẾU.

    Bẫy đã sửa 16/08: bảng khoá cũ gộp mọi biến của một provider thành quan hệ
    "hoặc", nên chỉ cần đặt `CLOUDFLARE_API_TOKEN` là hệ thống báo đủ khoá rồi
    mới nổ giữa buổi thi khi provider đọc tới account id.
    """
    for bien in _MOI_KHOA:
        monkeypatch.delenv(bien, raising=False)
    config = {"llm": {"enabled": True, "provider": "cloudflare_vlm"}}
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "x")
    thieu = cau_hinh.thieu_khoa_api(config)
    assert len(thieu) == 1 and "CLOUDFLARE_ACCOUNT_ID" in thieu[0]
    assert "CLOUDFLARE_API_TOKEN" not in thieu[0]

    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "y")
    assert cau_hinh.thieu_khoa_api(config) == []


def test_canh_bao_gemini_timeout_khong_du_cho_batch():
    """Lỗi thật, bắt được khi diễn tập 2.2 ngày 16/08.

    `timeout_seconds: 10` + `batch_max_images: 10` là cú pháp hợp lệ nhưng batch
    10 ảnh đo thật mất **22,8s**, nên Gemini luôn quá hạn và QA âm thầm rơi xuống
    provider dự phòng yếu hơn (0.1200 thay vì 0.2067). Không có triệu chứng nào ở
    đáp án, nên phải có cảnh báo.
    """
    xau = {"llm": {"provider": "failover_vlm",
                   "primary": {"enabled": True, "provider": "gemini_vlm",
                               "batch_max_images": 10, "timeout_seconds": 10}}}
    canh = cau_hinh.canh_bao_cau_hinh(xau)
    assert any("batch_max_images=10" in x and "timeout_seconds=10" in x
               for x in canh), canh

    tot = {"llm": {"provider": "failover_vlm",
                   "primary": {"enabled": True, "provider": "gemini_vlm",
                               "batch_max_images": 10, "timeout_seconds": 45}},
           "qa": {"image_roots": ["/kaggle/input/x"], "vlm_top_n": 10}}
    assert cau_hinh.canh_bao_cau_hinh(tot) == []

    # batch=1 thì timeout nhỏ là hợp lý, không cảnh báo
    mot_anh = {"llm": {"provider": "gemini_vlm", "enabled": True,
                       "batch_max_images": 1, "timeout_seconds": 10},
               "qa": {"vlm_top_n": 0}}
    assert cau_hinh.canh_bao_cau_hinh(mot_anh) == []


def test_canh_bao_bat_vlm_ma_khong_co_anh(tmp_path):
    config = {"llm": {"provider": "gemini_vlm", "enabled": True},
              "qa": {"vlm_top_n": 10, "image_roots": [],
                     "dev_image_dir": str(tmp_path / "khong-ton-tai")}}
    assert any("image_roots" in x for x in cau_hinh.canh_bao_cau_hinh(config))


def test_config_that_khong_con_canh_bao_nao():
    """Cấu hình thi phải sạch cảnh báo — nếu đỏ ở đây thì đừng đi thi bằng nó."""
    assert cau_hinh.canh_bao_cau_hinh(cau_hinh.load_config()) == []


def test_khong_con_grok_trong_bang_khoa():
    assert "grok_vlm" not in cau_hinh.KHOA_MOI_TRUONG
    assert cau_hinh.bien_moi_truong("gemini_vlm") == ["GEMINI_API_KEY",
                                                     "GOOGLE_API_KEY"]


def test_khong_biet_che_do_dich_dap_an_thi_no(tmp_path):
    with pytest.raises(cau_hinh.CauHinhSai, match="answer_translation"):
        cau_hinh._visual_qa({"qa": {"answer_translation": "google"}}, None)


def test_khong_biet_bo_sinh_ung_vien_thi_no():
    with pytest.raises(cau_hinh.CauHinhSai, match="answer_candidates"):
        cau_hinh._sinh_ung_vien({"qa": {"answer_candidates": "bậy"}})


# --------------------------------------------------------------------------- #
# Dựng hệ thống: cấu hình phải ĂN THẬT vào Pipeline
# --------------------------------------------------------------------------- #
CAU_HOI = "Người đàn ông mặc áo màu gì?"
KHUON_MAU = "a photo of a {} object"
DIM = 512


def _don_vi(i: int, dim: int = DIM):
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


def _events():
    ra = []
    for i, vid in enumerate(("L01_V001", "L02_V001", "L03_V001")):
        ra.append(Event(video_id=vid, event_id=f"{vid}#0000{i}",
                        t_start=i * 10.0, t_end=i * 10.0 + 10.0,
                        start_frame=i * 250, end_frame=i * 250 + 250,
                        keyframe_idx=i * 250 + 5))
    return ra


def _cache_npz(path: Path) -> Path:
    """Cache vector chữ đủ để chạy offline: câu truy hồi + 11 ứng viên màu.

    Đặt "red" trùng hướng với vector ảnh của event đầu, nên đáp án thắng là
    `red` -> phải ra `đỏ` khi config đặt `output_language: vi`.
    """
    compiled = compile_query("qa", CAU_HOI)
    truy_hoi = compiled.retrieval_query or compiled.visual_queries[0]
    keys, vecs = [truy_hoi], [_don_vi(0)]
    for j, mau in enumerate(NHOM_MAU):
        keys.append(KHUON_MAU.format(mau))
        vecs.append(_don_vi(0) if mau == "red" else _don_vi(10 + j))
    CLIPTextEncoder.luu_cache(path, keys, np.stack(vecs))
    return path


def _config_qa(tmp_path: Path, ngon_ngu: str) -> dict:
    return {
        "encoder": {"provider": "clip", "live": False,
                    "cache_path": str(_cache_npz(tmp_path / f"q_{ngon_ngu}.npz"))},
        "retrieval": {"fusion": "rrf", "top_k": 100, "khu_modality_gap": False,
                      "pool_per_query": 50, "candidate_videos": 3},
        "qa": {"output_language": ngon_ngu, "answer_translation": "bang_tay",
               "answer_candidates": "luat", "vlm_top_n": 10,
               "vlm_question_types": ["color"]},
        "portfolio": {"enabled": False},
        "online": {"max_submission_rows": 100},
    }


def _pipeline_qa(tmp_path: Path, ngon_ngu: str):
    config = _config_qa(tmp_path, ngon_ngu)
    events = _events()
    emb = np.stack([_don_vi(0), _don_vi(1), _don_vi(2)])
    return cau_hinh.dung_pipeline(config, events, emb, llm=RuleBasedProvider())


def test_dap_an_qa_ra_tieng_viet_khi_config_noi_vi(tmp_path):
    """Đây chính là lỗi 15/08: zip ra `Blue`/`Red`/`White` dù config đặt vi."""
    p = _pipeline_qa(tmp_path, "vi")
    rows = p.answer(Query(qid="qa-1", task="qa", text=CAU_HOI), top_k=3)
    dap_an = {r["answer"] for r in rows}
    assert dap_an == {"đỏ"}, dap_an
    assert not any(con_la_tieng_anh(x) for x in dap_an)


def test_cung_config_do_output_language_en_thi_giu_tieng_anh(tmp_path):
    p = _pipeline_qa(tmp_path, "en")
    rows = p.answer(Query(qid="qa-1", task="qa", text=CAU_HOI), top_k=3)
    assert {r["answer"] for r in rows} == {"red"}


def test_dung_pipeline_lay_dung_cac_nut_tu_config(tmp_path):
    p = _pipeline_qa(tmp_path, "vi")
    assert p.portfolio.enabled is False
    assert p.qa_vlm_top_n == 10
    assert p.qa_vlm_question_types == {"color"}
    assert p.retrieval_policy.output_top_k == 100
    assert p.retrieval_policy.candidate_videos == 3
    assert p.retriever.visual._khu_gap is False
    # Không có provider ảnh -> không dựng resolver, không đòi ảnh
    assert p.image_resolver is None


def test_portfolio_bat_trong_config_thi_pipeline_bat(tmp_path):
    config = _config_qa(tmp_path, "vi")
    config["portfolio"] = {"enabled": True, "calibrator": "rank"}
    p = cau_hinh.dung_pipeline(config, _events(),
                               np.stack([_don_vi(0), _don_vi(1), _don_vi(2)]),
                               llm=RuleBasedProvider())
    assert p.portfolio.enabled is True and p.portfolio.calibrator is not None


def test_lech_so_chieu_anh_va_chu_bao_ngay(tmp_path):
    config = _config_qa(tmp_path, "vi")
    emb = np.zeros((3, 256), dtype=np.float32)
    emb[:, 0] = 1.0
    with pytest.raises(cau_hinh.CauHinhSai, match="chiều"):
        cau_hinh.dung_pipeline(config, _events(), emb, llm=RuleBasedProvider())


def test_qa_cheap_ghi_de_provider_vlm(tmp_path):
    config = _config_qa(tmp_path, "vi")
    config["llm"] = {"provider": "gemini_vlm", "model": "gemini-3.6-flash",
                     "cache_path": str(tmp_path / "cache.json")}
    p = cau_hinh.dung_pipeline(config, _events(),
                               np.stack([_don_vi(0), _don_vi(1), _don_vi(2)]),
                               qa_cheap=True)
    assert isinstance(p.llm, RuleBasedProvider)
    assert p.image_resolver is None


# --------------------------------------------------------------------------- #
# Cấu hình THẬT của repo và hai script
# --------------------------------------------------------------------------- #
def test_lop_phu_gop_list_mapping_theo_chi_so(tmp_path):
    """Lớp phủ chỉ đặt MỘT khoá trong một mục của danh sách thì phần còn lại phải
    còn nguyên. Trước 16/08 `_gop_sau` thay hẳn list -> encoder phụ mất
    provider/model_id/cache và tụt về mặc định sai mà không có triệu chứng."""
    _ghi(tmp_path / "goc.yaml",
         "retrieval:\n  hop_nhat:\n    enabled: true\n    phu:\n"
         "      - provider: siglip2\n        model_id: google/x\n"
         "        cache_path: dev_set/a.npz\n        trong_so: 1\n")
    con = _ghi(tmp_path / "phu.yaml",
               "extends: goc.yaml\nretrieval:\n  hop_nhat:\n    phu:\n"
               "      - image_feature_root: D:/vector\n")
    c = cau_hinh.load_config(con)
    muc = c["retrieval"]["hop_nhat"]["phu"][0]
    assert muc["provider"] == "siglip2" and muc["model_id"] == "google/x"
    assert muc["trong_so"] == 1
    assert str(muc["image_feature_root"]).replace("\\", "/").endswith("vector")
    # đường dẫn tương đối trong list cũng phải quy về tuyệt đối
    assert Path(muc["cache_path"]).is_absolute()
    # list thường (không phải mapping) vẫn thay hẳn
    assert cau_hinh._gop_sau({"a": [1, 2, 3]}, {"a": [9]})["a"] == [9]


def test_hop_nhat_thieu_vector_anh_thi_bo_kem_canh_bao():
    config = {"offline": {"btc_data_root": "."},
              "retrieval": {"hop_nhat": {
                  "enabled": True,
                  "phu": [{"provider": "siglip2", "model_id": "google/x",
                           "image_feature_root": None}]}}}
    with pytest.warns(RuntimeWarning, match="hop_nhat"):
        assert cau_hinh.nap_hop_nhat(config, None) == []


def test_chinh_sach_truy_hoi_doc_tham_so_hop_nhat():
    p = cau_hinh.chinh_sach_truy_hoi(
        {"retrieval": {"top_k": 100,
                       "hop_nhat": {"pool": 400, "rrf_k": 30,
                                    "trong_so_chinh": 3}}})
    assert (p.fuse_pool, p.fuse_rrf_k, p.fuse_trong_so_chinh) == (400, 30.0, 3.0)


def test_config_that_khop_nhung_gi_da_chot():
    """Nếu đổi một trong các giá trị này thì phải đổi cả số đo đã ghi trong docs."""
    c = cau_hinh.load_config()
    assert c["encoder"]["provider"] == "siglip2"
    assert "so400m" in c["encoder"]["model_id"]
    assert str(c["qa"]["output_language"]).lower() == "vi"
    assert c["portfolio"]["enabled"] is False
    assert c["online"]["max_submission_rows"] == 100


def test_config_benchmark_gemini_only_bo_cloudflare_va_cheap_fallback():
    from do_diem_qa_dau_cuoi import cau_hinh_gemini_only

    source = {
        "llm": {
            "provider": "failover_vlm",
            "primary": {"provider": "gemini_vlm", "model": "gemini-test",
                        "batch_max_images": 10, "timeout_seconds": 45,
                        "max_retries": 0},
            "fallbacks": [{"provider": "cloudflare_vlm"}],
        },
        "qa": {"vlm_fallback_to_zero_shot": True},
    }
    result = cau_hinh_gemini_only(source, timeout_seconds=120, max_retries=2)
    assert result["llm"]["provider"] == "gemini_vlm"
    assert result["llm"]["model"] == "gemini-test"
    assert result["llm"]["batch_max_images"] == 10
    assert result["llm"]["timeout_seconds"] == 120
    assert result["llm"]["max_retries"] == 2
    assert result["qa"]["vlm_fallback_to_zero_shot"] is False
    assert source["llm"]["provider"] == "failover_vlm"


def test_nop_bai_khong_tu_dung_pipeline():
    """Chốt cách sửa của 1.5: đường nộp bài phải đi qua config, không dựng tay."""
    src = (GOC / "nop_bai.py").read_text(encoding="utf-8")
    assert "cau_hinh.dung_pipeline(" in src
    assert "Pipeline(" not in src, "nop_bai.py lại tự dựng Pipeline -> sẽ trôi tiếp"


def test_khong_con_ten_nop_bai_thu_gay_nham():
    assert not (GOC / "nop_bai_thu.py").exists()
    kiem = (GOC / "kiem_dinh_dang.py").read_text(encoding="utf-8")
    assert "nop_bai.py" in kiem      # chỉ đường sang đường nộp bài thật
