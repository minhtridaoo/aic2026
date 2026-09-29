"""Bộ viết lại truy vấn Việt -> chú thích tiếng Anh (việc 4.5a).

Không test nào ở đây gọi mạng: provider là bản giả ghi lại prompt đã nhận.
"""
import json

import pytest

from aic2026.core.viet_lai_truy_van import (PHIEN_BAN_PROMPT, VietLaiTruyVan,
                                            _khoa)


class ProviderGia:
    """Trả `so_ban` câu tiếng Anh bịa từ id, và đếm số request."""

    model = "gia-1.0"

    def __init__(self, so_ban=1, bo_qua=(), tra_ve=None):
        self.so_ban = so_ban
        self.bo_qua = set(bo_qua)          # id cố tình KHÔNG trả về
        self.tra_ve = tra_ve               # ép nguyên văn chuỗi trả lời
        self.prompts = []

    def tra_loi_chu(self, system_instruction, prompt, schema=None):
        self.prompts.append(prompt)
        if self.tra_ve is not None:
            return self.tra_ve
        rows = []
        for dong in prompt.splitlines():
            if not dong.startswith('{"id"'):
                continue
            muc = json.loads(dong)
            if muc["id"] in self.bo_qua:
                continue
            rows.append({"id": muc["id"],
                         "captions": [f"caption {muc['id']} #{k}"
                                      for k in range(self.so_ban)]})
        return json.dumps(rows, ensure_ascii=False)


def _bo(tmp_path, provider, **kwargs):
    return VietLaiTruyVan(provider, cache_path=str(tmp_path / "cache.json"),
                          **kwargs)


def test_viet_lai_tra_du_moi_cau(tmp_path):
    p = ProviderGia()
    ra = _bo(tmp_path, p).viet_lai(["một người nấu ăn", "hai người đi xe máy"])
    assert set(ra) == {"một người nấu ăn", "hai người đi xe máy"}
    assert all(len(v) == 1 and v[0].startswith("caption") for v in ra.values())
    assert p.prompts and len(p.prompts) == 1        # gộp lô, không phải 1/câu


def test_gop_lo_theo_batch(tmp_path):
    p = ProviderGia()
    bo = _bo(tmp_path, p, batch=2)
    bo.viet_lai([f"câu số {i}" for i in range(5)])
    assert bo.requests == 3                          # 2 + 2 + 1
    assert len(p.prompts) == 3


def test_cache_dia_khong_goi_lai(tmp_path):
    p1 = ProviderGia()
    bo1 = _bo(tmp_path, p1)
    bo1.viet_lai(["một người nấu ăn"])
    assert bo1.requests == 1

    p2 = ProviderGia()
    bo2 = _bo(tmp_path, p2)
    ra = bo2.viet_lai(["một người nấu ăn"])
    assert bo2.requests == 0 and bo2.cache_hits == 1
    assert p2.prompts == []
    assert ra["một người nấu ăn"][0].startswith("caption")


def test_doi_so_ban_thi_cache_cu_khong_dung_lai(tmp_path):
    """Khoá cache có `so_ban`: xin 3 bản mà nhận lại bản 1 câu là sai lặng lẽ."""
    p = ProviderGia()
    _bo(tmp_path, p, so_ban=1).viet_lai(["một người nấu ăn"])
    bo = _bo(tmp_path, ProviderGia(so_ban=3), so_ban=3)
    ra = bo.viet_lai(["một người nấu ăn"])
    assert bo.requests == 1
    assert len(ra["một người nấu ăn"]) == 3


def test_cau_bi_provider_bo_qua_thi_giu_nguyen_ban_goc(tmp_path):
    """Không bịa: mất câu nào thì trả lại chính câu tiếng Việt và ghi vào `thieu`."""
    p = ProviderGia(bo_qua={"q001"})
    bo = _bo(tmp_path, p)
    ra = bo.viet_lai(["câu một", "câu hai"])
    assert ra["câu hai"] == ["câu hai"]
    assert bo.thieu == ["câu hai"]
    assert ra["câu một"][0].startswith("caption")


def test_thieu_ban_thi_lap_lai_ban_cuoi_du_so_ban(tmp_path):
    tra_ve = json.dumps([{"id": "q000", "captions": ["chi mot ban"]}])
    bo = _bo(tmp_path, ProviderGia(tra_ve=tra_ve), so_ban=3)
    ra = bo.viet_lai(["câu một"])
    assert ra["câu một"] == ["chi mot ban"] * 3


def test_json_hong_thi_bao_loi_ro(tmp_path):
    bo = _bo(tmp_path, ProviderGia(tra_ve="xin chào, đây không phải JSON"))
    with pytest.raises(RuntimeError, match="JSON"):
        bo.viet_lai(["câu một"])


def test_cau_trung_nhau_chi_gui_mot_lan(tmp_path):
    p = ProviderGia()
    bo = _bo(tmp_path, p)
    bo.viet_lai(["giống hệt", "giống hệt", "khác"])
    muc = [d for d in p.prompts[0].splitlines() if d.startswith('{"id"')]
    assert len(muc) == 2


def test_khoa_cache_doi_khi_prompt_doi():
    """`PHIEN_BAN_PROMPT` nằm trong khoá — sửa prompt mà dùng lại cache là sai."""
    a = _khoa("m", 1, "câu")
    assert PHIEN_BAN_PROMPT in "caption-v1 caption-v2"
    assert a != _khoa("m", 2, "câu") and a != _khoa("khac", 1, "câu")


def test_doc_duoc_json_lan_trong_loi_dan():
    """Model open-weight (Groq/Cloudflare) hay thêm lời dẫn — không được vỡ vì thế."""
    from aic2026.core.viet_lai_truy_van import _doc_json_array
    rows = _doc_json_array(
        'Here is the JSON you asked for:\n'
        '[{"id":"q000","captions":["a chef cooks"]}]\nHope it helps!')
    assert rows == [{"id": "q000", "captions": ["a chef cooks"]}]


def test_groq_tra_loi_chu_khong_dung_max_tokens_cua_qa(tmp_path):
    """64 token là hạn của ĐÁP ÁN QA; viết lại cả lô mà cắt ở đó là JSON hỏng."""
    from aic2026.core.providers import GroqVLMProvider
    da_gui = {}

    def requester(url, headers, payload, timeout):
        da_gui.update(payload=payload, timeout=timeout)
        return {"choices": [{"message": {"content": "[]"}}]}

    p = GroqVLMProvider(cache_path=str(tmp_path / "c.json"), max_tokens=64,
                        api_key="k", requester=requester)
    p.tra_loi_chu("hướng dẫn", "viết lại giúp")
    assert da_gui["payload"]["max_completion_tokens"] >= 1024
    assert da_gui["payload"]["messages"][0]["role"] == "system"
    assert p.backend_requests == 1


def test_groq_lam_bo_viet_lai_duoc(tmp_path):
    """`VietLaiTruyVan` không biết provider nào: chỉ cần có `tra_loi_chu`."""
    from aic2026.core.providers import GroqVLMProvider

    def requester(url, headers, payload, timeout):
        return {"choices": [{"message": {"content": json.dumps(
            [{"id": "q000", "captions": ["a chef cooks in a kitchen"]}])}}]}

    p = GroqVLMProvider(cache_path=str(tmp_path / "c.json"), api_key="k",
                        requester=requester)
    bo = VietLaiTruyVan(p, cache_path=str(tmp_path / "vl.json"))
    ra = bo.viet_lai(["một đầu bếp đang nấu ăn"])
    assert ra["một đầu bếp đang nấu ăn"] == ["a chef cooks in a kitchen"]
    assert bo.thieu == []


def test_so_ban_va_batch_phai_duong(tmp_path):
    with pytest.raises(ValueError):
        _bo(tmp_path, ProviderGia(), so_ban=0)
    with pytest.raises(ValueError):
        _bo(tmp_path, ProviderGia(), batch=0)


# --------------------------------------------------------------------------- #
# Nối vào đường nộp bài (`ap_dung` + `tu_config`)
# --------------------------------------------------------------------------- #
class BanGhiGia:
    def __init__(self, qid, theo_chinh_sach):
        self.qid = qid
        self.theo_chinh_sach = theo_chinh_sach


def _queries():
    from aic2026.core.models import Query, SubEvent
    return [
        Query(qid="q1", task="kis", text="một đầu bếp đang nấu ăn"),
        Query(qid="q2", task="kis", text="A chef cooks"),          # đề có bản EN
        Query(qid="q3", task="trake", text="",
              sub_events=[SubEvent(text="người rót dầu"),
                          SubEvent(text="người đảo thức ăn")]),
    ]


def test_ap_dung_chi_dung_cau_khong_theo_chinh_sach(tmp_path):
    """Câu đề ĐÃ CÓ bản tiếng Anh của người ra đề thì không được đụng vào:
    bản người đo 0.3750, bản LLM 0.3450."""
    from aic2026.core.viet_lai_truy_van import ap_dung
    qs = _queries()
    bg = [BanGhiGia("q1", False), BanGhiGia("q2", True), BanGhiGia("q3", False)]
    bo = _bo(tmp_path, ProviderGia())
    ap_dung(bo, qs, bg)
    assert qs[0].text.startswith("caption")          # đã viết lại
    assert qs[1].text == "A chef cooks"              # GIỮ NGUYÊN
    assert all(se.text.startswith("caption") for se in qs[2].sub_events)


def test_ap_dung_lay_ban_dau_tien(tmp_path):
    """Bản [0], không phải trung bình: ensemble đo được TỆ HƠN ở so400m."""
    from aic2026.core.viet_lai_truy_van import ap_dung
    qs, bg = _queries()[:1], [BanGhiGia("q1", False)]
    ap_dung(_bo(tmp_path, ProviderGia(so_ban=3), so_ban=3), qs, bg)
    assert qs[0].text.endswith("#0")


def test_ap_dung_llm_chet_thi_giu_nguyen_khong_nem_loi(tmp_path):
    """Hỏng lớp này mất ~0.057 điểm; ném lỗi ra ngoài là mất CẢ BÀI NỘP."""
    from aic2026.core.viet_lai_truy_van import ap_dung

    class ProviderChet:
        model = "chet"

        def tra_loi_chu(self, *a, **k):
            raise RuntimeError("429 hết quota")

    qs, bg = _queries()[:1], [BanGhiGia("q1", False)]
    log = ap_dung(_bo(tmp_path, ProviderChet()), qs, bg)
    assert qs[0].text == "một đầu bếp đang nấu ăn"
    assert any("THẤT BẠI" in d for d in log)


def test_ap_dung_khong_cau_nao_can(tmp_path):
    from aic2026.core.viet_lai_truy_van import ap_dung
    p = ProviderGia()
    log = ap_dung(_bo(tmp_path, p), _queries()[:1], [BanGhiGia("q1", True)])
    assert p.prompts == [] and "không câu nào cần" in log[0]


def test_tu_config_tat_thi_tra_none():
    from aic2026.core.viet_lai_truy_van import tu_config
    bo, vi_sao = tu_config({"truy_van": {"viet_lai": {"enabled": False}}})
    assert bo is None and "tắt" in vi_sao


def test_tu_config_thieu_khoa_thi_tra_none_chu_khong_nem(monkeypatch):
    """Thiếu key lúc thi phải hạ cấp có báo, không được làm đổ `nop_bai.py`."""
    from aic2026.core.viet_lai_truy_van import tu_config
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    bo, vi_sao = tu_config({"truy_van": {"viet_lai": {
        "enabled": True, "provider": "gemini_vlm"}}})
    assert bo is None and "GEMINI_API_KEY" in vi_sao


def test_tu_config_dung_duoc(monkeypatch):
    from aic2026.core.viet_lai_truy_van import tu_config
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    bo, vi_sao = tu_config({"truy_van": {"viet_lai": {
        "enabled": True, "provider": "gemini_vlm",
        "model": "gemini-3.5-flash-lite", "so_ban": 3, "lo": 10}}})
    assert bo is not None and bo.so_ban == 3 and bo.batch == 10
    assert "gemini-3.5-flash-lite" in vi_sao
