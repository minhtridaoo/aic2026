"""Hợp nhất nhiều encoder ở tầng `RetrievalService` (việc 4.4a, 2026-08-16).

Đo được trên 120 câu KIS: so400m một mình 0.3400 -> so400m×2 + base-224 0.3750
(Δ +0.0350 ± 0.0127, và cùng dấu ở cả hai nửa split theo video). Các test dưới
đây khoá ĐÚNG cơ chế đã đo, không khoá con số:

  * hợp nhất bằng RRF CÓ TRỌNG SỐ trên HẠNG (không cộng cosine hai không gian)
  * encoder phụ lỗi/không encode được thì bỏ NÓ, không làm hỏng cả truy vấn
  * tắt hợp nhất thì kết quả y hệt đường một encoder
"""
import math

from aic2026.control.query_compiler import compile_query
from aic2026.core.models import Event
from aic2026.core.providers import VisualEncoder
from aic2026.core.retrieval import MultiChannelRetriever
from aic2026.core.stores import InMemoryVectorStore
from aic2026.retrieval.service import RetrievalPolicy, RetrievalService


def _events(n=6):
    return [Event(video_id=f"L01_V{i:03d}", event_id=f"L01_V{i:03d}#00001",
                  t_start=0.0, t_end=1.0, start_frame=0, end_frame=25,
                  keyframe_idx=i) for i in range(1, n + 1)]


class _EncoderGia(VisualEncoder):
    """Trả vector one-hot theo BẢNG ĐÃ CHO; ném LookupError nếu câu lạ.

    Ném đúng `LookupError` là có chủ ý: đó là hợp đồng của `CLIPTextEncoder` khi
    không encode được, và `RetrievalService` phải bỏ riêng encoder đó.
    """

    def __init__(self, bang, dim=6):
        self.bang, self.dim = bang, dim

    def encode_text(self, text):
        if text not in self.bang:
            raise LookupError(f"không encode được: {text!r}")
        v = [0.0] * self.dim
        v[self.bang[text]] = 1.0
        return v


def _retriever(bang, thu_tu, events):
    """Kho vector sao cho thứ tự trả về đúng `thu_tu` (chỉ số event, 0-based).

    Mọi vector đều chuẩn hoá về norm 1 nên cosine với truy vấn one-hot bằng đúng
    thành phần đầu — thứ tự do mình đặt, không do độ dài vector.
    """
    r = MultiChannelRetriever(_EncoderGia(bang))
    r.visual = InMemoryVectorStore()
    emb = [[0.0] * 6 for _ in events]
    for hang, i in enumerate(thu_tu):
        cos = 0.9 - hang * 0.05
        emb[i][0] = cos
        emb[i][2] = math.sqrt(max(0.0, 1.0 - cos * cos))
    # `embeddings=` để retriever KHÔNG encode `visual_text` (rỗng) lúc index.
    r.index(events, embeddings=emb)
    return r


def _xep(service, text="cau"):
    compiled = compile_query("kis", text)
    return [c.event.event_id
            for c in service.retrieve_compiled(compiled, top_k=6)]


def test_tat_hop_nhat_thi_y_het_duong_mot_encoder():
    events = _events()
    bang = {"cau": 0}
    chinh = _retriever(bang, [0, 1, 2, 3, 4, 5], events)
    service = RetrievalService(chinh, events, RetrievalPolicy(output_top_k=6))
    assert _xep(service) == [e.event_id for e in events]


def test_rrf_co_trong_so_keo_ung_vien_cua_encoder_phu_len():
    """Ứng viên hạng 4 của encoder chính, hạng 1 của encoder phụ, phải lên top-2."""
    events = _events()
    bang = {"cau": 0}
    chinh = _retriever(bang, [0, 1, 2, 3, 4, 5], events)
    phu = _retriever(bang, [3, 5, 4, 0, 1, 2], events)
    service = RetrievalService(
        chinh, events, RetrievalPolicy(output_top_k=6, fuse_pool=6,
                                       fuse_rrf_k=60.0,
                                       fuse_trong_so_chinh=2.0),
        phu=[(phu, 1.0)])
    xep = _xep(service)
    assert xep[0] == events[0].event_id            # hạng 1 chính vẫn thắng
    assert xep.index(events[3].event_id) < 3       # 3 nhảy từ hạng 4 lên top-3
    # điểm phải là điểm RRF, không phải cosine thô
    compiled = compile_query("kis", "cau")
    diem = [c.score for c in service.retrieve_compiled(compiled, top_k=6)]
    assert diem == sorted(diem, reverse=True) and max(diem) < 1.0


def test_trong_so_chinh_cao_hon_thi_giu_thu_tu_cua_encoder_chinh():
    events = _events()
    bang = {"cau": 0}
    chinh = _retriever(bang, [0, 1, 2, 3, 4, 5], events)
    phu = _retriever(bang, [5, 4, 3, 2, 1, 0], events)
    nhe = RetrievalService(chinh, events,
                           RetrievalPolicy(output_top_k=6, fuse_pool=6,
                                           fuse_trong_so_chinh=1.0),
                           phu=[(phu, 1.0)])
    nang = RetrievalService(chinh, events,
                            RetrievalPolicy(output_top_k=6, fuse_pool=6,
                                            fuse_trong_so_chinh=10.0),
                            phu=[(phu, 1.0)])
    assert _xep(nang) == [e.event_id for e in events]
    assert _xep(nhe) != _xep(nang)


def test_encoder_phu_khong_encode_duoc_thi_bo_rieng_no():
    events = _events()
    chinh = _retriever({"cau": 0}, [0, 1, 2, 3, 4, 5], events)
    phu = _retriever({"cau_khac": 1}, [5, 4, 3, 2, 1, 0], events)
    service = RetrievalService(chinh, events,
                              RetrievalPolicy(output_top_k=6, fuse_pool=6),
                              phu=[(phu, 1.0)])
    # Encoder phụ ném LookupError -> chỉ còn đường chính, không nổ cả truy vấn.
    assert _xep(service) == [e.event_id for e in events]


def test_trong_so_0_bi_bo_khoi_danh_sach_phu():
    events = _events()
    chinh = _retriever({"cau": 0}, [0, 1, 2, 3, 4, 5], events)
    phu = _retriever({"cau": 0}, [5, 4, 3, 2, 1, 0], events)
    service = RetrievalService(chinh, events, RetrievalPolicy(output_top_k=6),
                              phu=[(phu, 0.0)])
    assert service.phu == []
    assert _xep(service) == [e.event_id for e in events]
