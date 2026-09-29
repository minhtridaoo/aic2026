"""Q&A: trả lời câu hỏi về một khung hình, KHI TRONG TAY CHỈ CÓ VECTOR.

RÀNG BUỘC ĐỊNH HÌNH TOÀN BỘ THIẾT KẾ
------------------------------------
Bộ data BTC phát là 178 MB gồm vector CLIP + mapping + metadata. **Không có
ảnh.** Nên hai cách làm QA thông thường đều tắt đường:

  * VLM nhìn ảnh  -> không có ảnh để nhìn
  * LLM thuần chữ -> không thể biết "chiếc xe màu gì" nếu chưa thấy khung hình

Cách còn lại, và cũng là cách đúng ở đây: **chia đôi bài toán theo đúng năng lực
của từng công cụ**.

    LLM (hoặc luật)  : câu hỏi -> danh sách ĐÁP ÁN ỨNG VIÊN      (thuần chữ)
    CLIP             : chấm từng ứng viên với vector ảnh THẬT     (thị giác)

Bước hai chính là zero-shot classification kinh điển của CLIP: dựng câu
"a photo of a red car", encode bằng text tower, đo cosine với vector ảnh. Vector
ảnh thì ta CÓ — 177.321 cái. Nên QA chạy được mà không cần một tấm ảnh nào.

VÌ SAO ĐIỀU NÀY QUAN TRỌNG
--------------------------
Đo 2026-08-08: `RuleBasedProvider.answer()` trả về chuỗi RỖNG trên toàn bộ 100
dòng, vì nó bới chữ hoa trong `visual_text`/`ocr_text`/`asr_text` — cả ba đều
rỗng với bộ data này. R-Score của QA đòi `answer == GT_a`, nên QA đang được
**0 điểm tuyệt đối**, bất kể tìm đúng khung hay không.

TRẢ VỀ PHÂN PHỐI, KHÔNG PHẢI MỘT ĐÁP ÁN
---------------------------------------
Đáp án loại trừ lẫn nhau, nên mỗi đáp án phân biệt là một kịch bản riêng mà 100
slot có thể phủ độc lập. Nộp một đáp án ×100 dòng là phủ đúng một kịch bản.
Vì vậy `tra_loi()` trả `[(đáp án, xác suất)]` — khớp thẳng với
`optimize.submission_optimizer.qa_portfolio`.
"""
from __future__ import annotations

import math
import os
import re
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Sinh đáp án ứng viên (phần THUẦN CHỮ)
# --------------------------------------------------------------------------- #
class SinhUngVien(ABC):
    @abstractmethod
    def sinh(self, cau_hoi: str) -> List[str]:
        """Danh sách đáp án có thể, chưa xét ảnh."""


# Các nhóm đáp án đóng hay gặp trong đề KIS/QA về tin tức Việt Nam.
# Cố ý viết bằng TIẾNG ANH: chúng sẽ được nhét vào text tower của CLIP, mà CLIP
# không hiểu tiếng Việt (đo được: hạng 72.620 so với 283 — xem `dich.py`).
NHOM_MAU = ["red", "orange", "yellow", "green", "blue", "purple", "pink",
            "brown", "black", "white", "grey"]
NHOM_SO = ["one", "two", "three", "four", "five", "six", "seven", "eight",
           "nine", "ten"]
NHOM_XE = ["a car", "a motorbike", "a bicycle", "a bus", "a truck", "a train",
           "a boat", "an airplane"]
NHOM_NOI = ["indoors", "outdoors", "on a street", "in an office",
            "in a kitchen", "in a classroom", "on a stage", "in a stadium",
            "at a market", "on a beach", "in a forest", "in a field"]
NHOM_THOI_GIAN = ["during the day", "at night", "at sunset", "in the morning"]
NHOM_CO_KHONG = ["yes", "no"]

# Ánh xạ tay cho hai lớp đóng quan trọng nhất. Không dùng máy dịch cho chúng:
# model dịch dễ rút cả green/blue về "màu xanh", còn nhãn QA cần phân biệt đúng
# "xanh lá" / "xanh dương". Số xuất thẳng chữ số để khớp 6/30 nhãn hiện có.
_DAP_AN_DONG_EN_VI = {
    "red": "đỏ", "orange": "cam", "yellow": "vàng", "green": "xanh lá",
    "blue": "xanh dương", "purple": "tím", "pink": "hồng", "brown": "nâu",
    "black": "đen", "white": "trắng", "grey": "xám", "gray": "xám",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "a car": "ô tô", "a motorbike": "xe máy", "a bicycle": "xe đạp",
    "a bus": "xe buýt", "a truck": "xe tải", "a train": "tàu hỏa",
    "a boat": "thuyền", "an airplane": "máy bay",
    "indoors": "trong nhà", "outdoors": "ngoài trời",
    "on a street": "đường phố", "in an office": "văn phòng",
    "in a kitchen": "nhà bếp", "in a classroom": "lớp học",
    "on a stage": "sân khấu", "in a stadium": "sân vận động",
    "at a market": "chợ", "on a beach": "bãi biển",
    "in a forest": "rừng", "in a field": "cánh đồng",
    "during the day": "ban ngày", "at night": "ban đêm",
    "at sunset": "hoàng hôn", "in the morning": "buổi sáng",
    "yes": "có", "no": "không",
}


class DichDapAnVi:
    """Địa phương hoá đáp án thắng sau khi CLIP đã chấm xong.

    ``translator`` tuân giao diện ``Translator.dich(list[str])``. Để baseline
    không tự tải model, có thể để ``None``: màu/số vẫn được xử lý bằng bảng tay,
    còn lớp mở giữ nguyên tiếng Anh. Runtime thật nên truyền
    ``OpusMTEnViTranslator``; chỉ các đáp án ngoài bảng tay mới chạm model.
    """

    def __init__(self, translator=None):
        self.translator = translator
        self._cache: Dict[str, str] = {}

    @staticmethod
    def _khoa(answer: str) -> str:
        return " ".join(str(answer).lower().strip(" \t\r\n.,;:!?\"'").split())

    def dich_mot(self, answer: str) -> str:
        key = self._khoa(answer)
        if key in _DAP_AN_DONG_EN_VI:
            return _DAP_AN_DONG_EN_VI[key]
        if not key or self.translator is None:
            return str(answer).strip()
        if key not in self._cache:
            self._cache[key] = self.translator.dich([key])[0].strip()
        return self._cache[key]

    def dich_phan_phoi(self, answers: Sequence[Tuple[str, float]]) \
            -> List[Tuple[str, float]]:
        # Gộp xác suất nếu hai cách viết cùng dịch về một nhãn.
        merged: Dict[str, float] = {}
        for answer, prob in answers:
            vi = self.dich_mot(answer)
            merged[vi] = merged.get(vi, 0.0) + float(prob)
        return sorted(merged.items(), key=lambda kv: -kv[1])


def soi_dap_an_vlm(cap: Sequence[Tuple[str, str]],
                   toi_thieu_video: int = 3) -> Optional[str]:
    """VLM này có đang ĐỌC ẢNH, hay đang bịa từ chữ trong câu hỏi?

    Phép kiểm rẻ nhất mà lại quyết định được, phát hiện 20/08 và đo lại 28/08:
    một VLM **đọc thật** trả lời KHÁC NHAU theo video và NHẤT QUÁN trong cùng
    một video; một VLM **bịa** trả đúng một chuỗi cho mọi khung, vì nó chỉ suy
    từ chữ trong câu hỏi. Số thật, cùng một câu hỏi, cùng 10 khung:

        gemini-3.6-flash  "con số cuối trên cân"  -> `300` cho 8 video khác nhau
        gemini-3.6-flash  "sạt lở đường đèo"      -> `Đèo Bảo Lộc` cho 6 video
        gemini-3.5-flash-lite "bao nhiêu vị trí động đất cấp 4"
                          -> L21_V010: 5 · L21_V006: 4 · L21_V007: 2 · L25_V076: 10

    Nên điều kiện hét lên là **CHỈ MỘT đáp án phân biệt trải trên ≥3 video**.
    Đòi cả hai vì mỗi vế một mình đều sai:

      * "một đáp án cho ≥3 video" một mình thì báo động giả với câu hỏi có không
        gian đáp án nhỏ — 5 video đều có người mặc áo trắng là chuyện thường.
      * "chỉ một đáp án phân biệt" một mình thì báo động giả khi top-10 rơi cả
        vào MỘT video (lúc đó một đáp án là hợp lý, không phải triệu chứng).

    ⚠ CHỮ KÝ NÀY CÓ HAI CÁCH ĐỌC, đừng vội kết luận "model mù" — đo được cả hai
    trong cùng một buổi 28/08:

      (a) model không đọc ảnh   — `3.6-flash` trả một đáp án cho MỌI câu, mọi ảnh.
      (b) TẦNG TÌM trượt        — `3.5-flash-lite` (đã chứng minh đọc thật ở câu
          động đất) cũng trả `1` cho cả 10 khung câu "con số trên cân", đơn giản
          vì trong 10 ứng viên KHÔNG có tấm nào chụp cái cân: nó đoán cho xong.

    Việc phải làm thì giống nhau ở cả hai ca — **đừng tin đáp án này, đưa câu đó
    cho mắt người / tra OCR-ASR**. Nên hàm chỉ cảnh báo, không chặn.

    Nhớ cắt danh sách về `vlm_top_n` trước khi gọi: `AnswerTrace.answers` gồm cả
    90 dòng đuôi vốn dùng CHUNG một đáp án theo thiết kế (`qa.tail_answer`).
    """
    theo_dap_an: Dict[str, set] = {}
    for event_id, dap_an in cap:
        khoa = str(dap_an).strip()
        if not khoa:
            continue
        video = str(event_id).split("#")[0]
        theo_dap_an.setdefault(khoa, set()).add(video)
    if len(theo_dap_an) != 1:
        return None
    dap_an, video = next(iter(theo_dap_an.items()))
    if len(video) < int(toi_thieu_video):
        return None
    return (f"ĐÁP ÁN ĐÁNG NGỜ: đúng một đáp án {dap_an!r} cho {len(video)} video "
            f"khác nhau ({', '.join(sorted(video)[:4])}"
            f"{'…' if len(video) > 4 else ''}). Hoặc bậc VLM đang bịa, hoặc tầng "
            f"tìm không đưa được tấm ảnh liên quan nào vào top — cách nào thì "
            f"cũng ĐỪNG TIN đáp án này, đưa câu này cho người xem / tra OCR-ASR.")


def con_la_tieng_anh(answer: str) -> bool:
    """Đáp án này còn là ứng viên TIẾNG ANH chưa địa phương hoá?

    Có để đường nộp bài tự bắt lại đúng lỗi 1.5 (15/08): zip xuất ra `Blue` /
    `Red` / `White` trong khi config đặt `qa.output_language: vi`, và không một
    phép kiểm định dạng nào thấy bất thường. Chỉ nhận diện các lớp đóng có trong
    bảng tay — đủ để phát hiện cả một cột đáp án chưa dịch, không nhằm chấm đúng
    sai từng câu.
    """
    return DichDapAnVi._khoa(answer) in _DAP_AN_DONG_EN_VI

# (mẫu nhận dạng câu hỏi, nhóm đáp án, khuôn dựng câu để đưa vào CLIP)
#
# 🔴 THỨ TỰ CÓ Ý NGHĨA — luật ĐẦU TIÊN khớp là luật thắng, và nó quét trên CẢ
# đoạn "mô tả cảnh + câu hỏi" chứ không riêng câu hỏi. Đề BTC luôn tả cảnh
# trước, và cảnh thì đầy màu sắc, nên để luật MÀU trước luật SỐ là mời lỗi:
#
#   query-p1-9-qa  "…chiếc xe MÀU vàng, MÀU đỏ và MÀU đen… Con số được ghi
#                   trên biển báo… là bao nhiêu?"      -> nộp `trắng`
#   query-p1-15-qa "…kí hiệu nhiều MÀU sắc… có bao nhiêu vị trí ghi nhận
#                   động đất cấp độ 4?"                -> nộp `vàng`
#
# Cả hai là 0 điểm chắc chắn (sơ tuyển đợt 1, 21/08). Đảo lại: câu hỏi SỐ
# nhận dạng bằng "bao nhiêu/mấy" — mấy từ này gần như không xuất hiện trong
# phần TẢ CẢNH, còn "màu" thì xuất hiện liên tục. Nên hỏi-số-thắng-tả-màu là
# bất đối xứng ĐÚNG hướng, không phải một núm được fit.
LUAT = [
    (r"(?i)\b(bao nhiêu|mấy|how many|number of)\b", NHOM_SO,
     "a photo of {} people"),
    (r"(?i)\b(màu|color|colour)\b", NHOM_MAU, "a photo of a {} object"),
    (r"(?i)\b(xe|phương tiện|vehicle|transport)\b", NHOM_XE, "a photo of {}"),
    (r"(?i)\b(ở đâu|nơi nào|where|location|place)\b", NHOM_NOI, "a photo taken {}"),
    (r"(?i)\b(khi nào|lúc nào|when|time of day)\b", NHOM_THOI_GIAN,
     "a photo taken {}"),
    (r"(?i)\b(có phải|có .* không|is there|are there|does)\b", NHOM_CO_KHONG,
     "a photo where the answer is {}"),
]


class SinhTheoLuat(SinhUngVien):
    """Không cần mạng, không tốn tiền, tái lập được.

    Chỉ phủ các nhóm câu hỏi đóng hay gặp. Câu hỏi mở (`mô tả cảnh này`) sẽ ra
    danh sách rỗng — lúc đó `ClipZeroShotQA` lùi về `NHOM_NOI` để ít nhất còn
    trả lời được cái gì đó thay vì chuỗi rỗng.
    """

    def khuon(self, cau_hoi: str) -> Tuple[List[str], str]:
        for mau, nhom, khuon in LUAT:
            if re.search(mau, cau_hoi):
                return list(nhom), khuon
        return [], "a photo of {}"

    def sinh(self, cau_hoi: str) -> List[str]:
        return self.khuon(cau_hoi)[0]


class SinhBangLLM(SinhUngVien):
    """Dùng Claude sinh ứng viên. Chỉ là việc THUẦN CHỮ — không đưa ảnh vào.

    Bật khi có `ANTHROPIC_API_KEY`. Có cache trong bộ nhớ nên một câu hỏi chỉ
    gọi API đúng một lần.
    """

    MODEL = "claude-sonnet-5"

    def __init__(self, model: Optional[str] = None, n: int = 8):
        self.model = model or self.MODEL
        self.n = int(n)
        self._client = None
        self._cache: Dict[str, List[str]] = {}

    def sinh(self, cau_hoi: str) -> List[str]:
        if cau_hoi in self._cache:
            return self._cache[cau_hoi]
        try:
            if self._client is None:
                import anthropic
                self._client = anthropic.Anthropic()
            r = self._client.messages.create(
                model=self.model, max_tokens=300,
                messages=[{"role": "user", "content":
                           "Câu hỏi về MỘT khung hình trong video tin tức:\n"
                           f"{cau_hoi}\n\n"
                           f"Liệt kê tối đa {self.n} đáp án ngắn có thể đúng, "
                           "MỖI DÒNG MỘT ĐÁP ÁN, bằng TIẾNG ANH, không đánh số, "
                           "không giải thích. Đáp án phải loại trừ lẫn nhau."}])
            ra = [ln.strip(" -•\t") for ln in r.content[0].text.splitlines()]
            ra = [x for x in ra if x][:self.n]
        except Exception:                                      # noqa: BLE001
            ra = SinhTheoLuat().sinh(cau_hoi)                  # lùi về luật
        self._cache[cau_hoi] = ra
        return ra


# --------------------------------------------------------------------------- #
# Chấm ứng viên bằng CLIP (phần THỊ GIÁC)
# --------------------------------------------------------------------------- #
class ClipZeroShotQA:
    """Chấm từng đáp án ứng viên với vector ảnh của các khung hàng đầu.

    `lay_vector(event_id) -> vector ảnh | None` do phía gọi cung cấp (thường là
    `VectorStore.lay_vector`), nên lớp này không phụ thuộc vào kiểu store nào.
    """

    def __init__(self, encoder, sinh: Optional[SinhUngVien] = None,
                 so_khung: int = 5, nhiet: float = 100.0,
                 dich_dap_an: Optional[DichDapAnVi] = None):
        self.encoder = encoder
        self.sinh = sinh or SinhTheoLuat()
        self.so_khung = int(so_khung)
        # CLIP huấn luyện với logit_scale ~100; dùng lại để softmax không bị bẹt.
        self.nhiet = float(nhiet)
        self.dich_dap_an = dich_dap_an

    def _khuon(self, cau_hoi: str) -> str:
        if isinstance(self.sinh, SinhTheoLuat):
            return self.sinh.khuon(cau_hoi)[1]
        return "a photo of {}"

    def tra_loi(self, cau_hoi: str, candidates: Sequence,
                lay_vector: Callable) -> List[Tuple[str, float]]:
        ung_vien = self.sinh.sinh(cau_hoi)
        if not ung_vien:
            ung_vien = list(NHOM_NOI)
        khuon = self._khuon(cau_hoi)

        vecs = []
        for c in candidates[:self.so_khung]:
            v = lay_vector(c.event.event_id)
            if v is not None:
                vecs.append([float(x) for x in v])
        if not vecs:
            ra = [(ung_vien[0], 1.0)]
            return (self.dich_dap_an.dich_phan_phoi(ra)
                    if self.dich_dap_an is not None else ra)

        diem: List[float] = []
        for a in ung_vien:
            t = self.encoder.encode_text(khuon.format(a))
            # Trung bình cosine trên vài khung đầu: một khung có thể bị chọn sai,
            # nhưng vài khung cùng nghiêng về một đáp án thì đáng tin hơn.
            diem.append(sum(_cos(t, v) for v in vecs) / len(vecs))

        m = max(diem)
        mu = [math.exp(self.nhiet * (d - m)) for d in diem]
        tong = sum(mu) or 1.0
        ra = sorted(zip(ung_vien, [x / tong for x in mu]),
                    key=lambda kv: -kv[1])
        if self.dich_dap_an is not None:
            ra = self.dich_dap_an.dich_phan_phoi(ra)
        return ra


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def dung_sinh_ung_vien() -> SinhUngVien:
    """LLM nếu có key, không thì luật. Không bao giờ ném lỗi vì thiếu key."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return SinhBangLLM()
    return SinhTheoLuat()
