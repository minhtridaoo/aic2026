"""Định tuyến hoặc dịch truy vấn tiếng Việt trước khi encode.

VÌ SAO CẦN
----------
CLIP ViT-B/32 (đúng model BTC dùng để tạo 177.321 vector ảnh) huấn luyện trên
alt-text tiếng Anh. Đo 2026-08-08 trên dev set 60 câu, hạng của keyframe đúng
trong 177.321 (thấp là tốt; ~88.660 = xếp ngẫu nhiên):

    CLIP + tiếng Việt thô      hạng 72.620   R@100  0.0%   ❌ ~ngẫu nhiên
    CLIP + tiếng Anh (tay)     hạng    283   R@100 28.3%   ✅
    đa ngữ + tiếng Việt thô    hạng  1.183   R@100 18.3%

Lúc thi, BTC đưa câu TIẾNG VIỆT. Không được đưa nguyên câu đó vào CLIP gốc.
Runtime mặc định dùng mCLIP vì nó hiểu trực tiếp tiếng Việt và đã căn vào cùng
không gian ảnh BTC. Nhánh dịch -> CLIP vẫn được giữ để A/B có chủ đích; bản dịch
người hiện còn tốt hơn mọi bản máy nên không được dùng nhánh máy dịch làm mặc
định khi chưa có dev set lớn hơn.

THIẾT KẾ
--------
`ChonEncoderTheoNgonNgu` đưa tiếng Anh vào CLIP và tiếng Việt vào mCLIP.
`DichRoiEncode` là nhánh thí nghiệm tách biệt: bọc một `VisualEncoder` bất kỳ,
nên đổi encoder hay đổi model dịch không đụng tầng retrieval.

Chỉ dịch câu CÓ dấu tiếng Việt. Câu tiếng Anh đi thẳng — không tốn thời gian mà
cũng tránh việc model dịch "sửa" một câu vốn đã đúng.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .providers import VisualEncoder

# Các dấu chỉ có trong tiếng Việt (không có trong tiếng Anh). Đủ để phân biệt
# hai ngôn ngữ này; KHÔNG phải bộ nhận dạng ngôn ngữ tổng quát.
_DAU_VIET = set("ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩị"
                "òóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ")


def co_dau_tieng_viet(text: str) -> bool:
    return any(c in _DAU_VIET for c in str(text).lower())


class Translator(ABC):
    @abstractmethod
    def dich(self, texts: List[str]) -> List[str]:
        """Dịch cả lô. Trả về đúng số phần tử, cùng thứ tự."""


class KhongDich(Translator):
    """Không dịch gì. Dùng khi truy vấn vốn đã là tiếng Anh."""

    def dich(self, texts: List[str]) -> List[str]:
        return list(texts)


class ChonEncoderTheoNgonNgu(VisualEncoder):
    """Route English to BTC CLIP and Vietnamese to the aligned mCLIP tower.

    This is deliberately a text-encoder router, not a translator.  Both text
    towers must emit vectors in the same image space because the downstream
    index is shared.  Explicit translation experiments still use
    ``DichRoiEncode`` below and bypass this router in ``Pipeline``.
    """

    def __init__(self, english_encoder: VisualEncoder,
                 vietnamese_encoder: VisualEncoder):
        en_dim = getattr(english_encoder, "dim", None)
        vi_dim = getattr(vietnamese_encoder, "dim", None)
        if en_dim is not None and vi_dim is not None and en_dim != vi_dim:
            raise ValueError(
                "Hai encoder phải cùng không gian vector: "
                f"English={en_dim}, Vietnamese={vi_dim}")
        self.english_encoder = english_encoder
        self.vietnamese_encoder = vietnamese_encoder
        self.dim = en_dim if en_dim is not None else vi_dim
        self.route_counts = {"english": 0, "vietnamese": 0}

    def encode_text(self, text: str) -> List[float]:
        if co_dau_tieng_viet(text):
            self.route_counts["vietnamese"] += 1
            return self.vietnamese_encoder.encode_text(text)
        self.route_counts["english"] += 1
        return self.english_encoder.encode_text(text)


class OpusMTTranslator(Translator):
    """Helsinki-NLP/opus-mt-vi-en — Marian, ~300 MB, chạy CPU được.

    Vì sao model này: `VietAI/envit5-translation` chất lượng nhỉnh hơn nhưng
    tokenizer T5 của nó vỡ trên transformers 5.14 (`TypeError: argument 'vocab':
    'dict' object cannot be converted to 'Sequence'`). Opus-MT nhẹ hơn 5 lần và
    chạy được ngay, nên dùng làm baseline dịch. Nó không còn là đường runtime
    mặc định; nếu sau này T5 sửa được thì thêm một lớp `Translator` nữa rồi ĐO.

    ⚠ Cần `sentencepiece`. `MarianTokenizer` không nằm trong `AutoTokenizer` của
    transformers 5.x nên phải gọi lớp trực tiếp.
    """

    MODEL_ID = "Helsinki-NLP/opus-mt-vi-en"

    def __init__(self, batch: int = 16, num_beams: int = 4):
        self.batch = int(batch)
        self.num_beams = int(num_beams)
        self._mod = None
        self._tok = None
        self._cache: Dict[str, str] = {}

    def _nap(self):
        if self._mod is not None:
            return
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        self._torch = torch
        self._tok = MarianTokenizer.from_pretrained(self.MODEL_ID)
        self._mod = MarianMTModel.from_pretrained(self.MODEL_ID).eval()

    def dich(self, texts: List[str]) -> List[str]:
        texts = [str(t) for t in texts]
        can = [t for t in texts if t not in self._cache]
        if can:
            self._nap()
            for i in range(0, len(can), self.batch):
                lo = can[i:i + self.batch]
                x = self._tok(lo, return_tensors="pt", padding=True,
                              truncation=True, max_length=256)
                with self._torch.no_grad():
                    y = self._mod.generate(**x, max_length=256,
                                           num_beams=self.num_beams)
                for goc, ra in zip(lo, self._tok.batch_decode(
                        y, skip_special_tokens=True)):
                    self._cache[goc] = ra.strip()
        return [self._cache[t] for t in texts]


class OpusMTEnViTranslator(Translator):
    """Helsinki-NLP/opus-mt-en-vi — chỉ dùng ở ĐẦU RA của QA.

    Đây là chiều đối xứng với :class:`OpusMTTranslator`. Nó tuyệt đối không
    tham gia chấm cosine: ứng viên vẫn là tiếng Anh trong không gian ảnh–chữ,
    chỉ đáp án đã thắng mới được dịch. Model được nạp lười và có cache trong
    bộ nhớ để đường màu/số (đã có bảng tay trong ``core.qa``) không phải tải
    model chỉ để dịch các lớp đóng.
    """

    MODEL_ID = "Helsinki-NLP/opus-mt-en-vi"

    def __init__(self, batch: int = 16, num_beams: int = 4,
                 local_files_only: bool = False):
        self.batch = int(batch)
        self.num_beams = int(num_beams)
        self.local_files_only = bool(local_files_only)
        self._mod = None
        self._tok = None
        self._cache: Dict[str, str] = {}

    def _nap(self):
        if self._mod is not None:
            return
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        kw = {"local_files_only": True} if self.local_files_only else {}
        self._torch = torch
        self._tok = MarianTokenizer.from_pretrained(self.MODEL_ID, **kw)
        self._mod = MarianMTModel.from_pretrained(self.MODEL_ID, **kw).eval()

    def dich(self, texts: List[str]) -> List[str]:
        texts = [str(t) for t in texts]
        can = [t for t in texts if t not in self._cache]
        if can:
            self._nap()
            for i in range(0, len(can), self.batch):
                lo = can[i:i + self.batch]
                x = self._tok(lo, return_tensors="pt", padding=True,
                              truncation=True, max_length=64)
                with self._torch.no_grad():
                    y = self._mod.generate(**x, max_length=64,
                                           num_beams=self.num_beams)
                for goc, ra in zip(lo, self._tok.batch_decode(
                        y, skip_special_tokens=True)):
                    self._cache[goc] = ra.strip()
        return [self._cache[t] for t in texts]


class NLLBTranslator(Translator):
    """facebook/nllb-200-distilled-600M — ~2.4 GB, chậm hơn nhưng dịch sát hơn.

    Dùng khi opus-mt làm mất chi tiết. NLLB là model đa ngữ 200 thứ tiếng của
    Meta, bản distilled 600M là mức nhỏ nhất còn giữ được chất lượng.

    Tokenizer bản FAST (dựa trên `tokenizers`), nên không vướng lỗi chuyển vocab
    của T5 trên transformers 5.x.
    """

    MODEL_ID = "facebook/nllb-200-distilled-600M"
    NGUON, DICH = "vie_Latn", "eng_Latn"

    def __init__(self, batch: int = 8, num_beams: int = 4):
        self.batch = int(batch)
        self.num_beams = int(num_beams)
        self._mod = None
        self._cache: Dict[str, str] = {}

    def _nap(self):
        if self._mod is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, NllbTokenizerFast

        self._torch = torch
        self._tok = NllbTokenizerFast.from_pretrained(self.MODEL_ID,
                                                      src_lang=self.NGUON)
        self._mod = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_ID).eval()
        self._bos = self._tok.convert_tokens_to_ids(self.DICH)

    def dich(self, texts: List[str]) -> List[str]:
        texts = [str(t) for t in texts]
        can = [t for t in texts if t not in self._cache]
        if can:
            self._nap()
            for i in range(0, len(can), self.batch):
                lo = can[i:i + self.batch]
                x = self._tok(lo, return_tensors="pt", padding=True,
                              truncation=True, max_length=256)
                with self._torch.no_grad():
                    y = self._mod.generate(**x, forced_bos_token_id=self._bos,
                                           max_length=256,
                                           num_beams=self.num_beams)
                for goc, ra in zip(lo, self._tok.batch_decode(
                        y, skip_special_tokens=True)):
                    self._cache[goc] = ra.strip()
        return [self._cache[t] for t in texts]


class DichRoiEncode(VisualEncoder):
    """Bọc một encoder: câu tiếng Việt thì dịch trước, rồi mới encode.

    Giữ nguyên giao diện `VisualEncoder` nên cắm thẳng vào `MultiChannelRetriever`
    mà không sửa gì ở tầng dưới.

    `chi_dich_tieng_viet=False` ép dịch mọi câu — chỉ dùng để đo, vì nó khiến câu
    tiếng Anh cũng bị chạy qua model dịch.
    """

    def __init__(self, encoder: VisualEncoder, translator: Translator,
                 chi_dich_tieng_viet: bool = True):
        self.encoder = encoder
        self.translator = translator
        self.chi_dich_tieng_viet = chi_dich_tieng_viet
        self.dim = getattr(encoder, "dim", None)
        self.ban_dich: Dict[str, str] = {}      # câu gốc -> câu đã dịch

    def _dich_mot(self, text: str) -> str:
        if self.chi_dich_tieng_viet and not co_dau_tieng_viet(text):
            return text
        if text not in self.ban_dich:
            self.ban_dich[text] = self.translator.dich([text])[0]
        return self.ban_dich[text]

    def nap_truoc(self, texts: List[str]) -> None:
        """Dịch sẵn cả lô. Nhanh hơn nhiều so với gọi lẻ từng câu."""
        can = [t for t in texts
               if t not in self.ban_dich
               and not (self.chi_dich_tieng_viet and not co_dau_tieng_viet(t))]
        if can:
            for goc, ra in zip(can, self.translator.dich(can)):
                self.ban_dich[goc] = ra

    def encode_text(self, text: str) -> List[float]:
        return self.encoder.encode_text(self._dich_mot(text))
