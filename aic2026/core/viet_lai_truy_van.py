"""Viết lại truy vấn tiếng Việt thành chú thích ảnh tiếng Anh (việc 4.5a).

VÌ SAO CẦN
----------
Đo 18–19/08 trên 120 câu KIS (`thu_nghiem_dich_vi.py`, base-224):

    EN người gõ tay   0.2950              <- lúc thi KHÔNG có
    VI thô            0.1750              <- đường thi cũ
    opus-mt vi-en     0.1767  +0.0017     <- dịch máy KHÔNG bù được
    LLM viết lại      0.2600  +0.0850*    <- LỚP NÀY
    tb(LLM×3, VI)     0.2617  +0.0867*    <- tốt nhất, đã chia đôi kiểm chứng

`*` = vượt 2·SE. Lấy lại **72%** khoảng cách người-dịch, R@1 từ 3,3% lên 9,2%
(người gõ tay là 10,8%). Model viết lại: `gemini-3.5-flash-lite`.

Chẩn đoán: **không phải rào ngôn ngữ, mà là TỪ NGỮ**. Vector câu opus dịch gần
trùng vector câu người viết (cos 0,93–0,98) nhưng hạng keyframe đúng vẫn nhảy
58 → 177, vì "đầu bếp" ra `cook` còn người viết `chef`. SigLIP2 học trên alt-text
nên nó thưởng đúng lối viết chú thích ảnh, không thưởng bản dịch đúng ngữ pháp.

⇒ Việc của lớp này KHÔNG phải dịch, mà là **viết lại thành chú thích ảnh**: chọn
danh từ cụ thể, thì hiện tại, chỉ tả thứ nhìn thấy được.

THIẾT KẾ
--------
* Nhận bất kỳ provider nào có `tra_loi_chu(system_instruction, prompt, schema)` —
  hôm nay là `GeminiVLMProvider`, mai đổi provider không phải sửa lớp này.
* **Gộp lô**: mỗi request viết lại `batch` câu, nên 120 câu = 6 request chữ.
* **Cache đĩa** khoá theo (model · phiên bản prompt · số bản · câu gốc): chạy lại
  không tốn quota, và Kaggle dùng lại được file cache của máy này.
* Xin `so_ban` cách diễn đạt cho mỗi câu, để đo cả ensemble truy vấn.
* Câu nào provider trả thiếu thì **giữ nguyên bản gốc** và ghi vào `thieu` —
  không bịa, không im lặng.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Đổi chuỗi này mỗi khi sửa prompt: nó nằm trong khoá cache nên bản dịch cũ không
# lẫn với bản mới.
PHIEN_BAN_PROMPT = "caption-v1"

HUONG_DAN = (
    "Bạn viết CHÚ THÍCH ẢNH tiếng Anh cho hệ thống tìm kiếm khung hình bằng "
    "SigLIP/CLIP. Mỗi mục là mô tả tiếng Việt của MỘT khung hình video.\n"
    "Với mỗi mục, viết câu tiếng Anh:\n"
    "  * 8–20 từ, thì hiện tại, giọng chú thích ảnh (\"A chef and a woman talk "
    "in a modern kitchen\").\n"
    "  * CHỈ tả thứ nhìn thấy được: chủ thể, hành động, vật thể, bối cảnh, màu "
    "sắc, chữ nổi bật. Không suy diễn, không cảm xúc, không tên riêng nếu bản "
    "gốc không có.\n"
    "  * Chọn DANH TỪ CỤ THỂ mà người chú thích ảnh hay dùng (chef, courtroom, "
    "warehouse, podium, motorbike) thay vì từ chung chung.\n"
    "  * Giữ nguyên mọi chi tiết đếm được và màu sắc của bản gốc.\n"
    "Không thêm bất kỳ chữ nào ngoài JSON."
)

SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "captions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["id", "captions"],
    },
}


def tu_config(config: Dict):
    """Dựng bộ viết lại từ `truy_van.viet_lai`, hoặc None nếu tắt/thiếu khoá.

    Trả None thay vì ném lỗi khi thiếu API key: mất lớp này chỉ mất ~0.057 điểm,
    còn ném lỗi giữa lúc thi là mất cả bài nộp. Bên gọi PHẢI in cảnh báo.
    """
    from .providers import build_llm

    cfg = dict(((config.get("truy_van") or {}).get("viet_lai") or {}))
    if not cfg.pop("enabled", False):
        return None, "tắt trong config"
    kind = str(cfg.pop("provider", "gemini_vlm"))
    khoa = {"gemini_vlm": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            "groq_vlm": ("GROQ_API_KEY",)}.get(kind, ())
    if khoa and not any(os.environ.get(k) for k in khoa):
        return None, f"thiếu {' hoặc '.join(khoa)}"

    so_ban = int(cfg.pop("so_ban", 3))
    lo = int(cfg.pop("lo", 10))
    cache = cfg.pop("cache_path", "out/viet_lai_truy_van.json")
    model = cfg.get("model") or "gemini-3.5-flash-lite"
    prov = build_llm({"llm": {
        "provider": kind, "model": model,
        "cache_path": f"out/vlm_qa_cache_{kind}_{model}.json",
        # Hạn giờ riêng: 45s của config là cho lô ẢNH; sinh chữ cả lô cần dài
        # hơn (đo 18/08: lô 20 câu vượt 45s).
        "timeout_seconds": float(cfg.get("timeout_seconds", 180)),
        "max_retries": int(cfg.get("max_retries", 1))}})
    if not hasattr(prov, "tra_loi_chu"):
        return None, f"{type(prov).__name__} không gọi được chỉ-chữ"
    return VietLaiTruyVan(prov, so_ban=so_ban, batch=lo,
                          cache_path=cache), f"{kind}/{model}"


def _khoa(model: str, so_ban: int, text: str) -> str:
    goc = "|".join([model, PHIEN_BAN_PROMPT, str(so_ban), text])
    return hashlib.sha1(goc.encode("utf-8")).hexdigest()


def _doc_json_array(text: str) -> List[Dict]:
    """JSON array từ câu trả lời, chịu được ```fence``` và lời dẫn thừa.

    Gemini bám định dạng vì có `response_format`; model open-weight qua Groq/
    Cloudflare thì hay thêm "Here is the JSON:" hoặc bọc code fence. Cắt lấy
    khối `[...]` ngoài cùng RẺ HƠN NHIỀU so với việc mất cả lô request.
    """
    sach = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(text).strip(),
                  flags=re.IGNORECASE)
    try:
        rows = json.loads(sach)
    except json.JSONDecodeError as exc:
        dau, cuoi = sach.find("["), sach.rfind("]")
        if dau < 0 or cuoi <= dau:
            raise RuntimeError(
                f"Bản viết lại không phải JSON hợp lệ: {sach[:300]!r}") from exc
        try:
            rows = json.loads(sach[dau:cuoi + 1])
        except json.JSONDecodeError:
            raise RuntimeError(
                f"Bản viết lại không phải JSON hợp lệ: {sach[:300]!r}") from exc
    if not isinstance(rows, list):
        raise RuntimeError("Bản viết lại phải là JSON array")
    return [r for r in rows if isinstance(r, dict)]


def ap_dung(bo, queries: Sequence, ban_ghi: Sequence) -> List[str]:
    """Viết lại TẠI CHỖ những câu đang phải dùng bản tiếng Việt. Trả log.

    Chỉ đụng vào câu có `theo_chinh_sach=False` — tức đề chỉ phát bản tiếng
    Việt nên `chon_van_ban` đã phải đi chệch chính sách `en`. Câu nào đề đã có
    sẵn bản tiếng Anh của người ra đề thì KHÔNG đụng: bản người viết đo được
    0.3750 so với 0.3450 của bản LLM, viết lại chỉ làm tệ đi.

    Không ném lỗi ra ngoài: LLM chết thì giữ nguyên bản tiếng Việt và ghi vào
    log. Mất ~0.057 điểm còn hơn hỏng cả bài nộp.
    """
    can = [q for q, bg in zip(queries, ban_ghi)
           if not getattr(bg, "theo_chinh_sach", True)]
    if not can:
        return ["viết lại truy vấn: không câu nào cần (đề đã có bản tiếng Anh)"]

    goc: List[str] = []
    for q in can:
        if getattr(q, "sub_events", None):
            goc += [se.text for se in q.sub_events if str(se.text).strip()]
        elif str(getattr(q, "text", "")).strip():
            goc.append(q.text)

    try:
        bang = bo.viet_lai(goc)
    except Exception as exc:                      # noqa: BLE001 - xem docstring
        return [f"⚠ viết lại truy vấn THẤT BẠI, giữ nguyên bản tiếng Việt: "
                f"{str(exc)[:200]}"]

    def _moi(t: str) -> str:
        ban = bang.get(str(t).strip())
        # Lấy bản [0] của lô `so_ban`: đúng cấu hình đã đo trên Kaggle 19/08
        # (+0.0567 ± 0.0209). Trung bình nhiều bản ĐO ĐƯỢC LÀ TỆ HƠN ở so400m.
        return ban[0] if ban else t

    doi = 0
    for q in can:
        if getattr(q, "sub_events", None):
            for se in q.sub_events:
                t = _moi(se.text)
                doi += t != se.text
                se.text = t
        else:
            t = _moi(q.text)
            doi += t != q.text
            # Giữ nguyên văn tiếng Việt cho kênh OCR — xem `Query.text_goc`.
            if not getattr(q, "text_goc", ""):
                try:
                    q.text_goc = q.text
                except Exception:                     # noqa: BLE001
                    pass
            q.text = t

    log = [f"viết lại truy vấn: {doi}/{len(goc)} chuỗi · {bo.requests} request · "
           f"{bo.cache_hits} cache hit · model {bo.model}"]
    if bo.thieu:
        log.append(f"⚠ {len(bo.thieu)} chuỗi LLM không trả — giữ nguyên bản Việt")
    return log


class VietLaiTruyVan:
    """Việt -> `so_ban` chú thích tiếng Anh, gộp lô, có cache đĩa."""

    def __init__(self, provider, so_ban: int = 1, batch: int = 20,
                 cache_path: Optional[str] = "out/viet_lai_truy_van.json"):
        if so_ban < 1:
            raise ValueError("so_ban phải >= 1")
        if batch < 1:
            raise ValueError("batch phải >= 1")
        self.provider = provider
        self.so_ban = int(so_ban)
        self.batch = int(batch)
        self.cache_path = Path(cache_path) if cache_path else None
        self.model = str(getattr(provider, "model", type(provider).__name__))
        self.cache_hits = 0
        self.requests = 0
        self.thieu: List[str] = []
        self._cache: Dict[str, List[str]] = self._nap_cache()

    # -- cache ------------------------------------------------------------- #
    def _nap_cache(self) -> Dict[str, List[str]]:
        if self.cache_path is None or not self.cache_path.is_file():
            return {}
        try:
            du_lieu = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}
        return {k: [str(x) for x in v]
                for k, v in du_lieu.items() if isinstance(v, list)}

    def _ghi_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=1),
            encoding="utf-8")

    # -- việc chính -------------------------------------------------------- #
    def viet_lai(self, cau_vi: Sequence[str]) -> Dict[str, List[str]]:
        """{câu tiếng Việt: [chú thích tiếng Anh]}. Câu trượt giữ nguyên bản gốc."""
        duy_nhat = [t for t in dict.fromkeys(str(x).strip() for x in cau_vi) if t]
        thieu = []
        for text in duy_nhat:
            khoa = _khoa(self.model, self.so_ban, text)
            if khoa in self._cache:
                self.cache_hits += 1
            else:
                thieu.append(text)

        for i in range(0, len(thieu), self.batch):
            lo = thieu[i:i + self.batch]
            self._mot_lo(lo)
            self._ghi_cache()

        ra: Dict[str, List[str]] = {}
        for text in duy_nhat:
            ban = self._cache.get(_khoa(self.model, self.so_ban, text))
            if not ban:
                self.thieu.append(text)
                ban = [text]
            ra[text] = ban
        return ra

    def _mot_lo(self, lo: List[str]) -> None:
        muc = [f'{{"id": "q{j:03d}", "vi": {json.dumps(t, ensure_ascii=False)}}}'
               for j, t in enumerate(lo)]
        prompt = (
            f"Viết lại {len(lo)} mục dưới đây. Mỗi mục trả đúng {self.so_ban} "
            f"câu tiếng Anh KHÁC NHAU về cách diễn đạt (khác từ vựng, cùng nội "
            f"dung), theo thứ tự đã cho.\n"
            f'Định dạng trả về: [{{"id":"q000","captions":["...","..."]}}]\n\n'
            + "\n".join(muc))
        self.requests += 1
        tra_loi = self.provider.tra_loi_chu(HUONG_DAN, prompt, SCHEMA)
        theo_id = {str(r.get("id")): r for r in _doc_json_array(tra_loi)}
        for j, text in enumerate(lo):
            row = theo_id.get(f"q{j:03d}")
            if row is None:
                continue
            ban = [str(x).strip() for x in (row.get("captions") or [])
                   if str(x).strip()]
            if not ban:
                continue
            # Thiếu bản thì lặp lại bản cuối: giữ đúng số phần tử để bên gọi khỏi
            # phải xử lý riêng, mà vẫn không bịa nội dung mới.
            while len(ban) < self.so_ban:
                ban.append(ban[-1])
            self._cache[_khoa(self.model, self.so_ban, text)] = ban[:self.so_ban]
