"""Deterministic first-stage compiler for KIS, QA and TRAKE queries."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class CompiledEvent:
    text: str
    semantic_type: str = "MOMENT"
    modality_hint: Optional[str] = None


@dataclass(frozen=True)
class CompiledQuery:
    task: str
    original_text: str
    visual_queries: List[str]
    ocr_queries: List[str] = field(default_factory=list)
    asr_queries: List[str] = field(default_factory=list)
    events: List[CompiledEvent] = field(default_factory=list)
    answer_type: str = "free_text"
    retrieval_query: str = ""
    question: str = ""
    # Nguyên văn tiếng Việt, dành cho kênh chữ/OCR. Xem `Query.text_goc`.
    text_goc: str = ""


_TEXT = re.compile(r'(?:chữ|biển|logo|nhãn|tiêu đề|dòng chữ|text|sign)\s+["“]?([^"”;,]+)', re.I)
_SPEECH = re.compile(r'(?:nói|hỏi|trả lời|phát biểu|announce|says?)\s+["“]?([^"”;,]+)', re.I)
_ORDER = re.compile(r"\b(?:sau đó|tiếp theo|rồi|sau khi|then|next|afterwards)\b", re.I)


def _answer_type(text: str) -> str:
    low = text.lower().strip()
    rules = (("person", r"\b(ai|who)\b"), ("location", r"\b(ở đâu|where)\b"),
             ("time", r"\b(khi nào|mấy giờ|when)\b"),
             ("number", r"\b(bao nhiêu|how many|how much)\b"),
             ("color", r"\b(màu gì|what colo(?:u)?r)\b"),
             ("boolean", r"\b(có phải|is there|does|do)\b"))
    return next((name for name, pattern in rules if re.search(pattern, low)), "free_text")


def _channels(text: str) -> tuple[List[str], List[str]]:
    return ([m.group(1).strip() for m in _TEXT.finditer(text)],
            [m.group(1).strip() for m in _SPEECH.finditer(text)])


def compile_query(task: str, text: str,
                  sub_events: Optional[List[str]] = None,
                  text_goc: str = "") -> CompiledQuery:
    task = task.lower().strip()
    if task not in {"kis", "qa", "trake"}:
        raise ValueError(f"unsupported task: {task}")
    clean = " ".join(text.split())
    ocr, asr = _channels(clean)
    if task == "kis":
        return CompiledQuery(task=task, original_text=clean,
                             visual_queries=[clean], ocr_queries=ocr,
                             asr_queries=asr, retrieval_query=clean,
                             text_goc=text_goc)
    if task == "qa":
        retrieval, question = _split_qa(clean)
        return CompiledQuery(
            task=task, original_text=clean, visual_queries=[retrieval],
            ocr_queries=ocr, asr_queries=asr,
            answer_type=_answer_type(question), retrieval_query=retrieval,
            question=question, text_goc=text_goc)
    pieces = sub_events or [x.strip(" ,.;") for x in _ORDER.split(clean)
                            if x.strip(" ,.;")]
    if not pieces:
        pieces = [clean]
    events: List[CompiledEvent] = []
    for i, part in enumerate(pieces):
        po, pa = _channels(part)
        low = part.lower()
        if i == 0 and re.search(r"\b(bắt đầu|đầu tiên|first|starts?)\b", low):
            semantic = "FIRST_TRANSITION"
        elif re.search(r"\b(chạm|va|bắt tay|contact|touch|hit)\b", low):
            semantic = "FIRST_CONTACT"
        elif re.search(r"\b(cao nhất|cực đại|đỉnh|apex|highest)\b", low):
            semantic = "APEX"
        elif re.search(r"\b(trong khi|suốt|đang|while|during)\b", low):
            semantic = "INTERVAL"
        else:
            semantic = "MOMENT"
        hint = "ocr" if po else ("asr" if pa else None)
        events.append(CompiledEvent(part, semantic, hint))
    return CompiledQuery(task=task, original_text=clean,
                         visual_queries=[e.text for e in events],
                         ocr_queries=ocr, asr_queries=asr, events=events,
                         retrieval_query=clean, text_goc=text_goc)


def _split_qa(text: str) -> tuple[str, str]:
    """Separate scene description from the question without an LLM.

    The deterministic split is deliberately conservative. A learned/LLM
    compiler can replace it after the expanded labelled set exists.
    """
    stripped = text.rstrip(" ?")
    parts = [part.strip() for part in re.split(r"[,;]", stripped)
             if part.strip()]
    if len(parts) >= 2:
        question = parts[-1]
        answer_type = _answer_type(question)
        if answer_type == "free_text":
            return stripped, stripped
        anchor = _qa_visual_anchor(question, answer_type)
        retrieval = " ".join(parts[:-1] + ([anchor] if anchor else []))
        return retrieval.strip() or stripped, question
    match = re.match(
        r"(?i)^(?:trong cảnh|in the scene)\s+(.+?)\s+"
        r"((?:ai|what|who|where|when|how|màu gì|bao nhiêu|ở đâu).+)$",
        stripped)
    if match:
        question = match.group(2).strip()
        if _answer_type(question) != "free_text":
            return match.group(1).strip(), question

    # ĐÃ THỬ VÀ BỎ 2026-08-12: viết lại câu hỏi một cụm thành cụm tả cảnh
    # ("What color shirt is the man wearing?" -> "shirt the man wearing") bằng cách
    # cắt khung nghi vấn ở đầu. Đo trên 30 câu QA thật, trần định vị:
    #     giữ nguyên câu hỏi   0.0667 ± 0.0337
    #     cắt khung nghi vấn   0.0467 ± 0.0248      -> KÉM HƠN
    # Không phải vì cắt sai (30/30 câu cắt ra cụm đọc được), mà vì dạng câu KHÔNG
    # phải nút cổ chai: câu hỏi tả một CHI TIẾT ("cái áo"), còn encoder ảnh xếp
    # hạng theo TOÀN CẢNH. Bỏ mấy token nghi vấn không thêm thông tin cảnh nào.
    #
    # Nút cổ chai thật là THIẾU MÔ TẢ CẢNH trong truy vấn. Đề thi thật hỏi
    # "thương hiệu nào trên container ĐÓ?" — có cảnh đi trước. Vì vậy dev set dựng
    # câu QA thành "<mô tả cảnh>, <câu hỏi>" (xem `tao_dev_set.doc_qa`), và nhánh
    # nhiều cụm ở trên lo phần tách. Đừng thử lại cách cắt khung nghi vấn.
    return stripped, stripped


def _qa_visual_anchor(question: str, answer_type: str) -> str:
    """Keep entities/attributes from the question in the retrieval query."""
    patterns = {
        "color": [
            r"(?i)\b(?:có\s+)?màu\s+gì\b",
            r"(?i)^what\s+colo(?:u)?r\s+(?:is|are)\s+",
        ],
        "number": [
            r"(?i)\b(?:có\s+)?bao\s+nhiêu\b",
            r"(?i)^how\s+many\s+",
        ],
        "person": [r"(?i)^ai\b", r"(?i)^who\b"],
        "location": [r"(?i)\bở\s+đâu\b", r"(?i)^where\b"],
        "time": [r"(?i)\b(?:khi|lúc)\s+nào\b", r"(?i)^when\b"],
    }
    anchor = question
    for pattern in patterns.get(answer_type, []):
        anchor = re.sub(pattern, " ", anchor)
    anchor = re.sub(r"(?i)\b(?:là|is|are|có)\s*$", " ", anchor)
    return " ".join(anchor.split()).strip(" ?")
