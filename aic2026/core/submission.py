"""Submission path (fix #3): build, diversify, write, validate, zip.

Hard rules from the official guide encoded here:
  * <= 100 rows per query file, no header row.
  * UTF-8, comma delimiter, CRLF line endings, csv-minimal quoting.
  * video name has NO .mp4; frame ids are bare integers.
  * TRAKE row must carry EXACTLY N frame ids (N = number of query events).
Strategy encoded here:
  * always fill toward 100 rows (free recall — no penalty for extra guesses);
  * keep the top ranks by score, but diversify VIDEOS at lower ranks to hedge
    R@20/50/100 without disturbing R@1/R@5.
A malformed submission still burns 1 of the 3 attempts, so validate before zip.
"""
from __future__ import annotations

import csv
import io
import os
import zipfile
from typing import Dict, List, Optional, Sequence, Tuple

MAX_ROWS = 100


# --------------------------------------------------------------------------- #
# Row -> CSV fields
# --------------------------------------------------------------------------- #
def _row_fields(task: str, row: Dict) -> List[str]:
    if task == "kis":
        return [row["video_id"], str(int(row["frame_idx"]))]
    if task == "qa":
        return [row["video_id"], str(int(row["frame_idx"])), str(row.get("answer", ""))]
    if task == "trake":
        return [row["video_id"], *[str(int(f)) for f in row["frames"]]]
    raise ValueError(f"unknown task: {task!r}")


def rows_to_csv(task: str, rows: Sequence[Dict]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL)
    for r in rows[:MAX_ROWS]:
        writer.writerow(_row_fields(task, r))
    return buf.getvalue()


def write_csv(path: str, task: str, rows: Sequence[Dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(rows_to_csv(task, rows))


# --------------------------------------------------------------------------- #
# Ranked-candidate -> rows, with diversity fill toward 100
# --------------------------------------------------------------------------- #
def _khoa_dong(r: Dict) -> Tuple:
    """Khoá NHẬN DẠNG một dòng nộp, đúng như BTC nhìn thấy nó.

    Dễ tính với dòng thiếu trường: `diversify_tail` là hàm dùng chung và có
    caller (kể cả test) truyền dict hình dạng khác. Khử trùng là tính năng phụ
    trợ — nó không được phép làm nổ đường nộp bài vì một dict lạ.
    """
    vid = r.get("video_id")
    if "frames" in r:
        return (vid, tuple(int(f) for f in r["frames"]))
    if "frame_idx" in r:
        return (vid, int(r["frame_idx"]))
    return (vid, tuple(sorted((str(k), str(v)) for k, v in r.items())))


def khu_trung(rows: Sequence[Dict]) -> List[Dict]:
    """Bỏ dòng render ra GIỐNG HỆT nhau, giữ thứ tự.

    Hai `Event` khác nhau vẫn có thể ra cùng một dòng: audit dữ liệu BTC đếm
    được **614 `frame_idx` trùng** trên 192 video, nên hai keyframe khác nhau
    của cùng một video có thể mang cùng `frame_idx`. `_hop_nhat` khử trùng theo
    `event_id` nên không bắt được ca này.

    Hậu quả không lớn nhưng có thật: `Final` lấy MAX trên tiền tố, nên dòng
    trùng là một slot vứt đi — quan sát thật trên bài nộp 7,6 điểm,
    `query-p1-9-kis` mất một slot kiểu này.
    """
    ra, da = [], set()
    for r in rows:
        k = _khoa_dong(r)
        if k in da:
            continue
        da.add(k)
        ra.append(r)
    return ra


def diversify_tail(rows: List[Dict], key=lambda r: r["video_id"],
                   head: int = 5, max_rows: int = MAX_ROWS) -> List[Dict]:
    """Keep the first `head` rows fixed; round-robin the rest across distinct
    keys so lower ranks cover more videos. Order within a key is preserved.

    Khử trùng TRƯỚC khi cắt `max_rows`, để chỗ trống được lấp bằng một giả
    thuyết mới chứ không bỏ không."""
    rows = khu_trung(rows)
    top = rows[:head]
    rest = rows[head:]
    buckets: Dict[str, List[Dict]] = {}
    order: List[str] = []
    for r in rest:
        k = key(r)
        if k not in buckets:
            buckets[k] = []
            order.append(k)
        buckets[k].append(r)
    interleaved: List[Dict] = []
    while any(buckets[k] for k in order):
        for k in order:
            if buckets[k]:
                interleaved.append(buckets[k].pop(0))
    return (top + interleaved)[:max_rows]


def kis_rows(candidates, max_rows: int = MAX_ROWS) -> List[Dict]:
    rows = [{"video_id": c.video_id, "frame_idx": c.frame_idx} for c in candidates]
    return diversify_tail(rows, max_rows=max_rows)


def qa_rows(candidates, answer: str, max_rows: int = MAX_ROWS) -> List[Dict]:
    rows = [{"video_id": c.video_id, "frame_idx": c.frame_idx, "answer": answer}
            for c in candidates]
    return diversify_tail(rows, max_rows=max_rows)


def qa_rows_per_candidate(candidate_answers, max_rows: int = MAX_ROWS) -> List[Dict]:
    """QA rows when a VLM answered each candidate frame independently."""
    rows = [{"video_id": c.video_id, "frame_idx": c.frame_idx,
             "answer": answer}
            for c, answer in candidate_answers]
    return diversify_tail(rows, max_rows=max_rows)


def trake_rows(align_results, max_rows: int = MAX_ROWS) -> List[Dict]:
    """align_results: iterable of AlignResult (each already carries N frames)."""
    rows = [{"video_id": a.video_id, "frames": list(a.frames)} for a in align_results]
    return diversify_tail(rows, max_rows=max_rows)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_csv(path: str, task: str, n_events: Optional[int] = None
                 ) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            reader = list(csv.reader(fh))
    except UnicodeDecodeError:
        return False, ["file is not valid UTF-8"]
    except FileNotFoundError:
        return False, [f"file not found: {path}"]

    if not reader:
        errors.append("empty file")
    if len(reader) > MAX_ROWS:
        errors.append(f"{len(reader)} rows > {MAX_ROWS} limit")

    for i, fields in enumerate(reader, start=1):
        if not fields:
            errors.append(f"row {i}: empty")
            continue
        vid = fields[0]
        if vid.lower().endswith(".mp4"):
            errors.append(f"row {i}: video name must not include .mp4 ({vid})")
        if task == "kis":
            if len(fields) != 2:
                errors.append(f"row {i}: KIS needs 2 fields, got {len(fields)}")
            else:
                _check_int(fields[1], i, errors)
        elif task == "qa":
            if len(fields) < 3:
                errors.append(f"row {i}: QA needs >=3 fields, got {len(fields)}")
            else:
                _check_int(fields[1], i, errors)
                ans = ",".join(fields[2:])  # answer may itself contain commas
                if len(ans) > 100:
                    errors.append(f"row {i}: answer exceeds 100 chars")
        elif task == "trake":
            frames = fields[1:]
            if not frames:
                errors.append(f"row {i}: TRAKE has no frames")
            for f in frames:
                _check_int(f, i, errors)
            if n_events is not None and len(frames) != n_events:
                errors.append(
                    f"row {i}: TRAKE expects {n_events} frames, got {len(frames)}")
        else:
            errors.append(f"unknown task {task!r}")
    return (len(errors) == 0), errors


def _check_int(value: str, row: int, errors: List[str]) -> None:
    v = value.strip()
    if v != value:
        errors.append(f"row {row}: field has stray whitespace ({value!r})")
    try:
        int(v)
    except ValueError:
        errors.append(f"row {row}: frame id not an integer ({value!r})")


# --------------------------------------------------------------------------- #
# Packaging: submission/ dir -> zip
# --------------------------------------------------------------------------- #
def build_submission_zip(zip_path: str, files: Dict[str, Tuple[str, Sequence[Dict]]],
                         n_events: Dict[str, int] | None = None
                         ) -> Tuple[bool, List[str]]:
    """files: {"query-1-kis.csv": ("kis", rows), ...}. Validates every file, then
    writes a zip whose members live under a top-level submission/ folder.
    Returns (ok, errors); on error nothing is written."""
    n_events = n_events or {}
    all_errors: List[str] = []
    rendered: Dict[str, str] = {}
    for name, (task, rows) in files.items():
        content = rows_to_csv(task, rows)
        rendered[name] = content
        # validate the rendered content via a round-trip
        reader = list(csv.reader(io.StringIO(content)))
        ok, errs = _validate_rows(reader, task, n_events.get(name))
        all_errors.extend(f"{name}: {e}" for e in errs)
    if all_errors:
        return False, all_errors

    os.makedirs(os.path.dirname(os.path.abspath(zip_path)) or ".", exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in rendered.items():
            zf.writestr(f"submission/{name}", content)
    return True, []


def _validate_rows(reader, task, n_events):
    errors: List[str] = []
    if len(reader) > MAX_ROWS:
        errors.append(f"{len(reader)} rows > {MAX_ROWS}")
    for i, fields in enumerate(reader, start=1):
        if task == "trake" and n_events is not None and len(fields[1:]) != n_events:
            errors.append(f"row {i}: expects {n_events} frames, got {len(fields[1:])}")
    return (len(errors) == 0), errors
