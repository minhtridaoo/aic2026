#!/usr/bin/env python
"""GHÉP BÀI NỘP từ nhiều zip — lấy file tốt nhất của mỗi lần chạy.

    python -X utf8 ghep_bai_nop.py --goc out/L4.zip \
        --lay out/L3.zip:query-p1-22-qa.csv \
        --doi-dap-an query-p1-19-qa.csv="..." \
        --ra out/L5.zip

VÌ SAO CẦN
----------
Ngày thi có thể có nhiều lần chạy trong nhiều môi trường (local nhanh nhưng không
có ảnh; Kaggle có ảnh nhưng quota VLM tụt bậc model giữa chừng). Kết quả là bản
tốt nhất cho từng CÂU nằm rải ở các zip khác nhau — đúng chuyện đã xảy ra
20/08: bản có ghim thắng ở câu 19/4/20, bản không ghim thắng ở câu 22 vì lượt đó
Gemini 3.7 đọc ảnh thật còn lượt kia 3.6 bịa một đáp án cho mọi khung.

Cách sai để làm việc này là mở zip ra sửa tay. Script này giữ hai bảo đảm mà sửa
tay không có: **chỉ ghép nguyên file CSV**, không bao giờ ghép nửa file; và
**chạy lại toàn bộ phép kiểm định dạng** trên mọi file trước khi ghi zip, kể cả
những file chỉ chép qua. Nộp sai định dạng vẫn trừ một trong ba lần nộp.
"""
import argparse
import csv
import io
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026.core import submission as sub


def _loai(ten: str) -> str:
    t = ten.lower()
    for x in ("trake", "qa", "kis"):
        if x in t:
            return x
    raise ValueError(f"{ten}: không suy được loại câu từ tên file")


def doc(path: str):
    """{tên file csv: nội dung bytes} — bỏ tiền tố thư mục."""
    with zipfile.ZipFile(path) as z:
        return {os.path.basename(n): z.read(n)
                for n in z.namelist() if n.lower().endswith(".csv")}


def _args():
    p = argparse.ArgumentParser(description="Ghép bài nộp từ nhiều zip")
    p.add_argument("--goc", required=True, help="zip nền")
    p.add_argument("--lay", action="append", default=[],
                   help="zip.zip:ten_file.csv — lấy file này đè lên bản nền")
    p.add_argument("--doi-dap-an", action="append", default=[],
                   help="ten_file.csv=<đáp án mới> — đè cột answer của MỌI dòng")
    p.add_argument("--ra", required=True)
    return p.parse_args()


def main():
    a = _args()
    files = doc(a.goc)
    print(f"nền: {a.goc}  ({len(files)} file)")

    for muc in a.lay:
        zp, _, ten = muc.rpartition(":")
        if not zp:
            raise SystemExit(f"--lay phải có dạng zip.zip:ten.csv, nhận {muc!r}")
        nguon = doc(zp)
        if ten not in nguon:
            raise SystemExit(f"{zp} không có {ten}. Có: {sorted(nguon)[:5]} ...")
        cu, moi = files.get(ten), nguon[ten]
        files[ten] = moi
        print(f"  lấy {ten} từ {os.path.basename(zp)}"
              + ("  (giống bản nền — không đổi gì)" if cu == moi else ""))

    for muc in a.doi_dap_an:
        ten, _, moi = muc.partition("=")
        if ten not in files:
            raise SystemExit(f"không có {ten} trong zip nền")
        if len(moi) > 100:
            raise SystemExit(f"{ten}: đáp án {len(moi)} ký tự > trần 100 của BTC")
        rows = list(csv.reader(io.StringIO(files[ten].decode("utf-8"))))
        cu = {",".join(r[2:]) for r in rows}
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL)
        for r in rows:
            w.writerow([r[0], r[1], moi])
        files[ten] = buf.getvalue().encode("utf-8")
        print(f"  đổi đáp án {ten}: {len(cu)} đáp án cũ -> 1 đáp án mới")
        print(f"     cũ : {sorted(cu)[0][:78]}")
        print(f"     mới: {moi[:78]}")

    # Số sự kiện TRAKE lấy từ CHÍNH file, để phép kiểm bắt được file lệch cột.
    n_events, ra = {}, {}
    for ten, b in files.items():
        rows = list(csv.reader(io.StringIO(b.decode("utf-8"))))
        task = _loai(ten)
        if task == "trake":
            so = {len(r) - 1 for r in rows}
            if len(so) != 1:
                raise SystemExit(f"{ten}: số cột khung không đồng nhất: {sorted(so)}")
            n_events[ten] = so.pop()
        ra[ten] = (task, [_row(task, r) for r in rows])

    ok, loi = sub.build_submission_zip(a.ra, ra, n_events=n_events)
    print(f"\nzip ok={ok}")
    for e in loi[:10]:
        print("  ❌", e)
    if not ok:
        sys.exit(1)
    print(f"  -> {a.ra}")
    print(f"  {len(ra)} file · TRAKE n_events={sorted(set(n_events.values()))}")


def _row(task, r):
    if task == "kis":
        return {"video_id": r[0], "frame_idx": int(r[1])}
    if task == "qa":
        return {"video_id": r[0], "frame_idx": int(r[1]), "answer": ",".join(r[2:])}
    return {"video_id": r[0], "frames": [int(x) for x in r[1:]]}


if __name__ == "__main__":
    main()
