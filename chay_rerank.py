#!/usr/bin/env python
"""XẾP LẠI BÀI NỘP BẰNG VLM — chạy trên Kaggle (cần ảnh keyframe).

    python -X utf8 chay_rerank.py out/bai_nop.zip \
        --config config_kaggle.yaml --de data/de_thunghiem.json \
        --sau 100 --ra out/bai_rerank.zip

    # bảo thủ / mạnh tay
    ... --che-do rrf     (mặc định, VLM chỉ góp phiếu)
    ... --che-do diem    (VLM toàn quyền xếp top đầu)

VÌ SAO
------
`R@100 = 57,5%` nhưng điểm thật chỉ `0,467/câu`: khung đúng đã nằm trong 100
dòng, chỉ là không đứng hạng 1. Script này để VLM nhìn 100 ảnh đó rồi xếp lại —
đúng việc mà mắt người làm, nhưng tự động nên đo được và mang đi thi được.

Nền tảng đo: `thu_vlm_cham_anh.py` trên 120 ảnh dev có GT thật (21/08) —
`gemini-3.5-flash-lite` chọn đúng 9/10 với mô tả TIẾNG VIỆT (đoán mò 1/10).

CHI PHÍ
-------
`sau=100`, `lo=10` -> 10 request/câu. 24 câu = 240 request, ~6s/request đo thật
=> **~24 phút**. Bậc lite có RPD 500 nên thoải mái. Cache đĩa: chạy lại không
tốn request lần hai.

TRAKE BỊ BỎ QUA, CÓ CHỦ Ý
-------------------------
Một dòng TRAKE là N khung phải khớp N sự kiện theo đúng thứ tự thời gian, không
phải một ảnh khớp một mô tả. Chấm nó cần một prompt khác hẳn và một phép đo
riêng; ghép bừa vào đây thì rủi ro cao mà chưa có gì bảo chứng. Các file TRAKE
được chép nguyên vẹn sang zip mới.
"""
import argparse
import csv
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026 import config as cau_hinh
from aic2026.core import submission as sub
from aic2026.core.providers import GeminiVLMProvider
from aic2026.data_plane.image_resolver import (EventImageResolver,
                                               ImageResolutionError)
from aic2026.retrieval.rerank_vlm import (CauHinhRerank, _Cache,
                                          rerank_mot_truy_van)
from xem_anh import bang_khung

GOC = Path(os.path.dirname(os.path.abspath(__file__)))


def _args():
    p = argparse.ArgumentParser(description="Xếp lại bài nộp bằng VLM chấm ảnh")
    p.add_argument("zip")
    p.add_argument("--config", default=None)
    p.add_argument("--de", default=str(GOC / "data" / "de_thunghiem.json"))
    p.add_argument("--ra", default=str(GOC / "out" / "bai_rerank.zip"))
    p.add_argument("--sau", type=int, default=100)
    p.add_argument("--lo", type=int, default=10)
    p.add_argument("--che-do", default="rrf", choices=("rrf", "diem"))
    p.add_argument("--model", default="gemini-3.5-flash-lite")
    p.add_argument("--nhip", type=float, default=5.0)
    p.add_argument("--chi", default="", help="chỉ làm các qid này")
    p.add_argument("--cache", default=str(GOC / "out" / "rerank_vlm_cache.json"))
    return p.parse_args()


def main():
    a = _args()
    config = cau_hinh.load_config(a.config)
    resolver = EventImageResolver.from_config(config)
    if not resolver.roots:
        raise SystemExit("❌ `qa.image_roots` rỗng — phải chạy trên Kaggle.")

    with open(a.de, encoding="utf-8") as fh:
        de = {q["qid"]: q for q in json.load(fh)["queries"]}

    with zipfile.ZipFile(a.zip) as z:
        bai = {os.path.basename(n)[:-4]:
               list(csv.reader(io.StringIO(z.read(n).decode("utf-8"))))
               for n in z.namelist() if n.lower().endswith(".csv")}

    chi = {x.strip() for x in a.chi.split(",") if x.strip()}
    if chi:
        bai = {k: v for k, v in bai.items() if k in chi}

    def loai(qid):
        return de.get(qid, {}).get("task") or (
            "trake" if "trake" in qid else "qa" if "qa" in qid else "kis")

    can = [q for q in bai if loai(q) != "trake"]
    videos = {r[0].strip() for q in can for r in bai[q][:a.sau]}
    print(f"{len(bai)} câu · {len(can)} câu đem rerank (bỏ TRAKE) · "
          f"{len(videos)} video")
    print("Nạp map-keyframes ...")
    khung2n, thieu_map = bang_khung(
        (config.get("offline") or {}).get("btc_data_root"), videos)
    print(f"  {len(khung2n):,d} cặp (video, frame_idx)"
          + (f" · ⚠ {len(thieu_map)} video thiếu map" if thieu_map else ""))

    bo = GeminiVLMProvider(
        cache_path=str(GOC / "out" / f"_rerank_qa_{a.model}.json"),
        model=a.model, max_tokens=1024, timeout_seconds=120, max_retries=1,
        min_interval_seconds=a.nhip)
    cache = _Cache(a.cache, a.model)
    cfg = CauHinhRerank(sau=a.sau, lo=a.lo, che_do=a.che_do)

    print(f"\nmodel {a.model} · sâu {a.sau} · lô {a.lo} · chế độ {a.che_do}\n")
    t0, files, n_events, doi = time.time(), {}, {}, 0

    for qid in sorted(bai, key=lambda t: int(
            "".join(c for c in t.split("-")[2] if c.isdigit()) or 0)):
        rows_csv, task = bai[qid], loai(qid)
        rows = [_row(task, r) for r in rows_csv]
        if task == "trake":
            n_events[f"{qid}.csv"] = len(rows_csv[0]) - 1
            files[f"{qid}.csv"] = (task, rows)
            print(f"  {qid:22s} TRAKE — chép nguyên, không rerank")
            continue

        # Mô tả gửi cho VLM là bản TIẾNG VIỆT gốc của BTC, không phải bản LLM
        # viết lại thành chú thích tiếng Anh: bản viết lại tồn tại để hợp với
        # encoder ảnh, còn Gemini đọc tiếng Việt tốt (đo 9/10) và bản gốc giữ
        # được chi tiết mà bản viết lại đã lược đi.
        mo_ta = de.get(qid, {}).get("text", "")
        if not mo_ta:
            print(f"  {qid:22s} ⚠ không có mô tả trong đề — bỏ qua")
            files[f"{qid}.csv"] = (task, rows)
            continue

        def duong_anh(r, _v=khung2n):
            n = _v.get((r["video_id"].strip().upper(), int(r["frame_idx"])))
            if n is None:
                return None
            try:
                return resolver.resolve(f"{r['video_id'].strip().upper()}#{n:05d}")
            except ImageResolutionError:
                return None

        moi, bc = rerank_mot_truy_van(rows, mo_ta, bo, duong_anh, cfg,
                                      cache=cache, qid=qid)
        cache.luu()                       # lưu sau MỖI câu: quota chết vẫn giữ
        files[f"{qid}.csv"] = (task, moi)
        doi += bc.doi_hang_1
        print(f"  {bc.mo_ta()}"
              + (f" · hạng-1-cũ giờ ở #{bc.hang_moi_cua_top1_cu}"
                 if bc.doi_hang_1 else ""))

    ok, loi = sub.build_submission_zip(a.ra, files, n_events=n_events)
    print(f"\n  {doi}/{len(can)} câu bị ĐỔI hạng 1")
    print(f"  {bo.backend_requests} request · {time.time() - t0:.0f}s")
    print(f"  zip ok={ok}")
    for e in loi[:8]:
        print("   ❌", e)
    if not ok:
        sys.exit(1)
    print(f"  -> {a.ra}")


def _row(task, r):
    if task == "kis":
        return {"video_id": r[0], "frame_idx": int(r[1])}
    if task == "qa":
        return {"video_id": r[0], "frame_idx": int(r[1]),
                "answer": ",".join(r[2:])}
    return {"video_id": r[0], "frames": [int(x) for x in r[1:]]}


if __name__ == "__main__":
    main()
