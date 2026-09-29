#!/usr/bin/env python
"""SOI BÀI NỘP KHI KHÔNG CÓ ĐÁP ÁN — đối chiếu video đã chọn với metadata BTC.

    python -X utf8 soi_ket_qua.py out/thunghiem_v2.zip --de data/de_thunghiem.json
    python -X utf8 soi_ket_qua.py out/thunghiem_v2.zip --de ... --sau 20

VÌ SAO CẦN
----------
Ngày thi không có ground truth, nên câu hỏi "bài nộp này tốt hay rác?" thường
chỉ trả lời được sau khi leaderboard hiện lên — tức là sau khi đã tiêu một
trong ba lần nộp. Nhưng bộ dữ liệu nhẹ của BTC có `media-info/<VIDEO>.json` với
`title` 100% và `description` 99,7% coverage. Với đề dạng BẢN TIN ("đàn hổ quý
hiếm ở miền Nam", "nghiên cứu ở Lausanne", "thị trấn ven biển có cá mập") thì
tiêu đề video nói thẳng nội dung. Đối chiếu tiêu đề với truy vấn cho một phép
kiểm rẻ, offline, 0 request: KHÔNG chứng minh được đúng, nhưng phát hiện được
sai rành rành và ước lượng được câu nào có cửa.

Đây cũng là cách bắt lại đúng chủng loại lỗi đã xảy ra ngày 20/08 (hệ thống đi
tìm bằng chuỗi "query-p1-1-kis"): lúc đó mọi phép kiểm định dạng đều xanh, chỉ
có nhìn vào NỘI DUNG kết quả mới thấy.

CÁCH ĐỌC
--------
* `video khác nhau` thấp = hệ thống đang dồn slot vào ít video. Tốt nếu đúng
  video, thảm hoạ nếu sai — vì R@k lấy max trên tiền tố, 100 dòng cùng một video
  sai là 0 điểm y hệt 1 dòng sai.
* Cột `trùng từ` = số từ khoá nội dung của truy vấn xuất hiện trong tiêu đề +
  mô tả + keywords của video. Đây là tín hiệu YẾU và một chiều: trùng cao là
  dấu hiệu tốt, trùng 0 KHÔNG có nghĩa là sai (cảnh quay không phải chủ đề bản
  tin thì tiêu đề chẳng nhắc tới).
"""
import argparse
import csv
import io
import json
import os
import re
import sys
import unicodedata
import zipfile
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026 import config as cau_hinh
from aic2026.core import dataset

GOC = os.path.dirname(os.path.abspath(__file__))

# Từ chức năng tiếng Việt: có mặt ở mọi câu nên không phân biệt được gì.
DUNG_TU = set("""
la cua va co mot nguoi nhung trong tren duoi voi den tu cho khi da se
doan clip video canh phan tim kiem chinh xac can hay biet o cac nhung
tai boi vi nhu the nay do day duoc thi ma hoac cung deu ca sau truoc
""".split())


def khong_dau(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "d")


def tu_noi_dung(text: str) -> set:
    """Từ mang nội dung của truy vấn, đã bỏ dấu và bỏ từ chức năng."""
    return {t for t in re.findall(r"[a-z0-9]+", khong_dau(text))
            if len(t) >= 3 and t not in DUNG_TU}


def _args():
    p = argparse.ArgumentParser(description="Soi bài nộp bằng metadata BTC")
    p.add_argument("zip", help="file .zip đã dựng bằng nop_bai.py")
    p.add_argument("--de", default=os.path.join(GOC, "data", "de_thunghiem.json"))
    p.add_argument("--config", default=None)
    p.add_argument("--sau", type=int, default=5, help="in N dòng đầu mỗi câu")
    p.add_argument("--ra", default=None, help="ghi thêm bản đầy đủ ra file .md")
    return p.parse_args()


def doc_zip(path):
    """{tên file csv: [dòng đã tách cột]} — giữ nguyên thứ tự dòng."""
    ra = {}
    with zipfile.ZipFile(path) as z:
        for ten in z.namelist():
            if not ten.lower().endswith(".csv"):
                continue
            noi_dung = z.read(ten).decode("utf-8")
            rows = [r for r in csv.reader(io.StringIO(noi_dung)) if r]
            ra[os.path.basename(ten)] = rows
    return ra


def main():
    args = _args()
    config = cau_hinh.load_config(args.config)
    goc_data = (config.get("offline") or {}).get("btc_data_root")

    print("Đang đọc metadata BTC (title/description/keywords) ...")
    paths = dataset.scan(goc_data)
    metas = {}
    for vid, vp in paths.items():
        if vp.meta is None:
            continue
        try:
            metas[vid] = dataset.load_media_info(vp.meta, vid)
        except Exception:                                     # noqa: BLE001
            pass
    print(f"  {len(metas)}/{len(paths)} video có metadata\n")

    with open(args.de, "r", encoding="utf-8") as fh:
        de = {q["qid"]: q for q in json.load(fh)["queries"]}

    files = doc_zip(args.zip)
    but = []

    def ra(s=""):
        print(s)
        but.append(s)

    ra(f"# Soi bài nộp `{os.path.basename(args.zip)}`")
    ra()
    tong = []
    for ten in sorted(files, key=lambda t: (
            int(re.search(r"(\d+)", t).group(1)) if re.search(r"(\d+)", t)
            else 0)):
        rows = files[ten]
        qid = ten[:-4]
        q = de.get(qid, {})
        text = q.get("text", "")
        task = q.get("task", "?")
        tu_q = tu_noi_dung(text)

        videos = [r[0].strip() for r in rows]
        dem = Counter(videos)
        khac = len(dem)

        ra(f"## {qid}  ·  {task.upper()}  ·  {len(rows)} dòng  ·  "
           f"{khac} video khác nhau")
        ra(f"> {text[:300]}")
        ra()
        ra("| # | video | frame | trùng từ | tiêu đề video |")
        ra("|---|---|---|---|---|")
        diem_dau = 0
        for i, r in enumerate(rows[:args.sau], 1):
            vid = r[0].strip()
            m = metas.get(vid)
            kho = tu_noi_dung(
                f"{m.title} {m.description} {' '.join(m.keywords)}") if m else set()
            trung = tu_q & kho
            if i == 1:
                diem_dau = len(trung)
            tieu_de = (m.title[:70] if m else "(không có metadata)")
            ra(f"| {i} | {vid} | {', '.join(x.strip() for x in r[1:])} | "
               f"{len(trung)} | {tieu_de} |")
        # Video nào trùng từ nhiều nhất trong CẢ 100 dòng — có thể đang bị chôn.
        tot = []
        for vid in dem:
            m = metas.get(vid)
            if not m:
                continue
            kho = tu_noi_dung(
                f"{m.title} {m.description} {' '.join(m.keywords)}")
            tot.append((len(tu_q & kho), videos.index(vid), vid, m.title))
        tot.sort(key=lambda x: (-x[0], x[1]))
        if tot and tot[0][0] > diem_dau:
            n, hang, vid, tieu = tot[0]
            ra()
            ra(f"⚠ Trong 100 dòng, video trùng từ NHIỀU NHẤT là **{vid}** "
               f"({n} từ) nhưng nó đứng hạng **{hang + 1}**, không phải hạng 1: "
               f"_{tieu[:90]}_")
        ra()
        tong.append((qid, task, khac, diem_dau,
                     tot[0][0] if tot else 0, tot[0][1] + 1 if tot else 0))

    ra("## Tổng hợp")
    ra()
    ra("| truy vấn | loại | #video | trùng từ @1 | trùng từ tốt nhất | hạng của nó |")
    ra("|---|---|---|---|---|---|")
    for qid, task, khac, d1, dmax, hang in tong:
        ra(f"| {qid} | {task} | {khac} | {d1} | {dmax} | {hang} |")

    if args.ra:
        with open(args.ra, "w", encoding="utf-8") as fh:
            fh.write("\n".join(but))
        print(f"\n-> {args.ra}")


if __name__ == "__main__":
    main()
