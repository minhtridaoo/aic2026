#!/usr/bin/env python
"""BUNG RA ĐỂ SỬA TAY / ĐÓNG LẠI ĐỂ NỘP — an toàn cho việc sửa giữa giờ thi.

    python -X utf8 dong_goi.py bung  out/NOP/dot1_C.zip      # -> out/NOP/SUA/
    python -X utf8 dong_goi.py dong  out/NOP/dot1_D.zip      # SUA/ -> zip + kiểm

VÌ SAO CÓ FILE NÀY
------------------
21/08, `dot1_B.zip` bị giải nén ra rồi **nén lại bằng Windows**. Windows bọc
thêm một tầng thư mục mang tên file, nên trong zip thành `dot1_B/submission/...`
thay vì `submission/...`. Luật BTC đòi `submission/` nằm NGAY GỐC — nộp bản đó
là 0 điểm toàn bộ, mà nhìn bằng mắt thì zip vẫn "trông đúng".

Nên: sửa tay thì sửa trong `SUA/submission/`, và đóng gói bằng lệnh này chứ
đừng bấm chuột phải > Send to > Compressed folder.

Lệnh `dong` KHÔNG ghi ra zip nếu còn bất kỳ lỗi định dạng nào.
"""
import argparse
import csv
import io
import json
import os
import shutil
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

GOC = os.path.dirname(os.path.abspath(__file__))
SUA = os.path.join(GOC, "out", "NOP", "SUA")


def _de(path):
    with open(path, encoding="utf-8") as fh:
        return {q["qid"]: q for q in json.load(fh)["queries"]}


def _so(qid):
    return int("".join(c for c in qid.split("-")[2] if c.isdigit()) or 0)


def bung(a):
    de = _de(a.de)
    if os.path.isdir(SUA):
        shutil.rmtree(SUA)
    ra = os.path.join(SUA, "submission")
    os.makedirs(ra)
    z = zipfile.ZipFile(a.zip)
    n = 0
    for ten in z.namelist():
        if not ten.endswith(".csv"):
            continue
        with open(os.path.join(ra, os.path.basename(ten)), "wb") as fh:
            fh.write(z.read(ten))
        n += 1
    print(f"  {n} file -> {ra}")

    # Bảng tra để sửa cho đúng câu: mở cạnh CSV, khỏi phải lật lại đề.
    doc = [f"BẢNG TRA — sửa xong chạy: python -X utf8 dong_goi.py dong <ra.zip>",
           f"Sửa file trong: {ra}", "",
           "CỘT: KIS = video,frame | QA = video,frame,đáp án | "
           "TRAKE = video,f1..fN", ""]
    for qid in sorted(de, key=_so):
        q = de[qid]
        n_ev = len(q.get("sub_events") or [])
        can = {"kis": 2, "qa": 3}.get(q["task"], n_ev + 1)
        doc.append(f"{'=' * 88}\n{qid}   [{q['task'].upper()}]   "
                   f"PHẢI CÓ ĐÚNG {can} CỘT, 100 DÒNG")
        if q["task"] == "trake":
            doc.append(f"   {n_ev} sự kiện -> {n_ev} số khung mỗi dòng")
            for i, e in enumerate(q.get("sub_events") or [], 1):
                doc.append(f"     E{i}: {e[:110]}")
        else:
            doc.append(f"   {q.get('text', '')[:400]}")
        if q.get("_cau_hoi_vi"):
            doc.append(f"   HỎI: {q['_cau_hoi_vi']}")
        p = os.path.join(ra, f"{qid}.csv")
        if os.path.exists(p):
            with open(p, encoding="utf-8", newline="") as fh:
                rows = list(csv.reader(fh))
            doc.append("   5 dòng đầu đang có:")
            for r in rows[:5]:
                doc.append(f"     {','.join(r)}")
    p = os.path.join(SUA, "BANG_TRA.txt")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(doc))
    print(f"  bảng tra -> {p}")
    print(f"\n  Sửa xong: python -X utf8 dong_goi.py dong out/NOP/dot1_D.zip")


def kiem(thu_muc, de):
    """[lỗi] theo ĐÚNG luật BTC. Rỗng = nộp được."""
    loi, canh = [], []
    ten = sorted(x[:-4] for x in os.listdir(thu_muc) if x.endswith(".csv"))
    la = sorted(x for x in os.listdir(thu_muc) if not x.endswith(".csv"))
    if la:
        loi.append(f"có file KHÔNG phải .csv trong submission/: {la}")
    if ten != sorted(de):
        thieu, thua = sorted(set(de) - set(ten)), sorted(set(ten) - set(de))
        if thieu:
            loi.append(f"THIẾU file: {thieu}")
        if thua:
            loi.append(f"THỪA file lạ: {thua}")

    for qid in sorted(set(ten) & set(de), key=_so):
        with open(os.path.join(thu_muc, f"{qid}.csv"), "rb") as fh:
            raw = fh.read()
        try:
            s = raw.decode("utf-8")
        except UnicodeDecodeError:
            loi.append(f"{qid}: KHÔNG phải UTF-8 (lưu lại bằng UTF-8)")
            continue
        if s.startswith("﻿"):
            loi.append(f"{qid}: có BOM ở đầu file — lưu 'UTF-8' chứ không "
                       f"'UTF-8 with BOM'")
        rows = [r for r in csv.reader(io.StringIO(s)) if r]
        task = de[qid]["task"]
        n_ev = len(de[qid].get("sub_events") or [])
        can = {"kis": 2, "qa": 3}.get(task, n_ev + 1)

        if not rows:
            loi.append(f"{qid}: RỖNG")
            continue
        if len(rows) > 100:
            loi.append(f"{qid}: {len(rows)} dòng > 100")
        elif len(rows) < 100:
            canh.append(f"{qid}: chỉ {len(rows)}/100 dòng — bỏ phí slot")
        if rows[0][0].strip().lower() in ("video", "video_id"):
            loi.append(f"{qid}: có dòng HEADER, phải bỏ")
        cot = {len(r) for r in rows}
        if cot != {can}:
            loi.append(f"{qid}: phải đúng {can} cột, đang có {sorted(cot)}"
                       + (f" ({n_ev} sự kiện)" if task == "trake" else ""))
            continue

        khoa = [(r[0], tuple(r[1:])) if task == "trake" else (r[0], r[1])
                for r in rows]
        if len(set(khoa)) != len(khoa):
            loi.append(f"{qid}: {len(khoa) - len(set(khoa))} dòng TRÙNG "
                       f"(trùng = phí slot)")
        for i, r in enumerate(rows, 1):
            v = r[0].strip()
            if not v or v != r[0] or v.lower().endswith(".mp4"):
                loi.append(f"{qid} dòng {i}: tên video xấu {r[0]!r} "
                           f"(không khoảng trắng thừa, không đuôi .mp4)")
                break
            so = r[1:2] if task == "qa" else r[1:]
            xau = [x for x in so if not x.strip().isdigit() or x != x.strip()]
            if xau:
                loi.append(f"{qid} dòng {i}: khung phải là số nguyên, "
                           f"nhận {xau}")
                break
        if task == "qa":
            d = max(len(r[2]) for r in rows)
            if d > 100:
                loi.append(f"{qid}: đáp án dài {d} ký tự > 100")
            rong = [i for i, r in enumerate(rows, 1) if not r[2].strip()]
            if rong:
                loi.append(f"{qid}: {len(rong)} dòng đáp án RỖNG "
                           f"(dòng {rong[:5]})")
    return loi, canh


def dong(a):
    de = _de(a.de)
    thu_muc = os.path.join(SUA, "submission")
    if not os.path.isdir(thu_muc):
        raise SystemExit(f"Chưa có {thu_muc} — chạy lệnh `bung` trước.")

    loi, canh = kiem(thu_muc, de)
    for c in canh:
        print(f"  ⚠ {c}")
    if loi:
        print("\n❌ KHÔNG ĐÓNG GÓI — sửa những chỗ này đã:")
        for x in loi:
            print("   ", x)
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(a.zip)) or ".", exist_ok=True)
    with zipfile.ZipFile(a.zip, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(thu_muc)):
            if not f.endswith(".csv"):
                continue
            with open(os.path.join(thu_muc, f), "rb") as fh:
                noi = fh.read()
            # Ép CRLF: Excel/Notepad trên Windows giữ CRLF, nhưng nhiều editor
            # (VS Code, Notepad++) lưu LF và luật BTC ghi rõ CRLF.
            noi = noi.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            if not noi.endswith(b"\r\n"):
                noi += b"\r\n"
            z.writestr(f"submission/{f}", noi)

    z = zipfile.ZipFile(a.zip)
    goc = sorted({n.split("/")[0] for n in z.namelist()})
    print(f"\n  -> {a.zip}")
    print(f"  {len(z.namelist())} file · thư mục gốc {goc}")
    print("  ✅ HỢP LỆ THEO LUẬT BTC — nộp được"
          if goc == ["submission"] else "  ❌ SAI THƯ MỤC GỐC")


def main():
    p = argparse.ArgumentParser(description="Bung ra sửa tay / đóng lại để nộp")
    p.add_argument("lenh", choices=["bung", "dong"])
    p.add_argument("zip")
    p.add_argument("--de", default=os.path.join(GOC, "data", "de_dot1.json"))
    a = p.parse_args()
    (bung if a.lenh == "bung" else dong)(a)


if __name__ == "__main__":
    main()
