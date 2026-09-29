#!/usr/bin/env python
"""GHÉP HAI HỆ: giữ ĐẦU của hệ mình, lấy ĐUÔI của hệ kia.

    python -X utf8 ghep_hai_he.py --nen out/NOP/AIC2026Dot1C.zip \
        --kia "C:/.../New folder" --ra out/NOP/dot1_D.zip

VÌ SAO GHÉP KIỂU NÀY
--------------------
`Final = (1/5)·Σ_{k∈{1,5,20,50,100}} R@k`, `R@k = max` trên tiền tố. Slot 1 được
tính trong cả 5 số hạng, slot 61 chỉ trong 1. Nên:

  * Thay slot 1–60 = đánh cược lại toàn bộ câu -> KHÔNG làm, trừ khi hệ kia có
    bằng chứng mạnh hơn.
  * Thay slot 61–100 = giữ nguyên 4/5 số hạng của mình, mà nếu cả 60 dòng đầu
    của mình sai hết còn hệ kia đúng thì vẫn vớt được số hạng k=100 (0,2 điểm).

Bất đối xứng đó là lý do duy nhất để ghép: mất rất ít, được tới 0,2/câu.

KHÔNG ghép ở câu đã ghim video có bằng chứng chữ đọc được — ở đó 100 slot đổ
vào một video là quyết định đã có cơ sở, pha loãng nó là tự phá.
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

GOC = os.path.dirname(os.path.abspath(__file__))

# Câu ĐÃ ghim video bằng chữ đọc được (OCR/ASR/tiêu đề nói thẳng ra đáp án).
# Giữ nguyên 100%, không cho hệ kia chen vào.
GIU_NGUYEN = {
    "query-p1-4-kis",    # OCR "london zoo ... zsl conservation"
    "query-p1-5-kis",    # tiêu đề "MỰC XÀO ĐẬU HÀ LAN" + ASR nguyên văn
    "query-p1-6-kis",    # ASR "Lucara Diamond ... mỏ Karowe"
    "query-p1-12-kis",   # OCR "dâu mazut còn 15.562 dông/kg"  (hệ kia CŨNG chọn)
    "query-p1-18-kis",   # tiêu đề "BÚN GÀ XÁO SẢ"
    "query-p1-22-kis",   # tiêu đề "Môn Anh văn CĐ6 Động từ thêm 'ing'"
    "query-p1-24-kis",   # tiêu đề "Nghề đan đát"  (hệ kia CŨNG chọn)
    "query-p1-15-qa", "query-p1-17-qa", "query-p1-9-qa", "query-p1-3-qa",
    "query-p1-16-trake",
}


def _args():
    p = argparse.ArgumentParser(description="Ghép đuôi bài của hệ khác vào")
    p.add_argument("--nen", required=True, help="zip nền (hệ mình)")
    p.add_argument("--kia", required=True, help="thư mục CSV của hệ kia")
    p.add_argument("--ra", required=True)
    p.add_argument("--giu", type=int, default=60,
                   help="giữ bao nhiêu dòng đầu của hệ mình")
    return p.parse_args()


def main():
    a = _args()
    z = zipfile.ZipFile(a.nen)
    mp = {n.split("/")[-1][:-4]: n for n in z.namelist() if n.endswith(".csv")}
    kia = {f[:-4]: os.path.join(a.kia, f)
           for f in os.listdir(a.kia) if f.endswith(".csv")}

    ra, bao = {}, []
    for qid, ten in sorted(mp.items()):
        s = z.read(ten).decode("utf-8")
        rows = [r for r in csv.reader(io.StringIO(s)) if r]
        n_cot = len(rows[0])

        if qid in GIU_NGUYEN or qid not in kia:
            ra[qid] = rows
            continue

        with open(kia[qid], encoding="utf-8-sig") as fh:
            rk = [r for r in csv.reader(fh) if r]
        # Chỉ nhận dòng CÙNG SỐ CỘT — hệ kia là hệ khác, không đảm bảo cùng
        # dạng, mà sai số cột là 0 điểm cả câu.
        rk = [r for r in rk if len(r) == n_cot]

        khoa = {(r[0], tuple(r[1:])) for r in rows[:a.giu]}
        moi = list(rows[:a.giu])
        them = 0
        for r in rk:
            if len(moi) >= 100:
                break
            k = (r[0], tuple(r[1:]))
            if k in khoa:
                continue
            khoa.add(k)
            moi.append(r)
            them += 1
        for r in rows[a.giu:]:                      # còn chỗ thì trả lại của mình
            if len(moi) >= 100:
                break
            k = (r[0], tuple(r[1:]))
            if k in khoa:
                continue
            khoa.add(k)
            moi.append(r)
        ra[qid] = moi
        bao.append(f"  {qid:18s} giữ {a.giu} dòng đầu + {them} dòng của hệ kia"
                   f"  (top1 họ: {rk[0][0] if rk else '—'})")

    for b in bao:
        print(b)
    print(f"\n  giữ nguyên 100%: {len(mp) - len(bao)} câu")

    with zipfile.ZipFile(a.ra, "w", zipfile.ZIP_DEFLATED) as out:
        for qid, rows in ra.items():
            buf = io.StringIO()
            w = csv.writer(buf, lineterminator="\r\n")
            w.writerows(rows)
            out.writestr(f"submission/{qid}.csv", buf.getvalue().encode("utf-8"))

    z2 = zipfile.ZipFile(a.ra)
    goc = sorted({n.split("/")[0] for n in z2.namelist()})
    xau = [n for n in z2.namelist()
           if len([r for r in csv.reader(io.StringIO(
               z2.read(n).decode("utf-8"))) if r]) != 100]
    print(f"\n  -> {a.ra}  ·  {len(z2.namelist())} file · gốc {goc}")
    print("  ✅ mọi file đủ 100 dòng" if not xau else f"  ❌ {xau}")


if __name__ == "__main__":
    main()
