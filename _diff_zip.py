"""So HAI bài nộp theo TỪNG FILE — cổng an toàn cho LẦN NỘP CUỐI.

    python -X utf8 _diff_zip.py out/NOP/dot2_L5.zip out/NOP/dot2_L6.zip query-p2-7-qa,query-p2-27-qa

VÌ SAO CẦN
----------
Luật BTC tính LẦN NỘP CUỐI, nên bài cuối không có cách nào thử lại. Cách duy
nhất để "thêm mà không làm hỏng" là chứng minh bài mới GIỐNG HỆT bài cũ ở mọi
câu ta KHÔNG cố ý sửa. Pipeline này tái lập từng byte giữa hai lần chạy (đo
20/08), nên bất kỳ file đổi ngoài danh sách cố ý = có gì đó đã trôi (config,
cache, thứ tự nạp) và phải điều tra TRƯỚC khi nộp, không phải sau.

Tham số thứ ba là danh sách qid ĐƯỢC PHÉP đổi (ngăn bằng phẩy). Bỏ trống nghĩa
là "không câu nào được đổi".
"""
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8")
cu, moi = sys.argv[1], sys.argv[2]
cho_phep = {x.strip() for x in (sys.argv[3] if len(sys.argv) > 3 else "").split(",")
            if x.strip()}

za, zb = zipfile.ZipFile(cu), zipfile.ZipFile(moi)
ta, tb = set(za.namelist()), set(zb.namelist())
if ta != tb:
    print(f"❌ danh sách file KHÁC nhau: chỉ ở cũ {sorted(ta - tb)} · "
          f"chỉ ở mới {sorted(tb - ta)}")
    sys.exit(1)

doi, y_nguyen, ngoai_y = [], [], []
for ten in sorted(ta):
    if not ten.endswith(".csv"):
        continue
    qid = ten.split("/")[-1][:-4]
    if za.read(ten) == zb.read(ten):
        y_nguyen.append(qid)
        continue
    a = za.read(ten).decode("utf-8").splitlines()
    b = zb.read(ten).decode("utf-8").splitlines()
    khac = next((i for i, (x, y) in enumerate(zip(a, b), 1) if x != y), 0)
    doi.append((qid, khac, a[0] if a else "", b[0] if b else ""))
    if qid not in cho_phep:
        ngoai_y.append(qid)

print(f"{cu}  ->  {moi}")
print(f"  y nguyên : {len(y_nguyen)} câu")
print(f"  đã đổi   : {len(doi)} câu")
for qid, khac, a0, b0 in doi:
    dau = "✅ cố ý" if qid in cho_phep else "❌ NGOÀI Ý ĐỊNH"
    print(f"     {dau}  {qid}  (khác từ dòng {khac})")
    print(f"          cũ: {a0}")
    print(f"          mới: {b0}")
thieu = sorted(cho_phep - {q for q, *_ in doi})
if thieu:
    print(f"  ⚠ khai là sửa mà file KHÔNG đổi: {thieu} — ghim có chạy không?")
print("\n" + ("❌ DỪNG: có file đổi ngoài ý định — điều tra trước khi nộp"
              if ngoai_y else "✅ chỉ đúng các câu cố ý sửa bị đổi"))
sys.exit(1 if ngoai_y else 0)
