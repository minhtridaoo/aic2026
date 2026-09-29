"""Soi cấu trúc zip nộp bài — bắt đúng lỗi 21/08 (Windows bọc thêm một tầng
thư mục nên `submission/` không nằm ở gốc, mắt nhìn zip vẫn 'trông đúng' mà
điểm 0 toàn bộ).

    python -X utf8 _soi_zip.py out/NOP/dot2_L1may.zip
"""
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ten = sys.argv[1]
z = zipfile.ZipFile(ten)
n = z.namelist()
goc = sorted({x.split("/")[0] for x in n})
print(f"zip     : {ten}")
print(f"số file : {len(n)}")
print(f"gốc zip : {goc}")
print(f"ví dụ   : {n[:3]}")
ok = goc == ["submission"]
print("cấu trúc: " + ("✅ `submission/` nằm ở GỐC"
                      if ok else "❌ SAI — gốc phải đúng một thư mục `submission`"))
mot = [x for x in n if x.endswith(".csv")][:1]
if mot:
    dong = z.read(mot[0]).decode("utf-8").splitlines()
    print(f"\n{mot[0]}: {len(dong)} dòng")
    for d in dong[:3]:
        print("   ", d)
sys.exit(0 if ok else 1)
