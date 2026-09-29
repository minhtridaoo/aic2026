"""So hai bài nộp trên cùng một câu — in N dòng đầu của mỗi bên.

    python -X utf8 _so_hai_zip.py out/NOP/dot2_L1may.zip out/NOP/dot2_L2ghim.zip query-p2-23-qa 6
"""
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8")
a, b, cau = sys.argv[1], sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 6

for ten in (a, b):
    z = zipfile.ZipFile(ten)
    duong = f"submission/{cau}.csv"
    dong = z.read(duong).decode("utf-8").splitlines()
    print(f"=== {ten}  ({len(dong)} dòng)")
    for i, d in enumerate(dong[:n], 1):
        canh = "  ⚠ TRƯỜNG RỖNG" if d.rstrip().endswith(",") or ",," in d else ""
        print(f"   {i:3d}. {d}{canh}")
    rong = sum(1 for d in dong if d.rstrip().endswith(",") or ",," in d)
    if rong:
        print(f"   >>> {rong}/{len(dong)} dòng có trường RỖNG")
