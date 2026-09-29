"""Đếm đáp án theo dòng cho một câu QA trong bài nộp."""
import collections
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8")
z = zipfile.ZipFile(sys.argv[1])
cau = sys.argv[2]
dong = z.read(f"submission/{cau}.csv").decode("utf-8").splitlines()
dap = [d.split(",", 2)[2] if d.count(",") >= 2 else "" for d in dong]
print(f"{sys.argv[1]}  {cau}")
print("  20 dòng đầu:", dap[:20])
print("  đếm cả 100 :", dict(collections.Counter(dap)))
