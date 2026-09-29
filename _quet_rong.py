"""Quét CẢ bài nộp tìm dòng QA có trường đáp án RỖNG.

Đáp án rỗng = 0 điểm chắc chắn (R-Score QA đòi `answer == GT_a`), mà
`validate_csv` không coi đó là lỗi định dạng nên zip vẫn "hợp lệ".

    python -X utf8 _quet_rong.py out/NOP/dot2_L2ghim.zip
"""
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8")
z = zipfile.ZipFile(sys.argv[1])
tong = 0
for ten in sorted(z.namelist()):
    if not ten.endswith(".csv"):
        continue
    dong = z.read(ten).decode("utf-8").splitlines()
    rong = [(i, d) for i, d in enumerate(dong, 1)
            if d.rstrip().endswith(",") or ",," in d]
    if rong:
        tong += len(rong)
        print(f"⚠ {ten}: {len(rong)}/{len(dong)} dòng RỖNG "
              f"(dòng {[i for i, _ in rong][:8]})")
print(f"\nTỔNG dòng rỗng: {tong}" if tong else "\n✅ không có dòng rỗng nào")
sys.exit(1 if tong else 0)
