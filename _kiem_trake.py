"""Đếm SỐ CỘT thật của mọi file TRAKE trong zip nộp bài.

Đợt 1 (21/08) BTC bác trọn `query-p1-16-trake` với log
`Row 1..100: Expected 4 frame IDs, got 3`. `submission.validate_csv` KHÔNG bắt
được vì nó so CSV với chính bản đọc sai của mình — phép kiểm tự nhất quán không
thay được việc đếm bằng mắt. Cái này đếm cột THẬT trong zip.

    python -X utf8 _kiem_trake.py out/NOP/dot2_L5.zip 4
"""
import collections
import sys
import zipfile

sys.stdout.reconfigure(encoding="utf-8")
z = zipfile.ZipFile(sys.argv[1])
mong = int(sys.argv[2]) if len(sys.argv) > 2 else None
hong = False
for ten in sorted(z.namelist()):
    if "trake" not in ten:
        continue
    dong = z.read(ten).decode("utf-8").splitlines()
    dem = collections.Counter(d.count(",") for d in dong)   # cột = phẩy + 1
    print(f"{ten}: {len(dong)} dòng")
    for phay, n in sorted(dem.items()):
        khung = phay                      # cột đầu là video, còn lại là frame
        dau = "  ← " + ("✅ đúng" if mong is None or khung == mong
                        else f"❌ MONG {mong}")
        print(f"   {n:3d} dòng có {khung} frame ID{dau}")
        if mong is not None and khung != mong:
            hong = True
    print(f"   dòng đầu: {dong[0]}")
print("\n" + ("❌ CÓ FILE SAI SỐ FRAME — BTC sẽ bác cả câu"
              if hong else "✅ mọi dòng TRAKE đúng số frame ID"))
sys.exit(1 if hong else 0)
