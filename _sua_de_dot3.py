"""Sửa tay `data/de_dot3.json` — ba chỗ máy đọc sai, đã đếm lại bằng mắt 04/09.

⚠ ĐỢT 3 DÙNG LẠI MÃ GÓI `p2` GIỐNG HỆT ĐỢT 2 nhưng nội dung khác và loại câu
  khác (số 12 đợt 2 là qa, đợt 3 là kis; số 8 đợt 2 là trake, đợt 3 là qa...).
  Mọi file ghim/đề của đợt 2 KHÔNG được dùng lại — kiểm bằng `_kiem_de_dot3.py`.

1) CẢ HAI câu TRAKE lại bị tính CÂU DẪN thành sự kiện (5 thay vì 4) — lần thứ
   ba liên tiếp. Đếm bằng mắt: cả hai đều có đúng E1..E4.
     p2-21 : câu dẫn "Một món ăn làm từ tôm và các loại gia vị."   (tả MÓN, không phải khoảnh khắc)
     p2-34 : câu dẫn "Con lân màu đỏ đang biểu diễn trên các cột trụ." (tả CẢNH, sau đó còn một dòng trống)
   Cả hai câu dẫn đều KHÔNG bắt đầu bằng "Khoảnh khắc"/"Người ... bắt đầu" như
   các sự kiện thật trong chính câu đó -> chắc chắn là câu dẫn.

2) `query-p2-14-qa` không có dấu '?' (câu hỏi kết bằng dấu ngoặc đơn) nên máy
   không thấy câu hỏi -> tầng đáp án sẽ đoán mò. Gán thẳng `_cau_hoi_vi`.
"""
import json
import sys
from pathlib import Path

P = Path(__file__).with_name("data") / "de_dot3.json"
qs = json.loads(P.read_text(encoding="utf-8"))
L = qs if isinstance(qs, list) else qs["queries"]

CAU_HOI_14 = "Bé gái này sắp lên học lớp mấy?"

doi = []
for q in L:
    if q.get("task") == "trake" and len(q.get("sub_events") or []) == 5:
        cu = q["sub_events"]
        moi = list(cu[1:])
        q["_cau_dan"] = cu[0]          # giữ lại để tra, KHÔNG dùng làm sự kiện
        q["sub_events"] = moi
        q["n_events"] = 4
        q["text"] = " ".join(moi)
        doi.append(f"{q['qid']}: 5 -> {len(moi)} sự kiện (bỏ câu dẫn)")
    if q["qid"] == "query-p2-14-qa":
        q["_cau_hoi_vi"] = CAU_HOI_14
        q["question_vi"] = CAU_HOI_14
        doi.append("query-p2-14-qa: gán câu hỏi tay")

P.write_text(json.dumps(qs, ensure_ascii=False, indent=1), encoding="utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
for d in doi:
    print("  ✓", d)
for q in L:
    if q.get("task") == "trake":
        print(f"\n{q['qid']}  n_events={len(q['sub_events'])}   [câu dẫn bỏ ra: {q.get('_cau_dan','')}]")
        for i, s in enumerate(q["sub_events"], 1):
            print(f"   E{i}. {s}")
for q in L:
    if q["qid"] == "query-p2-14-qa":
        print(f"\nquery-p2-14-qa question_vi = {q['question_vi']}")
