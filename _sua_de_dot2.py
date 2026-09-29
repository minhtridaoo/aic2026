"""Sửa tay `data/de_dot2.json` — HAI chỗ máy đọc sai, đã đếm lại bằng mắt.

1) CẢ HAI câu TRAKE bị tính thêm CÂU DẪN thành sự kiện (5 thay vì 4). Đây đúng
   chỗ BTC bác `query-p1-16-trake` đợt 1 ("Expected 4 frame IDs, got 3"). Lần
   này bằng chứng còn nằm ngay trong đề: cả hai câu TỰ GHI SỐ 4 —
     p2-8  : "Đây là chuỗi liên tiếp các cảnh quay về **4 loại trái cây**"  + E1..E4
     p2-21 : "**4 cảnh này** xảy ra liên tiếp nhau."                        + Cảnh 1..4
   Bỏ phần tử đầu. Đồng thời bỏ tiền tố "Cảnh N:" khỏi p2-21 để truy vấn không
   mang rác vào encoder.

2) `query-p2-27-qa` không có dấu '?' nên máy không thấy câu hỏi. Câu hỏi thật là
   mệnh đề cuối; gán thẳng vào `_cau_hoi_vi`.
"""
import json
import sys
from pathlib import Path

P = Path(__file__).with_name("data") / "de_dot2.json"
qs = json.loads(P.read_text(encoding="utf-8"))
L = qs if isinstance(qs, list) else qs["queries"]

CAU_HOI_27 = ("Trong các số từ 1-8 trên các cột, số nào không được nhìn thấy "
              "từ góc nhìn của camera trong 16 giây đầu tiên của video?")

doi = []
for q in L:
    if q.get("task") == "trake" and len(q.get("sub_events") or []) == 5:
        cu = q["sub_events"]

        def bo_tien_to(s: str) -> str:
            dau = s[:10].lower()
            if dau.startswith("cảnh") and ":" in s[:10]:
                return s.split(":", 1)[1].strip()
            return s

        moi = [bo_tien_to(s) for s in cu[1:]]
        q["sub_events"] = moi
        q["n_events"] = 4
        q["text"] = " ".join(moi)
        doi.append(f"{q['qid']}: 5 -> {len(moi)} sự kiện")
    if q["qid"] == "query-p2-27-qa":
        q["_cau_hoi_vi"] = CAU_HOI_27
        q["question_vi"] = CAU_HOI_27
        doi.append("query-p2-27-qa: gán câu hỏi tay")

P.write_text(json.dumps(qs, ensure_ascii=False, indent=1), encoding="utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
for d in doi:
    print("  ✓", d)
for q in L:
    if q.get("task") == "trake":
        print(f"\n{q['qid']}  n_events={len(q['sub_events'])}")
        for i, s in enumerate(q["sub_events"], 1):
            print(f"   {i}. {s}")
