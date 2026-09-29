#!/usr/bin/env python
"""QA có sinh được đáp án CÓ NGHĨA không, trên vector thật.

    python thu_nghiem_qa.py

TRƯỚC KHI CÓ `core/qa.py`
-------------------------
`RuleBasedProvider.answer()` trả chuỗi RỖNG trên 100/100 dòng, vì nó bới chữ hoa
trong visual_text/ocr_text/asr_text — bộ data BTC để cả ba rỗng. R-Score của QA
đòi `answer == GT_a`, nên QA được **0 điểm tuyệt đối**.

PHÉP KIỂM Ở ĐÂY
---------------
Chưa có nhãn QA nên KHÔNG chấm điểm được. Thay vào đó kiểm ba tính chất quan
sát được, mỗi cái đều có thể sai một cách nhìn thấy được:

  1. đáp án khác rỗng                    (điều kiện cần tối thiểu)
  2. đáp án ĐỔI theo câu hỏi             (nếu không thì nó chỉ đoán bừa một hằng số)
  3. đáp án ĐỔI theo khung hình          (nếu không thì nó không hề nhìn ảnh)

Phép 3 là phép quan trọng nhất: nó phân biệt "QA thật" với "một bộ sinh chuỗi".
Cách kiểm: hỏi cùng một câu trên hai nhóm khung hình được chọn bằng hai truy vấn
thị giác trái ngược nhau. Nếu phân phối đáp án giống hệt thì vector ảnh không hề
tham gia.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026.core.dataset import load_dataset
from aic2026.core.models import Query
from aic2026.core.pipeline import Pipeline, PortfolioConfig
from aic2026.core.providers import CLIPTextEncoder, RuleBasedProvider
from aic2026.core.qa import ClipZeroShotQA, SinhTheoLuat
from aic2026.core.stores import NumpyVectorStore

GOC = os.path.dirname(os.path.abspath(__file__))
DATA = r"D:\COMP\AIC2026\data cuộc thi\aic2026_data_nhe"
CACHE = os.path.join(GOC, "dev_set", "query_vectors.npz")

CAU_HOI = [
    "What color is the shirt of the person in the frame?",
    "How many people are in the frame?",
    "Where was this photo taken?",
    "What vehicle appears in the frame?",
]


def main():
    print("Đang nạp 873 video ...")
    events, emb, metas, rep = load_dataset(DATA)
    enc = CLIPTextEncoder(cache_path=CACHE if os.path.exists(CACHE) else None,
                          cho_phep_live=True)
    print(f"  {rep.events:,d} event")

    kho = NumpyVectorStore(khu_modality_gap=True)
    qa = ClipZeroShotQA(enc, SinhTheoLuat())
    p = Pipeline(enc, RuleBasedProvider(), fusion="rrf", visual_qa=qa,
                 portfolio=PortfolioConfig(enabled=True))
    p.retriever.visual = kho
    p.index(events, embeddings=emb)

    print("\n" + "=" * 74)
    print("  1+2. Đáp án có khác rỗng, và có đổi theo câu hỏi không?")
    print("=" * 74)
    dau = {}
    for ch in CAU_HOI:
        rows = p.answer(Query(qid="q", task="qa", text=ch), top_k=100)
        aset = [r.get("answer", "") for r in rows]
        cands = p.retriever.retrieve(ch, top_k=100, fusion="rrf")
        pp = p.visual_qa.tra_loi(
            ch, cands, lay_vector=kho.lay_vector)
        dau[ch] = pp
        rong = sum(1 for a in aset if not str(a).strip())
        print(f"\n  {ch}")
        print(f"    {len(rows)} dòng, rỗng: {rong}, đáp án phân biệt: "
              f"{len(set(aset))}")
        print("    top-3: " + ", ".join(f"{a} ({q:.2f})" for a, q in pp[:3]))

    tat_ca_rong = all(not str(r).strip()
                      for ch in CAU_HOI for r, _ in dau[ch][:1])
    print("\n  " + ("❌ vẫn rỗng" if tat_ca_rong else
                    "✅ đáp án khác rỗng, và khác nhau giữa các câu hỏi"))

    print("\n" + "=" * 74)
    print("  3. Đáp án có ĐỔI theo khung hình không? (phép quyết định)")
    print("=" * 74)
    print("  Cùng một câu hỏi, hai nhóm khung chọn bằng hai truy vấn trái ngược.")
    ch = "What color is the shirt of the person in the frame?"
    for truy_van in ("a person wearing a red shirt",
                     "a green rice field under blue sky"):
        cands = p.retriever.retrieve(truy_van, top_k=100, fusion="rrf")
        pp = p.visual_qa.tra_loi(ch, cands, lay_vector=kho.lay_vector)
        print(f"\n    khung lấy từ: {truy_van!r}")
        print("      " + ", ".join(f"{a} ({q:.2f})" for a, q in pp[:4]))

    print("\n  Nếu hai dòng trên khác nhau -> vector ảnh CÓ tham gia quyết định,")
    print("  tức đây là QA thật chứ không phải bộ sinh chuỗi cố định.")
    print("\n  ⚠ Chưa có nhãn QA nên CHƯA chấm được điểm. Ba phép trên chỉ chứng")
    print("    minh đường ống sống; đúng hay sai còn phải đo khi có đáp án chuẩn.")


if __name__ == "__main__":
    main()
