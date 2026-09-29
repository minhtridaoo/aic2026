#!/usr/bin/env python
"""PHÉP THỬ SỐNG-CÒN — model VLM có THẬT SỰ nhìn ảnh không?

    python -X utf8 thu_vlm_cham_anh.py --model gemini-3.5-flash-lite --lo 10
    python -X utf8 thu_vlm_cham_anh.py --model gemini-3.7-flash --so-lo 4

VÌ SAO PHẢI THỬ TRƯỚC KHI XÂY
-----------------------------
Cả hướng "để VLM làm việc của mắt người" dựa trên một giả định: model nhìn được
ảnh. Ngày 20/08 giả định đó đã SAI một lần, và sai theo kiểu không nhìn ra được:
`gemini-3.6-flash` trả cùng một đáp án cho keyframe của 6 video khác nhau, tức
nó đọc chữ trong câu hỏi rồi bịa, không hề mở ảnh. Zip vẫn hợp lệ, đáp án vẫn
đọc như thật.

Nên trước khi viết reranker, hỏi thẳng: cho model 10 ảnh và mô tả của ĐÚNG MỘT
trong 10 ảnh đó, nó có chấm ảnh đúng cao nhất không?

Bộ thử có sẵn và không tốn gì: `dev_set/anh` có 120 keyframe thật, và
`data/dev_set_tu_lam.json` có mô tả người gõ cho từng ảnh. Đáp án biết trước,
nên đây là phép đo chứ không phải cảm nhận.

ĐỌC KẾT QUẢ
-----------
`top1` = tỷ lệ ảnh đúng được chấm cao nhất trong lô.

    ~10% (= 1/lô)   model đoán mò hoặc không nhìn -> HƯỚNG NÀY SẬP
    50-70%          nhìn được, dùng làm tín hiệu hợp nhất thì có lãi
    >80%            nhìn tốt, đáng cho nó quyền xếp lại top đầu

Kèm hai phép kiểm chống-bịa, vì `top1` cao vẫn có thể là ăn may:

  * `hoa`     — số lô mà MỌI ảnh cùng một điểm. Model không nhìn thì hay hoà.
  * `vi_tri`  — phân bố VỊ TRÍ được chấm cao nhất. Ảnh đúng đã được rải ngẫu
                nhiên trong lô, nên nếu model cứ chọn vị trí 1 (hoặc vị trí
                cuối) thì đó là thiên lệch vị trí, không phải nhìn ảnh.
"""
import argparse
import collections
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026.core.providers import GeminiVLMProvider

GOC = Path(os.path.dirname(os.path.abspath(__file__)))


def _args():
    p = argparse.ArgumentParser(description="VLM có nhìn ảnh thật không?")
    p.add_argument("--model", default="gemini-3.5-flash-lite")
    p.add_argument("--lo", type=int, default=10, help="số ảnh mỗi lô")
    p.add_argument("--so-lo", type=int, default=12, help="số lô đem thử")
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--nhip", type=float, default=5.0,
                   help="giây giữa hai request (lite RPM 15 -> 5s là an toàn)")
    p.add_argument("--tieng-viet", action="store_true",
                   help="dùng mô tả tiếng Việt thay vì bản tiếng Anh người gõ")
    p.add_argument("--ra", default=str(GOC / "out" / "thu_vlm_cham_anh.json"))
    return p.parse_args()


def nap_dev():
    """[(mô tả EN, mô tả VI, đường dẫn ảnh)] cho các câu KIS có ảnh."""
    with open(GOC / "data" / "dev_set_tu_lam.json", encoding="utf-8") as fh:
        qs = json.load(fh)["queries"]
    thu_muc = GOC / "dev_set" / "anh"
    ra = []
    for q in qs:
        if q.get("task") != "kis":
            continue
        ten = (q.get("_ghi_chu") or {}).get("anh")
        if not ten:
            continue
        p = thu_muc / ten
        if p.is_file():
            ra.append((q["text"], q.get("_mo_ta_vi", ""), p))
    return ra


def main():
    a = _args()
    mau = nap_dev()
    print(f"{len(mau)} ảnh dev có mô tả + đáp án\n")
    if len(mau) < a.lo:
        raise SystemExit(f"cần ít nhất {a.lo} ảnh, chỉ có {len(mau)}")

    rng = random.Random(a.seed)
    bo = GeminiVLMProvider(
        cache_path=str(GOC / "out" / f"_thu_cham_anh_{a.model}.json"),
        model=a.model, max_tokens=512, timeout_seconds=90, max_retries=1,
        min_interval_seconds=a.nhip)

    print(f"model {a.model} · lô {a.lo} ảnh · {a.so_lo} lô "
          f"· nhịp {a.nhip:g}s\n")
    dung, hoa, vi_tri, ket, t0 = 0, 0, collections.Counter(), [], time.time()

    for lan in range(a.so_lo):
        cum = rng.sample(mau, a.lo)
        dich = rng.randrange(a.lo)                    # ảnh đúng nằm ở đâu
        mo_ta = cum[dich][1 if a.tieng_viet else 0]
        anh = [x[2] for x in cum]
        try:
            diem = bo.cham_anh(mo_ta, anh)
        except Exception as exc:                       # noqa: BLE001
            print(f"  lô {lan + 1:2d}  ❌ {str(exc)[:150]}")
            ket.append({"lo": lan + 1, "loi": str(exc)[:300]})
            continue

        cao = max(range(a.lo), key=lambda i: diem[i])
        ok = cao == dich
        dung += ok
        bang = len(set(diem)) == 1
        hoa += bang
        vi_tri[cao] += 1
        ket.append({"lo": lan + 1, "dich": dich, "chon": cao, "dung": ok,
                    "diem": diem, "mo_ta": mo_ta[:90],
                    "anh": [p.name for p in anh]})
        print(f"  lô {lan + 1:2d}  đúng ở #{dich + 1:2d} · model chọn #{cao + 1:2d} "
              f"{'✅' if ok else '❌'}  điểm ảnh đúng={diem[dich]:5.1f} "
              f"max={diem[cao]:5.1f}" + ("  ⚠ CẢ LÔ HOÀ ĐIỂM" if bang else ""))

    n = sum(1 for r in ket if "loi" not in r)
    if not n:
        raise SystemExit("\n❌ Không lô nào chạy được — xem lỗi phía trên.")

    print(f"\n{'=' * 70}")
    print(f"  model      : {a.model}")
    print(f"  ngôn ngữ   : {'tiếng Việt' if a.tieng_viet else 'tiếng Anh'}")
    print(f"  top1       : {dung}/{n} = {dung / n:.1%}   "
          f"(đoán mò = {1 / a.lo:.1%})")
    print(f"  lô hoà điểm: {hoa}/{n}" + ("  ⚠ dấu hiệu KHÔNG NHÌN ẢNH" if hoa else ""))
    print(f"  vị trí chọn: {dict(sorted(vi_tri.items()))}")
    lech = max(vi_tri.values()) / n if vi_tri else 0
    if lech > 0.5:
        print(f"  ⚠ {lech:.0%} lô cùng chọn một VỊ TRÍ — nghi thiên lệch vị trí, "
              f"không phải nhìn ảnh")
    print(f"  request    : {bo.backend_requests} · {time.time() - t0:.0f}s "
          f"({(time.time() - t0) / max(n, 1):.1f}s/lô)")

    ket_luan = ("HƯỚNG NÀY SẬP — model không phân biệt được ảnh"
                if dung / n < 2.0 / a.lo else
                "dùng được, nhưng chỉ nên HỢP NHẤT với hạng gốc"
                if dung / n < 0.6 else
                "nhìn tốt — đáng cho quyền xếp lại top đầu")
    print(f"  => {ket_luan}")

    Path(a.ra).parent.mkdir(parents=True, exist_ok=True)
    with open(a.ra, "w", encoding="utf-8") as fh:
        json.dump({"model": a.model, "lo": a.lo, "top1": dung / n,
                   "hoa": hoa, "vi_tri": dict(vi_tri), "ket_qua": ket},
                  fh, ensure_ascii=False, indent=1)
    print(f"  -> {a.ra}")


if __name__ == "__main__":
    main()
