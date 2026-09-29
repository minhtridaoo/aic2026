#!/usr/bin/env python
"""Diễn tập ĐƯỜNG ẢNH ngày thi: corpus keyframe -> VLM -> đáp án (việc 2.2).

    # trên máy này (corpus một phần)
    python -X utf8 dien_tap_qa_anh.py --goc-anh "D:\\COMP\\AIC2026\\data cuộc thi\\keyframes(1 phần nhỏ)"

    # trên Kaggle (corpus đầy đủ) — đặt qa.image_roots trong config rồi chỉ cần
    python -X utf8 dien_tap_qa_anh.py --so 3

    python -X utf8 dien_tap_qa_anh.py --khong-goi-vlm    # chỉ kiểm resolve, 0 API

VÌ SAO CẦN
----------
`kiem_resolver_anh.py` kiểm resolver **đứng một mình**. Việc còn thiếu (2.2) là
đường THẬT lúc chạy: `Pipeline` tự resolve ảnh cho từng ứng viên trong top-N của
nó, gửi cho VLM, rồi ghép đáp án vào 100 dòng nộp bài. Hai thứ đó hỏng theo hai
kiểu khác nhau — resolver đúng mà `qa.image_roots` chưa gắn thì vẫn mất điểm QA.

Script này chạy **y hệt nhau** ở máy này và trên Kaggle. Nó chỉ nạp những video
CÓ ẢNH dưới `image_roots`, nên trên máy chỉ có một phần corpus vẫn diễn tập được
đúng đường ống; trên Kaggle thì phần giao đó là toàn bộ 873 video.

Nghiệm thu (in ở cuối):
  1. **0** `ImageResolutionError` trong top-N của mọi câu
  2. đáp án đến từ VLM thật, không phải đường lùi zero-shot
  3. số request thật + độ trễ mỗi câu — để tính quota ngày thi
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026 import config as cau_hinh
from aic2026.control.query_compiler import compile_query
from aic2026.core.models import Query
from aic2026.core.providers import RuleBasedProvider
from aic2026.data_plane.image_resolver import ImageResolutionError

GOC = Path(__file__).resolve().parent
DEV_JSON = GOC / "data" / "dev_set_tu_lam.json"

# Câu hỏi mặc định: tiếng Việt, loại `color` (đúng loại mà router gửi cho VLM).
CAU_HOI_MAC_DINH = [
    ("dt-1", "Người đang phát biểu mặc áo màu gì?"),
    ("dt-2", "Chiếc xe trong cảnh có màu gì?"),
    ("dt-3", "Người dẫn chương trình mặc áo màu gì?"),
]


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--goc-anh", action="append", default=[],
                   help="ghi đè qa.image_roots (lặp được)")
    p.add_argument("--so", type=int, default=2,
                   help="số câu diễn tập (mỗi câu ~1 request nếu batch=10)")
    p.add_argument("--cau-hoi", action="append", default=[],
                   help="câu hỏi tiếng Việt tự đặt (lặp được)")
    p.add_argument("--truy-van", default=None,
                   help="file dev set: lấy các câu QA trong đó thay cho câu mẫu")
    p.add_argument("--khong-goi-vlm", action="store_true",
                   help="chỉ kiểm resolve ảnh, không gọi API")
    return p.parse_args()


def _cau_hoi(args):
    if args.cau_hoi:
        return [(f"dt-{i}", c) for i, c in enumerate(args.cau_hoi, start=1)]
    if args.truy_van or DEV_JSON.exists():
        path = Path(args.truy_van) if args.truy_van else DEV_JSON
        data = json.loads(path.read_text(encoding="utf-8"))
        ra = [(q["qid"], q.get("_cau_hoi_vi") or q.get("text", ""))
              for q in data.get("queries", []) if q.get("task") == "qa"]
        if ra:
            return ra
    return CAU_HOI_MAC_DINH


def main():
    args = _args()
    config = cau_hinh.load_config(args.config)
    if args.goc_anh:
        config.setdefault("qa", {})["image_roots"] = [str(Path(x)) for x in args.goc_anh]
    roots = [Path(x) for x in (config.get("qa") or {}).get("image_roots") or []]
    roots = [r for r in roots if r.is_dir()]
    if not roots:
        print("❌ `qa.image_roots` chưa trỏ tới thư mục ảnh nào tồn tại.")
        print("   Trên Kaggle: mount dataset keyframe rồi đặt khoá đó trong config.")
        print("   Ở máy này: truyền --goc-anh <thư mục keyframe>.")
        sys.exit(1)

    co_anh = sorted({d.name.upper() for r in roots for d in r.iterdir() if d.is_dir()})
    print(f"Thư mục ảnh: {', '.join(str(r) for r in roots)}")
    print(f"  {len(co_anh)} video có ảnh (vd {', '.join(co_anh[:3])})")
    for dong in cau_hinh.canh_bao_cau_hinh(config):
        print(f"  ⚠ {dong}")

    if not args.khong_goi_vlm:
        thieu = cau_hinh.thieu_khoa_api(config)
        if thieu:
            print("❌ Provider VLM đang bật nhưng thiếu khoá:")
            for dong in thieu:
                print(f"   - {dong}")
            print("   Đặt biến môi trường, hoặc chạy --khong-goi-vlm.")
            sys.exit(1)

    t = time.time()
    events, emb, _, rep = cau_hinh.nap_du_lieu(config, video_ids=co_anh)
    print(f"  nạp {rep.events:,d} event / {rep.videos_da_nap} video "
          f"({time.time() - t:.1f}s)")
    if not events:
        print("❌ Không nạp được event nào cho các video có ảnh.")
        sys.exit(1)

    p = cau_hinh.dung_pipeline(
        config, events, emb,
        **({"llm": RuleBasedProvider()} if args.khong_goi_vlm else {}))
    top_n = p.qa_vlm_top_n or 10
    print(f"  QA: {type(p.llm).__name__} · top-N gửi VLM: {top_n} · "
          f"resolver: {'có' if p.image_resolver is not None else 'KHÔNG'}\n")
    # Chế độ `--khong-goi-vlm` dùng provider không nhận ảnh nên Pipeline không dựng
    # resolver — vẫn dựng riêng một cái để kiểm đúng phần resolve.
    resolver = p.image_resolver or cau_hinh.dung_image_resolver(config)
    if p.image_resolver is None and not args.khong_goi_vlm:
        print("❌ Pipeline không dựng resolver -> provider không nhận ảnh. "
              "Kiểm `llm.provider` trong config.")
        sys.exit(1)

    cau = _cau_hoi(args)[:max(1, args.so)]
    tong_thieu_anh, tong_request, nguon = 0, 0, []
    for qid, text in cau:
        # 1. Đúng tập ứng viên mà đường QA sẽ gửi cho VLM.
        compiled = compile_query("qa", text)
        cands = p.retrieval_service.retrieve_compiled(
            compiled, top_k=cau_hinh.so_dong_nop(config))[:top_n]
        thieu_anh = []
        for c in cands:
            try:
                resolver.resolve(c.event.event_id)
            except (ImageResolutionError, ValueError) as exc:
                thieu_anh.append((c.event.event_id, str(exc).splitlines()[0]))
        tong_thieu_anh += len(thieu_anh)

        # 2. Đường thật: Pipeline tự resolve + gọi VLM + ghép 100 dòng.
        truoc = getattr(p.llm, "backend_requests", 0)
        t = time.time()
        with warnings.catch_warnings(record=True) as bat:
            warnings.simplefilter("always")
            rows = p.answer(Query(qid=qid, task="qa", text=text,
                                  question_vi=text),
                            top_k=cau_hinh.so_dong_nop(config))
        giay = time.time() - t
        request = getattr(p.llm, "backend_requests", 0) - truoc
        tong_request += request
        loi_vlm = getattr(p.last_trace, "vlm_error", None)
        dau = []
        for r in rows[:top_n]:
            if r.get("answer") not in dau:
                dau.append(r["answer"])
        # AI NÓI: provider nào thật sự trả lời. Config ghi Gemini là chính, nhưng
        # nếu Gemini lỗi thì failover trả lời bằng đường dự phòng YẾU HƠN mà không
        # có gì hiện ra ở đáp án — phải in ra mới thấy.
        ai_tra = getattr(p.llm, "last_provider", None) or type(p.llm).__name__
        nguon.append("zero_shot" if loi_vlm else ai_tra)
        print(f"  {qid:8s} {text[:44]:44s} {len(rows):>3d} dòng · "
              f"{request} request · {giay:.1f}s · trả lời bởi **{ai_tra}**")
        for that_bai in list(getattr(p.llm, "failures", []))[-3:]:
            print(f"           ⚠ provider lỗi: {that_bai[:200]}")
        print(f"           resolve top-{top_n}: "
              f"{len(cands) - len(thieu_anh)}/{len(cands)} có ảnh"
              + (f"  ❌ thiếu {len(thieu_anh)}" if thieu_anh else ""))
        for eid, msg in thieu_anh[:3]:
            print(f"             {eid}: {msg[:90]}")
        print(f"           đáp án top-{top_n}: {dau[:6]}")
        duoi = {r.get("answer") for r in rows[top_n:]}
        print(f"           đáp án đuôi ({len(rows) - top_n} dòng, chính sách "
              f"`{p.qa_tail_answer}`): {sorted(duoi)[:4]}")
        for w in bat:
            print(f"           ⚠ {str(w.message)[:160]}")
        if loi_vlm:
            print(f"           ⚠ đã lùi về zero-shot: {str(loi_vlm)[:160]}")

    print("\n  NGHIỆM THU 2.2")
    print(f"    ImageResolutionError trong top-{top_n}: "
          + ("✅ 0" if tong_thieu_anh == 0 else f"❌ {tong_thieu_anh}"))
    chinh = None
    if not args.khong_goi_vlm:
        muc = cau_hinh.provider_vlm_dang_bat(config)
        # Tên lớp của provider CHÍNH theo config, để so với ai thật sự trả lời.
        chinh = {"gemini_vlm": "GeminiVLMProvider",
                 "cloudflare_vlm": "CloudflareVLMProvider",
                 "anthropic_vlm": "AnthropicVLMProvider"}.get(
                     muc[0].get("provider") if muc else "", "")
        tu_vlm = sum(1 for x in nguon if x != "zero_shot")
        tu_chinh = nguon.count(chinh)
        print(f"    đáp án đến từ VLM thật : "
              + ("✅ " if tu_vlm == len(nguon) else "❌ ")
              + f"{tu_vlm}/{len(nguon)} câu")
        print(f"    đúng provider CHÍNH    : "
              + ("✅ " if tu_chinh == len(nguon) else "❌ ")
              + f"{tu_chinh}/{len(nguon)} câu bằng {chinh or '?'}"
              + ("" if tu_chinh == len(nguon)
                 else f"  (thật: {sorted(set(nguon))})"))
        print(f"    request thật           : {tong_request} cho {len(cau)} câu "
              f"(~{tong_request / max(1, len(cau)):.1f}/câu)")
        if getattr(p.llm, "cache_hits", 0):
            print(f"    cache hit              : {p.llm.cache_hits}")
    hong = tong_thieu_anh or (not args.khong_goi_vlm
                              and nguon.count(chinh) != len(nguon))
    if hong:
        print("\n  ❌ Đường ảnh CHƯA sẵn sàng — xem lỗi ở trên.")
        sys.exit(1)
    print("\n  ✅ Đường ảnh chạy thật: corpus -> resolver -> VLM -> 100 dòng.")


if __name__ == "__main__":
    main()
