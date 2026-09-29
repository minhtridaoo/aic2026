#!/usr/bin/env python
"""ĐƯỜNG NỘP BÀI THẬT — đọc config, chạy hệ thống thi, đóng zip.

    python nop_bai.py                                  # 3 truy vấn khói, cấu hình thi
    python nop_bai.py --truy-van data/truy_van_thi.json --ra out/nop_bai.zip
    python nop_bai.py --qa-cheap                       # chưa có khoá VLM/ảnh
    python nop_bai.py --config config.yaml --gioi-han-video 30   # chạy nhanh

VÌ SAO TÁCH KHỎI `kiem_dinh_dang.py` (việc 1.5, 2026-08-16)
-----------------------------------------------------------
Script cũ (`nop_bai_thu.py`) tự dựng `Pipeline` nên đã trôi khỏi cấu hình thi:
diễn tập 15/08 xuất ra zip có đáp án QA **tiếng Anh**, chạy **base CLIP** thay vì
SigLIP2 so400m, và **bật portfolio** — trong khi cả ba đã chốt ngược lại. Không
một phép kiểm định dạng nào bắt được, vì zip vẫn hợp lệ.

File này KHÔNG được phép tự dựng `Pipeline`. Nó gọi `aic2026.config.dung_pipeline`
— một điểm nối duy nhất giữa YAML và hệ thống — và in cấu hình ra trước khi chạy,
để nhìn là biết đang nộp bằng gì. Có một test hồi quy khoá đúng điều này
(`tests/test_cau_hinh.py::test_nop_bai_khong_tu_dung_pipeline`).

ĐỊNH DẠNG FILE TRUY VẤN
-----------------------
    {"queries": [
      {"qid": "query-1-kis",   "task": "kis",   "text": "..."},
      {"qid": "query-3-qa",    "task": "qa",    "text": "...",
       "_cau_hoi_vi": "..."},
      {"qid": "query-4-trake", "task": "trake", "sub_events": ["...", "..."]}
    ]}

⚠ `qid` NÊN đúng bằng tên file đề của BTC (`query-3-qa.txt` -> `qid`
`"query-3-qa"`), vì tên file nộp suy từ đó và Codabench ghép file với truy vấn
THEO TÊN. Không đặt được thì thêm khoá `"ten_file": "query-3-qa"`. Đặt kiểu dev
set (`qa-1`) vẫn chạy nhưng sẽ bị cảnh báo to.

Bộ đọc CỐ Ý dễ tính về tên khoá, vì chưa ai thấy file đề thật: nó thử một danh
sách tên quen (`text`, `text_en`, `text_vi`, `query`, `description`, `mo_ta`...)
rồi mới quét các khoá lạ. `sub_events` nhận cả danh sách lẫn một chuỗi ngăn bằng
"→". Nhưng nó KHÔNG dễ tính về ngôn ngữ: bản dùng để truy hồi chọn theo
`truy_van.ngon_ngu` trong config, và in ra đã dùng bản nào cho từng câu. Xem
`aic2026/core/de_bai.py` để biết vì sao hai chỗ đó từng là lỗi câm.

Dev set (`data/dev_set_tu_lam.json`) dùng được luôn — phần `gt` bị bỏ qua. Chưa có
nguồn nào xác nhận layout/tên file BTC yêu cầu; hiện đặt `query-<qid>.csv` trong
thư mục `submission/` như `build_submission_zip` vẫn làm. Kiểm lại khi BTC mở
trang nộp.
"""
import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026 import config as cau_hinh
from aic2026.core import de_bai
from aic2026.core import ghim as ghim_mod
from aic2026.core import qa as qa_mod
from aic2026.core import submission as sub
from aic2026.core import viet_lai_truy_van
from aic2026.core.models import Query, SubEvent
from aic2026.core.qa import con_la_tieng_anh

GOC = os.path.dirname(os.path.abspath(__file__))

# Ba truy vấn KHÓI: chỉ để chạy hết đường ống khi chưa có đề. Câu QA có bản tiếng
# Việt để lớp đáp án tiếng Việt được đi qua thật, đúng chỗ đã hỏng ngày 15/08.
TRUY_VAN_KHOI = [
    Query(qid="kis-1", task="kis",
          text="a man in a suit speaking at a microphone"),
    Query(qid="qa-1", task="qa",
          text="What color is the shirt of the person speaking?",
          question_vi="Người đang phát biểu mặc áo màu gì?"),
    Query(qid="trake-1", task="trake", text="",
          sub_events=[SubEvent(text="a chef pouring oil into a pan"),
                      SubEvent(text="a chef stirring food in a pan"),
                      SubEvent(text="a finished dish on a plate")]),
]


def doc_truy_van(path: str, chinh_sach: str = "en"):
    """Đọc file đề. Trả `(queries, ban_ghi, ten_files)` — hai cái sau để IN RA.

    Chấp cả dev set (bỏ `gt`) để khỏi phải chuyển định dạng.

    Hai lỗi câm mà hàm này chặn nằm ở `aic2026/core/de_bai.py`:

      * bản cũ lấy `q.get("text") or q.get("_mo_ta_vi")` — không có chính sách
        ngôn ngữ, không log. Đề của BTC có cả hai bản; nếu bản tiếng Việt nằm
        dưới đúng khoá tên `text` thì hệ thống truy hồi bằng tiếng Việt, chạy
        trơn tru, và mất ~41% điểm KIS mà không có triệu chứng nào.
      * bản cũ lặp thẳng `q.get("sub_events", [])`. Một CHUỖI "a → b" bị Python
        lặp theo KÝ TỰ thành 5 sub-event, cũng im lặng, và sai `n_events` là
        R-Score 0 cho cả truy vấn TRAKE.

    `ten_files[qid]` là tên file CSV sẽ nộp cho truy vấn đó. Codabench ghép file
    với truy vấn THEO TÊN nên đây là chỗ thứ ba có thể cho 0 điểm mà không có
    triệu chứng — xem `de_bai.ten_file_nop`.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    muc = data["queries"] if isinstance(data, dict) else data
    ra, ban_ghi, ten_files = [], [], {}
    for i, q in enumerate(muc, start=1):
        qid = str(q.get("qid") or q.get("id") or f"q{i}")
        task = str(q.get("task") or q.get("loai") or "").lower()
        ten_files[qid] = de_bai.ten_file_nop(qid, q.get("ten_file") or "")
        if task == "trake":
            se, bg = de_bai.chon_sub_events(q, chinh_sach, qid)
            ra.append(Query(qid=qid, task=task, text=str(q.get("text") or ""),
                            sub_events=[SubEvent(**d) for d in se]))
        else:
            if task not in ("kis", "qa"):
                raise ValueError(
                    f"{qid}: `task` không biết: {task!r} (có: kis | qa | trake)")
            text, bg = de_bai.chon_van_ban(q, chinh_sach, qid)
            ra.append(Query(qid=qid, task=task, text=text,
                            question_vi=de_bai.chon_cau_hoi(q)))
        ban_ghi.append(bg)
    if not ra:
        raise ValueError(f"{path}: không có truy vấn nào")
    return ra, ban_ghi, ten_files


def _args():
    p = argparse.ArgumentParser(description="Đường nộp bài thật, chạy theo config")
    p.add_argument("--config", default=None,
                   help="mặc định: config.yaml, không có thì config.example.yaml")
    p.add_argument("--truy-van", default=None,
                   help="file JSON truy vấn; không có thì dùng 3 truy vấn khói")
    p.add_argument("--ra", default=os.path.join(GOC, "out", "nop_bai.zip"))
    p.add_argument("--qa-cheap", action="store_true",
                   help="ghi đè CÓ Ý THỨC: QA zero-shot, không gọi VLM")
    p.add_argument("--gioi-han-video", type=int, default=None,
                   help="chỉ nạp N video đầu — để thử nhanh, KHÔNG dùng khi nộp")
    p.add_argument("--ghim", default=None,
                   help="file JSON ghim video/đáp án do người vận hành xác "
                        "minh (xem aic2026/core/ghim.py)")
    return p.parse_args()


def main():
    args = _args()
    config = cau_hinh.load_config(args.config)

    print("=" * 74)
    print("  NỘP BÀI — CẤU HÌNH ĐANG DÙNG")
    print("=" * 74)
    for dong in cau_hinh.mo_ta_cau_hinh(config, qa_cheap=args.qa_cheap):
        print("  " + dong)
    for dong in cau_hinh.canh_bao_cau_hinh(config):
        print(f"\n  ⚠ {dong}")
    print()

    # Thiếu khoá API là lỗi CẤU HÌNH: chặn ngay, đừng để nạp 177.321 vector xong
    # rồi mới nổ ở câu QA đầu tiên.
    if not args.qa_cheap:
        thieu = cau_hinh.thieu_khoa_api(config)
        if thieu:
            print("❌ Provider VLM đang bật nhưng thiếu khoá môi trường:")
            for dong in thieu:
                print(f"   - {dong}")
            print("\n   Chọn một:")
            print("   (a) đặt biến môi trường cho tiến trình này (đừng ghi vào file)")
            print("   (b) tắt provider đó trong config (`enabled: false`)")
            print("   (c) chạy lại với `--qa-cheap` — QA zero-shot, ĐÁP ÁN YẾU HƠN")
            sys.exit(1)

    chinh_sach = cau_hinh.ngon_ngu_truy_van(config)
    ban_ghi, ten_files = [], {}
    if args.truy_van:
        truy_van, ban_ghi, ten_files = doc_truy_van(args.truy_van, chinh_sach)
        print(f"ĐỀ: {args.truy_van}  ·  chính sách ngôn ngữ `{chinh_sach}`")
        for bg in ban_ghi:
            print("  " + bg.mo_ta() + f"  -> {ten_files.get(bg.qid, '?')}")
        lech_ten = [c for c in (de_bai.canh_bao_ten_file(t)
                                for t in ten_files.values()) if c]
        if lech_ten:
            print(f"\n  ⚠ {len(lech_ten)}/{len(ten_files)} tên file KHÔNG khớp "
                  f"quy ước BTC:")
            for c in lech_ten[:5]:
                print(f"     {c}")
        lech = [bg for bg in ban_ghi if not bg.theo_chinh_sach]
        if lech:
            print(f"\n  ⚠ {len(lech)}/{len(ban_ghi)} câu KHÔNG theo được chính "
                  f"sách `{chinh_sach}`:")
            for bg in lech[:5]:
                print(f"     {bg.canh_bao}")
            # Đúng những câu này là chỗ mất điểm: đo trên Kaggle 19/08 (so400m
            # + hợp nhất, 120 câu KIS) bản tiếng Việt được 0.2883 còn bản người
            # gõ tay 0.3750. Cho LLM viết lại thành chú thích ảnh tiếng Anh kéo
            # lên 0.3450, tức +0.0567 ± 0.0209 (vượt 2·SE).
            bo, vi_sao = viet_lai_truy_van.tu_config(config)
            if bo is None:
                print(f"\n  ⚠ KHÔNG viết lại truy vấn ({vi_sao}) — đang chạy "
                      f"đường tiếng Việt, thấp hơn ~0.057 điểm.")
            else:
                print(f"\n  Viết lại truy vấn bằng {vi_sao} ...", flush=True)
                for dong in viet_lai_truy_van.ap_dung(bo, truy_van, ban_ghi):
                    print(f"     {dong}")
                for bg, q in zip(ban_ghi, truy_van):
                    if not bg.theo_chinh_sach and not q.sub_events:
                        print(f"     {bg.qid}: {q.text[:78]}")
                        break
        print()
    else:
        truy_van = TRUY_VAN_KHOI
        print("⚠ Chưa có file đề: đang dùng 3 TRUY VẤN KHÓI, không phải đề thi.\n")

    t = time.time()
    print("Đang nạp dữ liệu ...")
    kwargs = ({"gioi_han_video": args.gioi_han_video}
              if args.gioi_han_video else {})
    events, emb, metas, rep = cau_hinh.nap_du_lieu(config, **kwargs)
    print(f"  {rep.events:,d} event / {rep.videos_da_nap} video  "
          f"({time.time() - t:.1f}s)")
    if rep.thieu_file:
        print(f"  ⚠ {len(rep.thieu_file)} video thiếu file, đã bỏ "
              f"(vd {', '.join(rep.thieu_file[:3])})")
    if args.gioi_han_video:
        print("  ⚠ ĐANG GIỚI HẠN VIDEO — zip này KHÔNG dùng để nộp thật.")

    p = cau_hinh.dung_pipeline(config, events, emb, qa_cheap=args.qa_cheap)
    top_k = cau_hinh.so_dong_nop(config)
    print(f"  encoder: {type(p.base_encoder).__name__} "
          f"{getattr(p.base_encoder, 'MODEL_ID', '')} ({p.base_encoder.dim} chiều)")
    for _kho, _w in (getattr(p, "kenh_chu", None) or ()):
        # In SAU index: đây mới là lúc biết kênh chữ có thật sự gắn được event
        # hay không. Kênh gắn 0 mục chạy êm và không đóng góp gì — đúng loại
        # lỗi câm mà file này sinh ra để chặn.
        n = getattr(getattr(_kho, "bao_cao", None), "so_event_gan", 0) or len(
            getattr(_kho, "_row2event", None) or {}) or sum(
            len(v) for v in (getattr(_kho, "_event_theo_khoang", None)
                             or {}).values())
        print(f"  kênh chữ: {type(_kho).__name__} · trọng số {_w} · "
              + (f"{n:,d} event đã gắn" if n else "❌ KHÔNG GẮN ĐƯỢC EVENT"))
    print(f"  QA: {type(p.llm).__name__}"
          + (f" + {type(p.visual_qa).__name__}" if p.visual_qa is not None else "")
          + f" · portfolio {'BẬT' if p.portfolio.enabled else 'tắt'}\n")

    ds_ghim = ghim_mod.doc_ghim(args.ghim) if args.ghim else {}
    if ds_ghim:
        print(f"GHIM: {len(ds_ghim)} truy vấn có tri thức người vận hành")
        for g in ds_ghim.values():
            print("  " + g.mo_ta())
        la = sorted(set(ds_ghim) - {q.qid for q in truy_van})
        if la:
            print(f"\n  ⚠ {len(la)} qid trong file ghim KHÔNG có trong đề: {la}")
            print("     Sai qid là ghim rơi vào hư không — kiểm lại tên.")
        print()

    files, n_events, dap_an_qa, thieu_dong = {}, {}, [], []
    for q in truy_van:
        with warnings.catch_warnings(record=True) as bat:
            warnings.simplefilter("always")
            rows = p.answer(q, top_k=top_k)
        ten = ten_files.get(q.qid) or de_bai.ten_file_nop(q.qid)
        files[ten] = (q.task, rows)
        if q.task == "trake":
            n_events[ten] = q.n_events
        if len(rows) < top_k:
            thieu_dong.append((q.qid, len(rows)))
        print(f"  {q.task.upper():6s} {q.qid:12s} {len(rows):>3d} dòng"
              + (f"  ⚠ thiếu {top_k - len(rows)}" if len(rows) < top_k else ""))
        for w in bat:
            print(f"         ⚠ {w.message}")
        if getattr(p.last_trace, "vlm_error", None):
            print(f"         ⚠ VLM lỗi -> đã hạ xuống QA rẻ: "
                  f"{str(p.last_trace.vlm_error)[:160]}")
        if q.task == "qa":
            phan_biet = []
            for r in rows:
                if r.get("answer") not in phan_biet:
                    phan_biet.append(r["answer"])
            dap_an_qa.extend(phan_biet)
            print(f"         đáp án phân biệt: {phan_biet[:5]}")
            # Bịa hay đọc? Xem `qa.soi_dap_an_vlm`. Chỉ đọc được từ `trace`
            # vì `rows` đã lan đáp án phổ biến ra 90 dòng đuôi (`tail_answer`),
            # nên trên `rows` thì mọi câu đều "một đáp án" — mất hết triệu chứng.
            #
            # CẮT về `vlm_top_n`: `trace.answers` cũng gồm cả phần đuôi (xem
            # `pipeline._answer_qa`, `answered.extend(... for candidate in tail)`),
            # mà đuôi thì dùng CHUNG một đáp án theo thiết kế. Không cắt thì số
            # video trong cảnh báo bị thổi lên (đo thật: báo 35 video trong khi
            # VLM chỉ nhìn 10 khung) — cảnh báo đúng nhưng con số vô nghĩa.
            nghi = qa_mod.soi_dap_an_vlm(
                (getattr(p.last_trace, "answers", None) or [])
                [:p.qa_vlm_top_n or 10])
            if nghi:
                print(f"         ⚠ {nghi}")

    # --- GHIM: chạy SAU khi cả đề đã chạy xong trên chỉ mục đầy đủ ----------- #
    # Thứ tự này là bắt buộc, không phải sở thích: `ap_dung` re-index pipeline
    # xuống còn vài trăm keyframe của video được ghim. Ghim xen giữa vòng lặp
    # trên thì các câu sau sẽ chạy trên chỉ mục đã bị cắt — mất sạch, âm thầm.
    if ds_ghim:
        print("\nĐANG GHIM ...")
        theo_qid = {q.qid: q for q in truy_van}
        for qid, g in ds_ghim.items():
            q = theo_qid.get(qid)
            if q is None:
                continue
            ten = ten_files.get(qid) or de_bai.ten_file_nop(qid)
            task, rows_goc = files[ten]
            try:
                rows, bao = ghim_mod.ap_dung(p, q, g, rows_goc, events, emb,
                                             top_k=top_k)
            except LookupError as exc:
                print(f"  ❌ {qid}: {exc}")
                continue
            files[ten] = (task, rows)
            print(f"  {bao}")
            if q.task == "qa":
                print(f"     đáp án sau ghim: "
                      f"{sorted({r['answer'] for r in rows})[:3]}")
        p.retriever.reset()           # `index` cộng dồn — phải xoá trước
        p.index(events, emb)          # trả lại chỉ mục đầy đủ
        # Nghiệm thu đáp án QA ở cuối phải soi đáp án SAU khi ghim, không phải
        # đáp án đã thu trong vòng lặp trên — ghim có quyền đè đáp án.
        dap_an_qa = [r["answer"] for task, rows in files.values()
                     if task == "qa" for r in rows]
        print()

    ok, loi = sub.build_submission_zip(args.ra, files, n_events=n_events)
    print(f"\n  zip ok={ok} lỗi={loi}")
    print(f"  -> {args.ra}")

    # --- nghiệm thu, đúng ba điều đã trôi ngày 15/08 ------------------------- #
    ngon_ngu = str((config.get("qa") or {}).get("output_language", "vi")).lower()
    chua_dich = [x for x in dap_an_qa if con_la_tieng_anh(x)]
    lech = [bg for bg in ban_ghi if not bg.theo_chinh_sach]
    lech_ten = [c for c in (de_bai.canh_bao_ten_file(t) for t in files) if c]
    print("\n  NGHIỆM THU")
    print(f"    tên file khớp quy ước BTC   : "
          + ("✅" if not lech_ten
             else f"⚠ {len(lech_ten)}/{len(files)} lệch — xem cảnh báo trên"))
    if ban_ghi:
        print(f"    ngôn ngữ theo chính sách    : "
              + (f"✅ {len(ban_ghi)}/{len(ban_ghi)} bản `{chinh_sach}`"
                 if not lech else
                 f"⚠ {len(lech)}/{len(ban_ghi)} câu phải dùng bản khác"))
    print(f"    đủ {top_k} dòng mỗi truy vấn : "
          + ("✅" if not thieu_dong else f"❌ {thieu_dong}"))
    print(f"    portfolio theo config       : "
          + ("✅ tắt" if not p.portfolio.enabled else "BẬT (theo config)"))
    if ngon_ngu == "vi" and dap_an_qa:
        print(f"    đáp án QA tiếng Việt        : "
              + ("✅" if not chua_dich
                 else f"❌ còn tiếng Anh: {sorted(set(chua_dich))[:5]}"))
    if not ok or thieu_dong or (ngon_ngu == "vi" and chua_dich):
        sys.exit(1)
    print("\n  ✅ Zip hợp lệ và khớp cấu hình đã chốt.")


if __name__ == "__main__":
    main()
