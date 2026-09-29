#!/usr/bin/env python
"""Vì sao câu trượt: **SAI VIDEO** hay **ĐÚNG VIDEO NHƯNG SAI KHUNG**? (việc 4.1)

    python -X utf8 chan_doan_qa_truot.py              # 30 câu QA
    python -X utf8 chan_doan_qa_truot.py --task all   # cả kis/qa/trake-free
    python -X utf8 chan_doan_qa_truot.py --top-k 100 --ra out/chan_doan.csv

CÂU HỎI NÀY QUYẾT ĐỊNH LÀM 4.2 HAY 4.3
--------------------------------------
Tầng tìm đang là nút cổ chai chung của cả ba dạng. Nhưng hai cách sửa nhắm vào
hai bệnh khác nhau:

    nghiêng "SAI VIDEO"            -> 4.3 gộp bằng chứng mức video
    nghiêng "ĐÚNG VIDEO SAI KHUNG" -> 4.2 khử trùng vector + rerank trong video

Đo trước khi sửa, vì hai việc đó tốn ~2 giờ mỗi việc và 4.3 rủi ro cao (28 cấu
hình xếp slot đã thua).

KHÔNG TỐN API, KHÔNG CẦN ẢNH
----------------------------
Chỉ chấm tầng ĐỊNH VỊ: script lấy đúng danh sách ứng viên mà đường QA dùng
(`compile_query` -> `retrieve_compiled`), nên không gọi VLM, không cần dịch, và
không chạm tới chất lượng câu trả lời. Vì `R@k = max` trên tiền tố, chỉ cần biết
HẠNG đầu tiên chạm đúng là đủ nói về điểm.

Ba nhóm kết quả (theo top-K, mặc định 100 = đúng số dòng nộp bài):

    A  CHẠM KHUNG   có ít nhất một dòng nằm trong [frame_start, frame_end] -> có điểm
    B  ĐÚNG VIDEO   video đúng có trong top-K nhưng không dòng nào vào cửa sổ
    C  SAI VIDEO    video đúng không xuất hiện lần nào trong top-K
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026 import config as cau_hinh
from aic2026.control.query_compiler import compile_query
from aic2026.core.offline_eval import load_queries
from aic2026.core.providers import RuleBasedProvider

GOC = os.path.dirname(os.path.abspath(__file__))
DEV_JSON = os.path.join(GOC, "data", "dev_set_tu_lam.json")

# Số video được cấp ngân sách slot trong phép mô phỏng. 5/10/20 để thấy đánh đổi
# "phủ rộng nhiều video" vs "để dành slot cho video hạng đầu".
SO_VIDEO_THU = (5, 10, 20)


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--task", default="qa", choices=("qa", "kis", "all"))
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--ra", default=os.path.join(GOC, "out", "chan_doan_truot.csv"))
    p.add_argument("--khong-do-trung", action="store_true",
                   help="bỏ phép đếm vector trùng khít (tiết kiệm ~30s)")
    p.add_argument("--khong-mo-phong", action="store_true",
                   help="bỏ mô phỏng ngân sách slot theo video")
    p.add_argument("--khong-do-hang", action="store_true",
                   help="bỏ phép đo hạng của keyframe đúng trên toàn kho")
    return p.parse_args()


def nhom_vector_trung(emb):
    """{chỉ số hàng -> cỡ nhóm} cho các vector TRÙNG KHÍT (byte y hệt).

    Có 904 nhóm / 6.494 keyframe trên corpus này (audit). Rơi trúng một nhóm là
    hàng trăm video hoà điểm tuyệt đối, thứ tự do numpy quyết -> đó là thứ 4.2
    định sửa. Đếm ở đây để biết bệnh này có thật sự nằm trong các câu trượt hay
    không, thay vì suy diễn từ con số toàn corpus.
    """
    theo_bam = {}
    for i in range(emb.shape[0]):
        khoa = hashlib.sha1(emb[i].tobytes()).digest()
        theo_bam.setdefault(khoa, []).append(i)
    return {i: len(ds) for ds in theo_bam.values() if len(ds) > 1 for i in ds}


def mo_phong_ngan_sach_video(p, compiled, cands, gt, top_k, so_video):
    """Nở timeline **và** chia lại slot theo NGÂN SÁCH VIDEO (hướng của 4.3).

    ⚠ CÁI BẪY ĐÃ ĐO 16/08: chỉ "nở timeline của các video đã lọt top-K rồi chấm
    lại, lấy top-K" là **no-op về toán học** — mọi khung thêm vào đều có điểm
    THẤP HƠN khung hạng 100 toàn kho (nếu cao hơn thì nó đã nằm trong top-100
    rồi), nên top-K không đổi một dòng nào. Đo ra đúng 12/30 → 12/30, hạng trung
    vị 13 → 13. Nở ứng viên chỉ có nghĩa khi ĐI KÈM một cách tiêu slot khác.

    Nên phép mô phỏng này làm đúng thứ 4.3 đề xuất:

      1. xếp hạng VIDEO bằng bằng chứng gộp (RRF trên hạng của ứng viên)
      2. lấy `so_video` video đầu, chấm lại TOÀN BỘ timeline của từng video
      3. tiêu slot **round-robin** theo hạng video: mỗi video một dòng tốt nhất
         trước, rồi vòng hai... — vì `R@k = max` trên tiền tố, slot 1 đáng 5×
         slot 51, nên phải phủ nhiều video ở phần đầu

    Trả hạng dòng đầu tiên chạm cửa sổ đúng, hoặc None. Đây là TRẦN TRÊN của
    hướng đó chứ không phải bản cài đặt: nó có thể ĐẨY RA các câu đang có điểm,
    nên phải đọc kèm cột "A giữ được".
    """
    text = compiled.retrieval_query or compiled.visual_queries[0]
    qv = p.retriever.encoder.encode_text(text)
    diem_video = {}
    for hang, c in enumerate(cands, start=1):
        diem_video[c.video_id] = diem_video.get(c.video_id, 0.0) + 1.0 / (60 + hang)
    thu_tu = [v for v, _ in sorted(diem_video.items(),
                                   key=lambda kv: (-kv[1], kv[0]))][:so_video]

    theo_video = []
    for v in thu_tu:
        su_kien = {e.event_id: e for e in p.retrieval_service.full_timeline(v)}
        diem = p.retriever.visual.score_many(list(su_kien), qv)
        xep = sorted(diem.items(), key=lambda kv: -kv[1])
        theo_video.append([su_kien[eid] for eid, _ in xep])

    hang = 0
    for vong in range(max((len(x) for x in theo_video), default=0)):
        for ds in theo_video:
            if vong >= len(ds):
                continue
            hang += 1
            if hang > top_k:
                return None
            e = ds[vong]
            if (e.video_id == gt.video_id
                    and gt.frame_start <= e.keyframe_idx <= gt.frame_end):
                return hang
    return None


def hang_toan_kho(p, compiled, gt, su_kien_dung):
    """Hạng của keyframe ĐÚNG trong TOÀN BỘ 177.321 event, đúng hình học của search.

    Khác `so_sanh_clip_siglip2.hang_keyframe_dung` ở chỗ đó: bản kia dùng cosine
    THÔ (`emb @ v`), còn `search` có khử modality gap — nên hai con số không so
    trực tiếp được với nhau. Đây là số quyết định tiên lượng: hạng 150 thì rerank
    còn cứu được, hạng 20.000 thì phải đổi encoder/truy vấn.
    """
    if not su_kien_dung:
        return None
    text = compiled.retrieval_query or compiled.visual_queries[0]
    qv = p.retriever.encoder.encode_text(text)
    xep = p.retriever.visual.search(qv, top_k=len(p.retriever.events))
    for hang, (eid, _) in enumerate(xep, start=1):
        if eid in su_kien_dung:
            return hang
    return None


def main():
    args = _args()
    if not os.path.exists(DEV_JSON):
        print(f"❌ Chưa có dev set: {DEV_JSON}")
        sys.exit(1)
    config = cau_hinh.load_config(args.config)
    print("  encoder:", (config.get("encoder") or {}).get("model_id"))

    t = time.time()
    events, emb, metas, rep = cau_hinh.nap_du_lieu(config)
    print(f"  {rep.events:,d} event / {rep.videos_da_nap} video "
          f"({time.time() - t:.1f}s)")
    chi_so_cua = {e.event_id: i for i, e in enumerate(events)}

    co_trung = {}
    if not args.khong_do_trung:
        t = time.time()
        co_trung = nhom_vector_trung(emb)
        print(f"  vector trùng khít: {len(co_trung):,d} keyframe "
              f"({len(co_trung) / max(1, emb.shape[0]):.1%}) "
              f"({time.time() - t:.1f}s)")

    p = cau_hinh.dung_pipeline(config, events, emb, llm=RuleBasedProvider())
    theo_video_toan_bo = {}
    for e in events:
        theo_video_toan_bo.setdefault(e.video_id, []).append(e)
    queries = [(q, gt) for q, gt in load_queries(DEV_JSON)
               if args.task == "all" or q.task == args.task]
    queries = [(q, gt) for q, gt in queries if q.task in ("kis", "qa")]
    print(f"  dev set: {len(queries)} câu {args.task}\n")

    chi_tiet = []
    for q, gt in queries:
        compiled = compile_query(q.task, q.text)
        cands = p.retrieval_service.retrieve_compiled(compiled,
                                                      top_k=args.top_k)
        hang_video = hang_khung = None
        cung_video = []
        n_trung = 0
        for hang, c in enumerate(cands, start=1):
            if co_trung.get(chi_so_cua.get(c.event.event_id, -1)):
                n_trung += 1
            if c.video_id != gt.video_id:
                continue
            hang_video = hang_video or hang
            cung_video.append((hang, c))
            if gt.frame_start <= c.frame_idx <= gt.frame_end:
                hang_khung = hang_khung or hang

        # Lệch bao nhiêu khung so với cửa sổ đúng, tính trên dòng gần nhất của
        # ĐÚNG video — nói được "sai khung" là sai xa hay sai suýt.
        lech = None
        if cung_video and hang_khung is None:
            lech = min(min(abs(c.frame_idx - gt.frame_start),
                           abs(c.frame_idx - gt.frame_end))
                       for _, c in cung_video)

        nhom = "A_cham_khung" if hang_khung else (
            "B_dung_video_sai_khung" if hang_video else "C_sai_video")
        su_kien_dung = {e.event_id for e in theo_video_toan_bo.get(gt.video_id, [])
                        if gt.frame_start <= e.keyframe_idx <= gt.frame_end}
        hang_kho = (hang_toan_kho(p, compiled, gt, su_kien_dung)
                    if not args.khong_do_hang else None)
        hang_ns = {}
        for so_video in SO_VIDEO_THU:
            hang_ns[so_video] = (
                mo_phong_ngan_sach_video(p, compiled, cands, gt, args.top_k,
                                         so_video)
                if not args.khong_mo_phong and cands else None)
        chi_tiet.append({
            "qid": q.qid, "task": q.task, "nhom": nhom,
            "hang_khung": hang_khung or "", "hang_video": hang_video or "",
            "hang_toan_kho": hang_kho or "", "co_event_dung": int(bool(su_kien_dung)),
            **{f"hang_ns_{k}video": (hang_ns[k] or "") for k in SO_VIDEO_THU},
            "so_dong_dung_video": len(cung_video),
            "lech_khung_gan_nhat": lech if lech is not None else "",
            "so_dong_trung_vector": n_trung,
            "video_dung": gt.video_id, "cua_so": f"{gt.frame_start}-{gt.frame_end}",
            "text": q.text[:70],
        })

    os.makedirs(os.path.dirname(args.ra), exist_ok=True)
    with open(args.ra, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(chi_tiet[0]))
        w.writeheader()
        w.writerows(chi_tiet)

    # --- tổng hợp ------------------------------------------------------------ #
    def _in_nhom(ten, ds):
        print(f"  {ten:26s} {len(ds):>3d}/{len(chi_tiet)} "
              f"({len(ds) / len(chi_tiet):.0%})")

    A = [r for r in chi_tiet if r["nhom"] == "A_cham_khung"]
    B = [r for r in chi_tiet if r["nhom"] == "B_dung_video_sai_khung"]
    C = [r for r in chi_tiet if r["nhom"] == "C_sai_video"]
    print("=" * 74)
    print(f"  CHẨN ĐOÁN {len(chi_tiet)} CÂU (top-{args.top_k})")
    print("=" * 74)
    _in_nhom("A chạm khung (có điểm)", A)
    _in_nhom("B đúng video, sai khung", B)
    _in_nhom("C sai video", C)

    if A:
        print(f"\n  A: hạng trung vị của dòng chạm khung: "
              f"{statistics.median(int(r['hang_khung']) for r in A):.0f}")
    if B:
        lech = [int(r["lech_khung_gan_nhat"]) for r in B
                if r["lech_khung_gan_nhat"] != ""]
        print(f"  B: hạng trung vị của dòng đúng video: "
              f"{statistics.median(int(r['hang_video']) for r in B):.0f}"
              f" · số dòng đúng video trung vị: "
              f"{statistics.median(r['so_dong_dung_video'] for r in B):.0f}")
        if lech:
            print(f"     lệch khung gần nhất: trung vị {statistics.median(lech):.0f} "
                  f"khung (~{statistics.median(lech) / 25:.1f}s @25fps) · "
                  f"nhỏ nhất {min(lech)} · lớn nhất {max(lech)}")
    if not args.khong_do_hang:
        thieu = [r for r in chi_tiet if not r["co_event_dung"]]
        print(f"\n  HẠNG CỦA KEYFRAME ĐÚNG trong {rep.events:,d} event "
              f"(hình học của search)")
        for ten, ds in (("A chạm khung", A), ("B đúng video sai khung", B),
                        ("C sai video", C)):
            h = [int(r["hang_toan_kho"]) for r in ds if r["hang_toan_kho"] != ""]
            if h:
                print(f"    {ten:24s} trung vị {statistics.median(h):>9,.0f} "
                      f"· nhỏ nhất {min(h):>6,d} · lớn nhất {max(h):>9,d}")
        if thieu:
            print(f"    ⚠ {len(thieu)} câu KHÔNG có event nào khớp cửa sổ GT "
                  "(lỗi dữ liệu, không phải lỗi encoder): "
                  f"{[r['qid'] for r in thieu][:5]}")
        # Trần trên của "truy hồi sâu hơn rồi rerank": nếu pool lấy tới độ sâu D
        # thì keyframe đúng của bao nhiêu câu nằm trong pool? Đây là TRẦN với giả
        # định reranker HOÀN HẢO — không phải điểm hứa được.
        hang_all = [int(r["hang_toan_kho"]) for r in chi_tiet
                    if r["hang_toan_kho"] != ""]
        print("    trần theo độ sâu pool (giả định rerank hoàn hảo): " + " · ".join(
            f"D={d}: {sum(1 for h in hang_all if h <= d)}/{len(chi_tiet)}"
            for d in (100, 200, 500, 1000, 2000, 5000)))
    if co_trung:
        tv = statistics.median(r["so_dong_trung_vector"] for r in chi_tiet)
        xau = [r for r in chi_tiet if r["nhom"] != "A_cham_khung"]
        tv_xau = statistics.median(r["so_dong_trung_vector"] for r in xau) if xau else 0
        print(f"  Dòng trùng vector trong top-{args.top_k}: trung vị {tv:.0f} "
              f"(riêng câu trượt: {tv_xau:.0f})")

    # --- mô phỏng ngân sách slot theo video (hướng 4.3) ---------------------- #
    tot_nhat = (0, None)
    if not args.khong_mo_phong:
        print(f"\n  NGÂN SÁCH SLOT THEO VIDEO — nở timeline + round-robin "
              f"(trần trên của 4.3)")
        print(f"    {'V video':>8s}  {'chạm khung':>12s}  {'A giữ':>8s}  "
              f"{'cứu B':>6s}  {'cứu C':>6s}  {'hạng TV':>8s}")
        for k in SO_VIDEO_THU:
            cot = f"hang_ns_{k}video"
            duoc = [r for r in chi_tiet if r[cot] != ""]
            giu = [r for r in A if r[cot] != ""]
            cuu_b = [r for r in B if r[cot] != ""]
            cuu_c = [r for r in C if r[cot] != ""]
            tv = (statistics.median(int(r[cot]) for r in duoc) if duoc else 0)
            print(f"    {k:>8d}  {len(duoc):>7d}/{len(chi_tiet):<4d}  "
                  f"{len(giu):>4d}/{len(A):<3d}  {len(cuu_b):>6d}  "
                  f"{len(cuu_c):>6d}  {tv:>8.0f}")
            if len(duoc) > tot_nhat[0]:
                tot_nhat = (len(duoc), k)
        print(f"    hiện tại (không ngân sách): {len(A)}/{len(chi_tiet)} chạm khung, "
              f"hạng trung vị "
              f"{statistics.median(int(r['hang_khung']) for r in A) if A else 0:.0f}")
        print("    ⚠ 'chạm khung' chỉ là ĐIỀU KIỆN CẦN. Điểm thật còn phụ thuộc "
              "hạng (R@k lấy max trên tiền tố) — phải chạy do_diem_thuc_te để chốt.")

    print("\n  KẾT LUẬN")
    if not B and not C:
        print("    Mọi câu đều chạm khung — không có gì để chẩn đoán ở tầng tìm.")
    else:
        print(f"    B (đúng video, sai khung) {len(B)} · C (sai video) {len(C)}"
              f" -> {'nghiêng SAI VIDEO' if len(C) > len(B) else ('nghiêng ĐÚNG VIDEO SAI KHUNG' if len(B) > len(C) else 'cân bằng')}")
    if co_trung:
        xau = [r for r in chi_tiet if r["nhom"] != "A_cham_khung"]
        tong_trung = sum(r["so_dong_trung_vector"] for r in xau)
        if tong_trung == 0:
            print("    ⚠ 4.2 KHỬ TRÙNG không nhắm vào bệnh này: các câu trượt có "
                  f"**0** dòng trùng vector trong top-{args.top_k}. 904 nhóm trùng "
                  "là chuyện thật của corpus, nhưng chúng không lọt vào top nên "
                  "khử trùng sẽ không đổi điểm ở dev set này.")
    if not args.khong_mo_phong and tot_nhat[1]:
        delta = tot_nhat[0] - len(A)
        print(f"    Ngân sách slot theo video (V={tot_nhat[1]}): {len(A)} -> "
              f"{tot_nhat[0]} câu chạm khung ({delta:+d}). Đây mới là con số của "
              f"hướng 4.3; 'nở ứng viên' một mình là no-op (xem docstring).")
    print(f"\n  Chi tiết từng câu: {args.ra}")


if __name__ == "__main__":
    main()
