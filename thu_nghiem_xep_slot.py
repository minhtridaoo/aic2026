#!/usr/bin/env python
"""100 slot nên chia thế nào: nhiều video ít khung, hay ít video nhiều khung?

    python thu_nghiem_xep_slot.py

CÂU HỎI
-------
Đo được trên dev set: **video**-R@100 = 68.3% nhưng **frame**-R@100 chỉ 28.3%,
R@1 = 0%. Tức 41/60 câu đã lôi đúng video vào 100 dòng, mà chỉ 17 câu chạm đúng
khung. Chênh lệch đó là điểm đang bỏ trên bàn.

Nghi phạm: `submission.diversify_tail` rải 100 slot ra ~100 video KHÁC NHAU, nên
mỗi video chỉ được 1 khung. Một video có ~203 keyframe, cửa sổ đáp án rộng đúng 1
keyframe -> bắn 1 phát vào 203 ô thì trúng là may.

Đổi lại: xếp hạng VIDEO trước, rồi tiêu nhiều khung cho ít video.

BA TRỤC ĐƯỢC QUÉT
-----------------
1. chia slot     (số video × số khung mỗi video ≈ 100)
2. điểm của video   max khung tốt nhất, hay tổng 3 khung tốt nhất
3. thứ tự xuất   xen kẽ (khung tốt nhất của mọi video trước) hay theo khối
                 (đủ khung của video 1, rồi video 2, ...)

Trục 3 quan trọng vì điểm chính thức là `max` trên TIỀN TỐ: xen kẽ đặt khung tốt
nhất của video hạng 7 vào slot 7, còn theo khối đẩy nó xuống slot ~31.

TRUY HỒI SÂU
------------
Muốn 5 khung cho một video thì top-100 không đủ — nó thường chỉ chứa 1-2 khung
mỗi video. Vì vậy script truy hồi TOP-3000 rồi mới chia slot. Tầng TÌM không đổi,
chỉ lấy sâu hơn.
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026.core import scoring
from aic2026.core import submission as sub
from aic2026.core.dataset import load_dataset
from aic2026.core.offline_eval import load_queries
from aic2026.core.providers import CLIPTextEncoder, SigLIP2TextEncoder
from aic2026.core.retrieval import MultiChannelRetriever
from aic2026.core.stores import NumpyVectorStore

GOC = os.path.dirname(os.path.abspath(__file__))
DATA = r"D:\COMP\AIC2026\data cuộc thi\aic2026_data_nhe"
DEV_JSON = os.path.join(GOC, "data", "dev_set_tu_lam.json")
CACHE = os.path.join(GOC, "dev_set", "query_vectors.npz")

SAU = 3000            # truy hồi sâu bao nhiêu trước khi chia slot
MAX_ROWS = 100
CHIA = [(100, 1), (50, 2), (33, 3), (20, 5), (10, 10), (5, 20), (2, 50)]


def tb_va_se(v):
    n = len(v)
    tb = sum(v) / n
    if n < 2:
        return tb, float("inf")
    var = sum((x - tb) ** 2 for x in v) / (n - 1)
    return tb, math.sqrt(var / n)


def gom_theo_video(cands):
    """{video: [ứng viên theo điểm giảm dần]}. `cands` đã sắp giảm dần."""
    per = {}
    for c in cands:
        per.setdefault(c.video_id, []).append(c)
    return per


def tb_n(ds, n):
    t = sorted(ds, reverse=True)[:n]
    return sum(t) / len(t)


# Luật gom điểm khung -> điểm video. Đây là knob quyết định, xem phần
# "QUÉT CÁCH GOM ĐIỂM CHO VIDEO" ở cuối output.
LUAT = [
    ("max",        lambda ds: max(ds)),
    ("tb top2",    lambda ds: tb_n(ds, 2)),
    ("tb top3",    lambda ds: tb_n(ds, 3)),
    ("tb top5",    lambda ds: tb_n(ds, 5)),
    ("tb top10",   lambda ds: tb_n(ds, 10)),
    ("tổng top3",  lambda ds: sum(sorted(ds, reverse=True)[:3])),
    ("tổng top5",  lambda ds: sum(sorted(ds, reverse=True)[:5])),
    ("tổng top10", lambda ds: sum(sorted(ds, reverse=True)[:10])),
    ("đếm khung",  lambda ds: float(len(ds))),
]
LUAT_MAP = dict(LUAT)


def xep_slot(cands, n_video, m_frame, diem_video="max", thu_tu="xen"):
    per = gom_theo_video(cands)
    f = LUAT_MAP[diem_video]

    def diem(v):
        return f([c.score for c in per[v]])

    xep = sorted(per, key=lambda v: -diem(v))[:n_video]
    rows = []
    if thu_tu == "xen":
        for i in range(m_frame):
            for v in xep:
                if i < len(per[v]):
                    rows.append(per[v][i])
    else:
        for v in xep:
            rows.extend(per[v][:m_frame])
    return [{"video_id": c.video_id, "frame_idx": c.frame_idx}
            for c in rows[:MAX_ROWS]]


def _nap_ddt():
    """Dùng chung `BIEN_THE_SIGLIP2` của do_diem_thuc_te thay vì khai lại đường dẫn."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ddt", os.path.join(GOC, "do_diem_thuc_te.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    # Mặc định SigLIP2 (encoder chính từ 2026-08-12). `--clip` = đường cũ của BTC.
    # Cách chia slot phụ thuộc MẬT ĐỘ ứng viên đúng trong một video, mà cái đó do
    # encoder quyết định — nên số đo trên CLIP không suy ra được cho SigLIP2.
    if "--clip" in sys.argv:
        goc_feature, cache = None, CACHE
        lop_enc, model_id = CLIPTextEncoder, CLIPTextEncoder.MODEL_ID
        ten_enc = "CLIP ViT-B/32 (BTC cấp)"
    else:
        model_id, goc_feature, cache = _nap_ddt().duong_siglip2("base", vi=False)
        lop_enc, ten_enc = SigLIP2TextEncoder, f"SigLIP2 {model_id}"

    thieu = [p for p in (DATA, DEV_JSON, cache) if not os.path.exists(p)]
    if goc_feature and not os.path.isdir(goc_feature):
        thieu.append(goc_feature)
    if thieu:
        raise SystemExit("❌ Thiếu:\n" + "\n".join(f"   - {p}" for p in thieu)
                         + "\n   (dùng `--clip` để chạy trên encoder cũ)")

    print(f"Encoder: {ten_enc}")
    print("Đang nạp 873 video ...")
    events, emb, metas, rep = load_dataset(DATA, goc_feature=goc_feature)
    print(f"  {rep.events:,d} event · {emb.shape[1]} chiều")

    enc = lop_enc(cho_phep_live=True)
    enc.MODEL_ID = model_id       # trước nap_cache, để phép kiểm khoá `model` có tác dụng
    enc.nap_cache(cache)
    r = MultiChannelRetriever(enc, vector_store=NumpyVectorStore(
        khu_modality_gap=True))
    r.index(events, embeddings=emb)

    queries = load_queries(DEV_JSON)
    print(f"  {len(queries)} câu dev set")
    N = len(queries)      # KHONG hardcode: dev set da doi 60 -> 120 cau

    print(f"\nTruy hồi top-{SAU} (một lần) ...")
    t = time.time()
    goi = [(gt, r.retrieve(q.text, top_k=SAU, fusion="rrf"))
           for q, gt in queries]
    print(f"  xong {time.time()-t:.1f}s")

    # --- premise check: top-100 hiện rải ra bao nhiêu video? --------------- #
    sl = [len(gom_theo_video(c[:100])) for _, c in goi]
    print(f"\n  Kiểm tiền đề: top-100 ứng viên trải trên trung vị "
          f"{sorted(sl)[len(sl)//2]} video khác nhau")
    khung_moi_video = [len(v) for _, c in goi
                       for v in gom_theo_video(c[:100]).values()]
    print(f"  -> trung vị {sorted(khung_moi_video)[len(khung_moi_video)//2]} "
          f"khung mỗi video. Tiền đề đứng: mỗi video gần như chỉ được 1 phát.")

    def cham(lam_rows):
        d, r1, r5, r100 = [], [], [], []
        for gt, c in goi:
            rows = lam_rows(c)
            d.append(scoring.final_score(rows, gt))
            r1.append(scoring.recall_at_k(rows, gt, 1, threshold=1.0))
            r5.append(scoring.recall_at_k(rows, gt, 5, threshold=1.0))
            r100.append(scoring.recall_at_k(rows, gt, 100, threshold=1.0))
        return d, r1, r5, r100

    nen_v, nen_r1, nen_r5, nen_r100 = cham(lambda c: sub.kis_rows(c[:100]))
    nen, _ = tb_va_se(nen_v)
    print("\n" + "=" * 78)
    print("  NỀN: submission.kis_rows trên top-100 (cách đang dùng)")
    print("=" * 78)
    print(f"  điểm {nen:.4f} | R@1 {sum(nen_r1)/N:.1%} "
          f"| R@5 {sum(nen_r5)/N:.1%} | R@100 {sum(nen_r100)/N:.1%}")

    print("\n" + "=" * 78)
    print(f"  QUÉT CHIA SLOT (truy hồi sâu top-{SAU})")
    print("=" * 78)
    print(f"  {'video×khung':>12s} {'điểm video':>10s} {'thứ tự':>7s} "
          f"{'điểm':>7s} {'Δ so nền':>17s} {'R@1':>6s} {'R@5':>6s} {'R@100':>7s}")
    print("  " + "-" * 76)

    ket = []
    vec_cau_hinh = {}          # (nv, mf, dv, tt) -> điểm từng câu
    for dv in ("max", "tổng top5", "tổng top10"):
        for tt in ("xen", "khoi"):
            for nv, mf in CHIA:
                d, a1, a5, a100 = cham(
                    lambda c, nv=nv, mf=mf, dv=dv, tt=tt:
                    xep_slot(c, nv, mf, diem_video=dv, thu_tu=tt))
                tb, _ = tb_va_se(d)
                dd, se = tb_va_se([x - y for x, y in zip(d, nen_v)])
                sao = "*" if abs(dd) > 2 * se else " "
                ket.append((tb, dd, se, nv, mf, dv, tt,
                            sum(a1) / N, sum(a5) / N, sum(a100) / N))
                vec_cau_hinh[(nv, mf, dv, tt)] = d
                print(f"  {f'{nv}×{mf}':>12s} {dv:>10s} {tt:>7s} "
                      f"{tb:>7.4f} {dd:>+9.4f}±{se:.4f}{sao} "
                      f"{sum(a1)/N:>5.1%} {sum(a5)/N:>5.1%} "
                      f"{sum(a100)/N:>6.1%}")

    print("\n" + "=" * 78)
    print("  TỐT NHẤT")
    print("=" * 78)
    for tb, dd, se, nv, mf, dv, tt, a1, a5, a100 in sorted(ket, reverse=True)[:5]:
        y = "*" if abs(dd) > 2 * se else " "
        print(f"  {tb:.4f}  {nv}×{mf} {dv}/{tt}  Δ {dd:+.4f}±{se:.4f}{y}  "
              f"R@1 {a1:.1%} R@5 {a5:.1%} R@100 {a100:.1%}")
    print("\n  '*' = |Δ| > 2·SE. Không có dấu thì đừng tin con số.")

    # ----------------------------------------------------------------------- #
    # TẬP GIỮ LẠI — bảng trên vừa chọn quán quân trong 63 cấu hình TRÊN CHÍNH
    # 60 câu dùng để chấm. Đó là chọn trên tập test: với SE ±0.025 và 63 lần
    # thử, riêng nhiễu cũng đủ dựng một đỉnh +0.05. Nên phần thắng ở trên KHÔNG
    # đáng tin cho tới khi kiểm trên dữ liệu chưa dùng để chọn.
    #
    # Chia theo VIDEO (không theo câu): hai câu cùng video mà nằm hai bên thì
    # thông tin rỉ qua, vì cách gom điểm video là chính thứ đang được chọn.
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("  KIỂM TRÊN TẬP GIỮ LẠI (chia theo video)")
    print("=" * 78)

    vids = sorted({gt.video_id for gt, _ in goi})
    nua = {v: (i % 2) for i, v in enumerate(vids)}
    phan = [nua[gt.video_id] for gt, _ in goi]
    print(f"  {len(vids)} video -> nửa A: {phan.count(0)} câu, "
          f"nửa B: {phan.count(1)} câu")

    def loc(v, p):
        return [x for x, q in zip(v, phan) if q == p]

    for fit, danh_gia in ((0, 1), (1, 0)):
        # chọn cấu hình tốt nhất CHỈ dựa trên nửa `fit`
        tot = max(vec_cau_hinh,
                  key=lambda k: sum(loc(vec_cau_hinh[k], fit)))
        nv, mf, dv, tt = tot
        v_fit = loc(vec_cau_hinh[tot], fit)
        v_dg = loc(vec_cau_hinh[tot], danh_gia)
        n_fit, n_dg = loc(nen_v, fit), loc(nen_v, danh_gia)
        d_fit = sum(v_fit) / len(v_fit) - sum(n_fit) / len(n_fit)
        dd, se = tb_va_se([a - b for a, b in zip(v_dg, n_dg)])
        ten_nua = "AB"[fit], "AB"[danh_gia]
        print(f"\n  chọn trên nửa {ten_nua[0]} -> {nv}×{mf} {dv}/{tt}")
        print(f"    trên nửa {ten_nua[0]} (đã dùng để chọn): Δ {d_fit:+.4f}"
              "   <- con số này BỊ THIÊN VỊ")
        print(f"    trên nửa {ten_nua[1]} (giữ lại)         : Δ {dd:+.4f} ± {se:.4f}"
              + ("  ✅ giữ được dấu dương" if dd > 0 else "  ❌ mất khi giữ lại"))

    print("\n  Đọc: nếu Δ trên nửa giữ lại tụt gần 0 hoặc âm, thì +0.0233 ở bảng")
    print("  trên chỉ là hiệu ứng chọn lọc, không phải cải thiện thật.")

    # Cỡ mẫu: cần bao nhiêu câu để phân biệt được hiệu ứng cỡ này?
    tot_chung = max(vec_cau_hinh, key=lambda k: sum(vec_cau_hinh[k]))
    _, se60 = tb_va_se([a - b for a, b in zip(vec_cau_hinh[tot_chung], nen_v)])
    hieu_ung = 0.02
    can = int(math.ceil(len(nen_v) * (2 * se60 / hieu_ung) ** 2))
    print(f"\n  CỠ MẪU: SE ghép cặp của cấu hình tốt nhất trên {len(nen_v)} câu là "
          f"±{se60:.4f}.")
    print(f"  Để 2·SE < {hieu_ung:.2f} (tức phân biệt được hiệu ứng ~+0.02) cần "
          f"khoảng {can} câu dev set.")
    print(f"  Hiện có {len(nen_v)}. Đây là NÚT CỔ CHAI THẬT: không phải thiếu ý")
    print("  tưởng tối ưu, mà là không đủ dữ liệu để biết ý tưởng nào đúng.")

    # ----------------------------------------------------------------------- #
    # Vì sao chia slot không cứu được: trong ĐÚNG video, khung đúng hạng mấy?
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("  CHẨN ĐOÁN: bên trong đúng video, khung đúng xếp hạng thứ mấy?")
    print("=" * 78)
    print("  Nếu khung đúng nằm ở hạng ~giữa số keyframe của video, tức encoder")
    print("  KHÔNG phân biệt được các khung trong cùng một video -> chia slot lại")
    print("  bao nhiêu cũng vô ích, phải đổi tín hiệu chấm điểm.")

    kf_moi_video = {}
    for e in events:
        kf_moi_video[e.video_id] = kf_moi_video.get(e.video_id, 0) + 1

    hang_video, hang_khung, ty_le = [], [], []
    for (gt, cands) in goi:
        per = gom_theo_video(cands)
        if gt.video_id not in per:
            continue
        xep_v = sorted(per, key=lambda v: max(c.score for c in per[v]),
                       reverse=True)
        hang_video.append(xep_v.index(gt.video_id) + 1)
        ds = per[gt.video_id]                       # đã theo điểm giảm dần
        vi_tri = next((i + 1 for i, c in enumerate(ds)
                       if gt.frame_start <= c.frame_idx <= gt.frame_end), None)
        if vi_tri is None:
            continue
        hang_khung.append(vi_tri)
        ty_le.append(vi_tri / max(kf_moi_video[gt.video_id], 1))

    def tv(v):
        return sorted(v)[len(v) // 2] if v else float("nan")

    print(f"\n  {len(hang_video)}/{N} câu có đúng video trong top-{SAU}")
    print(f"  hạng của ĐÚNG VIDEO (trong ~873): trung vị {tv(hang_video)}")
    print(f"  {len(hang_khung)}/{N} câu có đúng khung trong danh sách ứng viên")
    print(f"  hạng của ĐÚNG KHUNG trong video đó: trung vị {tv(hang_khung)}")
    print(f"  hạng đó / số keyframe của video   : trung vị {tv(ty_le):.2f}")
    print(f"    (0.50 = ngẫu nhiên hoàn toàn; càng gần 0 càng phân biệt tốt)")
    if not math.isnan(tv(ty_le)) and tv(ty_le) > 0.25:
        print("\n  => Encoder gần như KHÔNG phân biệt được khung trong cùng video.")
        print("     Đó là lý do 68.3% 'đúng video' không đổi thành điểm được, và")
        print("     là lý do mọi cách chia slot đều thất bại. Đòn tiếp theo phải")
        print("     nhắm vào TÍN HIỆU (encoder mạnh hơn / rerank bằng VLM), không")
        print("     phải vào cách tiêu slot.")
    else:
        print("\n  => Encoder CÓ phân biệt được trong video.")
        print("     Nhưng xem bảng QUÉT CHIA SLOT ở trên trước khi kết luận là còn")
        print("     dư địa: nếu mọi cấu hình đều thua nền, nghĩa là chỗ mất điểm")
        print("     KHÔNG nằm ở cách tiêu slot mà ở việc lôi đúng video vào danh")
        print("     sách — dồn slot cho ít video chỉ làm mất thêm recall.")

    tv_video, tv_khung = tv(hang_video), tv(hang_khung)

    # ----------------------------------------------------------------------- #
    # Nút cổ chai là XẾP HẠNG VIDEO. Cách gom điểm nào đưa đúng video lên cao?
    # ----------------------------------------------------------------------- #
    print("\n" + "=" * 78)
    print("  QUÉT CÁCH GOM ĐIỂM CHO VIDEO")
    print("=" * 78)
    print(f"  Khung đúng ở hạng ~{tv_khung} trong video của nó, nhưng video đúng")
    print(f"  lại hạng ~{tv_video}. Nên thứ cần cải thiện là điểm của VIDEO. `max` chỉ nghe một")
    print("  khung duy nhất; gộp nhiều khung có thể vững hơn — hoặc tệ hơn, vì")
    print("  video dài tự nhiên có nhiều ứng viên. Đo mới biết.\n")

    MOC = (1, 5, 10, 25, 50, 100)

    print(f"  {'luật':>11s} {'hạng tv':>8s} " +
          " ".join(f"{'V@'+str(m):>7s}" for m in MOC))
    print("  " + "-" * 66)
    for ten, f in LUAT:
        hang = []
        for gt, cands in goi:
            per = gom_theo_video(cands)
            if gt.video_id not in per:
                continue
            diem_v = {v: f([c.score for c in cs]) for v, cs in per.items()}
            xep = sorted(diem_v, key=lambda v: -diem_v[v])
            hang.append(xep.index(gt.video_id) + 1)
        print(f"  {ten:>11s} {tv(hang):>8d} " +
              " ".join(f"{sum(1 for h in hang if h <= m)/N:>6.1%} "
                       for m in MOC))

    print(f"\n  V@N = tỉ lệ câu có ĐÚNG VIDEO nằm trong N video đầu (trên {N} câu).")
    print("  So với V@N hiện tại của `max`, luật nào kéo V@10 / V@25 lên rõ thì")
    print(f"  đó là chỗ đáng đổi — vì trong video, khung đúng đã sẵn ở hạng ~{tv_khung}.")


if __name__ == "__main__":
    main()
