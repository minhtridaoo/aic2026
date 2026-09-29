#!/usr/bin/env python
"""Đo điểm trên DỮ LIỆU THẬT — con số có nghĩa đầu tiên của hệ thống.

    python do_diem_thuc_te.py              # SigLIP2 (MẶC ĐỊNH) + A/B hai cách xếp slot
    python do_diem_thuc_te.py --clip       # đường CŨ: vector CLIP BTC cấp sẵn
    python do_diem_thuc_te.py --thu        # chỉ thử tìm bằng chữ, xem mắt thường
    python do_diem_thuc_te.py --siglip2=so400m       # checkpoint mạnh hơn, 1152 chiều
    python do_diem_thuc_te.py --vi                   # truy vấn TIẾNG VIỆT thẳng
    python do_diem_thuc_te.py --config               # ĐÚNG cấu hình thi
    python do_diem_thuc_te.py --config --vi          # cấu hình thi + tiếng Việt

CẦN CÓ
------
  1. data/dev_set_tu_lam.json   <- chạy tao_dev_set.py sau khi gán nhãn
  2. bộ data nhẹ 179 MB         <- cấp map-keyframes và media-info
  3. thư mục vector ảnh SigLIP2 <- kaggle_encode_siglip2.py (GPU T4, 260 MB)
  4. dev_set/query_vectors_siglip2.npz <- kaggle_encode_truy_van_siglip2.py

(3) và (4) PHẢI sinh ra từ cùng một checkpoint. Đường `--clip` thì cần
`dev_set/query_vectors.npz` thay cho (4) và không cần (3).

MẶC ĐỊNH LÀ SigLIP2 KỂ TỪ 2026-08-12
------------------------------------
Ghép cặp trên 120 câu: 0.2950 so với 0.1867, tức **+0.1083 ± 0.0313** (3.5·SE),
hạng trung vị của keyframe đúng 246 -> 119. Xem docstring `SigLIP2TextEncoder`.

Muốn so CLIP với SigLIP2 thì dùng `so_sanh_clip_siglip2.py` — nó chấm cả hai trên
cùng tập câu nên tính được SE ghép cặp. Cột `delta ± paired SE` in ở cuối script
này CHỈ ghép cặp giữa core/ và +optimize trong một lần chạy, không so hai encoder.

Thiếu cái nào script sẽ nói rõ thiếu gì và làm sao có, không chạy nửa vời.

KHÁC GÌ `run_benchmark.py`
-------------------------
`run_benchmark.py` chạy trên 8 sự kiện bịa với encoder giả — chỉ chứng minh
đường ống thông. Script này chạy trên 177.321 keyframe thật với vector CLIP thật,
nên con số nó in ra mới nói được điều gì đó về chất lượng tìm kiếm.
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from aic2026 import evaluation as scoring
from aic2026.core.dataset import load_dataset
from aic2026.core.offline_eval import load_queries
from aic2026.core.pipeline import Pipeline, PortfolioConfig
from aic2026.core.providers import (CLIPTextEncoder, MCLIPTextEncoder,
                                    RuleBasedProvider, SigLIP2TextEncoder)
from aic2026.core.stores import NumpyVectorStore
from aic2026.optimize.calibration import RankCalibrator

GOC = os.path.dirname(os.path.abspath(__file__))
DATA = r"D:\COMP\AIC2026\data cuộc thi\aic2026_data_nhe"
DEV_JSON = os.path.join(GOC, "data", "dev_set_tu_lam.json")
CACHE = os.path.join(GOC, "dev_set", "query_vectors.npz")
MCLIP_VI_CACHE = os.path.join(GOC, "dev_set", "query_mclip_vi.npz")

# --- đường SigLIP2 (cờ --siglip2) ------------------------------------------- #
# Vector ảnh SigLIP2 nằm ở thư mục RIÊNG, KHÔNG giải nén vào DATA: `phan_loai`
# coi mọi .npy có chữ "feature" là clip_features và `scan` ghi đè im lặng, nên
# trộn chung là mỗi video một không gian vector khác nhau. Xem `dataset._thay_feature`.
#
# Mỗi checkpoint một thư mục và một cache RIÊNG. Dùng chung thư mục là trộn 768
# chiều với 1152 chiều — `_thay_feature` sẽ thấy hai .npy cho cùng video và nổ,
# nhưng chỉ khi hai bộ nằm cùng chỗ; tách sẵn thì không có cơ hội sai.
_DATA_GOC = os.path.dirname(DATA)
BIEN_THE_SIGLIP2 = {
    "base": {
        "model_id": "google/siglip2-base-patch16-224",
        "features": os.path.join(_DATA_GOC, "aic2026_siglip2"),
        "cache": "query_vectors_siglip2.npz",
        "cache_vi": "query_vectors_siglip2_vi.npz",
    },
    "so400m": {
        "model_id": "google/siglip2-so400m-patch14-384",
        "features": os.path.join(_DATA_GOC, "aic2026_siglip2_so400m"),
        "cache": "query_vectors_siglip2_so400m.npz",
        "cache_vi": "query_vectors_siglip2_so400m_vi.npz",
    },
}


def duong_siglip2(bien_the: str, vi: bool):
    """Trả (model_id, thư mục feature, đường dẫn cache truy vấn) của một biến thể."""
    if bien_the not in BIEN_THE_SIGLIP2:
        print(f"❌ Biến thể SigLIP2 không biết: {bien_the!r}. "
              f"Có: {', '.join(BIEN_THE_SIGLIP2)}")
        sys.exit(1)
    c = BIEN_THE_SIGLIP2[bien_the]
    return (c["model_id"], c["features"],
            os.path.join(GOC, "dev_set", c["cache_vi" if vi else "cache"]))


def kiem_tra_dieu_kien(can_dev_set: bool, siglip2: str = "", vi: bool = False):
    thieu = []
    if not os.path.isdir(DATA):
        thieu.append(f"Bộ data nhẹ: {DATA}\n      -> chạy kaggle_lay_data_nhe.py")
    if siglip2:
        _, feat, cache = duong_siglip2(siglip2, vi)
        if not os.path.exists(cache):
            # KHÔNG chặn: `nap_he_thong` dựng SigLIP2TextEncoder(cho_phep_live=
            # True) nên thiếu cache chỉ là chậm hơn, không phải không chạy được.
            # Chặn ở đây từng khoá cứng việc đo tiếng Việt ở so400m (chỉ có cache
            # tiếng Việt cho base-224), trong khi phép đo đó là thứ cần nhất.
            them = " với BAN_TIENG_VIET = True" if vi else ""
            print(f"⚠ Không có cache vector truy vấn: {cache}")
            print(f"   Sẽ ENCODE LIVE (chậm hơn, cần tải model). Muốn nhanh thì "
                  f"chạy kaggle_encode_truy_van_siglip2.py{them} rồi tải về.\n")
        if not os.path.isdir(feat):
            thieu.append(f"Vector ảnh SigLIP2 ({siglip2}): {feat}\n"
                         "      -> chạy kaggle_encode_siglip2.py trên Kaggle GPU,\n"
                         "         tải zip về rồi giải nén vào ĐÚNG thư mục trên\n"
                         "         (đừng giải nén vào bộ data nhẹ — sẽ trộn hai không gian)")
    elif not os.path.exists(CACHE):
        thieu.append(f"Vector truy vấn: {CACHE}\n"
                     "      -> chạy kaggle_encode_truy_van.py trên Kaggle, tải về")
    if can_dev_set and not os.path.exists(DEV_JSON):
        thieu.append(f"Dev set: {DEV_JSON}\n"
                     "      -> gán nhãn dev_set/dan_nhan.csv rồi chạy tao_dev_set.py")
    if thieu:
        print("❌ Còn thiếu:\n")
        for t in thieu:
            print(f"   - {t}")
        sys.exit(1)


def nap_he_thong(siglip2: str = "", vi: bool = False):
    model_id, feat, cache = (duong_siglip2(siglip2, vi) if siglip2
                             else ("", None, CACHE))
    print("Đang nạp 873 video ...")
    t = time.time()
    events, emb, metas, rep = load_dataset(DATA, goc_feature=feat)
    print(f"  {rep.events:,d} event / {rep.videos_da_nap} video  ({time.time()-t:.1f}s)")
    if rep.thieu_file:
        print(f"  ⚠ {len(rep.thieu_file)} video thiếu file, đã bỏ "
              f"(vd {', '.join(rep.thieu_file[:3])})")
    if siglip2:
        enc = SigLIP2TextEncoder(cho_phep_live=True)
        # MODEL_ID phải đặt TRƯỚC nap_cache: phép kiểm chéo khoá `model` trong
        # .npz chạy lúc nạp, gán sau thì nó so với giá trị cũ và mất tác dụng.
        enc.MODEL_ID = model_id
        if os.path.exists(cache):
            enc.nap_cache(cache)
        print(f"  encoder: SigLIP2 {enc.MODEL_ID}  ({enc.dim} chiều)"
              + ("  [TRUY VẤN TIẾNG VIỆT]" if vi else ""))
        # Vector ảnh và vector chữ phải cùng số chiều, nếu không `NumpyVectorStore`
        # chỉ nổ ở tận lúc matmul với thông báo shape khó hiểu.
        if emb.size and enc.dim is not None and emb.shape[1] != enc.dim:
            print(f"\n❌ Vector ảnh {emb.shape[1]} chiều, vector chữ {enc.dim} chiều.")
            print("   Hai bên phải cùng MỘT checkpoint. Kiểm `MODEL_ID` ở")
            print("   kaggle_encode_siglip2.py và kaggle_encode_truy_van_siglip2.py.")
            sys.exit(1)
    else:
        enc = CLIPTextEncoder(cache_path=CACHE, cho_phep_live=True)
    print(f"  cache truy vấn: {len(enc._cache)} câu")
    return events, emb, metas, enc


def truong_text_dev_set(vi: bool) -> str:
    """Cột nào của dev set làm câu truy vấn. `--vi` = mô tả tiếng Việt gốc."""
    return "_mo_ta_vi" if vi else "text"


def dung_pipeline(events, emb, enc, portfolio=None, khu_gap=True,
                  qa_output_language="vi", qa_answer_translator=None,
                  llm=None, image_resolver=None, qa_vlm_top_n=10,
                  qa_vlm_question_types=("color", "open")):
    if isinstance(enc, SigLIP2TextEncoder):
        # mCLIP nằm trong không gian ViT-B/32 512 chiều, KHÔNG ghép được với ảnh
        # SigLIP2. Dùng chính encoder SigLIP2 cho câu tiếng Việt — SigLIP2 vốn
        # huấn luyện đa ngữ, nên đây là đường đúng chứ không phải giải pháp tạm.
        vi_enc = enc
    else:
        vi_enc = MCLIPTextEncoder(
            cache_path=MCLIP_VI_CACHE if os.path.exists(MCLIP_VI_CACHE) else None,
            cho_phep_live=True)
    p = Pipeline(enc, llm or RuleBasedProvider(), portfolio=portfolio,
                 fusion="rrf", vietnamese_encoder=vi_enc,
                 qa_output_language=qa_output_language,
                 qa_answer_translator=qa_answer_translator,
                 image_resolver=image_resolver,
                 qa_vlm_top_n=qa_vlm_top_n,
                 qa_vlm_question_types=qa_vlm_question_types)
    p.retriever.visual = NumpyVectorStore(khu_modality_gap=khu_gap)
    p.index(events, embeddings=emb)
    return p


def paired_delta_se(details_a, details_b):
    """Mean paired delta and its standard error, matched by query id."""
    by_id = {row[0]: row[2] for row in details_a}
    if set(by_id) != {row[0] for row in details_b}:
        raise ValueError("Hai cấu hình không chấm cùng tập query")
    diffs = [row[2] - by_id[row[0]] for row in details_b]
    mean = sum(diffs) / len(diffs)
    if len(diffs) < 2:
        return mean, float("inf")
    variance = sum((value - mean) ** 2 for value in diffs) / (len(diffs) - 1)
    return mean, math.sqrt(variance / len(diffs))


def optimizer_recommendation(delta, se):
    """Never recommend a default switch inside the paired noise band."""
    if abs(delta) < 1e-12:
        return "Tầng optimize không đổi gì. Giữ mặc định TẮT."
    if not math.isfinite(se) or abs(delta) <= 2 * se:
        return ("Chênh lệch chưa vượt 2·SE, chưa phân biệt được với nhiễu. "
                "Giữ PortfolioConfig.enabled=False.")
    if delta > 0:
        return ("Tầng optimize tăng điểm vượt 2·SE. Xác nhận lại trên validation "
                "split trước khi đổi mặc định.")
    return "Tầng optimize làm mất điểm vượt 2·SE. Giữ TẮT."


# --------------------------------------------------------------------------- #
def _co_dau_tieng_viet(s: str) -> bool:
    return any(ord(c) > 127 for c in s)


def chan_doan_cung_khong_gian(emb, enc, cau):
    """Truy vấn chữ có PHÂN BIỆT được nội dung không?

    Bài học rút ra ngày 2026-08-07: dùng "độ nổi bật" z = (max − tb)/sd là SAI,
    vì với N mẫu nhiễu Gauss thì E[max z] ≈ sqrt(2·ln N) — với N = 177.321 là
    4.92. Truy vấn tiếng Việt đo được z ≈ 4.8, tức ĐÚNG BẰNG MỨC NHIỄU, nhưng
    ngưỡng cũ lại chấm nó là "ranh giới, tạm được". Suýt thì tin nhầm.

    Phép đo quyết định là ĐỘ CHỒNG LẤN: hai truy vấn KHÁC NGHĨA mà trả về cùng
    một tập kết quả thì chúng không mang thông tin nội dung nào cả.

        chồng lấn thấp (~10%)  -> có phân biệt, dùng được
        chồng lấn cao (>40%)   -> gom cụm, mọi câu đổ về một vùng => vô dụng

    Đo trên chính corpus này: tiếng Anh 12.7%, tiếng Việt 53.4%.
    """
    N = emb.shape[0]
    nguong_nhieu = math.sqrt(2 * math.log(N))
    print("\n" + "=" * 74)
    print("  CHẨN ĐOÁN: truy vấn chữ có phân biệt được nội dung không?")
    print("=" * 74)
    print(f"  Đối chiếu trên TOÀN BỘ {N:,d} vector ảnh")
    print(f"  Mức nhiễu lý thuyết cho N này: z = sqrt(2·ln N) = {nguong_nhieu:.2f}")
    print("  -> z quanh mức đó KHÔNG chứng minh được gì.\n")

    print(f"  {'truy vấn':40s} {'max':>7s} {'z':>6s}")
    print("  " + "-" * 58)
    top100 = {}
    for text in cau:
        try:
            q = np.asarray(enc.encode_text(text), dtype=np.float32)
        except LookupError:
            continue
        d = emb @ q
        z = (float(d.max()) - float(d.mean())) / (float(d.std()) or 1.0)
        top100[text] = set(np.argsort(-d)[:100].tolist())
        co = "!" if z <= nguong_nhieu else " "
        print(f"  {text[:40]:40s} {float(d.max()):7.3f} {z:6.1f} {co}")
    if len(top100) < 2:
        return

    def chong_lan(nhom):
        cap = [(a, b) for i, a in enumerate(nhom) for b in nhom[i + 1:]]
        if not cap:
            return None
        return sum(len(top100[a] & top100[b]) for a, b in cap) / (100 * len(cap))

    print("\n  --- Độ chồng lấn top-100 giữa các truy vấn KHÁC NGHĨA ---")
    print("      (thấp = phân biệt tốt; cao = mọi câu đổ về cùng một vùng)")
    nhom = {"tiếng Việt": [c for c in top100 if _co_dau_tieng_viet(c)],
            "tiếng Anh": [c for c in top100 if not _co_dau_tieng_viet(c)]}
    ket = {}
    for ten, ds in nhom.items():
        if len(ds) >= 2:
            ket[ten] = chong_lan(ds)
            print(f"      {ten:12s} ({len(ds)} câu): {ket[ten]:.1%}")
    if not ket:
        ket["tất cả"] = chong_lan(list(top100))
        print(f"      {'tất cả':12s}: {ket['tất cả']:.1%}")

    print()
    for ten, v in ket.items():
        if v < 0.25:
            print(f"  ✅ {ten}: phân biệt tốt — dùng được để đo.")
        elif v < 0.40:
            print(f"  ⚠️  {ten}: phân biệt yếu — dùng được nhưng kết quả sẽ nhiễu.")
        else:
            print(f"  ❌ {ten}: GOM CỤM ({v:.0%}). Mọi câu trả về gần như cùng một tập")
            print(f"     kết quả => không mang thông tin nội dung. KHÔNG dùng để đo.")
    if ket.get("tiếng Việt", 0) >= 0.40 and ket.get("tiếng Anh", 1) < 0.25:
        print("\n  => KẾT LUẬN: CLIP ViT-B/32 không hiểu tiếng Việt.")
        print("     DỊCH truy vấn sang tiếng Anh trước khi encode.")


def che_do_thu(events, emb, metas, enc):
    """Tìm bằng chữ rồi in ra để nhìn bằng mắt — trước khi tin vào con số."""
    cau = sorted(enc._cache)
    print(f"\nCó {len(cau)} câu trong cache.")
    chan_doan_cung_khong_gian(emb, enc, cau)

    p = dung_pipeline(events, emb, enc)
    print(f"\n\nKết quả tìm kiếm — 4 câu đầu:\n")
    for text in cau[:4]:
        print("=" * 74)
        print(f"  TRUY VẤN: {text}")
        print("=" * 74)
        try:
            cands = p.retriever.retrieve(text, top_k=5)
        except LookupError as e:
            print(f"  ❌ {e}")
            continue
        for i, c in enumerate(cands, 1):
            tieu_de = metas.get(c.video_id).title[:44] if c.video_id in metas else ""
            print(f"  {i}. {c.video_id}  frame {c.event.keyframe_idx:<7d} "
                  f"t={c.event.t_start:7.1f}s  điểm={c.score:.4f}")
            print(f"     {tieu_de}")
        print()
    print("Nhìn bằng mắt: các video trả về có liên quan tới câu hỏi không?")
    print("Nếu không, vấn đề nằm ở CHẤT LƯỢNG TÌM KIẾM, không phải ở cách xếp slot.")


# --------------------------------------------------------------------------- #
def che_do_cham_diem(events, emb, metas, enc, vi: bool = False, dung=None):
    queries = load_queries(DEV_JSON, truong_text=truong_text_dev_set(vi))
    print(f"\nDev set: {len(queries)} câu"
          + ("  (truy vấn TIẾNG VIỆT, không qua lớp dịch)" if vi else ""))

    thieu = [q.text for q, _ in queries
             if CLIPTextEncoder._khoa(q.text) not in enc._cache]
    if thieu and not enc._lay_live():
        print(f"\n❌ {len(thieu)}/{len(queries)} câu chưa có vector trong cache.")
        print("   Encode chúng trên Kaggle rồi tải lại query_vectors.npz.")
        print(f"   Ví dụ câu thiếu: {thieu[0][:60]!r}")
        sys.exit(1)

    cau_hinh = [
        ("core/     (heuristic)", PortfolioConfig(enabled=False)),
        ("+optimize (portfolio)", PortfolioConfig(enabled=True,
                                                  calibrator=RankCalibrator())),
    ]
    ket_qua = {}
    for ten, cfg in cau_hinh:
        p = dung(cfg) if dung is not None else dung_pipeline(
            events, emb, enc, portfolio=cfg)
        t = time.time()
        diem, chi_tiet = [], []
        for q, gt in queries:
            rows = p.answer(q, top_k=100)
            fs = scoring.final_score(rows, gt)
            diem.append(fs)
            chi_tiet.append((q.qid, q.text, fs,
                             scoring.recall_at_k(rows, gt, 1, threshold=1.0),
                             scoring.recall_at_k(rows, gt, 100, threshold=1.0)))
        ket_qua[ten] = (sum(diem) / len(diem), chi_tiet, time.time() - t)
        print(f"  {ten}: {ket_qua[ten][0]:.4f}   ({ket_qua[ten][2]:.1f}s)")

    print("\n" + "=" * 74)
    print("  ĐIỂM CHÍNH THỨC (Mean of Top-k R-Scores) TRÊN DỮ LIỆU THẬT")
    print("=" * 74)
    for ten, (tb, _, _) in ket_qua.items():
        print(f"  {ten:24s} {tb:.4f}")

    # Tách theo LOẠI CÂU. Dev set có cả kis/qa/trake nên con số gộp KHÔNG so được
    # với các mốc lịch sử đo trên dev set chỉ có KIS — và ba loại này ở ba mức
    # hoàn hảo khác nhau (QA còn cần một VLM mới có điểm).
    task_cua = {q.qid: q.task for q, _ in queries}
    for ten, (_, ct, _) in ket_qua.items():
        theo = {}
        for qid, _, fs, _, _ in ct:
            theo.setdefault(task_cua[qid], []).append(fs)
        if len(theo) > 1:
            phan = "  ·  ".join(
                f"{t}: {sum(v)/len(v):.4f} ({len(v)} câu)"
                for t, v in sorted(theo.items()))
            print(f"    └ {ten.split()[0]:9s} {phan}")
    if len(set(task_cua.values())) > 1:
        print("\n  ⚠ Con số gộp ở trên trộn cả ba loại câu. Mốc lịch sử (CLIP 0.1867,")
        print("    SigLIP2 0.2950) đo trên dev set CHỈ CÓ KIS — chỉ so với dòng `kis`.")
        print("    QA/TRAKE đo riêng bằng: python do_diem_qa_trake.py")
    ten_a, ten_b = [t for t, _ in cau_hinh]
    delta, paired_se = paired_delta_se(
        ket_qua[ten_a][1], ket_qua[ten_b][1])
    print(f"  {'delta ± paired SE':24s} {delta:+.4f} ± {paired_se:.4f}")
    print()
    print("  -> " + optimizer_recommendation(delta, paired_se))

    # Chẩn đoán: hỏng ở khâu tìm hay khâu xếp slot?
    _, ct, _ = ket_qua[ten_a]
    r1 = sum(c[3] for c in ct) / len(ct)
    r100 = sum(c[4] for c in ct) / len(ct)
    print(f"\n  Recall@1   : {r1:.1%}   (đáp án đúng nằm ngay dòng đầu)")
    print(f"  Recall@100 : {r100:.1%}   (đáp án đúng lọt vào 100 dòng)")
    if r100 < 0.3:
        print("\n  ⚠ Recall@100 thấp -> vấn đề ở TẦNG TÌM. Xếp slot khéo mấy cũng vô ích")
        print("    khi đáp án còn không lọt nổi vào 100 dòng. Ưu tiên: đổi encoder,")
        print("    thêm kênh chữ từ metadata, dịch truy vấn sang tiếng Anh.")
    elif r1 / max(r100, 1e-9) < 0.5:
        print("\n  ⚠ Lọt top-100 nhiều nhưng ít khi lên được hạng 1 -> đây ĐÚNG LÀ")
        print("    chỗ tầng optimize ăn tiền. Fit tol_frames rồi chạy lại.")

    print("\n  5 câu tệ nhất:")
    for qid, text, fs, _, _ in sorted(ct, key=lambda c: c[2])[:5]:
        print(f"    {fs:.2f}  {qid}  {text[:52]}")
    print("\n  5 câu tốt nhất:")
    for qid, text, fs, _, _ in sorted(ct, key=lambda c: -c[2])[:5]:
        print(f"    {fs:.2f}  {qid}  {text[:52]}")


def duong_config(argv):
    """`--config` / `--config=<file>` -> đo ĐÚNG cấu hình thi, không phải mặc định
    riêng của script này.

    Có để khép chỗ lệch đã ghi trong 1.5: script này mặc định SigLIP2 **base** còn
    cấu hình thi là **so400m** (và từ 16/08 còn bật hợp nhất nhiều encoder). Giữ
    mặc định cũ để mọi số lịch sử còn so được, nhưng `--config` cho phép đo đúng
    thứ sẽ mang đi thi.
    """
    if "--config" not in argv and not any(a.startswith("--config=") for a in argv):
        return None
    return next((a.split("=", 1)[1] for a in argv if a.startswith("--config=")),
                None)


def cham_diem_theo_config(path, vi: bool = False):
    """Đo ĐÚNG cấu hình thi. `vi=True` -> truy vấn tiếng Việt (`--config --vi`).

    Trước 2026-08-17 hàm này không nhận `vi`, nên `--config --vi` chạy êm và trả
    về số TIẾNG ANH — đúng kiểu hỏng tệ nhất: một phép đo trả lời sai câu hỏi mà
    người đo tưởng mình đang hỏi.
    """
    from aic2026 import config as CFG
    from aic2026.core.providers import RuleBasedProvider

    c = CFG.load_config(path)
    print("Đo theo CẤU HÌNH THI:" + ("  [TRUY VẤN TIẾNG VIỆT]" if vi else ""))
    for dong in CFG.mo_ta_cau_hinh(c):
        print("  " + dong)
    for dong in CFG.canh_bao_cau_hinh(c):
        print(f"  ⚠ {dong}")
    print()
    events, emb, metas, rep = CFG.nap_du_lieu(c)
    print(f"  {rep.events:,d} event / {rep.videos_da_nap} video")
    phu = CFG.nap_hop_nhat(c, events)
    if phu:
        print(f"  hợp nhất: {len(phu)} encoder phụ "
              f"(trọng số chính {CFG.chinh_sach_truy_hoi(c).fuse_trong_so_chinh:g})")
    enc = CFG.dung_encoder(c)
    print(f"  encoder: {type(enc).__name__} {getattr(enc, 'MODEL_ID', '')} "
          f"({enc.dim} chiều) · cache {len(enc._cache)} câu")

    def dung(portfolio):
        return CFG.dung_pipeline(c, events, emb, encoder=enc,
                                 llm=RuleBasedProvider(), portfolio=portfolio,
                                 retrieval_phu=phu)

    che_do_cham_diem(events, emb, metas, enc, vi=vi, dung=dung)


def main():
    thu = "--thu" in sys.argv
    vi = "--vi" in sys.argv

    path_config = duong_config(sys.argv)
    if path_config is not None or "--config" in sys.argv:
        cham_diem_theo_config(path_config, vi=vi)
        return

    # SigLIP2 base là MẶC ĐỊNH kể từ 2026-08-12 (+0.1083 ± 0.0313 so với CLIP,
    # tức 3.5·SE — xem docstring SigLIP2TextEncoder). `--clip` để về đường cũ.
    siglip2 = "" if "--clip" in sys.argv else "base"
    for a in sys.argv[1:]:
        if a == "--siglip2":
            siglip2 = "base"
        elif a.startswith("--siglip2="):
            siglip2 = a.split("=", 1)[1]

    if vi and not siglip2:
        # CLIP gốc + tiếng Việt đã đo là XẾP NGẪU NHIÊN (hạng trung vị
        # 72.620/177.321, R@100 = 0.0%) — xem docstring MCLIPTextEncoder. Chạy
        # lại phép đó chỉ tốn thời gian để in ra 0.0000.
        print("❌ `--vi` chỉ dùng cùng `--siglip2`.")
        print("   CLIP ViT-B/32 + tiếng Việt đã đo là xếp ngẫu nhiên, đo lại vô ích.")
        sys.exit(1)

    kiem_tra_dieu_kien(can_dev_set=not thu, siglip2=siglip2, vi=vi)
    events, emb, metas, enc = nap_he_thong(siglip2=siglip2, vi=vi)
    if thu:
        che_do_thu(events, emb, metas, enc)
    else:
        che_do_cham_diem(events, emb, metas, enc, vi=vi)


if __name__ == "__main__":
    main()
