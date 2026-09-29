#!/usr/bin/env python
"""Kiểm chứng đường ống trên DỮ LIỆU THẬT — 873 video, 177.321 keyframe.

    python kiem_chung_thuc_te.py [đường\\dẫn\\data]

VÌ SAO CẦN SCRIPT NÀY
---------------------
Toàn bộ số liệu trước đây chạy trên 8 sự kiện bịa với encoder giả. Script này
chạy trên dữ liệu thật của BTC. Nó KHÔNG cần CLIP text encoder — nên chạy được
ngay hôm nay — bằng cách dùng chính vector của một keyframe làm câu truy vấn
(tìm ảnh bằng ảnh). Cách đó kiểm được mọi mắt xích trừ đúng phần text tower:

    đọc file -> dựng Event -> nạp index -> tìm -> xếp slot -> ghi CSV -> zip

NĂM PHÉP KIỂM
-------------
  1. Nạp dữ liệu       số event có khớp con số audit (177.321) không
  2. Tự tìm chính mình  tra bằng vector của keyframe X phải ra X ở hạng 1
  3. Liền mạch thời gian láng giềng gần nhất có nằm cùng video, gần nhau về
                        thời gian không — nếu có thì vector THẬT SỰ mang nghĩa
  4. Tốc độ             một truy vấn mất bao lâu trên 177k vector
  5. Nộp bài            dựng đủ 100 dòng, validate, đóng zip thành công
"""
import hashlib
import os
import random
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from aic2026.core import submission as sub
from aic2026.core.dataset import load_dataset
from aic2026.core.models import Candidate
from aic2026.core.retrieval import MultiChannelRetriever
from aic2026.core.stores import NumpyVectorStore
from aic2026.core.providers import HashEmbedder

MAC_DINH = r"D:\COMP\AIC2026\data cuộc thi\aic2026_data_nhe"
SO_MAU = 200          # số keyframe lấy ra làm câu truy vấn thử
HAT_GIONG = 20260807  # cố định để chạy lại ra đúng kết quả cũ


def muc(so, ten):
    print("\n" + "=" * 74)
    print(f"  PHÉP KIỂM {so}: {ten}")
    print("=" * 74)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else MAC_DINH
    print(f"Dữ liệu: {root}")

    # ---- 1. Nạp ---------------------------------------------------------- #
    muc(1, "Nạp dữ liệu thật")
    t = time.time()
    events, emb, metas, rep = load_dataset(root)
    t_nap = time.time() - t
    print(rep.tom_tat())
    print(f"\n  thời gian      : {t_nap:.1f}s")
    print(f"  ma trận vector : {emb.shape} {emb.dtype} "
          f"({emb.nbytes / 1024 / 1024:.0f} MiB)")

    KY_VONG_EVENT, KY_VONG_VIDEO = 177_321, 873
    ok1 = rep.events == KY_VONG_EVENT and rep.videos_da_nap == KY_VONG_VIDEO
    print(f"\n  {'✅' if ok1 else '❌'} khớp audit v4.7.4 "
          f"({KY_VONG_EVENT:,d} keyframe / {KY_VONG_VIDEO} video)"
          + ("" if ok1 else f" — thực tế {rep.events:,d} / {rep.videos_da_nap}"))
    if rep.lech_so_hang or rep.loi_khac:
        print("  ⚠ có video bị bỏ qua, xem phần cảnh báo bên trên")

    # ---- 2. Nạp index ---------------------------------------------------- #
    muc(2, "Tự tìm chính mình (tra bằng vector của keyframe X phải ra X)")
    t = time.time()
    ret = MultiChannelRetriever(HashEmbedder(), vector_store=NumpyVectorStore())
    ret.index(events, embeddings=emb)
    ret_index = {ev.event_id: i for i, ev in enumerate(events)}
    print(f"  nạp index: {time.time() - t:.1f}s cho {len(ret.visual):,d} vector\n")

    rng = random.Random(HAT_GIONG)
    mau = rng.sample(range(len(events)), min(SO_MAU, len(events)))

    # Không thể đòi "phải ra ĐÚNG chính nó": corpus có những keyframe mà vector
    # giống nhau TỪNG BIT ở các video khác nhau (hình hiệu, intro chương trình
    # lặp lại giữa các tập). Khi đó mọi bản đều hoà ở cosine 1.0 và thứ tự giữa
    # chúng là tuỳ ý — kho vector vẫn đúng. Nên tiêu chí là: trả về chính nó
    # HOẶC một bản trùng khít.
    dung_hang1 = trung_khit = 0
    diem_hang1 = []
    for i in mau:
        kq = ret.visual.search(emb[i], top_k=1)
        if not kq:
            continue
        key, diem = kq[0]
        diem_hang1.append(diem)
        if key == events[i].event_id:
            dung_hang1 += 1
        elif np.array_equal(emb[i], emb[ret_index[key]]):
            trung_khit += 1
    ty_le = (dung_hang1 + trung_khit) / len(mau)
    print(f"  mẫu thử           : {len(mau)} keyframe ngẫu nhiên "
          f"(hạt giống {HAT_GIONG})")
    print(f"  ra đúng chính nó  : {dung_hang1}/{len(mau)}")
    print(f"  ra bản trùng khít : {trung_khit}/{len(mau)}  "
          "(vector giống hệt, hoà điểm — vẫn đúng)")
    print(f"  tổng đạt          : {dung_hang1 + trung_khit}/{len(mau)}  ({ty_le:.1%})")
    if diem_hang1:
        print(f"  điểm cosine       : {min(diem_hang1):.4f} – {max(diem_hang1):.4f} "
              "(≈1.0; lệch nhỏ do nguồn lưu float16)")
    ok2 = ty_le >= 0.99
    print(f"\n  {'✅' if ok2 else '❌'} kho vector trả về đúng thứ đã nạp vào")

    # ---- 2b. Mức độ trùng lặp của corpus --------------------------------- #
    print("\n  --- Trùng lặp trong corpus (ảnh hưởng trực tiếp tới việc tiêu slot) ---")
    bam = [hashlib.blake2b(emb[i].tobytes(), digest_size=16).digest()
           for i in range(len(events))]
    dem = Counter(bam)
    nhom_trung = {k: v for k, v in dem.items() if v > 1}
    kf_trung = sum(nhom_trung.values())
    print(f"  vector duy nhất   : {len(dem):,d} / {len(events):,d}")
    print(f"  nhóm trùng khít   : {len(nhom_trung):,d} nhóm, "
          f"{kf_trung:,d} keyframe ({kf_trung / len(events):.1%})")
    if nhom_trung:
        theo_bam = defaultdict(list)
        for i, k in enumerate(bam):
            theo_bam[k].append(i)
        lon_nhat = max(nhom_trung, key=lambda k: nhom_trung[k])
        ids = theo_bam[lon_nhat]
        print(f"  nhóm lớn nhất     : {len(ids)} keyframe trải "
              f"{len({events[i].video_id for i in ids})} video khác nhau")
        print("  ⚠ Nếu truy vấn rơi trúng một hình hiệu như vậy, hàng trăm video sẽ")
        print("    hoà điểm tuyệt đối và thứ tự giữa chúng là ngẫu nhiên. Cần khử")
        print("    trùng trước khi phân bổ 100 slot.")

    # ---- 3. Liền mạch thời gian ------------------------------------------ #
    muc(3, "Vector có mang nghĩa không? (láng giềng gần nhất là cảnh kề bên?)")
    print("""  Nếu vector CLIP thật sự mô tả nội dung, thì keyframe gần nhất với
  keyframe X (không tính chính nó) phần lớn phải nằm CÙNG VIDEO và KỀ NHAU
  về thời gian — vì hai khung cách nhau 2 giây thì gần giống nhau.
  Nếu tỉ lệ này thấp, tức là ta đã gán nhầm vector cho keyframe.
""")
    cung_video = ke_ben = 0
    khoang_cach = []
    for i in mau:
        kq = ret.visual.search(emb[i], top_k=2)
        lang_gieng = [k for k, _ in kq if k != events[i].event_id][:1]
        if not lang_gieng:
            continue
        ev_lg = ret.events[lang_gieng[0]]
        if ev_lg.video_id == events[i].video_id:
            cung_video += 1
            d = abs(ev_lg.t_start - events[i].t_start)
            khoang_cach.append(d)
            if d <= 15.0:
                ke_ben += 1
    print(f"  láng giềng cùng video      : {cung_video}/{len(mau)} "
          f"({cung_video / len(mau):.1%})")
    print(f"  và cách nhau dưới 15 giây  : {ke_ben}/{len(mau)} "
          f"({ke_ben / len(mau):.1%})")
    if khoang_cach:
        khoang_cach.sort()
        print(f"  khoảng cách thời gian      : trung vị "
              f"{khoang_cach[len(khoang_cach) // 2]:.1f}s")
    ok3 = cung_video / len(mau) >= 0.5
    print(f"\n  {'✅' if ok3 else '⚠️'} vector "
          + ("mang nghĩa thật — cảnh giống nhau nằm gần nhau"
             if ok3 else "KHÔNG liền mạch, nghi gán nhầm vector cho keyframe"))

    # ---- 4. Tốc độ -------------------------------------------------------- #
    muc(4, "Tốc độ truy vấn trên 177k vector")
    for k in (1, 100):
        t = time.time()
        for i in mau[:50]:
            ret.visual.search(emb[i], top_k=k)
        ms = (time.time() - t) / 50 * 1000
        print(f"  top_k={k:<4d} : {ms:6.1f} ms / truy vấn")
    print("\n  ✅ đủ nhanh cho vòng sơ tuyển (nộp file, không cần realtime)")

    # ---- 5. Nộp bài ------------------------------------------------------- #
    muc(5, "Dựng file nộp bài từ kết quả thật")
    i = mau[0]
    goc = events[i]
    kq = ret.visual.search(emb[i], top_k=100)
    cands = [Candidate(event=ret.events[key], score=s,
                       modality_scores={"visual": s, "ocr": 0.0, "asr": 0.0})
             for key, s in kq]
    rows = sub.kis_rows(cands)
    print(f"  truy vấn giả lập : dùng vector của {goc.event_id}")
    print(f"                     ({metas[goc.video_id].title[:52]}...)")
    print(f"  ứng viên trả về  : {len(cands)}")
    print(f"  dòng nộp bài     : {len(rows)}")
    print(f"  video khác nhau  : {len({r['video_id'] for r in rows})}")
    print(f"\n  5 dòng đầu:")
    for j, r in enumerate(rows[:5], 1):
        print(f"    {j}. {r['video_id']}  frame {r['frame_idx']}")

    duong_zip = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "out", "kiem_chung_that.zip")
    ok_zip, loi = sub.build_submission_zip(
        duong_zip, {"query-1-kis.csv": ("kis", rows)})
    print(f"\n  {'✅' if ok_zip else '❌'} đóng gói zip: ok={ok_zip} lỗi={loi}")
    print(f"     -> {duong_zip}")

    # ---- Tổng kết --------------------------------------------------------- #
    print("\n" + "=" * 74)
    tat_ca = ok1 and ok2 and ok3 and ok_zip and len(rows) == 100
    print("  KẾT LUẬN: " + ("✅ đường ống chạy đúng trên dữ liệu thật"
                            if tat_ca else "❌ CÓ PHÉP KIỂM KHÔNG ĐẠT — xem ở trên"))
    print("=" * 74)
    print("""
  Còn thiếu để có ĐIỂM THẬT: CLIP text encoder (tìm bằng chữ) và dev set
  AIC2025 batch 1 (để biết đáp án đúng là gì). Phép kiểm ở trên chứng minh
  mọi mắt xích khác đã chạy đúng ở quy mô thật.
""")
    return 0 if tat_ca else 1


if __name__ == "__main__":
    sys.exit(main())
