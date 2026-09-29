#!/usr/bin/env python
"""Biến file nhãn bạn gõ tay thành dev set chấm điểm được.

    python tao_dev_set.py [thư_mục_dev_set] [thư_mục_data]

ĐẦU VÀO   dev_set/dan_nhan.csv  — cột `mo_ta` do bạn điền
ĐẦU RA    data/dev_set_tu_lam.json      — query + đáp án, đọc bằng load_queries()
          dev_set/truy_van_can_encode.txt — danh sách câu cần encode trên Kaggle

CỬA SỔ ĐÁP ÁN ĐƯỢC ĐỊNH NGHĨA THẾ NÀO
-------------------------------------
BTC không công bố cửa sổ chấm của họ, nên dev set này tự định nghĩa: một câu trả
lời được tính ĐÚNG nếu khung nộp lên nằm trong đoạn mà keyframe đã gán nhãn đại
diện, tức [frame_idx của keyframe này, frame_idx của keyframe kế tiếp − 1].

Đó là định nghĩa rộng rãi và có thể giải thích được: bạn mô tả cảnh ở keyframe
này, thì trả về bất kỳ khung nào trong cùng đoạn đó là đã tìm đúng chỗ. Cửa sổ
lấy từ chính file mapping của BTC nên không phải số bịa.

⚠ Điểm đo trên dev set này KHÔNG phải điểm thi. Nó dùng để SO SÁNH giữa các
phiên bản hệ thống của chính bạn — cái đó mới là thứ cần để tối ưu.
"""
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026.core.dataset import load_map_keyframes, scan

GOC = os.path.dirname(os.path.abspath(__file__))
DEV_SET = os.path.join(GOC, "dev_set")
DATA = r"D:\COMP\AIC2026\data cuộc thi\aic2026_data_nhe"
RA_JSON = os.path.join(GOC, "data", "dev_set_tu_lam.json")


def doc_qa(dev, cua_so):
    """VIỆC 4.1 — `dev_set/dan_nhan_qa.csv` -> query task=qa.

    Cửa sổ đáp án lấy CÙNG luật với KIS (keyframe nào thì đoạn đó), vì câu hỏi QA
    được đặt trên đúng keyframe đó. Đáp án giữ NGUYÊN VĂN — `scoring.answers_match`
    tự chuẩn hoá (NFC, hạ chữ, gộp khoảng trắng); sửa tay ở đây sẽ làm nhãn khác
    với thứ người gán nhìn thấy.
    """
    p = os.path.join(dev, "dan_nhan_qa.csv")
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    # Mô tả cảnh của ĐÚNG keyframe đó, lấy từ nhãn KIS. Truy vấn QA = "<mô tả
    # cảnh>, <câu hỏi>" chứ KHÔNG phải câu hỏi trơ.
    #
    # VÌ SAO: đề thi thật hỏi "thương hiệu nào trên container ĐÓ?" — chữ "đó" đòi
    # một cảnh đi trước. Đo 2026-08-12 trên câu hỏi trơ: trần định vị 0.0667 so
    # với 0.1733 khi truy hồi bằng câu mô tả cùng keyframe (chênh +0.1067 ±
    # 0.0523, vượt 2·SE). Câu hỏi tả một CHI TIẾT, còn encoder xếp hạng theo TOÀN
    # CẢNH — nên câu hỏi trơ là truy vấn nghèo thông tin, và cũng không giống đề
    # thi. Dấu phẩy là chỗ `query_compiler._split_qa` tách cảnh khỏi câu hỏi.
    canh = {}
    p_kis = os.path.join(dev, "dan_nhan.csv")
    if os.path.exists(p_kis):
        with open(p_kis, "r", encoding="utf-8-sig", newline="") as fh:
            for k in csv.DictReader(fh):
                canh[(k["video_id"].upper(), int(k["n"]))] = (
                    (k.get("mo_ta_en") or "").strip(),
                    (k.get("mo_ta") or "").strip())

    def ghep(mo_ta, hoi):
        if not mo_ta:
            return hoi
        return f"{mo_ta.rstrip(' .')}, {hoi}"

    ra, thieu, khong_canh = [], 0, 0
    for r in rows:
        hoi_en = (r.get("cau_hoi_en") or "").strip()
        hoi_vi = (r.get("cau_hoi") or "").strip()
        dap = (r.get("dap_an") or "").strip()
        if not (dap and (hoi_en or hoi_vi)):
            thieu += 1
            continue
        vid = r["video_id"].upper()
        n = int(r["n"])
        cs = cua_so(vid, n)
        if cs is None:
            thieu += 1
            continue
        ta_en, ta_vi = canh.get((vid, n), ("", ""))
        if not ta_en and not ta_vi:
            khong_canh += 1
        ra.append({
            "qid": f"qa-{len(ra) + 1:03d}",
            "task": "qa",
            "text": ghep(ta_en or ta_vi, hoi_en or hoi_vi),
            "_mo_ta_vi": ghep(ta_vi, hoi_vi or hoi_en),
            "_cau_hoi": hoi_en or hoi_vi,      # giữ riêng để soi lại khi cần
            "_cau_hoi_vi": hoi_vi or hoi_en,   # prompt VLM; không dùng để retrieval
            "gt": {"video_id": vid, "frame_start": cs[0], "frame_end": cs[1],
                   "answer": dap,
                   # Cột `dap_an_alt` (tuỳ chọn, ngăn bằng `;`) — các cách diễn
                   # đạt KHÁC cũng tính đúng. Chỗ này để ghi bản dịch: đáp án
                   # chuẩn là tiếng Việt, nhưng VLM tiếng Anh trả "green" cho
                   # "xanh lá" thì về nghĩa vẫn đúng. Xem GroundTruth.answer_alt.
                   "answer_alt": [x.strip() for x in
                                  (r.get("dap_an_alt") or "").split(";")
                                  if x.strip()]},
            "_ghi_chu": {"keyframe_n": n, "anh": r.get("anh", "")},
        })
    if khong_canh:
        print(f"     ⚠ {khong_canh} câu QA không tìm được mô tả cảnh trong "
              "dan_nhan.csv — dùng câu hỏi trơ, định vị sẽ kém hơn")
    print(f"  QA: {len(ra)} câu" + (f"  (bỏ {thieu} dòng thiếu ô)" if thieu else ""))
    return ra


# Tách mô tả chuỗi thành các bước. Cùng bộ dấu với `RuleBasedProvider.parse_query`
# để nhãn và runtime hiểu câu giống nhau.
_TACH_MOC = re.compile(r"\s*(?:->|→|;|,\s*(?:sau đó|rồi|tiếp theo)\s+|\s+(?:sau đó|rồi|tiếp theo)\s+)\s*")


def doc_trake(dev, cua_so):
    """VIỆC 4.2 — `dev_set/dan_nhan_trake.csv` -> query task=trake.

    `event_windows[j]` PHẢI ứng với mốc thứ j, đúng thứ tự người gán đã bấm —
    `scoring.r_score_trake` so `frames[j]` với `event_windows[j]` theo chỉ số.
    Đảo thứ tự là nhãn sai mà không có triệu chứng gì.

    Số bước trong mô tả phải KHỚP số mốc đã chọn. Lệch thì bỏ dòng: 3 mốc mà mô tả
    chỉ tách được 2 bước nghĩa là hệ thống sẽ tìm 2 mốc rồi bị chấm trên 3, tức đo
    ra điểm thấp vì lỗi nhãn chứ không vì hệ thống kém.
    """
    p = os.path.join(dev, "dan_nhan_trake.csv")
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    ra, bo = [], []
    for r in rows:
        vid = r["video_id"].upper()
        ns = [int(x) for x in (r.get("cac_n") or "").split()]
        en = (r.get("mo_ta_en") or "").strip()
        vi = (r.get("mo_ta") or "").strip()
        if not ns or not (en or vi):
            bo.append((vid, "thiếu mốc hoặc mô tả"))
            continue

        buoc = [b.strip() for b in _TACH_MOC.split(en or vi) if b.strip()]
        if len(buoc) != len(ns):
            bo.append((vid, f"{len(ns)} mốc nhưng mô tả tách được {len(buoc)} bước"))
            continue

        cs = [cua_so(vid, n) for n in ns]
        if any(c is None for c in cs):
            bo.append((vid, "không tra được cửa sổ cho một mốc"))
            continue
        if ns != sorted(ns):
            # Mốc phải tăng dần theo thời gian. Người gán bấm sai thứ tự thì mô tả
            # và cửa sổ lệch nhau, và điểm từng phần sẽ đo sai chứ không báo lỗi.
            bo.append((vid, f"mốc không tăng dần: {ns}"))
            continue

        ra.append({
            "qid": f"trake-{len(ra) + 1:03d}",
            "task": "trake",
            "sub_events": [{"text": b} for b in buoc],
            "_mo_ta_vi": vi,
            "gt": {"video_id": vid,
                   "event_windows": [[c[0], c[1]] for c in cs]},
            "_ghi_chu": {"cac_n": ns},
        })

    print(f"  TRAKE: {len(ra)} chuỗi" + (f"  (bỏ {len(bo)} dòng)" if bo else ""))
    for vid, ly_do in bo[:5]:
        print(f"     ⚠ {vid}: {ly_do}")
    return ra


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else DEV_SET
    data = sys.argv[2] if len(sys.argv) > 2 else DATA
    csv_path = os.path.join(dev, "dan_nhan.csv")

    if not os.path.exists(csv_path):
        raise SystemExit(
            f"Không thấy {csv_path}\n"
            "  Chạy kaggle_chuan_bi_dev_set.py trên Kaggle trước, tải zip về,\n"
            f"  giải nén vào {dev}")

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    print(f"Đọc {len(rows)} dòng từ dan_nhan.csv")

    def lay_text(r):
        """Ưu tiên bản tiếng Anh.

        Đo 2026-08-08 trên chính 60 câu này, đối chiếu 177.321 keyframe thật —
        hạng của keyframe đúng (thấp là tốt; ~88.660 = xếp ngẫu nhiên):

            CLIP + tiếng Việt   hạng 72.620   R@100  0.0%   ❌ ~ngẫu nhiên
            CLIP + tiếng Anh    hạng    283   R@100 28.3%   ✅
            đa ngữ + tiếng Việt hạng  1.183   R@100 18.3%

        OpenAI CLIP huấn luyện trên alt-text tiếng Anh, nó không biết tiếng Việt.
        Dịch sang tiếng Anh vẫn hơn dùng text tower đa ngữ với tiếng Việt, nên
        cột `mo_ta_en` là đường chính chứ không phải phụ.

        (Con số 53.4% / 12.7% ghi ở đây trước ngày 2026-08-08 đã bị bỏ: nó sinh
        ra từ đường CLIP bị project hai lần, xem `CLIPTextEncoder._lay_live`.)
        """
        return (r.get("mo_ta_en") or "").strip() or (r.get("mo_ta") or "").strip()

    co_nhan = [r for r in rows if lay_text(r)]
    co_en = [r for r in rows if (r.get("mo_ta_en") or "").strip()]
    print(f"Đã gán nhãn : {len(co_nhan)}")
    print(f"Còn trống   : {len(rows) - len(co_nhan)}")
    print(f"Có bản tiếng Anh: {len(co_en)}")
    if not co_nhan:
        raise SystemExit("\nChưa có dòng nào được gán nhãn. Mở dan_nhan.csv và "
                         "điền cột `mo_ta` (mô tả tiếng Việt cảnh trong ảnh).")
    if len(co_en) < len(co_nhan):
        print("\n  ⚠️  CẢNH BÁO: còn "
              f"{len(co_nhan) - len(co_en)} dòng chưa có bản tiếng Anh.")
        print("     CLIP ViT-B/32 không hiểu tiếng Việt (đã đo: chồng lấn 53%).")
        print("     Thêm cột `mo_ta_en` và dịch sang tiếng Anh, nếu không điểm")
        print("     đo được sẽ là điểm của nhiễu.")

    # Tra cửa sổ đáp án từ file mapping gốc của BTC.
    print("\nĐang tra cửa sổ đáp án từ map-keyframes ...")
    paths = scan(data)
    cache = {}
    queries, bo_qua = [], []
    for r in co_nhan:
        vid = r["video_id"].upper()
        if vid not in paths or paths[vid].mapping is None:
            bo_qua.append((vid, "không thấy file mapping"))
            continue
        if vid not in cache:
            cache[vid] = load_map_keyframes(paths[vid].mapping)
        kf = cache[vid]
        n = int(r["n"])
        i = next((j for j, k in enumerate(kf) if k.n == n), None)
        if i is None:
            bo_qua.append((vid, f"không thấy keyframe n={n}"))
            continue
        dau = kf[i].frame_idx
        cuoi = (kf[i + 1].frame_idx - 1) if i + 1 < len(kf) else dau + int(kf[i].fps)
        queries.append({
            "qid": f"kis-{int(r['stt']):03d}",
            "task": "kis",
            "text": lay_text(r),
            "_mo_ta_vi": (r.get("mo_ta") or "").strip(),
            "gt": {"video_id": vid, "frame_start": dau, "frame_end": max(cuoi, dau)},
            "_ghi_chu": {"keyframe_n": n, "pts_time": float(r["pts_time"]),
                         "anh": r.get("anh", "")},
        })

    if bo_qua:
        print(f"  ⚠ bỏ qua {len(bo_qua)} dòng:")
        for vid, ly_do in bo_qua[:5]:
            print(f"     {vid}: {ly_do}")

    def cua_so(vid, n):
        """Cửa sổ đáp án của keyframe n trong video — cùng luật với KIS ở trên."""
        if vid not in paths or paths[vid].mapping is None:
            return None
        if vid not in cache:
            cache[vid] = load_map_keyframes(paths[vid].mapping)
        kf = cache[vid]
        i = next((j for j, k in enumerate(kf) if k.n == n), None)
        if i is None:
            return None
        dau = kf[i].frame_idx
        cuoi = (kf[i + 1].frame_idx - 1) if i + 1 < len(kf) else dau + int(kf[i].fps)
        return dau, max(cuoi, dau)

    queries += doc_qa(dev, cua_so)
    queries += doc_trake(dev, cua_so)

    os.makedirs(os.path.dirname(RA_JSON), exist_ok=True)
    with open(RA_JSON, "w", encoding="utf-8") as fh:
        json.dump({
            "_note": "Dev set TỰ LÀM (known-target). Cửa sổ đáp án = đoạn mà "
                     "keyframe đã gán nhãn đại diện, lấy từ map-keyframes của BTC. "
                     "KHÔNG phải bộ đề chính thức; dùng để so sánh giữa các phiên "
                     "bản hệ thống của chính mình.",
            "queries": queries,
        }, fh, ensure_ascii=False, indent=2)

    theo_task = {}
    for q in queries:
        theo_task[q["task"]] = theo_task.get(q["task"], 0) + 1
    print(f"\n✅ Ghi {len(queries)} câu -> {RA_JSON}")
    print("   " + " · ".join(f"{t}: {n}" for t, n in sorted(theo_task.items())))

    # TRAKE không có `frame_start`/`frame_end` — nó có `event_windows`. Gộp bừa
    # hai hình dạng GT vào một phép thống kê là cách chắc chắn để nổ KeyError,
    # hoặc tệ hơn là in ra một con số không có nghĩa.
    rong = sorted(q["gt"]["frame_end"] - q["gt"]["frame_start"]
                  for q in queries if "frame_end" in q["gt"])
    if rong:
        print(f"   độ rộng cửa sổ đáp án (kis+qa): trung vị {rong[len(rong)//2]} "
              f"khung (nhỏ nhất {rong[0]}, lớn nhất {rong[-1]})")
    rong_tk = sorted(e - s for q in queries
                     for s, e in q["gt"].get("event_windows", []))
    if rong_tk:
        print(f"   độ rộng cửa sổ mỗi mốc (trake) : trung vị "
              f"{rong_tk[len(rong_tk)//2]} khung")

    # --- danh sách câu cần encode ------------------------------------------- #
    # TRAKE encode TỪNG BƯỚC, không encode cả câu: `MultiChannelRetriever` truy hồi
    # riêng cho mỗi sub-event. Thiếu một bước trong cache là `CLIPTextEncoder` ném
    # LookupError giữa lúc chấm điểm.
    def cac_text(q, vi=False):
        khoa = "_mo_ta_vi" if vi else "text"
        if q["task"] == "trake":
            if vi:
                return [b.strip() for b in _TACH_MOC.split(q.get("_mo_ta_vi") or "")
                        if b.strip()]
            return [s["text"] for s in q["sub_events"]]
        t = (q.get(khoa) or "").strip()
        return [t] if t else []

    def ghi_danh_sach(ten, vi):
        seen, ds = set(), []
        for q in queries:
            for t in cac_text(q, vi=vi):
                k = " ".join(t.split())     # cùng cách chuẩn hoá với _khoa của cache
                if k and k not in seen:
                    seen.add(k)
                    ds.append(t)
        p = os.path.join(dev, ten)
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("\n".join(ds) + "\n")
        return p, len(ds)

    txt, n_txt = ghi_danh_sach("truy_van_can_encode.txt", vi=False)
    print(f"\n   Bước kế: encode {n_txt} chuỗi này thành vector "
          f"({len(queries)} câu, TRAKE tách theo từng bước).")
    print(f"   -> {txt}")
    print("   Ở máy đã có transformers thì chạy thẳng:")
    print("     python kaggle_encode_truy_van_siglip2.py")
    print("   (CLIP cũ: kaggle_encode_truy_van.py)")

    # Bản TIẾNG VIỆT — cho việc 2.6 (`do_diem_thuc_te.py --siglip2 --vi`).
    # Chỉ SigLIP2 dùng được; CLIP gốc + tiếng Việt đã đo là xếp ngẫu nhiên
    # (hạng trung vị 72.620/177.321), xem docstring MCLIPTextEncoder.
    thieu_vi = [q["qid"] for q in queries if not cac_text(q, vi=True)]
    if thieu_vi:
        print(f"\n   ⚠ {len(thieu_vi)} câu chưa có mô tả tiếng Việt "
              f"(vd {', '.join(thieu_vi[:3])}), chưa xuất được bản tiếng Việt.")
    else:
        txt_vi, n_vi = ghi_danh_sach("truy_van_vi_can_encode.txt", vi=True)
        print(f"\n   Bản tiếng Việt ({n_vi} chuỗi, cho việc 2.6): {txt_vi}")
        print("     python kaggle_encode_truy_van_siglip2.py --vi")


if __name__ == "__main__":
    main()
