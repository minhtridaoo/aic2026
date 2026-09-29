#!/usr/bin/env python
"""SĂN BẰNG CHỨNG — tra OCR + ASR cho những câu tầng ảnh còn mò.

    python -X utf8 san_bang_chung.py --config config_ocr_asr.yaml
    python -X utf8 san_bang_chung.py --cau query-p1-1-kis --tu "cực quang,phi hành gia"

VÌ SAO
------
Cách này đã ăn +1,4 điểm thật trên leaderboard (6,0 -> 7,4): với đề dạng bản
tin, OCR/ASR thường đọc ra NGUYÊN VĂN thứ đề hỏi. Nhưng nó chỉ chạy tự động khi
truy vấn và chữ trên màn hình dùng chung từ ngữ. Khi đề nói vòng ("phim của
Spielberg năm 1975") thì phải người gõ đúng từ khoá vào ("cá mập").

Script này in ra, cho mỗi câu: top OCR, top ASR, kèm video + tiêu đề + khung —
đủ để mắt người quyết trong vài giây rồi ghim.

Không tự ghim. Ghim là quyết định của người, và mỗi ghim phải có chữ đọc được
kèm theo trong `ghi_chu` để 21 giờ tối còn kiểm lại được.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026 import config as cau_hinh
from aic2026.core import dataset

GOC = os.path.dirname(os.path.abspath(__file__))

# Từ khoá NGƯỜI thêm cho những câu đề nói vòng. Đây là chỗ tri thức ngoài đi
# vào hệ thống — encoder ảnh và OCR đều không tự suy ra được.
TU_KHOA_THEM = {
    "query-p1-1-kis":  ["phóng tàu vũ trụ tư nhân phi hành gia",
                        "cực quang vùng cực", "spacex crew dragon polaris"],
    "query-p1-7-kis":  ["chim rừng Nam Bộ mắt đỏ", "loài chim quý"],
    "query-p1-13-kis": ["vệ sinh máy ảnh ống kính", "nhiếp ảnh máy ảnh"],
    "query-p1-24-kis": ["cuộc đua xe đạp ba tay đua cùng đội",
                        "cúp truyền hình chặng đua"],
    "query-p1-25-kis": ["flycam đua xe đạp vượt lên dẫn đầu",
                        "về đích dẫn đầu chặng"],
    "query-p1-17-kis": ["trao kinh phí hỗ trợ trẻ em mồ côi COVID-19",
                        "bệnh viện trao quà xuân 2024"],
    "query-p1-12-kis": ["bánh rán chocolate chuối dâu tây",
                        "trang trí bánh rán"],
    "query-p1-16-trake": ["múa lân cột mai hoa thung", "lân sư rồng chào rồng"],
    "query-p1-18-trake": ["nấu ăn nấm củ năng đậu hũ", "cắt nấm sơ chế"],
}


def _args():
    p = argparse.ArgumentParser(description="Tra OCR/ASR cho câu còn nghi ngờ")
    p.add_argument("--config", default=os.path.join(GOC, "config_ocr_asr.yaml"))
    p.add_argument("--de", default=os.path.join(GOC, "data", "de_thunghiem.json"))
    p.add_argument("--cau", default="", help="chỉ tra các qid này (ngăn bằng phẩy)")
    p.add_argument("--tu", default="", help="từ khoá tự gõ, ngăn bằng phẩy")
    # `TU_KHOA_THEM` viết cứng trong file này là bộ từ khoá của VÒNG THỬ NGHIỆM.
    # Mỗi vòng đề khác nhau nên phải nạp được từ ngoài, nếu không thì hoặc sửa mã
    # nguồn giữa giờ thi (đã suýt làm) hoặc mất hẳn kênh tri thức ngoài.
    p.add_argument("--tu-json", default="",
                   help="file JSON {qid: [từ khoá, ...]} thay cho TU_KHOA_THEM")
    p.add_argument("--sau", type=int, default=4)
    return p.parse_args()


def main():
    a = _args()
    config = cau_hinh.load_config(a.config)
    kenh = cau_hinh.nap_kenh_chu(config)
    if not kenh:
        raise SystemExit("Chưa bật kênh chữ — xem `retrieval.kenh_chu`")

    print("Nạp map-keyframes ...")
    paths = dataset.scan(config["offline"]["btc_data_root"])
    moc, tong, tieu = [], 0, {}
    for vid in sorted(paths):
        vp = paths[vid]
        if not vp.du_toi_thieu:
            continue
        rows = dataset.load_map_keyframes(vp.mapping)
        moc.append((tong, tong + len(rows), vid, rows))
        tong += len(rows)
        if vp.meta is not None:
            try:
                tieu[vid] = dataset.load_media_info(vp.meta, vid).title
            except Exception:                                  # noqa: BLE001
                pass
    events = [type("E", (), {"video_id": vid, "event_id": f"{vid}#{r.n:05d}",
                             "t_start": float(r.pts_time)})()
              for _, _, vid, rows in moc for r in rows]
    for kho, _ in kenh:
        kho.gan_event(events)
    print(f"  {tong:,d} keyframe · {len(kenh)} kênh chữ\n")

    def tra_frame_row(fr):
        for x, y, vid, rows in moc:
            if x <= fr < y:
                r = rows[fr - x]
                return vid, r.n, r.frame_idx
        return None, None, None

    def tra_event(eid):
        vid = eid.split("#")[0]
        n = int(eid.split("#")[1])
        for x, y, v, rows in moc:
            if v == vid:
                r = next((z for z in rows if z.n == n), None)
                return vid, n, (r.frame_idx if r else None)
        return vid, n, None

    with open(a.de, encoding="utf-8") as fh:
        de = {q["qid"]: q for q in json.load(fh)["queries"]}

    them = TU_KHOA_THEM
    if a.tu_json:
        with open(a.tu_json, encoding="utf-8") as fh:
            them = {k: v for k, v in json.load(fh).items()
                    if not k.startswith("_")}
        la = sorted(set(them) - set(de))
        if la:
            raise SystemExit(f"{a.tu_json}: qid không có trong đề: {la}")

    if a.tu:
        muc = {"(tự gõ)": [x.strip() for x in a.tu.split(",") if x.strip()]}
    else:
        chi = {x.strip() for x in a.cau.split(",") if x.strip()}
        muc = {q: [de[q].get("text", "")] + them.get(q, [])
               for q in (chi or them) if q in de}

    for qid, cau_hoi in muc.items():
        print("=" * 96)
        print(f"▸ {qid}")
        if qid in de:
            print(f"  ĐỀ: {de[qid].get('text', '')[:150]}")
        for t in cau_hoi:
            print(f"\n  ── tra: {t[:78]}")
            for kho, _w in kenh:
                ten = type(kho).__name__
                if ten == "KhoOcrFts":
                    kq = kho.cham(t)[:a.sau]
                    for fr, s in kq:
                        vid, n, fi = tra_frame_row(fr)
                        print(f"     OCR {s:.3f} {vid} kf={n} frame={fi}"
                              f"  {tieu.get(vid, '')[:40]}")
                        print(f"              {kho.van_ban(fr)[:98]}")
                    if not kq:
                        print("     OCR (không khớp)")
                else:
                    kq = kho.cham_khoang(t)[:a.sau]
                    for i, s in kq:
                        k = kho.khoang[i]
                        print(f"     ASR {s:.3f} {k['video_id']} "
                              f"{float(k['start_pts_time_technical']):.0f}s"
                              f"  {tieu.get(k['video_id'], '')[:40]}")
                        print(f"              "
                              f"{str(k.get('asr_text_compact'))[:98]}")
                    if not kq:
                        print("     ASR (không khớp)")


if __name__ == "__main__":
    main()
