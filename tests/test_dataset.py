"""Test loader dữ liệu thật, chạy trên file giả dựng ngay trong tmp_path.

Cố ý KHÔNG đụng tới bộ 179 MB trên đĩa: test phải chạy được trên máy chưa tải
dữ liệu, và phải xong trong tích tắc.
"""
import json

import pytest

np = pytest.importorskip("numpy")

from aic2026.core.dataset import (KeyframeRow, events_cua_video, load_clip_features,
                                  load_dataset, load_map_keyframes, load_media_info,
                                  phan_loai, scan)


# --------------------------------------------------------------------------- #
# Dựng một bộ dữ liệu giả đúng hình dạng thật
# --------------------------------------------------------------------------- #
def dung_video(root, vid, n_kf=4, fps=25.0, n_vector=None, co_meta=True,
               long_sau=True):
    """Tạo 3 file của một video. `n_vector` khác `n_kf` để test phát hiện lệch."""
    if long_sau:                     # bắt chước cây thư mục thật, lồng nhiều cấp
        d_clip = root / "clip_features" / "ds" / "clip-features-32"
        d_map = root / "keyframe_mapping" / "ds" / "map-keyframes"
        d_meta = root / "metadata" / "ds" / "media-info"
    else:
        d_clip = d_map = d_meta = root
    for d in (d_clip, d_map, d_meta):
        d.mkdir(parents=True, exist_ok=True)

    rows = ["n,pts_time,fps,frame_idx"]
    for i in range(1, n_kf + 1):
        pts = (i - 1) * 2.0
        rows.append(f"{i},{pts},{fps},{int(pts * fps)}")
    (d_map / f"{vid}.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    m = np.zeros((n_kf if n_vector is None else n_vector, 512), dtype=np.float16)
    m[:, 0] = 1.0                                    # vector đơn vị đơn giản
    np.save(str(d_clip / f"{vid}.npy"), m)

    if co_meta:
        (d_meta / f"{vid}.json").write_text(json.dumps({
            "title": f"Tiêu đề {vid}", "description": "mô tả dài " * 5,
            "author": "Kênh Thử", "keywords": ["từ khoá a", "từ khoá b"],
            "length": 100, "publish_date": "01/08/2024",
            "watch_url": "https://youtube.com/watch?v=x",
        }, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
def test_phan_loai_khop_luat_audit():
    """Phải hiểu 'file nào là gì' y như audit v4.7.4, nếu không sẽ lấy nhầm."""
    assert phan_loai("clip-features-32/L01_V001.npy", ".npy") == "clip_features"
    assert phan_loai("bat_ky/L01.npy", ".npy") == "npy_unknown"
    assert phan_loai("map-keyframes/L01_V001.csv", ".csv") == "keyframe_mapping"
    assert phan_loai("media-info/L01_V001.json", ".json") == "metadata"
    assert phan_loai("objects/L01_V001/001.json", ".json") == "objects"


def test_scan_tim_duoc_du_ba_thanh_phan_du_long_sau(tmp_path):
    dung_video(tmp_path, "L01_V001")
    dung_video(tmp_path, "L01_V002")
    p = scan(tmp_path)
    assert set(p) == {"L01_V001", "L01_V002"}
    vp = p["L01_V001"]
    assert vp.clip and vp.mapping and vp.meta and vp.du_toi_thieu


def test_scan_bao_thieu_khi_khuyet_file(tmp_path):
    dung_video(tmp_path, "L01_V001", co_meta=False)
    (tmp_path / "clip_features" / "ds" / "clip-features-32" / "L01_V001.npy").unlink()
    vp = scan(tmp_path)["L01_V001"]
    assert vp.clip is None and not vp.du_toi_thieu


def test_scan_bao_loi_ro_khi_thu_muc_khong_ton_tai(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan(tmp_path / "khong-co")


# --------------------------------------------------------------------------- #
def test_doc_map_keyframes(tmp_path):
    dung_video(tmp_path, "L01_V001", n_kf=3, fps=25.0)
    rows = load_map_keyframes(
        tmp_path / "keyframe_mapping" / "ds" / "map-keyframes" / "L01_V001.csv")
    assert [r.n for r in rows] == [1, 2, 3]
    assert rows[1].pts_time == 2.0
    assert rows[1].frame_idx == 50          # 2.0s * 25fps
    assert rows[0].fps == 25.0


def test_map_keyframes_hong_bao_dung_so_dong(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("n,pts_time,fps,frame_idx\n1,0.0,25.0,0\n2,LOI,25.0,50\n",
                 encoding="utf-8")
    with pytest.raises(ValueError, match="dòng 3"):
        load_map_keyframes(p)


def test_doc_media_info_va_ghep_text(tmp_path):
    dung_video(tmp_path, "L01_V001")
    m = load_media_info(
        tmp_path / "metadata" / "ds" / "media-info" / "L01_V001.json", "L01_V001")
    assert m.title == "Tiêu đề L01_V001"
    assert m.keywords == ["từ khoá a", "từ khoá b"]
    # description bị loại theo mặc định: nó toàn link/hashtag lặp giữa các video
    assert "mô tả dài" not in m.text()
    assert "mô tả dài" in m.text(kem_description=True)


# --------------------------------------------------------------------------- #
def test_clip_features_duoc_chuan_hoa_lai(tmp_path):
    """Nguồn float16 làm chuẩn L2 lệch khỏi 1.0 -> phải chuẩn hoá lại lúc nạp.

    Nếu bỏ bước này, cosine của một vector với chính nó bằng |v|² < 1 và có thể
    thua một khung hình gần-giống có |v| > 1.
    """
    d = tmp_path
    m = np.full((3, 512), 0.05, dtype=np.float16)     # chuẩn khác 1 rõ rệt
    np.save(str(d / "a.npy"), m)

    tho = load_clip_features(d / "a.npy", chuan_hoa_lai=False)
    assert not np.allclose(np.linalg.norm(tho, axis=1), 1.0, atol=1e-3)

    sach = load_clip_features(d / "a.npy")
    assert np.allclose(np.linalg.norm(sach, axis=1), 1.0, atol=1e-6)
    assert sach.dtype == np.float32
    assert float(sach[0] @ sach[0]) == pytest.approx(1.0, abs=1e-6)


def test_clip_features_tu_choi_mang_sai_chieu(tmp_path):
    np.save(str(tmp_path / "b.npy"), np.zeros((5,), dtype=np.float16))
    with pytest.raises(ValueError, match=r"mong \(N, D\)"):
        load_clip_features(tmp_path / "b.npy")


# --------------------------------------------------------------------------- #
def test_event_lop_kin_truc_thoi_gian():
    """t_end của event này phải nối vào t_start của event kế — DP cần khoảng
    cách thời gian có nghĩa, không phải một chuỗi điểm rời rạc."""
    rows = [KeyframeRow(n=1, pts_time=0.0, fps=25.0, frame_idx=0),
            KeyframeRow(n=2, pts_time=2.0, fps=25.0, frame_idx=50),
            KeyframeRow(n=3, pts_time=5.0, fps=25.0, frame_idx=125)]
    ev = events_cua_video("L01_V001", rows)
    assert [e.t_start for e in ev] == [0.0, 2.0, 5.0]
    assert ev[0].t_end == 2.0 and ev[1].t_end == 5.0
    assert ev[0].end_frame == 49                 # dừng ngay trước khung kế
    assert [e.keyframe_idx for e in ev] == [0, 50, 125]
    assert ev[-1].t_end > ev[-1].t_start         # keyframe cuối không có bề rộng 0


def test_event_id_giu_duoc_thu_tu_khi_sap_xep_chuoi():
    rows = [KeyframeRow(n=i, pts_time=i * 1.0, fps=25.0, frame_idx=i * 25)
            for i in range(1, 12)]
    ev = events_cua_video("L01_V001", rows)
    assert ev[0].event_id == "L01_V001#00001"
    assert sorted(e.event_id for e in ev) == [e.event_id for e in ev]


def test_meta_vao_kenh_chu_chi_khi_duoc_bat():
    rows = [KeyframeRow(n=1, pts_time=0.0, fps=25.0, frame_idx=0)]
    from aic2026.core.dataset import VideoMeta
    m = VideoMeta(video_id="L01_V001", title="Tin tức", keywords=["thời sự"])
    assert events_cua_video("L01_V001", rows, m)[0].asr_text == ""
    co = events_cua_video("L01_V001", rows, m, meta_vao_kenh_chu=True)[0]
    assert "Tin tức" in co.asr_text and "thời sự" in co.asr_text


# --------------------------------------------------------------------------- #
def test_load_dataset_tra_ve_event_va_vector_cung_thu_tu(tmp_path):
    dung_video(tmp_path, "L01_V001", n_kf=4)
    dung_video(tmp_path, "L01_V002", n_kf=3)
    ev, emb, metas, rep = load_dataset(tmp_path)
    assert len(ev) == 7 and emb.shape == (7, 512)
    assert rep.videos_da_nap == 2 and rep.events == 7
    assert set(metas) == {"L01_V001", "L01_V002"}
    assert rep.fps_phan_bo == {25.0: 2}


def test_load_dataset_bo_qua_video_lech_so_hang(tmp_path):
    """Lệch csv/npy = gán nhầm vector cho keyframe. Phải BỎ HẲN video đó, không
    được cắt bớt cho khớp — cắt bớt là hỏng âm thầm."""
    dung_video(tmp_path, "L01_V001", n_kf=4)
    dung_video(tmp_path, "L01_V002", n_kf=4, n_vector=3)   # cố tình lệch
    ev, emb, metas, rep = load_dataset(tmp_path)
    assert rep.videos_da_nap == 1
    assert rep.lech_so_hang == [("L01_V002", 4, 3)]
    assert {e.video_id for e in ev} == {"L01_V001"}
    assert "LỆCH SỐ HÀNG" in rep.tom_tat()


def test_load_dataset_loc_theo_video_va_gioi_han(tmp_path):
    for i in range(1, 6):
        dung_video(tmp_path, f"L01_V00{i}", n_kf=2)
    _, _, _, rep = load_dataset(tmp_path, gioi_han_video=2)
    assert rep.videos_da_nap == 2
    ev, _, _, rep2 = load_dataset(tmp_path, video_ids=["l01_v003"])   # không phân biệt hoa thường
    assert rep2.videos_da_nap == 1 and ev[0].video_id == "L01_V003"


def test_load_dataset_ghi_nhan_video_thieu_file(tmp_path):
    dung_video(tmp_path, "L01_V001", n_kf=2)
    dung_video(tmp_path, "L01_V002", n_kf=2)
    (tmp_path / "clip_features" / "ds" / "clip-features-32" / "L01_V002.npy").unlink()
    _, _, _, rep = load_dataset(tmp_path)
    assert rep.thieu_file == ["L01_V002"] and rep.videos_da_nap == 1


def test_load_dataset_thu_muc_rong(tmp_path):
    ev, emb, metas, rep = load_dataset(tmp_path)
    assert ev == [] and emb.shape == (0, 512) and rep.events == 0
