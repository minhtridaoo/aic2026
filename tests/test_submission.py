import csv
import io
import os
import zipfile

from aic2026.core import submission as sub


def test_kis_row_format():
    out = sub.rows_to_csv("kis", [{"video_id": "L01_V028", "frame_idx": 25300}])
    assert out == "L01_V028,25300\r\n"


def test_qa_answer_with_comma_is_quoted():
    out = sub.rows_to_csv("qa", [{"video_id": "L01", "frame_idx": 5,
                                  "answer": "Năm người, gồm nam và nữ"}])
    assert '"Năm người, gồm nam và nữ"' in out
    assert out.endswith("\r\n")


def test_trake_row_all_frames():
    out = sub.rows_to_csv("trake", [{"video_id": "L10", "frames": [1200, 1850, 2100]}])
    assert out == "L10,1200,1850,2100\r\n"


def test_validate_catches_common_errors(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("L01.mp4,12\nL02,abc\n", encoding="utf-8", newline="")
    ok, errs = sub.validate_csv(str(p), "kis")
    assert not ok
    assert any(".mp4" in e for e in errs)
    assert any("integer" in e for e in errs)


def test_validate_trake_frame_count(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("L10,1,2,3\nL11,1,2\n", encoding="utf-8", newline="")
    ok, errs = sub.validate_csv(str(p), "trake", n_events=3)
    assert not ok
    assert any("expects 3 frames" in e for e in errs)


def test_validate_row_cap(tmp_path):
    p = tmp_path / "big.csv"
    p.write_text("".join(f"L01,{i}\n" for i in range(101)), encoding="utf-8", newline="")
    ok, errs = sub.validate_csv(str(p), "kis")
    assert not ok
    assert any("101 rows" in e for e in errs)


def test_diversify_keeps_head_and_spreads_tail():
    # Mỗi dòng có `frame_idx` RIÊNG — dòng nộp thật luôn có, và fixture cũ
    # (chỉ `video_id`) khiến 5 dòng "A" giống hệt nhau nên bị `khu_trung` gộp.
    # Ý định của test không đổi: giữ nguyên phần đầu, phần đuôi rải vòng tròn.
    rows = [{"video_id": "A", "frame_idx": i} for i in range(4)]
    rows += [{"video_id": "B", "frame_idx": 100},
             {"video_id": "A", "frame_idx": 4},
             {"video_id": "C", "frame_idx": 200}]
    out = sub.diversify_tail(rows, head=3)
    assert [r["video_id"] for r in out[:3]] == ["A", "A", "A"]   # head untouched
    tail = [r["video_id"] for r in out[3:]]
    assert tail[:3] == ["A", "B", "C"]                          # round-robin spread


class _A:  # minimal AlignResult stand-in for trake_rows
    def __init__(self, video_id, frames):
        self.video_id = video_id
        self.frames = frames


def test_build_zip_has_submission_prefix_and_validates(tmp_path):
    zip_path = tmp_path / "s.zip"
    rows = sub.trake_rows([_A("L10", [1, 2, 3]), _A("L11", [4, 5, 6])])
    ok, errs = sub.build_submission_zip(
        str(zip_path), {"query-1-trake.csv": ("trake", rows)},
        n_events={"query-1-trake.csv": 3})
    assert ok, errs
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert names == ["submission/query-1-trake.csv"]


def test_build_zip_rejects_wrong_trake_count(tmp_path):
    zip_path = tmp_path / "s.zip"
    rows = [{"video_id": "L10", "frames": [1, 2]}]
    ok, errs = sub.build_submission_zip(
        str(zip_path), {"query-1-trake.csv": ("trake", rows)},
        n_events={"query-1-trake.csv": 3})
    assert not ok
    assert not os.path.exists(zip_path)   # nothing written on validation failure


def test_khu_trung_dong_render_giong_het_nhau():
    """614 `frame_idx` trùng trong map-keyframes BTC -> hai event ra một dòng.

    `Final` lấy max trên tiền tố nên dòng trùng là một slot vứt đi. Quan sát
    thật trên bài nộp 7,6 điểm: `query-p1-9-kis` mất một slot kiểu này.
    """
    from aic2026.core.submission import diversify_tail, khu_trung

    rows = [{"video_id": "V1", "frame_idx": 10},
            {"video_id": "V1", "frame_idx": 10},      # trùng khít
            {"video_id": "V2", "frame_idx": 10},
            {"video_id": "V1", "frame_idx": 11}]
    assert len(khu_trung(rows)) == 3
    assert khu_trung(rows)[0] == rows[0]              # giữ thứ tự, giữ bản đầu

    ra = diversify_tail(rows, head=1)
    khoa = [(r["video_id"], r["frame_idx"]) for r in ra]
    assert len(khoa) == len(set(khoa)) == 3


def test_khu_trung_TRAKE_theo_ca_chuoi_khung():
    from aic2026.core.submission import khu_trung
    a = {"video_id": "V", "frames": [1, 2, 3, 4]}
    b = {"video_id": "V", "frames": [1, 2, 3, 5]}
    assert len(khu_trung([a, dict(a), b])) == 2


def test_khu_trung_lap_day_du_100_dong_khi_con_ung_vien():
    """Khử trùng phải chạy TRƯỚC khi cắt, nếu không là tự bỏ trống slot."""
    from aic2026.core.submission import diversify_tail
    rows = ([{"video_id": "V1", "frame_idx": 7}] * 3
            + [{"video_id": f"V{i}", "frame_idx": i} for i in range(2, 130)])
    ra = diversify_tail(rows, head=1)
    khoa = [(r["video_id"], r["frame_idx"]) for r in ra]
    assert len(ra) == 100 and len(set(khoa)) == 100


def test_so_ung_vien_du_bien_de_khu_trung_van_du_100():
    """Xin đúng 100 ứng viên rồi khử trùng = nộp 99 dòng. Bộ nghiệm thu của
    `nop_bai.py` bắt được ca này ngày 21/08."""
    from aic2026.core.pipeline import Pipeline
    assert Pipeline._so_ung_vien(100) >= 110
    assert Pipeline._so_ung_vien(10) >= 20


def test_answer_ton_trong_top_k_khong_tra_du_dong():
    """`answer(top_k=72)` phải trả 72 dòng, không phải 86.

    Từ khi `_so_ung_vien` xin dư biên ứng viên (để khử trùng vẫn đủ 100), quên
    truyền `max_rows=top_k` xuống `kis_rows` làm nó trả 86 dòng. Lớp ghim dựa
    vào con số đó để chia hạn ngạch slot, nên lệch ở đây là cấp sai hạn ngạch
    mà không có triệu chứng.
    """
    from aic2026.core.submission import kis_rows

    class C:
        def __init__(self, v, f):
            self.video_id, self.frame_idx = v, f

    cands = [C(f"V{i // 5}", i) for i in range(200)]
    assert len(kis_rows(cands, max_rows=72)) == 72
    assert len(kis_rows(cands)) == 100
