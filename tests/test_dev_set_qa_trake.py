"""VIỆC 4 — nhãn QA/TRAKE chảy đúng vào dev set, và hai hàm chấm nhận được.

`r_score_qa` và `r_score_trake` CHƯA TỪNG chạy trên dữ liệu thật, nên hình dạng
ground truth mà `tao_dev_set.py` sinh ra chưa từng được đối chiếu với thứ hai hàm
đó thực sự đọc. File này khớp hai bên lại, trên nhãn giả — để lỗi lộ ra trước khi
người gán bỏ 2-3 giờ gõ 50 nhãn.

Ba thứ được canh, cả ba đều sai được mà không có triệu chứng:
  1. `event_windows[j]` phải ứng với mốc thứ j, ĐÚNG THỨ TỰ đã bấm.
  2. Số bước tách được từ mô tả phải khớp số mốc đã chọn.
  3. Đáp án QA giữ nguyên văn — chuẩn hoá là việc của `answers_match`.
"""
import csv
import importlib.util
import json
import pathlib

import pytest

from aic2026.core import scoring
from aic2026.core.offline_eval import load_queries

GOC = pathlib.Path(__file__).resolve().parents[1]


def _nap_tao_dev_set():
    spec = importlib.util.spec_from_file_location("tds", GOC / "tao_dev_set.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


tds = _nap_tao_dev_set()


def _cua_so_gia(vid, n):
    """Giả `map-keyframes`: keyframe n phủ [n*100, n*100+99]."""
    if n < 1:
        return None
    return n * 100, n * 100 + 99


def _ghi_csv(path, cot, dong):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cot)
        w.writerows(dong)


def _ghi_dev_set(path, muc):
    path.write_text(json.dumps({"_note": "test", "queries": muc},
                               ensure_ascii=False), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------- #
# QA
# --------------------------------------------------------------------------- #
QA_COT = ["stt", "video_id", "n", "pts_time", "fps", "frame_idx", "anh",
          "cau_hoi", "cau_hoi_en", "dap_an"]


def test_qa_sinh_gt_ma_r_score_qa_cham_dung(tmp_path):
    _ghi_csv(tmp_path / "dan_nhan_qa.csv", QA_COT, [
        [1, "L21_V001", 5, 12.0, 25.0, 500, "a.jpg",
         "áo màu gì?", "what colour is the shirt?", "đỏ"],
    ])
    q = tds.doc_qa(str(tmp_path), _cua_so_gia)
    assert len(q) == 1
    gt = q[0]["gt"]
    assert gt == {"video_id": "L21_V001", "frame_start": 500, "frame_end": 599,
                  "answer": "đỏ", "answer_alt": []}
    assert q[0]["task"] == "qa"
    assert q[0]["text"] == "what colour is the shirt?"   # tiếng Anh là đường chính
    assert q[0]["_mo_ta_vi"] == "áo màu gì?"

    # Hàm chấm thật phải cho 1.0 khi đúng cả video + khung + đáp án.
    from aic2026.core.models import GroundTruth
    g = GroundTruth(task="qa", video_id="L21_V001", frame_start=500,
                    frame_end=599, answer="đỏ")
    assert scoring.r_score_qa(
        {"video_id": "L21_V001", "frame_idx": 550, "answer": "ĐỎ"}, g) == 1.0
    assert scoring.r_score_qa(
        {"video_id": "L21_V001", "frame_idx": 550, "answer": "đỏ."}, g) == 0.0
    assert scoring.r_score_qa(
        {"video_id": "L21_V001", "frame_idx": 700, "answer": "đỏ"}, g) == 0.0
    assert scoring.r_score_qa(
        {"video_id": "L21_V002", "frame_idx": 550, "answer": "đỏ"}, g) == 0.0


def test_qa_bo_dong_thieu_dap_an(tmp_path):
    _ghi_csv(tmp_path / "dan_nhan_qa.csv", QA_COT, [
        [1, "L21_V001", 5, 12.0, 25.0, 500, "a.jpg", "hỏi?", "ask?", ""],
        [2, "L21_V001", 6, 15.0, 25.0, 600, "b.jpg", "", "", "xanh"],
        [3, "L21_V001", 7, 18.0, 25.0, 700, "c.jpg", "hỏi?", "ask?", "vàng"],
    ])
    q = tds.doc_qa(str(tmp_path), _cua_so_gia)
    assert [x["gt"]["answer"] for x in q] == ["vàng"]
    assert q[0]["qid"] == "qa-001", "qid phải đánh lại liên tục sau khi bỏ dòng"


def test_qa_giu_nguyen_van_dap_an(tmp_path):
    """Chuẩn hoá là việc của answers_match. Sửa ở đây = nhãn khác thứ người gán thấy."""
    _ghi_csv(tmp_path / "dan_nhan_qa.csv", QA_COT, [
        [1, "L21_V001", 5, 1.0, 25.0, 500, "a.jpg", "h", "q", "  Áo  Đỏ  "],
    ])
    q = tds.doc_qa(str(tmp_path), _cua_so_gia)
    assert q[0]["gt"]["answer"] == "Áo  Đỏ", "chỉ strip hai đầu, không đụng bên trong"


def test_khong_co_file_qa_thi_tra_ve_rong(tmp_path):
    assert tds.doc_qa(str(tmp_path), _cua_so_gia) == []
    assert tds.doc_trake(str(tmp_path), _cua_so_gia) == []


# --------------------------------------------------------------------------- #
# TRAKE
# --------------------------------------------------------------------------- #
TK_COT = ["stt", "video_id", "cac_n", "mo_ta", "mo_ta_en"]


def test_trake_windows_dung_thu_tu_moc(tmp_path):
    _ghi_csv(tmp_path / "dan_nhan_trake.csv", TK_COT, [
        [1, "L21_V001", "3 7 12",
         "mở cửa → bước vào → ngồi xuống",
         "opens the door → walks in → sits down"],
    ])
    q = tds.doc_trake(str(tmp_path), _cua_so_gia)
    assert len(q) == 1
    assert q[0]["task"] == "trake"
    assert [s["text"] for s in q[0]["sub_events"]] == [
        "opens the door", "walks in", "sits down"]
    # Cửa sổ thứ j PHẢI của mốc thứ j
    assert q[0]["gt"]["event_windows"] == [[300, 399], [700, 799], [1200, 1299]]


def test_trake_diem_tung_phan_dung_nhu_thiet_ke(tmp_path):
    _ghi_csv(tmp_path / "dan_nhan_trake.csv", TK_COT, [
        [1, "L21_V001", "3 7 12", "a → b → c", "a → b → c"],
    ])
    q = tds.doc_trake(str(tmp_path), _cua_so_gia)
    from aic2026.core.models import GroundTruth
    g = GroundTruth(task="trake", video_id="L21_V001",
                    event_windows=[tuple(w) for w in q[0]["gt"]["event_windows"]])

    assert scoring.r_score_trake(
        {"video_id": "L21_V001", "frames": [350, 750, 1250]}, g) == 1.0
    assert scoring.r_score_trake(
        {"video_id": "L21_V001", "frames": [350, 750, 9999]}, g) == pytest.approx(2 / 3)
    # Đảo thứ tự phải MẤT điểm — đó là lý do thứ tự bấm mốc quan trọng
    assert scoring.r_score_trake(
        {"video_id": "L21_V001", "frames": [1250, 750, 350]}, g) == pytest.approx(1 / 3)
    assert scoring.r_score_trake(
        {"video_id": "L21_V002", "frames": [350, 750, 1250]}, g) == 0.0


def test_trake_bo_dong_khi_so_buoc_lech_so_moc(tmp_path):
    _ghi_csv(tmp_path / "dan_nhan_trake.csv", TK_COT, [
        [1, "L21_V001", "3 7 12", "a → b", "a → b"],          # 3 mốc, 2 bước
        [2, "L21_V002", "3 7", "a → b → c", "a → b → c"],      # 2 mốc, 3 bước
        [3, "L21_V003", "3 7 12", "a → b → c", "a → b → c"],   # khớp
    ])
    q = tds.doc_trake(str(tmp_path), _cua_so_gia)
    assert [x["gt"]["video_id"] for x in q] == ["L21_V003"]
    assert q[0]["qid"] == "trake-001"


def test_trake_bo_dong_khi_moc_khong_tang_dan(tmp_path):
    _ghi_csv(tmp_path / "dan_nhan_trake.csv", TK_COT, [
        [1, "L21_V001", "12 7 3", "a → b → c", "a → b → c"],
    ])
    assert tds.doc_trake(str(tmp_path), _cua_so_gia) == []


def test_trake_tach_duoc_bang_rồi_va_sau_do(tmp_path):
    """Người gán có thể gõ 'rồi'/'sau đó' thay vì dấu →."""
    _ghi_csv(tmp_path / "dan_nhan_trake.csv", TK_COT, [
        [1, "L21_V001", "3 7 12", "mở cửa rồi bước vào rồi ngồi xuống", ""],
    ])
    q = tds.doc_trake(str(tmp_path), _cua_so_gia)
    assert [s["text"] for s in q[0]["sub_events"]] == [
        "mở cửa", "bước vào", "ngồi xuống"]


# --------------------------------------------------------------------------- #
# load_queries đọc lại được cả ba loại
# --------------------------------------------------------------------------- #
def test_load_queries_doc_duoc_ca_kis_qa_trake(tmp_path):
    p = tmp_path / "dev.json"
    p.write_text(json.dumps({"queries": [
        {"qid": "kis-001", "task": "kis", "text": "a",
         "gt": {"video_id": "L21_V001", "frame_start": 0, "frame_end": 9}},
        {"qid": "qa-001", "task": "qa", "text": "b",
         "gt": {"video_id": "L21_V001", "frame_start": 0, "frame_end": 9,
                "answer": "đỏ"}},
        {"qid": "trake-001", "task": "trake",
         "sub_events": [{"text": "a"}, {"text": "b"}, {"text": "c"}],
         "gt": {"video_id": "L21_V001",
                "event_windows": [[0, 9], [10, 19], [20, 29]]}},
    ]}, ensure_ascii=False), encoding="utf-8")

    qs = load_queries(str(p))
    assert [q.task for q, _ in qs] == ["kis", "qa", "trake"]
    assert qs[1][1].answer == "đỏ"
    assert len(qs[2][0].sub_events) == 3
    assert len(qs[2][1].event_windows) == 3
    # final_score chạy được trên cả ba mà không ném lỗi
    for q, gt in qs:
        row = ({"video_id": "L21_V001", "frames": [5, 15, 25]} if gt.task == "trake"
               else {"video_id": "L21_V001", "frame_idx": 5, "answer": "đỏ"})
        assert 0.0 <= scoring.final_score([row], gt) <= 1.0


# --------------------------------------------------------------------------- #
# answer_alt — tách "trả lời sai" khỏi "đúng nghĩa, khác ngôn ngữ"
# --------------------------------------------------------------------------- #
def test_answer_alt_duoc_tinh_dung(tmp_path):
    from aic2026.core.models import GroundTruth

    g = GroundTruth(task="qa", video_id="L21_V001", frame_start=0, frame_end=99,
                    answer="xanh lá", answer_alt=["green", "màu xanh lá"])
    hang = lambda a: {"video_id": "L21_V001", "frame_idx": 50, "answer": a}
    assert scoring.r_score_qa(hang("xanh lá"), g) == 1.0
    assert scoring.r_score_qa(hang("GREEN"), g) == 1.0        # biến thể, có chuẩn hoá
    assert scoring.r_score_qa(hang("Màu  Xanh  Lá"), g) == 1.0
    assert scoring.r_score_qa(hang("đỏ"), g) == 0.0           # vẫn phải sai
    # Khung sai thì biến thể cũng không cứu được
    assert scoring.r_score_qa(
        {"video_id": "L21_V001", "frame_idx": 500, "answer": "green"}, g) == 0.0


def test_answer_alt_rong_thi_hanh_vi_y_nhu_cu():
    """Nới lỏng phải CÓ KIỂM SOÁT: không khai biến thể thì không đổi gì."""
    from aic2026.core.models import GroundTruth

    g = GroundTruth(task="qa", video_id="L21_V001", frame_start=0, frame_end=99,
                    answer="đỏ")
    hang = lambda a: {"video_id": "L21_V001", "frame_idx": 50, "answer": a}
    assert scoring.r_score_qa(hang("đỏ"), g) == 1.0
    assert scoring.r_score_qa(hang("red"), g) == 0.0
    assert g.answer_alt == []


def test_tao_dev_set_doc_cot_dap_an_alt(tmp_path):
    _ghi_csv(tmp_path / "dan_nhan_qa.csv",
             QA_COT + ["dap_an_alt"], [
                 [1, "L21_V001", 5, 1.0, 25.0, 500, "a.jpg", "h", "q", "xanh lá",
                  "green; light green"],
                 [2, "L21_V001", 6, 1.0, 25.0, 600, "b.jpg", "h", "q", "nồi", ""],
             ])
    q = tds.doc_qa(str(tmp_path), _cua_so_gia)
    assert q[0]["gt"]["answer_alt"] == ["green", "light green"]
    assert q[1]["gt"]["answer_alt"] == []


def test_load_queries_chuyen_answer_alt_sang_GroundTruth(tmp_path):
    p = _ghi_dev_set(tmp_path / "dev.json", [{
        "qid": "qa-001", "task": "qa", "text": "q",
        "gt": {"video_id": "L21_V001", "frame_start": 0, "frame_end": 99,
               "answer": "xanh lá", "answer_alt": ["green"]},
    }])
    (_, gt), = load_queries(p)
    assert gt.answer_alt == ["green"]
    assert scoring.r_score_qa(
        {"video_id": "L21_V001", "frame_idx": 50, "answer": "green"}, gt) == 1.0


def test_qa_truy_van_ghep_mo_ta_canh_voi_cau_hoi(tmp_path):
    """Câu hỏi TRƠ định vị kém — đo được 0.0667 vs 0.1800 khi có mô tả cảnh."""
    _ghi_csv(tmp_path / "dan_nhan.csv",
             ["stt", "video_id", "n", "pts_time", "fps", "frame_idx", "anh",
              "mo_ta", "mo_ta_en"],
             [[1, "L21_V001", 5, 1.0, 25.0, 500, "a.jpg",
               "một đầu bếp trong bếp.", "a chef in a kitchen."]])
    _ghi_csv(tmp_path / "dan_nhan_qa.csv", QA_COT, [
        [1, "L21_V001", 5, 1.0, 25.0, 500, "a.jpg",
         "áo màu gì?", "what colour is the shirt?", "đỏ"],
    ])
    q = tds.doc_qa(str(tmp_path), _cua_so_gia)
    assert q[0]["text"] == "a chef in a kitchen, what colour is the shirt?"
    assert q[0]["_mo_ta_vi"] == "một đầu bếp trong bếp, áo màu gì?"
    assert q[0]["_cau_hoi"] == "what colour is the shirt?"

    # Không có mô tả cảnh -> vẫn chạy, chỉ là câu hỏi trơ
    (tmp_path / "dan_nhan.csv").unlink()
    q2 = tds.doc_qa(str(tmp_path), _cua_so_gia)
    assert q2[0]["text"] == "what colour is the shirt?"
