"""Tầng TIÊU — 100 slot nộp bài phân bổ thế nào.

Không truy hồi gì cả. Nhận danh sách ứng viên `core/` tìm được, rồi quyết định
thứ tự 100 dòng sao cho tối đa công thức của BTC:

    Final = (1/5) * Σ_{k∈{1,5,20,50,100}} R@k,   R@k = max_{i≤k}

Vì R@k lấy max trên tiền tố nên 100 slot là một *danh mục đầu tư*, không phải
top-100:

  * slot 1 được tính vào cả 5 mốc k nên đáng 5× slot 51;
  * một dòng đúng trong cùng hoàn cảnh với dòng đứng trước nó thì cộng thêm 0 —
    nộp 100 khung hình của cùng một cảnh có giá trị đúng bằng nộp 1 dòng;
  * đoán sai không bị trừ, nên luôn điền đủ 100 dòng.

    calibration          điểm thô -> xác suất (Platt / Isotonic / Rank)
                         kèm ECE, Brier, risk-coverage để biết xác suất có đáng tin
    submission_optimizer xếp slot theo lợi ích biên kỳ vọng

Bật bằng `PortfolioConfig(enabled=True)`. Mặc định TẮT cho tới khi
`python run_benchmark.py --ab` trên dev set thật cho thấy nó ăn điểm hơn đường
heuristic trong `core/submission.py`.

Cố ý để trống phần import ở file này — xem ghi chú chống vòng lặp trong
`core/__init__.py`.
"""
