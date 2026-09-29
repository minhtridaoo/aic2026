"""Tầng TÌM — từ câu truy vấn ra danh sách ứng viên.

Trả lời đúng một câu hỏi: *video nào, khung hình nào*. Mọi thứ ở đây hỏng thì
sửa bằng cách đổi encoder, đổi index, đổi cách trộn kênh — không phải bằng cách
đổi thứ tự dòng nộp bài (việc đó thuộc `optimize/`).

    models       kiểu dữ liệu dùng chung (Event, Candidate, Query, ...)
    ingestion    frame -> event window 10s / overlap 2s
    providers    VisualEncoder + LLMProvider  ← ĐANG LÀ STUB, xem README
    stores       vector store + text store trong RAM
    retrieval    3 kênh visual/OCR/ASR + RRF / Borda / weighted-sum
    temporal_dp  DP căn thời gian cho TRAKE (nhánh MISS, cross-modal, đủ N khung)
    scoring      công thức chính thức Mean-of-Top-k R-Score
    submission   dựng dòng CSV, validate, đóng zip
    offline_eval chạy benchmark trên dev set
    pipeline     ráp tất cả thành một đường chạy; là chỗ duy nhất gọi sang optimize/

Cố ý để trống phần import ở file này: `core.pipeline` gọi sang `optimize`, còn
`optimize.submission_optimizer` gọi ngược về `core.scoring`. Nếu ở đây import sẵn
submodule thì hai chiều đó thành vòng tròn lúc nạp.
"""
