"""AIC HCMC 2026 — hệ thống video retrieval.

Một hệ thống, chia làm hai tầng theo việc chúng làm:

    core/       TÌM. Từ câu truy vấn ra danh sách ứng viên: video nào, khung
                hình nào. ingestion -> providers -> stores -> retrieval ->
                temporal_dp, cộng scoring/submission, ráp trong `pipeline`.

    optimize/   TIÊU. Không tìm gì cả; nhận danh sách ứng viên rồi quyết định
                xếp 100 dòng nộp bài ra sao để tối đa
                Final = (1/5) Σ_{k∈{1,5,20,50,100}} R@k.

Tách như vậy vì hai việc đó hỏng theo hai kiểu khác nhau và sửa bằng hai loại
công việc khác nhau: tầng TÌM hỏng thì đổi encoder/index, tầng TIÊU hỏng thì
đổi cách phân bổ slot. Trộn chung sẽ không biết mất điểm ở đâu.

Nối giữa hai tầng là cờ `PortfolioConfig.enabled` trong `core.pipeline`:

    False (mặc định)  core/submission — sort theo điểm, rải video ở đuôi
    True              optimize/       — greedy theo lợi ích biên kỳ vọng

Giữ cả hai đường KHÔNG phải vì lịch sử, mà vì đó là cách duy nhất đo được tầng
`optimize` đáng giữ hay đáng xoá: `python run_benchmark.py --ab` chạy cùng một
retriever qua cả hai rồi in chênh lệch. Chừng nào chưa có số trên dev set thật
thì mặc định vẫn để False. Nếu đo ra thua, xoá `optimize/` không tiếc.

Đường import chuẩn là đường đầy đủ:

    from aic2026.core.models import Event
    from aic2026.optimize.calibration import RankCalibrator

Đường phẳng (`from aic2026.models import Event`) vẫn chạy nhờ bảng alias bên
dưới, cho notebook và script cũ.

Toàn bộ chạy trên stdlib thuần. Các thành phần thật (SigLIP2, FAISS/Milvus,
Elasticsearch, Whisper, VietOCR) cắm vào tại các plug point đã đánh dấu.
"""

__version__ = "0.2.0"

# --------------------------------------------------------------------------- #
# Tương thích ngược: ánh xạ tên phẳng cũ -> vị trí mới.
#
# Đăng ký thẳng vào sys.modules thay vì viết file shim `from .core.models import *`
# vì `*` bỏ qua tên bắt đầu bằng gạch dưới (ví dụ `offline_eval._print_report`,
# đang được run_benchmark.py dùng). Alias trả về đúng object module gốc nên
# `aic2026.models is aic2026.core.models`.
#
# Import sẵn (không lười) để thứ tự nạp cố định: các submodule chỉ dùng import
# tương đối, mà `aic2026` đã nằm trong sys.modules ở thời điểm này nên không vỡ.
# --------------------------------------------------------------------------- #
_MOVED = {
    "models": "core.models",
    "ingestion": "core.ingestion",
    "providers": "core.providers",
    "stores": "core.stores",
    "retrieval": "core.retrieval",
    "temporal_dp": "core.temporal_dp",
    "scoring": "core.scoring",
    "submission": "core.submission",
    "offline_eval": "core.offline_eval",
    "pipeline": "core.pipeline",
    "calibration": "optimize.calibration",
    "submission_optimizer": "optimize.submission_optimizer",
}


def _install_legacy_aliases() -> None:
    import importlib
    import sys

    for old, new in _MOVED.items():
        # retrieval is now a real package. Its __init__ re-exports the old
        # core API, so legacy imports keep working without hiding new modules
        # such as retrieval.service and retrieval.persistent.
        if old == "retrieval":
            module = importlib.import_module(".retrieval", __name__)
            globals()[old] = module
            continue
        module = importlib.import_module("." + new, __name__)
        sys.modules[__name__ + "." + old] = module
        globals()[old] = module


_install_legacy_aliases()
