"""Khoá bố cục gói: core/ (tìm ứng viên) và optimize/ (phân bổ 100 slot).

Hai điều được giữ ở đây: ranh giới giữa hai tầng chỉ đi một chiều
(core -> optimize, không ngược lại), và đường import phẳng vẫn chạy cho
notebook/script cũ.
"""
import pkgutil

import aic2026
from aic2026 import core, optimize


def test_legacy_flat_imports_still_resolve():
    """`from aic2026.models import ...` phải trỏ vào đúng module đã dời."""
    import aic2026.models
    import aic2026.calibration

    assert aic2026.models is aic2026.core.models
    assert aic2026.calibration is aic2026.optimize.calibration


# Các module đã tồn tại lúc tách gói ngày 2026-08-07. Chỉ những cái này mới cần
# alias phẳng, vì chỉ chúng mới có notebook/script cũ đang import kiểu cũ.
# Module thêm sau (vd `core.dataset`) sinh ra đã có đường đầy đủ nên KHÔNG thêm
# vào đây — danh sách này là lịch sử, đóng băng, không nới ra.
MODULE_CU = {
    "models", "ingestion", "providers", "stores", "retrieval", "temporal_dp",
    "scoring", "submission", "offline_eval", "pipeline",
    "calibration", "submission_optimizer",
}


def test_alias_phu_het_module_cu():
    """Mọi module có từ lúc tách đều phải còn đường import phẳng."""
    declared = set(aic2026._MOVED)
    thieu = MODULE_CU - declared
    assert not thieu, f"module cũ mất alias: {sorted(thieu)}"


def test_alias_khong_tro_vao_khoang_khong():
    """Không được để lại alias trỏ tới module đã xoá/đổi tên."""
    present = {m.name for pkg in (core, optimize)
               for m in pkgutil.iter_modules(pkg.__path__)}
    declared = {new.split(".", 1)[1] for new in aic2026._MOVED.values()}
    treo = declared - present
    assert not treo, f"alias trỏ vào module không tồn tại: {sorted(treo)}"


def test_underscore_names_survive_the_alias():
    """Lý do dùng sys.modules thay vì `import *`: tên có gạch dưới phải còn."""
    from aic2026.offline_eval import _print_report  # run_benchmark.py đang dùng
    assert callable(_print_report)


def test_optimize_does_not_import_core_pipeline():
    """optimize/ chỉ được dùng scoring + calibration, không kéo ngược pipeline.

    Nếu vi phạm thì core.pipeline -> optimize -> core.pipeline thành vòng tròn,
    và tầng optimize hết tách rời để A/B.
    """
    import aic2026.optimize.submission_optimizer as opt
    assert not hasattr(opt, "Pipeline")
