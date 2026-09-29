"""In chuỗi QA VLM thật sự sẽ được gọi, theo đúng thứ tự failover.

Chạy TRƯỚC mỗi buổi thi:  python -X utf8 _kiem_chuoi_vlm.py config_xem.yaml
0 request, 0 nạp dữ liệu.
"""
import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026 import config as C

ten = sys.argv[1] if len(sys.argv) > 1 else "config_xem.yaml"
c = C.load_config(ten)
print(f"config: {ten}")
print("chuỗi QA VLM đang bật:")
for i, m in enumerate(C.provider_vlm_dang_bat(c), 1):
    print(f"  {i}. {m.get('provider')}/{m.get('model')}"
          f"   interval={m.get('min_interval_seconds')}s"
          f"   batch={m.get('batch_max_images')}")
print("thiếu khoá API :", C.thieu_khoa_api(c) or "không")
qa = c.get("qa") or {}
print("loại câu -> VLM:", qa.get("vlm_question_types"))
print("vlm_top_n      :", qa.get("vlm_top_n"), "· tail:", qa.get("tail_answer"))
print("số thư mục ảnh :", len(qa.get("image_roots") or []))
