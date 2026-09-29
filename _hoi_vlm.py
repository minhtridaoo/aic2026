"""Hỏi VLM một câu trên các KHUNG CỤ THỂ của MỘT video — thay mắt người.

    python -X utf8 _hoi_vlm.py L30_V043 "Biển tên đường trong ảnh ghi chữ gì?" 40 120 8

Tham số: video, câu hỏi, keyframe đầu, keyframe cuối, số ảnh lấy mẫu.

VÌ SAO KHÔNG DÙNG ĐƯỜNG QA CỦA PIPELINE: đường đó chọn khung theo điểm truy hồi
rồi mới hỏi. Ở đây ta ĐÃ CHẮC video (OCR/ASR chứng minh) và chỉ còn thiếu đáp
án, nên phải quét ĐỀU trên timeline của đúng video đó.

Đáp án đổi theo khung = VLM đang đọc thật. Một đáp án cho mọi khung = nó đoán
(xem `qa.soi_dap_an_vlm`).
"""
import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
sys.stdout.reconfigure(encoding="utf-8")

from aic2026 import config as C
from aic2026.core import providers
from aic2026.data_plane.image_resolver import EventImageResolver

VIDEO = sys.argv[1].upper()
CAU_HOI = sys.argv[2]
DAU = int(sys.argv[3]) if len(sys.argv) > 3 else 1
CUOI = int(sys.argv[4]) if len(sys.argv) > 4 else 200
SO = int(sys.argv[5]) if len(sys.argv) > 5 else 10

cfg = C.load_config("config_xem.yaml")
resolver = EventImageResolver.from_config(cfg)
llm = providers.dung_llm(cfg) if hasattr(providers, "dung_llm") else None
if llm is None:                      # tên hàm dựng đổi theo bản; dò cho chắc
    for ten in ("tao_llm", "dung_provider", "tu_config", "build_llm"):
        if hasattr(providers, ten):
            llm = getattr(providers, ten)(cfg)
            break
if llm is None:
    raise SystemExit("Không tìm được hàm dựng LLM trong aic2026.core.providers — "
                     f"có: {[x for x in dir(providers) if 'llm' in x.lower()]}")

buoc = max(1, (CUOI - DAU) // max(1, SO - 1))
ns = list(range(DAU, CUOI + 1, buoc))[:SO]
yeu_cau = []
for n in ns:
    try:
        p = resolver.resolve(f"{VIDEO}#{n:05d}")
    except OSError as exc:
        print(f"  (bỏ n={n}: {exc})".split("\n")[0])
        continue
    yeu_cau.append((CAU_HOI, str(p), f"{VIDEO}#{n:05d}", ""))

print(f"video {VIDEO} · {len(yeu_cau)} ảnh (keyframe {ns[0]}..{ns[-1]}, bước {buoc})")
print(f"câu hỏi: {CAU_HOI}\n")
ra = llm.answer_images(yeu_cau)
for (_, _, eid, _), dap in zip(yeu_cau, ra):
    print(f"  {eid:22s} -> {dap}")
print(f"\nprovider cuối: {getattr(llm, 'last_provider', '?')}")
