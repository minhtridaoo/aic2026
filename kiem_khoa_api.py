#!/usr/bin/env python
"""Khoá API còn sống không? — 1 request mỗi provider đang BẬT trong config.

    python -X utf8 kiem_khoa_api.py              # theo config: primary + fallbacks
    python -X utf8 kiem_khoa_api.py --anh <file>  # dùng ảnh khác để thử
    python -X utf8 kiem_khoa_api.py --cloudflare-agree-license

Chạy cái này **đầu buổi thi**. Nó trả lời đúng câu hỏi vận hành: khoá đã đặt
chưa, còn hiệu lực chưa, provider dự phòng có sẵn sàng thay không — trước khi
đề về, chứ không phải lúc đang chấm.

Chỉ đúng MỘT request mỗi provider và ghi vào cache TẠM (xoá sau khi chạy), nên
không làm bẩn cache đo điểm và không tốn quota đáng kể.

Khoá CHỈ đọc từ biến môi trường. Nếu thiếu, script nói rõ thiếu biến nào và
không đoán thêm gì.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026 import config as cau_hinh
from aic2026.core.providers import _build_llm_cfg

CAU_HOI = "Trong ảnh có người không?"


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None)
    p.add_argument("--anh", default=None, help="ảnh dùng để thử (mặc định: ảnh đầu "
                                              "trong qa.dev_image_dir)")
    p.add_argument("--cloudflare-agree-license", action="store_true",
                   help="gửi acknowledgement Meta license cho Workers AI rồi thoát")
    return p.parse_args()


def _anh_thu(config, chi_dinh):
    if chi_dinh:
        p = Path(chi_dinh)
        if not p.is_file():
            raise SystemExit(f"❌ Không thấy ảnh: {p}")
        return p
    dev = (config.get("qa") or {}).get("dev_image_dir")
    if dev and Path(dev).is_dir():
        for path in sorted(Path(dev).iterdir()):
            if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                return path
    for root in (config.get("qa") or {}).get("image_roots") or []:
        for path in sorted(Path(root).rglob("*.jpg"))[:1]:
            return path
    raise SystemExit("❌ Không tìm được ảnh nào để thử. Truyền --anh <file>.")


def main():
    args = _args()
    config = cau_hinh.load_config(args.config)
    muc = cau_hinh.provider_vlm_dang_bat(config)
    if not muc:
        print("Config không bật provider VLM nào (QA sẽ chạy zero-shot).")
        return
    anh = _anh_thu(config, args.anh)
    print(f"Ảnh thử: {anh.name}  ·  câu hỏi: {CAU_HOI!r}\n")

    ket_qua = []
    with tempfile.TemporaryDirectory() as tam:
        for thu_tu, cfg in enumerate(muc, start=1):
            ten = cfg.get("provider", "?")
            nhan = f"{thu_tu}. {ten} / {cfg.get('model', '?')}"
            bien = cau_hinh.bien_moi_truong(ten)
            thieu = cau_hinh.thieu_khoa_api({"llm": dict(cfg, provider=ten)})
            if thieu:
                print(f"{nhan}\n   ❌ {thieu[0]}")
                ket_qua.append((ten, False, "thiếu khoá"))
                continue

            # Cache TẠM: không đụng cache đo điểm, và luôn là một request thật.
            rieng = dict(cfg)
            rieng["cache_path"] = os.path.join(tam, f"probe_{ten}.json")
            provider = _build_llm_cfg(rieng)
            if args.cloudflare_agree_license and ten == "cloudflare_vlm":
                try:
                    print(f"{nhan}\n   license: {provider.agree_to_model_license()}")
                    ket_qua.append((ten, True, "license"))
                except Exception as exc:                       # noqa: BLE001
                    print(f"{nhan}\n   ❌ license lỗi: {str(exc)[:300]}")
                    ket_qua.append((ten, False, "license lỗi"))
                continue
            t = time.perf_counter()
            try:
                dap_an = provider.answer_image(CAU_HOI, str(anh), "PROBE#00001")
                giay = time.perf_counter() - t
                print(f"{nhan}\n   ✅ {giay:.2f}s · đáp án: {dap_an!r} "
                      f"· biến môi trường: {', '.join(bien)}")
                ket_qua.append((ten, True, f"{giay:.2f}s"))
            except Exception as exc:                           # noqa: BLE001
                print(f"{nhan}\n   ❌ {type(exc).__name__}: {str(exc)[:300]}")
                ket_qua.append((ten, False, type(exc).__name__))

    tot = [x for x in ket_qua if x[1]]
    print(f"\n  {len(tot)}/{len(ket_qua)} provider trả lời được.")
    if ket_qua and not ket_qua[0][1]:
        print("  ⚠ PROVIDER CHÍNH đang lỗi — QA sẽ rơi xuống fallback"
              " (kém hơn) hoặc zero-shot.")
    if not tot:
        print("  ❌ Không provider nào chạy. QA ngày thi sẽ là zero-shot.")
        sys.exit(1)


if __name__ == "__main__":
    main()
