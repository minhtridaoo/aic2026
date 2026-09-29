"""Chuẩn bị dev set + vector truy vấn — chạy TRÊN KAGGLE, một lần cho cả hai việc.

VÌ SAO CẦN
----------
Không có bộ query + đáp án chính thức của AIC2025 trên máy (đã tìm khắp D:\\COMP
và C:\\tmp). Không có đáp án thì chạy xong không biết đúng hay sai — vẫn là đoán.

Cách rẻ nhất để có đáp án thật: chọn ngẫu nhiên N keyframe, bạn TỰ MÔ TẢ nó bằng
tiếng Việt. Keyframe đó chính là đáp án đúng cho câu mô tả của bạn. Đây đúng là
"Track 1 — known-target coverage" trong tài liệu của Triết: không có chuyện mơ hồ
"tìm không ra hay cảnh không tồn tại", vì ta biết chắc cảnh đó có thật.

VIỆC SCRIPT NÀY LÀM
-------------------
  A. Xuất N ảnh keyframe đã chọn (~10 MB) + file dan_nhan.csv để bạn điền
  B. Encode sẵn vài câu truy vấn thử -> query_vectors.npz (xem ngay được kết quả)

CÁCH DÙNG
---------
1. Kaggle -> New Notebook -> Add Input: dataset batch 1 -> dán file này -> Run
   (KHÔNG cần GPU)
2. Output -> tải `dev_set_kit.zip` về, giải nén vào thư mục `system thử\\dev_set\\`
3. Mở `dan_nhan.csv`, xem ảnh cùng tên, gõ mô tả tiếng Việt vào cột `mo_ta`
4. Chạy `python tao_dev_set.py` để biến nó thành benchmark chấm điểm được

CÁCH CHỌN KEYFRAME
------------------
  * mỗi video lấy tối đa 1 khung -> câu hỏi trải rộng, không dồn vào vài video
  * bỏ 15% đầu và 10% cuối mỗi video -> tránh hình hiệu, intro, credit
  * bỏ khung có vector trùng khít khung khác -> tránh cảnh không phân biệt được
    (corpus có 904 nhóm trùng, nhóm lớn nhất trải 123 video)
"""
import hashlib
import json
import os
import random
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
STAGING = WORK / "dev_set_kit"
ZIP_PATH = WORK / "dev_set_kit.zip"

# Số keyframe đem đi gán nhãn.
#
# 60 -> 120 (2026-08-10). Đây là mốc thực tế trước mắt: thêm khoảng 60 câu để
# mở khoá so sánh mCLIP/Opus và giảm SE ghép cặp ước tính từ ±0.025 xuống
# khoảng ±0.018. Nó chưa đủ chứng minh hiệu ứng rất nhỏ; báo cáo vẫn phải in SE.
#
# Mỗi video chỉ lấy 1 keyframe, nên 120 câu = 120 video khác nhau.
#
# Ảnh xuất ra ước tính ~20 MB (60 ảnh hiện tại là 10 MB).
#
# 60 nhãn đã gõ KHÔNG mất: chạy `gop_nhan_cu.py` để chuyển sang file mới, nó
# khớp theo (video_id, n) chứ không giả định thứ tự bốc lặp lại y hệt.
SO_CAU = 120
HAT_GIONG = 20260807
VIDEO_ID_PATTERN = re.compile(r"(?i)(L\d{2,3}_V\d{3,5})")

# Câu truy vấn thử, encode sẵn để xem hệ thống chạy ngay hôm nay.
# CLIP gốc mạnh tiếng Anh hơn tiếng Việt -> để cả hai bản, đo rồi hãy chọn.
TRUY_VAN_THU = [
    "người đàn ông mặc áo đỏ",              "a man wearing a red shirt",
    "bãi biển và hàng dừa",                 "a beach with palm trees",
    "người phát biểu trước micro",          "a person speaking at a microphone",
    "món ăn trên bàn trong nhà hàng",       "food on a table in a restaurant",
    "đường phố đông xe máy",                "a street crowded with motorbikes",
    "phòng học có bảng trắng",              "a classroom with a whiteboard",
]


# --------------------------------------------------------------------------- #
def phan_loai(rel: str, ext: str) -> str:
    low = rel.lower()
    if ext == ".npy":
        return "clip" if any(t in low for t in ("clip", "feature", "embedding")) else ""
    if ext in {".jpg", ".jpeg", ".png", ".webp"}:
        return "anh"
    if ext in {".csv", ".json"}:
        if "object" in low:
            return ""
        if "map" in low and "keyframe" in low:
            return "map"
        if any(t in low for t in ("media-info", "media_info", "metadata")):
            return "meta"
    return ""


def quet():
    """Gom đường dẫn theo video_id. Ảnh keyframe gom theo (video, số thứ tự)."""
    clip, mp, meta = {}, {}, {}
    anh = defaultdict(dict)
    for goc, _, files in os.walk(INPUT_ROOT):
        for ten in files:
            p = Path(goc) / ten
            m = VIDEO_ID_PATTERN.search(str(p))
            if not m:
                continue
            vid = m.group(1).upper()
            loai = phan_loai(str(p.relative_to(INPUT_ROOT)), p.suffix.lower())
            if loai == "clip":
                clip[vid] = p
            elif loai == "map":
                mp[vid] = p
            elif loai == "meta":
                meta[vid] = p
            elif loai == "anh":
                so = re.findall(r"\d+", p.stem)
                if so:
                    anh[vid][int(so[-1])] = p
    return clip, mp, meta, anh


def doc_map(path):
    import csv
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return [(int(r["n"]), float(r["pts_time"]), float(r["fps"]),
                 int(r["frame_idx"])) for r in csv.DictReader(fh)]


# --------------------------------------------------------------------------- #
def phan_a_chon_va_xuat_anh(clip, mp, meta, anh):
    print("=== A. Chọn keyframe và xuất ảnh ===")
    if not anh:
        print("  ⚠ Không thấy ảnh keyframe trong input.")
        print("    Phải Add dataset ĐẦY ĐỦ (có thư mục keyframes), không phải bản nhẹ.")
        return []

    dung_chung = sorted(set(clip) & set(mp) & set(anh))
    print(f"  video đủ clip+map+ảnh: {len(dung_chung)}")

    # Băm vector để loại khung trùng khít (hình hiệu lặp giữa các tập).
    print("  đang quét vector trùng lặp ...")
    dem_bam = Counter()
    bam_theo_video = {}
    for vid in dung_chung:
        m = np.load(str(clip[vid]))
        bs = [hashlib.blake2b(m[i].tobytes(), digest_size=12).digest()
              for i in range(m.shape[0])]
        bam_theo_video[vid] = bs
        dem_bam.update(bs)
    print(f"  vector trùng khít: "
          f"{sum(v for v in dem_bam.values() if v > 1):,d} khung")

    rng = random.Random(HAT_GIONG)
    chon = []
    for vid in rng.sample(dung_chung, len(dung_chung)):
        if len(chon) >= SO_CAU:
            break
        rows = doc_map(mp[vid])
        if len(rows) < 10:
            continue
        lo, hi = int(len(rows) * 0.15), int(len(rows) * 0.90)   # bỏ intro/outro
        ung_vien = [i for i in range(lo, hi)
                    if dem_bam[bam_theo_video[vid][i]] == 1
                    and rows[i][0] in anh[vid]]
        if not ung_vien:
            continue
        i = rng.choice(ung_vien)
        n, pts, fps, fidx = rows[i]
        chon.append({"vid": vid, "n": n, "pts_time": pts, "fps": fps,
                     "frame_idx": fidx, "anh": anh[vid][n]})

    print(f"  đã chọn: {len(chon)} keyframe từ {len({c['vid'] for c in chon})} video")

    d_anh = STAGING / "anh"
    d_anh.mkdir(parents=True, exist_ok=True)
    dong = ["stt,video_id,n,pts_time,fps,frame_idx,anh,mo_ta"]
    for k, c in enumerate(chon, 1):
        ten = f"{k:03d}_{c['vid']}_n{c['n']}{c['anh'].suffix}"
        shutil.copy2(c["anh"], d_anh / ten)
        dong.append(f"{k},{c['vid']},{c['n']},{c['pts_time']},{c['fps']},"
                    f"{c['frame_idx']},{ten},")
    (STAGING / "dan_nhan.csv").write_text("\n".join(dong) + "\n", encoding="utf-8-sig")

    # Kèm metadata của đúng những video này, để lúc gán nhãn có ngữ cảnh.
    ctx = {}
    for c in chon:
        if c["vid"] in meta:
            with open(meta[c["vid"]], "r", encoding="utf-8") as fh:
                d = json.load(fh)
            ctx[c["vid"]] = {"title": d.get("title", ""),
                             "watch_url": d.get("watch_url", "")}
    (STAGING / "ngu_canh_video.json").write_text(
        json.dumps(ctx, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  đã xuất ảnh + dan_nhan.csv + ngu_canh_video.json")
    return chon


# --------------------------------------------------------------------------- #
def lay_vector_text(mod, tok, texts, torch):
    """Encode text -> vector 512 chiều ĐÃ QUA PROJECTION, chuẩn hoá L2.

    ⚠ KHÔNG dùng `CLIPModel.get_text_features()`. Bug đã trả giá 2026-08-08: ở
    transformers 5.x hàm đó trả `BaseModelOutputWithPooling` mà `pooler_output`
    ĐÃ được project sẵn. Code cũ thấy thiếu `text_embeds` nên nhân
    `text_projection` LẦN NỮA -> project hai lần -> cosine với vector đúng chỉ
    còn 0.096, tức gần vuông góc = nhiễu.

    Vì ViT-B/32 có hidden size = projection dim = 512, vector sai vẫn đúng 512
    chiều nên phép kiểm shape bên dưới KHÔNG bắt được. Hậu quả: chấm điểm ra
    0.0000 cho cả 60 câu, trông y như "encoder yếu".

    `CLIPTextModelWithProjection` có ở cả transformers 4.x và 5.x và project
    đúng một lần theo thiết kế — hết chuyện đoán theo phiên bản.
    """
    x = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        v = mod(**x).text_embeds
        v = v / v.norm(dim=-1, keepdim=True)
    if v.shape[-1] != 512:
        raise ValueError(
            f"vector text {v.shape[-1]} chiều, vector ảnh của BTC là 512.")
    return v.cpu().numpy().astype(np.float32)


def kiem_tra_cung_khong_gian(mat, clip, texts):
    """Kiểm vector text có THẬT SỰ cùng không gian với vector ảnh không.

    CLIP khớp text-ảnh thường cho cosine cỡ 0.15–0.35 với cặp đúng nghĩa. Nếu
    lấy nhầm vector chưa projection, cosine sẽ tụt về quanh 0 — đây là cái lưới
    duy nhất chặn được lỗi im lặng đó.
    """
    print("\n  Kiểm tra cùng không gian với vector ảnh:")
    mau = list(clip.values())[:20]
    if not mau:
        print("    ⚠ không có file clip để đối chiếu, bỏ qua")
        return
    anh = np.vstack([np.load(str(p)).astype(np.float32) for p in mau])
    anh /= np.linalg.norm(anh, axis=1, keepdims=True)
    print(f"    đối chiếu với {anh.shape[0]:,d} vector ảnh")
    cao_nhat = 0.0
    for i, t in enumerate(texts[:4]):
        d = anh @ mat[i]
        cao_nhat = max(cao_nhat, float(d.max()))
        print(f"    {t[:34]:34s} cos cao nhất={float(d.max()):.3f} "
              f"trung bình={float(d.mean()):+.3f}")
    if cao_nhat < 0.15:
        print("\n    ❌ CẢNH BÁO: cosine quá thấp. Vector text nhiều khả năng KHÔNG")
        print("       cùng không gian với vector ảnh. ĐỪNG dùng file này để đo.")
    else:
        print(f"\n    ✅ cosine cao nhất {cao_nhat:.3f} — nằm trong khoảng bình thường")


def phan_b_encode_truy_van(clip):
    print("\n=== B. Encode câu truy vấn thử ===")
    try:
        import torch
        from transformers import (CLIPTextModelWithProjection,
                                  CLIPTokenizerFast)
    except ImportError:
        print("  ⚠ Không có transformers, bỏ qua phần B.")
        return

    MODEL_ID = "openai/clip-vit-base-patch32"     # đúng model BTC dùng cho ảnh
    print(f"  tải {MODEL_ID} ...")
    tok = CLIPTokenizerFast.from_pretrained(MODEL_ID)
    mod = CLIPTextModelWithProjection.from_pretrained(MODEL_ID).eval()

    mat = lay_vector_text(mod, tok, TRUY_VAN_THU, torch)

    keys = [" ".join(t.split()) for t in TRUY_VAN_THU]
    np.savez_compressed(str(STAGING / "query_vectors.npz"),
                        keys=np.array(keys, dtype=object).astype(str), vectors=mat)
    print(f"  encode {len(keys)} câu -> query_vectors.npz")

    print("\n  Độ tương đồng cặp Việt–Anh cùng nghĩa (càng cao càng dịch sát):")
    for i in range(0, len(TRUY_VAN_THU) - 1, 2):
        print(f"    {TRUY_VAN_THU[i][:32]:32s} | {float(mat[i] @ mat[i+1]):.3f}")

    kiem_tra_cung_khong_gian(mat, clip, TRUY_VAN_THU)


# --------------------------------------------------------------------------- #
def main():
    if not INPUT_ROOT.exists():
        raise SystemExit("Phải chạy TRÊN KAGGLE (không thấy /kaggle/input).")
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    clip, mp, meta, anh = quet()
    print(f"Quét xong: {len(clip)} clip, {len(mp)} map, {len(meta)} meta, "
          f"{len(anh)} video có ảnh\n")

    phan_a_chon_va_xuat_anh(clip, mp, meta, anh)

    # Phần B hỏng thì vẫn phải đóng gói phần A: chọn ảnh mất vài phút quét, không
    # để mất trắng chỉ vì lỗi phía model.
    try:
        phan_b_encode_truy_van(clip)
    except Exception as exc:                                   # noqa: BLE001
        import traceback
        print(f"\n  ❌ Phần B hỏng: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        print("  -> Vẫn đóng gói phần A bên dưới; encode truy vấn chạy riêng sau.")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for goc, _, files in os.walk(STAGING):
            for ten in files:
                p = Path(goc) / ten
                zf.write(p, p.relative_to(STAGING))
    shutil.rmtree(STAGING)
    print(f"\n✅ Xong: {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024 / 1024:.1f} MiB)")
    print("   Output -> tải về -> giải nén vào  system thử\\dev_set\\")


if __name__ == "__main__":
    main()
