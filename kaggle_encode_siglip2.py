"""Encode 177.321 keyframe bằng SigLIP2 — chạy TRÊN KAGGLE, GPU T4.

VÌ SAO CẦN
----------
BTC cấp sẵn vector ảnh bằng `clip-ViT-B-32` — model 2021, chỉ hiểu tiếng Anh, và
ta đang dùng nó VÌ ĐƯỢC CHO KHÔNG, không phải vì nó tốt. Đây là đòn lớn nhất
chưa đánh (VIEC_NGAY_MAI.md, việc 2).

CÁCH DÙNG
---------
1. Kaggle -> New Notebook
2. Add Input: `nguyenminhtric/data-batch-1-aic` — bản ĐẦY ĐỦ, có thư mục
   `keyframes/`. Bản nhẹ của `kaggle_lay_data_nhe.py` cố tình bỏ ảnh -> vô dụng ở đây.
3. Accelerator: **GPU T4**. Không bật thì chạy hàng chục giờ. Quota 30h/tuần.
4. Dán file này -> Run All
5. Output -> tải `siglip2_features.zip` về, giải nén vào thư mục dữ liệu

Đo so400m 3 video trước khi chạy full::

    python kaggle_encode_siglip2.py --so400m --smoke-3

Nếu script ước tính <= 4 giờ, bỏ ``--smoke-3`` để encode full.

XUẤT RA GÌ
----------
    siglip2-features/L21_V001.npy   (N, DIM) float16, ĐÃ chuẩn hoá L2
    siglip2-features/_meta.json     model, dim, số video, số frame, fingerprint

Cấu trúc và quy ước GIỐNG HỆT `clip-features-32/` của BTC (xem docstring
`aic2026/core/dataset.py`): hàng thứ i của .npy là keyframe thứ i của
`map-keyframes/<vid>.csv` xếp theo cột `n` tăng dần. Nhờ vậy `dataset.py` đọc
được mà không phải sửa gì ngoài đường dẫn.

⚠ SigLIP2 KHÔNG phải 512 chiều như CLIP. Mọi chỗ hardcode 512 (`CLIPTextEncoder.DIM`,
khoá kiểm trong cache, `luu_cache`) phải bỏ trước khi dùng bộ feature này. Khoá
`model` trong `_meta.json` là để chặn việc trộn nhầm hai không gian vector — cả hai
đều là "vector ảnh", trộn nhầm thì không có triệu chứng gì nhìn thấy được.

TÍNH ĐÚNG ĐẮN — BA THỨ ĐƯỢC KIỂM, KHÔNG TIN SUÔNG
--------------------------------------------------
1. Số hàng .npy PHẢI khớp số dòng CSV của đúng video đó. Lệch một hàng là lệch
   toàn bộ kết quả mà không báo lỗi ở đâu cả -> video đó bị loại, ghi vào lỗi.
2. Thiếu ảnh cho một `n` nào đó -> loại cả video, KHÔNG chèn vector 0 và cũng
   không âm thầm dồn hàng lên.
3. Norm của mọi vector PHẢI ≈ 1 sau chuẩn hoá.

Chạy lại được: video nào đã có .npy hợp lệ thì bỏ qua. Session chết giữa đường
(GPU timeout, mạng đứt) chỉ mất phần chưa xong, không mất hết.
"""
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------- #
# Cấu hình — chỉ cần đụng phần này
# --------------------------------------------------------------------------- #

# ĐỔI ĐÚNG MỘT DÒNG NÀY để chuyển checkpoint. Tên thư mục output đổi theo, nên hai
# bộ không bao giờ nằm chồng lên nhau — và `kaggle_encode_truy_van_siglip2.py`
# nhận cùng tên biến thể qua cờ `--so400m`, khỏi phải sửa MODEL_ID ở hai chỗ.
#
#   base   : ~30-45 phút trên T4, 768 chiều, ~260 MB. Đã chạy 2026-08-12,
#            điểm 0.2950 so với 0.1867 của CLIP (+0.1083 ± 0.0313).
#   so400m : mạnh hơn, chậm 4-6 lần, 1152 chiều, ~390 MB. CHƯA ĐO.
CHECKPOINT = "so400m" if "--so400m" in sys.argv else "base"

_MODEL = {
    "base": "google/siglip2-base-patch16-224",
    "so400m": "google/siglip2-so400m-patch14-384",
}
if CHECKPOINT not in _MODEL:
    raise SystemExit(f"CHECKPOINT phải là một trong {list(_MODEL)}, không phải "
                     f"{CHECKPOINT!r}")
MODEL_ID = _MODEL[CHECKPOINT]
_HAU_TO = "" if CHECKPOINT == "base" else f"-{CHECKPOINT}"

BATCH = 128          # ảnh mỗi forward. T4 16 GB thừa sức với base-224 fp16.
NUM_WORKERS = 4      # Kaggle cho 4 vCPU. Decode JPEG là nút cổ chai, không phải GPU.

# CÁCH THỬ NHANH — dùng cái này, ĐỪNG dùng GIOI_HAN_VIDEO.
# Liệt kê vài video_id: `quet()` sẽ prune hẳn các thư mục video khác, nên chỉ đọc
# vài trăm file thay vì ~178k -> quét xong trong giây thay vì phút.
_SMOKE_3 = ["L22_V006", "L26_V124", "L30_V012"]
CHI_VIDEO = _SMOKE_3 if "--smoke-3" in sys.argv else []

# >0 = chỉ encode N video đầu. LƯU Ý: cắt SAU khi quét, nên KHÔNG rút ngắn được
# thời gian quét /kaggle/input. Muốn thử nhanh thì dùng CHI_VIDEO.
GIOI_HAN_VIDEO = 0

# Zip to hơn ngưỡng này (MB) thì cắt thành nhiều phần — Kaggle hay fail khi tải
# một file to từ session đang chạy.
NGUONG_MB = 90

INPUT_ROOT = Path("/kaggle/input")
OUT_DIR = Path(f"/kaggle/working/siglip2-features{_HAU_TO}")
TEN_ZIP = f"/kaggle/working/siglip2_features{_HAU_TO}"

VIDEO_ID_PATTERN = re.compile(r"(?i)(L\d{2,3}_V\d{3,5})")
ANH_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# --------------------------------------------------------------------------- #
# Quét input
# --------------------------------------------------------------------------- #
def quet():
    """Gom đường dẫn theo video_id. Ảnh gom theo (video, số thứ tự n).

    Copy nguyên luật phân loại của `kaggle_chuan_bi_dev_set.quet()` để hai script
    không hiểu khác nhau về "thế nào là ảnh keyframe của video X, số n".
    """
    mp = {}
    anh = defaultdict(dict)
    goc_anh = {}          # (vid, n) -> dataset đã cấp file đó, để bắt trộn nguồn
    tron = []
    loc = {v.upper() for v in CHI_VIDEO}
    dem_file, t0 = 0, time.time()
    for goc, dirs, files in os.walk(INPUT_ROOT):
        if loc:
            # Prune: không bước vào thư mục của video không nằm trong CHI_VIDEO.
            # Ảnh keyframe nằm theo từng thư mục video nên đây là chỗ cắt hiệu quả nhất.
            dirs[:] = [d for d in dirs
                       if not (m := VIDEO_ID_PATTERN.fullmatch(d))
                       or m.group(1).upper() in loc]
        dem_file += len(files)
        if dem_file % 25000 < len(files):
            print(f"  ... đã quét {dem_file:,} file ({time.time() - t0:.0f}s)", flush=True)
        for ten in files:
            p = Path(goc) / ten
            m = VIDEO_ID_PATTERN.search(str(p))
            if not m:
                continue
            vid = m.group(1).upper()
            if loc and vid not in loc:
                continue          # ảnh/CSV phẳng không nằm trong thư mục theo video
            ext = p.suffix.lower()
            rel = p.relative_to(INPUT_ROOT)
            low = str(rel).lower()
            mount = rel.parts[0] if rel.parts else "?"
            if ext in ANH_EXTS:
                # số CUỐI trong tên file là n — `L22_V006/209.jpg` và
                # `keyframe_209.jpg` đều ra 209, còn video_id ở thư mục cha
                # không bị nhặt lẫn vì ta chỉ đọc `p.stem`.
                so = re.findall(r"\d+", p.stem)
                if so:
                    n = int(so[-1])
                    truoc = goc_anh.get((vid, n))
                    if truoc is not None and truoc != mount:
                        # Hai dataset khác nhau cùng khai "đây là keyframe n của
                        # video này". Ảnh BTC đánh số theo `n`, còn bộ
                        # `beit3-extract-keyframe` tự làm đánh số theo `frame_idx`
                        # — trộn vào là encode sai ảnh mà không có triệu chứng gì.
                        tron.append((vid, n, truoc, mount))
                        continue
                    goc_anh[(vid, n)] = mount
                    anh[vid][n] = p
            elif ext in {".csv", ".tsv"} and "map" in low and "keyframe" in low:
                mp[vid] = p

    if tron:
        mounts = sorted({m for t in tron for m in t[2:]})
        raise SystemExit(
            f"Ảnh keyframe đến từ NHIỀU dataset khác nhau: {mounts}\n"
            f"  {len(tron):,} khoá (video, n) bị tranh chấp — ví dụ "
            f"{tron[0][0]} n={tron[0][1]}: {tron[0][2]} vs {tron[0][3]}\n"
            "  Chỉ Add DUY NHẤT dataset BTC (data-batch-1-aic), remove các dataset\n"
            "  keyframe tự trích. Ảnh BTC đánh số theo `n`; bộ tự trích đánh số theo\n"
            "  `frame_idx` — trộn hai cách đánh số là encode sai ảnh trong im lặng.")
    return mp, anh


def doc_map(path):
    """Đọc map-keyframes -> list `n` theo đúng thứ tự hàng của .npy BTC."""
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return [int(r["n"]) for r in csv.DictReader(fh)]


# --------------------------------------------------------------------------- #
# Encode
# --------------------------------------------------------------------------- #
class BoAnh(Dataset):
    """Trả pixel_values đã tiền xử lý. Decode JPEG chạy trong worker, không chặn GPU."""

    def __init__(self, paths, proc):
        self.paths = paths
        self.proc = proc

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        px = self.proc(images=img, return_tensors="pt")["pixel_values"][0]
        return px


def lay_vector_anh(model, px):
    """`get_image_features` -> tensor (B, dim), chịu được cả transformers 4.x và 5.x.

    ⚠ 4.x trả thẳng tensor; 5.x trả `BaseModelOutputWithPooling` mà
    `pooler_output` ĐÃ được project sẵn. Đúng họ bug đã trả giá 2026-08-08 ở phía
    text (xem `kaggle_encode_truy_van.lay_vector_text`), chỉ khác là lần này nó
    ném AttributeError ngay nên không âm thầm.

    Ta KHÔNG tự nhân thêm projection nào — nên không có nguy cơ project hai lần.
    Cũng KHÔNG rơi về `last_hidden_state`: nó là (B, số token, dim), lấy nhầm thì
    vstack vẫn chạy mà vector thì vô nghĩa.
    """
    out = model.get_image_features(pixel_values=px)
    if hasattr(out, "shape"):
        v = out
    else:
        v = getattr(out, "image_embeds", None)
        if v is None:
            v = getattr(out, "pooler_output", None)
        if v is None:
            raise TypeError(
                f"get_image_features trả {type(out).__name__} không có "
                f"`image_embeds` lẫn `pooler_output`; các field có: "
                f"{list(getattr(out, 'keys', lambda: [])())}")
    if v.dim() != 2:
        raise ValueError(f"vector ảnh phải là (B, dim), đang là {tuple(v.shape)}")
    return v


def encode_video(model, proc, paths, device, dim):
    """Encode danh sách ảnh ĐÃ XẾP ĐÚNG THỨ TỰ -> (N, dim) float16 chuẩn L2."""
    dl = DataLoader(BoAnh(paths, proc), batch_size=BATCH, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)
    ra = []
    for px in dl:
        px = px.to(device, non_blocking=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                             enabled=device.type == "cuda"):
            v = lay_vector_anh(model, px)
        v = v.float()
        v = v / v.norm(dim=-1, keepdim=True)
        ra.append(v.cpu().numpy())
    mat = np.vstack(ra).astype(np.float16)
    if mat.shape != (len(paths), dim):
        raise ValueError(f"shape {mat.shape}, chờ {(len(paths), dim)}")
    return mat


def main():
    if not INPUT_ROOT.exists():
        raise SystemExit("Không thấy /kaggle/input — script này phải chạy TRÊN KAGGLE.")

    print("Đang quét /kaggle/input ...")
    mp, anh = quet()
    print(f"  map-keyframes: {len(mp)} video")
    print(f"  có ảnh keyframe: {len(anh)} video")
    if not anh:
        raise SystemExit(
            "Không thấy ảnh keyframe nào.\n"
            "  Phải Add dataset ĐẦY ĐỦ (có thư mục keyframes/), không phải bản nhẹ.")

    vids = sorted(set(mp) & set(anh))
    print(f"  đủ cả map + ảnh: {len(vids)} video")
    if len(vids) < len(mp):
        print(f"  ⚠ {len(mp) - len(vids)} video có map nhưng không có ảnh — sẽ bỏ")
    if GIOI_HAN_VIDEO:
        vids = vids[:GIOI_HAN_VIDEO]
        print(f"  ⚠ GIOI_HAN_VIDEO={GIOI_HAN_VIDEO} — chạy thử, KHÔNG phải bộ đầy đủ")

    # Báo ngay nếu không có mạng, thay vì traceback ba tầng của huggingface_hub.
    # Kaggle mặc định TẮT Internet và bật thì phải xác thực số điện thoại.
    import socket
    try:
        socket.getaddrinfo("huggingface.co", 443)
    except OSError:
        raise SystemExit(
            "❌ Notebook KHÔNG có internet — không tải được checkpoint SigLIP2.\n"
            "   Sửa: panel phải -> Settings -> Internet -> On, rồi Run lại.\n"
            "   (Kaggle bắt xác thực số điện thoại mới cho bật Internet.)\n"
            "   Việc này KHÔNG chạy được offline: 1.5 GB trọng số phải tải về.")

    from transformers import AutoModel, AutoProcessor

    print(f"\nTải model {MODEL_ID} ...")
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("  ⚠ KHÔNG CÓ GPU. Bật Accelerator -> GPU T4 rồi chạy lại.")
        print("    Chạy CPU sẽ mất hàng chục giờ cho 177k ảnh.")
    model = model.to(device)

    # Đo số chiều bằng một forward thật thay vì đọc config: tên field khác nhau
    # giữa các checkpoint (`text_config.hidden_size`, `vision_config.hidden_size`,
    # `projection_dim`), và đoán sai thì phép kiểm shape bên dưới sẽ loại sạch
    # 873 video mà không nói vì sao.
    with torch.no_grad():
        _px = proc(images=Image.new("RGB", (384, 384)), return_tensors="pt")
        dim = int(lay_vector_anh(model, _px["pixel_values"].to(device)).shape[-1])
    print(f"  số chiều: {dim}   device: {device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    xong, bo_qua, loi, tong_frame = 0, 0, [], 0
    t0 = time.time()

    for i, vid in enumerate(vids, 1):
        out = OUT_DIR / f"{vid}.npy"
        ns = doc_map(mp[vid])

        if out.exists():
            # Chạy lại: chỉ tin file cũ nếu số hàng và số chiều đều khớp.
            try:
                cu = np.load(out, mmap_mode="r")
                if cu.shape == (len(ns), dim):
                    bo_qua += 1
                    tong_frame += len(ns)
                    continue
                print(f"[{i}/{len(vids)}] {vid}: file cũ shape {cu.shape} sai -> encode lại")
            except Exception as exc:
                print(f"[{i}/{len(vids)}] {vid}: file cũ đọc không được ({exc}) -> encode lại")

        thieu = [n for n in ns if n not in anh[vid]]
        if thieu:
            # KHÔNG chèn vector 0 và KHÔNG dồn hàng lên: cả hai đều làm .npy lệch
            # so với CSV một cách im lặng. Loại hẳn video này.
            loi.append(f"{vid}: thiếu ảnh cho {len(thieu)}/{len(ns)} keyframe "
                       f"(n={thieu[:5]}{'...' if len(thieu) > 5 else ''})")
            continue

        paths = [anh[vid][n] for n in ns]      # thứ tự = thứ tự hàng CSV
        try:
            mat = encode_video(model, proc, paths, device, dim)
        except Exception as exc:
            loi.append(f"{vid}: encode lỗi — {type(exc).__name__}: {exc}")
            continue

        norm = np.linalg.norm(mat.astype(np.float32), axis=1)
        if not np.allclose(norm, 1.0, atol=2e-3):
            loi.append(f"{vid}: norm lệch, min={norm.min():.4f} max={norm.max():.4f}")
            continue

        np.save(out, mat)
        xong += 1
        tong_frame += len(mat)

        if i % 20 == 0 or i == len(vids):
            tr = time.time() - t0
            toc = tong_frame / tr if tr else 0
            con = (len(vids) - i) * (tr / i) if i else 0
            print(f"[{i}/{len(vids)}] {tong_frame:,} frame · {toc:.0f} frame/s · "
                  f"còn ~{con / 60:.0f} phút")

    # ----------------------------------------------------------------------- #
    meta = {
        "model": MODEL_ID,
        "dim": dim,
        "dtype": "float16",
        "chuan_hoa": "L2",
        "so_video": xong + bo_qua,
        "so_frame": tong_frame,
        "thu_tu_hang": "theo cot `n` tang dan cua map-keyframes/<vid>.csv",
        "loi": loi,
    }
    (OUT_DIR / "_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    tr = time.time() - t0
    print(f"\n{'=' * 62}")
    print(f"encode mới: {xong} video · bỏ qua (đã có): {bo_qua} · lỗi: {len(loi)}")
    print(f"tổng: {tong_frame:,} frame · {dim} chiều · {tr / 60:.1f} phút")
    toc = tong_frame / tr if tr else 0.0
    if toc:
        gio_full = 177_321 / toc / 3600
        print(f"tốc độ thật: {toc:.2f} frame/s · ước tính 177.321 frame: "
              f"{gio_full:.2f} giờ")
        if gio_full > 4:
            print("⛔ Ước tính > 4 giờ: dừng so400m theo ngưỡng đã chốt, "
                  "không chạy full.")
    if loi:
        print(f"\n⚠ {len(loi)} video KHÔNG có feature:")
        for d in loi[:15]:
            print(f"  - {d}")
        if len(loi) > 15:
            print(f"  ... còn {len(loi) - 15} dòng nữa trong _meta.json")

    if tong_frame < 177_321:
        print(f"\n⚠ {tong_frame:,} frame < 177.321 của BTC. Đừng so điểm với CLIP")
        print("  khi hai bên không cùng tập ảnh — chênh lệch sẽ do dữ liệu, không")
        print("  phải do encoder. Xử lý hết `loi` trước rồi chạy lại.")

    import shutil
    zip_path = shutil.make_archive(TEN_ZIP, "zip",
                                   root_dir=OUT_DIR.parent, base_dir=OUT_DIR.name)
    mb = os.path.getsize(zip_path) / 1024 / 1024
    print(f"\n✅ {zip_path}  ({mb:.0f} MB)")

    # Zip xong thì .npy gốc chỉ còn là bản trùng, mà /kaggle/working có hạn và
    # Output tab đếm cả hai -> xoá thư mục để khỏi phải nhìn 2x dung lượng.
    # Giữ lại `_meta.json` ở ngoài để đọc được mà không cần giải nén.
    shutil.copy(OUT_DIR / "_meta.json", "/kaggle/working/_meta.json")
    shutil.rmtree(OUT_DIR)

    # Kaggle rất hay fail khi tải một file to từ session đang chạy. Cắt thành
    # nhiều phần <= NGUONG_MB để browser tải được từng phần.
    if mb > NGUONG_MB:
        print(f"\n{mb:.0f} MB > {NGUONG_MB} MB -> cắt thành nhiều phần "
              f"để tải cho chắc:")
        buoc = NGUONG_MB * 1024 * 1024
        phan = 0
        with open(zip_path, "rb") as fh:
            while True:
                khoi = fh.read(buoc)
                if not khoi:
                    break
                phan += 1
                ten = f"/kaggle/working/siglip2_features.zip.{phan:03d}"
                with open(ten, "wb") as ra:
                    ra.write(khoi)
                print(f"   {ten}  ({len(khoi) / 1024 / 1024:.0f} MB)")
        os.remove(zip_path)          # bỏ bản nguyên khối, khỏi tốn chỗ 2 lần
        print(f"\n   Tải cả {phan} phần về CÙNG một thư mục, rồi ghép ở máy:")
        print("     PowerShell:  cmd /c copy /b siglip2_features.zip.* siglip2_features.zip")
        print("     Bash:        cat siglip2_features.zip.* > siglip2_features.zip")
    print("   Giải nén cạnh clip-features-32/")


if __name__ == "__main__":
    main()
