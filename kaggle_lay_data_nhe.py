"""Lấy ~179 MB cần dùng ra khỏi dataset 100+ GB, chạy TRÊN KAGGLE.

CÁCH DÙNG
---------
1. Vào kaggle.com -> Create -> New Notebook
2. Panel bên phải -> Add Input -> tìm dataset batch 1 -> Add
3. Dán TOÀN BỘ file này vào một cell -> Run
4. Xong: panel phải -> Output -> tải file aic2026_data_nhe.zip về máy
5. Giải nén vào D:\\COMP\\AIC2026\\du lieu nhe\\

VÌ SAO KHÔNG TẢI THẲNG
----------------------
Kaggle chỉ cho tải nguyên dataset hoặc từng file lẻ. Dataset này có 358.134
file nên tải lẻ là không khả thi, tải nguyên là 100+ GB. Cách vòng qua notebook
là: Kaggle đã mount sẵn dataset (không tốn gì), ta chỉ copy phần cần rồi nén
lại thành một file output nhỏ để tải.

QUY TẮC PHÂN LOẠI
-----------------
Copy đúng theo luật `classify_component` trong audit v4.7.4 của nhóm trưởng,
nên không phụ thuộc vào tên thư mục cụ thể:

    clip_features    .npy  và đường dẫn chứa clip / feature / embedding
    keyframe_mapping .csv|.json|... và đường dẫn chứa 'map' VÀ 'keyframe'
    metadata         .csv|.json|... và chứa metadata / media-info / media_info

Script CHỈ ĐỌC dataset gốc, không sửa gì. Nó in bảng dung lượng trước khi copy.
"""
import os
import shutil
import zipfile
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
STAGING = WORK / "aic2026_data_nhe"
ZIP_PATH = WORK / "aic2026_data_nhe.zip"

TABLE_EXTS = {".csv", ".tsv", ".txt", ".parquet", ".json", ".jsonl"}
MUON_LAY = {"clip_features", "keyframe_mapping", "metadata"}


def phan_loai(relative_path: str, ext: str) -> str:
    """Copy nguyên luật của audit v4.7.4 (cell 12) để khỏi lệch cách hiểu."""
    low = relative_path.lower()
    if ext in {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".ts"}:
        return "videos"
    if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return "keyframes"
    if ext == ".npy":
        return "clip_features" if any(
            t in low for t in ("clip", "feature", "embedding")) else "npy_unknown"
    if ext in TABLE_EXTS:
        if "object" in low:
            return "objects"
        if "map" in low and "keyframe" in low:
            return "keyframe_mapping"
        if any(t in low for t in ("media-info", "media_info", "metadata")):
            return "metadata"
        return "structured_unknown"
    return "other"


def doc_dep(n: float) -> str:
    for don_vi in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or don_vi == "GiB":
            return f"{n:,.1f} {don_vi}"
        n /= 1024


def main() -> None:
    if not INPUT_ROOT.exists():
        raise SystemExit("Không thấy /kaggle/input — script này phải chạy TRÊN KAGGLE.")

    print("Đang quét /kaggle/input ...\n")
    theo_nhom = {}          # component -> [(path, size), ...]
    for goc, _, ten_files in os.walk(INPUT_ROOT):
        for ten in ten_files:
            p = Path(goc) / ten
            rel = str(p.relative_to(INPUT_ROOT))
            nhom = phan_loai(rel, p.suffix.lower())
            try:
                kich_thuoc = p.stat().st_size
            except OSError:
                continue
            theo_nhom.setdefault(nhom, []).append((p, kich_thuoc))

    # ---- Bảng tổng quan: thấy TOÀN BỘ dataset có gì, trước khi copy ---------
    print(f"{'thành phần':<20s} {'số file':>10s} {'dung lượng':>14s}   lấy?")
    print("-" * 58)
    tong_lay = 0
    for nhom in sorted(theo_nhom, key=lambda k: -sum(s for _, s in theo_nhom[k])):
        ds = theo_nhom[nhom]
        tong = sum(s for _, s in ds)
        lay = nhom in MUON_LAY
        if lay:
            tong_lay += tong
        print(f"{nhom:<20s} {len(ds):>10,d} {doc_dep(tong):>14s}   "
              f"{'✓ LẤY' if lay else '– bỏ qua'}")
    print("-" * 58)
    print(f"{'TỔNG SẼ LẤY':<20s} {'':>10s} {doc_dep(tong_lay):>14s}\n")

    thieu = MUON_LAY - set(theo_nhom)
    if thieu:
        print(f"⚠️  KHÔNG tìm thấy: {sorted(thieu)}")
        print("   Kiểm tra lại đã Add đúng dataset batch 1 chưa.\n")

    # ---- Copy, giữ nguyên cây thư mục để loader nhận ra cấu trúc -----------
    if STAGING.exists():
        shutil.rmtree(STAGING)
    da_copy = 0
    for nhom in sorted(MUON_LAY & set(theo_nhom)):
        for p, _ in theo_nhom[nhom]:
            dich = STAGING / nhom / p.relative_to(INPUT_ROOT)
            dich.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dich)
            da_copy += 1
    print(f"Đã copy {da_copy:,d} file vào {STAGING}")

    # ---- Nén. .npy float16 gần như không nén được nên dùng ZIP_STORED cho
    #      nhanh; phần .csv/.json thì nén thật.
    print("Đang nén ...")
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for goc, _, ten_files in os.walk(STAGING):
            for ten in ten_files:
                p = Path(goc) / ten
                zf.write(p, p.relative_to(STAGING))

    print(f"\n✅ XONG: {ZIP_PATH}  ({doc_dep(ZIP_PATH.stat().st_size)})")
    print("   Panel bên phải -> Output -> tải file này về.")
    shutil.rmtree(STAGING)   # xoá bản chưa nén cho nhẹ output


if __name__ == "__main__":
    main()
