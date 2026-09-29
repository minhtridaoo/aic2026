"""So 3 cách xử lý truy vấn tiếng Việt — chạy TRÊN KAGGLE, ~5 phút, không cần GPU.

CÂU HỎI CẦN TRẢ LỜI
-------------------
CLIP ViT-B/32 không hiểu tiếng Việt (đã đo: 6 truy vấn khác nghĩa chồng lấn
top-100 tới 53.4%, câu nào cũng ra bản tin 70-90%). Có 3 cách chữa mà KHÔNG
phải encode lại 177.321 ảnh:

  A. clip_vi   CLIP gốc, nhét thẳng tiếng Việt          <- bản đối chứng, biết là hỏng
  B. clip_en   CLIP gốc, câu đã dịch sang tiếng Anh
  C. mclip_vi  Text encoder ĐA NGỮ, nhét thẳng tiếng Việt

Cách C dùng model được huấn luyện để nằm ĐÚNG CÙNG KHÔNG GIAN với vector ảnh
CLIP ViT-B/32 — tức vẫn xài được 173 MB vector BTC cho sẵn, chỉ thay nửa chữ.

Vì sao KHÔNG có BEiT-3 trong danh sách: nó cũng chỉ huấn luyện trên tiếng Anh
(bản checkpoint của đàn anh còn fine-tune riêng trên COCO), VÀ nó bắt encode lại
toàn bộ 177.321 ảnh. Trả giá đắt để mua đúng thứ mà cách B cho không.

SCRIPT NÀY KHÔNG ĐỌC DATASET
----------------------------
Nó chỉ encode 10 cặp câu chữ, tổng cộng 40 câu. KHÔNG cần Add Input, KHÔNG cần
mount 100+ GB, KHÔNG cần GPU. Lý do phải chạy trên Kaggle chỉ là vì máy bạn chưa
có `transformers` / `sentence-transformers` (~600 MB, mạng chậm).

Phần đối chiếu với 177.321 vector ảnh diễn ra Ở MÁY BẠN, bằng `so_sanh_encoder.py`.

CÁCH DÙNG
---------
1. Kaggle -> New Notebook
2. Settings bên phải -> bật **Internet ON**   <- BẮT BUỘC, để tải model về
   (Accelerator để None cũng được, không cần GPU)
3. Dán file này -> Run   (~5 phút)
4. Output -> tải 4 file .npz về, bỏ vào  system thử\\dev_set\\
5. Ở máy:  python so_sanh_encoder.py
   -> in ra độ chồng lấn của từng cách, thấp nhất là cách thắng.
"""
import os
import subprocess
import sys

import numpy as np

# Chạy được cả trên Kaggle lẫn ở máy (nếu sau này bạn cài transformers cục bộ).
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dev_set")
os.makedirs(OUT, exist_ok=True)

# Cặp Việt–Anh cùng nghĩa. Cùng một bộ khái niệm cho cả 3 cách, để so được.
CAP = [
    ("người đàn ông mặc áo đỏ",                 "a man wearing a red shirt"),
    ("bãi biển và hàng dừa",                    "a beach with palm trees"),
    ("người phát biểu trước micro",             "a person speaking at a microphone"),
    ("món ăn trên bàn trong nhà hàng",          "food on a table in a restaurant"),
    ("đường phố đông xe máy",                   "a street crowded with motorbikes"),
    ("phòng học có bảng trắng",                 "a classroom with a whiteboard"),
    ("hai người bắt tay nhau",                  "two people shaking hands"),
    ("ruộng lúa xanh trải dài",                 "green rice fields"),
    ("chợ đông người bán rau củ",               "a busy market selling vegetables"),
    ("cầu bắc qua sông vào ban đêm",            "a bridge over a river at night"),
]
VI = [a for a, _ in CAP]
EN = [b for _, b in CAP]


def luu(ten, keys, mat):
    mat = np.asarray(mat, dtype=np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    if mat.shape[1] != 512:
        raise ValueError(f"{ten}: {mat.shape[1]} chiều, vector ảnh BTC là 512")
    p = os.path.join(OUT, f"query_{ten}.npz")
    np.savez_compressed(p, keys=np.array([" ".join(k.split()) for k in keys],
                                         dtype=object).astype(str), vectors=mat)
    print(f"  ✅ {ten}: {mat.shape} -> {p}")


def cach_A_va_B():
    """CLIP gốc. Encode cả bản Việt (đối chứng) và bản Anh."""
    import torch
    from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

    MID = "openai/clip-vit-base-patch32"
    print(f"\n=== A + B: {MID} ===")
    tok = CLIPTokenizerFast.from_pretrained(MID)
    # ⚠ KHÔNG dùng CLIPModel.get_text_features(): ở transformers 5.x nó trả
    # `pooler_output` ĐÃ project sẵn, nhân `text_projection` lần nữa là project
    # hai lần -> vector vẫn 512 chiều nhưng gần vuông góc với vector đúng
    # (cosine 0.096). Lớp này project đúng một lần ở mọi phiên bản.
    mod = CLIPTextModelWithProjection.from_pretrained(MID).eval()

    def enc(texts):
        x = tok(texts, return_tensors="pt", padding=True,
                truncation=True, max_length=77)
        with torch.no_grad():
            v = mod(**x).text_embeds
        return v.cpu().numpy()

    luu("clip_vi", VI, enc(VI))
    luu("clip_en", EN, enc(EN))


def cach_C():
    """Text encoder đa ngữ, căn sẵn vào không gian ảnh của CLIP ViT-B/32."""
    MID = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    print(f"\n=== C: {MID} ===")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  cài sentence-transformers ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "sentence-transformers"], check=True)
        from sentence_transformers import SentenceTransformer

    mod = SentenceTransformer(MID)
    # Nhét THẲNG tiếng Việt — đó chính là điều đang muốn kiểm.
    luu("mclip_vi", VI, mod.encode(VI, convert_to_numpy=True))
    luu("mclip_en", EN, mod.encode(EN, convert_to_numpy=True))


def main():
    print(f"{len(CAP)} cặp truy vấn Việt–Anh cùng nghĩa\n")
    for ten, fn in (("A+B (CLIP gốc)", cach_A_va_B), ("C (đa ngữ)", cach_C)):
        try:
            fn()
        except Exception as exc:                            # noqa: BLE001
            import traceback
            print(f"\n  ❌ {ten} hỏng: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            print("  -> các cách còn lại vẫn chạy tiếp")

    co = [f for f in os.listdir(OUT) if f.startswith("query_") and f.endswith(".npz")]
    print(f"\n✅ Xong. Tải {len(co)} file về dev_set\\: {sorted(co)}")
    print("   Rồi chạy ở máy:  python so_sanh_encoder.py")


if __name__ == "__main__":
    main()
