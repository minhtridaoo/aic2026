"""Encode câu truy vấn thành vector CLIP 512 chiều — chạy TRÊN KAGGLE.

VÌ SAO CẦN
----------
BTC đã encode sẵn 177.321 ẢNH bằng `clip-ViT-B-32`. Muốn tìm bằng CHỮ thì câu
truy vấn phải nằm cùng không gian đó, tức phải chạy qua nửa text của đúng model
ấy. Máy bạn chưa có `transformers` (~600 MB), Kaggle thì có sẵn.

CÁCH DÙNG
---------
1. Sửa danh sách TRUY_VAN bên dưới (hoặc trỏ FILE_TRUY_VAN vào file .txt/.json)
2. Kaggle -> New Notebook -> dán file này -> Run  (KHÔNG cần bật GPU)
3. Output -> tải `query_vectors.npz` về
4. Ở máy:

       from aic2026.core.providers import CLIPTextEncoder
       enc = CLIPTextEncoder(cache_path="query_vectors.npz", cho_phep_live=False)

Encode lại lúc nào cũng ra đúng vector cũ, nên kết quả đo được tái lập chính xác.

LƯU Ý QUAN TRỌNG
----------------
CLIP gốc của OpenAI huấn luyện chủ yếu trên tiếng Anh. Truy vấn tiếng Việt vẫn
chạy nhưng chất lượng kém hơn rõ rệt. Script vì vậy encode CẢ hai bản nếu bạn
cung cấp bản dịch — xem TRUY_VAN dạng cặp bên dưới. Đo rồi hãy quyết dùng bản
nào; đừng đoán.
"""
import json
import os

import numpy as np
import torch
from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

MODEL_ID = "openai/clip-vit-base-patch32"   # đúng model BTC dùng cho ảnh

# Chạy được cả trên Kaggle lẫn ở máy. Máy đã cài transformers thì khỏi qua Kaggle.
_GOC = os.path.dirname(os.path.abspath(__file__))
_TREN_KAGGLE = os.path.isdir("/kaggle/working")
OUT = ("/kaggle/working/query_vectors.npz" if _TREN_KAGGLE
       else os.path.join(_GOC, "dev_set", "query_vectors.npz"))

# Trỏ vào file nếu có sẵn (mỗi dòng một truy vấn, hoặc JSON list[str]).
# Ở máy: tự lấy danh sách mà `tao_dev_set.py` vừa xuất ra, nếu có.
_MAC_DINH = os.path.join(_GOC, "dev_set", "truy_van_can_encode.txt")
FILE_TRUY_VAN = None if _TREN_KAGGLE or not os.path.exists(_MAC_DINH) else _MAC_DINH

# Hoặc gõ thẳng vào đây để thử nhanh.
TRUY_VAN = [
    "người đàn ông mở cửa container",
    "a man opening a shipping container",
    "cảng biển có xe tải và hàng hoá",
    "a seaport with trucks and cargo",
    "người phát biểu trước micro trên sân khấu",
    "a person speaking at a microphone on stage",
    "dòng chữ thông báo khẩn trên màn hình",
    "breaking news text on screen",
]


def doc_truy_van():
    if not FILE_TRUY_VAN:
        return list(TRUY_VAN)
    with open(FILE_TRUY_VAN, "r", encoding="utf-8") as fh:
        if FILE_TRUY_VAN.endswith(".json"):
            d = json.load(fh)
            if isinstance(d, dict):          # {"qid": "text", ...}
                return list(d.values())
            return [str(x) for x in d]
        return [ln.strip() for ln in fh if ln.strip()]


def lay_vector_text(mod, tok, texts, torch):
    """Encode text -> vector 512 chiều ĐÃ QUA PROJECTION ĐÚNG MỘT LẦN, chuẩn L2.

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

    Chuẩn hoá L2 để tích vô hướng ĐÚNG BẰNG cosine — cùng quy ước với vector ảnh
    sau khi `dataset.load_clip_features` chuẩn hoá lại.
    """
    x = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        v = mod(**x).text_embeds
        v = v / v.norm(dim=-1, keepdim=True)
    if v.shape[-1] != 512:
        raise ValueError(
            f"vector text {v.shape[-1]} chiều, vector ảnh của BTC là 512.")
    return v.cpu().numpy().astype(np.float32)


def main():
    texts = doc_truy_van()
    if not texts:
        raise SystemExit("Không có truy vấn nào để encode.")
    print(f"Số truy vấn: {len(texts)}")

    print(f"Tải model {MODEL_ID} ...")
    tok = CLIPTokenizerFast.from_pretrained(MODEL_ID)
    mod = CLIPTextModelWithProjection.from_pretrained(MODEL_ID).eval()

    ra = []
    B = 64
    for i in range(0, len(texts), B):
        # max_length=77 là giới hạn cứng của CLIP; câu dài hơn bị cắt.
        ra.append(lay_vector_text(mod, tok, texts[i:i + B], torch))
        print(f"  {min(i + B, len(texts))}/{len(texts)}")

    mat = np.vstack(ra)
    assert mat.shape == (len(texts), 512), mat.shape
    assert np.allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-5)

    # Khoá cache phải khớp cách chuẩn hoá của CLIPTextEncoder._khoa
    keys = [" ".join(t.split()) for t in texts]
    # Ghi kèm tên model để `CLIPTextEncoder.nap_cache` chặn việc nạp cache của
    # text tower này vào một text tower khác — cả hai đều 512 chiều nên trộn
    # nhầm không có triệu chứng gì.
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    np.savez_compressed(OUT, keys=np.array(keys, dtype=object).astype(str),
                        vectors=mat, model=np.array(MODEL_ID))
    print(f"\n✅ Xong: {OUT}  ({os.path.getsize(OUT) / 1024:.0f} KB)")
    print("   Output -> tải về -> CLIPTextEncoder(cache_path=...)")

    # Kiểm nhanh: cặp Việt/Anh cùng nghĩa phải gần nhau hơn cặp khác nghĩa.
    if len(texts) >= 4:
        print("\nĐộ tương đồng giữa các truy vấn (cosine):")
        for i in range(0, min(len(texts), 4), 2):
            print(f"  [{texts[i][:34]:34s}] vs [{texts[i+1][:34]:34s}] "
                  f"= {float(mat[i] @ mat[i+1]):.3f}")


if __name__ == "__main__":
    main()
