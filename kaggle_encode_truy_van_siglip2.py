"""Encode câu truy vấn bằng TEXT TOWER của SigLIP2 — chạy Kaggle hoặc ở máy.

VÌ SAO CẦN RIÊNG MỘT FILE
-------------------------
`kaggle_encode_truy_van.py` encode vào không gian CLIP ViT-B/32 512 chiều, để
ghép với `clip-features-32` BTC cấp sẵn. SigLIP2 là không gian KHÁC HẲN: vector
ảnh phải tự encode bằng `kaggle_encode_siglip2.py`, và câu truy vấn phải đi qua
đúng text tower của cùng checkpoint đó. Trộn hai bên là điểm tụt về mức ngẫu
nhiên mà không có lỗi nào ném ra.

CÁCH DÙNG
---------
1. Kaggle -> New Notebook (KHÔNG cần GPU, 120 câu chạy trong ~1 phút trên CPU)
   Không cần Add Input: script chỉ cần danh sách câu + checkpoint từ HF.
2. Dán file này. Nếu chạy trên Kaggle thì dán nội dung
   `dev_set/truy_van_can_encode.txt` vào TRUY_VAN bên dưới.
3. Run -> Output -> tải `query_vectors_siglip2.npz` về `dev_set/`
4. Ở máy: `python do_diem_thuc_te.py --siglip2`

Chạy thẳng ở máy cũng được nếu đã `pip install transformers` — lúc đó nó tự đọc
`dev_set/truy_van_can_encode.txt`, khỏi phải dán tay.

⚠ MODEL_ID PHẢI KHỚP `kaggle_encode_siglip2.py`
-----------------------------------------------
Đây là chỗ dễ sai nhất và sai thì im lặng. Số chiều của hai bên tình cờ bằng
nhau (cùng base-224 = 768) là điều kiện CẦN nhưng không đủ — hai checkpoint khác
nhau cùng 768 chiều vẫn không so sánh được. Khoá `model` ghi trong .npz là để
`SigLIP2TextEncoder.nap_cache` chặn đúng trường hợp đó.
"""
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Biến thể PHẢI GIỐNG `CHECKPOINT` trong kaggle_encode_siglip2.py.
# Ở máy thì truyền cờ, khỏi sửa file: `--so400m`, `--vi`, hoặc cả hai.
CHECKPOINT = "so400m" if "--so400m" in sys.argv else "base"

_MODEL = {
    "base": "google/siglip2-base-patch16-224",
    "so400m": "google/siglip2-so400m-patch14-384",
}
MODEL_ID = _MODEL[CHECKPOINT]
_HAU_TO = "" if CHECKPOINT == "base" else f"_{CHECKPOINT}"

# SigLIP huấn luyện với text đệm CỨNG tới 64 token. Dùng `padding=True` (đệm tới
# câu dài nhất trong batch) là lệch quy ước huấn luyện -> vector khác, điểm kém
# hơn, và KHÔNG có lỗi nào báo. Đây là lỗi phổ biến nhất khi chuyển CLIP -> SigLIP.
MAX_LEN = 64

# True = encode bản TIẾNG VIỆT (`truy_van_vi_can_encode.txt`) cho việc 2.6:
# SigLIP2 vốn huấn luyện đa ngữ, nếu nó đọc tiếng Việt đủ tốt thì bỏ được cả lớp
# dịch lẫn ô tiếng Anh khi gán nhãn các đợt sau. Chạy HAI lần — một lần False,
# một lần True — rồi so hai con số.
# Chạy ở máy thì tiện hơn: `python kaggle_encode_truy_van_siglip2.py --vi`
BAN_TIENG_VIET = "--vi" in sys.argv

_TREN_KAGGLE = os.path.isdir("/kaggle/working")
_GOC = "/kaggle/working" if _TREN_KAGGLE else os.path.dirname(os.path.abspath(__file__))

# Tên khớp `BIEN_THE_SIGLIP2` trong do_diem_thuc_te.py — đổi ở đây thì phải đổi ở đó.
_TEN_OUT = (f"query_vectors_siglip2{_HAU_TO}_vi.npz" if BAN_TIENG_VIET
            else f"query_vectors_siglip2{_HAU_TO}.npz")
OUT = (f"/kaggle/working/{_TEN_OUT}" if _TREN_KAGGLE
       else os.path.join(_GOC, "dev_set", _TEN_OUT))

_TEN_IN = ("truy_van_vi_can_encode.txt" if BAN_TIENG_VIET
           else "truy_van_can_encode.txt")
_MAC_DINH = os.path.join(_GOC, "dev_set", _TEN_IN)
FILE_TRUY_VAN = None if _TREN_KAGGLE or not os.path.exists(_MAC_DINH) else _MAC_DINH

# Chạy trên Kaggle thì dán nội dung truy_van_can_encode.txt vào đây.
TRUY_VAN = [
    "a man opening a shipping container",
    "a seaport with trucks and cargo",
]


def doc_truy_van():
    if not FILE_TRUY_VAN:
        return list(TRUY_VAN)
    with open(FILE_TRUY_VAN, "r", encoding="utf-8") as fh:
        if FILE_TRUY_VAN.endswith(".json"):
            d = json.load(fh)
            return list(d.values()) if isinstance(d, dict) else [str(x) for x in d]
        return [ln.strip() for ln in fh if ln.strip()]


def go_vector(out):
    """Lấy tensor (B, D) từ `get_text_features`, chịu cả transformers 4.x và 5.x.

    5.x trả `BaseModelOutputWithPooling` mà `pooler_output` đã project sẵn — cùng
    họ bug đã trả giá 2026-08-08. KHÔNG rơi về `last_hidden_state`: nó là
    (B, token, D), lấy nhầm thì mọi phép kiểm chiều vẫn qua mà vector vô nghĩa.
    """
    if hasattr(out, "shape"):
        return out
    for ten in ("text_embeds", "pooler_output"):
        v = getattr(out, ten, None)
        if v is not None:
            if v.dim() != 2:
                raise ValueError(f"`{ten}` phải là (B, D), đang là {tuple(v.shape)}")
            return v
    raise TypeError(
        f"{type(out).__name__} không có text_embeds/pooler_output; "
        f"field có: {list(getattr(out, 'keys', lambda: [])())}")


def kiem_mang():
    """Báo NGAY nếu notebook không có internet, thay vì 50 dòng traceback.

    Kaggle mặc định TẮT Internet, và muốn bật thì phải xác thực số điện thoại —
    acc mới thường chưa verify. Không có mạng thì `from_pretrained` chết với
    `Temporary failure in name resolution` lồng trong ba tầng exception, đọc mãi
    mới ra nguyên nhân thật.
    """
    import socket
    try:
        socket.getaddrinfo("huggingface.co", 443)
        return
    except OSError:
        pass
    raise SystemExit(
        "❌ Notebook KHÔNG có internet — không tải được checkpoint từ HuggingFace.\n"
        "   Sửa: panel phải -> Settings -> Internet -> On, rồi Run lại.\n"
        "   (Kaggle bắt xác thực số điện thoại mới cho bật Internet.)\n"
        "   Không bật được thì chạy thẳng ở máy: pip install transformers\n"
        "   rồi `python kaggle_encode_truy_van_siglip2.py` — 120 câu CPU ~30 giây,\n"
        "   và từ đó khỏi cần Kaggle cho việc encode chữ nữa.")


def main():
    texts = doc_truy_van()
    if not texts:
        raise SystemExit("Không có truy vấn nào để encode.")
    print(f"Số truy vấn: {len(texts)}")
    print(f"Nguồn: {FILE_TRUY_VAN or 'danh sách TRUY_VAN trong file'}")
    print(f"Bản: {'TIẾNG VIỆT' if BAN_TIENG_VIET else 'tiếng Anh'}  ->  {OUT}")

    kiem_mang()
    print(f"\nTải model {MODEL_ID} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mod = AutoModel.from_pretrained(MODEL_ID).eval()

    ra = []
    B = 64
    for i in range(0, len(texts), B):
        x = tok(texts[i:i + B], return_tensors="pt", padding="max_length",
                truncation=True, max_length=MAX_LEN)
        with torch.no_grad():
            v = go_vector(mod.get_text_features(**x))
            # Chuẩn L2 để tích vô hướng ĐÚNG BẰNG cosine — cùng quy ước với
            # vector ảnh sau khi `dataset.load_clip_features` chuẩn hoá lại.
            v = v / v.norm(dim=-1, keepdim=True)
        ra.append(v.cpu().numpy().astype(np.float32))
        print(f"  {min(i + B, len(texts))}/{len(texts)}")

    mat = np.vstack(ra)
    assert mat.shape[0] == len(texts), mat.shape
    assert np.allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-5)
    print(f"\nsố chiều: {mat.shape[1]}  "
          f"(base-224 = 768, so400m-384 = 1152)")

    # Khoá cache phải khớp cách chuẩn hoá của CLIPTextEncoder._khoa
    keys = [" ".join(t.split()) for t in texts]
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    np.savez_compressed(OUT, keys=np.array(keys, dtype=object).astype(str),
                        vectors=mat, model=np.array(MODEL_ID))
    print(f"✅ Xong: {OUT}  ({os.path.getsize(OUT) / 1024:.0f} KB)")

    # Kiểm nhanh có phân biệt được nội dung hay không. Vector suy biến (mọi câu
    # gần như trùng nhau) là triệu chứng của đệm sai hoặc lấy nhầm field.
    if len(texts) >= 3:
        G = mat @ mat.T
        np.fill_diagonal(G, np.nan)
        print(f"\ncosine giữa các câu khác nhau: trung vị {np.nanmedian(G):.3f}, "
              f"cao nhất {np.nanmax(G):.3f}")
        if np.nanmedian(G) > 0.95:
            print("  ⚠ Mọi câu gần như TRÙNG nhau -> vector suy biến, đừng dùng.")
            print("    Nghi trước tiên: đệm sai (phải padding='max_length'), hoặc")
            print("    lấy nhầm field trong output của get_text_features.")


if __name__ == "__main__":
    main()
