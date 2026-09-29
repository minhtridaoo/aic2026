# Kiến trúc & lựa chọn — bản tra cứu

Ghi lại **đã chọn gì, vì sao, và số đo nào chống lưng cho lựa chọn đó**. Mọi con
số ở đây đo trên dev set 60 câu KIS có nhãn + 177.321 vector thật của BTC, ngày
2026-08-08.

Quy ước: `Δ ± SE` là chênh lệch **ghép cặp** trên cùng bộ câu. `*` = vượt 2·SE
(đo được thật). Không có `*` = **chưa kết luận được**, dù con số trông thế nào.

---

## 1. ENCODER — câu chữ → vector 512 chiều

### Đã chọn: `CLIPTextEncoder` (text tower của `openai/clip-vit-base-patch32`)

**Vì sao bắt buộc là model này:** BTC phát sẵn vector ảnh sinh bằng
`clip-ViT-B-32`. Muốn tìm bằng chữ thì câu truy vấn phải nằm **cùng không gian**,
nên phải dùng đúng nửa còn lại của chính model đó. Đây là ràng buộc, không phải
lựa chọn.

**Số đo** (hạng của keyframe đúng trong 177.321; ~88.660 = xếp ngẫu nhiên):

| đường | hạng trung vị | R@100 | đúng video @100 |
|---|---|---|---|
| CLIP + tiếng Anh | **283** | **28.3%** | 61.7% |
| CLIP + tiếng Việt thô | 72.620 | 0.0% | 3.3% |
| đa ngữ + tiếng Việt thô | 1.183 | 18.3% | 56.7% |
| đa ngữ + tiếng Anh | 468 | 23.3% | 55.0% |

### Bug đã sửa: project hai lần

`CLIPModel.get_text_features()` ở transformers 5.x trả `BaseModelOutputWithPooling`
mà `pooler_output` **đã** được project. Code cũ nhân `text_projection` lần nữa →
cosine với vector đúng chỉ còn **0.096** (gần vuông góc).

Vì ViT-B/32 có hidden size = projection dim = **512**, vector sai vẫn đúng 512
chiều, vẫn chuẩn hoá được — **không phép kiểm shape nào bắt được**. Hậu quả: điểm
**0.0000** cho cả 60 câu, trông y hệt "encoder yếu".

Sửa bằng `CLIPTextModelWithProjection` (có ở cả 4.x và 5.x, project đúng một lần
theo thiết kế). Khoá bằng test: cosine(CLIP-en, mCLIP-en) phải > 0.80 — đo được
0.958, bản hỏng 0.096.

### Giữ thêm: `MCLIPTextEncoder`

`sentence-transformers/clip-ViT-B-32-multilingual-v1` — text tower đa ngữ đã căn
sẵn vào cùng không gian ảnh. **Đường dùng lúc thi**, vì BTC đưa câu tiếng Việt.

### Đã loại: BEiT-3

Cũng chỉ huấn luyện tiếng Anh, **và** bắt encode lại toàn bộ 177.321 ảnh. Trả giá
đắt để mua đúng thứ mà việc dịch cho không.

### Đã dựng nhưng KHÔNG bật: lớp dịch (`core/dich.py`)

| đường | điểm | Δ so người dịch |
|---|---|---|
| người dịch tay | 0.1233 | (nền) |
| NLLB-600M | 0.0967 | −0.0267 ± 0.0181 |
| opus-mt-vi-en | 0.0867 | −0.0367 ± 0.0210 |
| **đa ngữ, không dịch** | **0.0933** | −0.0300 ± 0.0222 |

Ba đường máy **không phân biệt được với nhau**. Nên chọn theo chi phí: mCLIP là
**một** component, dịch+CLIP là **hai** cộng 300MB–2.4GB và ~275 ms/câu. Giữ
`dich.py` lại vì rẻ, có thể thắng khi đủ nhãn.

⚠ Cả ba đều dưới bản người dịch → **ô tiếng Anh khi gán nhãn vẫn phải gõ**.

---

## 2. EMBEDDING — ảnh → vector

### Đã chọn: dùng thẳng `clip-features-32` BTC cấp

873 file `.npy` = **177.321 × 512**, đã chuẩn hoá L2, tổng 173 MB.

**Hệ quả lớn nhất:** không phải encode ảnh, nên **không cần tải bản >100 GB**.
Bản rút gọn 178 MB là đủ cho toàn bộ phần KIS.

### Lưu trữ: một ma trận numpy, không phải list trong từng `Event`

177.321 list Python 512 phần tử tốn vài GB chỉ riêng overhead object; ma trận
numpy tốn **363 MB** và một truy vấn là **một phép nhân ma trận**. Hợp đồng:
`embeddings[i]` là vector của `events[i]` — cùng thứ tự.

### Khử modality gap: **BẬT**

CLIP đặt vector ảnh và vector chữ ở hai nón tách rời, nên cosine chữ–ảnh bị chi
phối bởi hướng trung bình chung. Chiếu bỏ hướng đó đi.

| | điểm | R@100 | đúng video @100 |
|---|---|---|---|
| khử gap | 0.1233 | 28.3% | **68.3%** |
| thô | 0.1167 | 28.3% | 61.7% |

Δ điểm **không** vượt 2·SE. Lý do vẫn bật là cột cuối: **+6.6 điểm** tỉ lệ đúng
video mà không mất gì đo được — đó chính là phần một bước rerank có thể biến
thành điểm.

### Khoá chống trộn text tower

File cache `.npz` giờ ghi kèm tên model, `nap_cache` từ chối nạp chéo. Cần thiết
vì CLIP và mCLIP **đều cho vector 512 chiều và đều ghép được với vector ảnh** —
trộn nhầm không có triệu chứng nào, chỉ có điểm tụt không rõ lý do.

---

## 3. RETRIEVAL — vector → 100 dòng nộp bài

### Kênh: 3 kênh visual / OCR / ASR, trộn bằng **RRF thật**

Bug cũ: `_rrf_fallback_score` thực chất là **trung bình cộng** mang nhãn RRF. Đã
thay bằng `Σ 1/(k+rank)` thật, khoá bằng `test_fusion.py`.

### Kênh chữ metadata: **TẮT** — đo ra là hại

| | điểm | R@100 | đúng video @100 |
|---|---|---|---|
| tắt | 0.1233 | 28.3% | 68.3% |
| bật | 0.0933 | 20.0% | 51.7% |

**Δ = −0.0300 ± 0.0104\*** — hại thật, vượt 2·SE.

Không phải "kênh chết": 3.000 ứng viên có điểm chữ > 0, nó chạy hết công suất.
Vấn đề là truy vấn **tiếng Anh** còn title/keywords **tiếng Việt** → TF-IDF token
thô gần như không có token chung, nhưng RRF vẫn cấp cho nó trọng số **bằng** kênh
visual → mỗi câu bị chèn ~50 event ngẫu nhiên. Kèm chậm **40×** (283s vs 7s).

Điều kiện để bật lại: truy vấn giữ tiếng Việt, hoặc Elasticsearch có analyzer
tiếng Việt, và hạ trọng số kênh chữ.

### Kho vector: `NumpyVectorStore`

`argpartition` lấy top-k trong O(N) thay vì sort O(N log N). Thêm `lay_vector(key)`
để QA zero-shot tra ngược vector của một khung đã biết.

### Xếp 100 slot: **heuristic**, chưa bật tầng `optimize/`

`Final = (1/5) Σ_k max_{i≤k} R(rᵢ)` là **max trên tiền tố**, nên một dòng đúng
"cùng hoàn cảnh" với dòng trước cộng thêm **đúng bằng 0**.

- `KIS_TOL_FRAMES` **đã fit**: quét 17 giá trị 0→500, plateau **0..100** phẳng ở
  0.1300, tol 150 sụt còn 0.0900. Giữ **25** (giữa plateau). Đỉnh là tol=18
  nhưng hơn plateau 0.0033 < SE ±0.0067 → lấy đỉnh là chạy theo nhiễu.
- `--ab`: `0.1233 → 0.1300`, **Δ +0.0067 ± 0.0067** → chưa vượt 2·SE, **giữ tắt**.

### Bug đã sửa: calibrator nhận sai loại đầu vào

`RankCalibrator.predict()` chờ **hạng**, nhưng cả 3 builder truyền **điểm RRF**
(luôn ≤ 1/61). Vì có `max(1.0, score)` → **mọi ứng viên nhận cùng prior 0.5**,
tầng optimize mất sạch thứ hạng của retrieval. Đo được: 0.1233 → 0.1033, **đúng
bằng** phần "optimize làm mất 0.0200" từng báo.

Sửa bằng `Calibrator.dung_hang` + `Calibrator.prior(score, hạng)` — giao diện tự
khai báo nó cần gì thay vì để phía gọi đoán.

### Chưa chốt: gom điểm theo video

Chẩn đoán: encoder phân biệt **trong** video rất tốt (khung đúng hạng **4**/203),
nút cổ chai là hạng **video** (trung vị 27).

| luật gom điểm video | hạng video tv | V@10 | V@25 |
|---|---|---|---|
| `max` (đang dùng) | 27 | 35.0% | 48.3% |
| `tổng top10` | **14** | **45.0%** | **55.0%** |

Cấu hình `10 video × 10 khung` cho 0.1467 vs nền 0.1233. Nhưng kiểm tập giữ lại
(chia theo video): chọn trên nửa A → nửa B `+0.0467`; chọn trên nửa B → nửa A
`−0.0067`. **Cơ chế vững, độ lớn chưa chốt → chưa đổi mặc định.**

---

## 4. QA — trả lời câu hỏi khi chỉ có vector

### Vấn đề: không có ảnh

Bộ data BTC là vector + mapping + metadata. **Không ảnh.** Nên VLM nhìn ảnh bất
khả thi, LLM thuần chữ thì không thể biết "chiếc xe màu gì".

Trước khi có `core/qa.py`: `RuleBasedProvider.answer()` bới chữ hoa trong
`visual_text`/`ocr_text`/`asr_text` — cả ba **rỗng** → đáp án rỗng trên **100/100
dòng** → QA được **0 điểm tuyệt đối**.

### Đã chọn: chia đôi bài toán theo năng lực công cụ

```
LLM (hoặc luật)  :  câu hỏi  →  đáp án ứng viên       (thuần chữ)
CLIP             :  chấm từng ứng viên với vector ảnh  (thị giác)
```

Bước hai là zero-shot classification kinh điển của CLIP: dựng
`"a photo of a red object"`, encode bằng text tower, đo cosine với vector ảnh —
**vector ảnh thì ta có**. Nên QA chạy được mà không cần một tấm ảnh nào.

Trả về **phân phối** `[(đáp án, xác suất)]` chứ không một đáp án, vì đáp án loại
trừ lẫn nhau nên mỗi đáp án là một kịch bản riêng mà 100 slot phủ độc lập được.

### Kiểm chứng — phép quyết định

Cùng câu hỏi "màu áo là gì", đổi nhóm khung hình:

```
khung lấy từ 'a person wearing a red shirt'    →  red   (0.75)
khung lấy từ 'a green rice field under blue sky' →  green (0.81)
```

Vector ảnh **thật sự** tham gia quyết định — đây là QA thị giác, không phải bộ
sinh chuỗi. Đáp án rỗng: **0/100 dòng**.

⚠ Chưa có nhãn QA nên **chưa chấm được điểm**. Ba phép kiểm chỉ chứng minh đường
ống sống.

---

## 5. TRAKE — trần cấu trúc và cách nới

### Vấn đề: keyframe thưa gấp ~10 lần cửa sổ chấm

| | |
|---|---|
| cửa sổ chấm của BTC | "thường dưới 10 khung" |
| khoảng cách keyframe **thật** | trung vị **96** khung (74.773 khoảng, 300 video) |
| → nộp đúng chỉ số keyframe phủ được | **11.5%** khoảng trống |

**Không thuật toán nào lách được.** Nhưng hai điều cứu vãn: R-Score của TRAKE trả
**điểm từng phần** `(1/N)Σⱼ`, và Final lấy **max trên tiền tố** — nên nộp nhiều
dòng lệch nhau thì chỉ cần một dòng may.

### Đã chọn: lưới jitter bước 10, ±6 nhịp (13 dòng)

Mô phỏng với phân bố khoảng cách thật, N=3 mốc, giả định đã tin đúng video +
căn chỉnh (tức **trần trên**):

| lưới | số dòng | Final kỳ vọng |
|---|---|---|
| `(0,)` — mặc định cũ | 1 | 0.2521 |
| **bước 10, ±6 nhịp** | **13** | **0.3952** |
| bước 10, ±12 nhịp | 25 | 0.3958 (bão hoà) |
| bước 20, ±6 nhịp | 13 | 0.3458 (bước thưa, hở) |

Chọn ±6 chứ không ±12: thêm 12 dòng chỉ được +0.0006, mà 12 slot đó đáng giá hơn
khi dành cho căn chỉnh/video khác. Bước 10 = 2×tol để mỗi dòng là giả thuyết mới.
Thứ tự 0 trước rồi toả hai bên vì slot 1 đáng 5× slot 51.

**Cách duy nhất phá trần: khung dày hơn**, tức tự trích frame từ video gốc.

---

## 6. Trạng thái từng loại truy vấn

| loại | trạng thái | số đo (cập nhật 2026-08-16) |
|---|---|---|
| **KIS** | ✅ chạy thật, đo được | **0.3750** · R@1 15,8% · R@100 57,5% (so400m×2 + base-224 RRF) |
| **QA** | ✅ VLM thật, có nhãn 30 câu | **0.2067 ± 0.0618** (Gemini) · trần định vị 0.2133 · đáp án đúng 26/30 |
| **TRAKE** | ✅ chạy, đã nới trần | 0.2733 ± 0.0481 thật (trần mô phỏng 0.2521 → 0.3952) |

Lịch sử KIS: CLIP BTC 0.1233 → SigLIP2 base 0.2950 → so400m 0.3400 → hợp nhất
encoder 0.3750. Ba bước đầu là đổi encoder, bước cuối là RRF trên hạng giữa hai
checkpoint đã có sẵn trên đĩa (0 đồng, +545 MB RAM).

`nop_bai.py` (đường nộp bài thật, đọc config) chạy cả ba trên dữ liệu thật → zip
hợp lệ, validator `ok=True`, đáp án QA tiếng Việt. `kiem_dinh_dang.py` là bộ kiểm
ĐỊNH DẠNG riêng, cố ý dùng cấu hình rẻ — trước 16/08 hai việc này nằm trong cùng
một script (`nop_bai_thu.py`) và nó đã trôi khỏi cấu hình thi mà không ai thấy.

---

## 7. Đo được vs còn giả định

**Đã đo, không cần hỏi lại:** công thức Final (khớp ví dụ 0.74 của đề) · encoder
nào thắng · tol_frames · kênh chữ metadata · khử gap · lưới jitter TRAKE · QA có
nhìn ảnh thật.

**Chưa đo, đang là giả định:** điểm QA/TRAKE thật (thiếu nhãn) · `tổng top10`
(thiếu mẫu) · Codabench, cách đặt tên file zip, số lượt nộp, mọi mốc thời gian
(BTC chưa công bố).

**Nút cổ chai:** dev set 60 câu → SE ghép cặp ±0.025. Mốc trước mắt là gõ thêm
~60 để đạt khoảng 120 câu, SE ước tính ±0.018 và đo lại hai quyết định encoder/
dịch. Nếu hiệu ứng vẫn nằm trong 2·SE thì mới mở tiếp lên 200/400; không được
ép kết luận. Công cụ đã sẵn (`SO_CAU = 120`, `gop_nhan_cu.py`), chỉ còn việc
gõ nhãn.
