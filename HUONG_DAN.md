# Hiểu hệ thống này từ số 0

Tài liệu này giải thích **toàn bộ** hệ thống, từ chỗ chưa biết gì. Viết ngày
2026-08-07.

## Đọc thế nào

Sáu cấp, đi từ dễ đến khó. **Đọc hết Cấp 2 là đã đủ để ra quyết định và giao
việc.** Cấp 3–5 đọc khi cần, không cần thuộc.

| Cấp | Nội dung | Cần bao lâu |
|---|---|---|
| 0 | Cuộc thi là gì, chấm điểm ra sao | 5 phút |
| 1 | Hệ thống làm gì — một bức tranh | 5 phút |
| 2 | **Tự tay chạy và nhìn kết quả** | 15 phút |
| 3 | Từ điển thuật ngữ | tra khi cần |
| 4 | Từng file làm gì | tra khi cần |
| 5 | Phần nâng cao — vì sao thiết kế vậy | 20 phút |
| 6 | Hiện trạng thật và việc còn lại | 5 phút |

---

# CẤP 0 — Cuộc thi và cách chấm điểm

## Đề bài

BTC đưa **873 video** (batch 1, 107.8 GiB). Họ hỏi 3 kiểu câu:

| Kiểu | Câu hỏi | Ta phải nộp |
|---|---|---|
| **KIS** | "tìm cảnh người đàn ông mở container" | video nào + khung hình nào |
| **QA** | "thương hiệu nào trên container đó?" | video + khung + **chuỗi trả lời** |
| **TRAKE** | "tìm chuỗi: mở cửa → viết bảng → phát biểu" | video + **N khung theo thứ tự** |

## Luật chấm — phần quan trọng nhất

Mỗi câu hỏi được nộp **tối đa 100 dòng đáp án**. Điểm tính thế này:

$$\text{Điểm} = \frac{1}{5}\left(R@1 + R@5 + R@20 + R@50 + R@100\right)$$

trong đó **`R@k` = dòng đúng nhất trong `k` dòng đầu tiên**.

Nghe rối, nhưng hệ quả cực kỳ đơn giản và bạn phải nhớ 3 điều:

### ① Đặt đúng thứ tự quan trọng ngang với tìm đúng

| Đáp án đúng nằm ở slot | Điểm |
|---|---|
| 1 | **1.00** |
| 3 | 0.80 |
| 10 | 0.60 |
| 30 | 0.40 |
| 80 | 0.20 |
| 101 | **0.00** |

Cùng một đáp án đúng, đặt slot 1 được gấp **5 lần** đặt slot 80.
→ *Chạy `python giai_thich.py 1` để tự thấy.*

### ② Nộp trùng lặp = phí slot

Vì `R@k` lấy **max**, dòng thứ hai đúng trong *cùng hoàn cảnh* với dòng thứ nhất
thì **cộng thêm 0**. Nộp 100 khung hình của cùng một cảnh có giá trị đúng bằng
nộp 1 dòng.
→ *Chạy `python giai_thich.py 2`.*

### ③ Đoán sai KHÔNG bị trừ điểm

Nên **luôn nộp đủ 100 dòng**, kể cả khi không tự tin. Bỏ trống = tự nguyện nhận 0.

> **Kết luận Cấp 0:** 100 slot không phải bảng xếp hạng. Nó là **danh mục đầu
> tư** — phủ càng nhiều khả năng khác nhau càng tốt, khả năng nào chắc nhất đặt
> lên trên.

---

# CẤP 1 — Hệ thống làm gì

```
  Câu hỏi: "người đàn ông mở container Maersk"
       │
  ┌────▼─────────────────── TẦNG TÌM (thư mục core/) ─────────────┐
  │                                                                │
  │   ingestion    cắt video thành "event" — đoạn 10 giây          │
  │       ↓                                                        │
  │   providers    biến chữ và hình thành dãy số (vector)          │
  │       ↓                                                        │
  │   stores       cất vector vào kho để tra nhanh                 │
  │       ↓                                                        │
  │   retrieval    tìm song song 3 kênh, trộn 3 bảng xếp hạng      │
  │       ↓                                                        │
  │   temporal_dp  (chỉ TRAKE) chọn chuỗi khoảnh khắc đúng thứ tự  │
  │       ↓                                                        │
  │   ┌────── ~100 ỨNG VIÊN, ĐÃ XẾP HẠNG SƠ BỘ ──────┐             │
  └───┼─────────────────────────────────────────────┼─────────────┘
      │                                             │
      │      Câu hỏi kế: XẾP 100 DÒNG NỘP RA SAO?   │
      │                                             │
  ┌───▼──────────────────┐              ┌───────────▼──────────────┐
  │ Cách A — core/       │              │ Cách B — optimize/       │
  │ sort theo điểm,      │  chọn bằng   │ coi 100 slot là danh mục │
  │ giữ top 5, đuôi rải  │◄─ cờ bật/tắt─│ đầu tư, xếp theo lợi ích │
  │ đều các video        │              │ kỳ vọng                  │
  └──────────┬───────────┘              └───────────┬──────────────┘
             └───────────────┬───────────────────────┘
                             ▼
                  file CSV → kiểm tra → submission.zip
```

**Hai tầng, hai việc khác nhau:**

- `core/` trả lời **"video nào, khung nào?"** → hỏng thì sửa bằng cách đổi model
- `optimize/` trả lời **"đã có ứng viên rồi, xếp sao?"** → hỏng thì sửa bằng cách đổi phân bổ slot

Tách vậy để khi mất điểm còn biết mất ở đâu.

---

# CẤP 2 — Tự tay chạy

Mở terminal, vào thư mục `system thử`, chạy 5 lệnh sau theo thứ tự.

## Lệnh 1 — Kiểm tra sức khoẻ

```bash
python -m pytest -q
```

**Thấy gì:** `84 passed in ~0.4s`

**Nghĩa là:** 84 phép kiểm tự động đều đúng. Chạy lệnh này **trước và sau mỗi
lần sửa code**. Nếu đỏ → có gì đó vừa hỏng, dừng lại.

## Lệnh 2 — Sáu thí nghiệm giải thích khái niệm

```bash
python giai_thich.py          # chạy hết
python giai_thich.py 3        # chỉ thí nghiệm 3
```

| # | Thí nghiệm | Cho thấy |
|---|---|---|
| 1 | Vị trí quyết định điểm | Slot 1 = 1.00, slot 80 = 0.20 |
| 2 | Trùng lặp = phí slot | Dồn 1 cửa = 0.00, phủ 100 cửa = 0.40 |
| 3 | "Event" là gì | 31 khung → 4 event, mỗi event 10s |
| 4 | Ba kênh + cách trộn | Điểm từng kênh và điểm trộn |
| 5 | Hai cách xếp slot | Thứ tự dòng nộp khác nhau ra sao |
| 6 | TRAKE có điểm từng phần | Đúng 1/3 bước vẫn được 0.333 |

**Đây là lệnh quan trọng nhất để hiểu hệ thống.** Chạy thật, đừng chỉ đọc.

## Lệnh 3 — Chạy toàn bộ đường ống

```bash
python demo.py
```

**Thấy gì (dòng cuối):** `submission zip ok=True errors=[]`

**Nghĩa là:** đi trọn từ video → tìm kiếm → sinh file CSV → đóng gói zip, và
**định dạng file nộp hợp lệ**. Quan trọng thật, vì nộp sai định dạng là mất lượt.

## Lệnh 4 — Chấm điểm theo công thức BTC

```bash
python run_benchmark.py
```

**Thấy gì:** bảng điểm từng task (kis / qa / trake / OVERALL).

## Lệnh 5 — So hai cách xếp slot

```bash
python run_benchmark.py --ab
```

## Lệnh 6 ⭐ — Kiểm chứng trên DỮ LIỆU THẬT

```bash
python kiem_chung_thuc_te.py
```

Khác hẳn 5 lệnh trên: lệnh này chạy trên **873 video, 177.321 keyframe thật**
của BTC. Nó không cần CLIP text encoder, vì dùng chính vector của một keyframe
làm câu truy vấn (tìm ảnh bằng ảnh).

| Phép kiểm | Kết quả đo được |
|---|---|
| 1. Nạp dữ liệu | 177.321 event / 873 video — **khớp audit** |
| 2. Tự tìm chính mình | **200/200** |
| 3. Vector có mang nghĩa | 54,5% láng giềng cùng video, trung vị cách 5,0s |
| 4. Tốc độ | **11 ms** / truy vấn trên 177k vector |
| 5. Nộp bài | 100 dòng, 30 video, zip hợp lệ |

Đây là bằng chứng đường ống chạy đúng ở quy mô thật.

**Thấy gì:**

```
                          kis       qa    trake  OVERALL
core/     (heuristic)  0.9000   0.8000   1.0000   0.9000
+optimize (portfolio)  0.9000   1.0000   0.8000   0.9000
delta                 +0.0000  +0.2000  -0.2000  +0.0000
```

**Đọc thế nào:** tầng `optimize` thắng QA nhưng thua TRAKE, tổng bằng 0 → **hiện
chưa chứng minh được nó đáng bật.**

## ⚠️ Cảnh báo lớn nhất về Cấp 2

**Lệnh 1–5 chạy trên dữ liệu BỊA.** (Lệnh 6 thì trên dữ liệu thật.)

File `data/benchmark_queries.sample.json` chỉ có **8 sự kiện + 4 câu hỏi**, gõ
tay, không có khung hình video thật nào. Đáp án đúng cũng do người viết file tự
đặt.

Và encoder hiện tại là `HashEmbedder` — nó **băm chuỗi ký tự ra số ngẫu nhiên**
rồi giả vờ đó là vector ngữ nghĩa. Nó không "nhìn" thấy gì cả.

Nên con số `0.9` kia **không nói gì** về việc hệ thống tìm kiếm tốt hay dở. Nó
chỉ chứng minh: đường ống thông, định dạng nộp đúng, công thức tính điểm đúng.

---

# CẤP 3 — Từ điển thuật ngữ

Tra khi gặp từ lạ.

| Từ | Nghĩa dễ hiểu |
|---|---|
| **keyframe** | Một khung hình đại diện được rút ra từ video. BTC cấp sẵn 177.321 cái cho 873 video |
| **event** | Đoạn video 10 giây. Đơn vị mà hệ thống đi tìm (thay vì tìm từng ảnh lẻ) |
| **embedding / vector** | Dãy số mô tả nội dung. Hai thứ giống nhau → hai dãy số gần nhau. CLIP của BTC dùng **512 số** |
| **encoder** | Cái máy biến ảnh hoặc chữ thành vector |
| **CLIP** | Model của OpenAI, biến ảnh và chữ về **cùng một không gian số** → so ảnh với chữ được. BTC đã encode sẵn |
| **index / store** | Kho chứa vector, tra nhanh "cái nào gần nhất" |
| **OCR** | Đọc chữ hiện trên màn hình |
| **ASR** | Chuyển lời nói thành chữ |
| **RRF** | Cách trộn nhiều bảng xếp hạng: cộng `1/(60 + thứ hạng)`. Chỉ dùng **thứ hạng**, không dùng điểm thô → kênh cho điểm 0–1 và kênh 0–1000 vẫn trộn công bằng |
| **DP** (quy hoạch động) | Thuật toán chọn chuỗi tối ưu. Dùng cho TRAKE: chọn 3 khoảnh khắc đúng thứ tự thời gian |
| **k-best / list-Viterbi** | DP thường trả 1 đáp án tốt nhất. k-best trả **k đáp án tốt nhất khác nhau** |
| **calibration** | Đổi "điểm số" thành "xác suất đúng". Điểm 0.9 không có nghĩa là 90% đúng — calibration sửa việc đó |
| **portfolio** | Coi 100 slot như tiền đầu tư: chia vào nhiều cửa, cửa chắc ăn đặt trước |
| **R@k** | Dòng đúng nhất trong k dòng đầu |
| **tol_frames** | Ngưỡng "hai khung cách nhau bao nhiêu thì coi là cùng một chỗ" |
| **stub** | Đồ giả tạm thời, để code chạy được trong khi chờ đồ thật. `HashEmbedder` là stub |
| **pytest** | Công cụ chạy các phép kiểm tự động |

---

# CẤP 4 — Từng file làm gì

## `aic2026/core/` — tầng TÌM

| File | Việc |
|---|---|
| `models.py` | Định nghĩa các kiểu dữ liệu dùng chung (Event, Candidate, Query…) |
| **`dataset.py`** | **Đọc dữ liệu thật của BTC** (npy + csv + json) → `Event`. Cần numpy |
| `ingestion.py` | Cắt khung hình thành event 10s / chồng 2s (đường cũ, cho dữ liệu tự dựng) |
| `providers.py` | Encoder + bộ trả lời câu hỏi. `HashEmbedder` là stub, **`CLIPTextEncoder`** là hàng thật |
| `stores.py` | Kho vector + kho chữ. `InMemoryVectorStore` (stdlib) và **`NumpyVectorStore`** (177k vector) |
| `retrieval.py` | Tìm 3 kênh + trộn bằng RRF/Borda |
| `temporal_dp.py` | DP căn chuỗi thời gian cho TRAKE (+ bản k-best) |
| `scoring.py` | Công thức chấm điểm chính thức của BTC |
| `submission.py` | Sinh dòng CSV, kiểm tra hợp lệ, đóng zip |
| `offline_eval.py` | Chạy benchmark, in báo cáo |
| `pipeline.py` | Ráp tất cả lại. **Chỗ duy nhất gọi sang `optimize/`** |

## `aic2026/optimize/` — tầng TIÊU

| File | Việc |
|---|---|
| `calibration.py` | Điểm → xác suất, kèm thước đo độ tin cậy (ECE, Brier) |
| `submission_optimizer.py` | Xếp 100 slot theo lợi ích kỳ vọng |

## Ở thư mục gốc

| File | Việc |
|---|---|
| `demo.py` | Chạy thử toàn bộ đường ống (dữ liệu bịa) |
| `run_benchmark.py` | Chấm điểm (`--ab` để so hai cách xếp slot) |
| `giai_thich.py` | 6 thí nghiệm giải thích khái niệm |
| **`kiem_chung_thuc_te.py`** | **5 phép kiểm trên 177k keyframe thật** |
| `kaggle_lay_data_nhe.py` | Lấy 179 MB ra khỏi dataset 100+ GB (chạy trên Kaggle) |
| `kaggle_encode_truy_van.py` | Encode câu truy vấn thành vector CLIP (chạy trên Kaggle) |
| `data/benchmark_queries.sample.json` | Dữ liệu bịa để test khói |
| `tests/` | 119 phép kiểm tự động |

---

# CẤP 5 — Nâng cao: vì sao thiết kế như vậy

## 5.1 Vì sao 100 slot là "danh mục đầu tư"

Gọi $\theta$ là sự thật (video nào, cửa sổ ở đâu, đáp án là gì). Điểm kỳ vọng:

$$E[\text{Điểm}] = \frac{1}{5}\sum_k E_\theta\left[\max_{i\le k} R(r_i;\theta)\right]$$

Vì `max` trên tiền tố là **một dãy tăng dần**, nó tách được thành các phần tăng.
Nếu dòng đặt ở slot `pos` làm dãy đó tăng thêm $\delta$, phần tăng ấy được **mọi
mốc $k \ge pos$ đếm**, tức đúng $w(pos)$ lần:

$$w(1)=5,\quad w(2..5)=4,\quad w(6..20)=3,\quad w(21..50)=2,\quad w(51..100)=1$$

Suy ra hai điều:

1. **Tham lam (greedy) là đúng thuật toán** — mỗi slot cứ lấy dòng có phần tăng
   kỳ vọng lớn nhất.
2. **Dòng không tăng thêm gì thì đáng 0 điểm — nhưng cũng không tốn gì**, vì luật
   không phạt sai. Nên vẫn điền đủ 100.

Trường hợp đơn giản nhất, và đó là toàn bộ chiến thuật KIS trong một câu:

> Gom các giả thuyết vào từng **nhóm loại trừ nhau** (cùng nhóm = cùng đúng
> trong một hoàn cảnh), **giữ đúng 1 dòng mỗi nhóm**, xếp nhóm theo xác suất giảm dần.

## 5.2 Vì sao cần calibration

Bộ tìm kiếm trả về "điểm". Nhưng điểm 0.9 **không có nghĩa 90% đúng**. Mà thuật
toán xếp slot cần **xác suất thật** mới so sánh được các cửa với nhau.

`calibration.py` làm việc đổi đó (Platt / Isotonic / Rank), kèm thước đo:

- **ECE** — sai lệch giữa "tự tin 80%" và "thực tế đúng 80%"
- **Brier** — sai số bình phương của xác suất
- **risk-coverage** — nếu chỉ trả lời 50% câu tự tin nhất thì sai bao nhiêu?

## 5.3 Vì sao TRAKE cần k-best chứ không phải đáp án tốt nhất

Luật TRAKE:

$$R = \begin{cases} 0 & \text{sai video} \\ \dfrac{1}{N}\sum_j \mathbb{1}(\text{khung}_j \in \text{cửa sổ}_j) & \text{đúng video} \end{cases}$$

Hai điều rút ra:

- **Có điểm từng phần.** Đúng video, trúng 1/3 bước = 0.333.
- **Đúng video rồi thì mọi slot còn giá trị.** Một cách ghép chuỗi khác của cùng
  video có thể trúng những bước mà cách tốt nhất trượt.

Mà cửa sổ chấm TRAKE **dưới 10 khung** (~0.4 giây ở 25fps), trong khi keyframe
BTC cấp cách nhau **2.16 giây** (trung vị). Nên chuyện cách ghép tốt nhất lệch
một keyframe ở một bước là **bình thường, không phải ngoại lệ**.

→ Vì vậy `align_k_best` trả **k chuỗi khác nhau**, để tầng `optimize` rải slot
lên nhiều chuỗi của cùng một video đúng.

## 5.4 Vì sao giữ cả hai cách xếp slot

Không phải để lưu niệm. Mà vì `run_benchmark.py --ab` chạy **cùng một bộ tìm
kiếm** qua cả hai cách rồi in chênh lệch — đó là cách **duy nhất** biết tầng
`optimize` đáng giữ hay đáng xoá.

Mặc định `enabled=False` cho tới khi có số trên dữ liệu thật. **Nếu đo ra thua,
xoá cả thư mục `optimize/`, không tiếc.**

---

# CẤP 6 — Hiện trạng thật

*Cập nhật 2026-08-07, sau khi nối được dữ liệu thật.*

## Đã có

- ✅ Đường ống chạy trọn vẹn, **119 phép kiểm** đúng
- ✅ Công thức chấm điểm đúng luật BTC (đã đối chiếu file PDF)
- ✅ Sinh file nộp bài đúng định dạng, có kiểm tra trước khi đóng gói
- ✅ DP căn chuỗi cho TRAKE + bản k-best
- ✅ Tầng phân bổ 100 slot theo công thức chấm
- ✅ **Đọc được dữ liệu thật**: 873 video / 177.321 keyframe, khớp audit
- ✅ **Kho vector chịu quy mô thật**: 11 ms/truy vấn trên 177k
- ✅ **Đã chứng minh chạy đúng** ở quy mô thật (`kiem_chung_thuc_te.py`, 5/5 đạt)

## Chưa có

| Thiếu gì | Hậu quả |
|---|---|
| **Vector cho câu truy vấn chữ** | Có `CLIPTextEncoder` rồi nhưng máy chưa cài `transformers`. Phải encode trên Kaggle (`kaggle_encode_truy_van.py`) hoặc cài cục bộ |
| **Dev set thật đã convert** | Có query + đáp án AIC2025 batch 1 nhưng chưa biết nằm ở đâu |
| **Kênh OCR / ASR** | Bộ data BTC không có 2 thành phần này. Metadata (title/keywords) thay được một phần, đã đọc sẵn nhưng chưa bật |
| **Bộ trả lời câu hỏi QA** | `RuleBasedProvider` là stub |
| **`tol_frames` đã đo** | 25 và 5 là số **đoán**, chưa fit trên dữ liệu thật |
| **Khử trùng lặp** | Corpus có 904 nhóm vector trùng khít (6.494 keyframe, 3,7%); nhóm lớn nhất 216 keyframe trải **123 video**. Truy vấn rơi trúng hình hiệu → hàng trăm video hoà điểm, thứ tự ngẫu nhiên |
| **ID khung nộp bài** | Audit báo `frame_idx` lẫn quy ước, chưa chốt. **Rủi ro 0 điểm toàn tập** |

## Tin tốt về dữ liệu

Audit của nhóm trưởng cho biết BTC **đã encode sẵn CLIP** cho toàn bộ 177.321
keyframe: 512 chiều, đã chuẩn hoá L2, phủ 100%.

| Thành phần | Dung lượng | Cần không? |
|---|---|---|
| videos | 77.28 GiB | chưa cần |
| keyframes (ảnh) | 28.66 GiB | chưa cần |
| objects | 1.68 GiB | tuỳ chọn |
| **clip_features** | **173 MiB** | ⭐ cần |
| **keyframe_mapping** | **3.62 MiB** | ⭐ cần |
| **metadata** | **1.86 MiB** | ⭐ cần |

→ Chỉ cần tải **~179 MB**, không phải 30 GB.

## Việc tiếp theo, theo thứ tự

1. **Tải 3 thư mục nhẹ** (~179 MB)
2. **Viết loader** `core/dataset.py` — cầu nối dữ liệu thật vào hệ thống
3. **Thay `HashEmbedder`** bằng `CLIPFeatureEncoder` dùng vector BTC cấp sẵn
4. **Convert dev set** AIC2025 batch 1 → đo lần đầu có nghĩa
5. **Fit `tol_frames`**, chạy `--ab` → biết nên dồn sức vào đâu

> **Điểm mấu chốt:** sau bước 4 thì hết phải đoán. Mọi ưu tiên trước đó đều là
> phỏng đoán trên dữ liệu bịa.

---

## Tài liệu liên quan

- `README.md` — chi tiết kỹ thuật, luật đã xác nhận vs chưa xác nhận
- `giai_thich.py` — 6 thí nghiệm chạy được
- Backup trước khi tách kiến trúc:
  `D:\COMP\AIC2026\tự làm\_backup_system-thu_truoc-khi-tach_2026-08-07.zip`
