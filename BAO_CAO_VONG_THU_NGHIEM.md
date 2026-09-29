# Báo cáo — Vòng thử nghiệm AIC2026 (bộ đề 24 câu, phát 20/08/2026)

> Viết ngày **2026-08-20**. Bộ đề: `D:\COMP\AIC2026\data cuộc thi\THUNGHIEM-bo-de-thi\`
> Luật: `D:\COMP\AIC2026\Tài liệu\info cuộc thi\aic2026-vongsotuyen.md`
> Hệ thống: `D:\COMP\AIC2026\tự làm\system thử\` — **371 test xanh**

---

## 0. Tóm tắt điều hành

Đã dựng xong **hai** bài nộp hợp lệ cho vòng thử nghiệm, khác nhau đúng 3/24 file:

| file | là gì | dùng để |
|---|---|---|
| `out/NOP/aic2026thunghiemL1may.zip` | **thuần máy** — không có can thiệp của người | đo THẬT mức của hệ thống |
| `out/NOP/aic2026thunghiemL2ghim.zip` | máy + **3 video do người xác minh bằng metadata BTC** | đo giá trị của lớp người-trong-vòng-lặp |

Nộp **L1 trước, L2 sau**. Vòng thử nghiệm cho 3 lần nộp và **chỉ tính lần cuối**, nên
thứ tự này cho hai số đo mà vẫn giữ bản tốt hơn ở lần cuối, và còn dư 1 lần.

Trong lúc dựng bài đã tìm và sửa **5 lỗi**, trong đó **3 lỗi làm mất điểm hoàn toàn mà
không có triệu chứng gì** — zip vẫn hợp lệ, mọi phép nghiệm thu vẫn xanh. Nếu nộp ngay
lúc chưa sửa thì kết quả gần như bằng 0 và sẽ không hiểu tại sao.

**Dự đoán điểm public leaderboard** (thang 12,0 điểm — xem §6): bản thuần máy
**2,5 – 4,0**, nhiều khả năng quanh **3,2**. Bảng xếp hạng hiện tại có đỉnh 9,8 và trung
vị ~7,2, nên vị trí dự kiến nằm ở **nhóm cuối, khoảng hạng 29–34/44**.

**Kết luận quan trọng nhất của báo cáo này** (§8): khoảng cách tới nhóm đầu **không phải
do encoder yếu**. Chính số đo của hệ thống nói ra điều đó — nó tìm đúng khung trong
top-100 ở **57,5%** số câu, nhưng chỉ ăn điểm **34,5%**, vì không có ai NHÌN 100 khung đó
để kéo khung đúng lên hạng 1. Bịt đúng khe hở này (tải keyframe về + một trang xem ảnh)
đáng giá **+0,23 điểm/câu**, lớn hơn tổng mọi cải tiến thuật toán của cả tháng 8 cộng lại.

---

## 1. Cách nộp bài — từng bước

### 1.1 Chuẩn bị (đã xong)

File nộp đã đúng chuẩn, đã kiểm bằng máy chứ không phải bằng mắt:

```
aic2026thunghiemL1may.zip
└── submission/                    ← BẮT BUỘC phải có thư mục này
    ├── query-p1-1-kis.csv         100 dòng · 2 cột
    ├── query-p1-2-kis.csv         100 dòng · 2 cột
    ├── query-p1-4-trake.csv       100 dòng · 5 cột (video + 4 khung)
    ├── query-p1-15-qa.csv         100 dòng · 3 cột
    └── ... đủ 24 file
```

Đã xác minh: 24/24 file · đúng 100 dòng/file · UTF-8 · CRLF · không header ·
tên video không có `.mp4` · frame là số nguyên không khoảng trắng thừa ·
TRAKE đúng 4 cột khung · đáp án QA dài nhất 66 ký tự (trần 100) ·
tên zip chỉ gồm chữ và số theo khuyến cáo BTC.

### 1.2 Nộp

1. **Đăng nhập** hệ thống thi bằng **tài khoản BTC đã cấp** cho đội (không đăng ký gì
   thêm ở đâu khác). Mỗi đội **chỉ được dùng một tài khoản duy nhất**.
2. Vào đúng bảng của mình — **Thanh niên, sinh viên** (hoặc **Học sinh THPT**) — rồi vào
   vòng **Vòng thử nghiệm**.
3. Chọn tab **Nộp bài**.
4. Tải lên `aic2026thunghiemL1may.zip`. **Không giải nén, không đổi cấu trúc.**
5. Sang tab **Bài của tôi** kiểm hai điều **trước khi nộp lần tiếp theo**:
   - hệ thống có **parse được** không (nếu báo lỗi định dạng thì vẫn đã trừ 1 lần nộp);
   - điểm public hiện ra là bao nhiêu.
6. Ghi lại điểm. Rồi nộp `aic2026thunghiemL2ghim.zip` làm **lần 2**.
7. Ghi lại điểm lần 2. **Còn dư 1 lần** — để dành phòng khi phát hiện sai sót.

### 1.3 Ba điều dễ mất điểm nhất khi nộp

| Bẫy | Vì sao chết | Đã phòng chưa |
|---|---|---|
| Nén thẳng các file CSV thay vì nén **thư mục** `submission/` | BTC ghi rõ BẮT BUỘC có thư mục | ✅ zip đã dựng bằng code, luôn có |
| Tên file CSV không trùng tên file đề | Hệ thống ghép file với truy vấn **theo tên** → câu đó 0 điểm, zip vẫn hợp lệ | ✅ tên lấy nguyên văn từ tên file `.txt` |
| Nộp sai định dạng | **Vẫn tính là 1 trong 3 lần** | ✅ `nop_bai.py` tự nghiệm thu rồi `exit 1` nếu lệch |

---

## 2. Nhận xét bộ đề sơ tuyển

### 2.1 Cấu trúc

24 câu: **18 KIS · 3 Q&A · 3 TRAKE**. Đánh số `query-p1-1` … `query-p1-25`, **khuyết số 3**
(BTC bỏ một câu). Tiền tố `p1` = gói 1 — năm ngoái không có tiền tố này, và nó chính là
nguyên nhân của một lỗi nặng, xem §4.

**Toàn bộ đề ra bằng TIẾNG VIỆT, không có bản tiếng Anh.** Đây là điều đã lường trước và
đã có đối sách (§3), nhưng nó vẫn là khoản mất điểm lớn nhất của hệ thống.

### 2.2 Kho dữ liệu quyết định dạng câu hỏi

873 video của batch 1 gom thành mấy nhóm chương trình rất rõ:

| nhóm | nội dung | câu rơi vào |
|---|---|---|
| L21–L22 | bản tin **60 Giây** (HTV), mỗi video ~30 phút, nhiều mẩu tin ngắn | 1, 2, 8, 14, 21, 23 |
| L23 | tường thuật **đua xe đạp** (Cúp Truyền hình) | 24, 25 |
| L24 | **Lân Sư Rồng** Cúp Chợ Lớn | 16 (và 19 sau khi ghim) |
| L26 | **Món ngon mỗi ngày** (VIVU TV), ~460 tập | 4, 6, 12, 18, 20, 22 |
| L27 | **Việt Nam đi là ghiền** (du lịch) | 5, 9 |
| L28–L29 | **Tản Mạn Mê Kông**, **Đôi Mắt Mê Kông** | 7 |
| L30 | **Lan tỏa năng lượng tích cực** (phóng sự cộng đồng) | 10, 11, 13, 15, 17 |

Nhận xét: đề được ra rất đều tay trên các nhóm chương trình. Không có câu nào rơi vào
vùng trống của kho.

### 2.3 Bốn loại độ khó, rất khác nhau

**(a) Tả cảnh thuần thị giác — DỄ NHẤT với hệ thống hiện tại** (câu 5, 6, 9, 10, 11, 12,
13, 20). Câu 20 là ví dụ mẫu: "ly panna cotta, lát nho đỏ, lá bạc hà, hai bông hoa ăn
được". Toàn danh từ cụ thể, màu sắc rõ, đúng thứ encoder ảnh giỏi.

**(b) Cần TRI THỨC NGOÀI mới dịch được sang hình ảnh — nguy hiểm nhất** (câu 1, 7, 21, 23).
Câu 23 không hề nói chữ "cá mập": nó nói *"loài động vật biển nguy hiểm nổi tiếng trong
phim của Steven Spielberg năm 1975"*. Encoder ảnh không suy luận được. Muốn tìm được thì
phải có một tầng biết `Jaws (1975) → cá mập trắng`. Tương tự: câu 21 `Lausanne → EPFL`,
câu 7 `chim lông đen ánh xanh, mắt đỏ, Nam Bộ → chim cu ngói / chích choè`, câu 1
`phóng tàu tư nhân + 4 phi hành gia + cực quang vùng cực → chuyến bay quanh cực`.
Lớp LLM viết lại truy vấn CÓ giải được một phần việc này — và đó là lý do nó đáng giá hơn
nhiều so với "chỉ để dịch".

**(c) Mẩu tin ngắn chôn trong video dài — khó về mặt tỷ lệ** (câu 1, 2, 8, 14, 21, 23).
Một bản tin "60 Giây" dài ~30 phút chứa 15–20 mẩu tin; mẩu cần tìm chỉ vài giây. Tệ hơn:
`media-info` của các video này **chỉ có boilerplate quảng cáo kênh**, không liệt kê từng
mẩu tin (đã mở ra xem tận mắt). Nghĩa là **kênh chữ hoàn toàn vô dụng** với 6 câu này,
buộc phải tìm thuần thị giác giữa 177.321 khung.

**(d) Đọc chữ trong ảnh — hệ thống hiện KHÔNG có kênh này** (câu 15, 17, 19, 22).
Cả 3 câu Q&A đều là câu đọc chữ: tên xã trên băng-rôn, hai câu thơ trên câu đối, tiêu đề
trên tờ công thức nấu ăn. Bộ dữ liệu BTC **không cấp OCR**, và hệ thống chưa tự dựng.
Đường duy nhất là VLM đọc ảnh — mà muốn thế thì phải có ảnh keyframe (§7).

### 2.4 Hai lỗi trong chính đề bài

- `query-p1-18-trake.txt` đánh nhãn sự kiện là **E1, E2, E2, E4** — trùng E2, thiếu E3.
  Bốn sự kiện, đánh nhãn sai một chỗ. Bộ đọc đề nay lấy **thứ tự xuất hiện** làm thứ tự
  thời gian và bỏ qua con số trong nhãn, nên vẫn ra đúng 4.
- `query-p1-14-kis.txt` thiếu dấu chấm câu ("...hai người trượt ván Nền phía sau là...").
  Không ảnh hưởng.

### 2.5 Đánh giá chung về đề

Đề **ra tốt và khó đúng chỗ**. Nó không thưởng cho việc nhồi thêm tham số hay encoder to
hơn; nó thưởng cho ba thứ khác: (1) hiểu tiếng Việt, (2) suy luận tri thức ngoài để dịch
câu chữ thành hình ảnh, (3) đọc được chữ trong khung hình. Ba thứ này đúng là ba khe hở
của hệ thống hiện tại, xếp theo đúng thứ tự đó.

---

## 3. Hệ thống đem đi thi

```
đề .txt (tiếng Việt)
   │  doc_de_thi.py          → tách loại câu, chuỗi sự kiện TRAKE, câu hỏi QA
   ▼
LLM viết lại truy vấn         gemini-3.5-flash-lite, 3 bản, lấy bản [0]
   │                          (VI thô 0.2883 → 0.3450, Δ +0.0567 ± 0.0209)
   ▼
SigLIP2 so400m-384 (1152 chiều) mã hoá truy vấn
   │                          + khử modality gap
   ▼
HỢP NHẤT 2 ENCODER            RRF có trọng số trên hạng: so400m ×2 + base-224 ×1
   │                          (0.3400 → 0.3750, Δ +0.0350 ± 0.0127)
   ▼
177.321 keyframe / 873 video
   │
   ├── KIS   → 100 dòng, đa dạng hoá video ở phần đuôi
   ├── QA    → VLM Gemini đọc top-10 ảnh (CẦN ẢNH — xem §7), đáp án tiếng Việt
   └── TRAKE → chấm lại toàn timeline từng video ứng viên + căn thời gian k-best
   ▼
GHIM (mới hôm nay)            người xác minh video bằng metadata → ghim lên đầu
   ▼
submission/*.csv → zip
```

Mọi con số trong sơ đồ là **số đo ghép cặp trên dev set 170 câu tự làm**, không phải ước
lượng. Những gì đã thử và **bị loại vì đo ra kém**: dịch máy opus-mt (−0.0400), rerank
theo lân cận thời gian (−0.0317), gộp bằng chứng cấp video, khử vector trùng, BEiT-3,
ensemble truy vấn ở so400m.

---

## 4. Năm lỗi tìm ra hôm nay

Ba lỗi đầu đều thuộc loại tệ nhất: **chạy trơn tru, zip hợp lệ, nghiệm thu xanh, chỉ có
điểm là sai**.

### 🔴 Lỗi 1 — cả 21 câu KIS/QA đi tìm bằng chuỗi `"query-p1-1-kis"`

Bộ đọc đề ghi kèm khoá `ten_file` vào mỗi câu để tên CSV nộp luôn khớp tên file đề. Bộ
chọn ngôn ngữ có một phép quét dự phòng nhặt mọi khoá lạ dài **≥ 12 ký tự** làm ứng viên
văn bản truy vấn. Chuỗi `"query-p1-1-kis"` dài **14** ký tự, không có dấu tiếng Việt nên
bị xếp là `en`; chính sách cũng là `en`; nó **thắng** bản mô tả tiếng Việt thật.

Kết quả: một lượt chạy đầy đủ trên 177.321 khung, ra 24 file CSV hợp lệ, và **21 câu đi
tìm bằng đúng chuỗi mã truy vấn**. Bản in vẫn ✓ xanh vì nó chỉ ghi *tên khoá*.

Vì sao chưa bao giờ lộ: tên năm ngoái `query-1-kis` dài **11** ký tự — dưới ngưỡng 12.
Mã gói `p1` của năm nay đẩy nó qua ngưỡng. Chưa lộ là **may**, không phải thiết kế đúng.

Đã sửa ba lớp: (a) `ten_file` vào danh sách khoá bỏ qua; (b) khoá lạ phải là **một CÂU**
(≥ 3 từ) mới được coi là văn bản — chặn theo hình dạng, không phụ thuộc đoán trước tên
khoá; (c) bản in nay in **nội dung** chứ không chỉ tên khoá.

### 🔴 Lỗi 2 — hai câu TRAKE bị đọc thành 5 sự kiện thay vì 4

`query-p1-16-trake` và `query-p1-18-trake` có một **câu dẫn** đứng trước danh sách E1..E4.
Bộ đọc cũ coi mọi dòng là một sự kiện → 5. Luật BTC: *"Số lượng Frame ID phải khớp chính
xác với số events yêu cầu"* → **R-Score 0 cho cả truy vấn**, mà zip vẫn hợp lệ.

Đã sửa: hễ có ≥ 2 dòng mang nhãn `E<n>` thì chỉ những dòng đó là sự kiện; câu dẫn được
**ghép vào đầu mỗi sự kiện** làm ngữ cảnh cấp video (TRAKE sai video là 0 điểm, nên độ
chính xác cấp video đáng giá hơn độ sắc nét giữa các sự kiện).

### 🔴 Lỗi 3 — "ghim" không ghim được gì

`MultiChannelRetriever.index()` **cộng dồn** chứ không thay thế — đọc tên hàm không ai
đoán ra. Lớp ghim thu chỉ mục xuống 204 keyframe của một video, nhưng 177.321 khung cũ
vẫn nằm nguyên trong kho, nên CSV "đã ghim `L24_V011`" vẫn mở đầu bằng `L28_V007`.
Kèm theo: `RetrievalService._hop_nhat` có nhánh dự phòng cho event lạ đi thẳng ra kết quả,
và encoder phụ của lớp hợp nhất không hề bị cắt theo.

Đã sửa: thêm `reset()` cho cả ba store và cho retriever, bỏ encoder phụ trong lượt ghim,
và thêm một lưới cuối **ném lỗi** nếu còn dòng nào lọt ra ngoài video được ghim.

### 🟡 Lỗi 4 — 24/24 câu bị báo sai tên file

Mẫu tên `query-<số>-<loại>` không khớp `query-p1-<số>-<loại>` của năm nay. Cảnh báo sai
còn nguy hơn không có cảnh báo: nó dạy người đọc bỏ qua đúng dòng lẽ ra phải chặn họ lại.

### 🟡 Lỗi 5 — `query-p1-15-qa` bị coi là không có câu hỏi

Câu hỏi kết thúc bằng `? (tại thời điểm đó)` — dấu `?` không ở cuối dòng. Bộ đọc trả câu
hỏi rỗng, tầng đáp án đoán mò.

**Mỗi lỗi đều kèm test hồi quy dán nguyên văn nội dung đề thật. 371 test xanh.**

---

## 5. Tự kiểm chất lượng khi không có đáp án

Ngày thi không có ground truth. Nhưng bộ dữ liệu nhẹ của BTC có `media-info/<VIDEO>.json`
với **title 100% coverage, keywords 100%**. Đã viết hai công cụ mới:

- `soi_ket_qua.py` — đọc zip, in cho từng câu: top-N dòng kèm **tiêu đề video**, số video
  khác nhau trong 100 dòng, và cảnh báo khi video trùng từ nhiều nhất KHÔNG đứng hạng 1.
- `tra_metadata.py` — tra vét cạn tiêu đề/keywords/mô tả, ~10 giây, 0 request.

### Kết quả soi bản thuần máy

**Video hạng 1 gần như chắc chắn ĐÚNG** (8 câu — đọc tiêu đề là biết):

| câu | truy vấn | video hạng 1 | tiêu đề |
|---|---|---|---|
| 5 | hai phụ nữ cho dê ăn | L27_V014 | *Một ngày làm nông dân chăn dê* |
| 6 | gỏi cuốn chay, đậu hũ | L26_V056 | *CUỐN CHAY ĐẬU HỦ RAU RỪNG* |
| 9 | thu hoạch dứa miền Tây | L27_V013 | *Trăm Năm làng Khóm* |
| 10 | nhạc cụ kim loại tròn rỗng | L30_V017 | *Thanh âm diệu kỳ của trống Hang-drum* |
| 11 | mảnh bìa đổ bóng thành chân dung | L30_V057 | *Hãy làm cuộc sống trở nên thú vị* |
| 12 | trang trí bánh rán | L26_V279 | *BÁNH RÁN SOCOLA XỐT TÁO* |
| 17 | trao quà trẻ mồ côi COVID | L30_V091 | *Hãy cho nhau những gì ta có thể!* |
| 20 | panna cotta | L26_V004 | *PANNA COTTA KEM MUỐI* — **video panna cotta duy nhất trong 873 video** |

**Video hạng 1 gần như chắc chắn SAI** (6 câu): 1, 2, 8, 14, 21, 23 — tất cả đều là mẩu
tin ngắn trong bản tin 60 Giây hoặc cần tri thức ngoài. Câu 21 (Lausanne/robot bọ) hạng 1
là *"BÍ QUYẾT ÔN THI THPT — Môn Hóa học"*, tức tầng tìm đang lạc hoàn toàn.

### Ba ghim đã đưa vào bản L2

| câu | ghim | bằng chứng | slot |
|---|---|---|---|
| `query-p1-19-qa` | `L24_V011` | `tra_metadata.py "nguyen trung truc"` → **đúng 1/873 video**: *"…tưởng nhớ anh hùng Nguyễn Trung Trực — Đình Nguyễn Trung Trực — Kiên Giang"*. Máy để video này **ngoài cả 100 dòng**. Đáp án là câu đối Huỳnh Mẫn Đạt khắc tại đình. | 100/100 |
| `query-p1-4-trake` | `L26_V194` | `"mang tay"` → 6 video, **chỉ một** có chiên: *MĂNG TÂY CHIÊN BIA XỐT CÁ NGỪ*. Chuỗi sự kiện của đề (bột vào tô → chạm dầu → rời chảo → lên dĩa) đúng quy trình tẩm bột chiên; 5 video kia là salad/xào/súp, không có chảo dầu. Máy để nó hạng 4. | 50/100 |
| `query-p1-20-kis` | `L26_V004` | `"panna cotta"` → **đúng 1/873**. Máy đã tự để hạng 1, nên ghim ở đây không sửa video mà sửa **phân bổ slot**: bản gốc rải 100 dòng ra 48 video, ghim kéo 50 dòng đầu về đúng video. | 50/100 |

Núm `so_dong` đặt theo hình dạng thang điểm, không theo cảm tính: ghim 50 slot đầu = nếu
ghim **đúng** thì ăn trọn 4/5 số hạng (k = 1, 5, 20, 50) kể cả khi bài của máy vô dụng;
nếu ghim **sai** thì vẫn còn 50 slot của máy để cứu số hạng k = 100.

---

## 6. Dự đoán điểm

### 6.1 Thang điểm

`R-Score` của một dòng: KIS đúng khi `video = GT ∧ frame ∈ [s,e]`; QA thêm điều kiện
`answer = GT`; TRAKE **0 nếu sai video**, đúng video thì `(1/N)·Σ I(id_j ∈ [s_j,e_j])`
(có điểm thành phần).

`Final = (1/5)·Σ_{k∈{1,5,20,50,100}} R@k` với `R@k` = max trên 100 dòng đầu. Nghĩa là
điểm mỗi câu chỉ nhận 6 giá trị:

| khung đúng nằm ở hạng | 1 | 2–5 | 6–20 | 21–50 | 51–100 | không có |
|---|---|---|---|---|---|---|
| điểm câu | **1,0** | 0,8 | 0,6 | 0,4 | 0,2 | 0 |

Public leaderboard tính trên **50% đáp án** = 12 câu → **thang tối đa 12,0**. Khớp với
bảng xếp hạng thật: mọi điểm đều là **bội của 0,2** (9,8 · 9,6 · 9,4 · 9,0 …).

### 6.2 Ước lượng

Số đo gần nhất, đúng cấu hình thi, 120 câu KIS tiếng Việt + LLM viết lại: **Final 0,3450**,
**R@100 = 57,5%**. Điều chỉnh cho đề thật:

| loại | số câu | điểm/câu ước tính | căn cứ |
|---|---|---|---|
| KIS nhóm (a) tả cảnh | 8 | 0,45 – 0,60 | soi metadata thấy video hạng 1 đúng |
| KIS nhóm (b)(c) tri thức ngoài / mẩu tin | 10 | 0,05 – 0,15 | soi metadata thấy hạng 1 sai rõ |
| TRAKE | 3 | 0,10 – 0,25 | dev 0,2733 nhưng cửa sổ chấm thật hẹp hơn nhiều |
| Q&A | 3 | ~0,00 (bản L1) | không có ảnh → không đọc được chữ |

Tổng trên 24 câu ≈ **5,5 – 8,0**, tức trung bình **0,23 – 0,33 điểm/câu**.
→ **Public LB (12 câu) ≈ 2,8 – 4,0**, điểm giữa **~3,2**.

Bản **L2 (có ghim)** cộng thêm ước chừng **+0,3 – +0,9** trên thang 12, chủ yếu từ câu 19
(từ chắc chắn 0 thành có cửa) và câu 4.

### 6.3 So với bảng xếp hạng

Đỉnh **9,8/12 = 81,7%**. Trung vị ~7,2. Với ~3,2 thì vị trí rơi vào **hạng 29–34/44**.

Con số 81,7% nói lên một điều quan trọng, và nó không phải chuyện encoder: **không có
mô hình tự động nào của một đội sinh viên đạt 80% R@1 trên bài toán này**. Các đội đầu
bảng gần như chắc chắn có **người ngồi xem ảnh** và chọn khung. Đó là §8.

---

## 7. Điểm mạnh và điểm yếu

### Điểm mạnh

1. **Kỷ luật đo lường.** Mọi lựa chọn kiến trúc đều có số ghép cặp kèm sai số chuẩn, và
   có cả một danh sách "đã thử, đo ra kém, đã bỏ". Rất hiếm ở mức đồ án sinh viên, và nó
   chính là thứ đã giữ hệ thống không đi lạc suốt tháng 8.
2. **Chống lỗi câm.** 371 test, và mỗi lỗi từng gặp đều có một test dán nguyên văn dữ liệu
   thật gây ra nó. Đường nộp bài tự nghiệm thu rồi `exit 1`.
3. **Đường nộp bài đã diễn tập thật**, không phải lý thuyết. Từ thư mục `.txt` của BTC ra
   zip hợp lệ bằng hai lệnh.
4. **Chuỗi dự phòng LLM 6 tầng** (5 model Gemini + Cloudflare) — không chết giữa buổi thi.
5. **Xử lý tiếng Việt bằng LLM viết lại, không phải dịch máy.** Đây là phát hiện thật và
   ngược trực giác: dịch máy làm HỎNG điểm (−0.0400), viết lại thành chú thích ảnh tiếng
   Anh thì ăn (+0.0567). Lấy lại 65% khoảng cách so với bản người gõ tay.

### Điểm yếu, xếp theo mức thiệt hại

1. **🔴 KHÔNG CÓ NGƯỜI TRONG VÒNG LẶP.** Hệ thống tìm đúng khung trong top-100 ở 57,5% số
   câu nhưng chỉ ăn 34,5% điểm. Chênh lệch **0,23 điểm/câu** nằm ở chỗ không ai nhìn
   100 khung để kéo khung đúng lên hạng 1. Máy không có ảnh keyframe để mà hiển thị.
2. **🔴 KHÔNG CÓ KÊNH ĐỌC CHỮ (OCR).** Cả 3 câu Q&A của đề này đều là câu đọc chữ trong
   khung hình. Không có OCR thì mất trắng 12,5% số câu, và còn mất cả những câu KIS có
   chữ trên băng-rôn (câu 17 chẳng hạn).
3. **🟠 Không suy luận được tri thức ngoài ở tầng tìm.** "Phim Spielberg 1975" phải thành
   "cá mập" trước khi encode. LLM viết lại có làm một phần nhưng chưa được giao việc này
   một cách tường minh, và chưa có phép đo riêng cho nhóm câu đó.
4. **🟠 Mẩu tin ngắn trong video dài.** Không có bước nào thu hẹp phạm vi theo mẩu tin.
5. **🟡 TRAKE bấp bênh về cấu trúc.** Keyframe của BTC cách nhau trung vị **2,16 giây**
   trong khi cửa sổ chấm TRAKE "thường dưới 10 khung" ≈ **0,4 giây**. Nghĩa là ngay cả khi
   chọn đúng keyframe gần nhất thì xác suất rơi vào cửa sổ cũng chỉ ~18%/sự kiện. Đây là
   giới hạn của **dữ liệu**, không phải của mô hình — muốn vượt phải giải mã khung dày.
6. **🟡 Phụ thuộc Kaggle cho đường ảnh.** 28,66 GiB keyframe không có ở máy local, nên QA
   bằng VLM chỉ chạy được trên Kaggle. Bản L1/L2 hôm nay chạy `--qa-cheap`.

---

## 8. Lộ trình khắc phục, xếp theo lợi ích chia chi phí

### ⭐ Việc số 1 — TẢI KEYFRAME VỀ MÁY + DỰNG TRANG XEM ẢNH (≈ 4 giờ)

Đây là việc đáng làm nhất, và không phải vì nó "hay" mà vì **chính số đo của hệ thống chỉ
thẳng vào nó**:

```
R@100  = 57,5%      ← hệ thống ĐÃ tìm thấy khung đúng, nó nằm đâu đó trong 100 dòng
Final  = 0,3450     ← nhưng chỉ ăn được chừng này, vì khung đúng không ở hạng 1
```

Nếu có một trang hiện 100 ảnh dạng lưới và người chỉ cần bấm chọn ảnh đúng, điểm câu đó
nhảy thẳng lên **1,0**. Trần lý thuyết của việc này là **0,575 điểm/câu**, tức
**+0,23 so với hiện tại** — lớn hơn tổng của toàn bộ: đổi sang SigLIP2 (+0,108) + hợp nhất
encoder (+0,035) + LLM viết lại (+0,057) cộng lại.

Và còn đi xa hơn được: phân tích `chan_doan_qa_truot.py` đã đo trần theo độ sâu pool —
xem tới **D = 500** thì trần lên **75%**, tới D = 2000 thì **88%**. Với 24 câu và 3 giờ
làm bài, một người xem 500 ảnh/câu là hoàn toàn khả thi (≈ 7 phút/câu).

Ước tính điểm sau việc này: **6,9 – 9,0 trên thang 12** → **hạng 5–22**.

Việc cụ thể:
- tải bộ keyframe (28,66 GiB) về ổ D, hoặc tối thiểu là bản thu nhỏ 224px (~3 GiB);
- một trang HTML tĩnh: đọc CSV → hiện lưới ảnh → bấm chọn → xuất CSV đã xếp lại;
- nối vào `ghim_video.py`/`--ghim` đã có, để lựa chọn của người đi thẳng vào bài nộp.

### ⭐ Việc số 2 — OCR cho keyframe (≈ 3 giờ máy + 1 đêm GPU Kaggle)

Cả 3 câu Q&A của đề này đều cần đọc chữ. Có sẵn keyframe rồi thì chạy PaddleOCR /
VietOCR trên toàn bộ 177.321 khung là việc một đêm trên Kaggle. Sau đó kênh `ocr_text`
trong `MultiChannelRetriever` — **vốn đã được dựng sẵn từ đầu và đang rỗng** — tự có dữ
liệu, không phải sửa kiến trúc.

Lợi: 3 câu Q&A từ ~0 lên có cửa thật; thêm cả các câu KIS có chữ (câu 17: băng-rôn
*"Trao kinh phí hỗ trợ cho trẻ em mồ côi do dịch COVID-19"* — một truy vấn OCR là ra ngay).

### ⭐ Việc số 3 — giao cho LLM việc SUY LUẬN TRI THỨC, không chỉ viết lại (≈ 1 giờ)

Đổi prompt của `viet_lai_truy_van` để nó **giải mã ẩn ý** trước khi tả cảnh:
`"phim của Steven Spielberg năm 1975"` → `"a great white shark near a beach town"`.
Hạ tầng đã có đủ, chỉ là sửa prompt + đo lại. Nhóm (b) có 4/24 câu trong đề này.

### Việc số 4 — kênh chữ từ metadata, có trọng số thấp (≈ 2 giờ)

`meta_vao_kenh_chu` đang TẮT vì đo ngày 08/08 ra −0.0300. Nhưng phép đo đó chạy trên dev
set **tiếng Anh** còn tiêu đề BTC là **tiếng Việt** — hai bên gần như không có token
chung nên RRF chỉ chèn nhiễu. **Tiền đề đó nay không còn đúng**: đề thi ra tiếng Việt.
Đã thấy tận mắt hôm nay là tra tiêu đề định vị được video duy nhất cho 3 câu. Cần: hạ
trọng số kênh chữ trong RRF, tách token tiếng Việt tử tế, rồi đo lại trên dev set bản VI.

⚠ Giới hạn phải nhớ: **vô dụng với 6 câu rơi vào bản tin 60 Giây** (mô tả chỉ là
boilerplate quảng cáo kênh).

### Việc số 5 — giải mã khung dày quanh ứng viên TRAKE (≈ 3 giờ)

Cửa sổ chấm 0,4 giây so với keyframe cách nhau 2,16 giây là bài toán **thiếu mẫu**. Sau
khi đã khoanh được video, giải mã lại đoạn đó ở 25 fps rồi chấm dày sẽ nâng trần TRAKE.
Rẻ hơn: bật `portfolio` chỉ cho TRAKE để rải các khung lệch ±10…±60 quanh mỗi mốc — hạ
tầng `trake_jitters` đã có sẵn, chỉ chưa được bật vì phép A/B toàn cục chưa vượt 2·SE.
**Nên đo lại riêng cho TRAKE**, đừng để phép đo gộp che mất.

### Không nên làm

- Đổi/thêm encoder ảnh nữa. Đã đo: CLIP BTC → SigLIP2 base → so400m → hợp nhất, mỗi bước
  càng về sau càng ít lãi, và tất cả cộng lại vẫn nhỏ hơn việc số 1.
- Dịch máy truy vấn. Đã đo ra **âm**.
- Rerank theo lân cận thời gian. Đã đo: 0.3750 → 0.3433.

---

## 9. Checklist cho Sơ tuyển đợt 1 — **19:30–22:30 ngày 21/08/2026**

> Chỉ **3 giờ**, và vòng thử nghiệm đóng lúc **19:30 cùng ngày** — tức không có khoảng
> nghỉ giữa hai vòng. Mọi thứ phải sẵn sàng **trước 19:00**.

**Ngân sách thời gian đã đo (không phải ước lượng).** Chạy đủ 24 câu trên máy này, cache
nóng: **40,4 giây**. Lần chạy nguội thêm ~25 giây nạp dữ liệu + ~30 giây nạp model + vài
giây gọi Gemini viết lại truy vấn. Hai lần chạy độc lập cho zip **giống nhau từng byte**
(đã kiểm) — hệ thống tái lập được, nên có thể chạy đi chạy lại thoải mái trong 3 giờ làm
bài. Nút cổ chai của đêm thi là **mắt người**, không phải máy.

**Trước 19:00**
- [ ] `python -X utf8 -m pytest -q` → phải xanh
- [ ] `python kiem_khoa_api.py` → xác nhận khoá Gemini còn quota (1 request/provider)
- [ ] Nếu chạy Kaggle: notebook đã có `!pip install -U "google-genai>=2.3.0"` **và đã
      restart kernel** (bản cài sẵn của Kaggle < 2.0.0 làm mọi lời gọi Gemini trả
      `400 legacy Interactions API`, đọc y hệt lỗi hết quota)
- [ ] Nạp trước dữ liệu để cache OS nóng (lần đầu 23 giây, sau đó 2 giây)
- [ ] Mở sẵn `tra_metadata.py` trong một cửa sổ terminal riêng

**19:30 — nhận đề**
1. Tải thư mục `.txt` của BTC về `de_dot1/`
2. `python -X utf8 doc_de_thi.py de_dot1 --ra data/de_dot1.json --xem 3`
3. **ĐỌC HẾT PHẦN CẢNH BÁO.** Đặc biệt: đếm bằng mắt số sự kiện của từng câu TRAKE trong
   đề gốc và đối chiếu với con số script in ra. Sai số sự kiện = 0 điểm cả câu.
4. `python -X utf8 nop_bai.py --config config.yaml --truy-van data/de_dot1.json --ra out/dot1_L1.zip`
   → **Đọc dòng «…» của từng câu**: đó là chuỗi thật đem đi tìm. Nếu thấy mã truy vấn
   hay chuỗi lạ ở đó thì dừng lại ngay.
5. `python -X utf8 soi_ket_qua.py out/dot1_L1.zip --de data/de_dot1.json --ra out/soi_dot1.md`
6. **Nộp L1 ngay** (~20:15). Đừng chờ hoàn hảo — có một điểm thật trong tay trước đã.

**20:15–21:45 — vòng người**
7. Với mỗi câu mà `soi_ket_qua.py` báo tiêu đề hạng 1 không ăn nhập: chạy
   `tra_metadata.py "<từ khoá>"`. Câu nào ra **đúng một video** thì ghi vào `data/ghim_dot1.json`
   **kèm lệnh đã dùng** ở trường `ghi_chu`.
8. `python -X utf8 nop_bai.py ... --ghim data/ghim_dot1.json --ra out/dot1_L2.zip`
9. Nộp L2 (~21:45).

**Giữ nguyên lần nộp thứ 3 làm dự phòng.** Chỉ dùng khi phát hiện lỗi thật, vì
**chỉ lần cuối được tính**.

---

## 10. Phụ lục — file mới/đổi trong phiên này

| file | việc |
|---|---|
| `aic2026/core/ghim.py` | **mới** — lớp ghim video/đáp án của người vận hành |
| `soi_ket_qua.py` | **mới** — soi bài nộp bằng metadata, không cần đáp án |
| `tra_metadata.py` | **mới** — tra vét cạn tiêu đề/keywords/mô tả video |
| `tests/test_ghim.py` | **mới** — 15 test |
| `aic2026/core/de_bai.py` | sửa lỗi 1 và 4; bản ghi in thêm nội dung truy vấn |
| `aic2026/core/doc_de_txt.py` | sửa lỗi 2, 4, 5 |
| `aic2026/core/retrieval.py`, `core/stores.py` | thêm `reset()` (lỗi 3) |
| `nop_bai.py` | thêm `--ghim` |
| `tests/test_de_bai.py`, `tests/test_doc_de_txt.py` | +9 test dán nguyên văn đề thật |
| `data/de_thunghiem.json` | đề đã chuyển sang JSON |
| `data/ghim_thunghiem.json` | 3 ghim, mỗi ghim kèm lệnh kiểm chứng |
| `out/NOP/aic2026thunghiemL1may.zip` | **bài nộp 1 — thuần máy** |
| `out/NOP/aic2026thunghiemL2ghim.zip` | **bài nộp 2 — có ghim** |
| `out/soi_v2.md` | bản soi đầy đủ 24 câu |
