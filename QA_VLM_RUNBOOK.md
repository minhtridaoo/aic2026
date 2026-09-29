# QA/VLM — trạng thái nghiệm thu và vận hành

## Kết quả ngày 2026-08-14

Baseline nguyên trạng, trước khi thêm `dap_an_alt` và dịch đầu ra:

- QA strict: **0.0000 ± 0.0000**
- Trần định vị: **0.1800 ± 0.0578**
- TRAKE: **0.2267 ± 0.0455**

Sau khi khai biến thể tiếng Anh cho 30/30 nhãn và dùng base-224:

- 13/30 dự đoán zero-shot đúng nghĩa bằng tiếng Anh.
- 5/30 vừa đúng nghĩa vừa có dòng chạm đúng khung.
- Điểm semantic-with-alt: **0.1333 ± 0.0570**.

Sau khi bật đầu ra tiếng Việt (bảng tay màu + số):

- 13/30 dự đoán khớp chính xác nhãn tiếng Việt (baseline tiếng Anh: 0/30).
- QA strict: **0.1333 ± 0.0570**.
- Khoảng cách tới trần định vị còn **0.0467**; trên 30 câu, mức này chưa đủ
  phân giải chắc chắn so với nhiễu khoảng ±0.03.

Chi tiết từng câu nằm ở `out/qa_predictions_en.csv` và
`out/qa_predictions_vi.csv`.

Với encoder chính so400m-384:

- QA semantic-with-alt tiếng Anh: **0.1067 ± 0.0467**, nhưng strict nhãn Việt
  vẫn **0.0000**.
- Sau Việt hoá đầu ra: QA strict **0.1067 ± 0.0467**; 16/30 đáp án đúng
  nghĩa, 12/30 câu có dòng chạm đúng khung.
- Trần định vị tăng lên **0.2133 ± 0.0614**; phần tầng trả lời còn có thể ăn là
  **0.1067**.
- TRAKE tăng từ **0.2267 ± 0.0455** lên **0.2733 ± 0.0481**.

## Hạ tầng đã chốt

Máy 4 GB VRAM chỉ chạy retrieval. **Chuỗi provider đã CHỐT 16/08:**

    Gemini native (chính)  ->  Cloudflare Workers AI (dự phòng, BẬT)  ->  QA cheap
    Claude Haiku: giữ nhưng TẮT (trả phí, chỉ bật khi hai đường trên chết)
    Grok/xAI: ĐÃ BỎ HẲN — không còn provider, không còn CLI, không còn trong config

Failover là **tuần tự**: Cloudflare chỉ được gọi khi Gemini đã ném lỗi, nên không
bao giờ gửi cùng một ảnh cho hai API. Hết cả chuỗi thì hạ xuống ClipZeroShotQA
tiếng Việt.

Gemini gọi **Google Gen AI SDK native** (`google-genai>=2.3.0`) qua
`client.interactions.create`; không có raw REST hay `generateContent`. Cloudflare
gọi REST Workers AI bằng `urllib` — **không cần SDK nào**, chỉ cần
`CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` (cần CẢ HAI). Nếu bật Anthropic,
timeout là 30 giây và không retry mặc định để failover không bị treo lâu.

**Kiểm khoá trước khi thi:** `python -X utf8 kiem_khoa_api.py` — 1 request mỗi
provider đang bật, cache tạm, in latency + đáp án. Đo 16/08: Gemini 6,96s trả
"Có" (đúng), Cloudflare 1,74s trả "Không" (sai) trên cùng một ảnh — đúng khoảng
cách chất lượng đã đo (0.2067 vs 0.1200), nên Cloudflare chỉ là đường khẩn cấp.

## Điểm VLM thật đủ 30 câu — 2026-08-14 (Anthropic Haiku 4.5)

Chốt đỏ "chưa có điểm VLM đủ 30 câu" đã **mở**, bằng `ANTHROPIC_API_KEY` chứ
không chờ quota Gemini: ảnh dev set nằm sẵn ở `dev_set/anh` (120 file
1280×720), nên phép đo 30 câu không cần corpus 28,66 GiB.

| phép đo | điểm | đáp án khớp |
|---|---|---|
| QA cheap (nền) | 0.1067 ± 0.0467 | 16/30 |
| QA VLM — prompt cũ, `max_tokens=32` | **0.0600 ± 0.0334** | 9/30 |
| QA VLM — prompt chung, `max_tokens=64` | **0.1800 ± 0.0586** | 23/30 |
| Trần định vị | 0.2133 ± 0.0614 | — |

Delta ghép cặp VLM − cheap: **+0.0733 ± 0.0571 = 1.28·SE**; thắng/hòa/thua theo
câu **5/24/1**. Vượt ngưỡng 0.06 đã đặt, **chưa vượt 2·SE** — xem "Kết luận".

**Lần đo đầu 0.0600 là lỗi thước đo, không phải model kém.** 10/30 câu mất điểm
vì dạng đáp án chứ không vì nhìn sai: `Áo trắng` vs `trắng`, `Hai người` vs `2`,
`Tạp chí,` (dấu phẩy cuối), và `Xám và` / `Xám (hoặc` bị `max_tokens=32` cắt
giữa câu. Hai sửa đổi:

1. `vlm_image_prompt()` — **một prompt dùng chung cho cả ba provider**. Trước đó
   Anthropic dùng prompt riêng KHÔNG có luật "hỏi số lượng trả lời bằng chữ số"
   mà Gemini đã có; mọi so sánh chéo provider trước mốc này đều không hợp lệ.
   Thêm luật chỉ-nêu-thuộc-tính và một-đáp-án-duy-nhất.
2. `max_tokens` 32 → 64 ở constructor, factory và config của cả ba provider.
3. Prompt hiện tại cho phép cụm 1–3 từ, giữ trọn màu ghép/cụm danh từ. Hậu xử lý
   không còn cắt cứng hai từ; số đếm và màu chỉ được chuẩn hóa khi đúng loại câu.

Cache hiện khóa theo `(model, prompt fingerprint, normalizer version, event_id,
câu hỏi)`. Sửa prompt/hậu xử lý sẽ tự cache miss; không cần dời file thủ công và
không thể vô tình đọc đáp án cũ. Cache lưu cả `raw_answer` để lần sau có thể đổi
hậu xử lý mà vẫn kiểm toán được. Bản cache prompt cũ vẫn giữ ở
`out/vlm_qa_cache_anthropic_claude-haiku-4-5.PROMPT-CU.json` để đối chứng.

Số vận hành thật (thay cho 1.01 s của provider GIẢ):

- Độ trễ VLM thật: **1.393 s/đáp án**, 41.8 s cho 30 ảnh tuần tự.
- Retrieval 4.633 s/câu; end-to-end TB 6.027 s, max 21.610 s.
- Chi phí thật: 60 request Haiku 4.5 (hai lượt đo) ≈ **0,09 USD**.
- Chạy lần hai: **30 cache hit · 0 đáp án mới · 0 API request.** ✔

Ở bản đo 0.1800, 7 câu còn trượt — 5 là model sai thật (`Đũa`/thìa,
`Đỏ`/trắng, `Đũa`/xẻng gỗ, `Vàng nhạt`/nâu, `4`/5), còn 2 là hiện vật thước đo:

- `qa-013`: đáp án thật "cây đánh trứng" bị `_short_answer` chặn 2 từ → `Cây đánh`.
- `qa-015`: `Xanh` vs `xanh dương` — câu này **trước đúng, sau khi siết prompt thì
  sai**. Luật "không kèm danh từ" đang áp hơi rộng, làm model bỏ cả bổ ngữ màu.

Hai hiện vật trên đã được sửa trong code nhưng **prompt v2 chưa được gọi cloud
đủ 30 ảnh**. Vì fingerprint cache đã đổi, lần nghiệm thu kế tiếp sẽ là 30 request
mới; không được gọi điểm 0.1800 là điểm của prompt v2 trước khi chạy lại.

### Kết luận

Tầng trả lời gần như đã cạn: 0.1800 so với trần 0.2133 chỉ còn **0.0333**. Nút
cổ chai QA quay lại **tầng tìm** — chỉ 12/30 câu chạm nổi khung, và đó chính là
thứ sinh ra trần 0.2133.

Chưa nên coi là "chốt VLM". Delta chỉ 1.28·SE và đứng trên 6 câu đổi kết quả;
theo đúng chuẩn 2·SE của dự án thì mức này **chưa phân giải được**. Muốn kết
luận chắc phải mở rộng nhãn QA quá 30 câu, không phải đo lại.

Trạng thái các provider cloud khác:

- Gemini native SDK đã trả thành công một ảnh: `xanh lá`, 1.180 input token,
  3 output token, tổng 1.509 token tính cả thinking; latency 17,7 giây.
- Cache phép đo 30 câu mới có **1/30** đáp án. Các request tiếp theo bị 429:
  `free_tier_requests, limit: 20, model: gemini-3.6-flash` dù đã retry theo
  cửa sổ server. Vì vậy chưa có điểm VLM đủ 30 câu.
- Grok key chạm được native gRPC nhưng team trả `PERMISSION_DENIED`: chưa có
  credits/licenses. Không có inference Grok và không phát sinh phí. *(16/08: Grok
  đã bỏ hẳn khỏi hệ thống — giữ dòng này chỉ để tra lịch sử.)*
- Unit test native SDK, batch/cache, watchdog, router loại câu, top-N hybrid và
  failover đều qua; toàn suite hiện **222 test xanh**. Hạ tầng corpus thật vẫn
  chưa đạt gate đi thi.

Không ghi API key vào source, YAML, notebook hay cache. Chỉ đặt
`GEMINI_API_KEY` / `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` /
`ANTHROPIC_API_KEY` trong secret hoặc biến môi trường của tiến trình. Nếu key
từng xuất hiện trong chat/log, phải thu hồi và tạo key mới. Đã quét toàn bộ
`D:\COMP\AIC2026` ngày 16/08: **không có khoá nào nằm trong file của hệ thống**
(còn một HF token hard-code trong notebook của đàn anh — không thuộc hệ này).

Trên Kaggle:

1. Mount dataset keyframe đầy đủ.
2. Đặt `qa.image_roots` tới thư mục mount và `llm.cache_path` tới
   `/kaggle/working/vlm_qa_cache.json`.
3. Đặt secret `GEMINI_API_KEY` (chính) và `CLOUDFLARE_API_TOKEN` +
   `CLOUDFLARE_ACCOUNT_ID` (dự phòng). Anthropic chỉ khi thật sự muốn trả phí.
4. Cài `requirements-vlm.txt`, chạy `kiem_khoa_api.py` rồi `kiem_resolver_anh.py`
   (cả hai ~0 quota) trước khi chạy tầng QA, rồi tải cache cùng CSV kết quả về.

Resolver cố ý ném `ImageResolutionError` khi thiếu ảnh. Không có đường lùi trả
chuỗi rỗng; vì vậy thiếu corpus sẽ lộ ngay trước khi nộp bài.

## Khối 3 — điểm Gemini thật (2026-08-16)

| số đo | giá trị |
|---|---|
| QA VLM (Gemini 3.6 flash, so400m) | **0.2067 ± 0.0618** |
| Trần định vị | 0.2133 ± 0.0614 |
| Khoảng cách ghép cặp (trần − thật) | +0.0067 ± 0.0067 |
| Đáp án khớp chuỗi/alt | **26/30** |
| Trần theo top-N | N=1 0.0667 · N=5 0.1467 · N=10 0.1667 · N=50 0.1800 · N=100 0.2133 |
| Tái lập | chạy lần hai: **30 cache hit · 0 API request** |

**Quota giải bằng batch, không bằng tiền.** Đo thật: batch 10 ảnh = **22,8 giây,
1 request**. Runtime với `vlm_top_n: 10` + `batch_max_images: 10` ⇒ **1 request
mỗi câu QA** (không phải 10). 24 câu QA = 24 request. Vì vậy giữ free tier;
Cloudflare là đường tràn khi hết quota.

**⚠ Hai giá trị phải đi cùng nhau.** `timeout_seconds: 10` + `batch_max_images: 10`
là bẫy đã đo: batch luôn quá hạn → failover trả lời bằng Cloudflare (0.1200) trong
khi config ghi Gemini (0.2067), **không có dấu hiệu gì ở đáp án**. Đã đặt
`timeout_seconds: 45` và thêm `config.canh_bao_cau_hinh()` để lần sau nó tự báo.
`dien_tap_qa_anh.py` in `last_provider` nên luôn thấy AI thật sự trả lời.

**Chính sách ĐUÔI (`qa.tail_answer`, mặc định `vlm` từ 16/08).** 90 dòng ngoài
top-N: dùng đáp án VLM phổ biến nhất thay cho đáp án rẻ. Đo ghép cặp 30 câu:
`0.1933 → 0.2067`, **+0.0133 ± 0.0093**, **0 request thêm** — bằng mức lợi của
việc tăng `vlm_top_n` lên 50 nhưng không tốn 5× request.

## Lệnh nghiệm thu

```powershell
# Baseline tiếng Anh + đủ 30 dự đoán
python do_diem_qa_trake.py --qa-en

# Đòn rẻ: bảng tay màu/số, đầu ra tiếng Việt
python do_diem_qa_trake.py

# Thêm Opus-MT cho lớp đáp án mở (model chỉ nạp khi bảng tay không phủ)
python do_diem_qa_trake.py --opus-en-vi

# Toàn tuyến ảnh/cache không gọi mạng
python do_diem_vlm_qa.py --fake

# Gemini native SDK + encoder so400m
python do_diem_vlm_qa.py --provider gemini --model gemini-3.6-flash --encoder so400m

# Thử native batch sau khi quota reset (một request cho tối đa 30 ảnh dev)
python do_diem_vlm_qa.py --provider gemini --model gemini-3.6-flash --encoder so400m --batch-max-images 30

# Auto mặc định chỉ Gemini; Cloudflare/Claude không tự bật dù có biến môi trường
python do_diem_vlm_qa.py --provider auto --encoder so400m --timeout 10 --max-retries 0

# Chỉ khi chủ động muốn Claude làm fallback
python do_diem_vlm_qa.py --provider auto --encoder so400m --enable-anthropic-fallback

# Cloudflare Workers AI: dự phòng đã BẬT trong config; đo riêng bằng
python do_diem_vlm_qa.py --provider cloudflare --encoder so400m
# Lần đầu dùng model Meta phải chấp nhận license (một lần, không chạy QA):
python do_diem_vlm_qa.py --provider cloudflare --cloudflare-agree-license

# Provider chính đã đo
python do_diem_vlm_qa.py --provider anthropic --model claude-haiku-4-5 --encoder so400m
```

Script VLM báo QA thật ± SE, trần định vị ± SE, khoảng cách ghép cặp, số cache
hit/backend call và độ trễ end-to-end. Thước đo chính vẫn là `answers_match` +
R-Score có nhãn; không dùng LLM-judge.

Ở chế độ `auto`, chuỗi cloud chỉ chạy tuần tự và dừng ngay khi một provider trả
đủ batch. Nếu tất cả cùng lỗi, evaluator không crash: nó tận dụng đáp án VLM đã
có trong cache và dùng ClipZeroShotQA cho phần còn lại.
CSV có cột `answer_source`; console ghi `QA HYBRID` và đếm riêng `vlm_cache` /
`zero_shot`. Chế độ provider chỉ định rõ (`gemini`, `cloudflare`, `anthropic`) vẫn
ném lỗi để phục vụ nghiệm thu thuần provider.

## Chi phí và độ trễ ngày thi

### Một key hay nhiều provider?

Router hiện dựa trên đối chứng 30 câu (số nhỏ, cần kiểm lại khi mở rộng nhãn):

| loại câu | số câu | QA cheap đúng | VLM đúng | chiến thuật |
|---|---:|---:|---:|---|
| đếm | 6 | 5 | 5 | local, không gọi API |
| màu | 16 | 10 | 13 | Gemini top-10 |
| mở/vật thể | 8 | 1 | 5 | Gemini top-10 |

- **Đường thi hiện tại (chốt 16/08):** Gemini cho câu màu/câu mở; câu đếm dùng
  QA cheap. Cloudflare BẬT làm dự phòng tuần tự (0.1200 — kém Gemini nhưng hơn
  cheap 0.1067); Claude tắt; Grok đã bỏ. Cuối chuỗi luôn là QA cheap. Provider
  phụ là failover TUẦN TỰ, không phải ensemble.
- **Không gửi song song cùng ảnh cho nhiều provider.** Nó nhân chi phí/quota và
  chưa có dữ liệu ghép cặp cho thấy đồng thuận/đa số làm tăng điểm; trong khi
  khoảng cách tầng trả lời tới trần chỉ 0.0333.
- **Song song request trong cùng provider là chuyện khác.** Chỉ cân nhắc 2–4
  request đồng thời sau khi đo rate-limit/latency trên chính tài khoản thi. Code
  hiện gọi provider tuần tự để ưu tiên ổn định và cache nhất quán.
- **Nhiều API key không phải nhiều quota một cách mặc định.** Không chia key để
  lách rate limit; quota/credit phải được cấp hợp lệ cho project/team tương ứng.

Runtime hiện dùng **hybrid top-10**: VLM trả lời 10 candidate đầu, 90 dòng đuôi
dùng ClipZeroShotQA. Với số Haiku đã đo, cache lạnh xấp xỉ **14 giây và 0,015
USD/câu QA** nếu gọi tuần tự; chạy lại cùng prompt/model/event/câu hỏi là cache
hit. Không chạy ba provider song song—provider sau chỉ được gọi khi provider
trước lỗi.

Trần định vị đo không gọi mạng trên 30 câu: **N=1 0.0667 · N=3 0.1200 · N=5
0.1467 · N=10 0.1667 · N=20 0.1667 · N=50 0.1800 · N=100 0.2133**. Vì N=20
không thêm chút trần nào so với N=10, mặc định 10 là điểm vận hành hợp lý. N=50
tốn gấp năm chỉ mua tối đa +0.0133 trần trên dev set này. Đây vẫn là ước lượng
từ 30 câu; phải kiểm lại khi mở rộng nhãn.

## 2.7 so400m — hoàn tất

- Đã encode đủ **873 video / 177.321 frame**, vector ảnh 1152 chiều; thư mục
  `D:\COMP\AIC2026\data cuộc thi\aic2026_siglip2_so400m` có 880 file, 0,73 GiB.
- Cache 210 câu truy vấn nằm ở
  `dev_set/query_vectors_siglip2_so400m.npz` (907.823 byte).
- KIS core: **0.3400** so với base **0.2950**; delta ghép cặp
  **+0.0450 ± 0.0223**. Median hạng GT **119 → 34**; Recall@100
  **47,5% → 62,5%**.
- `config.example.yaml` đã trỏ mặc định tới model/file so400m. Portfolio vẫn
  tắt vì mức tăng **+0.0122 ± 0.0083** chưa vượt 2·SE.
- **16/08 (việc 2.3):** quy ước tên ảnh của corpus BTC là
  `<root>/<VIDEO>/001.jpg` — **3 chữ số**, khác `event_id` 5 chữ số.
  `EventImageResolver` phủ sẵn nên không phải sửa code. Đã đối chiếu **byte** 4
  ảnh `dev_set/anh` với ảnh corpus cùng `(video, n)`: giống hệt ⇒ `n` không lệch
  đơn vị nào. Dùng `python kiem_resolver_anh.py` (đọc `qa.image_roots`) để chạy
  lại phép kiểm này trên Kaggle **trước khi** gọi VLM — 10 giây, 0 request.
- **16/08:** config không còn là ghi chú — `aic2026/config.py` đọc nó thật và
  `nop_bai.py` nộp bài qua đó (việc 1.5). Đường dẫn riêng của máy nằm ở
  `config.yaml` (`extends: config.example.yaml`). Trên Kaggle, việc 2.1/2.2 chỉ
  cần thêm `qa.image_roots` vào `config.yaml` là tầng QA có ảnh, không phải sửa
  code. Khoá API vẫn CHỈ đọc từ biến môi trường; `nop_bai.py` chặn trước và nói
  rõ thiếu khoá nào.

<!-- B4_KAGGLE_2026_START -->

## Nghiệm thu Kaggle B0–B4 — 17/08/2026

### Môi trường đã nghiệm thu

- Dataset: 873 video / 177.321 keyframe.
- Encoder chính: `google/siglip2-so400m-patch14-384`, 1152 chiều.
- Encoder phụ: SigLIP2 base-224, hợp nhất RRF với trọng số encoder chính = 2.
- Portfolio: tắt.
- Chính sách ngôn ngữ truy vấn: ưu tiên `en`.
- Số dòng nộp tối đa: 100 dòng/truy vấn.
- Bộ kiểm thử: 291 test xanh.

### B0 — Đường tiếng Anh chính

| Chỉ số | Giá trị |
|---|---:|
| Điểm gộp core | 0.3122 |
| KIS | 0.3750 |
| QA cheap | 0.0867 |
| TRAKE | 0.2733 |
| Recall@1 | 11,8% |
| Recall@100 | 44,7% |

Kết luận: mốc KIS 0.3750 được tái lập đúng trên Kaggle. Cấu hình dữ liệu,
so400m và encoder phụ đều hoạt động đúng.

### B1 — Đường ảnh QA

- Corpus keyframe BTC đã được mount qua `qa.image_roots`.
- Resolver ảnh đã được diễn tập trên Kaggle.
- Gemini là provider chính.
- Cloudflare Workers AI là provider dự phòng.
- Kaggle Secrets cần nạp vào `os.environ` sau mỗi lần restart session.
- Kaggle cần `google-genai >= 2.0.0` để dùng Interactions API.

### B2 — Đề song ngữ và TRAKE

- Hệ thống ưu tiên khóa `text_en` khi đề có cả hai bản.
- Câu chỉ có tiếng Việt được cảnh báo rõ ràng.
- Chuỗi TRAKE dùng dấu `→` được tách đúng thành các sự kiện.
- File diễn tập tạo đủ 100 dòng mỗi truy vấn.
- ZIP nghiệm thu: `ok=True`, không có lỗi.
- Đáp án QA xuất ra bằng tiếng Việt.

### B3 — Đường tiếng Việt trực tiếp

| Chỉ số | Giá trị |
|---|---:|
| Điểm gộp core | 0.2463 |
| KIS | 0.2883 |
| QA cheap | 0.0867 |
| TRAKE | 0.2333 |
| Recall@1 | 8,2% |
| Recall@100 | 35,3% |

So với tiếng Anh:

- KIS: 0.3750 → 0.2883, giảm 0.0867, khoảng 23%.
- TRAKE: 0.2733 → 0.2333, giảm 0.0400, khoảng 15%.
- Recall@100: 44,7% → 35,3%.

Quyết định: giữ tiếng Anh làm đường truy hồi chính. Tiếng Việt trực tiếp là
đường dự phòng khi truy vấn không có bản tiếng Anh.

Không bật portfolio dựa trên riêng phép đo tiếng Việt; cần validation split độc
lập trước khi đổi mặc định.

### Quy trình batch 2

BTC thông báo dữ liệu vòng sơ tuyển đầy đủ sẽ có thêm batch 2.

Khi batch 2 xuất hiện:

1. Gắn video/keyframe/map-keyframes batch 2 vào Kaggle.
2. Encode ảnh batch 2 bằng SigLIP2 so400m cho encoder chính.
3. Encode lại batch 2 bằng SigLIP2 base-224 cho encoder phụ.
4. Không trộn vector của hai checkpoint vào cùng thư mục.
5. Cập nhật đường dẫn Dataset trong `config_kaggle.yaml`.
6. Kiểm tra số video và số vector ảnh khớp nhau.
7. Chạy lại 291 test.
8. Chạy lại B0 và yêu cầu KIS không giảm bất thường.
9. Chạy lại `kiem_resolver_anh.py` với corpus ảnh đầy đủ.
10. Diễn tập `nop_bai.py` và yêu cầu ZIP `ok=True`.

Dành tối thiểu 5 giờ GPU trong quota 30 giờ/tuần để encode cả so400m và base
cho batch 2.

### Quyết định cuối

- Đường thi chính: tiếng Anh + so400m×2 + base-224.
- Đường dự phòng: tiếng Việt trực tiếp.
- QA chính: Gemini VLM.
- QA dự phòng: Cloudflare VLM.
- Portfolio: tắt.
- Không thay model hoặc tinh chỉnh thêm trước khi có batch 2/validation mới.

<!-- B4_KAGGLE_2026_END -->
