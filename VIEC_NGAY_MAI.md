# Việc tới SƠ TUYỂN ĐỢT 2 — **19:30–22:30 ngày 28/08/2026**

> Viết lại **26/08**. Bản trước đứng ở 18/08 và đã lệch thực tế: nó chưa có
> vòng thử nghiệm (20/08), chưa có sơ tuyển đợt 1 (21/08), chưa có kênh OCR/ASR,
> chưa có lớp ghim. Đọc bản cũ mà chia việc là chia sai.

**Còn 2 ngày.** Lịch BTC: đợt 2 **28/08 19:30–22:30**, đợt 3 **04/09 19:30–22:30**.
Mỗi gói 3 lần nộp, **tính lần cuối**.

Trạng thái: **450 test xanh** · 177.321 event / 873 video · encoder SigLIP2
so400m + hợp nhất base · kênh OCR (177.321 dòng) + kênh ASR (18.660 khoảng) ·
LLM viết lại truy vấn Việt→chú thích Anh (đang bật, cache đĩa) · lớp ghim
(`videos` / `nhom` / `khung` / `answer`).

---

## ✅ VIỆC SỐ 1 — BỘ KEYFRAME VỀ ĐĨA *(XONG 28/08 chiều)*

Bộ **đầy đủ** đã về `D:\COMP\AIC2026\data cuộc thi\keyframes\archive (10)\`
(có cả `Videos_*` 77 GiB, chưa dùng). Cấu trúc là
`Keyframes_L2x[_a..e]\keyframes\<VIDEO>\001.jpg` nên `qa.image_roots` phải liệt
kê **14 gói**, không trỏ thư mục mẹ — `EventImageResolver` chỉ dò `<root>/<VIDEO>/`,
`<root>/<3 chữ đầu>/<VIDEO>/`, `<root>/`, cố tình không quét sâu 28 GiB.

Nghiệm thu `kiem_resolver_anh.py --config config_xem.yaml` — **xanh hết**:

```
1) resolve 20 event_id ngẫu nhiên : 20/20 ra file thật
2) đếm ảnh == đếm keyframe        : 873/873 video khớp
3) so BYTE ảnh dev vs corpus      : 120 giống · 0 khác · 0 bỏ qua
   -> `n` trong event_id = số thứ tự file ảnh, hai quy ước TRÙNG NHAU
```

🔴 **Từ giờ CẤU HÌNH THI LÀ `config_xem.yaml`**, không phải `config_ocr_asr.yaml`.
Nó = cấu hình cũ + 14 đường ảnh + hai bản sửa QA dưới đây. Mọi lệnh trong file
này đã đổi theo.

### Hai lỗi QA 0-điểm bắt được nhờ chạy lại đề đợt 1 với ảnh thật

**(a) Câu ĐẾM SỐ nhận đáp án MÀU.** Đợt 1 có **3/4 câu QA là câu đếm số**, và cả
ba nộp `1` / `trắng` / `vàng`. Hai nguyên nhân cộng dồn:

* `qa.vlm_question_types: [color, open]` mà `Pipeline._qa_question_type` trả
  `"count"` cho "bao nhiêu|mấy|số lượng" → câu đếm **không bao giờ tới VLM**
  (`pipeline.py:334`), rơi về `ClipZeroShotQA`.
* trong `qa.LUAT`, luật **MÀU đứng trước luật SỐ** và quét trên CẢ đoạn "tả cảnh
  + câu hỏi". Đề BTC luôn tả cảnh trước, cảnh thì đầy màu → câu hỏi số nhận
  nhóm đáp án màu.

Đã sửa: bật `[color, count, open]` (CHỈ trong `config_xem.yaml` — nhánh VLM ném
`RuntimeError` NGOÀI khối try khi thiếu `image_roots`) + đảo `LUAT`. Đo lại:
`1`/`trắng`/`vàng` → `300` / `3,10` / `10,5,4,2`.

**(b) `gemini-3.6-flash` BỊA ĐÁP ÁN, và nó đang đứng ĐẦU chuỗi.** Đo trên đề
đợt 1, đúng chữ ký đã bắt 20/08:

| model | câu | đáp án |
|---|---|---|
| 3.6-flash | "con số cuối trên cân" | `300` cho **8 video khác nhau** |
| 3.6-flash | "sạt lở đường đèo" | `Đèo Bảo Lộc` cho **6 video khác nhau** |
| 3.5-flash-lite | "bao nhiêu vị trí động đất cấp 4" | L21_V010: 5 · L21_V006: 4 · L21_V007: 2 · L25_V076: 10 |

Dòng cuối = **đọc thật** (đổi theo video, nhất quán trong video). Đã đảo chuỗi
trong `config_xem.yaml` thành `3.5-flash-lite → 3.1-flash-lite → 3.7-flash →
… → cloudflare`, loại hẳn `3.6-flash`. Lợi kép: bậc lite là RPM 15 / **RPD 500**
còn bậc flash chỉ **RPD 20**, mà ba tiếng thi thì đứng đầu bằng RPD 20 là tự
đặt vách quota vào giữa buổi. **Thời gian chạy 25 câu: 365,7s → 52,3s.**

Kiểm chuỗi trước khi thi, 0 request: `python -X utf8 _kiem_chuoi_vlm.py config_xem.yaml`

**(c) Đã cài phép kiểm "VLM bịa"** mà ghi chú 20/08 xếp là việc-cần-làm rồi bỏ
quên: `qa.soi_dap_an_vlm` hét khi **chỉ một đáp án phân biệt trải trên ≥3 video**
(đòi cả hai vế: một vế thôi thì báo động giả với câu không gian đáp án nhỏ, hoặc
với câu top-10 rơi cả vào một video). `nop_bai.py` in cảnh báo ngay dưới dòng
"đáp án phân biệt". **454 test xanh.**

Nó bắt được ngay một ca thật: p1-17 đổi từ `Đèo Bảo Lộc` (3.6) sang
`Đèo Khau Phạ` (lite) — **vẫn một đáp án cho cả 10 khung**, tức ảnh không chứa
tên đèo và model đang đoán bằng tri thức ngoài. Câu như vậy phải để người tra
OCR/ASR, đừng tin VLM.

## ✅ VIỆC SỐ 1 CŨ — giữ để tra lại lý do làm *(đã hoàn thành)*

Đây là **hạng mục đắt nhất còn lại**, và nó không phải ý tưởng mới — báo cáo
20/08 §8 đã xếp nó số 1 rồi bị kẹt vì máy không có ảnh:

```
R@100  = 57,5%      khung ĐÚNG đã nằm trong 100 dòng máy trả về
Final  = 0,3450     chỉ ăn chừng này, vì nó không đứng hạng 1
                    chênh 0,23 điểm/CÂU — lớn hơn (SigLIP2 +0,108) +
                    (hợp nhất +0,035) + (LLM viết lại +0,057) CỘNG LẠI
```

Điều đã đổi so với lúc viết báo cáo: **ổ D còn trống 234 GB**, thừa cho 28,66 GiB.

- [ ] Tải dataset keyframe của BTC trên Kaggle (thư mục con tên `keyframes`,
      chính là cái notebook đợt 1 dò bằng `tim_thu_muc('keyframes')`)
- [ ] Giải nén sao cho ra đúng dạng **`<gốc>/<VIDEO>/001.jpg`** — 3 chữ số, không
      phải 5 (quy ước đã đối chiếu BYTE ngày 16/08, việc 2.3 cũ). Đích đề nghị:
      `D:\COMP\AIC2026\data cuộc thi\keyframes`
- [ ] Sửa `config_xem.yaml`: `qa.image_roots` trỏ vào thư mục đó (giờ đang trỏ
      bản mẫu 29 video)
- [ ] Nghiệm thu, **10 giây, 0 API**:
      `python -X utf8 kiem_resolver_anh.py --config config_xem.yaml`
      Phải thấy: resolve 20/20 · đếm ảnh == đếm keyframe ở đủ 873 video · và
      phép mạnh nhất là **so byte ảnh dev set với ảnh corpus cùng `(video, n)`**
      — bắt được cả lệch một đơn vị trong cách đánh số

Tải xong thì mở thêm được thứ hai, không mất công gì: **QA VLM chạy tại máy**.
Đợt 1 phải `--qa-cheap` vì không có ảnh (dev set: VLM 0.2067 vs cheap 0.1067).

---

## ✅ VIỆC SỐ 2 — MÁY XEM ẢNH *(xong 26/08)*

`may_xem.py` (mới) — server tại chỗ, 2–3 người cùng mở bằng trình duyệt, **mỗi
cú bấm ghi thẳng vào file ghim**, không dán JSON.

    python -X utf8 may_xem.py --config config_xem.yaml \
        --de data/de_dot2.json --ghim data/ghim_dot2.json --sau 300 --host 0.0.0.0

- [x] Hai chế độ, ứng với hai kiểu trượt đã chẩn đoán (`chan_doan_qa_truot.py`):
      `/q/<qid>` lưới ứng viên của máy (nhóm A: 56% câu KIS, máy đã chạm khung
      đúng nhưng xếp sai hạng) · `/v/<VIDEO>` **toàn bộ keyframe một video theo
      thời gian** (nhóm B: 22% câu, hạng trung vị 358 — máy không bao giờ tự đưa
      lên top, người biết video thì lướt 200 ảnh là thấy)
- [x] Truy vấn đi qua **đúng bước LLM viết lại** như `nop_bai.py`, nếu không thì
      người bấm trên một danh sách khác với danh sách được nộp
- [x] Ghi file ghim **nguyên tử** (`os.replace`) và **giữ nguyên** các khoá người
      vận hành viết tay (`videos`/`nhom`/`answer`/`so_dong`/`ghi_chu`)
- [x] **10 test** (`tests/test_may_xem.py`), không test nào gọi mạng/nạp model →
      **450 test xanh** (440 → +10)
- [x] **Số đo thật 26/08** (máy này, 25 câu đề đợt 1, pool sâu 300):

      | việc | thời gian |
      |---|---:|
      | nạp dữ liệu (cache OS nóng / nguội) | 2,0s / 20,7s |
      | **nung sẵn cả 25 câu** | **34s** (câu đầu 15,4s, các câu sau 0,5s) |
      | mở một trang câu (300 ô) | 0,08s |
      | mở một trang video (307 keyframe) | 0,10s |
      | một ảnh thu nhỏ (đã nung) | 0,002s |

      ⇒ **mở máy lúc 19:30, tới 19:31 là cả bộ đề sẵn sàng để nhìn.**
- [ ] Chưa kiểm được ở quy mô thật: máy mới có ảnh **29/873 video**. Chạy lại
      sau việc số 1.

> 🔎 Phát hiện lúc đo: **TRAKE chỉ ra 100 ứng viên dù xin 300**
> (`TrakeConfig.paths_per_video` × số video ứng viên). Không phải lỗi của trang
> xem, nhưng nghĩa là *xem sâu hơn* không áp dụng cho TRAKE — muốn sâu hơn thì
> phải nâng `paths_per_video`, chưa đo.

---

## 🟠 VIỆC SỐ 3 — DIỄN TẬP BẤM GIỜ *(cần việc số 1 xong trước)*

Chạy lại **nguyên** quy trình 3 giờ trên bộ đề đợt 1, coi như chưa biết đáp án,
để tìm chỗ nghẽn TRƯỚC khi nó xảy ra lúc thi thật.

- [x] Từ lúc có `.txt` tới lúc ra zip: `doc_de_thi.py` → `nop_bai.py` →
      `may_xem.py` → bấm → `nop_bai.py --ghim` → `kiem_dinh_dang.py`
- [x] **(a) máy chạy xong bài nền = 52,3s** cho cả 25 câu, `config_xem.yaml`,
      ảnh thật, QA VLM thật, zip nghiệm thu ✅. (Chuỗi VLM cũ: 365,7s — xem
      việc số 1.) Lượt toàn cache hit: **39,6s**.
      `may_xem.py` nung sẵn 25 câu/300 ứng viên = **37s** ở quy mô 873 video
      (câu đầu 17,5s vì nạp encoder, các câu sau 0,5s; câu QA 3,2–3,8s).
- [ ] **(b) một người xem hết 300 ảnh một câu = ? phút** ← SỐ DUY NHẤT CÒN
      THIẾU, và là số quyết định cách chia câu. Máy xem đang chạy sẵn:
      `http://127.0.0.1:8000`, ghim ghi ra `data/ghim_dientap.json`.
- [ ] (c) từ lúc bấm xong tới lúc có zip mới
- [ ] Với (b): báo cáo 20/08 ước 7 phút/câu. **Chưa ai đo.** 24 câu × 3 người
      thì mỗi người 8 câu — nếu thật sự 7 phút/câu thì hết 56 phút, còn dư giờ;
      nếu hoá ra 20 phút/câu thì phải cắt `--sau` xuống và chọn câu để xem
- [ ] Chốt cách chia câu cho 2–3 người (trang chủ đã có ô "nhận câu này")

---

## 🟡 VIỆC SỐ 4 — CÂU CẦN TRI THỨC NGOÀI *(báo cáo 20/08, việc số 3)*

4/24 câu đề thử nghiệm thuộc loại encoder ảnh không thể suy ra:
*"loài động vật biển nguy hiểm trong phim Spielberg 1975"* → phải thành
**cá mập trắng** TRƯỚC khi encode. Hạ tầng đã đủ (`viet_lai_truy_van` + chuỗi
failover Gemini→Groq→Cloudflare), chỉ là **sửa prompt + đo lại**.

- [ ] Sửa `PHIEN_BAN_PROMPT` để nó giải mã ẩn ý rồi mới tả cảnh
- [ ] Đo bằng `thu_nghiem_dich_vi.py --llm --to-hop` (cache theo lô, không tốn
      lại request). Nền hiện tại: viết lại được **+0.0567 ± 0.0209** so tiếng
      Việt thô — phải không được làm hỏng số đó
- [ ] ⚠ Đổi prompt là **đổi khoá cache** → phải encode lại toàn bộ; làm trước
      28/08, đừng làm trong giờ thi

---

## ⚪ VIỆC SỐ 5 — CHỐT SỔ TRƯỚC 19:00 NGÀY 28/08

- [x] `python -X utf8 -m pytest -q` → **454 xanh** (28/08, 16:30)
- [ ] `python -X utf8 _kiem_chuoi_vlm.py config_xem.yaml` → phải ra
      `3.5-flash-lite → 3.1-flash-lite → 3.7-flash → … → cloudflare`,
      `thiếu khoá API: không`, `loại câu: [color, count, open]`, `14` đường ảnh.
      **0 request.** Chạy cái này TRƯỚC `kiem_khoa_api.py`.
- [ ] `python kiem_khoa_api.py` → Gemini còn quota (1 request/provider).
      Bậc **lite RPD 500** nên đầu chuỗi không còn là vách nữa; bậc flash vẫn
      RPD 20, đừng đốt vào việc thử nghiệm chiều nay.
- [ ] Nạp trước dữ liệu cho cache OS nóng (20,7s → 2,0s)
- [ ] `may_xem.py` bật sẵn, đồng đội mở được `http://<IP>:8000` trong LAN
- [ ] Mở sẵn cửa sổ riêng: `tra_metadata.py`, `ghim_tu_ocr.py`, `tim_khung.py`
- [ ] `dong_goi.py` để sẵn — **KHÔNG nén zip bằng chuột phải Windows**
      (21/08: Windows bọc thêm một tầng thư mục → `submission/` không nằm ở gốc
      → 0 điểm toàn bộ, mà nhìn bằng mắt zip vẫn "trông đúng")

---

## Thứ tự đêm 28/08 (dự kiến, chốt lại sau khi diễn tập)

| giờ | việc |
|---|---|
| 19:00 | mở máy, nạp trước, `_kiem_chuoi_vlm.py`, kiểm khoá API, `pytest` |
| 19:30 | nhận đề → `doc_de_thi.py` → **ĐẾM SỰ KIỆN TRAKE BẰNG MẮT** (dòng không nhãn = sự kiện 1; `validate_csv` không bắt được vì nó so CSV với chính bản đọc sai của mình) → `nop_bai.py --config config_xem.yaml` (**~1 phút**) → **nộp lần 1** bằng `dong_goi.py` |
| 19:35 | `may_xem.py --config config_xem.yaml` lên, chia câu, mọi người bắt đầu nhìn. Soi log câu QA: dòng `⚠ ĐÁP ÁN ĐÁNG NGỜ` = câu đó VLM không đọc ra, **ưu tiên đưa cho người xem trước** |
| 20:45 | ghép ghim → `nop_bai.py --ghim` → **nộp lần 2** |
| 21:00–22:00 | dồn giờ vào câu còn trống; tra OCR/ASR/tiêu đề cho câu không nhìn ra |
| 22:15 | **nộp lần 3 (bản cuối, được tính điểm)** — chốt sổ, không sửa nữa |
| 22:30 | đóng cổng |

> Ba lần nộp chỉ tính lần cuối, nên lần 1 và lần 2 là **miễn phí về điểm** —
> giá trị của chúng là biết hệ thống parse được bài và thấy điểm public.

---

## ĐÃ XONG — đừng làm lại

- ✅ Chống sập QA thiếu ảnh (1.1–1.3, 15/08) · config đọc thật (1.5, 16/08)
- ✅ Hợp nhất encoder RRF **+0.0350 ± 0.0127** (16/08)
- ✅ LLM viết lại truy vấn Việt→Anh **+0.0567 ± 0.0209** (18–19/08) — dịch máy
  thì ÂM, đừng quay lại
- ✅ Kênh OCR per-keyframe (w=2.0) + kênh ASR per-khoảng (w=1.0), trọng số đã dò
  (21/08) — chọn vùng phẳng ×1.00, không lấy đỉnh nhọn ×1.25
- ✅ Lớp ghim: `videos` · `nhom` (thu hẹp theo chương trình) · `khung` (người
  chọn) · `answer`/nhiều đáp án rải luân phiên
- ✅ `dong_goi.py` bung/đóng zip an toàn · `ghep_hai_he.py` giữ đầu mình, lấy
  đuôi hệ khác
- ✅ **Máy xem ảnh** (26/08) — xem mục việc số 2

## KHÔNG làm

- ❌ Đổi/thêm encoder ảnh nữa — mỗi bước sau càng ít lãi, tất cả cộng lại vẫn
  nhỏ hơn việc số 1
- ❌ Dịch máy truy vấn (đo ra âm) · rerank lân cận thời gian (0.3750 → 0.3433)
- ❌ Khử trùng vector · gộp bằng chứng mức video · "nở ứng viên" không đổi cách
  tiêu slot (no-op về toán học)
- ❌ Nén zip bằng Windows Explorer

---

## Hai điều cần nhớ khi đọc mọi số

1. Dev set: 120 câu KIS / 30 câu QA. SE ghép cặp ±0.02–0.06; cải tiến dưới
   2·SE **không phân giải được**. Mọi số "đo trên dev" là để CHỌN GIỮA CÁC
   PHƯƠNG ÁN, không phải để hứa điểm thi.
2. Thang BTC là `Final = (1/5)·Σ_{k∈{1,5,20,50,100}} R@k`, `R@k` lấy max trên
   tiền tố. Vì thế **thứ tự dòng đáng giá ngang chất lượng mô hình**: cùng một
   khung, hạng 100 được 0,2 điểm, hạng 1 được 1,0. Đó là toàn bộ lý do của việc
   số 1 và số 2.
