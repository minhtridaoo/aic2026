# AIC HCMC 2026 — Video Retrieval system (rebuild theo new_arch)

Hệ thống này được chỉnh trực tiếp từ codebase cũ, không làm lại từ số 0.
Thứ tự thẩm quyền: luật vòng sơ tuyển 2026 → new_arch.md đã nghiệm thu → code
đã chạy của nhóm → notebook tham khảo.

Kiến trúc hiện chia thành hai session:

1. **Offline data plane hiện có**: canonical manifest PTS-aware → BTC mapping
   validation → persistent BTC vector bundle. Adaptive keyframes và multimodal
   evidence vẫn là phase kế tiếp, chưa có code production.
2. **Online control plane**: task-aware query compiler → retrieval/fusion →
   confidence gate → fast/escalated refinement → evidence verifier → official
   submission.

TRAKE không còn căn DP trực tiếp trên các ANN hit rời rạc. Nó gom video ứng
viên, hydrate toàn timeline, chấm event × time, rồi mới chạy temporal DP.

### Trạng thái trung thực

- **Đã chạy trên Batch 1 thật:** canonical manifest, 873 video / 177321 rows,
  0 error, 873 PTS_UNKNOWN và 614 DUP_FRAME_ID.
- **Đã nối vào Pipeline thật và có integration test:** compiler, CLIP/mCLIP
  language routing, shared retrieval, visual-only gate, verifier contract, full-timeline
  TRAKE và rank/submission hiện hữu.
- **Pilot 60 GT đã chạy lại:** Việt thô + BTC CLIP = 0.0000; Opus → CLIP
  = 0.0867; mCLIP tiếng Việt = 0.0933; NLLB → CLIP = 0.0967; Anh người →
  CLIP = 0.1233. NLLB hơn mCLIP +0.0033 với KTC xấp xỉ
  [-0.0364, +0.0431], chưa phân biệt được. Runtime mặc định dùng mCLIP cho
  tiếng Việt vì chỉ cần một component; Opus/NLLB là các nhánh A/B.
  Dev set đang được mở từ 60 lên khoảng 120 câu để chốt lại quyết định.
- **Chưa benchmark chất lượng:** ngưỡng gate, DANTE beam 64, máy dịch và
  optimizer. Không được coi các default này là đã tune.
- **Chưa triển khai:** production AutoShot job, dense raw decode, OCR/ASR/object/
  motion/caption indexes, semantic verifier, VLM reranker và API/UI. Gate hiện
  chỉ ghi `fast/escalate` vào trace; chưa có expensive path để gọi.

```
python -m pytest -q            # 163 tests tại thời điểm rebuild 2026-08-10
python offline_session.py --help
python giai_thich.py           # 6 thí nghiệm giải thích khái niệm (tiếng Việt)
python -X utf8 demo.py         # ingestion -> retrieval -> submission.zip
python nop_bai.py              # ĐƯỜNG NỘP BÀI THẬT, đọc config.yaml
python kiem_dinh_dang.py       # chỉ kiểm ĐỊNH DẠNG file (cấu hình rẻ, cố ý khác)
python kiem_resolver_anh.py    # event_id -> ảnh trên tên file THẬT của corpus
python kiem_khoa_api.py        # khoá VLM còn sống? 1 request/provider, cache tạm
python dien_tap_qa_anh.py      # diễn tập đường ảnh runtime (chạy được cả trên Kaggle)
python chan_doan_qa_truot.py   # câu trượt vì sai video hay sai khung?
python thu_nghiem_hop_nhat_encoder.py   # RRF nhiều encoder, in Δ ± SE
python do_diem_thuc_te.py --config      # đo ĐÚNG cấu hình thi (so400m + hợp nhất)
python run_benchmark.py        # official Mean-of-Top-k R-Scores
python run_benchmark.py --ab   # core/ vs core+optimize/, in chênh lệch
python kiem_chung_thuc_te.py   # 5 phép kiểm trên 177.321 keyframe THẬT
```

> 📘 **Chưa quen hệ thống? Đọc [`HUONG_DAN.md`](HUONG_DAN.md) trước.**
> Giải thích từ số 0 bằng tiếng Việt: luật chấm điểm, hệ thống làm gì, chạy thử
> từng bước, từ điển thuật ngữ, rồi mới đến phần nâng cao. README này là tài
> liệu kỹ thuật, đọc sau.

## Why this exists / what changed after review

The design was reorganised around one fact taken straight from the official 2026
preliminary document: the ranking metric is **fully specified there** (§2.1–2.2 —
per-task R-Score, `R@k = max_{i≤k}`, `Final` = mean over `k ∈ {1,5,20,50,100}`,
max 100 answers per query). It is therefore **computable at home on a labelled
dev set**, which makes it the #1 development metric — not the interactive-round
speed KPIs (time-to-correct, etc.), and not a proxy borrowed from 2025.

The delivery mechanics (platform, zip layout, attempt count, dates) are a
separate question and are **not** in that document — see
"What the 2026 rules actually confirm" below before relying on any of them.

The three fixes baked in here:

1. **Official metric is metric #1** — `scoring.py` implements
   `Final = (1/|K|) Σ_k max_{i≤k} R-Score(r_i)`, `K={1,5,20,50,100}`, with the
   correct per-task R-Score including **TRAKE full-chain per-frame partial
   credit** (the old offline eval scored only event #1). `offline_eval.py`
   reports it per task. Verified line-by-line against the official 2026
   document §2.1–2.2, including its worked example (0.74). `K` stays
   parameterised for batch 2, but it is not an open question today.
2. **Real data/encoder plug points** — `HashEmbedder`/in-memory stores are
   explicit stubs behind the `VisualEncoder`/`VectorStore`/`TextStore`
   interfaces; swap in SigLIP2 + FAISS/Milvus + Elasticsearch without touching
   retrieval/DP.
3. **Submission path is correct** — `submission.py`: TRAKE always emits exactly
   N frames (MISS steps are **interpolated**, never blank), rows fill toward 100
   with video-diversified tails, CRLF/UTF-8/csv-minimal quoting, a `validate_csv`
   guard, and a `submission/`-prefixed zip builder. A malformed file still burns
   1 of 3 attempts, so validation runs before the zip is written.

## Pipeline / files

Chỉ có **một runtime path**: `core.pipeline.Pipeline`. Bốn layer mới được gọi
trực tiếp từ Pipeline; không còn `control/online.py` hay orchestrator thứ hai.

```
aic2026/
├── data_plane/  canonical manifest + BTC mapping validator
├── retrieval/   persistent bundle + shared candidate/timeline service
├── reasoning/   full-timeline TRAKE; dùng DP hiện hữu trong core
├── control/     compiler, gate, verifier
├── evaluation/  official scorer/offline-eval entry point
├── core/        Pipeline thật + providers/stores/retrieval/submission tương thích
└── optimize/    rank optimizer được Pipeline gọi trực tiếp
```

Việc chưa di chuyển hết file vật lý khỏi `core/` là có chủ ý: giữ compatibility
cho notebook cũ trong khi chỉ duy trì một implementation và một đường đo.

### `core/` — tầng TÌM

| Module | Role | Plug point |
|---|---|---|
| `ingestion.py` | frames → 10s/2s event windows | shot-boundary detection |
| `providers.py` | `LLMProvider` + `VisualEncoder` (rule-based / hash stubs) | OpenAI/Gemini · SigLIP2/BEiT3 · pyvi/underthesea |
| `stores.py` | in-memory vector + TF-IDF text stores | FAISS/Milvus · Elasticsearch (VN analyzer) |
| `retrieval.py` | 3 channels (visual/OCR/ASR) + **real RRF**/Borda | — |
| `temporal_dp.py` | TRAKE DP: MISS branch, cross-modal, N-frame guarantee | — |
| `scoring.py` | official Final Score + diagnostics | swap QA answer comparator |
| `submission.py` | CSV rows, validator, zip | — |
| `offline_eval.py` | benchmark trên dev set | — |
| `pipeline.py` | wires it all; used by demo + benchmark | — |

### `optimize/` — tầng TIÊU

| Module | Role | Plug point |
|---|---|---|
| `calibration.py` | score→probability (Platt/Isotonic/Rank) + ECE/Brier/risk-coverage | — |
| `submission_optimizer.py` | **metric-aware 100-slot allocation** | — |

Điểm nối giữa hai tầng chỉ có **một chỗ**: cờ `PortfolioConfig.enabled` trong
`core/pipeline.py`. Tắt → dùng heuristic trong `core/submission.py` (sort theo
điểm, rải video ở đuôi). Bật → gọi sang `optimize/`.

Giữ song song hai đường **không phải để lưu niệm**, mà vì đó là cách duy nhất đo
được tầng `optimize` đáng giữ hay đáng xoá — xem `--ab` bên dưới. Ba chỗ trong
`core/` tồn tại chỉ để phục vụ tầng `optimize`, đã ghi chú tại chỗ:
`temporal_dp.align_k_best` (list-Viterbi), `providers.LLMProvider.answer_distribution`,
và nhánh portfolio trong `pipeline._answer_{kis,qa,trake}`.

### Đo xem tầng `optimize` có đáng giữ không

```bash
python run_benchmark.py --ab        # cùng retriever, hai cách tiêu slot
```

Cùng dữ liệu, cùng ứng viên, khác mỗi cách phân bổ 100 slot → chênh lệch đọc
được chính là phần `optimize` kiếm thêm hoặc làm mất. Mặc định `enabled=False`
cho tới khi có số trên dev set THẬT. Trên dữ liệu synthetic hiện tại delta = 0
(thắng QA, thua TRAKE) — chưa kết luận được gì.

### Import

Đường chuẩn là đường đầy đủ, `from aic2026.core.models import Event` /
`from aic2026.optimize.calibration import RankCalibrator`. Đường phẳng cũ
(`from aic2026.models import Event`) vẫn chạy nhờ bảng alias trong
`aic2026/__init__.py`, giữ cho notebook viết trước 07/08 không phải sửa;
`tests/test_layout_compat.py` khoá cả hai.

## Metric-aware submission (`submission_optimizer.py`)

The official score is `Final = (1/|K|) Σ_k max_{i≤k} R(r_i)` — a **max over a
prefix**, not a sum. Two things follow, and a plain score-sort throws both away:

1. Position only buys tiers. `w(i) = #{k ∈ K : k ≥ i}` gives
   `w(1)=5, w(2..5)=4, w(6..20)=3, w(21..50)=2, w(51..100)=1`. Slot 1 is worth
   5× slot 51, and there are only five distinct slot prices.
2. Because it is a max, a row that is correct *under the same circumstances* as
   an earlier row adds **nothing**. Score-sort spends the budget on
   near-duplicates: 100 frames of one shot, one answer ×100, one alignment/video.

Introducing a latent truth state θ and telescoping the running max gives

```
E[Final] = (1/|K|) · Σ over placements  w(pos) · Σ_θ p_θ · Δ(row, θ)
```

a sum of independent (increment × slot-price) terms ⇒ greedy on marginal gain is
the right algorithm, and optimal under this model (rearrangement inequality;
(1−1/e) in the general submodular case). A zero-gain row still costs nothing —
the metric has no penalty term — so we keep filling to 100 ("free recall") and
just mark those rows non-scoring.

Per task, the hypothesis/scenario granularity is where the score is:

- **KIS** — `(video, frame)` vs `(video, true moment)`. Frames closer than
  `tol_frames` are one hypothesis. `frame_offsets` hedges the unresolved
  frame-index convention (audit: `MIXED_CONVENTION`, offset 0 wins ~79%); the
  hedge is self-limiting because a redundant offset scores zero gain.
- **QA** — `(video, frame, answer)`. Answers are mutually exclusive, so each
  distinct answer is its own scenario. Needs
  `LLMProvider.answer_distribution()`, not `answer()`.
- **TRAKE** — `(video, alignment)`, **fractional** payoff. Wrong video ⇒ 0, but
  right video ⇒ every slot still pays, so `temporal_dp.align_k_best`
  (list-Viterbi) supplies many alignments per video instead of one.

`expected_final_of(ordering, ...)` scores an arbitrary ordering, so the optimizer
can be A/B'd against `submission.diversify_tail` on identical inputs.

**Unverified defaults**: `KIS_TOL_FRAMES=25` (1.0s at the corpus-dominant 25 fps)
and `TRAKE_TOL_FRAMES=5` (the BTC "under 10 frame" window) are starting points.
Fit both on the labelled dev set as the median half-width of the accepted
window, on a video-level split, choosing a plateau not a peak. `expected_final`
is only meaningful with a fitted calibrator — `IdentityCalibrator` preserves
ordering but carries no probabilistic meaning and says so.

## Team plug-in map (5 roles)

- **Encoder/vision**: implement `SigLIP2Encoder(VisualEncoder)` in `providers.py`,
  precompute window embeddings offline, register in `build_encoder`.
- **Vector/index**: implement `FaissVectorStore`/`MilvusVectorStore(VectorStore)`.
- **Text/VN**: implement `ElasticTextStore(TextStore)` + VN word segmentation in
  `providers.tokenize`; wire Whisper (ASR) and VietOCR/PaddleOCR into ingestion.
- **LLM**: implement `OpenAIProvider`/`GeminiProvider(LLMProvider)`; add a VLM
  cross-check of QA answers against the frame before submitting.
- **Eval/submission**: own `offline_eval.py` + `submission.py`; build the
  past-year dev sets (Kaggle 2023 / official 2025) and the video-level tuning
  split.

## Known limitations / intentional stubs

- `HashEmbedder` is deterministic but **not semantically meaningful** — every
  number from `run_benchmark.py` on the synthetic data means nothing about real
  performance. It only proves the wiring + scorer are correct.
- `InMemoryTextStore` tokenises on whitespace only (NOT Vietnamese-aware).
- Config loader landed 2026-08-16 (`aic2026/config.py`): `config.example.yaml` is
  read for real by `nop_bai.py`; `config.yaml` is the per-machine overlay
  (`extends:` + data paths). The measurement scripts (`do_diem_thuc_te.py`,
  `do_diem_vlm_qa.py`) still carry their own CLI defaults on purpose — they A/B
  encoders, and their historical numbers were produced with those defaults.
  `do_diem_thuc_te.py --config` scores the real competition config (so400m +
  encoder fusion); without the flag it keeps its historical default (SigLIP2
  **base**) so old numbers stay comparable.
- DP tunables are **unverified defaults**; tune on real past-year queries with a
  video-level split and pick a stable plateau (overfitting guard).
- TRAKE DP allows two distinct events to share a window but forbids reusing the
  **same** event across steps (see bug #2 below).

## Bugs found & fixed while rebuilding (regression-tested)

1. **`_rrf_fallback_score` was a mean, not RRF** (prior codebase) → `retrieval.rrf_fuse`
   now implements true reciprocal-rank fusion; `test_fusion.py` locks the formula.
2. **DP degeneracy**: with no modality hints, the DP would pick one globally
   strong event repeatedly at every step (equal timestamps ⇒ zero gap penalty),
   beating the temporally-correct chain. Fixed by forbidding same-event reuse
   along a path; `test_no_degenerate_single_event_repeat` locks it.
3. **Single-candidate step collapsed to 0 reward** under min-max normalisation,
   making the DP open with a MISS to dodge a gap penalty. Switched to
   max-normalisation (best→1.0, absent→0.0).
4. **MISS emitted `None`** for TRAKE frames → now interpolated so every row has N
   submittable frames; `test_miss_step_is_interpolated_never_none` locks it.

## What the 2026 rules actually confirm (and what they don't)

Source of truth: `Thong-tin-vong-So-tuyen-AIC2026.pdf` (BTC, preliminary round).
Split explicitly, because the two halves were previously mixed together.

**CONFIRMED by the official 2026 document — no longer needs re-checking:**

- The three query types: Textual KIS, Q&A, TRAKE (§1).
- **Max 100 answers per query** (§2).
- The exact R-Score per task (§2.1): KIS `I(v=GTᵥ ∧ id∈[s,e])`; QA the same plus
  `answer=GTₐ`; TRAKE `0` on video mismatch else `(1/N)Σⱼ I(idⱼ∈[sⱼ,eⱼ])`.
- **`Final = (1/5) Σ_{k∈{1,5,20,50,100}} R@k`, `R@k = max_{1≤i≤k} R(rᵢ)`** (§2.2),
  worked example = 0.74. So `scoring.DEFAULT_KS` is confirmed against the 2026
  rule, not inferred from 2025. The audit's `aic2026_scoring_spec` gate is
  over-conservative and can be flipped to PASS with this document as the source.
- TRAKE per-moment windows are **"thường là dưới 10 frame"** (§2.1.3) — ~0.4s at
  the corpus-dominant 25 fps. This is why `TRAKE_TOL_FRAMES=5` and why sparse
  official keyframes cannot carry TRAKE.
- Batch-1 data is **also AIC 2025 batch 1** (§3), so past-year queries whose GT
  lands in these 873 videos are a legitimate dev set. Batch 2 is still to come.
- Official CLIP features are `clip-ViT-B-32` (§3) — English-only and weak; a
  multilingual encoder (SigLIP2) is an upgrade, not a preference.

**NOT stated anywhere in that document — treat as unverified assumptions:**

- Codabench as the platform, the submission file/zip layout and naming.
- "Max 3 submissions per package, last one counts."
- Any calendar date, including the preliminary opening date.
- Whether an agent-vs-agent form exists in the final round.

Everything in the second list is kept behind config (`submission.py` layout,
`MAX_ROWS`, file names) and must not drive design decisions until BTC publishes
the submission guide.

## Lịch sử bản dựng

Ghi lại ở đây cho việc đấu nội bộ; **không dùng để chia code**. Trong code hiện
tại, file nằm ở đâu là do nó *làm gì*, không phải do ai viết hay viết khi nào.

| Mốc | Nội dung |
|---|---|
| 2026-07-16 | Dựng khung retrieval: 11 module `core/` + 5 file test |
| 2026-08-06 | Thêm tầng metric-aware: `calibration`, `submission_optimizer`, `align_k_best`, `answer_distribution`, nhánh portfolio trong `pipeline` + 4 file test |
| 2026-08-07 | Tách `core/` + `optimize/`, thêm `test_layout_compat.py` và `run_benchmark.py --ab` |

Ảnh chụp trước khi tách:
`D:\COMP\AIC2026\tự làm\_backup_system-thu_truoc-khi-tach_2026-08-07.zip`
