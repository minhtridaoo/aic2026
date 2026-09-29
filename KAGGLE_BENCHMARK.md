# Benchmark cần chạy trên Kaggle

Máy local chỉ chạy unit/integration test nhỏ. Các phép dưới đây cần dataset thật
nguyenminhtric/data-batch-1-aic và nên chạy trên Kaggle.

## Gate 1 — canonical manifest

Tìm đúng mount của dataset trong /kaggle/input, sau đó:

    python offline_session.py build-btc-manifest \
      /kaggle/input/data-batch-1-aic /kaggle/working/canonical
    python offline_session.py validate-manifest /kaggle/working/canonical

Điều kiện đạt: 873 video, khoảng 177321 frame samples, không có error. Cảnh báo
PTS_UNKNOWN là dự kiến với mapping BTC vì raw PTS chưa được BTC cung cấp; các
frame ID trùng phải hiện thành warning, không bị âm thầm gộp.

## Gate 2 — persistent bundle

Sau khi tạo embeddings.npy và frame_metadata.jsonl cùng thứ tự:

    python offline_session.py validate-bundle /kaggle/working/vector_bundle \
      --encoder-fingerprint <fingerprint-dung-khi-build>

Điều kiện đạt: checksum, số hàng, số chiều, finite/norm, unique event ID và
fingerprint đều đúng. Chỉ cần sửa một byte metadata thì loader phải từ chối.

## Gate 3 — retrieval/official score

Chạy trên dev set có ground truth theo video-level split; báo riêng KIS, QA,
TRAKE với Final mean-of-top-k và first-correct-rank. A/B tối thiểu:

1. BTC CLIP baseline.
2. Baseline + RRF OCR/ASR.
3. Baseline + adaptive keyframes.
4. TRAKE ANN-hit DP cũ so với full-timeline TRAKE mới.
5. Fast path so với Auto escalation, kèm latency p50/p95.

Không dùng nhãn model-assisted để tuyên bố Recall chính thức. Mọi kết luận chỉ
được giữ khi tăng official score trên query/video chưa dùng để tune.
