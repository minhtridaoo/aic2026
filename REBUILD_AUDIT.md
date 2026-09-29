# AIC26 rebuild acceptance map

Authority order used by this rebuild:

1. Official AIC26 preliminary rules.
2. `D:/COMP/AIC2026/tự làm/new_arch.md` (accepted target architecture).
3. Executed, reproducible team artifacts.
4. Experimental notebooks and technical references.

## Current integration truth

There is one executable path: every real script calls
`aic2026.core.pipeline.Pipeline`. The layer packages are used by that Pipeline;
they are not a second online system.

| Capability | Actual path | Current evidence |
|---|---|---|
| Official scorer | `aic2026/evaluation` re-exports verified `core/scoring.py` | official formula tests |
| Manifest | `data_plane/manifest.py` | real Batch 1: 873 videos, 177321 rows, 0 errors |
| Persistent vectors | `retrieval/persistent.py` | checksum/shape/id/fingerprint fail-closed tests |
| Shared retrieval | `core/retrieval.py` + `retrieval/service.py` | called by Pipeline for all tasks |
| Query compiler | `control/query_compiler.py` | called by Pipeline before KIS/QA/TRAKE |
| Vietnamese routing | `core/dich.py` routes English -> BTC CLIP, Vietnamese -> mCLIP | 163-test suite; 60-GT rerun: raw VI 0.0000, Opus 0.0867, mCLIP 0.0933, NLLB 0.0967, human EN 0.1233; expanding to about 120 GT |
| Gate/verifier | `control/gates.py` and `control/verifier.py` | trace-only decision; no expensive path yet |
| TRAKE | `reasoning/trake.py` + `core/temporal_dp.py` | full timeline score then bounded DANTE beam |
| Rank optimizer | `optimize/submission_optimizer.py` | real A/B +0.0067 ± 0.0047 paired SE; below 2·SE, default remains off |
| Submission | `core/submission.py` | called directly by Pipeline; schema tests |

Not implemented yet: AutoShot/adaptive-keyframe production jobs, shared
multimodal evidence store, dense raw-video refinement, real
OCR/ASR/object/motion/caption indexes, semantic evidence verifier, VLM reranker
and API/UI. No placeholder module is kept for these jobs.

## Do not copy as-is

- Two-query/1,500-frame temporal boost is not TRAKE. Replace with per-event pools,
  candidate-video aggregation, full-timeline hydration and DP.
- `frame_idx / fps` is not a canonical timestamp conversion. Preserve PTS/time-base
  when available and keep BTC mapping evidence explicit.
- Model-assisted semantic-discovery labels are not official ground truth.
- Raw scores from CLIP/BM25/OCR are not calibrated confidence and cannot be summed.
- The current `demo-server-test` checkout is flattened (`server/` and `docs/` are
  missing), its tests and contract constants disagree, so only its tested ideas are
  integrated.
- Notebooks with hard-coded paths, credentials, no row identity or silent image
  skips remain references, not production modules.
- Security audit found a hard-coded access token in the OpenCLIP bigG notebook
  under CỦA ĐÀN ANH. It was not copied into this rebuild; revoke the token and
  redact the notebook before sharing or committing it.

## Known corpus facts carried into the design

- Batch 1 audit: 873 videos, 177,321 official keyframes, 512-D official CLIP rows.
- Official keyframe median gap is about 2.16 s; it cannot solve sub-10-frame TRAKE
  alignment without dense raw-video refinement.
- Adjacent duplicate `frame_idx` mappings exist and no single global frame offset is
  proven. The manifest must therefore preserve evidence/status instead of applying a
  hidden correction.
- BTC keyframes and team adaptive keyframes are complementary sources; neither
  deletes the other automatically.

## Rebuild sequence

1. Canonical PTS-aware manifest and validator.
2. Fail-closed persistent BTC CLIP baseline and official scorer.
3. AutoShot plus adaptive keyframe selection.
4. Shared evidence store and retrieval service.
5. Task-aware query compiler.
6. KIS, QA and full-timeline TRAKE reasoning.
7. Confidence gate, evidence verifier and final rank optimizer.
8. Optional API/UI integration after the internal contracts stabilize.
