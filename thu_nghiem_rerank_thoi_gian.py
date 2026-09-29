#!/usr/bin/env python
"""A/B pool sâu + rerank lân cận thời gian, không gọi API.

Ví dụ::

    python -X utf8 thu_nghiem_rerank_thoi_gian.py --config config.yaml
    python -X utf8 thu_nghiem_rerank_thoi_gian.py --task qa --pool 500

Script lấy pool từ đúng retrieval nhiều encoder đang dùng, sau đó so baseline
top-100 với một lưới reranker.  Siêu tham số được chọn trên nửa A theo VIDEO và
chỉ được kết luận bằng nửa B chưa dùng để chọn.  Không tự sửa config production.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from aic2026 import config as cau_hinh
from aic2026.control.query_compiler import compile_query
from aic2026.core import scoring
from aic2026.core import submission as sub
from aic2026.core.offline_eval import load_queries
from aic2026.core.providers import RuleBasedProvider
from aic2026.retrieval.temporal_rerank import (
    TemporalRerankConfig,
    rerank_temporal,
)

GOC = Path(__file__).resolve().parent
DEV_JSON = GOC / "data" / "dev_set_tu_lam.json"


def _args():
    parser = argparse.ArgumentParser(
        description="Đo pool sâu + temporal rerank trên dev set")
    parser.add_argument("--config", default=None)
    parser.add_argument("--task", choices=("kis", "qa", "all"), default="kis")
    parser.add_argument("--pool", type=int, default=500)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--windows", default="2,5,10")
    parser.add_argument("--alphas", default="0.25,0.5,1,2")
    parser.add_argument("--max-neighbours", default="2,4")
    return parser.parse_args()


def _numbers(value, cast=float):
    return [cast(x.strip()) for x in str(value).split(",") if x.strip()]


def _mean_se(values):
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, float("inf")
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(variance / len(values))


def _half(video_id):
    # Stable across Python processes; built-in hash() is intentionally salted.
    digest = hashlib.sha1(video_id.encode("utf-8")).digest()
    return "A" if digest[0] % 2 == 0 else "B"


def _localization_score(candidates, ground_truth, top_k):
    rows = sub.kis_rows(candidates[:top_k], max_rows=top_k)
    values = [scoring.r_score_kis(row, ground_truth) for row in rows]
    return sum(max(values[:k], default=0.0) for k in scoring.DEFAULT_KS) / len(
        scoring.DEFAULT_KS)


def main():
    args = _args()
    if args.pool < args.top_k:
        raise SystemExit("--pool phải >= --top-k")
    config = cau_hinh.load_config(args.config)
    queries = [(q, gt) for q, gt in load_queries(DEV_JSON)
               if q.task in ("kis", "qa")
               and (args.task == "all" or q.task == args.task)]
    if not queries:
        raise SystemExit("Không có câu phù hợp trong dev set")

    print(f"Nạp pipeline · {len(queries)} câu {args.task} · "
          f"pool {args.pool} -> top {args.top_k}")
    started = time.perf_counter()
    events, embeddings, _, report = cau_hinh.nap_du_lieu(config)
    auxiliaries = cau_hinh.nap_hop_nhat(config, events)
    pipeline = cau_hinh.dung_pipeline(
        config, events, embeddings, llm=RuleBasedProvider(),
        retrieval_phu=auxiliaries)
    print(f"  {report.events:,} event · {len(auxiliaries)} encoder phụ · "
          f"{time.perf_counter() - started:.1f}s")

    pools = []
    for index, (query, ground_truth) in enumerate(queries, start=1):
        compiled = compile_query(query.task, query.text)
        candidates = pipeline.retrieval_service.retrieve_compiled(
            compiled, top_k=args.pool)
        pools.append((query, ground_truth, candidates))
        if index % 20 == 0 or index == len(queries):
            print(f"  retrieval {index}/{len(queries)}")

    baseline = [_localization_score(c, gt, args.top_k)
                for _, gt, c in pools]
    configs = [TemporalRerankConfig(w, a, 60.0, n)
               for w in _numbers(args.windows)
               for a in _numbers(args.alphas)
               for n in _numbers(args.max_neighbours, int)]
    half_a = [i for i, (_, gt, _) in enumerate(pools)
              if _half(gt.video_id) == "A"]
    half_b = [i for i in range(len(pools)) if i not in set(half_a)]
    if not half_a or not half_b:
        raise SystemExit("Dev set không tạo được hai nửa video A/B")

    results = []
    print("\nwindow alpha n  full score       Δ full ± SE       Δ A       Δ B")
    print("-" * 72)
    for cfg in configs:
        scores = [_localization_score(rerank_temporal(c, cfg), gt, args.top_k)
                  for _, gt, c in pools]
        delta = [x - y for x, y in zip(scores, baseline)]
        full, _ = _mean_se(scores)
        d_full, se_full = _mean_se(delta)
        d_a, _ = _mean_se([delta[i] for i in half_a])
        d_b, _ = _mean_se([delta[i] for i in half_b])
        results.append((d_a, cfg, scores, d_full, se_full, d_b))
        print(f"{cfg.window_seconds:6.1f} {cfg.alpha:5.2f} "
              f"{cfg.max_neighbours:1d}  {full:10.4f}  "
              f"{d_full:+8.4f} ± {se_full:.4f}  {d_a:+7.4f}  {d_b:+7.4f}")

    # Select only on A.  B is the untouched confirmation half.
    _, winner, winner_scores, d_full, se_full, d_b = max(
        results, key=lambda row: (row[0], -row[1].alpha,
                                  -row[1].window_seconds))
    d_a = max(results, key=lambda row: (row[0], -row[1].alpha,
                                        -row[1].window_seconds))[0]
    base_mean, _ = _mean_se(baseline)
    winner_mean, _ = _mean_se(winner_scores)
    print("\nKẾT LUẬN HOLDOUT THEO VIDEO")
    print(f"  baseline              : {base_mean:.4f}")
    print(f"  chọn trên nửa A       : window={winner.window_seconds:g}s · "
          f"alpha={winner.alpha:g} · neighbours={winner.max_neighbours}")
    print(f"  rerank full           : {winner_mean:.4f} "
          f"(Δ {d_full:+.4f} ± {se_full:.4f})")
    print(f"  nửa A / nửa B         : Δ {d_a:+.4f} / {d_b:+.4f}")
    if d_b <= 0:
        print("  ❌ Không qua holdout B — KHÔNG bật vào đường thi.")
    elif d_full <= 2 * se_full:
        print("  ⚪ B dương nhưng full chưa vượt 2·SE — cần thêm nhãn, chưa bật.")
    else:
        print("  ✅ B dương và full vượt 2·SE — đủ lý do để làm bước nối config,"
              " rồi nghiệm thu lại đúng đường nộp bài.")


if __name__ == "__main__":
    main()
