#!/usr/bin/env python
"""Đo VLM thật trên 30 ảnh QA, có cache và không dùng LLM-judge.

Mỗi câu dùng đúng ảnh đã gán nhãn để tách chất lượng trả lời khỏi định vị. Đáp
án VLM sau đó được ghép vào 100 dòng retrieval của cùng câu; điểm thu được so
trực tiếp với trần định vị. Đây là đo offline có nhãn, không phải đường runtime
ngày thi (runtime phải resolve ảnh của từng candidate trong ``Pipeline``).
"""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import re
import sys
import time
import argparse
from pathlib import Path

from aic2026.core import scoring
from aic2026.core.offline_eval import load_queries
from aic2026.core.pipeline import PortfolioConfig
from aic2026.core.providers import (AnthropicVLMProvider,
                                    CachedImageLLMProvider,
                                    CloudflareVLMProvider,
                                    FailoverVLMProvider,
                                    GeminiVLMProvider)
from aic2026.data_plane.image_resolver import EventImageResolver

GOC = Path(__file__).resolve().parent


def _nap_ddt():
    spec = importlib.util.spec_from_file_location("ddt", GOC / "do_diem_thuc_te.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tb_se(values):
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, float("inf")
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(variance / len(values))


def localization_ceiling(rows, gt):
    per = [scoring.r_score_kis(row, gt) for row in rows]
    return sum(max(per[:k], default=0.0) for k in scoring.DEFAULT_KS) / len(
        scoring.DEFAULT_KS)


class FakeVLM(CachedImageLLMProvider):
    """Smoke-test provider; never network and intentionally not an accuracy model."""

    def __init__(self, cache_path):
        super().__init__(cache_path, model="fake-no-network")

    def _request_image(self, question, image_path, context=""):
        return "không rõ"


def _args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider",
                        choices=("auto", "gemini", "cloudflare", "anthropic",
                                 "fake"),
                        default="auto")
    parser.add_argument("--model")
    parser.add_argument("--encoder", choices=("base", "so400m"),
                        default="so400m")
    parser.add_argument("--min-interval", type=float, default=4.0,
                        help="số giây tối thiểu giữa hai request Gemini")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=0)
    parser.add_argument("--batch-max-images", type=int, default=1,
                        help=">1 gom nhiều ảnh vào một Gemini Interaction")
    parser.add_argument("--enable-anthropic-fallback", action="store_true",
                        help="cho auto dùng Claude sau Gemini; mặc định TẮT")
    # Grok/xAI đã BỎ HẲN 16/08 — không còn provider, không còn cờ.
    parser.add_argument("--enable-cloudflare-fallback", action="store_true",
                        help="cho auto dùng Cloudflare sau Gemini; mặc định TẮT")
    parser.add_argument("--cloudflare-agree-license", action="store_true",
                        help="chỉ Cloudflare: chấp nhận Meta license, không chạy QA")
    parser.add_argument("--limit", type=int, default=0,
                        help="chỉ chạy N câu QA để smoke-test provider (0 = đủ 30)")
    parser.add_argument("--vlm-top-n", type=int, default=10,
                        help="số dòng đầu nhận đáp án VLM khi mô phỏng chính sách "
                             "đuôi của runtime (khớp qa.vlm_top_n trong config)")
    # tương thích lệnh cũ
    parser.add_argument("--fake", action="store_true")
    return parser.parse_args()


def _provider(args):
    kind = "fake" if args.fake else args.provider
    defaults = {"auto": "gemini-with-opt-in-fallbacks",
                "gemini": GeminiVLMProvider.DEFAULT_MODEL,
                "cloudflare": CloudflareVLMProvider.DEFAULT_MODEL,
                "anthropic": AnthropicVLMProvider.DEFAULT_MODEL,
                "fake": "fake-no-network"}
    model = args.model or defaults[kind]
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model)
    cache = GOC / "out" / f"vlm_qa_cache_{kind}_{safe_model}.json"
    if kind == "fake":
        return kind, model, FakeVLM(cache)
    if kind == "auto":
        # Chỉ gọi một provider thành công. Các provider sau là failover tuần tự,
        # không phải ensemble/song song nên không nhân ba chi phí.
        providers = []
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            providers.append(GeminiVLMProvider(
                GOC / "out" / "vlm_qa_cache_gemini_gemini-3.6-flash.json",
                model=GeminiVLMProvider.DEFAULT_MODEL,
                min_interval_seconds=args.min_interval,
                timeout_seconds=args.timeout, max_retries=args.max_retries,
                batch_max_images=args.batch_max_images))
        if args.enable_anthropic_fallback and os.environ.get("ANTHROPIC_API_KEY"):
            providers.append(AnthropicVLMProvider(
                GOC / "out" / "vlm_qa_cache_anthropic_claude-haiku-4-5.json",
                model="claude-haiku-4-5"))
        if (args.enable_cloudflare_fallback
                and os.environ.get("CLOUDFLARE_API_TOKEN")
                and os.environ.get("CLOUDFLARE_ACCOUNT_ID")):
            providers.append(CloudflareVLMProvider(
                GOC / "out" / "vlm_qa_cache_cloudflare_llama-3.2-11b-vision.json",
                model=CloudflareVLMProvider.DEFAULT_MODEL,
                timeout_seconds=args.timeout))
        if not providers:
            raise SystemExit(
                "Provider auto cần GEMINI_API_KEY/GOOGLE_API_KEY. Cloudflare/"
                "Claude chỉ được thêm khi có key và cờ --enable-*-fallback.")
        return kind, model, FailoverVLMProvider(providers)
    if kind == "gemini":
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise SystemExit(
                "Thiếu GEMINI_API_KEY. Không đặt key trong source/config; "
                "hãy đặt biến môi trường cho tiến trình chạy.")
        return kind, model, GeminiVLMProvider(
            cache, model=model, min_interval_seconds=args.min_interval,
            timeout_seconds=args.timeout, max_retries=args.max_retries,
            batch_max_images=args.batch_max_images)
    if kind == "cloudflare":
        if not (os.environ.get("CLOUDFLARE_API_TOKEN")
                and os.environ.get("CLOUDFLARE_ACCOUNT_ID")):
            raise SystemExit(
                "Thiếu CLOUDFLARE_API_TOKEN hoặc CLOUDFLARE_ACCOUNT_ID. "
                "Không đặt token trong source/config.")
        return kind, model, CloudflareVLMProvider(
            cache, model=model, timeout_seconds=args.timeout)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Thiếu ANTHROPIC_API_KEY cho provider Anthropic.")
    return kind, model, AnthropicVLMProvider(cache, model=model)


def _percentile(values, q):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def main():
    args = _args()
    kind, model, provider = _provider(args)
    if args.cloudflare_agree_license:
        if not isinstance(provider, CloudflareVLMProvider):
            raise SystemExit("--cloudflare-agree-license chỉ dùng cùng --provider cloudflare")
        print("Đang gửi xác nhận Meta license cho Cloudflare Workers AI...")
        print(provider.agree_to_model_license() or "Cloudflare đã xác nhận.")
        return
    resolver = EventImageResolver(dev_image_dir=GOC / "dev_set" / "anh")

    ddt = _nap_ddt()
    ddt.kiem_tra_dieu_kien(can_dev_set=True, siglip2=args.encoder)
    events, embeddings, _, encoder = ddt.nap_he_thong(siglip2=args.encoder)
    pipeline = ddt.dung_pipeline(
        events, embeddings, encoder, portfolio=PortfolioConfig(enabled=False))
    pairs = [(q, gt) for q, gt in load_queries(ddt.DEV_JSON) if q.task == "qa"]
    if args.limit:
        if args.limit < 1:
            raise SystemExit("--limit phải là số nguyên dương, hoặc bỏ cờ để chạy đủ 30")
        pairs = pairs[:args.limit]
    raw = json.loads(Path(ddt.DEV_JSON).read_text(encoding="utf-8"))
    raw_by_qid = {q["qid"]: q for q in raw["queries"] if q["task"] == "qa"}

    records = []
    for query, gt in pairs:
        raw_q = raw_by_qid[query.qid]
        ordinal = int(raw_q["_ghi_chu"]["keyframe_n"])
        event_id = f"{gt.video_id}#{ordinal:05d}"
        image = resolver.resolve(event_id)
        question_vi = query.question_vi or raw_q.get("_cau_hoi_vi") or query.text
        records.append((query, gt, event_id, image, question_vi))

    vlm_started = time.perf_counter()
    requests = [(question_vi, str(image), event_id, "")
                for _, _, event_id, image, question_vi in records]
    cloud_error = None
    try:
        answers = provider.answer_images(requests)
        answer_sources = [getattr(provider, "last_provider", None)
                          or type(provider).__name__] * len(records)
    except RuntimeError as exc:
        if kind != "auto":
            raise
        cloud_error = str(exc)
        peek = getattr(provider, "peek_cached_answer", None)
        answers, answer_sources = [], []
        for _, _, event_id, _, question_vi in records:
            cached = peek(question_vi, event_id) if peek is not None else None
            answers.append(cached)
            answer_sources.append("vlm_cache" if cached is not None else "zero_shot")
    batch_vlm_latency = time.perf_counter() - vlm_started
    amortized_vlm_latency = batch_vlm_latency / len(records)

    actual, ceiling, answer_ok, details = [], [], 0, []
    # Hai chính sách ĐUÔI, chấm ghép cặp trên cùng một bộ dòng (0 request thêm):
    #   `runtime_cheap` — như runtime hôm nay: VLM cho top-N, 90 dòng đuôi giữ đáp
    #                     án zero-shot rẻ (đáp án rẻ đúng ~13/30)
    #   `actual`        — đáp án VLM cho CẢ 100 dòng (đáp án VLM đúng 26/30)
    # Vì R-Score chỉ cộng điểm cho dòng NẰM TRONG cửa sổ đúng, và cửa sổ đúng chỉ
    # thuộc một video, nên dùng cùng một đáp án tốt nhất cho mọi dòng là tối ưu.
    runtime_cheap = []
    top_n_values = (1, 3, 5, 10, 20, 50, 100)
    top_n_ceilings = {n: [] for n in top_n_values}
    latencies, retrieval_latencies = [], []
    for index, ((query, gt, event_id, image, question_vi), answer) in enumerate(
            zip(records, answers)):
        retrieval_started = time.perf_counter()
        rows = pipeline.answer(query, top_k=100)
        retrieval_latency = time.perf_counter() - retrieval_started
        if answer is None:
            answer = str(rows[0].get("answer", "")) if rows else ""
            answer_sources[index] = "zero_shot"
        # Chính sách runtime hôm nay: chỉ top-N nhận đáp án VLM, đuôi giữ đáp án rẻ
        # mà `pipeline.answer` đã gán.
        rows_runtime = [dict(row, answer=(answer if i < args.vlm_top_n
                                         else row.get("answer", "")))
                        for i, row in enumerate(rows)]
        runtime_cheap.append(scoring.final_score(rows_runtime, gt))
        rows = [dict(row, answer=answer) for row in rows]
        score = scoring.final_score(rows, gt)
        cap = localization_ceiling(rows, gt)
        for n in top_n_values:
            top_n_ceilings[n].append(localization_ceiling(rows[:n], gt))
        latency = amortized_vlm_latency + retrieval_latency
        accepted = [gt.answer] + list(gt.answer_alt)
        matched = any(scoring.answers_match(answer, x) for x in accepted)
        actual.append(score)
        ceiling.append(cap)
        answer_ok += matched
        latencies.append(latency)
        retrieval_latencies.append(retrieval_latency)
        details.append({
            "qid": query.qid, "event_id": event_id, "image": str(image),
            "question_vi": question_vi, "ground_truth": gt.answer,
            "answer_alt": ";".join(gt.answer_alt), "prediction": answer,
            "answer_source": answer_sources[index],
            "answer_matches": int(matched), "final_score": f"{score:.6f}",
            "localization_ceiling": f"{cap:.6f}",
            "vlm_latency_seconds": f"{amortized_vlm_latency:.6f}",
            "retrieval_latency_seconds": f"{retrieval_latency:.6f}",
            "latency_seconds": f"{latency:.6f}",
        })

    safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model)
    out = GOC / "out" / f"vlm_qa_predictions_{kind}_{safe_model}_{args.encoder}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)

    mean, se = tb_se(actual)
    cap_mean, cap_se = tb_se(ceiling)
    gap, gap_se = tb_se([c - a for a, c in zip(actual, ceiling)])
    print(f"Provider: {kind} · model: {model} · encoder: {args.encoder}")
    label = "QA VLM" if not cloud_error else "QA HYBRID"
    print(f"{label}: {mean:.4f} ± {se:.4f}")
    print(f"Trần định vị: {cap_mean:.4f} ± {cap_se:.4f}")
    print("Trần theo top-N: " + " · ".join(
        f"N={n}: {tb_se(top_n_ceilings[n])[0]:.4f}"
        for n in top_n_values))
    print(f"Khoảng cách ghép cặp: {gap:+.4f} ± {gap_se:.4f}")
    rt_mean, rt_se = tb_se(runtime_cheap)
    d_duoi, d_duoi_se = tb_se([a - r for a, r in zip(actual, runtime_cheap)])
    print(f"Chính sách ĐUÔI (đo ghép cặp, 0 request thêm):")
    print(f"  VLM top-{args.vlm_top_n} + đuôi RẺ (runtime hôm nay): "
          f"{rt_mean:.4f} ± {rt_se:.4f}")
    print(f"  VLM cho CẢ 100 dòng                     : {mean:.4f} ± {se:.4f}")
    print(f"  delta ± paired SE                       : {d_duoi:+.4f} ± {d_duoi_se:.4f}"
          + ("  -> vượt 2·SE, ĐÁNG ĐỔI" if abs(d_duoi) > 2 * d_duoi_se
             else "  -> chưa vượt 2·SE"))
    print(f"Đáp án khớp chuỗi/alt: {answer_ok}/{len(pairs)}")
    source_counts = {source: answer_sources.count(source)
                     for source in sorted(set(answer_sources))}
    print("Nguồn đáp án: " + " · ".join(
        f"{name}={count}" for name, count in source_counts.items()))
    if cloud_error:
        print("⚠ Cloud VLM đều lỗi; auto đã hạ xuống cache/zero-shot.")
        print(f"  Lỗi cloud: {cloud_error[:500]}")
    print(f"VLM batch: {batch_vlm_latency:.3f}s cho {len(records)} ảnh · "
          f"{amortized_vlm_latency:.3f}s/đáp án quy đổi")
    print(f"Retrieval/câu: TB {sum(retrieval_latencies)/len(retrieval_latencies):.3f}s")
    print(f"End-to-end/câu: TB {sum(latencies)/len(latencies):.3f}s · "
          f"max {max(latencies):.3f}s")
    if provider.network_calls:
        print("Ước tính 100 candidate cache lạnh: tối đa 100 request ảnh")
    print(f"Cache hit: {provider.cache_hits} · đáp án mới: {provider.network_calls} · "
          f"API request thật: {provider.backend_requests}")
    if getattr(provider, "total_cost_usd", 0.0):
        print(f"Chi phí provider trả phí: ${provider.total_cost_usd:.6f}")
    print(f"Chi tiết: {out}")


if __name__ == "__main__":
    main()
