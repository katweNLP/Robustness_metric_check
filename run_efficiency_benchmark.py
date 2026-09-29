#!/usr/bin/env python3
"""
Efficiency Benchmark: Time each metric on SummEval samples.

Reports seconds-per-sample for each metric with warmup and multiple runs.
Categorises metrics as lightweight (< 100ms), moderate (< 1s), heavyweight (>= 1s).
"""

import json
import sys
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def load_summeval_samples(n_samples: int = 50):
    """Load first N SummEval triples for benchmarking."""
    from datasets import load_dataset
    ds = load_dataset('mteb/summeval', split='test')

    triples = []
    for item in ds:
        source = item['text']
        reference = item['human_summaries'][0] if item['human_summaries'] else ""
        for summary in item['machine_summaries']:
            triples.append({
                "source": source,
                "reference": reference,
                "generated": summary,
            })
            if len(triples) >= n_samples:
                return triples
    return triples


def _time_metric(metric_fn, triples, n_warmup=3, n_runs=10):
    """Time a metric function. Returns per-sample timing stats in ms."""
    # Warmup
    for t in triples[:n_warmup]:
        try:
            metric_fn(t)
        except Exception:
            pass

    # Timed runs
    all_times_ms = []
    for t in triples[:n_runs]:
        start = time.perf_counter()
        try:
            metric_fn(t)
        except Exception:
            pass
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        all_times_ms.append(elapsed_ms)

    arr = np.array(all_times_ms)
    mean_ms = float(np.mean(arr))
    std_ms = float(np.std(arr))
    median_ms = float(np.median(arr))

    if mean_ms < 100:
        category = "lightweight"
    elif mean_ms < 1000:
        category = "moderate"
    else:
        category = "heavyweight"

    return {
        "mean_ms": round(mean_ms, 2),
        "std_ms": round(std_ms, 2),
        "median_ms": round(median_ms, 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
        "category": category,
        "n_runs": n_runs,
    }


def main():
    parser = argparse.ArgumentParser(description="Efficiency benchmark for CHI metrics")
    parser.add_argument('--n-samples', type=int, default=50)
    parser.add_argument('--n-runs', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default='./results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Efficiency Benchmark: Metric Timing")
    print("=" * 70)

    print(f"\n  [Step 1] Loading {args.n_samples} SummEval samples...")
    triples = load_summeval_samples(args.n_samples)
    n = len(triples)
    print(f"  Loaded {n} triples")

    # Preload all models to ensure warmup is fair
    print("\n  [Step 2] Preloading models...")
    from src.ehi import compute_ehi_factors, compute_ehi, _get_nlp
    from src.rhi_star import compute_rhi_star_factors, compute_rhi_star, _get_st_model
    from src.qhi import compute_qhi_factors, compute_qhi
    from src.chi import compute_chi, score_text as chi_score_text
    from rouge_score import rouge_scorer

    _get_nlp()
    _get_st_model()
    rouge_sc = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # Define metric functions
    def fn_ehi(t):
        f = compute_ehi_factors(t["source"], t["reference"], t["generated"])
        return compute_ehi(f)

    def fn_rhi_star(t):
        f = compute_rhi_star_factors(t["source"], t["reference"], t["generated"])
        return compute_rhi_star(f)

    def fn_qhi(t):
        f = compute_qhi_factors(t["source"], t["reference"], t["generated"])
        return compute_qhi(f)

    def fn_chi_full(t):
        return chi_score_text(t["source"], t["reference"], t["generated"])

    def fn_rouge_l(t):
        return rouge_sc.score(t["reference"], t["generated"])['rougeL'].fmeasure

    def fn_bertscore(t):
        from bert_score import score as bert_score_fn
        P, R, F1 = bert_score_fn([t["generated"]], [t["reference"]],
                                  model_type="roberta-large", lang="en", verbose=False)
        return F1[0].item()

    metrics_to_bench = [
        ("EHI", fn_ehi),
        ("RHI*", fn_rhi_star),
        ("QHI", fn_qhi),
        ("CHI (full)", fn_chi_full),
        ("ROUGE-L", fn_rouge_l),
        ("BERTScore", fn_bertscore),
    ]

    # Run benchmarks
    print(f"\n  [Step 3] Benchmarking ({args.n_runs} runs per metric)...")
    results = {}
    for name, fn in metrics_to_bench:
        print(f"\n    Benchmarking {name}...")
        timing = _time_metric(fn, triples, n_warmup=3, n_runs=args.n_runs)
        results[name] = timing
        print(f"      Mean: {timing['mean_ms']:.1f} ms/sample  "
              f"(std: {timing['std_ms']:.1f}, median: {timing['median_ms']:.1f})  "
              f"[{timing['category']}]")

    # Summary table
    print(f"\n  {'='*70}")
    print(f"  EFFICIENCY BENCHMARK RESULTS ({args.n_runs} runs per metric)")
    print(f"  {'='*70}")
    print(f"\n  {'Metric':<14} {'Mean (ms)':>10} {'Std (ms)':>10} {'Median':>10} {'Category':>14}")
    print(f"  {'-'*60}")

    for name in sorted(results.keys(), key=lambda x: results[x]["mean_ms"]):
        r = results[name]
        print(f"  {name:<14} {r['mean_ms']:>10.1f} {r['std_ms']:>10.1f} "
              f"{r['median_ms']:>10.1f} {r['category']:>14}")

    # Save
    output = {
        "metrics": results,
        "n_samples_available": n,
        "n_runs": args.n_runs,
    }

    save_file = output_dir / "efficiency_benchmark.json"
    with open(save_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {save_file}")


if __name__ == "__main__":
    main()
