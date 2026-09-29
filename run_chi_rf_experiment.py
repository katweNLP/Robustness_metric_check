#!/usr/bin/env python3
"""
Evaluate CHI-RF (Reference-Free) on SummEval and HaluEval QA.

Compares CHI-RF (no reference), CHI-RF-NLI (NLI enriched), and CHI (full,
with reference) to quantify the cost of dropping the reference.

Output: results/chi_rf_results.json
"""

import json
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

sys.path.insert(0, str(Path(__file__).parent))


def load_summeval(max_articles=100):
    from datasets import load_dataset
    ds = load_dataset('mteb/summeval', split='test')
    triples = []
    for item in ds:
        source = item['text']
        reference = item['human_summaries'][0] if item['human_summaries'] else ""
        for i, (summary, cons) in enumerate(
                zip(item['machine_summaries'], item['consistency'])):
            triples.append({
                "source": source, "reference": reference,
                "generated": summary,
                "human_consistency": cons,
            })
        if len(triples) >= max_articles * 16:
            break
    print(f"  Loaded {len(triples)} SummEval triples", flush=True)
    return triples


def load_halueval_qa(max_samples=500):
    from datasets import load_dataset
    ds = load_dataset('notrichardren/HaluEval', 'qa', split='train',
                      streaming=True)
    samples = []
    for item in ds:
        knowledge = item.get('knowledge', '')
        question = item.get('question', '')
        right = item.get('right_answer', '')
        hallucinated = item.get('hallucinated_answer', '')
        if not knowledge or not right or not hallucinated:
            continue
        input_text = question + " " + knowledge
        samples.append({"source": input_text, "generated": right,
                        "label": 0})
        samples.append({"source": input_text, "generated": hallucinated,
                        "label": 1})
        if len(samples) >= max_samples:
            break
    print(f"  Loaded {len(samples)} HaluEval QA samples", flush=True)
    return samples


def main():
    parser = argparse.ArgumentParser(description="CHI-RF evaluation")
    parser.add_argument('--max-articles', type=int, default=100)
    parser.add_argument('--max-qa', type=int, default=500)
    parser.add_argument('--skip-nli', action='store_true',
                        help="Skip NLI enrichment (faster)")
    parser.add_argument('--output-dir', default='./results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  CHI-RF: Reference-Free Evaluation")
    print("=" * 70)

    # ------------------------------------------------------------------
    # SummEval
    # ------------------------------------------------------------------
    print("\n  [Step 1] Loading SummEval...", flush=True)
    triples = load_summeval(args.max_articles)
    n = len(triples)

    from src.chi_rf import score_text_rf, score_text_rf_nli
    from src.chi import score_text as score_text_full

    print(f"\n  [Step 2] Computing CHI-RF on SummEval ({n} triples)...",
          flush=True)
    t0 = time.time()
    rf_scores = []
    full_scores = []
    for i, t in enumerate(triples):
        try:
            rf = score_text_rf(t["source"], t["generated"])
        except Exception:
            rf = {"chi_rf": 0.5, "ehi_rf": 0.5, "rhi_rf": 0.5, "qhi_rf": 0.5}
        rf_scores.append(rf)

        try:
            full = score_text_full(t["source"], t["reference"], t["generated"])
        except Exception:
            full = {"chi": 0.4, "ehi": 0.4, "rhi_star": 0.333, "qhi": 0.4}
        full_scores.append(full)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"    {i+1}/{n} ({rate:.1f}/s, ETA: {(n-i-1)/rate:.0f}s)",
                  flush=True)

    print(f"    CHI-RF + CHI done in {time.time()-t0:.0f}s", flush=True)

    # NLI enrichment (optional, slow)
    nli_scores = None
    if not args.skip_nli:
        print(f"\n  [Step 3] Computing CHI-RF-NLI on SummEval...", flush=True)
        t0 = time.time()
        nli_scores = []
        for i, t in enumerate(triples):
            try:
                nli = score_text_rf_nli(t["source"], t["generated"])
            except Exception:
                nli = {"chi_rf_nli": 0.5, "ehi_rf_nli": 0.5,
                       "rhi_rf": 0.5, "qhi_rf": 0.5}
            nli_scores.append(nli)

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"    {i+1}/{n} ({rate:.2f}/s, ETA: {(n-i-1)/rate:.0f}s)",
                      flush=True)
        print(f"    CHI-RF-NLI done in {time.time()-t0:.0f}s", flush=True)

    # Correlations
    print(f"\n  [Step 4] Computing SummEval correlations...", flush=True)
    human = np.array([t["human_consistency"] for t in triples])

    summeval_results = {}
    metrics = {
        "CHI": np.array([s["chi"] for s in full_scores]),
        "EHI": np.array([s["ehi"] for s in full_scores]),
        "RHI*": np.array([s["rhi_star"] for s in full_scores]),
        "CHI-RF": np.array([s["chi_rf"] for s in rf_scores]),
        "EHI-RF": np.array([s["ehi_rf"] for s in rf_scores]),
        "RHI-RF": np.array([s["rhi_rf"] for s in rf_scores]),
        "QHI-RF": np.array([s["qhi_rf"] for s in rf_scores]),
    }
    if nli_scores:
        metrics["CHI-RF-NLI"] = np.array(
            [s["chi_rf_nli"] for s in nli_scores])
        metrics["EHI-RF-NLI"] = np.array(
            [s["ehi_rf_nli"] for s in nli_scores])

    print(f"\n  {'Metric':<18} {'Spearman':>10} {'Kendall':>10}")
    print(f"  {'-'*40}")
    for name, scores in sorted(metrics.items(),
                                key=lambda x: -spearmanr(x[1], human)[0]):
        rho, p = spearmanr(scores, human)
        tau, _ = kendalltau(scores, human)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:<18} {rho:>8.4f}{sig:>2} {tau:>10.4f}")
        summeval_results[name] = {
            "spearman": float(rho), "kendall": float(tau), "p_value": float(p)}

    # ------------------------------------------------------------------
    # HaluEval QA
    # ------------------------------------------------------------------
    print(f"\n  [Step 5] Loading HaluEval QA...", flush=True)
    qa_samples = load_halueval_qa(args.max_qa)
    n_qa = len(qa_samples)

    print(f"\n  [Step 6] Computing CHI-RF on HaluEval QA ({n_qa})...",
          flush=True)
    t0 = time.time()
    qa_rf_scores = []
    for i, s in enumerate(qa_samples):
        try:
            rf = score_text_rf(s["source"], s["generated"])
        except Exception:
            rf = {"chi_rf": 0.5, "ehi_rf": 0.5, "rhi_rf": 0.5, "qhi_rf": 0.5}
        qa_rf_scores.append(rf)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"    {i+1}/{n_qa} ({rate:.1f}/s)", flush=True)
    print(f"    Done in {time.time()-t0:.0f}s", flush=True)

    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    from scipy.stats import pointbiserialr

    qa_labels = np.array([s["label"] for s in qa_samples])
    qa_metrics = {
        "CHI-RF": np.array([s["chi_rf"] for s in qa_rf_scores]),
        "EHI-RF": np.array([s["ehi_rf"] for s in qa_rf_scores]),
        "RHI-RF": np.array([s["rhi_rf"] for s in qa_rf_scores]),
        "QHI-RF": np.array([s["qhi_rf"] for s in qa_rf_scores]),
    }

    halueval_results = {}
    print(f"\n  {'Metric':<18} {'Eff AUC':>10} {'Bal Acc':>10} {'|r_pb|':>10}")
    print(f"  {'-'*50}")
    for name, scores in sorted(
            qa_metrics.items(),
            key=lambda x: -max(roc_auc_score(qa_labels, x[1]),
                               1 - roc_auc_score(qa_labels, x[1]))):
        auc = roc_auc_score(qa_labels, scores)
        eff_auc = max(auc, 1 - auc)
        try:
            rpb, _ = pointbiserialr(qa_labels, scores)
            abs_rpb = abs(rpb)
        except Exception:
            abs_rpb = 0
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(qa_labels, scores)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]
        preds = (scores >= best_thresh).astype(int)
        bal_acc = balanced_accuracy_score(qa_labels, preds)
        print(f"  {name:<18} {eff_auc:>8.3f}   {bal_acc:>8.3f}   {abs_rpb:>8.3f}")
        halueval_results[name] = {
            "effective_auc": float(eff_auc),
            "balanced_acc": float(bal_acc),
            "abs_point_biserial": float(abs_rpb),
        }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    results = {
        "summeval": {
            "n": n,
            "correlations": summeval_results,
        },
        "halueval_qa": {
            "n": n_qa,
            "metrics": halueval_results,
        },
    }
    if nli_scores:
        total_reclassified = sum(
            s.get("nli_reclassified", 0) for s in nli_scores)
        total_fabricated = sum(
            s.get("nli_still_fabricated", 0) for s in nli_scores)
        results["nli_enrichment"] = {
            "total_reclassified": total_reclassified,
            "total_still_fabricated": total_fabricated,
            "reclassification_rate": (
                total_reclassified / (total_reclassified + total_fabricated)
                if (total_reclassified + total_fabricated) > 0 else 0),
        }

    out_file = output_dir / "chi_rf_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_file}")
    print("  DONE!")


if __name__ == "__main__":
    main()
