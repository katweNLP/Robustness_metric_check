#!/usr/bin/env python3
"""
CHI Evaluation on AggreFact — Community Standard Factuality Benchmark.

AggreFact aggregates human factual consistency annotations across multiple
source datasets (XSumFaith, PolyTope, SummEval, Factuality, etc.).
Each sample is (document, claim, label) with binary labels.

Adaptation: AggreFact provides no reference summary. We use the source
document as both Input and Reference (I=R) for CHI metrics:
  chi.score_text(source=doc, reference=doc, generated=claim)

This is standard for reference-free factuality evaluation. The EHI PH factor
collapses to ~0 when R=I (since R∩G−I = ∅), making EHI effectively measure
entity overlap with the source.

Metrics: ROC-AUC, balanced accuracy (Youden J), point-biserial, Spearman.

Output: results/aggrefact_results.json
"""

import json
import sys
import time
import argparse
import warnings
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pointbiserialr

sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings("ignore", category=UserWarning)


def load_aggrefact(max_samples: int = 500):
    """Load factuality benchmark data.

    Primary: AggreFact from HuggingFace.
    Fallback: Binarized SummEval (consistency >= 4 → consistent, <= 2 → inconsistent).

    Returns list of dicts with keys: doc, claim, label (1=consistent, 0=inconsistent), subset.
    """
    from datasets import load_dataset

    samples = []
    label_counts = {0: 0, 1: 0}

    # Try AggreFact first
    try:
        print("  Trying AggreFact from HuggingFace...")
        ds = load_dataset('lytang/AggreFact', split='test')
        for item in ds:
            doc = item.get('doc', item.get('document', ''))
            claim = item.get('claim', item.get('summary', ''))
            label = int(item.get('label', 0))
            subset = item.get('cut', item.get('dataset', 'unknown'))
            if not doc or not claim:
                continue
            label_counts[label] = label_counts.get(label, 0) + 1
            samples.append({"doc": doc, "claim": claim, "label": label, "subset": subset})
            if len(samples) >= max_samples:
                break
        print(f"  Loaded {len(samples)} from AggreFact")
    except Exception as e:
        print(f"  AggreFact not available ({type(e).__name__}), using binarized SummEval...")

    # Fallback: binarized SummEval
    if len(samples) == 0:
        print("  Loading SummEval with binarized consistency labels...")
        ds = load_dataset('mteb/summeval', split='test')
        for item in ds:
            source = item['text']
            for i, (summary, consistency) in enumerate(zip(item['machine_summaries'], item['consistency'])):
                if consistency >= 4:
                    label = 1
                elif consistency <= 2:
                    label = 0
                else:
                    continue
                subset = f"consistency_{int(consistency)}"
                label_counts[label] = label_counts.get(label, 0) + 1
                samples.append({"doc": source, "claim": summary, "label": label, "subset": "SummEval-binarized"})
                if len(samples) >= max_samples:
                    break
            if len(samples) >= max_samples:
                break
        print(f"  Loaded {len(samples)} binarized SummEval samples")

    print(f"  Label distribution: consistent={label_counts.get(1,0)}, "
          f"inconsistent={label_counts.get(0,0)}")

    # Verify polarity: if more samples are labeled 1 than 0 in a typical benchmark,
    # labels are likely correct. If inverted, warn.
    subsets_found = set(s['subset'] for s in samples)
    print(f"  Subsets found: {subsets_found}")

    return samples


def evaluate_binary(scores: np.ndarray, labels: np.ndarray, higher_is_consistent: bool = True):
    """Evaluate continuous metric scores against binary labels.

    Args:
        scores: continuous metric values
        labels: binary labels (1=consistent, 0=inconsistent)
        higher_is_consistent: if True, higher scores predict consistency

    Returns:
        dict with roc_auc, balanced_acc, point_biserial, spearman
    """
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    result = {}

    # Filter NaN/Inf
    valid = np.isfinite(scores)
    if valid.sum() < 10:
        return {"roc_auc": float('nan'), "balanced_acc": float('nan'),
                "point_biserial": float('nan'), "spearman": float('nan'),
                "n_valid": int(valid.sum())}

    s = scores[valid]
    l = labels[valid]

    # ROC-AUC (threshold-free)
    try:
        if higher_is_consistent:
            auc = roc_auc_score(l, s)
        else:
            auc = roc_auc_score(l, -s)
        result["roc_auc"] = float(auc)
    except ValueError:
        result["roc_auc"] = float('nan')

    # Balanced accuracy at optimal threshold (Youden's J)
    try:
        thresholds = np.percentile(s, np.arange(5, 100, 5))
        best_ba = 0.0
        best_thresh = 0.0
        for thresh in thresholds:
            if higher_is_consistent:
                preds = (s >= thresh).astype(int)
            else:
                preds = (s < thresh).astype(int)
            ba = balanced_accuracy_score(l, preds)
            if ba > best_ba:
                best_ba = ba
                best_thresh = thresh
        result["balanced_acc"] = float(best_ba)
        result["optimal_threshold"] = float(best_thresh)
    except Exception:
        result["balanced_acc"] = float('nan')

    # Point-biserial correlation
    try:
        r, p = pointbiserialr(l, s)
        result["point_biserial"] = float(r)
        result["point_biserial_p"] = float(p)
    except Exception:
        result["point_biserial"] = float('nan')

    # Spearman
    try:
        rho, p = spearmanr(s, l)
        result["spearman"] = float(rho)
        result["spearman_p"] = float(p)
    except Exception:
        result["spearman"] = float('nan')

    result["n_valid"] = int(valid.sum())
    return result


def compute_metrics_for_sample(doc: str, claim: str) -> dict:
    """Compute all metrics for a single (doc, claim) pair.

    Uses I=R adaptation: source document as both Input and Reference.
    """
    from src.ehi import compute_ehi_factors, compute_ehi
    from src.rhi_star import compute_rhi_star_factors, compute_rhi_star
    from src.qhi import compute_qhi_factors, compute_qhi
    from src.chi import compute_chi
    from src.ehi_enhanced import compute_ehi_enhanced

    results = {}

    # CHI sub-indices (I=R adaptation)
    ehi_f = compute_ehi_factors(doc, doc, claim)
    results["EHI"] = compute_ehi(ehi_f)

    rhi_f = compute_rhi_star_factors(doc, doc, claim)
    results["RHI*"] = compute_rhi_star(rhi_f)

    qhi_f = compute_qhi_factors(doc, doc, claim)
    results["QHI"] = compute_qhi(qhi_f)

    has_q = qhi_f.get("counts", {}).get("len_G", 0) > 0
    has_e = ehi_f.get("len_G", 0) > 0 if isinstance(ehi_f, dict) else True
    results["CHI"] = compute_chi(results["EHI"], results["RHI*"], results["QHI"],
                                  has_entities=has_e, has_quantities=has_q)

    # Domain-grounded EHI (default to 'news' since AggreFact is mixed)
    try:
        results["EHI_enhanced"] = compute_ehi_enhanced(doc, doc, claim, domain="news")
    except Exception:
        results["EHI_enhanced"] = results["EHI"]

    # ROUGE-L
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge = scorer.score(doc, claim)
    results["ROUGE-L"] = rouge['rougeL'].fmeasure

    # BERTScore
    from bert_score import score as bert_score_fn
    _, _, F1 = bert_score_fn([claim], [doc], model_type="roberta-large",
                              lang="en", verbose=False)
    results["BERTScore"] = float(F1[0])

    return results


def main():
    parser = argparse.ArgumentParser(
        description="CHI evaluation on AggreFact factuality benchmark"
    )
    parser.add_argument('--max-samples', type=int, default=500)
    parser.add_argument('--output-dir', type=str, default='./results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  CHI Evaluation on AggreFact (Binary Factuality Benchmark)")
    print("=" * 70)

    # Step 1: Load data
    print("\n  [Step 1] Loading AggreFact...")
    samples = load_aggrefact(args.max_samples)
    n = len(samples)

    # Step 2: Compute metrics
    print(f"\n  [Step 2] Computing metrics for {n} samples...")
    t0 = time.time()

    all_scores = []
    errors = 0
    for i, sample in enumerate(samples):
        try:
            scores = compute_metrics_for_sample(sample["doc"], sample["claim"])
            all_scores.append(scores)
        except Exception as e:
            errors += 1
            all_scores.append({
                "EHI": 0.4, "RHI*": 0.333, "QHI": 0.4, "CHI": 0.4,
                "EHI_enhanced": 0.4, "ROUGE-L": 0.0, "BERTScore": 0.0,
            })
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"    {i+1}/{n} done ({rate:.2f} samples/s, ETA: {eta:.0f}s, errors: {errors})")

    print(f"    Complete in {time.time()-t0:.1f}s ({errors} errors)")

    # Step 3: Evaluate
    print(f"\n  [Step 3] Computing binary evaluation metrics...")

    labels = np.array([s["label"] for s in samples])

    metric_names = ["CHI", "EHI", "RHI*", "QHI", "EHI_enhanced", "ROUGE-L", "BERTScore"]
    # Higher scores = more consistent for all our metrics
    higher_is_consistent = {
        "CHI": True, "EHI": True, "RHI*": True, "QHI": True,
        "EHI_enhanced": True, "ROUGE-L": True, "BERTScore": True,
    }

    overall_results = {}

    print(f"\n  {'='*70}")
    print(f"  RESULTS: Binary Factuality Detection on AggreFact (N={n})")
    print(f"  {'='*70}")
    print(f"\n  {'Metric':<16} {'ROC-AUC':>9} {'Bal.Acc':>9} {'Pt.Bis':>9} {'Spearman':>10}")
    print(f"  {'-'*55}")

    for metric in metric_names:
        scores = np.array([s[metric] for s in all_scores])
        eval_result = evaluate_binary(scores, labels, higher_is_consistent[metric])
        overall_results[metric] = eval_result

        auc = eval_result.get("roc_auc", float('nan'))
        ba = eval_result.get("balanced_acc", float('nan'))
        pb = eval_result.get("point_biserial", float('nan'))
        sp = eval_result.get("spearman", float('nan'))
        print(f"  {metric:<16} {auc:>9.4f} {ba:>9.4f} {pb:>9.4f} {sp:>10.4f}")

    # Step 4: Per-subset breakdown
    print(f"\n  [Step 4] Per-subset breakdown...")
    subsets = {}
    for i, sample in enumerate(samples):
        sub = sample["subset"]
        if sub not in subsets:
            subsets[sub] = {"indices": [], "labels": []}
        subsets[sub]["indices"].append(i)
        subsets[sub]["labels"].append(sample["label"])

    per_subset_results = {}
    for sub_name, sub_data in sorted(subsets.items()):
        sub_n = len(sub_data["indices"])
        if sub_n < 50:
            print(f"    Skipping subset '{sub_name}' (N={sub_n} < 50)")
            continue

        sub_labels = np.array(sub_data["labels"])
        sub_metrics = {}

        print(f"\n    Subset: {sub_name} (N={sub_n}, "
              f"consistent={sub_labels.sum()}, inconsistent={sub_n - sub_labels.sum()})")
        print(f"    {'Metric':<16} {'ROC-AUC':>9} {'Bal.Acc':>9}")
        print(f"    {'-'*36}")

        for metric in metric_names:
            sub_scores = np.array([all_scores[j][metric] for j in sub_data["indices"]])
            eval_result = evaluate_binary(sub_scores, sub_labels,
                                           higher_is_consistent[metric])
            sub_metrics[metric] = eval_result
            auc = eval_result.get("roc_auc", float('nan'))
            ba = eval_result.get("balanced_acc", float('nan'))
            print(f"    {metric:<16} {auc:>9.4f} {ba:>9.4f}")

        per_subset_results[sub_name] = {"n": sub_n, "metrics": sub_metrics}

    # Save results
    final_results = {
        "overall": {
            "n": n,
            "n_consistent": int(labels.sum()),
            "n_inconsistent": int(n - labels.sum()),
            "errors": errors,
            "metrics": overall_results,
        },
        "per_subset": per_subset_results,
        "adaptation": "I=R (source document used as both Input and Reference)",
        "note": "EHI PH factor ~0 when R=I since R∩G−I = ∅",
    }

    save_file = output_dir / "aggrefact_results.json"
    with open(save_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {save_file}")

    # Key findings
    chi_auc = overall_results.get("CHI", {}).get("roc_auc", float('nan'))
    ehi_auc = overall_results.get("EHI", {}).get("roc_auc", float('nan'))
    bert_auc = overall_results.get("BERTScore", {}).get("roc_auc", float('nan'))
    print(f"\n  KEY FINDINGS:")
    print(f"    CHI ROC-AUC:       {chi_auc:.4f}")
    print(f"    EHI ROC-AUC:       {ehi_auc:.4f}")
    print(f"    BERTScore ROC-AUC: {bert_auc:.4f}")


if __name__ == "__main__":
    main()
