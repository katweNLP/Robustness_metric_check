#!/usr/bin/env python3
"""
CHI Evaluation on HaluEval QA — Non-Summarization Task Transfer.

Tests whether CHI metrics generalise beyond summarisation to QA hallucination
detection. HaluEval provides (knowledge, question, answer, hallucination_label)
tuples with binary ground truth.

Mapping:
    Input     = question + " " + knowledge  (context for NER / SVO)
    Reference = right_answer                 (gold answer)
    Generated = answer                       (candidate to evaluate)
"""

import json
import sys
import time
import argparse
import random
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, kendalltau, pointbiserialr

sys.path.insert(0, str(Path(__file__).parent))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_halueval_qa(max_samples: int = 500):
    """Load HaluEval QA split from HuggingFace."""
    from datasets import load_dataset

    # Try known config names in priority order
    configs_to_try = [
        ("pminervini/HaluEval", "qa_samples"),
        ("pminervini/HaluEval", "qa"),
        ("pminervini/HaluEval", None),
    ]

    ds = None
    for repo, config in configs_to_try:
        try:
            if config:
                ds = load_dataset(repo, config, split="data", trust_remote_code=True)
            else:
                ds = load_dataset(repo, split="data", trust_remote_code=True)
            print(f"  Loaded HaluEval via ({repo}, {config})")
            break
        except Exception:
            try:
                if config:
                    ds = load_dataset(repo, config, trust_remote_code=True)
                else:
                    ds = load_dataset(repo, trust_remote_code=True)
                if hasattr(ds, 'keys'):
                    split_name = list(ds.keys())[0]
                    ds = ds[split_name]
                print(f"  Loaded HaluEval via ({repo}, {config}) split={split_name}")
                break
            except Exception:
                continue

    if ds is None:
        raise RuntimeError(
            "Could not load HaluEval. Tried: " +
            str([c for c in configs_to_try])
        )

    samples = []
    for item in ds:
        knowledge = item.get("knowledge", item.get("context", ""))
        question = item.get("question", "")
        answer = item.get("answer", item.get("response", ""))
        right_answer = item.get("right_answer", item.get("correct_answer", ""))
        hallucination = item.get("hallucination", item.get("label", ""))

        if not knowledge or not answer:
            continue

        if isinstance(hallucination, str):
            is_hallucinated = hallucination.strip().lower() == "yes"
        elif isinstance(hallucination, (int, float)):
            is_hallucinated = bool(hallucination)
        else:
            continue

        samples.append({
            "knowledge": knowledge,
            "question": question,
            "answer": answer,
            "right_answer": right_answer if right_answer else answer,
            "hallucination": is_hallucinated,
        })

        if len(samples) >= max_samples:
            break

    n_pos = sum(s["hallucination"] for s in samples)
    print(f"  Loaded {len(samples)} QA samples (hallucinated={n_pos}, faithful={len(samples)-n_pos})")
    return samples


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_chi_score(source, reference, generated):
    """Compute CHI and sub-indices."""
    from src.ehi import compute_ehi_factors, compute_ehi
    from src.rhi_star import compute_rhi_star_factors, compute_rhi_star
    from src.qhi import compute_qhi_factors, compute_qhi
    from src.chi import compute_chi

    ehi_f = compute_ehi_factors(source, reference, generated)
    ehi = compute_ehi(ehi_f)

    rhi_f = compute_rhi_star_factors(source, reference, generated)
    rhi_star = compute_rhi_star(rhi_f)

    qhi_f = compute_qhi_factors(source, reference, generated)
    qhi = compute_qhi(qhi_f)

    has_ent = ehi_f.get("len_G", 0) > 0 if isinstance(ehi_f, dict) else True
    has_q = qhi_f.get("counts", {}).get("len_G", 0) > 0
    chi = compute_chi(ehi, rhi_star, qhi, has_entities=has_ent, has_quantities=has_q)

    return {"chi": chi, "ehi": ehi, "rhi_star": rhi_star, "qhi": qhi}


def compute_rouge_score(reference, generated):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores['rougeL'].fmeasure


def compute_bertscore_batch(references, generateds):
    from bert_score import score as bert_score_fn
    P, R, F1 = bert_score_fn(generateds, references, model_type="roberta-large",
                              lang="en", verbose=False)
    return F1.numpy().tolist()


def extraction_statistics(samples):
    """Report median entity / SVO-triple counts in the answer field."""
    from src.ehi import extract_entities
    from src.rhi_star import extract_relations

    entity_counts = []
    triple_counts = []
    for s in samples[:100]:
        ents = extract_entities(s["answer"])
        entity_counts.append(len(ents))
        rels = extract_relations(s["answer"])
        triple_counts.append(len(rels))

    return {
        "median_entities_per_answer": float(np.median(entity_counts)),
        "mean_entities_per_answer": float(np.mean(entity_counts)),
        "median_triples_per_answer": float(np.median(triple_counts)),
        "mean_triples_per_answer": float(np.mean(triple_counts)),
        "pct_answers_with_entities": float(np.mean([c > 0 for c in entity_counts]) * 100),
        "pct_answers_with_triples": float(np.mean([c > 0 for c in triple_counts]) * 100),
    }


# ---------------------------------------------------------------------------
# Binary classification evaluation
# ---------------------------------------------------------------------------

def evaluate_binary(scores, labels):
    """ROC-AUC, balanced accuracy, point-biserial, Spearman vs binary labels."""
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    scores = np.array(scores, dtype=float)
    labels = np.array(labels, dtype=int)

    mask = np.isfinite(scores)
    scores = scores[mask]
    labels = labels[mask]

    if len(np.unique(labels)) < 2 or len(scores) < 10:
        return {"roc_auc": None, "balanced_acc": None, "point_biserial": None,
                "spearman": None, "n": int(len(scores))}

    # For CHI metrics, higher = better (faithful). HaluEval label 1 = hallucinated.
    # So faithful samples (label=0) should have HIGHER CHI scores.
    # For ROC-AUC we need P(positive=hallucinated | score). Negate score.
    try:
        roc = roc_auc_score(labels, -scores)
    except Exception:
        roc = None

    # Balanced accuracy at optimal threshold (Youden's J)
    try:
        thresholds = np.percentile(scores, np.arange(5, 100, 5))
        best_ba = 0.0
        for thresh in thresholds:
            preds = (scores < thresh).astype(int)  # below threshold = hallucinated
            ba = balanced_accuracy_score(labels, preds)
            if ba > best_ba:
                best_ba = ba
        bal_acc = best_ba
    except Exception:
        bal_acc = None

    # Point-biserial correlation
    try:
        pb_r, pb_p = pointbiserialr(labels, scores)
        pb = float(pb_r)
    except Exception:
        pb = None

    # Spearman
    try:
        rho, p = spearmanr(scores, labels)
        sp = float(rho)
    except Exception:
        sp = None

    return {
        "roc_auc": float(roc) if roc is not None else None,
        "balanced_acc": float(bal_acc) if bal_acc is not None else None,
        "point_biserial": pb,
        "spearman": sp,
        "n": int(len(scores)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CHI evaluation on HaluEval QA")
    parser.add_argument('--max-samples', type=int, default=500)
    parser.add_argument('--output-dir', type=str, default='./results')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  CHI Evaluation on HaluEval QA (Non-Summarization Transfer)")
    print("=" * 70)

    # Load
    print("\n  [Step 1] Loading HaluEval QA...")
    samples = load_halueval_qa(args.max_samples)
    n = len(samples)

    # Extraction statistics (on first 100)
    print("\n  [Step 2] Computing extraction statistics...")
    ext_stats = extraction_statistics(samples)
    print(f"    Median entities/answer: {ext_stats['median_entities_per_answer']:.1f}")
    print(f"    Median SVO triples/answer: {ext_stats['median_triples_per_answer']:.1f}")
    print(f"    Answers with entities: {ext_stats['pct_answers_with_entities']:.1f}%")
    print(f"    Answers with triples: {ext_stats['pct_answers_with_triples']:.1f}%")

    # Compute ROUGE-L
    print("\n  [Step 3] Computing ROUGE-L...")
    t0 = time.time()
    rouge_scores = []
    for s in samples:
        ref = s["right_answer"] if s["right_answer"] else s["answer"]
        rouge_scores.append(compute_rouge_score(ref, s["answer"]))
    print(f"    ROUGE-L time: {time.time()-t0:.1f}s")

    # Compute BERTScore
    print("\n  [Step 4] Computing BERTScore...")
    t0 = time.time()
    refs = [s["right_answer"] if s["right_answer"] else s["answer"] for s in samples]
    gens = [s["answer"] for s in samples]
    bert_f1s = compute_bertscore_batch(refs, gens)
    print(f"    BERTScore time: {time.time()-t0:.1f}s")

    # Compute CHI
    print(f"\n  [Step 5] Computing CHI metrics ({n} samples)...")
    t0 = time.time()
    chi_scores = []
    for i, s in enumerate(samples):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (n - i - 1) / rate
            print(f"    {i+1}/{n}... ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

        input_text = s["question"] + " " + s["knowledge"]
        reference = s["right_answer"] if s["right_answer"] else s["answer"]
        generated = s["answer"]

        try:
            scores = compute_chi_score(input_text, reference, generated)
        except Exception as e:
            scores = {"chi": 0.4, "ehi": 0.4, "rhi_star": 0.333, "qhi": 0.4}
        chi_scores.append(scores)
    print(f"    CHI time: {time.time()-t0:.1f}s")

    # Assemble
    labels = [int(s["hallucination"]) for s in samples]

    all_metrics = {
        "CHI": [s["chi"] for s in chi_scores],
        "EHI": [s["ehi"] for s in chi_scores],
        "RHI*": [s["rhi_star"] for s in chi_scores],
        "QHI": [s["qhi"] for s in chi_scores],
        "ROUGE-L": rouge_scores,
        "BERTScore": bert_f1s,
    }

    # Evaluate
    print(f"\n  {'='*70}")
    print(f"  RESULTS: Binary Classification — HaluEval QA (N={n})")
    print(f"  {'='*70}")
    print(f"\n  {'Metric':<12} {'ROC-AUC':>10} {'Bal.Acc':>10} {'Pt-Biserial':>12} {'Spearman':>10}")
    print(f"  {'-'*58}")

    results = {}
    for name, scores in all_metrics.items():
        ev = evaluate_binary(scores, labels)
        roc_str = f"{ev['roc_auc']:.4f}" if ev['roc_auc'] is not None else "N/A"
        ba_str = f"{ev['balanced_acc']:.4f}" if ev['balanced_acc'] is not None else "N/A"
        pb_str = f"{ev['point_biserial']:.4f}" if ev['point_biserial'] is not None else "N/A"
        sp_str = f"{ev['spearman']:.4f}" if ev['spearman'] is not None else "N/A"
        print(f"  {name:<12} {roc_str:>10} {ba_str:>10} {pb_str:>12} {sp_str:>10}")
        results[name] = ev

    # Save
    output = {
        "task": "HaluEval QA — Non-Summarization Transfer",
        "n_samples": n,
        "n_hallucinated": sum(labels),
        "n_faithful": n - sum(labels),
        "extraction_statistics": ext_stats,
        "results": results,
    }

    save_file = output_dir / "halueval_qa_results.json"
    with open(save_file, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)
    print(f"\n  Results saved to: {save_file}")

    # Summary
    print(f"\n  KEY FINDINGS:")
    best_metric = max(results.items(), key=lambda x: x[1].get("roc_auc") or 0)
    print(f"    Best ROC-AUC: {best_metric[0]} = {best_metric[1].get('roc_auc', 'N/A')}")
    for name in ["CHI", "EHI", "RHI*"]:
        auc = results[name].get("roc_auc")
        if auc:
            print(f"    {name}: ROC-AUC = {auc:.4f}")


if __name__ == "__main__":
    main()
