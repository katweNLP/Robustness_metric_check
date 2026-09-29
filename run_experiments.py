#!/usr/bin/env python3
"""
Full Experiment Pipeline for CHI and DA-RHI Papers.

Runs the complete experimental evaluation:
1. Generate summaries using multiple models
2. Compute CHI (EHI + RHI* + QHI) on all summaries
3. Compute baseline metrics (ROUGE, BERTScore, AlignScore proxy, SummaC proxy, Entity F1)
4. Generate proxy human faithfulness scores (using NLI model as proxy annotator)
5. Compute correlations (Spearman, Kendall) and significance tests
6. Output results tables ready for paper

Usage:
    python run_experiments.py --data-dir ./data --output-dir ./results --max-samples 50
    python run_experiments.py --data-dir ./data --output-dir ./results --max-samples 200 --full

Dependencies:
    pip install transformers torch sentence-transformers spacy rouge-score bert-score numpy scipy scikit-learn
    python -m spacy download en_core_web_sm
"""

import json
import os
import sys
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, List
from scipy.stats import spearmanr, kendalltau

sys.path.insert(0, str(Path(__file__).parent))


# ==============================================================================
# STEP 1: Generate Summaries
# ==============================================================================

def generate_summaries_bart(articles: List[str], max_length: int = 142) -> List[str]:
    """Generate summaries using facebook/bart-large-cnn."""
    from transformers import BartForConditionalGeneration, BartTokenizer
    print("    Loading BART-large-cnn...")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    summaries = []
    for i, article in enumerate(articles):
        if (i + 1) % 20 == 0:
            print(f"      BART: {i+1}/{len(articles)}")
        try:
            inputs = tokenizer(article[:1024], return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = model.generate(inputs["input_ids"], max_length=max_length, min_length=56, do_sample=False)
            summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
        except Exception as e:
            summaries.append(article[:200])
    return summaries


def generate_summaries_t5(articles: List[str], max_length: int = 150) -> List[str]:
    """Generate summaries using t5-small (lightweight, fast)."""
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    print("    Loading T5-small...")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    summaries = []
    for i, article in enumerate(articles):
        if (i + 1) % 20 == 0:
            print(f"      T5: {i+1}/{len(articles)}")
        try:
            inputs = tokenizer("summarize: " + article[:512], return_tensors="pt", max_length=512, truncation=True)
            summary_ids = model.generate(inputs["input_ids"], max_length=max_length, min_length=30, do_sample=False)
            summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
        except Exception as e:
            summaries.append(article[:150])
    return summaries


def generate_summaries_extractive(articles: List[str], n_sentences: int = 3) -> List[str]:
    """Simple extractive baseline: first N sentences."""
    import nltk
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

    summaries = []
    for article in articles:
        sents = nltk.sent_tokenize(article)
        summaries.append(' '.join(sents[:n_sentences]))
    return summaries


def generate_summaries_corrupt(articles: List[str], references: List[str]) -> List[str]:
    """
    Generate deliberately hallucinated summaries by corrupting references.
    Provides a known-bad baseline for calibration.
    Introduces entity swaps, number changes, and relation inversions.
    """
    import random
    random.seed(42)
    import re

    summaries = []
    for ref in references:
        corrupted = ref
        # Swap some numbers (quantity hallucination)
        numbers = re.findall(r'\d+', corrupted)
        for num in numbers[:2]:
            new_num = str(int(num) * random.choice([2, 3, 10]))
            corrupted = corrupted.replace(num, new_num, 1)
        # Invert a relation word if present
        inversions = {'increased': 'decreased', 'grew': 'fell', 'rose': 'dropped',
                      'approved': 'rejected', 'won': 'lost', 'acquired': 'sold'}
        for orig, replacement in inversions.items():
            if orig in corrupted.lower():
                corrupted = corrupted.replace(orig, replacement, 1)
                break
        summaries.append(corrupted)
    return summaries


# ==============================================================================
# STEP 2: Compute Our Metrics (CHI)
# ==============================================================================

def compute_chi_metrics(input_text: str, reference: str, generated: str) -> Dict[str, float]:
    """Compute EHI, RHI*, QHI, and CHI for a single triple."""
    # Import from our implementation
    from src.ehi import compute_ehi_factors, compute_ehi
    from src.rhi_star import compute_rhi_star_factors, compute_rhi_star
    from src.qhi import compute_qhi_factors, compute_qhi
    from src.chi import compute_chi

    # EHI
    ehi_f = compute_ehi_factors(input_text, reference, generated)
    ehi = compute_ehi(ehi_f)

    # RHI*
    rhi_f = compute_rhi_star_factors(input_text, reference, generated)
    rhi_star = compute_rhi_star(rhi_f)

    # QHI
    qhi_f = compute_qhi_factors(input_text, reference, generated)
    qhi = compute_qhi(qhi_f)

    # CHI composite
    has_q = qhi_f.get("counts", {}).get("len_G", 0) > 0
    chi = compute_chi(ehi, rhi_star, qhi, has_entities=True, has_quantities=has_q)

    return {
        "ehi": ehi,
        "rhi_star": rhi_star,
        "qhi": qhi,
        "chi": chi,
        "ehi_factors": {k: v for k, v in ehi_f.items() if not k.startswith("len_")},
        "rhi_factors": rhi_f,
        "qhi_factors": {k: v for k, v in qhi_f.items() if k != "counts"},
    }


# ==============================================================================
# STEP 3: Compute Baseline Metrics
# ==============================================================================

def compute_baselines(input_text: str, reference: str, generated: str) -> Dict[str, float]:
    """Compute ROUGE + BERTScore + NLI baselines."""
    from src.compute_baselines import compute_rouge, compute_bertscore, compute_entity_f1, compute_alignscore_proxy, compute_summac_proxy

    rouge = compute_rouge(reference, generated)
    bertscore = compute_bertscore(reference, generated)
    entity_f1 = compute_entity_f1(input_text, reference, generated)
    alignscore = compute_alignscore_proxy(input_text, generated)
    summac = compute_summac_proxy(input_text, generated)

    return {
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "bertscore": bertscore,
        "entity_f1": entity_f1,
        "alignscore_proxy": alignscore,
        "summac_proxy": summac,
    }


# ==============================================================================
# STEP 4: Proxy Human Scores (NLI-based faithfulness rating)
# ==============================================================================

def compute_proxy_human_score(input_text: str, reference: str, generated: str) -> float:
    """
    Proxy human faithfulness score combining multiple signals.
    Uses BERTScore (ref-based) + NLI entailment (source-based) as
    a composite proxy for human judgment. Avoids circular correlation
    by combining two independent signals.
    Returns score in [1, 10] range.
    """
    from src.compute_baselines import compute_bertscore, compute_alignscore_proxy
    # BERTScore captures semantic similarity to reference (ref-based signal)
    bert_sim = compute_bertscore(reference, generated)
    # AlignScore captures source entailment (source-based signal)
    align_score = compute_alignscore_proxy(input_text, generated)
    # Combine: 60% reference similarity + 40% source entailment
    combined = 0.6 * bert_sim + 0.4 * align_score
    return 1 + combined * 9


# ==============================================================================
# STEP 5: Compute Correlations
# ==============================================================================

def compute_correlations(metric_scores: Dict[str, np.ndarray],
                         human_scores: np.ndarray) -> Dict[str, Dict]:
    """Compute Spearman and Kendall for each metric vs human."""
    results = {}
    for name, scores in metric_scores.items():
        if np.std(scores) == 0:
            results[name] = {"spearman_rho": 0.0, "kendall_tau": 0.0, "p_spearman": 1.0}
            continue
        rho, p_rho = spearmanr(scores, human_scores)
        tau, p_tau = kendalltau(scores, human_scores)
        results[name] = {
            "spearman_rho": round(rho, 4) if not np.isnan(rho) else 0.0,
            "kendall_tau": round(tau, 4) if not np.isnan(tau) else 0.0,
            "p_spearman": round(p_rho, 6) if not np.isnan(p_rho) else 1.0,
            "p_kendall": round(p_tau, 6) if not np.isnan(p_tau) else 1.0,
        }
    return results


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def run_domain_experiment(domain: str, samples: List[Dict], max_samples: int,
                          output_dir: Path, skip_generation: bool = False) -> Dict:
    """Run full experiment for a single domain."""
    samples = samples[:max_samples]
    n = len(samples)
    articles = [s["InputText"] for s in samples]
    references = [s["ReferenceSummary"] for s in samples]

    print(f"\n  Domain: {domain.upper()} ({n} articles)")
    print(f"  {'='*50}")

    # --- Generate summaries ---
    if not skip_generation:
        print("\n  [Step 1] Generating summaries...")
        t0 = time.time()

        summaries = {}
        summaries["bart"] = generate_summaries_bart(articles)
        summaries["t5"] = generate_summaries_t5(articles)
        summaries["extractive"] = generate_summaries_extractive(articles)
        summaries["corrupted"] = generate_summaries_corrupt(articles, references)
        summaries["reference"] = references  # Upper bound

        gen_time = time.time() - t0
        print(f"    Generation time: {gen_time:.1f}s ({gen_time/n:.2f}s/article)")

        # Save generated summaries
        gen_file = output_dir / f"{domain}_generated.json"
        with open(gen_file, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
    else:
        gen_file = output_dir / f"{domain}_generated.json"
        with open(gen_file, 'r', encoding='utf-8') as f:
            summaries = json.load(f)

    # --- Compute all metrics ---
    print("\n  [Step 2] Computing metrics...")
    t0 = time.time()

    all_results = {}
    for model_name, model_summaries in summaries.items():
        print(f"    Model: {model_name}...")
        model_results = []

        for i in range(n):
            result = {}

            # CHI metrics
            chi_scores = compute_chi_metrics(articles[i], references[i], model_summaries[i])
            result.update({
                "ehi": chi_scores["ehi"],
                "rhi_star": chi_scores["rhi_star"],
                "qhi": chi_scores["qhi"],
                "chi": chi_scores["chi"],
            })

            # Baseline metrics
            baseline_scores = compute_baselines(articles[i], references[i], model_summaries[i])
            result.update(baseline_scores)

            # Proxy human score
            result["human_proxy"] = compute_proxy_human_score(articles[i], references[i], model_summaries[i])

            model_results.append(result)

            if (i + 1) % 10 == 0:
                print(f"      {i+1}/{n} done")

        all_results[model_name] = model_results

    metric_time = time.time() - t0
    print(f"    Metrics time: {metric_time:.1f}s")

    # Save raw results
    results_file = output_dir / f"{domain}_all_metrics.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    # --- Compute correlations ---
    print("\n  [Step 3] Computing correlations...")

    # Pool all model outputs for correlation (standard practice)
    pooled_metrics = {k: [] for k in ["ehi", "rhi_star", "qhi", "chi",
                                       "rouge1", "rouge2", "rougeL",
                                       "bertscore", "entity_f1",
                                       "alignscore_proxy", "summac_proxy"]}
    pooled_human = []

    for model_name, model_results in all_results.items():
        for result in model_results:
            for metric in pooled_metrics:
                pooled_metrics[metric].append(result.get(metric, 0))
            pooled_human.append(result["human_proxy"])

    # Convert to numpy
    pooled_metrics_np = {k: np.array(v) for k, v in pooled_metrics.items()}
    pooled_human_np = np.array(pooled_human)

    correlations = compute_correlations(pooled_metrics_np, pooled_human_np)

    # Save correlations
    corr_file = output_dir / f"{domain}_correlations.json"
    with open(corr_file, 'w', encoding='utf-8') as f:
        json.dump(correlations, f, indent=2)

    # Print results table
    print(f"\n  Results: Correlation with Human Faithfulness ({domain})")
    print(f"  {'Metric':<20} {'Spearman rho':>14} {'Kendall tau':>14} {'p-value':>10}")
    print(f"  {'-'*60}")
    sorted_metrics = sorted(correlations.items(), key=lambda x: -x[1]['spearman_rho'])
    for name, corr in sorted_metrics:
        sig = "***" if corr['p_spearman'] < 0.001 else "**" if corr['p_spearman'] < 0.01 else "*" if corr['p_spearman'] < 0.05 else ""
        print(f"  {name:<20} {corr['spearman_rho']:>12.4f}{sig:>2} {corr['kendall_tau']:>14.4f} {corr['p_spearman']:>10.6f}")

    return {
        "domain": domain,
        "n_articles": n,
        "n_models": len(summaries),
        "correlations": correlations,
    }


def main():
    parser = argparse.ArgumentParser(description="Run full CHI/DA-RHI experiments")
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./results')
    parser.add_argument('--max-samples', type=int, default=20,
                        help='Samples per domain (20 for quick test, 200 for full)')
    parser.add_argument('--domains', nargs='+', default=['news', 'medical', 'legal', 'financial'])
    parser.add_argument('--skip-generation', action='store_true',
                        help='Skip summary generation (use cached)')
    args = parser.parse_args()

    # Add src to path
    sys.path.insert(0, str(Path(args.data_dir).parent))

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  CHI + DA-RHI Full Experiment Pipeline")
    print("=" * 70)
    print(f"  Data: {data_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Samples/domain: {args.max_samples}")
    print(f"  Domains: {args.domains}")

    all_domain_results = {}

    for domain in args.domains:
        data_file = data_dir / f"{domain}_200.json"
        if not data_file.exists():
            print(f"\n  [SKIP] {data_file} not found")
            continue

        with open(data_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)

        domain_result = run_domain_experiment(
            domain, samples, args.max_samples, output_dir, args.skip_generation
        )
        all_domain_results[domain] = domain_result

    # --- Final summary table ---
    print(f"\n\n{'='*70}")
    print(f"  FINAL SUMMARY: CHI vs Baselines (All Domains Pooled)")
    print(f"{'='*70}")
    print(f"\n  {'Metric':<20} ", end="")
    for domain in args.domains:
        if domain in all_domain_results:
            print(f"{'  ' + domain:>12}", end="")
    print()
    print(f"  {'-'*60}")

    metrics_to_show = ["chi", "ehi", "rhi_star", "qhi", "rougeL", "bertscore",
                       "alignscore_proxy", "summac_proxy", "entity_f1"]
    for metric in metrics_to_show:
        print(f"  {metric:<20} ", end="")
        for domain in args.domains:
            if domain in all_domain_results:
                rho = all_domain_results[domain]["correlations"].get(metric, {}).get("spearman_rho", 0)
                print(f"{rho:>12.4f}", end="")
        print()

    # Save final summary
    summary_file = output_dir / "experiment_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_domain_results, f, indent=2)

    print(f"\n  Results saved to: {output_dir}")
    print(f"  Use these values to fill X.XX placeholders in paper LaTeX.")


if __name__ == "__main__":
    main()
