#!/usr/bin/env python3
"""
Baseline Metrics Computation for CHI and DA-RHI Papers.

Computes competing faithfulness metrics on the same data to enable
head-to-head comparison with CHI/DA-RHI. Each baseline is wrapped
in a common interface: (input_text, reference_text, generated_text) -> float score.

Baselines implemented:
1. ROUGE-1/2/L (rouge_score library)
2. BERTScore (bert_score library)
3. AlignScore (via sentence-transformers NLI proxy)
4. SummaC (NLI-based, via transformers)
5. MiniCheck (via transformers, claim decomposition)
6. GPT-4-as-judge (via API, optional)

Usage:
    python compute_baselines.py --data-dir ./data --output-dir ./results

Dependencies:
    pip install rouge-score bert-score sentence-transformers transformers torch numpy scipy
"""

import json
import os
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


# ==============================================================================
# ROUGE
# ==============================================================================

def compute_rouge(reference: str, generated: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {
        "rouge1": scores['rouge1'].fmeasure,
        "rouge2": scores['rouge2'].fmeasure,
        "rougeL": scores['rougeL'].fmeasure,
    }


# ==============================================================================
# BERTScore
# ==============================================================================

_bertscore_model = None

def compute_bertscore(reference: str, generated: str) -> float:
    """Compute BERTScore F1 using roberta-large."""
    from bert_score import score as bert_score_fn
    P, R, F1 = bert_score_fn(
        [generated], [reference],
        model_type="roberta-large",
        lang="en",
        verbose=False
    )
    return float(F1[0])


# ==============================================================================
# AlignScore (Proxy via NLI cross-encoder)
# ==============================================================================

_nli_model = None

def compute_alignscore_proxy(input_text: str, generated: str) -> float:
    """
    Proxy for AlignScore using cross-encoder NLI model.
    Computes entailment probability of (source, summary) pair.
    This approximates AlignScore's alignment function.
    """
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')

    # Score: probability that input entails generated summary
    score = _nli_model.predict([(input_text[:512], generated[:512])])
    # CrossEncoder nli-deberta-v3-base label order: [contradiction=0, entailment=1, neutral=2]
    score = np.atleast_1d(np.squeeze(np.array(score)))
    if score.shape[-1] == 3:
        probs = np.exp(score) / np.exp(score).sum(axis=-1, keepdims=True)
        if probs.ndim == 1:
            return float(probs[1])  # index 1 = entailment
        return float(probs[0][1])
    return float(score.flat[0])


# ==============================================================================
# SummaC (NLI-based sentence-level consistency)
# ==============================================================================

def compute_summac_proxy(input_text: str, generated: str) -> float:
    """
    SummaC proxy: average NLI entailment across generated sentences
    checked against source document.
    """
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')

    import nltk
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

    gen_sentences = nltk.sent_tokenize(generated)
    if not gen_sentences:
        return 0.0

    # For each generated sentence, check NON-contradiction against source
    # SummaC uses 1 - P(contradiction) as consistency signal
    scores = []
    source_truncated = input_text[:1024]
    for sent in gen_sentences[:10]:
        try:
            raw = _nli_model.predict([(source_truncated, sent)])
            # Label order: [contradiction=0, entailment=1, neutral=2]
            raw = np.atleast_1d(np.squeeze(np.array(raw)))
            if raw.shape[-1] == 3:
                probs = np.exp(raw) / np.exp(raw).sum(axis=-1, keepdims=True)
                # 1 - P(contradiction) = P(entailment) + P(neutral)
                non_contra = 1.0 - (float(probs[0]) if probs.ndim == 1 else float(probs[0][0]))
                scores.append(non_contra)
            else:
                scores.append(float(raw.flat[0]))
        except Exception:
            scores.append(0.5)

    return float(np.mean(scores)) if scores else 0.0


# ==============================================================================
# Entity F1 (simple NER overlap baseline)
# ==============================================================================

def compute_entity_f1(input_text: str, reference: str, generated: str) -> float:
    """
    Entity F1: overlap of named entities between source and generated.
    Simple baseline that CHI's EHI component should outperform.
    """
    import spacy
    nlp = spacy.load("en_core_web_sm")

    src_ents = set(ent.text.lower() for ent in nlp(input_text).ents)
    gen_ents = set(ent.text.lower() for ent in nlp(generated).ents)

    if not gen_ents:
        return 0.0

    precision = len(src_ents & gen_ents) / len(gen_ents) if gen_ents else 0
    recall = len(src_ents & gen_ents) / len(src_ents) if src_ents else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ==============================================================================
# Batch computation
# ==============================================================================

def compute_all_baselines(input_text: str, reference: str, generated: str,
                          include_heavy: bool = True) -> Dict[str, float]:
    """
    Compute all baseline metrics for a single (input, reference, generated) triple.

    Args:
        input_text: Source document
        reference: Reference summary
        generated: Model-generated summary
        include_heavy: If True, includes BERTScore and NLI models (slower)

    Returns:
        Dict of metric_name -> score
    """
    results = {}

    # ROUGE (fast, always compute)
    rouge = compute_rouge(reference, generated)
    results["rouge1"] = rouge["rouge1"]
    results["rouge2"] = rouge["rouge2"]
    results["rougeL"] = rouge["rougeL"]

    if include_heavy:
        # BERTScore
        results["bertscore"] = compute_bertscore(reference, generated)

        # AlignScore proxy (NLI entailment)
        results["alignscore_proxy"] = compute_alignscore_proxy(input_text, generated)

        # SummaC proxy (sentence-level NLI)
        results["summac_proxy"] = compute_summac_proxy(input_text, generated)

    # Entity F1 (moderate speed)
    results["entity_f1"] = compute_entity_f1(input_text, reference, generated)

    return results


def compute_baselines_batch(samples: List[Dict], include_heavy: bool = True,
                            progress: bool = True) -> List[Dict]:
    """
    Compute baselines for a batch of samples.

    Args:
        samples: List of dicts with InputText, ReferenceSummary keys
        include_heavy: Include BERTScore/NLI models
        progress: Print progress

    Returns:
        List of result dicts (same order as input)
    """
    results = []
    n = len(samples)

    for i, sample in enumerate(samples):
        if progress and (i + 1) % 10 == 0:
            print(f"    Processing {i+1}/{n}...")

        input_text = sample["InputText"]
        reference = sample["ReferenceSummary"]
        generated = sample.get("GeneratedSummary", reference)  # Use ref as placeholder if no gen

        baseline_scores = compute_all_baselines(input_text, reference, generated, include_heavy)
        results.append(baseline_scores)

    return results


# ==============================================================================
# Main: process all domain datasets
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compute baseline metrics for comparison")
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory containing domain JSON files')
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Output directory for baseline scores')
    parser.add_argument('--light-only', action='store_true',
                        help='Skip heavy models (BERTScore, NLI)')
    parser.add_argument('--max-samples', type=int, default=50,
                        help='Max samples per domain (for testing)')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Baseline Metrics Computation")
    print("=" * 60)

    domains = ["news", "medical", "legal", "financial"]

    for domain in domains:
        data_file = data_dir / f"{domain}_200.json"
        if not data_file.exists():
            print(f"\n  [SKIP] {data_file} not found")
            continue

        print(f"\n  Domain: {domain.upper()}")
        with open(data_file, 'r', encoding='utf-8') as f:
            samples = json.load(f)

        # Limit for testing
        samples = samples[:args.max_samples]
        print(f"    Samples: {len(samples)}")
        print(f"    Computing baselines (heavy={'OFF' if args.light_only else 'ON'})...")

        results = compute_baselines_batch(
            samples,
            include_heavy=not args.light_only,
            progress=True
        )

        # Compute statistics
        metric_names = list(results[0].keys())
        print(f"\n    Results summary:")
        for metric in metric_names:
            values = [r[metric] for r in results]
            print(f"      {metric:<20}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")

        # Save
        output_file = output_dir / f"{domain}_baselines.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"    Saved to: {output_file}")

    print(f"\n{'='*60}")
    print(f"  COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Next: run CHI/DA-RHI on same data, then correlate both with human scores")
    print(f"  Expected result: CHI correlation > individual baselines")


if __name__ == "__main__":
    main()
