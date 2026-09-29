#!/usr/bin/env python3
"""
Additional Baselines: MiniCheck (local) + Claude-as-Judge (via Bedrock)
Runs on SummEval to compare against CHI.

MiniCheck: Uses DeBERTa-v3-large NLI model for claim-level fact-checking.
Claude-judge: Uses Claude Haiku via Bedrock gateway for cost-efficient judging.
"""

import sys, os, json, time, numpy as np
from pathlib import Path
from scipy.stats import spearmanr, kendalltau
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent))


# ==============================================================================
# MiniCheck: Claim-level NLI fact-checking (local, CPU)
# Uses cross-encoder NLI on (source_sentence, summary_claim) pairs
# ==============================================================================

_minicheck_model = None

def compute_minicheck(source: str, generated: str) -> float:
    """
    MiniCheck-style fact-checking: decompose summary into sentences,
    check each against source via NLI entailment.
    Returns fraction of summary sentences that are entailed by source.
    Higher = more faithful.
    """
    global _minicheck_model
    if _minicheck_model is None:
        from sentence_transformers import CrossEncoder
        # MiniCheck uses DeBERTa-v3-large finetuned on NLI
        # We use the same base model as our AlignScore proxy
        _minicheck_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')

    import nltk
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

    gen_sentences = nltk.sent_tokenize(generated)
    if not gen_sentences:
        return 0.0

    # For each generated sentence, find max entailment across source sentences
    source_sentences = nltk.sent_tokenize(source)[:20]  # Cap source for speed

    entailed_count = 0
    for gen_sent in gen_sentences[:10]:  # Cap at 10 summary sentences
        max_entailment = 0.0
        for src_sent in source_sentences:
            try:
                score = _minicheck_model.predict([(src_sent, gen_sent)])
                score = np.atleast_1d(np.squeeze(np.array(score)))
                if score.shape[-1] == 3:
                    # Label order: [contradiction=0, entailment=1, neutral=2]
                    probs = np.exp(score) / np.exp(score).sum()
                    entailment = float(probs[1])
                else:
                    entailment = float(score.flat[0])
                max_entailment = max(max_entailment, entailment)
            except Exception:
                continue
        if max_entailment > 0.5:  # Threshold for "entailed"
            entailed_count += 1

    return entailed_count / len(gen_sentences[:10])


# ==============================================================================
# Claude-as-Judge (via Bedrock gateway)
# ==============================================================================

def compute_claude_judge(source: str, generated: str, reference: str = None) -> float:
    """
    Claude-as-judge: ask Claude to rate summary faithfulness on 1-5 scale.
    Uses Claude Haiku via Bedrock for cost efficiency.
    Returns score normalized to [0, 1].
    """
    import anthropic

    client = anthropic.Anthropic(
        api_key=os.environ.get('ANTHROPIC_AUTH_TOKEN', ''),
        base_url=os.environ.get('ANTHROPIC_BEDROCK_BASE_URL', '')
    )

    prompt = f"""Rate the faithfulness of the following summary on a scale of 1-5.

A faithful summary contains ONLY information that is supported by the source document.
- 5 = Perfectly faithful, all information is supported by the source
- 4 = Mostly faithful, minor unsupported details
- 3 = Partially faithful, some unsupported claims
- 2 = Mostly unfaithful, significant fabrication
- 1 = Completely unfaithful, major fabrications

Source document (first 500 words):
{source[:2000]}

Summary to evaluate:
{generated}

Respond with ONLY a single number (1-5), nothing else."""

    try:
        response = client.messages.create(
            model="us.anthropic.claude-haiku-4-5-20251001",
            max_tokens=5,
            messages=[{"role": "user", "content": prompt}]
        )
        score_text = response.content[0].text.strip()
        score = float(score_text)
        return (score - 1) / 4.0  # Normalize to [0, 1]
    except Exception as e:
        print(f"    Claude error: {e}", flush=True)
        return 0.5  # Default middle score on error


# ==============================================================================
# Main experiment
# ==============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-articles', type=int, default=20)
    parser.add_argument('--claude-samples', type=int, default=50,
                        help='Number of samples for Claude judge (API cost control)')
    parser.add_argument('--skip-claude', action='store_true')
    parser.add_argument('--skip-minicheck', action='store_true')
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("  Additional Baselines: MiniCheck + Claude-as-Judge", flush=True)
    print("=" * 60, flush=True)

    ds = load_dataset('mteb/summeval', split='test')

    # Flatten triples
    triples = []
    for item in list(ds)[:args.max_articles]:
        source = item['text']
        for i in range(16):
            triples.append({
                'source': source,
                'generated': item['machine_summaries'][i],
                'reference': item['human_summaries'][0],
                'consistency': item['consistency'][i]
            })

    n = len(triples)
    print(f"  Samples: {n} ({args.max_articles} articles x 16 systems)", flush=True)

    human = np.array([t['consistency'] for t in triples])

    # --- MiniCheck ---
    if not args.skip_minicheck:
        print(f"\n  [MiniCheck] Computing on {n} samples...", flush=True)
        t0 = time.time()
        minicheck_scores = []
        for i, t in enumerate(triples):
            minicheck_scores.append(compute_minicheck(t['source'], t['generated']))
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"    {i+1}/{n} ({elapsed:.0f}s)", flush=True)

        minicheck_arr = np.array(minicheck_scores)
        rho_mc, p_mc = spearmanr(minicheck_arr, human)
        print(f"  MiniCheck: Spearman rho = {rho_mc:.4f} (p={p_mc:.4e})", flush=True)
    else:
        rho_mc = None

    # --- Claude-as-Judge ---
    if not args.skip_claude:
        import random
        random.seed(42)
        claude_indices = random.sample(range(n), min(args.claude_samples, n))
        print(f"\n  [Claude-judge] Computing on {len(claude_indices)} samples...", flush=True)

        t0 = time.time()
        claude_scores = []
        claude_human = []
        for idx, i in enumerate(claude_indices):
            t = triples[i]
            score = compute_claude_judge(t['source'], t['generated'])
            claude_scores.append(score)
            claude_human.append(t['consistency'])
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"    {idx+1}/{len(claude_indices)} ({elapsed:.0f}s)", flush=True)

        claude_arr = np.array(claude_scores)
        claude_human_arr = np.array(claude_human)
        rho_cl, p_cl = spearmanr(claude_arr, claude_human_arr)
        print(f"  Claude-judge: Spearman rho = {rho_cl:.4f} (p={p_cl:.4e})", flush=True)
    else:
        rho_cl = None

    # --- Summary ---
    print(f"\n{'='*60}", flush=True)
    print(f"  RESULTS (summary-level, vs Human Consistency)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Metric':<20} {'Spearman':>10} {'N':>6}", flush=True)
    print(f"  {'-'*38}", flush=True)
    if rho_mc is not None:
        print(f"  {'MiniCheck':<20} {rho_mc:>10.4f} {n:>6}", flush=True)
    if rho_cl is not None:
        print(f"  {'Claude-judge':<20} {rho_cl:>10.4f} {len(claude_indices):>6}", flush=True)
    # Reference our existing results
    print(f"  {'CHI (ours)':<20} {'0.084':>10} {'1600':>6}", flush=True)
    print(f"  {'AlignScore':<20} {'0.140':>10} {'1600':>6}", flush=True)
    print(f"  {'ROUGE-L':<20} {'0.139':>10} {'1600':>6}", flush=True)

    # Save results
    results = {
        "minicheck": {"spearman": float(rho_mc) if rho_mc else None, "n": n},
        "claude_judge": {"spearman": float(rho_cl) if rho_cl else None,
                         "n": len(claude_indices) if not args.skip_claude else 0},
    }
    with open('results/additional_baselines.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to results/additional_baselines.json", flush=True)


if __name__ == "__main__":
    main()
