#!/usr/bin/env python3
"""
BARTScore Baseline on SummEval with Real Human Annotations.

BARTScore: log-probability of generating target text given source text
under a pre-trained BART model. Higher score = more faithful.

Uses facebook/bart-large-cnn (pre-trained on CNN/DailyMail summarization).
Direction: src→hyp (P(generated | source) — faithfulness direction).

Output: results/bartscore_summeval.json
"""

import json
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

sys.path.insert(0, str(Path(__file__).parent))


class BARTScorer:
    """BARTScore: log-probability scoring using a pre-trained BART model."""

    def __init__(self, checkpoint='facebook/bart-large-cnn', device='cpu'):
        import torch
        from transformers import BartTokenizer, BartForConditionalGeneration

        self.device = torch.device(device)
        self.tokenizer = BartTokenizer.from_pretrained(checkpoint)
        self.model = BartForConditionalGeneration.from_pretrained(checkpoint)
        self.model.eval()
        self.model.to(self.device)

    def score_pair(self, source: str, target: str, max_length: int = 1024) -> float:
        """Compute BARTScore for a single (source, target) pair.

        Returns negative NLL (higher = better).
        """
        import torch

        inputs = self.tokenizer(
            source, return_tensors='pt', max_length=max_length,
            truncation=True, padding=True
        ).to(self.device)

        labels = self.tokenizer(
            target, return_tensors='pt', max_length=max_length,
            truncation=True, padding=True
        ).to(self.device)

        with torch.no_grad():
            output = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                labels=labels['input_ids']
            )
        return -output.loss.item()

    def score_batch(self, sources: list, targets: list,
                    batch_size: int = 8, max_length: int = 1024) -> list:
        """Score multiple (source, target) pairs with batching."""
        import torch

        scores = []
        for i in range(0, len(sources), batch_size):
            batch_src = sources[i:i + batch_size]
            batch_tgt = targets[i:i + batch_size]

            inputs = self.tokenizer(
                batch_src, return_tensors='pt', max_length=max_length,
                truncation=True, padding=True
            ).to(self.device)

            labels = self.tokenizer(
                batch_tgt, return_tensors='pt', max_length=max_length,
                truncation=True, padding=True
            ).to(self.device)

            with torch.no_grad():
                for j in range(len(batch_src)):
                    inp_ids = inputs['input_ids'][j:j+1]
                    inp_mask = inputs['attention_mask'][j:j+1]
                    lab_ids = labels['input_ids'][j:j+1]

                    output = self.model(
                        input_ids=inp_ids,
                        attention_mask=inp_mask,
                        labels=lab_ids
                    )
                    scores.append(-output.loss.item())

        return scores


def load_summeval(max_articles: int = 100):
    """Load SummEval dataset and flatten into evaluation triples."""
    from datasets import load_dataset
    ds = load_dataset('mteb/summeval', split='test')

    triples = []
    for item in ds:
        source = item['text']
        reference = item['human_summaries'][0] if item['human_summaries'] else ""
        for i, (summary, consistency) in enumerate(
            zip(item['machine_summaries'], item['consistency'])
        ):
            triples.append({
                "source": source,
                "reference": reference,
                "generated": summary,
                "human_consistency": consistency,
                "human_coherence": item['coherence'][i],
                "human_relevance": item['relevance'][i],
                "human_fluency": item['fluency'][i],
                "article_id": item['id'],
                "summary_idx": i,
            })
        if len(triples) >= max_articles * 16:
            break

    print(f"  Loaded {len(triples)} evaluation triples from SummEval")
    return triples


def main():
    parser = argparse.ArgumentParser(
        description="BARTScore baseline evaluation on SummEval"
    )
    parser.add_argument('--max-articles', type=int, default=100)
    parser.add_argument('--output-dir', type=str, default='./results')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cpu',
                        help="Device: 'cpu' or 'cuda'")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  BARTScore Baseline Evaluation on SummEval")
    print("=" * 70)

    # Step 1: Load SummEval
    print("\n  [Step 1] Loading SummEval...")
    triples = load_summeval(args.max_articles)
    n = len(triples)

    # Step 2: Initialize BARTScorer
    print("\n  [Step 2] Loading BARTScore model (facebook/bart-large-cnn)...")
    t0 = time.time()
    scorer = BARTScorer(device=args.device)
    print(f"    Model loaded in {time.time()-t0:.1f}s")

    # Step 3: Compute BARTScore (src→hyp direction: faithfulness)
    print(f"\n  [Step 3] Computing BARTScore for {n} triples...")
    t0 = time.time()
    bartscore_values = []
    for i, t in enumerate(triples):
        score = scorer.score_pair(t["source"], t["generated"])
        bartscore_values.append(score)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"    {i+1}/{n} done ({rate:.1f} samples/s, ETA: {eta:.0f}s)")

    print(f"    BARTScore complete in {time.time()-t0:.1f}s")

    # Step 4: Compute correlations
    print(f"\n  [Step 4] Computing correlations with human judgments...")

    bart_arr = np.array(bartscore_values)

    dimensions = {
        "consistency": np.array([t["human_consistency"] for t in triples]),
        "coherence": np.array([t["human_coherence"] for t in triples]),
        "relevance": np.array([t["human_relevance"] for t in triples]),
        "fluency": np.array([t["human_fluency"] for t in triples]),
    }

    print(f"\n  {'='*60}")
    print(f"  RESULTS: BARTScore Correlations with Human Judgments")
    print(f"  N = {n} summaries (SummEval)")
    print(f"  {'='*60}")
    print(f"\n  {'Dimension':<15} {'Spearman':>10} {'Kendall':>10} {'p-value':>12}")
    print(f"  {'-'*49}")

    per_dimension = {}
    for dim_name, human_scores in dimensions.items():
        rho, p_rho = spearmanr(bart_arr, human_scores)
        tau, p_tau = kendalltau(bart_arr, human_scores)
        sig = "***" if p_rho < 0.001 else "**" if p_rho < 0.01 else "*" if p_rho < 0.05 else ""
        print(f"  {dim_name:<15} {rho:>8.4f}{sig:>2} {tau:>10.4f} {p_rho:>12.2e}")
        per_dimension[dim_name] = {
            "spearman": rho,
            "kendall": tau,
            "p_value": p_rho,
        }

    # Compute average correlation
    avg_human = np.mean([dimensions[d] for d in dimensions], axis=0)
    rho_avg, p_avg = spearmanr(bart_arr, avg_human)
    tau_avg, _ = kendalltau(bart_arr, avg_human)
    print(f"  {'average':<15} {rho_avg:>8.4f}    {tau_avg:>10.4f} {p_avg:>12.2e}")

    # Primary result: correlation with consistency (matches existing results format)
    rho_cons = per_dimension["consistency"]["spearman"]
    tau_cons = per_dimension["consistency"]["kendall"]
    p_cons = per_dimension["consistency"]["p_value"]

    results = {
        "BARTScore": {
            "spearman": rho_cons,
            "kendall": tau_cons,
            "p_value": p_cons,
        },
        "per_dimension": per_dimension,
        "average": {
            "spearman": rho_avg,
            "kendall": tau_avg,
            "p_value": p_avg,
        },
        "n": n,
        "model": "facebook/bart-large-cnn",
        "direction": "src_to_hyp",
        "score_stats": {
            "mean": float(np.mean(bart_arr)),
            "std": float(np.std(bart_arr)),
            "min": float(np.min(bart_arr)),
            "max": float(np.max(bart_arr)),
        },
    }

    # Save
    save_file = output_dir / "bartscore_summeval.json"
    with open(save_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {save_file}")

    print(f"\n  KEY FINDING:")
    print(f"    BARTScore achieves Spearman rho={rho_cons:.4f} on SummEval consistency")
    print(f"    (for comparison: MiniCheck=0.380, RHI*=0.136, CHI=0.114)")


if __name__ == "__main__":
    main()
