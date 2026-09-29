#!/usr/bin/env python3
"""
Compute system-level correlations for BARTScore and MiniCheck on SummEval.

System-level = aggregate scores per model system (16 systems in SummEval),
then correlate system means with human dimension means.

Adds results to summeval_system_level_full.json and regenerates the HTML report.
"""

import json
import sys
import time
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

sys.path.insert(0, str(Path(__file__).parent))


def load_summeval_with_systems(max_articles=100):
    from datasets import load_dataset
    ds = load_dataset('mteb/summeval', split='test')

    triples = []
    for item in ds:
        source = item['text']
        reference = item['human_summaries'][0] if item['human_summaries'] else ""
        for i, (summary, cons, coh, rel, flu) in enumerate(
            zip(item['machine_summaries'], item['consistency'],
                item['coherence'], item['relevance'], item['fluency'])):
            triples.append({
                "source": source, "reference": reference, "generated": summary,
                "system_idx": i,
                "human_consistency": cons, "human_coherence": coh,
                "human_relevance": rel, "human_fluency": flu,
            })
        if len(triples) >= max_articles * 16:
            break
    print(f"  Loaded {len(triples)} triples ({len(set(t['system_idx'] for t in triples))} systems)")
    return triples


def compute_system_level(per_sample_scores, triples, metric_name):
    n_systems = 16
    system_metric = {i: [] for i in range(n_systems)}
    system_human = {i: {"consistency": [], "coherence": [], "relevance": []} for i in range(n_systems)}

    for score, t in zip(per_sample_scores, triples):
        si = t["system_idx"]
        system_metric[si].append(score)
        system_human[si]["consistency"].append(t["human_consistency"])
        system_human[si]["coherence"].append(t["human_coherence"])
        system_human[si]["relevance"].append(t["human_relevance"])

    metric_means = [np.mean(system_metric[i]) for i in range(n_systems)]
    results = {}
    for dim in ["consistency", "coherence", "relevance"]:
        human_means = [np.mean(system_human[i][dim]) for i in range(n_systems)]
        rho, p = spearmanr(metric_means, human_means)
        tau, _ = kendalltau(metric_means, human_means)
        results[dim] = {"spearman": round(rho, 4), "kendall": round(tau, 4), "p": p}

    avg_human = [np.mean([np.mean(system_human[i][d]) for d in ["consistency", "coherence", "relevance"]]) for i in range(n_systems)]
    rho, p = spearmanr(metric_means, avg_human)
    tau, _ = kendalltau(metric_means, avg_human)
    results["average"] = {"spearman": round(rho, 4), "kendall": round(tau, 4), "p": p}

    return results


def main():
    output_dir = Path("./results")

    print("=" * 70)
    print("  System-Level Baselines: BARTScore + MiniCheck on SummEval")
    print("=" * 70)

    print("\n  [Step 1] Loading SummEval...")
    triples = load_summeval_with_systems()
    n = len(triples)

    # BARTScore
    print("\n  [Step 2] Computing BARTScore (system-level)...")
    from run_bartscore_baseline import BARTScorer
    import torch
    scorer = BARTScorer(device="cpu")
    t0 = time.time()
    bart_scores = []
    for i, t in enumerate(triples):
        score = scorer.score_pair(t["source"], t["generated"])
        bart_scores.append(score)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"    {i+1}/{n} ({rate:.1f}/s, ETA: {eta:.0f}s)", flush=True)
    print(f"    BARTScore done in {time.time()-t0:.0f}s")

    bart_system = compute_system_level(bart_scores, triples, "BARTScore")
    print(f"    System-level: avg spearman = {bart_system['average']['spearman']}")

    # MiniCheck (NLI-based)
    print("\n  [Step 3] Computing MiniCheck (system-level)...")
    from sentence_transformers import CrossEncoder
    import nltk
    nltk.download('punkt_tab', quiet=True)
    nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')

    t0 = time.time()
    mini_scores = []
    for i, t in enumerate(triples):
        source_sents = nltk.sent_tokenize(t["source"])[:10]
        gen_sents = nltk.sent_tokenize(t["generated"])
        if not gen_sents or not source_sents:
            mini_scores.append(0.5)
            continue
        claim_scores = []
        for claim in gen_sents:
            pairs = [(claim, src) for src in source_sents]
            preds = nli_model.predict(pairs)
            entail_probs = [p[0] if len(p) > 2 else max(p) for p in (preds if len(preds.shape) > 1 else [preds])]
            claim_scores.append(max(entail_probs))
        mini_scores.append(float(np.mean(claim_scores)))
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"    {i+1}/{n} ({rate:.1f}/s, ETA: {eta:.0f}s)", flush=True)
    print(f"    MiniCheck done in {time.time()-t0:.0f}s")

    mini_system = compute_system_level(mini_scores, triples, "MiniCheck")
    print(f"    System-level: avg spearman = {mini_system['average']['spearman']}")

    # Update system-level results
    print("\n  [Step 4] Updating system-level results...")
    sl_file = output_dir / "summeval_system_level_full.json"
    with open(sl_file) as f:
        sl = json.load(f)

    for dim in ["consistency", "coherence", "relevance", "average"]:
        sl[dim]["BARTScore"] = bart_system[dim]
        sl[dim]["MiniCheck"] = mini_system[dim]

    with open(sl_file, 'w') as f:
        json.dump(sl, f, indent=2)
    print(f"    Updated: {sl_file}")

    # Print summary
    print(f"\n  {'=' * 60}")
    print(f"  SYSTEM-LEVEL RESULTS (added)")
    print(f"  {'=' * 60}")
    for dim in ["consistency", "coherence", "relevance", "average"]:
        print(f"\n  {dim.title()}:")
        for m in sorted(sl[dim].keys(), key=lambda x: -sl[dim][x].get("spearman", 0)):
            sp = sl[dim][m]["spearman"]
            print(f"    {m:20s} {sp:.4f}")

    # Regenerate HTML
    print("\n  [Step 5] Regenerating HTML report...")
    import subprocess
    subprocess.run([sys.executable, "generate_html_report.py"], check=True)

    print("\n  DONE! System-level BARTScore + MiniCheck added to report.")


if __name__ == "__main__":
    main()
