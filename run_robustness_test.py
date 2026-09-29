#!/usr/bin/env python3
"""
Robustness Test: Entity-Swap and Relation-Inversion Perturbation.

Takes high-quality SummEval summaries and progressively perturbs them.
A good faithfulness metric should show monotonic score drops as
perturbation intensity increases.

Two perturbation types:
    1. Entity swap  — replace entities with type-matched out-of-source entities
    2. Relation inversion — swap verbs with antonyms
"""

import json
import sys
import time
import random
import argparse
import numpy as np
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent))

# ---------------------------------------------------------------------------
# Type-aware entity replacement pool
# ---------------------------------------------------------------------------

ENTITY_POOL = {
    "PERSON": ["John Smith", "Maria Garcia", "James Wilson", "Sarah Chen",
               "Robert Taylor", "Anna Schmidt", "David Kim", "Elena Rodriguez"],
    "ORG": ["Microsoft", "United Nations", "Goldman Sachs", "World Bank",
            "Tesla", "Red Cross", "NATO", "Spotify"],
    "GPE": ["Tokyo", "Paris", "Brazil", "Sydney", "Moscow", "Cairo",
            "Toronto", "Berlin"],
    "NORP": ["Republicans", "Democrats", "Europeans", "Catholics"],
    "LOC": ["Pacific Ocean", "Mount Everest", "Sahara Desert", "Amazon River"],
    "MONEY": ["$500 million", "$2.3 billion", "€1.5 million", "$750,000"],
    "DATE": ["January 2020", "March 2019", "last September", "2017"],
    "TIME": ["3 p.m.", "midnight", "noon", "8:30 a.m."],
    "CARDINAL": ["47", "1,200", "three", "85"],
    "ORDINAL": ["first", "third", "seventh", "tenth"],
    "PERCENT": ["15%", "42 percent", "7.5%", "90%"],
    "EVENT": ["World Cup", "Olympics", "Hurricane Sandy", "Davos Forum"],
    "FAC": ["White House", "Eiffel Tower", "Pentagon", "Wall Street"],
    "PRODUCT": ["iPhone", "Model S", "Boeing 737", "Surface Pro"],
    "WORK_OF_ART": ["Mona Lisa", "Hamlet", "The Matrix", "Bohemian Rhapsody"],
}

INVERSIONS = {
    'increased': 'decreased', 'decreased': 'increased',
    'increases': 'decreases', 'decreases': 'increases',
    'increasing': 'decreasing', 'decreasing': 'increasing',
    'grew': 'fell', 'fell': 'grew',
    'grows': 'falls', 'falls': 'grows',
    'rose': 'dropped', 'dropped': 'rose',
    'rises': 'drops', 'drops': 'rises',
    'approved': 'rejected', 'rejected': 'approved',
    'approves': 'rejects', 'rejects': 'approves',
    'won': 'lost', 'lost': 'won',
    'wins': 'loses', 'loses': 'wins',
    'winning': 'losing', 'losing': 'winning',
    'acquired': 'sold', 'sold': 'acquired',
    'supported': 'opposed', 'opposed': 'supported',
    'supports': 'opposes', 'opposes': 'supports',
    'praised': 'criticized', 'criticized': 'praised',
    'praises': 'criticizes', 'criticizes': 'praises',
    'accepted': 'denied', 'denied': 'accepted',
    'accepts': 'denies', 'denies': 'accepts',
    'improved': 'worsened', 'worsened': 'improved',
    'raised': 'lowered', 'lowered': 'raised',
    'expanded': 'contracted', 'contracted': 'expanded',
    'gained': 'lost', 'succeeded': 'failed', 'failed': 'succeeded',
    'opened': 'closed', 'closed': 'opened',
    'started': 'stopped', 'stopped': 'started',
    'said': 'denied', 'told': 'denied', 'announced': 'retracted',
    'confirmed': 'denied', 'agreed': 'disagreed', 'disagreed': 'agreed',
    'allowed': 'banned', 'banned': 'allowed',
    'added': 'removed', 'removed': 'added',
    'joined': 'left', 'left': 'joined',
    'helped': 'hindered', 'killed': 'saved', 'saved': 'killed',
    'arrested': 'released', 'released': 'arrested',
    'found': 'lost', 'recovered': 'lost',
    'attacked': 'defended', 'defended': 'attacked',
    'built': 'destroyed', 'destroyed': 'built',
    'elected': 'defeated', 'defeated': 'elected',
    'hired': 'fired', 'fired': 'hired',
    'signed': 'vetoed', 'vetoed': 'signed',
    'married': 'divorced', 'divorced': 'married',
    'entered': 'exited', 'exited': 'entered',
    'revealed': 'concealed', 'concealed': 'revealed',
    'promoted': 'demoted', 'demoted': 'promoted',
    'included': 'excluded', 'excluded': 'included',
    'convicted': 'acquitted', 'acquitted': 'convicted',
    'praised': 'condemned', 'condemned': 'praised',
    'launched': 'cancelled', 'cancelled': 'launched',
    'allowed': 'prohibited', 'prohibited': 'allowed',
    'passed': 'blocked', 'blocked': 'passed',
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_summeval_high_quality(max_samples: int = 200):
    """Load SummEval triples with human_consistency >= 4."""
    from datasets import load_dataset
    ds = load_dataset('mteb/summeval', split='test')

    triples = []
    for item in ds:
        source = item['text']
        reference = item['human_summaries'][0] if item['human_summaries'] else ""
        for i, (summary, consistency) in enumerate(zip(
                item['machine_summaries'], item['consistency'])):
            if consistency >= 4.0:
                triples.append({
                    "source": source,
                    "reference": reference,
                    "generated": summary,
                    "human_consistency": consistency,
                })
            if len(triples) >= max_samples:
                break
        if len(triples) >= max_samples:
            break

    print(f"  Loaded {len(triples)} high-quality SummEval triples (consistency >= 4)")
    return triples


# ---------------------------------------------------------------------------
# Perturbation functions
# ---------------------------------------------------------------------------

def entity_swap_perturbation(source: str, generated: str, swap_ratio: float, nlp) -> str:
    """Replace swap_ratio fraction of entities in generated with type-matched randoms."""
    if swap_ratio <= 0:
        return generated

    doc_gen = nlp(generated)
    doc_src = nlp(source)

    source_entities = {ent.text.lower() for ent in doc_src.ents}

    gen_ents = list(doc_gen.ents)
    if not gen_ents:
        return generated

    random.shuffle(gen_ents)
    n_to_swap = max(1, int(len(gen_ents) * swap_ratio))
    ents_to_swap = gen_ents[:n_to_swap]

    perturbed = generated
    for ent in sorted(ents_to_swap, key=lambda e: e.start_char, reverse=True):
        label = ent.label_
        pool = ENTITY_POOL.get(label, ENTITY_POOL.get("PERSON", []))
        candidates = [e for e in pool if e.lower() not in source_entities
                       and e.lower() != ent.text.lower()]
        if not candidates:
            candidates = pool

        if candidates:
            replacement = random.choice(candidates)
            perturbed = perturbed[:ent.start_char] + replacement + perturbed[ent.end_char:]

    return perturbed


def relation_inversion_perturbation(generated: str, inversion_ratio: float, nlp=None) -> str:
    """Invert relations by (1) antonym replacement or (2) negation insertion.

    Two strategies applied in order:
    - Dictionary: replace verbs with antonyms from INVERSIONS
    - Negation: for verbs without antonyms, insert 'not' before the verb

    This ensures every summary with verbs gets perturbed, not just the ~10%
    that happen to contain dictionary entries.
    """
    if inversion_ratio <= 0:
        return generated

    if nlp is None:
        import spacy
        nlp = spacy.load("en_core_web_sm")

    doc = nlp(generated)
    words = generated.split()

    invertible = []
    negatable = []
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ in ("ROOT", "relcl", "advcl", "ccomp", "xcomp", "conj"):
            w_lower = token.text.lower().rstrip('.,;:!?')
            word_idx = None
            for i, w in enumerate(words):
                if w.lower().rstrip('.,;:!?') == w_lower and i not in [x[0] for x in invertible + negatable]:
                    word_idx = i
                    break
            if word_idx is not None:
                if w_lower in INVERSIONS:
                    invertible.append((word_idx, "antonym"))
                else:
                    negatable.append((word_idx, "negate"))

    all_targets = invertible + negatable
    if not all_targets:
        return generated

    n_to_modify = max(1, int(len(all_targets) * inversion_ratio))
    random.shuffle(all_targets)
    to_modify = all_targets[:n_to_modify]

    offset = 0
    to_modify.sort(key=lambda x: x[0])
    for idx, strategy in to_modify:
        real_idx = idx + offset
        if real_idx >= len(words):
            continue
        w = words[real_idx]
        w_lower = w.lower().rstrip('.,;:!?')
        if strategy == "antonym" and w_lower in INVERSIONS:
            suffix = w[len(w_lower):]
            replacement = INVERSIONS[w_lower]
            if w[0].isupper():
                replacement = replacement.capitalize()
            words[real_idx] = replacement + suffix
        elif strategy == "negate":
            words.insert(real_idx, "not")
            offset += 1

    return ' '.join(words)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_chi_score(source, reference, generated):
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


def compute_rouge_l(reference, generated):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(reference, generated)['rougeL'].fmeasure


def compute_bertscore_batch(references, generateds):
    from bert_score import score as bert_score_fn
    P, R, F1 = bert_score_fn(generateds, references, model_type="roberta-large",
                              lang="en", verbose=False)
    return F1.numpy().tolist()


def compute_all_metrics_for_batch(triples):
    """Compute all metrics for a batch of (source, reference, generated) triples."""
    chi_scores = {"CHI": [], "EHI": [], "RHI*": [], "QHI": []}
    rouge_scores = []

    for t in triples:
        try:
            cs = compute_chi_score(t["source"], t["reference"], t["generated"])
        except Exception:
            cs = {"chi": 0.4, "ehi": 0.4, "rhi_star": 0.333, "qhi": 0.4}
        chi_scores["CHI"].append(cs["chi"])
        chi_scores["EHI"].append(cs["ehi"])
        chi_scores["RHI*"].append(cs["rhi_star"])
        chi_scores["QHI"].append(cs["qhi"])
        rouge_scores.append(compute_rouge_l(t["reference"], t["generated"]))

    refs = [t["reference"] for t in triples]
    gens = [t["generated"] for t in triples]
    bert_f1s = compute_bertscore_batch(refs, gens)

    return {
        "CHI": chi_scores["CHI"],
        "EHI": chi_scores["EHI"],
        "RHI*": chi_scores["RHI*"],
        "QHI": chi_scores["QHI"],
        "ROUGE-L": rouge_scores,
        "BERTScore": bert_f1s,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Robustness perturbation test")
    parser.add_argument('--max-samples', type=int, default=200)
    parser.add_argument('--output-dir', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import spacy
    nlp = spacy.load("en_core_web_sm")

    print("=" * 70)
    print("  Robustness Test: Entity-Swap & Relation-Inversion Perturbation")
    print("=" * 70)

    # Load
    print("\n  [Step 1] Loading high-quality SummEval triples...")
    triples = load_summeval_high_quality(args.max_samples)
    n = len(triples)

    # -----------------------------------------------------------------------
    # Entity-swap experiment
    # -----------------------------------------------------------------------
    swap_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    entity_swap_results = {"swap_ratios": swap_ratios, "metrics": {}}

    print(f"\n  [Step 2] Entity-swap perturbation ({n} samples x {len(swap_ratios)} levels)...")
    for ratio in swap_ratios:
        print(f"\n    --- Swap ratio: {ratio:.0%} ---")
        t0 = time.time()

        perturbed_triples = []
        for i, t in enumerate(triples):
            perturbed_gen = entity_swap_perturbation(t["source"], t["generated"], ratio, nlp)
            perturbed_triples.append({
                "source": t["source"],
                "reference": t["reference"],
                "generated": perturbed_gen,
            })
            if (i + 1) % 50 == 0:
                print(f"      Perturbed {i+1}/{n}")

        print(f"      Computing metrics...")
        batch_metrics = compute_all_metrics_for_batch(perturbed_triples)

        for metric_name, scores in batch_metrics.items():
            if metric_name not in entity_swap_results["metrics"]:
                entity_swap_results["metrics"][metric_name] = {"scores": []}
            entity_swap_results["metrics"][metric_name]["scores"].append(
                float(np.mean(scores))
            )

        elapsed = time.time() - t0
        print(f"      Done in {elapsed:.1f}s")

    # Compute deltas
    for metric_name, data in entity_swap_results["metrics"].items():
        scores = data["scores"]
        data["delta"] = scores[-1] - scores[0] if len(scores) > 1 else 0.0

    # -----------------------------------------------------------------------
    # Relation-inversion experiment
    # -----------------------------------------------------------------------
    inv_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    relation_inv_results = {"inversion_ratios": inv_ratios, "metrics": {}}

    print(f"\n  [Step 3] Relation-inversion perturbation ({n} samples x {len(inv_ratios)} levels)...")
    for ratio in inv_ratios:
        print(f"\n    --- Inversion ratio: {ratio:.0%} ---")
        t0 = time.time()

        perturbed_triples = []
        for i, t in enumerate(triples):
            perturbed_gen = relation_inversion_perturbation(t["generated"], ratio, nlp=nlp)
            perturbed_triples.append({
                "source": t["source"],
                "reference": t["reference"],
                "generated": perturbed_gen,
            })

        print(f"      Computing metrics...")
        batch_metrics = compute_all_metrics_for_batch(perturbed_triples)

        for metric_name, scores in batch_metrics.items():
            if metric_name not in relation_inv_results["metrics"]:
                relation_inv_results["metrics"][metric_name] = {"scores": []}
            relation_inv_results["metrics"][metric_name]["scores"].append(
                float(np.mean(scores))
            )

        elapsed = time.time() - t0
        print(f"      Done in {elapsed:.1f}s")

    for metric_name, data in relation_inv_results["metrics"].items():
        scores = data["scores"]
        data["delta"] = scores[-1] - scores[0] if len(scores) > 1 else 0.0

    # -----------------------------------------------------------------------
    # Sensitivity ranking (by absolute delta, entity-swap)
    # -----------------------------------------------------------------------
    sensitivity_ranking = sorted(
        entity_swap_results["metrics"].keys(),
        key=lambda m: abs(entity_swap_results["metrics"][m]["delta"]),
        reverse=True
    )

    # -----------------------------------------------------------------------
    # Save & report
    # -----------------------------------------------------------------------
    output = {
        "entity_swap": entity_swap_results,
        "relation_inversion": relation_inv_results,
        "n_samples": n,
        "sensitivity_ranking": sensitivity_ranking,
        "seed": args.seed,
    }

    save_file = output_dir / "robustness_entity_swap.json"
    with open(save_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {save_file}")

    print(f"\n  {'='*70}")
    print(f"  ENTITY-SWAP SENSITIVITY RANKING (|delta| from 0% to 100%):")
    print(f"  {'='*70}")
    print(f"\n  {'Metric':<12} {'Score@0%':>10} {'Score@100%':>10} {'Delta':>10}")
    print(f"  {'-'*44}")
    for m in sensitivity_ranking:
        data = entity_swap_results["metrics"][m]
        s0 = data["scores"][0]
        s100 = data["scores"][-1]
        delta = data["delta"]
        print(f"  {m:<12} {s0:>10.4f} {s100:>10.4f} {delta:>+10.4f}")

    print(f"\n  RELATION-INVERSION SENSITIVITY:")
    print(f"\n  {'Metric':<12} {'Score@0%':>10} {'Score@100%':>10} {'Delta':>10}")
    print(f"  {'-'*44}")
    for m in sensitivity_ranking:
        data = relation_inv_results["metrics"][m]
        s0 = data["scores"][0]
        s100 = data["scores"][-1]
        delta = data["delta"]
        print(f"  {m:<12} {s0:>10.4f} {s100:>10.4f} {delta:>+10.4f}")


if __name__ == "__main__":
    main()
