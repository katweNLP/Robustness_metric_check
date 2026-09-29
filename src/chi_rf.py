"""
CHI-RF: Reference-Free Composite Hallucination Index.

Variant of CHI that operates without a human reference summary.
Uses only Input (source document) and Generated (model output).

Three sub-indices:
  EHI_RF — entity-level, 3 factors (EF_rf, NH_rf, 1-Coverage)
  RHI_RF — relation-level, 2 factors (EF_rf, NH_rf)
  QHI_RF — quantity-level, 2 factors (QEF_rf, QNH_rf)

Optional NLI enrichment reclassifies surface-level "fabricated" entities
that are actually paraphrased and entailed by the source.
"""

import math
from typing import Dict, Optional

from .ehi import extract_entities
from .rhi_star import (extract_relations, _to_tuples,
                       _batch_encode_verbs, _intersect_2)
from .qhi import extract_quantities, quantities_match


def _harmonic_mean_2(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return 0.0
    return 2 / (1/a + 1/b)


def _harmonic_mean_3(a: float, b: float, c: float) -> float:
    if a <= 0 or b <= 0 or c <= 0:
        return 0.0
    return 3 / (1/a + 1/b + 1/c)


# ---------------------------------------------------------------------------
# EHI-RF: Entity Hallucination Index — Reference Free
# ---------------------------------------------------------------------------

def compute_ehi_rf_factors(input_text: str, generated_text: str) -> Dict:
    """
    Compute reference-free entity factors from I and G only.

    Factors:
        EF_rf:       |I ∩ G| / (|I| + |G|) — source-grounded entities
        NH_rf:       |G - I| / |G|          — fabricated entities
        1-Coverage:  1 - |I ∩ G| / |I|      — source content missed
    """
    I = extract_entities(input_text)
    G = extract_entities(generated_text)

    len_I = len(I)
    len_G = len(G)

    I_G = I & G

    ef_rf = len(I_G) / (len_I + len_G) if (len_I + len_G) > 0 else 0
    nh_rf = (len_G - len(I_G)) / len_G if len_G > 0 else 0
    coverage = len(I_G) / len_I if len_I > 0 else 0

    return {
        "ef_rf": ef_rf,
        "nh_rf": nh_rf,
        "coverage": coverage,
        "len_I": len_I,
        "len_G": len_G,
        "len_I_G": len(I_G),
        "fabricated_entities": G - I,
        "grounded_entities": I_G,
    }


def compute_ehi_rf(factors: Dict) -> float:
    """
    EHI_RF = e^EF_rf / (e^EF_rf + e^NH_rf + e^(1-Coverage))

    Output: [0, 1]. At equilibrium (all equal): 1/3 ≈ 0.333.
    """
    ef_rf = factors["ef_rf"]
    nh_rf = factors["nh_rf"]
    missed = 1 - factors["coverage"]

    numerator = math.exp(ef_rf)
    denominator = math.exp(ef_rf) + math.exp(nh_rf) + math.exp(missed)
    return numerator / denominator


# ---------------------------------------------------------------------------
# RHI-RF: Relation Hallucination Index — Reference Free
# ---------------------------------------------------------------------------

def compute_rhi_rf_factors(input_text: str, generated_text: str,
                           confidence_threshold: float = 0.4) -> Dict:
    """
    Compute reference-free relation factors from I and G SVO triples.

    Factors:
        EF_rf:  partial_intersect(I, G) / |G| — source-grounded relations
        NH_rf:  (|G| - matched) / |G|          — fabricated relations
    """
    inp_rels = extract_relations(input_text, confidence_threshold)
    gen_rels = extract_relations(generated_text, confidence_threshold)

    inp_t = _to_tuples(inp_rels)
    gen_t = _to_tuples(gen_rels)

    len_I = len(inp_t)
    len_G = len(gen_t)

    if len_G == 0:
        return {"ef_rf": 0, "nh_rf": 0, "len_I": len_I, "len_G": 0}

    verb_embs = _batch_encode_verbs(inp_t, gen_t)
    I_G = _intersect_2(inp_t, gen_t, verb_embs)

    ef_rf = I_G / len_G if len_G > 0 else 0
    nh_rf = max(0, (len_G - I_G)) / len_G if len_G > 0 else 0

    return {
        "ef_rf": ef_rf,
        "nh_rf": nh_rf,
        "len_I": len_I,
        "len_G": len_G,
        "I_G_partial": I_G,
    }


def compute_rhi_rf(factors: Dict) -> float:
    """
    RHI_RF = e^EF_rf / (e^EF_rf + e^NH_rf)

    Output: [0, 1]. At equilibrium: 0.5.
    """
    ef_rf = factors["ef_rf"]
    nh_rf = factors["nh_rf"]

    numerator = math.exp(ef_rf)
    denominator = math.exp(ef_rf) + math.exp(nh_rf)
    return numerator / denominator


# ---------------------------------------------------------------------------
# QHI-RF: Quantity Hallucination Index — Reference Free
# ---------------------------------------------------------------------------

def compute_qhi_rf_factors(input_text: str, generated_text: str,
                           epsilon: float = 0.05) -> Dict:
    """
    Compute reference-free quantity factors from I and G only.

    Factors:
        QEF_rf:  |I_quants matched in G| / (|I| + |G|) — source-grounded
        QNH_rf:  |G_quants not in I| / |G|              — fabricated numbers
    """
    I_quants = extract_quantities(input_text)
    G_quants = extract_quantities(generated_text)

    len_I = len(I_quants)
    len_G = len(G_quants)

    if len_G == 0 and len_I == 0:
        return {"qef_rf": 0, "qnh_rf": 0,
                "len_I": 0, "len_G": 0}

    g_matched = set()
    for k, qg in enumerate(G_quants):
        for qi in I_quants:
            if quantities_match(qg, qi, epsilon):
                g_matched.add(k)
                break

    qef_rf = len(g_matched) / (len_I + len_G) if (len_I + len_G) > 0 else 0
    qnh_rf = (len_G - len(g_matched)) / len_G if len_G > 0 else 0

    return {
        "qef_rf": qef_rf,
        "qnh_rf": qnh_rf,
        "len_I": len_I,
        "len_G": len_G,
        "n_matched": len(g_matched),
    }


def compute_qhi_rf(factors: Dict) -> float:
    """
    QHI_RF = e^QEF_rf / (e^QEF_rf + e^QNH_rf)

    Output: [0, 1]. At equilibrium: 0.5.
    """
    qef_rf = factors["qef_rf"]
    qnh_rf = factors["qnh_rf"]

    numerator = math.exp(qef_rf)
    denominator = math.exp(qef_rf) + math.exp(qnh_rf)
    return numerator / denominator


# ---------------------------------------------------------------------------
# CHI-RF: Composite — Reference Free
# ---------------------------------------------------------------------------

def compute_chi_rf(ehi_rf: float, rhi_rf: float, qhi_rf: float,
                   has_entities: bool = True,
                   has_quantities: bool = True) -> float:
    """Adaptive harmonic mean of reference-free sub-indices."""
    if not has_entities and not has_quantities:
        return rhi_rf
    elif not has_quantities:
        return _harmonic_mean_2(ehi_rf, rhi_rf)
    elif not has_entities:
        return _harmonic_mean_2(qhi_rf, rhi_rf)
    else:
        return _harmonic_mean_3(ehi_rf, rhi_rf, qhi_rf)


def score_text_rf(input_text: str, generated_text: str,
                  confidence_threshold: float = 0.4,
                  epsilon: float = 0.05) -> Dict:
    """
    End-to-end reference-free CHI scoring.

    Only needs source document + generated text. No human reference.
    """
    ehi_factors = compute_ehi_rf_factors(input_text, generated_text)
    ehi_rf = compute_ehi_rf(ehi_factors)

    rhi_factors = compute_rhi_rf_factors(input_text, generated_text,
                                         confidence_threshold)
    rhi_rf = compute_rhi_rf(rhi_factors)

    qhi_factors = compute_qhi_rf_factors(input_text, generated_text, epsilon)
    qhi_rf = compute_qhi_rf(qhi_factors)

    has_entities = ehi_factors["len_G"] > 0
    has_quantities = qhi_factors["len_G"] > 0

    chi_rf = compute_chi_rf(ehi_rf, rhi_rf, qhi_rf,
                            has_entities, has_quantities)

    return {
        "chi_rf": chi_rf,
        "ehi_rf": ehi_rf,
        "rhi_rf": rhi_rf,
        "qhi_rf": qhi_rf,
        "dimensions_used": sum([True, has_entities, has_quantities]),
        "has_entities": has_entities,
        "has_quantities": has_quantities,
        "factors": {
            "ehi_rf": {k: v for k, v in ehi_factors.items()
                       if k not in ("fabricated_entities", "grounded_entities")},
            "rhi_rf": dict(rhi_factors),
            "qhi_rf": dict(qhi_factors),
        }
    }


# ---------------------------------------------------------------------------
# NLI-Enriched variant
# ---------------------------------------------------------------------------

_nli_model = None


def _get_nli_model():
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-base')
    return _nli_model


def _nli_entailment(source_text: str, claim: str) -> float:
    """P(source entails claim) using NLI cross-encoder."""
    import numpy as np
    model = _get_nli_model()
    raw = model.predict([(source_text[:512], claim[:512])])
    raw = np.atleast_1d(np.squeeze(np.array(raw)))
    if raw.shape[-1] == 3:
        probs = np.exp(raw) / np.exp(raw).sum()
        return float(probs[1])
    return float(raw.flat[0])


def _nli_reclassify_entities(input_text: str, generated_text: str,
                             fabricated_entities: set,
                             threshold: float = 0.5) -> Dict:
    """
    For entities flagged as fabricated (not in source by surface match),
    check via NLI whether the source actually entails a statement
    containing that entity. Reclassify entailed ones as grounded.
    """
    import nltk
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

    source_sents = nltk.sent_tokenize(input_text)[:15]
    gen_sents = nltk.sent_tokenize(generated_text)

    reclassified = set()
    still_fabricated = set()

    for entity in fabricated_entities:
        entity_sents = [s for s in gen_sents if entity.lower() in s.lower()]
        if not entity_sents:
            still_fabricated.add(entity)
            continue

        best_entailment = 0.0
        claim = entity_sents[0]
        for src_sent in source_sents:
            score = _nli_entailment(src_sent, claim)
            best_entailment = max(best_entailment, score)
            if best_entailment > threshold:
                break

        if best_entailment > threshold:
            reclassified.add(entity)
        else:
            still_fabricated.add(entity)

    return {
        "reclassified": reclassified,
        "still_fabricated": still_fabricated,
        "n_reclassified": len(reclassified),
        "n_still_fabricated": len(still_fabricated),
    }


def score_text_rf_nli(input_text: str, generated_text: str,
                      confidence_threshold: float = 0.4,
                      epsilon: float = 0.05,
                      nli_threshold: float = 0.5) -> Dict:
    """
    Reference-free CHI with NLI enrichment.

    Step 1: Compute CHI-RF (surface matching, no reference).
    Step 2: For fabricated entities, run NLI verification.
    Step 3: Reclassify entailed entities → recalculate EHI_RF.
    """
    ehi_factors = compute_ehi_rf_factors(input_text, generated_text)
    fabricated = ehi_factors.get("fabricated_entities", set())

    nli_result = {"n_reclassified": 0, "n_still_fabricated": len(fabricated)}
    ehi_rf_nli = compute_ehi_rf(ehi_factors)

    if fabricated:
        nli_result = _nli_reclassify_entities(
            input_text, generated_text, fabricated, nli_threshold)

        n_reclassified = nli_result["n_reclassified"]
        if n_reclassified > 0:
            len_I = ehi_factors["len_I"]
            len_G = ehi_factors["len_G"]
            len_I_G_new = ehi_factors["len_I_G"] + n_reclassified

            ef_rf_new = len_I_G_new / (len_I + len_G) if (len_I + len_G) > 0 else 0
            nh_rf_new = (len_G - len_I_G_new) / len_G if len_G > 0 else 0
            coverage_new = len_I_G_new / len_I if len_I > 0 else 0

            ehi_factors_nli = {
                "ef_rf": ef_rf_new,
                "nh_rf": max(0, nh_rf_new),
                "coverage": min(1, coverage_new),
            }
            ehi_rf_nli = compute_ehi_rf(ehi_factors_nli)

    rhi_factors = compute_rhi_rf_factors(input_text, generated_text,
                                         confidence_threshold)
    rhi_rf = compute_rhi_rf(rhi_factors)

    qhi_factors = compute_qhi_rf_factors(input_text, generated_text, epsilon)
    qhi_rf = compute_qhi_rf(qhi_factors)

    has_entities = ehi_factors["len_G"] > 0
    has_quantities = qhi_factors["len_G"] > 0

    chi_rf_nli = compute_chi_rf(ehi_rf_nli, rhi_rf, qhi_rf,
                                has_entities, has_quantities)

    return {
        "chi_rf_nli": chi_rf_nli,
        "ehi_rf_nli": ehi_rf_nli,
        "ehi_rf_base": compute_ehi_rf(ehi_factors),
        "rhi_rf": rhi_rf,
        "qhi_rf": qhi_rf,
        "nli_reclassified": nli_result["n_reclassified"],
        "nli_still_fabricated": nli_result["n_still_fabricated"],
        "dimensions_used": sum([True, has_entities, has_quantities]),
    }
