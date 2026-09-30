"""
RHI Core: Relation Hallucination Index computation.

Extracted and modularized from the published RHI_Metric implementation
(Katwe et al., ACM FIRE 2024). Uses spaCy for SVO extraction and
sentence-transformers for semantic matching.

Six factors from SVO triple Venn diagram (I, R, G):
  Rewarded: EF (Extractiveness), PH (Positive Hallucination)
  Penalized: OF (Over Focus), NH (Negative Hallucination),
             LH (Lost Hallucination), LF (Lost Focus)

Formula: RHI = 1 + (EF * PH) / 2 - (OF + NH + LH + LF) / 4
"""

import math
from decimal import Decimal
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer, util

_nlp = None
_st_model = None

UNWANTED_SYMBOLS = {'.', ',', '(', ')', '{', '}', '[', ']', '%', '$', ';', ':', '&', ' ', '"'}


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _get_st_model():
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _st_model


def compute_confidence(subject, verb, obj) -> float:
    """Compute syntactic confidence score for an SVO triple."""
    confidence = 1.0
    if subject.dep_ in {"nsubj", "nsubjpass"}:
        confidence += 0.5
    if verb.pos_ == "VERB":
        confidence += 0.5
    if obj.dep_ in {"dobj", "attr", "pobj"}:
        confidence += 0.5
    if subject.dep_ not in {"nsubj", "nsubjpass"}:
        confidence -= 0.5
    if verb.pos_ != "VERB":
        confidence -= 0.5
    if obj.dep_ not in {"dobj", "attr", "pobj"}:
        confidence -= 0.5
    if subject.head == verb:
        confidence += 0.2
    if obj.head == verb:
        confidence += 0.2
    distance = abs(subject.i - verb.i) + abs(verb.i - obj.i)
    confidence -= (distance * 0.05)
    if subject.ent_type_:
        confidence += 0.1
    if obj.ent_type_:
        confidence += 0.1
    confidence = max(0.0, min(1.0, confidence / 2.5))
    return confidence


def extract_relations_with_confidence(text: str, confidence_threshold: float = 0.4) -> List[Dict]:
    """
    Extract SVO relation triples from text using spaCy dependency parsing.
    Returns list of dicts with keys: subject, verb, object, confidence.
    """
    nlp = _get_nlp()
    doc = nlp(text)
    relations = []

    for sent in doc.sents:
        subjects = [t for t in sent if "subj" in t.dep_]
        verbs = [t for t in sent if t.pos_ == "VERB"]
        objects = [t for t in sent if "obj" in t.dep_ or "attr" in t.dep_]
        preps = [t for t in sent if t.dep_ == "prep"]

        for verb in verbs:
            for subject in subjects:
                for obj in objects:
                    if len({subject.text, verb.text, obj.text}) < 3:
                        continue
                    conf = compute_confidence(subject, verb, obj)
                    if conf >= confidence_threshold:
                        relations.append({
                            "subject": subject.text,
                            "verb": verb.text,
                            "object": obj.text,
                            "confidence": conf
                        })

        for prep in preps:
            pobj = [child for child in prep.children if child.dep_ == "pobj"]
            if pobj and prep.head.text != pobj[0].text:
                conf = compute_confidence(prep.head, prep, pobj[0])
                if conf >= confidence_threshold:
                    relations.append({
                        "subject": prep.head.text,
                        "verb": prep.text,
                        "object": pobj[0].text,
                        "confidence": conf
                    })

    return relations


def relations_to_tuples(relations: List[Dict]) -> Set[Tuple[str, str, str, str]]:
    """Convert relation dicts to tuple set (subject, verb, object, confidence_str)."""
    tuples = set()
    for rel in relations:
        t = (rel['subject'], rel['verb'], rel['object'], str(rel['confidence']))
        if not any(sym in ''.join(t[:3]) for sym in UNWANTED_SYMBOLS):
            tuples.add(t)
    return tuples


def _check_antonym_core(v1: str, v2: str, s1: str = "", o1: str = "") -> bool:
    """Two-layer antonym check: WordNet + NLI contradiction."""
    try:
        from .antonym_check import is_antonym_pair
        return is_antonym_pair(v1, v2, use_nli=True, subject=s1 or None, obj=o1 or None)
    except ImportError:
        return False


def is_semantically_valid(tup1: tuple, tup2: tuple, threshold: float = 0.3) -> bool:
    """Check verb semantic similarity with antonym guard."""
    s1, v1, o1 = tup1[:3]
    s2, v2, o2 = tup2[:3]
    if s1.lower() == s2.lower() and v1.lower() == v2.lower() and o1.lower() == o2.lower():
        return True
    if _check_antonym_core(v1, v2, s1, o1):
        return False
    model = _get_st_model()
    embeddings = model.encode([v1, v2], convert_to_tensor=True)
    cosine_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
    return cosine_sim > threshold


def calculate_intersection_count(set1: Set[tuple], set2: Set[tuple]) -> float:
    """
    Calculate partial-match intersection count between two tuple sets.
    Each SVO component contributes 1/3 of a match.
    """
    match_count = Decimal(0)
    matched = set()
    for tup2 in sorted(set2):
        s2, v2, o2 = tup2[:3]
        for tup1 in sorted(set1):
            if tup1 in matched:
                continue
            s1, v1, o1 = tup1[:3]
            match_elements = sum([s1 == s2, v1 == v2, o1 == o2])
            if match_elements > 0 and is_semantically_valid(tup1, tup2):
                match_count += Decimal(match_elements) / Decimal(3)
                matched.add(tup1)
                break
    return float(match_count)


def calculate_triple_intersection_count(set1: Set[tuple], set2: Set[tuple], set3: Set[tuple]) -> float:
    """
    Calculate partial-match intersection count across three tuple sets.
    A triple must appear (partially) in all three sets to count.
    """
    match_count = Decimal(0)
    matched = set()

    set1_sorted = sorted(set1)
    set2_sorted = sorted(set2)
    set3_sorted = sorted(set3)

    if set2 == set3:
        for tup2 in set2_sorted:
            s2, v2, o2 = tup2[:3]
            for tup1 in set1_sorted:
                if tup1 in matched:
                    continue
                s1, v1, o1 = tup1[:3]
                match_elements = sum([s1 == s2, v1 == v2, o1 == o2])
                if match_elements > 0 and is_semantically_valid(tup1, tup2):
                    match_count += Decimal(match_elements) / Decimal(3)
                    matched.add(tup1)
                    break
    else:
        for tup3 in set3_sorted:
            s3, v3, o3 = tup3[:3]
            for tup2 in set2_sorted:
                s2, v2, o2 = tup2[:3]
                if (s2, v2, o2) == (s3, v3, o3):
                    for tup1 in set1_sorted:
                        if tup1 in matched:
                            continue
                        s1, v1, o1 = tup1[:3]
                        match_elements = sum([s1 == s2, v1 == v2, o1 == o2])
                        if match_elements > 0 and is_semantically_valid(tup1, tup2):
                            match_count += Decimal(match_elements) / Decimal(3)
                            matched.add(tup1)
                            break
    return float(match_count)


def compute_rhi_factors(inp_tuples: Set[tuple], ref_tuples: Set[tuple], gen_tuples: Set[tuple]) -> Dict[str, float]:
    """
    Compute the six RHI factors from SVO tuple sets.

    Args:
        inp_tuples: Relation tuples from Input (I)
        ref_tuples: Relation tuples from Reference (R)
        gen_tuples: Relation tuples from Generated (G)

    Returns:
        Dict with keys: ef, ph, of, nh, lh, lf (factor values)
        and intersection_counts for debugging.
    """
    len_I = len(inp_tuples)
    len_R = len(ref_tuples)
    len_G = len(gen_tuples)

    if len_G == 0:
        return {"ef": 0, "ph": 0, "of": 0, "nh": 0, "lh": 0, "lf": 0,
                "intersection_counts": {}}

    I_R = calculate_intersection_count(inp_tuples, ref_tuples)
    I_G = calculate_intersection_count(inp_tuples, gen_tuples)
    R_G = calculate_intersection_count(ref_tuples, gen_tuples)
    I_R_G = calculate_triple_intersection_count(inp_tuples, ref_tuples, gen_tuples)

    # EF: Extractiveness — relations in all three sets
    ef = (3 * I_R_G) / (len_I + len_R + len_G) if (len_I + len_R + len_G) > 0 else 0

    # PH: Positive Hallucination — overlap between Reference and Generated
    ph = (2 * R_G) / (len_R + len_G) if (len_R + len_G) > 0 else 0

    # OF: Over Focus — Input-Generated overlap minus triple intersection
    of = (2 * (I_G - I_R_G)) / (len_I + len_G) if (len_I + len_G) > 0 else 0

    # NH: Negative Hallucination — relations in Generated not in Input or Reference
    nh_num = len_G - (R_G + I_G - I_R_G)
    nh = abs(nh_num) / len_G if len_G > 0 else 0

    # LF: Lost Focus — relations in Input∩Reference but not in Generated
    lf_num = I_R - I_R_G
    lf = lf_num / len_G if len_G > 0 else 0

    # LH: Lost Hallucination — partially grounded, incorrectly assembled
    lh_num = 2 * (len_R - I_R - (R_G - I_R_G))
    lh = lh_num / (len_I + len_G) if (len_I + len_G) > 0 else 0

    return {
        "ef": ef, "ph": ph, "of": of, "nh": nh, "lh": lh, "lf": lf,
        "intersection_counts": {
            "I_R": I_R, "I_G": I_G, "R_G": R_G, "I_R_G": I_R_G,
            "len_I": len_I, "len_R": len_R, "len_G": len_G
        }
    }


def compute_rhi(factors: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    """
    Compute the RHI score from factors.

    Args:
        factors: Dict with keys ef, ph, of, nh, lh, lf
        weights: Optional dict with penalty weights (w_of, w_nh, w_lh, w_lf).
                 Must sum to 1. If None, uses uniform 0.25 (published formula).

    Returns:
        RHI score. Range [0, 1.5]. Threshold at 1.0.
    """
    ef = factors["ef"]
    ph = factors["ph"]
    of_val = factors["of"]
    nh = factors["nh"]
    lh = factors["lh"]
    lf = factors["lf"]

    reward = (ef * ph) / 2

    if weights is None:
        penalty = (of_val + nh + lh + lf) / 4
    else:
        w_of = weights.get("w_of", 0.25)
        w_nh = weights.get("w_nh", 0.25)
        w_lh = weights.get("w_lh", 0.25)
        w_lf = weights.get("w_lf", 0.25)
        penalty = w_of * of_val + w_nh * nh + w_lh * lh + w_lf * lf

    return 1 + reward - penalty


def compute_rhi_softmax(factors: Dict[str, float]) -> float:
    """
    Compute the softmax variant of RHI (RHI*) for use in CHI composite.

    RHI* = (e^EF + e^PH) / (e^EF + e^PH + e^OF + e^NH + e^LH + e^LF)

    Output: [0, 1]. At equilibrium (all equal): 2/6 = 0.333.
    """
    ef = factors["ef"]
    ph = factors["ph"]
    of_val = factors["of"]
    nh = factors["nh"]
    lh = factors["lh"]
    lf = factors["lf"]

    numerator = math.exp(ph) + math.exp(ef)
    denominator = math.exp(ph) + math.exp(ef) + math.exp(nh) + math.exp(of_val) + math.exp(lh) + math.exp(lf)
    return numerator / denominator


def score_text(input_text: str, reference_text: str, generated_text: str,
               weights: Optional[Dict[str, float]] = None,
               confidence_threshold: float = 0.4) -> Dict:
    """
    End-to-end RHI scoring: text in, scores out.

    Args:
        input_text: Source document
        reference_text: Human reference summary
        generated_text: Model-generated summary
        weights: Optional domain-adaptive penalty weights (sum to 1)
        confidence_threshold: SVO extraction confidence threshold

    Returns:
        Dict with rhi, rhi_softmax, factors, and metadata.
    """
    inp_rels = extract_relations_with_confidence(input_text, confidence_threshold)
    ref_rels = extract_relations_with_confidence(reference_text, confidence_threshold)
    gen_rels = extract_relations_with_confidence(generated_text, confidence_threshold)

    inp_tuples = relations_to_tuples(inp_rels)
    ref_tuples = relations_to_tuples(ref_rels)
    gen_tuples = relations_to_tuples(gen_rels)

    factors = compute_rhi_factors(inp_tuples, ref_tuples, gen_tuples)

    rhi = compute_rhi(factors, weights)
    rhi_star = compute_rhi_softmax(factors)

    return {
        "rhi": rhi,
        "rhi_softmax": rhi_star,
        "factors": {k: v for k, v in factors.items() if k != "intersection_counts"},
        "weights_used": weights if weights else {"w_of": 0.25, "w_nh": 0.25, "w_lh": 0.25, "w_lf": 0.25},
        "tuple_counts": {
            "input": len(inp_tuples),
            "reference": len(ref_tuples),
            "generated": len(gen_tuples)
        }
    }
