"""
RHI*: Relation Hallucination Index — Softmax Variant.

Softmax reformulation of RHI (Katwe et al., ACM FIRE 2024) for use
within the CHI composite. Same 6 factors, same extraction pipeline,
different aggregation function.

RHI* = (e^EF + e^PH) / (e^EF + e^PH + e^OF + e^NH + e^LH + e^LF)

Output: [0, 1]. At equilibrium: 2/6 = 0.333.
"""

import math
from typing import Dict, List, Set, Optional

import spacy
from sentence_transformers import SentenceTransformer, util
from decimal import Decimal

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
    """Compute syntactic confidence for SVO triple (from published code)."""
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
    return max(0.0, min(1.0, confidence / 2.5))


def extract_relations(text: str, confidence_threshold: float = 0.4) -> List[Dict]:
    """Extract SVO relations with confidence scores."""
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
                            "subject": subject.text, "verb": verb.text,
                            "object": obj.text, "confidence": conf
                        })

        for prep in preps:
            pobj = [c for c in prep.children if c.dep_ == "pobj"]
            if pobj and prep.head.text != pobj[0].text:
                conf = compute_confidence(prep.head, prep, pobj[0])
                if conf >= confidence_threshold:
                    relations.append({
                        "subject": prep.head.text, "verb": prep.text,
                        "object": pobj[0].text, "confidence": conf
                    })

    return relations


def _to_tuples(relations: List[Dict]) -> Set[tuple]:
    """Convert to filtered tuple set."""
    tuples = set()
    for r in relations:
        t = (r['subject'], r['verb'], r['object'], str(r['confidence']))
        if not any(sym in ''.join(t[:3]) for sym in UNWANTED_SYMBOLS):
            tuples.add(t)
    return tuples


def _batch_encode_verbs(*tuple_sets: Set[tuple]) -> Dict[str, object]:
    """Batch-encode all unique verbs across tuple sets once."""
    unique_verbs = set()
    for ts in tuple_sets:
        for t in ts:
            unique_verbs.add(t[1])
    if not unique_verbs:
        return {}
    verb_list = sorted(unique_verbs)
    model = _get_st_model()
    embeddings = model.encode(verb_list, convert_to_tensor=True)
    return {verb: embeddings[i] for i, verb in enumerate(verb_list)}


def _semantic_valid_cached(tup1: tuple, tup2: tuple, verb_embs: Dict[str, object]) -> bool:
    """Check verb similarity using pre-computed embeddings."""
    s1, v1, o1 = tup1[:3]
    s2, v2, o2 = tup2[:3]
    if s1.lower() == s2.lower() and v1.lower() == v2.lower() and o1.lower() == o2.lower():
        return True
    if v1 not in verb_embs or v2 not in verb_embs:
        return False
    return util.cos_sim(verb_embs[v1], verb_embs[v2]).item() > 0.3


def _semantic_valid(tup1: tuple, tup2: tuple) -> bool:
    """Check verb similarity via sentence-transformer (backward-compat wrapper)."""
    s1, v1, o1 = tup1[:3]
    s2, v2, o2 = tup2[:3]
    if s1.lower() == s2.lower() and v1.lower() == v2.lower() and o1.lower() == o2.lower():
        return True
    model = _get_st_model()
    embs = model.encode([tup1[1], tup2[1]], convert_to_tensor=True)
    return util.cos_sim(embs[0], embs[1]).item() > 0.3


def _intersect_2(set1: Set[tuple], set2: Set[tuple],
                 verb_embs: Optional[Dict[str, object]] = None) -> float:
    """Partial-match intersection count (1/3 per matching component)."""
    count = Decimal(0)
    matched = set()
    for t2 in sorted(set2):
        s2, v2, o2 = t2[:3]
        for t1 in sorted(set1):
            if t1 in matched:
                continue
            s1, v1, o1 = t1[:3]
            m = sum([s1 == s2, v1 == v2, o1 == o2])
            if m > 0:
                if verb_embs is not None:
                    valid = _semantic_valid_cached(t1, t2, verb_embs)
                else:
                    valid = _semantic_valid(t1, t2)
                if valid:
                    count += Decimal(m) / Decimal(3)
                    matched.add(t1)
                    break
    return float(count)


def _intersect_3(set1: Set[tuple], set2: Set[tuple], set3: Set[tuple],
                 verb_embs: Optional[Dict[str, object]] = None) -> float:
    """Three-way partial-match intersection count."""
    count = Decimal(0)
    matched = set()
    for t3 in sorted(set3):
        s3, v3, o3 = t3[:3]
        for t2 in sorted(set2):
            s2, v2, o2 = t2[:3]
            if (s2, v2, o2) == (s3, v3, o3):
                for t1 in sorted(set1):
                    if t1 in matched:
                        continue
                    s1, v1, o1 = t1[:3]
                    m = sum([s1 == s2, v1 == v2, o1 == o2])
                    if m > 0:
                        if verb_embs is not None:
                            valid = _semantic_valid_cached(t1, t2, verb_embs)
                        else:
                            valid = _semantic_valid(t1, t2)
                        if valid:
                            count += Decimal(m) / Decimal(3)
                            matched.add(t1)
                            break
    return float(count)


def compute_rhi_star_factors(input_text: str, reference_text: str, generated_text: str,
                             confidence_threshold: float = 0.4) -> Dict[str, float]:
    """
    Compute 6 RHI factors from text.

    Same factor computation as published RHI, different aggregation.
    """
    inp_rels = extract_relations(input_text, confidence_threshold)
    ref_rels = extract_relations(reference_text, confidence_threshold)
    gen_rels = extract_relations(generated_text, confidence_threshold)

    inp_t = _to_tuples(inp_rels)
    ref_t = _to_tuples(ref_rels)
    gen_t = _to_tuples(gen_rels)

    len_I = len(inp_t)
    len_R = len(ref_t)
    len_G = len(gen_t)

    if len_G == 0:
        return {"ef": 0, "ph": 0, "of": 0, "nh": 0, "lh": 0, "lf": 0}

    verb_embs = _batch_encode_verbs(inp_t, ref_t, gen_t)

    I_R = _intersect_2(inp_t, ref_t, verb_embs)
    I_G = _intersect_2(inp_t, gen_t, verb_embs)
    R_G = _intersect_2(ref_t, gen_t, verb_embs)
    I_R_G = _intersect_3(inp_t, ref_t, gen_t, verb_embs)

    ef = (3 * I_R_G) / (len_I + len_R + len_G) if (len_I + len_R + len_G) > 0 else 0
    ph = (2 * R_G) / (len_R + len_G) if (len_R + len_G) > 0 else 0
    of = (2 * (I_G - I_R_G)) / (len_I + len_G) if (len_I + len_G) > 0 else 0
    nh_num = len_G - (R_G + I_G - I_R_G)
    nh = abs(nh_num) / len_G if len_G > 0 else 0
    lf_num = I_R - I_R_G
    lf = lf_num / len_G if len_G > 0 else 0
    lh_num = 2 * (len_R - I_R - (R_G - I_R_G))
    lh = lh_num / (len_I + len_G) if (len_I + len_G) > 0 else 0

    return {"ef": ef, "ph": ph, "of": of, "nh": nh, "lh": lh, "lf": lf}


def compute_rhi_star(factors: Dict[str, float]) -> float:
    """
    Softmax RHI* aggregation.

    RHI* = (e^EF + e^PH) / (e^EF + e^PH + e^OF + e^NH + e^LH + e^LF)
    Output: [0, 1]. At equilibrium: 2/6 = 0.333.
    """
    numerator = math.exp(factors["ef"]) + math.exp(factors["ph"])
    denominator = (math.exp(factors["ef"]) + math.exp(factors["ph"]) +
                   math.exp(factors["of"]) + math.exp(factors["nh"]) +
                   math.exp(factors["lh"]) + math.exp(factors["lf"]))
    return numerator / denominator
