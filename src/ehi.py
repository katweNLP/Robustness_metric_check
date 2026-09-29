"""
EHI: Entity Hallucination Index.

Published: Katwe et al., Applied Computing and Informatics, 2025.
Softmax over 5 entity factors from Venn diagram of I, R, G entity sets.

EHI = (e^PH + e^EF) / (e^PH + e^EF + e^NH + e^OF + e^LF)
"""

import math
from typing import Dict, Set, Tuple

import spacy

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_entities(text: str) -> Set[str]:
    """Extract named entities from text using spaCy NER."""
    nlp = _get_nlp()
    doc = nlp(text)
    entities = set()
    for ent in doc.ents:
        entities.add(ent.text.lower().strip())
    return entities


def compute_ehi_factors(input_text: str, reference_text: str, generated_text: str) -> Dict:
    """
    Compute the five EHI factors from entity sets.

    Factors:
        EF (Extractiveness): I ∩ R ∩ G — faithfully preserved
        PH (Positive Hallucination): R ∩ G - I — beneficial abstraction
        OF (Over Focus): I ∩ G - R — irrelevant extraction
        NH (Negative Hallucination): G - (I ∪ R) — fabricated
        LF (Lost Focus): R - G — omitted

    Returns dict with factor values and set sizes.
    """
    I = extract_entities(input_text)
    R = extract_entities(reference_text)
    G = extract_entities(generated_text)

    I_count = len(I)
    R_count = len(R)
    G_count = len(G)

    I_R_G = I & R & G
    R_G = R & G
    I_G = I & G
    I_R = I & R

    ef = (3 * len(I_R_G)) / (I_count + R_count + G_count) if (I_count + R_count + G_count) > 0 else 0
    ph = (2 * len(R_G)) / (R_count + G_count) if (R_count + G_count) > 0 else 0
    of = (2 * len(I_G)) / (I_count + G_count) if (I_count + G_count) > 0 else 0
    nh = (G_count - (len(R_G) + len(I_G) - len(I_R_G))) / G_count if G_count > 0 else 0
    lf = (R_count - len(I_R) + len(I_R_G)) / (R_count + G_count) if (R_count + G_count) > 0 else 0

    return {
        "ef": ef, "ph": ph, "of": of, "nh": nh, "lf": lf,
        "len_I": I_count, "len_R": R_count, "len_G": G_count
    }


def compute_ehi(factors: Dict) -> float:
    """
    Softmax-normalized EHI formula.

    EHI = (e^PH + e^EF) / (e^PH + e^EF + e^NH + e^OF + e^LF)
    Output: [0, 1]. At equilibrium (all equal): 2/5 = 0.4.
    Higher = better.
    """
    ef = factors["ef"]
    ph = factors["ph"]
    of_val = factors["of"]
    nh = factors["nh"]
    lf = factors["lf"]

    numerator = math.exp(ph) + math.exp(ef)
    denominator = math.exp(ph) + math.exp(ef) + math.exp(nh) + math.exp(of_val) + math.exp(lf)
    return numerator / denominator
