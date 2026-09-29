"""
QHI: Quantity Hallucination Index.

Detects numerical fabrication in generated text by comparing quantity
expressions across Input (I), Reference (R), and Generated (G) sets.

Five factors (softmax, matching EHI architecture):
  Rewarded: QEF (Quantity Extractiveness), QPH (Quantity Positive Hallucination)
  Penalized: QOF (Quantity Over Focus), QNH (Quantity Negative Hallucination),
             QLF (Quantity Lost Focus)

Formula: QHI = (e^QPH + e^QEF) / (e^QPH + e^QEF + e^QNH + e^QOF + e^QLF)
"""

import math
import re
from typing import Dict, List, Set, Tuple, Optional

import spacy

_nlp = None

UNITS = {
    '%': 'percent', 'percent': 'percent', 'percentage': 'percent',
    '$': 'usd', 'dollar': 'usd', 'dollars': 'usd', 'usd': 'usd',
    '€': 'eur', 'euro': 'eur', 'euros': 'eur',
    '£': 'gbp', 'pound': 'gbp', 'pounds': 'gbp',
    'billion': 'B', 'bn': 'B', 'b': 'B',
    'million': 'M', 'mn': 'M', 'm': 'M',
    'thousand': 'K', 'k': 'K',
    'trillion': 'T', 'tn': 'T', 't': 'T',
}

MULTIPLIERS = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}

NUMBER_PATTERN = re.compile(
    r'(?:[\$€£])\s*'
    r'(\d+(?:[.,]\d+)*)'
    r'\s*(?:billion|million|thousand|trillion|bn|mn|tn|[BMKTbmkt])?'
    r'|'
    r'(\d+(?:[.,]\d+)*)'
    r'\s*(?:%|percent|billion|million|thousand|trillion|bn|mn|tn|[BMKTbmkt]|'
    r'dollars?|euros?|pounds?)?',
    re.IGNORECASE
)


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_quantities(text: str) -> List[Dict]:
    """
    Extract quantity expressions from text.

    Returns list of dicts: {value, raw_text, unit, normalized_value, context}
    """
    nlp = _get_nlp()
    doc = nlp(text)
    quantities = []
    seen_spans = set()

    for ent in doc.ents:
        if ent.label_ in ("MONEY", "PERCENT", "QUANTITY", "CARDINAL"):
            if ent.start_char in seen_spans:
                continue
            seen_spans.add(ent.start_char)
            q = _parse_quantity_entity(ent, doc)
            if q:
                quantities.append(q)

    for match in NUMBER_PATTERN.finditer(text):
        start = match.start()
        if start in seen_spans:
            continue
        q = _parse_quantity_match(match, text, doc)
        if q:
            seen_spans.add(start)
            quantities.append(q)

    return quantities


def _parse_quantity_entity(ent, doc) -> Optional[Dict]:
    """Parse a spaCy entity into a quantity dict."""
    raw = ent.text.strip()
    value, unit, multiplier = _parse_number_string(raw)
    if value is None:
        return None

    normalized = value * multiplier
    context = _get_context(ent.sent.text, ent.text)

    return {
        "value": value,
        "normalized_value": normalized,
        "unit": unit,
        "raw_text": raw,
        "context": context
    }


def _parse_quantity_match(match, text: str, doc) -> Optional[Dict]:
    """Parse a regex match into a quantity dict."""
    raw = match.group(0).strip()
    if not raw:
        return None
    value, unit, multiplier = _parse_number_string(raw)
    if value is None:
        return None

    normalized = value * multiplier
    start = max(0, match.start() - 50)
    end = min(len(text), match.end() + 50)
    context = text[start:end]

    return {
        "value": value,
        "normalized_value": normalized,
        "unit": unit,
        "raw_text": raw,
        "context": context
    }


def _parse_number_string(s: str) -> Tuple[Optional[float], str, float]:
    """
    Parse a number string like "$94.9 billion" into (94.9, 'usd', 1e9).
    Returns (value, unit, multiplier) or (None, '', 1) on failure.
    """
    s = s.strip()
    unit = ''
    multiplier = 1.0

    for sym, canonical in UNITS.items():
        if sym in s.lower():
            if canonical in MULTIPLIERS:
                multiplier = MULTIPLIERS[canonical]
            else:
                unit = canonical
            break

    nums = re.findall(r'\d+(?:[.,]\d+)*', s)
    if not nums:
        return None, '', 1.0

    num_str = nums[0].replace(',', '')
    try:
        value = float(num_str)
    except ValueError:
        return None, '', 1.0

    return value, unit, multiplier


def _get_context(sentence: str, quantity_text: str) -> str:
    """Extract surrounding context for a quantity."""
    idx = sentence.find(quantity_text)
    if idx == -1:
        return sentence[:100]
    start = max(0, idx - 30)
    end = min(len(sentence), idx + len(quantity_text) + 30)
    return sentence[start:end]


def quantities_match(q1: Dict, q2: Dict, epsilon: float = 0.05) -> bool:
    """
    Determine if two quantities match (tolerance-aware).

    Checks: exact value match, within-epsilon, or unit-converted match.
    """
    v1 = q1["normalized_value"]
    v2 = q2["normalized_value"]

    if v1 == v2:
        return True

    if v1 != 0 and abs(v1 - v2) / abs(v1) <= epsilon:
        return True
    if v2 != 0 and abs(v1 - v2) / abs(v2) <= epsilon:
        return True

    return False


def compute_quantity_sets(input_text: str, reference_text: str, generated_text: str,
                          epsilon: float = 0.05) -> Tuple[Set[int], Set[int], Set[int], List, List, List]:
    """
    Extract quantities and compute Venn diagram intersections.

    Returns indices into the respective quantity lists that form each region.
    """
    I_quants = extract_quantities(input_text)
    R_quants = extract_quantities(reference_text)
    G_quants = extract_quantities(generated_text)

    I_matched_by_R = set()
    I_matched_by_G = set()
    R_matched_by_G = set()
    G_matched_by_I = set()
    G_matched_by_R = set()
    R_matched_by_I = set()

    for i, qi in enumerate(I_quants):
        for j, qr in enumerate(R_quants):
            if quantities_match(qi, qr, epsilon):
                I_matched_by_R.add(i)
                R_matched_by_I.add(j)
                break

    for i, qi in enumerate(I_quants):
        for k, qg in enumerate(G_quants):
            if quantities_match(qi, qg, epsilon):
                I_matched_by_G.add(i)
                G_matched_by_I.add(k)
                break

    for j, qr in enumerate(R_quants):
        for k, qg in enumerate(G_quants):
            if quantities_match(qr, qg, epsilon):
                R_matched_by_G.add(j)
                G_matched_by_R.add(k)
                break

    I_R = I_matched_by_R
    I_G = I_matched_by_G
    R_G = R_matched_by_G
    I_R_G = I_matched_by_R & I_matched_by_G

    return I_R, I_G, R_G, I_quants, R_quants, G_quants


def compute_qhi_factors(input_text: str, reference_text: str, generated_text: str,
                        epsilon: float = 0.05) -> Dict[str, float]:
    """
    Compute the five QHI factors from quantity sets.

    Returns dict with qef, qph, qof, qnh, qlf.
    """
    I_quants = extract_quantities(input_text)
    R_quants = extract_quantities(reference_text)
    G_quants = extract_quantities(generated_text)

    len_I = len(I_quants)
    len_R = len(R_quants)
    len_G = len(G_quants)

    if len_G == 0 and len_R == 0 and len_I == 0:
        return {"qef": 0, "qph": 0, "qof": 0, "qnh": 0, "qlf": 0,
                "counts": {"len_I": 0, "len_R": 0, "len_G": 0}}

    I_in_R = 0
    I_in_G = 0
    R_in_G = 0
    I_in_R_in_G = 0

    i_matched_r = [False] * len_I
    i_matched_g = [False] * len_I
    r_matched_g = [False] * len_R

    for i, qi in enumerate(I_quants):
        for j, qr in enumerate(R_quants):
            if quantities_match(qi, qr, epsilon):
                i_matched_r[i] = True
                break

    for i, qi in enumerate(I_quants):
        for k, qg in enumerate(G_quants):
            if quantities_match(qi, qg, epsilon):
                i_matched_g[i] = True
                break

    for j, qr in enumerate(R_quants):
        for k, qg in enumerate(G_quants):
            if quantities_match(qr, qg, epsilon):
                r_matched_g[j] = True
                break

    I_R = sum(i_matched_r)
    I_G = sum(i_matched_g)
    R_G = sum(r_matched_g)
    I_R_G = sum(1 for i in range(len_I) if i_matched_r[i] and i_matched_g[i])

    # QEF: Quantity Extractiveness — in all three
    qef = (3 * I_R_G) / (len_I + len_R + len_G) if (len_I + len_R + len_G) > 0 else 0

    # QPH: Quantity Positive Hallucination — in R and G but not I
    qph = (2 * R_G) / (len_R + len_G) if (len_R + len_G) > 0 else 0

    # QOF: Quantity Over Focus — in I and G but not R
    qof = (2 * (I_G - I_R_G)) / (len_I + len_G) if (len_I + len_G) > 0 else 0

    # QNH: Quantity Negative Hallucination — in G only (fabricated numbers)
    g_matched = set()
    for k, qg in enumerate(G_quants):
        for qi in I_quants:
            if quantities_match(qg, qi, epsilon):
                g_matched.add(k)
                break
        if k not in g_matched:
            for qr in R_quants:
                if quantities_match(qg, qr, epsilon):
                    g_matched.add(k)
                    break
    qnh = (len_G - len(g_matched)) / len_G if len_G > 0 else 0

    # QLF: Quantity Lost Focus — in R but not G
    r_not_in_g = sum(1 for matched in r_matched_g if not matched)
    qlf = r_not_in_g / (len_R + len_G) if (len_R + len_G) > 0 else 0

    return {
        "qef": qef, "qph": qph, "qof": qof, "qnh": qnh, "qlf": qlf,
        "counts": {"len_I": len_I, "len_R": len_R, "len_G": len_G,
                   "I_R": I_R, "I_G": I_G, "R_G": R_G, "I_R_G": I_R_G}
    }


def compute_qhi(factors: Dict[str, float]) -> float:
    """
    Compute QHI using softmax aggregation (matching EHI architecture).

    QHI = (e^QPH + e^QEF) / (e^QPH + e^QEF + e^QNH + e^QOF + e^QLF)

    Output: [0, 1]. At equilibrium: 2/5 = 0.4.
    """
    qef = factors["qef"]
    qph = factors["qph"]
    qof = factors["qof"]
    qnh = factors["qnh"]
    qlf = factors["qlf"]

    numerator = math.exp(qph) + math.exp(qef)
    denominator = (math.exp(qph) + math.exp(qef) +
                   math.exp(qnh) + math.exp(qof) + math.exp(qlf))
    return numerator / denominator


def score_quantities(input_text: str, reference_text: str, generated_text: str,
                     epsilon: float = 0.05) -> Dict:
    """
    End-to-end QHI scoring: text in, score out.

    Returns dict with qhi score, factors, and quantity counts.
    """
    factors = compute_qhi_factors(input_text, reference_text, generated_text, epsilon)
    qhi = compute_qhi(factors)

    return {
        "qhi": qhi,
        "factors": {k: v for k, v in factors.items() if k != "counts"},
        "counts": factors.get("counts", {})
    }
