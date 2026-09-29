"""
EHI Enhanced: Entity Hallucination Index with Domain Knowledge Base Integration.

Demonstrates how domain-specific entity normalization (UMLS for medical,
legal ontology for legal, financial KB for financial) improves EHI accuracy
by resolving synonyms, aliases, and abbreviations before set comparison.

Three normalization strategies:
1. Plain (spaCy NER, exact string match) — current baseline
2. UMLS-style (medical synonym dictionary) — resolves drug/disease aliases
3. Domain-adaptive (auto-selects normalization based on detected domain)

Key insight: Without normalization, "heart attack" and "myocardial infarction"
are treated as DIFFERENT entities, causing false NH (fabrication) penalties.
With normalization, both map to the same canonical concept → correct EF match.
"""

import math
import re
from typing import Dict, Set, Tuple

import spacy

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ==============================================================================
# MEDICAL KNOWLEDGE BASE (UMLS-style synonym dictionary)
# Maps surface forms to canonical concepts (simulating CUI resolution)
# ==============================================================================

MEDICAL_SYNONYMS = {
    # Cardiovascular
    "heart attack": "myocardial_infarction",
    "myocardial infarction": "myocardial_infarction",
    "mi": "myocardial_infarction",
    "cardiac arrest": "cardiac_arrest",
    "heart failure": "heart_failure",
    "chf": "heart_failure",
    "congestive heart failure": "heart_failure",
    "hypertension": "hypertension",
    "high blood pressure": "hypertension",
    "htn": "hypertension",
    "stroke": "cerebrovascular_accident",
    "cva": "cerebrovascular_accident",
    "cerebrovascular accident": "cerebrovascular_accident",

    # Medications
    "amoxicillin": "amoxicillin",
    "amoxil": "amoxicillin",
    "augmentin": "amoxicillin_clavulanate",
    "metformin": "metformin",
    "glucophage": "metformin",
    "aspirin": "acetylsalicylic_acid",
    "acetylsalicylic acid": "acetylsalicylic_acid",
    "asa": "acetylsalicylic_acid",
    "ibuprofen": "ibuprofen",
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "acetaminophen": "acetaminophen",
    "tylenol": "acetaminophen",
    "paracetamol": "acetaminophen",
    "lisinopril": "lisinopril",
    "prinivil": "lisinopril",
    "zestril": "lisinopril",
    "atorvastatin": "atorvastatin",
    "lipitor": "atorvastatin",
    "omeprazole": "omeprazole",
    "prilosec": "omeprazole",

    # Conditions
    "diabetes": "diabetes_mellitus",
    "diabetes mellitus": "diabetes_mellitus",
    "dm": "diabetes_mellitus",
    "type 2 diabetes": "diabetes_mellitus_type2",
    "t2dm": "diabetes_mellitus_type2",
    "pneumonia": "pneumonia",
    "lung infection": "pneumonia",
    "copd": "chronic_obstructive_pulmonary_disease",
    "chronic obstructive pulmonary disease": "chronic_obstructive_pulmonary_disease",
    "asthma": "asthma",
    "bronchial asthma": "asthma",
    "cancer": "malignant_neoplasm",
    "malignancy": "malignant_neoplasm",
    "tumor": "neoplasm",
    "tumour": "neoplasm",

    # Anatomy
    "kidney": "kidney",
    "renal": "kidney",
    "liver": "liver",
    "hepatic": "liver",
    "brain": "brain",
    "cerebral": "brain",
    "lung": "lung",
    "pulmonary": "lung",

    # Procedures
    "surgery": "surgical_procedure",
    "operation": "surgical_procedure",
    "mri": "magnetic_resonance_imaging",
    "magnetic resonance imaging": "magnetic_resonance_imaging",
    "ct scan": "computed_tomography",
    "computed tomography": "computed_tomography",
    "cat scan": "computed_tomography",
    "ecg": "electrocardiogram",
    "ekg": "electrocardiogram",
    "electrocardiogram": "electrocardiogram",
}

# ==============================================================================
# LEGAL KNOWLEDGE BASE
# ==============================================================================

LEGAL_SYNONYMS = {
    "supreme court": "supreme_court",
    "sc": "supreme_court",
    "scotus": "supreme_court",
    "high court": "high_court",
    "hc": "high_court",
    "district court": "district_court",
    "trial court": "district_court",
    "plaintiff": "plaintiff",
    "petitioner": "plaintiff",
    "complainant": "plaintiff",
    "defendant": "defendant",
    "respondent": "defendant",
    "accused": "defendant",
    "article 21": "article_21",
    "right to life": "article_21",
    "article 14": "article_14",
    "right to equality": "article_14",
}

# ==============================================================================
# FINANCIAL KNOWLEDGE BASE
# ==============================================================================

FINANCIAL_SYNONYMS = {
    "revenue": "revenue",
    "sales": "revenue",
    "top line": "revenue",
    "net income": "net_income",
    "profit": "net_income",
    "bottom line": "net_income",
    "earnings": "net_income",
    "operating income": "operating_income",
    "ebit": "operating_income",
    "market cap": "market_capitalization",
    "market capitalization": "market_capitalization",
    "eps": "earnings_per_share",
    "earnings per share": "earnings_per_share",
    "q1": "quarter_1",
    "first quarter": "quarter_1",
    "q2": "quarter_2",
    "second quarter": "quarter_2",
    "q3": "quarter_3",
    "third quarter": "quarter_3",
    "q4": "quarter_4",
    "fourth quarter": "quarter_4",
    "fy": "fiscal_year",
    "fiscal year": "fiscal_year",
    "yoy": "year_over_year",
    "year over year": "year_over_year",
    "year-over-year": "year_over_year",
}


def normalize_entity(entity: str, domain: str = "general") -> str:
    """
    Normalize an entity string using domain knowledge base.
    Returns canonical form if found, otherwise lowercase original.
    """
    entity_lower = entity.lower().strip()

    if domain == "medical":
        if entity_lower in MEDICAL_SYNONYMS:
            return MEDICAL_SYNONYMS[entity_lower]
    elif domain == "legal":
        if entity_lower in LEGAL_SYNONYMS:
            return LEGAL_SYNONYMS[entity_lower]
    elif domain == "financial":
        if entity_lower in FINANCIAL_SYNONYMS:
            return FINANCIAL_SYNONYMS[entity_lower]

    # Try all domains if "general" or not found
    for kb in [MEDICAL_SYNONYMS, LEGAL_SYNONYMS, FINANCIAL_SYNONYMS]:
        if entity_lower in kb:
            return kb[entity_lower]

    return entity_lower


def extract_entities_enhanced(text: str, domain: str = "general") -> Set[str]:
    """
    Extract and normalize entities using domain KB.
    Returns set of canonical entity identifiers.
    """
    nlp = _get_nlp()
    doc = nlp(text)
    entities = set()

    for ent in doc.ents:
        normalized = normalize_entity(ent.text, domain)
        entities.add(normalized)

    # Also check for KB terms not caught by NER (pattern matching)
    text_lower = text.lower()
    if domain == "medical":
        kb = MEDICAL_SYNONYMS
    elif domain == "legal":
        kb = LEGAL_SYNONYMS
    elif domain == "financial":
        kb = FINANCIAL_SYNONYMS
    else:
        kb = {**MEDICAL_SYNONYMS, **LEGAL_SYNONYMS, **FINANCIAL_SYNONYMS}

    for term, canonical in kb.items():
        if term in text_lower:
            entities.add(canonical)

    return entities


def extract_entities_plain(text: str) -> Set[str]:
    """Plain entity extraction (spaCy NER, no normalization). Baseline."""
    nlp = _get_nlp()
    doc = nlp(text)
    return set(ent.text.lower().strip() for ent in doc.ents)


def compute_ehi_factors_from_sets(I: Set, R: Set, G: Set) -> Dict[str, float]:
    """Compute 5 EHI factors from pre-computed entity sets."""
    len_I, len_R, len_G = len(I), len(R), len(G)

    if len_G == 0 and len_R == 0 and len_I == 0:
        return {"ef": 0, "ph": 0, "of": 0, "nh": 0, "lf": 0}

    I_R_G = I & R & G
    R_G = R & G
    I_G = I & G
    I_R = I & R

    ef = (3 * len(I_R_G)) / (len_I + len_R + len_G) if (len_I + len_R + len_G) > 0 else 0
    ph = (2 * len(R_G)) / (len_R + len_G) if (len_R + len_G) > 0 else 0
    of = (2 * len(I_G)) / (len_I + len_G) if (len_I + len_G) > 0 else 0
    nh = (len_G - (len(R_G) + len(I_G) - len(I_R_G))) / len_G if len_G > 0 else 0
    lf = (len_R - len(I_R) + len(I_R_G)) / (len_R + len_G) if (len_R + len_G) > 0 else 0

    return {"ef": ef, "ph": ph, "of": of, "nh": nh, "lf": lf}


def compute_ehi_score(factors: Dict[str, float]) -> float:
    """Softmax EHI from factors."""
    numerator = math.exp(factors["ph"]) + math.exp(factors["ef"])
    denominator = (math.exp(factors["ph"]) + math.exp(factors["ef"]) +
                   math.exp(factors["nh"]) + math.exp(factors["of"]) + math.exp(factors["lf"]))
    return numerator / denominator


def compute_ehi_plain(source: str, reference: str, generated: str) -> float:
    """EHI with plain spaCy NER (baseline)."""
    I = extract_entities_plain(source)
    R = extract_entities_plain(reference)
    G = extract_entities_plain(generated)
    factors = compute_ehi_factors_from_sets(I, R, G)
    return compute_ehi_score(factors)


def compute_ehi_enhanced(source: str, reference: str, generated: str, domain: str = "medical") -> float:
    """EHI with domain-KB normalization (enhanced)."""
    I = extract_entities_enhanced(source, domain)
    R = extract_entities_enhanced(reference, domain)
    G = extract_entities_enhanced(generated, domain)
    factors = compute_ehi_factors_from_sets(I, R, G)
    return compute_ehi_score(factors)


def compare_plain_vs_enhanced(source: str, reference: str, generated: str, domain: str = "medical") -> Dict:
    """
    Compare plain EHI vs domain-enhanced EHI on same texts.
    Returns both scores + entity sets for analysis.
    """
    # Plain
    I_plain = extract_entities_plain(source)
    R_plain = extract_entities_plain(reference)
    G_plain = extract_entities_plain(generated)
    f_plain = compute_ehi_factors_from_sets(I_plain, R_plain, G_plain)
    ehi_plain = compute_ehi_score(f_plain)

    # Enhanced
    I_enh = extract_entities_enhanced(source, domain)
    R_enh = extract_entities_enhanced(reference, domain)
    G_enh = extract_entities_enhanced(generated, domain)
    f_enh = compute_ehi_factors_from_sets(I_enh, R_enh, G_enh)
    ehi_enhanced = compute_ehi_score(f_enh)

    return {
        "ehi_plain": ehi_plain,
        "ehi_enhanced": ehi_enhanced,
        "improvement": ehi_enhanced - ehi_plain,
        "plain_entities": {"I": len(I_plain), "R": len(R_plain), "G": len(G_plain)},
        "enhanced_entities": {"I": len(I_enh), "R": len(R_enh), "G": len(G_enh)},
        "factors_plain": f_plain,
        "factors_enhanced": f_enh,
    }
