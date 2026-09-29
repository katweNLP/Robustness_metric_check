"""
CHI: Composite Hallucination Index.

Unified softmax architecture fusing EHI (entity), RHI* (relation), and QHI (quantity).
CHI = Harmonic Mean(EHI, RHI*, QHI) — all three in [0, 1].

Uses:
- EHI: softmax over 5 entity factors (published, Katwe et al. ACI 2025)
- RHI*: softmax over 6 relation factors (variant of Katwe et al. FIRE 2024)
- QHI: softmax over 5 quantity factors (novel)
"""

import math
from typing import Dict, Optional

from .qhi import compute_qhi_factors, compute_qhi, score_quantities
from .ehi import compute_ehi_factors, compute_ehi
from .rhi_star import compute_rhi_star_factors, compute_rhi_star


def harmonic_mean_2(a: float, b: float) -> float:
    """Harmonic mean of two values."""
    if a <= 0 or b <= 0:
        return 0.0
    return 2 / (1/a + 1/b)


def harmonic_mean_3(a: float, b: float, c: float) -> float:
    """Harmonic mean of three values."""
    if a <= 0 or b <= 0 or c <= 0:
        return 0.0
    return 3 / (1/a + 1/b + 1/c)


def compute_chi(ehi: float, rhi_star: float, qhi: float,
                has_entities: bool = True, has_quantities: bool = True) -> float:
    """
    Compute Composite Hallucination Index.

    Adaptive: uses 2-way harmonic mean if one dimension is absent.

    Args:
        ehi: Entity Hallucination Index [0, 1]
        rhi_star: Relation Hallucination Index softmax variant [0, 1]
        qhi: Quantity Hallucination Index [0, 1]
        has_entities: Whether generated text has entities to evaluate
        has_quantities: Whether generated text has quantities to evaluate

    Returns:
        CHI score in [0, 1]. Higher = better.
    """
    if not has_entities and not has_quantities:
        return rhi_star
    elif not has_quantities:
        return harmonic_mean_2(ehi, rhi_star)
    elif not has_entities:
        return harmonic_mean_2(qhi, rhi_star)
    else:
        return harmonic_mean_3(ehi, rhi_star, qhi)


def score_text(input_text: str, reference_text: str, generated_text: str,
               confidence_threshold: float = 0.4, epsilon: float = 0.05) -> Dict:
    """
    End-to-end CHI scoring: text in, composite score out.

    Computes all three sub-indices and fuses via adaptive harmonic mean.

    Args:
        input_text: Source document
        reference_text: Human reference summary
        generated_text: Model-generated summary
        confidence_threshold: SVO extraction confidence threshold
        epsilon: Quantity matching tolerance

    Returns:
        Dict with chi, ehi, rhi_star, qhi, factors, and metadata.
    """
    # EHI: Entity dimension
    ehi_result = compute_ehi_factors(input_text, reference_text, generated_text)
    ehi_score = compute_ehi(ehi_result)

    # RHI*: Relation dimension (softmax variant)
    rhi_result = compute_rhi_star_factors(input_text, reference_text, generated_text,
                                          confidence_threshold)
    rhi_star_score = compute_rhi_star(rhi_result)

    # QHI: Quantity dimension
    qhi_result = compute_qhi_factors(input_text, reference_text, generated_text, epsilon)
    qhi_score = compute_qhi(qhi_result)

    # Determine which dimensions are present
    has_entities = ehi_result.get("len_G", 0) > 0 if isinstance(ehi_result, dict) else True
    has_quantities = qhi_result.get("counts", {}).get("len_G", 0) > 0

    # Composite
    chi_score = compute_chi(ehi_score, rhi_star_score, qhi_score,
                            has_entities, has_quantities)

    return {
        "chi": chi_score,
        "ehi": ehi_score,
        "rhi_star": rhi_star_score,
        "qhi": qhi_score,
        "dimensions_used": sum([True, has_entities, has_quantities]),
        "has_entities": has_entities,
        "has_quantities": has_quantities,
        "factors": {
            "ehi_factors": {k: v for k, v in ehi_result.items() if k != "len_G"} if isinstance(ehi_result, dict) else ehi_result,
            "rhi_factors": {k: v for k, v in rhi_result.items() if k != "intersection_counts"},
            "qhi_factors": {k: v for k, v in qhi_result.items() if k != "counts"}
        }
    }
