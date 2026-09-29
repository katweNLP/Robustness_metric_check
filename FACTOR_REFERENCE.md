# Canonical Factor Reference — EHI & RHI
## (Use this as the single source of truth for all paper writing)

---

## Three Entity/Relation Sets

| Set | Symbol | Source |
|-----|--------|--------|
| Input entities/relations | I | Source document |
| Reference entities/relations | R | Human-written reference summary |
| Generated entities/relations | G | Model-generated summary |

---

## EHI: Entity Hallucination Index (5 factors)

**Formula:**
```
EHI = (e^PH + e^EF) / (e^PH + e^EF + e^NH + e^OF + e^LF)
```

**Aggregation:** Softmax (bounded [0,1], smooth gradients, no arbitrary weights)

| Factor | Full Name | Venn Region | Meaning | Role |
|--------|-----------|-------------|---------|------|
| EF | Extractiveness | I ∩ R ∩ G | Entity in source, reference, AND generated — faithfully preserved | **Rewarded** (numerator) |
| PH | Positive Hallucination | R ∩ G − I | Entity in reference and generated but NOT source — beneficial abstraction | **Rewarded** (numerator) |
| OF | Over Focus | I ∩ G − R | Entity in source and generated but NOT reference — irrelevant extraction | **Penalized** (denominator only) |
| NH | Negative Hallucination | G − (I ∪ R) | Entity ONLY in generated — fabricated, no support anywhere | **Penalized** (denominator only) |
| LF | Lost Focus | R − G | Entity in reference but NOT generated — critical omission | **Penalized** (denominator only) |

**Individual Factor Formulas:**
```
EF = 3 × |I ∩ R ∩ G| / (|I| + |R| + |G|)
PH = 2 × |R ∩ G| / (|R| + |G|)               [Dice coefficient, R and G]
OF = 2 × |I ∩ G| / (|I| + |G|)               [Dice coefficient, I and G]
NH = (|G| - (|R ∩ G| + |I ∩ G| - |I ∩ R ∩ G|)) / |G|    [Fraction of G unsupported]
LF = (|R| - |I ∩ R| + |I ∩ R ∩ G|) / (|R| + |G|)         [Omission rate]
```

**Properties:**
- Output range: [0, 1]
- At equilibrium (all factors equal): EHI = 2/5 = 0.4 (pessimistic bias)
- Higher = better (more desirable factors dominate)
- Designed for RL training (smooth gradients, bounded)

---

## RHI: Relation Hallucination Index (6 factors)

**Formula:**
```
RHI = 1 + (EF × PH)/2 - (OF + NH + LH + LF)/4
```

**Aggregation:** Linear with base offset (interpretable threshold at 1.0)

| Factor | Full Name | Venn Region | Meaning | Role |
|--------|-----------|-------------|---------|------|
| EF | Extractiveness | I ∩ R ∩ G | Relation in source, reference, AND generated — faithfully preserved | **Rewarded** (multiplicative) |
| PH | Positive Hallucination | R ∩ G − I | Relation in reference and generated but NOT source — beneficial abstraction | **Rewarded** (multiplicative) |
| OF | Over Focus | I ∩ G − R | Relation in source and generated but NOT reference — irrelevant extraction | **Penalized** (additive) |
| NH | Negative Hallucination | G − (I ∪ R) | Relation ONLY in generated — completely fabricated | **Penalized** (additive) |
| LH | Lost Hallucination | G, partially grounded in I∩R overlap but incorrectly assembled | Relation with correct components but wrong combination — "close but wrong" | **Penalized** (additive) |
| LF | Lost Focus | R − G | Relation in reference but NOT generated — critical omission | **Penalized** (additive) |

**Individual Factor Formulas (same structure as EHI + LH):**
```
EF = 3 × |I ∩ R ∩ G| / (|I| + |R| + |G|)
PH = 2 × |R ∩ G| / (|R| + |G|)
OF = 2 × |I ∩ G| / (|I| + |G|)
NH = (|G| - (|R ∩ G| + |I ∩ G| - |I ∩ R ∩ G|)) / |G|
LH = 2 × (|G| - |I ∩ R| - |R ∩ G| + |I ∩ R ∩ G|) / (|I| + |G|)
LF = (|R| - |I ∩ R| + |I ∩ R ∩ G|) / (|R| + |G|)
```

**Five Design Principles:**
1. **AND-logic reward** (EF × PH multiplicative): Both source-grounding AND reference-matching abstraction required
2. **OR-logic penalty** (additive penalties): Independent failure modes accumulate
3. **/2 faithfulness-first**: Reward capped at [0, 0.5], penalty range [0, 1.0] — penalty has 2× dynamic range
4. **/4 equal-weight**: Maximum entropy prior — no domain-specific claim about which failure is worst
5. **Base 1 threshold**: RHI > 1 = faithful, RHI < 1 = hallucinated, no calibration needed

**Properties:**
- Output range: [0, 1.5]
- Threshold: 1.0 (universal, calibration-free)
- Designed for human-interpretable evaluation (clear threshold)
- Why NOT softmax: linear more robust to noisy relation extraction; threshold property lost with softmax

---

## CRITICAL SEMANTIC CLARIFICATIONS

### What each factor ACTUALLY means (correcting common mix-ups):

| Factor | IS | IS NOT |
|--------|----|----|
| **NH (Negative Hallucination)** | Fabrication — generated something supported by NOTHING | Missing info (that's LF) |
| **LF (Lost Focus)** | Omission — reference had it, model dropped it | Hallucination (that's NH) |
| **PH (Positive Hallucination)** | Beneficial abstraction — both ref and gen agree on something NOT in source | Bad hallucination (that's NH) |
| **OF (Over Focus)** | Irrelevant extraction — source had it but ref deemed it unimportant | Hallucination (it IS grounded in source) |
| **LH (Lost Hallucination)** | Partially-grounded error — components exist but triple is wrong | Complete fabrication (that's NH) |
| **EF (Extractiveness)** | Faithful preservation — all three sets agree | Just copying (it's validated copying) |

### Domain-specific severity (CORRECTED):

| Domain | Most Critical Penalty Factor | Why |
|--------|------------------------------|-----|
| Medical | **NH** (fabrication) | Fabricating a drug-disease relation can kill patients |
| Legal | **NH** + **LH** (fabrication + partial-grounding error) | Fabricated citations + "close but wrong" legal reasoning |
| Financial | **LF** (omission) | Omitting material risk disclosures violates regulations |
| News (general) | Approximately uniform | No single failure mode dominates |

---

## Why LH Exists for Relations But Not Entities

An entity is atomic — "John" either appears or doesn't. A relation is composite — ("John", "approved", "budget") can be hallucinated where ALL individual entities are correct but the COMBINATION is fabricated. The model has "John", "approved", and "budget" grounded in source, but the triple connecting them is invented.

LH captures this "close but wrong" failure mode unique to composite structures.
