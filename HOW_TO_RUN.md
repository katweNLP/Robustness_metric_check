# How to Run: CHI & DA-RHI Experiments

## Quick Start

```bash
cd C:\Users\pkatwe\PHD\PHD

# 1. Install dependencies (one-time)
pip install spacy sentence-transformers torch numpy scipy scikit-learn datasets rouge-score bert-score anthropic
python -m spacy download en_core_web_sm

# 2. Run CHI demo (no data needed, self-contained)
python run_demo_chi.py

# 3. Run DA-RHI demo (no data needed, self-contained)
python run_demo_darhi.py
```

---

## Directory Structure

```
C:\Users\pkatwe\PHD\PHD\
|
|-- src/                          # Core metric implementations
|   |-- ehi.py                    # Entity Hallucination Index (softmax, 5 factors)
|   |-- rhi_star.py               # Relation HI softmax variant (6 factors)
|   |-- rhi_core.py               # Published linear RHI + DA-RHI weight support
|   |-- qhi.py                    # Quantity Hallucination Index (5 factors)
|   |-- chi.py                    # Composite: harmonic mean of EHI + RHI* + QHI
|   |-- compute_baselines.py      # ROUGE, BERTScore, AlignScore proxy, SummaC proxy, MiniCheck
|   |-- da_rhi.py                 # Bayesian optimization, grid search, cross-domain transfer
|   |-- evaluation.py             # Spearman, Kendall, bootstrap CIs, Williams test
|
|-- data/                         # Downloaded datasets (800 articles)
|   |-- news_200.json
|   |-- medical_200.json
|   |-- legal_200.json
|   |-- financial_200.json
|
|-- results/                      # Computed experiment results
|   |-- summeval_system_level_full.json    # CHI system-level correlations
|   |-- summeval_full_comparison.json      # All baselines summary-level
|   |-- summeval_correct_chi.json          # CHI with actual RHI* (200 samples)
|   |-- additional_baselines.json          # MiniCheck N=1600 result
|   |-- news_generated.json                # Generated summaries per model
|   |-- (medical|legal|financial)_*.json   # Per-domain results
|
|-- paper/                        # LaTeX (symlinked from Gitea repos)
|
|-- prepare_datasets.py           # Download 4-domain datasets from HuggingFace
|-- run_experiments.py            # Full pipeline: generate + score + correlate
|-- run_summeval_experiment.py    # SummEval evaluation (with BERTScore)
|-- run_additional_baselines.py   # MiniCheck + Claude-judge
|
|-- FACTOR_REFERENCE.md           # Canonical factor definitions
|-- DATASETS.md                   # Dataset justification for papers
|-- Paper1_CHI_v7.pdf             # Latest CHI paper PDF
|-- Paper2_DARHI_v2.pdf           # Latest DA-RHI paper PDF
```

---

## Running Individual Metrics

### Compute CHI on any text triple:

```python
import sys
sys.path.insert(0, 'C:/Users/pkatwe/PHD/PHD')

from src.ehi import compute_ehi_factors, compute_ehi
from src.rhi_star import compute_rhi_star_factors, compute_rhi_star
from src.qhi import compute_qhi_factors, compute_qhi
from src.chi import compute_chi

source = "Apple reported revenue of $94.9 billion in Q3 2024."
reference = "Apple Q3 revenue was $94.9B."
generated = "Apple reported revenue of $94.9 billion."

# EHI (fast, ~0.1s)
ehi_f = compute_ehi_factors(source, reference, generated)
ehi = compute_ehi(ehi_f)
print(f"EHI: {ehi:.4f}")

# QHI (fast, ~0.1s)
qhi_f = compute_qhi_factors(source, reference, generated)
qhi = compute_qhi(qhi_f)
print(f"QHI: {qhi:.4f}")

# RHI* (slow, ~5-15s, needs sentence-transformer)
rhi_f = compute_rhi_star_factors(source, reference, generated)
rhi_star = compute_rhi_star(rhi_f)
print(f"RHI*: {rhi_star:.4f}")

# CHI composite
has_q = qhi_f.get("counts", {}).get("len_G", 0) > 0
chi = compute_chi(ehi, rhi_star, qhi, has_entities=True, has_quantities=has_q)
print(f"CHI: {chi:.4f}")  # High = faithful, Low = hallucinated
```

### Compute DA-RHI with custom weights:

```python
from src.rhi_core import score_text, compute_rhi

result = score_text(source, reference, generated)

# Uniform weights (published RHI)
print(f"RHI (uniform): {result['rhi']:.4f}")

# Domain-adaptive weights
medical_weights = {"w_of": 0.10, "w_nh": 0.45, "w_lh": 0.25, "w_lf": 0.20}
rhi_medical = compute_rhi(result['factors'], weights=medical_weights)
print(f"DA-RHI (medical): {rhi_medical:.4f}")

financial_weights = {"w_of": 0.12, "w_nh": 0.20, "w_lh": 0.25, "w_lf": 0.43}
rhi_financial = compute_rhi(result['factors'], weights=financial_weights)
print(f"DA-RHI (financial): {rhi_financial:.4f}")
```

### Compute baselines:

```python
from src.compute_baselines import (
    compute_rouge, compute_bertscore, 
    compute_alignscore_proxy, compute_summac_proxy,
    compute_entity_f1
)
from run_additional_baselines import compute_minicheck

rouge = compute_rouge(reference, generated)
print(f"ROUGE-L: {rouge['rougeL']:.4f}")

bertscore = compute_bertscore(reference, generated)
print(f"BERTScore: {bertscore:.4f}")

alignscore = compute_alignscore_proxy(source, generated)
print(f"AlignScore: {alignscore:.4f}")

summac = compute_summac_proxy(source, generated)
print(f"SummaC: {summac:.4f}")

minicheck = compute_minicheck(source, generated)
print(f"MiniCheck: {minicheck:.4f}")
```

---

## Running Full Experiments

### 1. Prepare datasets (download from HuggingFace):

```bash
python prepare_datasets.py --output-dir ./data --samples 200
```

### 2. Run 4-domain experiment (generate summaries + compute all metrics):

```bash
# Quick test (5 articles, ~5 min)
python run_experiments.py --max-samples 5 --domains news

# Full run (20 articles/domain, ~1 hour)
python run_experiments.py --max-samples 20

# Production (200 articles/domain, ~6 hours)
python run_experiments.py --max-samples 200
```

### 3. Run SummEval evaluation (with real human scores):

```bash
# Full 100 articles, EHI+QHI+ROUGE (fast, ~7 min)
python run_summeval_experiment.py --max-articles 100

# With RHI* (slow, ~60 min for 200 samples)
# Edit script to enable RHI* computation
```

### 4. Run MiniCheck baseline:

```bash
# Full 1600 samples (~3 hours on CPU)
python run_additional_baselines.py --max-articles 100 --skip-claude

# Quick test (10 articles, ~15 min)
python run_additional_baselines.py --max-articles 10 --skip-claude
```

### 5. Learn domain-adaptive weights (DA-RHI):

```python
from src.da_rhi import bayesian_optimization, cross_validate, cross_domain_transfer
from src.rhi_core import score_text

# Compute factors for your domain data
factors_list = []
human_scores = []
for article in your_annotated_data:
    result = score_text(article['source'], article['reference'], article['generated'])
    factors_list.append(result['factors'])
    human_scores.append(article['human_faithfulness_score'])

# Learn optimal weights
import numpy as np
result = bayesian_optimization(factors_list, np.array(human_scores), n_initial=200)
print(f"Optimal weights: {result['best_weights']}")
print(f"Best correlation: {result['best_rho']:.4f}")

# Cross-validate
cv_result = cross_validate(factors_list, np.array(human_scores), method='bayesian', n_folds=5)
print(f"CV avg test rho: {cv_result['avg_test_rho']:.4f}")
```

---

## Compiling Papers (LaTeX)

```bash
# CHI paper
cd /tmp/chi-push/paper    # or clone from Gitea
pdflatex main.tex
pdflatex main.tex         # second pass for references

# DA-RHI paper
cd /tmp/da-rhi-push/paper
pdflatex main.tex
pdflatex main.tex
```

Or use the Gitea repos directly:
```bash
git clone http://invp360dock01.informatica.com:3000/praveenkatwe/CHI-Composite-Hallucination-Index.git
git clone http://invp360dock01.informatica.com:3000/praveenkatwe/DA-RHI-Domain-Adaptive-Hallucination.git
```

---

## Modifying and Re-running

### To add a new baseline metric:

1. Add function to `src/compute_baselines.py`:
```python
def compute_your_metric(source, reference, generated):
    # Your implementation
    return score  # float, higher = more faithful
```

2. Add to `run_experiments.py` in `compute_baselines()` function
3. Re-run: `python run_experiments.py --max-samples 20 --skip-generation`

### To add a new domain:

1. Add loader function to `prepare_datasets.py`
2. Run: `python prepare_datasets.py`
3. Re-run experiments: `python run_experiments.py --domains news medical legal financial YOUR_DOMAIN`

### To change CHI fusion method:

Edit `src/chi.py` — the `compute_chi()` function. Replace harmonic mean with:
```python
# Arithmetic mean
chi = (ehi + rhi_star + qhi) / 3

# Geometric mean
chi = (ehi * rhi_star * qhi) ** (1/3)

# Weighted
chi = 0.4*ehi + 0.4*rhi_star + 0.2*qhi
```

### To update paper with new results:

1. Run experiments, get new numbers
2. Edit LaTeX: `C:\tmp\chi-paper\main.tex` or `C:\tmp\darhi-paper\main.tex`
3. Compile: `pdflatex main.tex && pdflatex main.tex`
4. Push: `cd /tmp/chi-push && git add -A && git commit -m "msg" && git push origin main`

---

## Key Configuration

| Setting | Value | Where |
|---------|-------|-------|
| spaCy model | en_core_web_sm | Used by EHI, RHI*, QHI |
| Sentence-transformer | all-MiniLM-L6-v2 | Used by RHI* for SVO matching |
| NLI model | cross-encoder/nli-deberta-v3-base | Used by AlignScore, SummaC, MiniCheck |
| NLI label order | [contradiction=0, entailment=1, neutral=2] | Critical! Index 1 = entailment |
| Confidence threshold | 0.4 | SVO extraction filter |
| Quantity epsilon | 0.05 (5%) | Tolerance for quantity matching |
| CHI direction | High = faithful, Low = hallucinated | Consistent across all sub-indices |
| RHI threshold | 1.0 (linear version) | Above = faithful, Below = hallucinated |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: src` | Run from `C:\Users\pkatwe\PHD\PHD` directory |
| Slow RHI* (~15s/doc) | Skip for quick tests; use `compute_chi(ehi, 0.333, qhi)` as approximation |
| `CUDA not available` | Normal — all code runs on CPU. GPU only speeds up BERTScore/sentence-transformer |
| MiniCheck slow (~8s/doc) | Sentence-level NLI is inherently expensive; no way to speed up without GPU |
| Output buffered (no prints) | Use `python -u script.py` for unbuffered output |
| LaTeX `[?]` citations | Run pdflatex twice; check bibitem keys match \cite keys |
