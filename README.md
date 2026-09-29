# Systematic Evaluation of Decomposable Hallucination Metrics: EHI, RHI, and QHI

**Paper**: *Systematic Evaluation of Decomposable Hallucination Metrics: EHI, RHI, and QHI for Robust Comparison Across Tasks and Domains*

**Authors**: Praveenkumar Katwe, Rakesh Chandra Balabantaray, Kali Prasad Vittala

## Overview

This repository contains the code, experiments, and results for a systematic evaluation of the CHI (Composite Hallucination Index) metric family across five dimensions:

1. **Human Correlation** — SummEval (N=1,600) at sample and system level
2. **Binary Factuality** — Binarized consistency annotations (N=500)
3. **Cross-Task Transfer** — QA hallucination detection on HaluEval (N=500)
4. **Robustness** — Entity-swap and relation-inversion perturbations
5. **Computational Efficiency** — Per-sample timing benchmarks

## Metrics

| Metric | Description | File |
|--------|-------------|------|
| **EHI** | Entity Hallucination Index (5 softmax factors) | `src/ehi.py` |
| **RHI*** | Relation Hallucination Index (6 softmax factors, SVO triples) | `src/rhi_star.py` |
| **QHI** | Quantity Hallucination Index (5 softmax factors) | `src/qhi.py` |
| **CHI** | Composite = HarmonicMean(EHI, RHI*, QHI) | `src/chi.py` |
| **CHI-RF** | Reference-free variant (experimental) | `src/chi_rf.py` |

## Key Results

| Finding | Value |
|---------|-------|
| CHI system-level avg (SummEval) | **0.656** (ties BARTScore, beats all ROUGE) |
| CHI on QA (HaluEval) | **0.869 AUC** (BERTScore = 0.504) |
| EHI entity-swap sensitivity | **-0.102** (BERTScore = -0.012) |
| RHI* optimized speed | **136ms/sample** (40x speedup) |
| Full CHI pipeline | **464ms/sample** (7.5x faster than BERTScore) |

## Quick Start

```bash
pip install spacy sentence-transformers torch numpy scipy scikit-learn datasets rouge-score bert-score
python -m spacy download en_core_web_sm

# Run all experiments
python run_summeval_experiment.py --max-articles 100
python run_bartscore_baseline.py --max-articles 100
python run_aggrefact_experiment.py --max-samples 500
python run_halueval_experiment.py --max-samples 500
python run_robustness_test.py --max-samples 200
python run_efficiency_benchmark.py

# Generate interactive HTML report
python generate_html_report.py
# Open results/chi_evaluation_report.html
```

## Repository Structure

```
src/                          # Metric implementations
  ehi.py                      # Entity Hallucination Index
  rhi_star.py                 # Relation Hallucination Index (softmax)
  rhi_core.py                 # RHI (original linear, FIRE 2024)
  qhi.py                      # Quantity Hallucination Index
  chi.py                      # Composite Hallucination Index
  chi_rf.py                   # Reference-free CHI (experimental)
  ehi_enhanced.py             # Domain-grounded EHI (UMLS)
  compute_baselines.py        # ROUGE, BERTScore, AlignScore, SummaC, Entity F1

run_*.py                      # Experiment scripts
generate_html_report.py       # Interactive HTML dashboard generator

results/                      # All experiment results (JSON)
  chi_evaluation_report.html  # Interactive dashboard

paper/                        # LaTeX paper + compiled PDF
  main.tex
  references.bib
  paper.pdf
```

## Citation

```bibtex
@article{katwe2026systematic,
  title={Systematic Evaluation of Decomposable Hallucination Metrics: EHI, RHI, and QHI for Robust Comparison Across Tasks and Domains},
  author={Katwe, Praveenkumar and Balabantaray, Rakesh Chandra and Vittala, Kali Prasad},
  year={2026}
}
```
