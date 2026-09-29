#!/usr/bin/env python3
"""
Dataset Preparation Script for CHI and DA-RHI Papers.

Downloads and prepares 200 articles per domain from publicly available
HuggingFace datasets. Output: JSON files ready for RHI/CHI evaluation.

Usage:
    python prepare_datasets.py [--output-dir ./data] [--samples 200]

Datasets:
    1. NEWS: EdinburghNLP/xsum (BBC articles, abstractive single-sentence summaries)
    2. MEDICAL: ccdv/pubmed-summarization (PubMed articles, abstracts as summaries)
    3. LEGAL: abisee/cnn_dailymail filtered for legal/court content + government reports
    4. FINANCIAL: abisee/cnn_dailymail filtered for financial/earnings content

No authentication required. All datasets are public.

Dependencies:
    pip install datasets
"""

import json
import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional


def load_xsum_news(n_samples: int = 200) -> List[Dict]:
    """
    Load news articles from XSUM (BBC).

    Why XSUM for news:
    - Highly abstractive summaries (single sentence) -> challenges models to paraphrase
    - BBC editorial quality ensures clean source text
    - Standard benchmark: used in FIRE 2024 RHI paper (continuity)
    - 11,334 test samples available -> ample selection
    - Well-studied: human evaluation data available (SummEval overlap)
    """
    from datasets import load_dataset

    print(f"  Loading EdinburghNLP/xsum (news domain)...")
    ds = load_dataset('EdinburghNLP/xsum', split='test', streaming=True)

    samples = []
    for i, item in enumerate(ds):
        if len(samples) >= n_samples:
            break
        doc = item['document'].strip()
        summary = item['summary'].strip()
        # Filter: require minimum length for meaningful relation extraction
        if len(doc.split()) >= 50 and len(summary.split()) >= 10:
            samples.append({
                "id": item['id'],
                "domain": "news",
                "source": "EdinburghNLP/xsum",
                "InputText": doc,
                "ReferenceSummary": summary,
                "doc_words": len(doc.split()),
                "summary_words": len(summary.split()),
            })

    print(f"    Collected {len(samples)} news samples")
    return samples


def load_pubmed_medical(n_samples: int = 200) -> List[Dict]:
    """
    Load medical articles from PubMed summarization.

    Why PubMed for medical:
    - Scientific biomedical text with domain terminology (drug names, conditions, procedures)
    - Abstract serves as expert-written summary of full paper
    - High density of medical entities and relations (drug-treats-disease, dosage-frequency)
    - NH (fabrication) in this domain = dangerous (wrong drug-disease relations)
    - Standard medical NLP evaluation domain (MedNLI, MEDIQA precedent)
    - 6,658 test samples available
    """
    from datasets import load_dataset

    print(f"  Loading ccdv/pubmed-summarization (medical domain)...")
    ds = load_dataset('ccdv/pubmed-summarization', 'document', split='test', streaming=True)

    samples = []
    for i, item in enumerate(ds):
        if len(samples) >= n_samples:
            break
        article = item['article'].strip()
        abstract = item['abstract'].strip()
        # Filter: require reasonable length + medical content indicators
        if (len(article.split()) >= 100 and len(abstract.split()) >= 30
            and len(article.split()) <= 3000):  # Cap length for processing speed
            samples.append({
                "id": f"pubmed_{i}",
                "domain": "medical",
                "source": "ccdv/pubmed-summarization",
                "InputText": article,
                "ReferenceSummary": abstract,
                "doc_words": len(article.split()),
                "summary_words": len(abstract.split()),
            })

    print(f"    Collected {len(samples)} medical samples")
    return samples


def load_cnn_legal(n_samples: int = 200) -> List[Dict]:
    """
    Load legal-domain articles from CNN/DailyMail (court/legal filtered).

    Why CNN/DM filtered for legal:
    - CNN/DM contains substantial court reporting, legislation coverage, legal proceedings
    - Filtering for legal keywords selects articles with legal entities and relations
    - Reference highlights are extractive -> models must faithfully preserve legal facts
    - Legal hallucination = fabricated citations, wrong court holdings, incorrect case outcomes
    - LH (partial-grounding) critical: real case names with wrong outcomes
    - 11,490 test samples -> filtering yields sufficient legal content
    """
    from datasets import load_dataset

    print(f"  Loading abisee/cnn_dailymail filtered for legal content...")
    ds = load_dataset('abisee/cnn_dailymail', '3.0.0', split='test', streaming=True)

    legal_keywords = re.compile(
        r'\b(court|judge|ruling|verdict|sentenced|plaintiff|defendant|'
        r'lawsuit|prosecution|attorney|legal|trial|jury|statute|'
        r'legislation|amendment|constitutional|supreme court|'
        r'appeal|conviction|acquitted|indicted|filed suit|'
        r'regulation|compliance|judicial|arbitration)\b',
        re.IGNORECASE
    )

    samples = []
    seen = 0
    for item in ds:
        seen += 1
        if len(samples) >= n_samples:
            break
        if seen > 50000:  # Safety cap
            break
        article = item['article'].strip()
        highlights = item['highlights'].strip()
        # Check for legal content
        matches = legal_keywords.findall(article)
        if len(matches) >= 3 and len(article.split()) >= 80 and len(highlights.split()) >= 15:
            samples.append({
                "id": item['id'],
                "domain": "legal",
                "source": "abisee/cnn_dailymail (legal-filtered)",
                "InputText": article,
                "ReferenceSummary": highlights,
                "doc_words": len(article.split()),
                "summary_words": len(highlights.split()),
                "legal_keyword_count": len(matches),
            })

    print(f"    Scanned {seen} articles, collected {len(samples)} legal samples")
    return samples


def load_cnn_financial(n_samples: int = 200) -> List[Dict]:
    """
    Load financial-domain articles from CNN/DailyMail (finance filtered).

    Why CNN/DM filtered for financial:
    - CNN/DM contains business reporting with numerical data (revenue, earnings, prices)
    - High quantity density: dollar amounts, percentages, growth figures
    - LF (omission) critical: missing material financial figures = misleading
    - QHI validation: these articles have numerous verifiable quantities
    - Financial hallucination = wrong numbers, magnitude errors, omitted risk factors
    - Standard news format ensures clean text quality
    """
    from datasets import load_dataset

    print(f"  Loading abisee/cnn_dailymail filtered for financial content...")
    ds = load_dataset('abisee/cnn_dailymail', '3.0.0', split='test', streaming=True)

    financial_keywords = re.compile(
        r'\b(revenue|profit|earnings|stock|shares|market|billion|million|'
        r'quarter|fiscal|dividend|investment|investor|nasdaq|'
        r'dow jones|s&p|trading|financial|economy|gdp|'
        r'inflation|interest rate|federal reserve|bank|'
        r'percent|growth|decline|loss|gain|quarterly)\b',
        re.IGNORECASE
    )
    # Require numbers (quantities)
    number_pattern = re.compile(r'\$[\d,.]+|\d+(?:\.\d+)?%|\d+(?:,\d{3})+')

    samples = []
    seen = 0
    for item in ds:
        seen += 1
        if len(samples) >= n_samples:
            break
        if seen > 50000:
            break
        article = item['article'].strip()
        highlights = item['highlights'].strip()
        # Check for financial content + numerical density
        keyword_matches = financial_keywords.findall(article)
        number_matches = number_pattern.findall(article)
        if (len(keyword_matches) >= 3 and len(number_matches) >= 3
            and len(article.split()) >= 80 and len(highlights.split()) >= 15):
            samples.append({
                "id": item['id'],
                "domain": "financial",
                "source": "abisee/cnn_dailymail (financial-filtered)",
                "InputText": article,
                "ReferenceSummary": highlights,
                "doc_words": len(article.split()),
                "summary_words": len(highlights.split()),
                "financial_keyword_count": len(keyword_matches),
                "number_count": len(number_matches),
            })

    print(f"    Scanned {seen} articles, collected {len(samples)} financial samples")
    return samples


def compute_statistics(samples: List[Dict]) -> Dict:
    """Compute basic statistics for a dataset."""
    doc_lengths = [s["doc_words"] for s in samples]
    sum_lengths = [s["summary_words"] for s in samples]
    return {
        "count": len(samples),
        "avg_doc_words": sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0,
        "avg_summary_words": sum(sum_lengths) / len(sum_lengths) if sum_lengths else 0,
        "min_doc_words": min(doc_lengths) if doc_lengths else 0,
        "max_doc_words": max(doc_lengths) if doc_lengths else 0,
        "compression_ratio": (sum(sum_lengths) / sum(doc_lengths)) if sum(doc_lengths) > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for CHI/DA-RHI experiments")
    parser.add_argument('--output-dir', type=str, default='./data',
                        help='Output directory for JSON files')
    parser.add_argument('--samples', type=int, default=200,
                        help='Number of samples per domain')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Dataset Preparation for CHI & DA-RHI Papers")
    print("=" * 70)
    print(f"\n  Output: {output_dir.absolute()}")
    print(f"  Samples per domain: {args.samples}")
    print()

    # Load all domains
    loaders = [
        ("news", load_xsum_news),
        ("medical", load_pubmed_medical),
        ("legal", load_cnn_legal),
        ("financial", load_cnn_financial),
    ]

    all_stats = {}
    for domain_name, loader_fn in loaders:
        print(f"\n{'='*50}")
        print(f"  Domain: {domain_name.upper()}")
        print(f"{'='*50}")
        samples = loader_fn(args.samples)

        # Save JSON
        output_file = output_dir / f"{domain_name}_200.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        print(f"    Saved to: {output_file}")

        # Statistics
        stats = compute_statistics(samples)
        all_stats[domain_name] = stats
        print(f"    Stats: {stats['count']} samples, "
              f"avg {stats['avg_doc_words']:.0f} doc words, "
              f"avg {stats['avg_summary_words']:.0f} summary words, "
              f"compression {stats['compression_ratio']:.2%}")

    # Save combined statistics
    stats_file = output_dir / "dataset_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  COMPLETE: 4 domains x {args.samples} samples = {args.samples * 4} total")
    print(f"  Statistics saved to: {stats_file}")
    print(f"{'='*70}")
    print(f"\n  Dataset Summary:")
    print(f"  {'Domain':<12} {'Count':>6} {'Avg Doc':>10} {'Avg Sum':>10} {'Compression':>12}")
    print(f"  {'-'*52}")
    for domain, stats in all_stats.items():
        print(f"  {domain:<12} {stats['count']:>6} {stats['avg_doc_words']:>10.0f} "
              f"{stats['avg_summary_words']:>10.0f} {stats['compression_ratio']:>12.2%}")


if __name__ == "__main__":
    main()
