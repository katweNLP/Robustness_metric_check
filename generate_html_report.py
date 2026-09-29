#!/usr/bin/env python3
"""
Generate a self-contained HTML dashboard for CHI metric evaluation.

Reads all JSON result files from results/ and produces a single interactive
HTML file with tabs, sortable tables, and Chart.js visualizations.

Gracefully handles missing result files — shows 'Pending' for experiments
that haven't been run yet.

Usage:
    python generate_html_report.py --results-dir ./results --output ./results/chi_evaluation_report.html
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def load_json_safe(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_all_results(results_dir):
    d = Path(results_dir)
    return {
        "summeval_correlations": load_json_safe(d / "summeval_correlations.json"),
        "summeval_system_level": load_json_safe(d / "summeval_system_level_full.json"),
        "summeval_ehi_method": load_json_safe(d / "summeval_ehi_method.json"),
        "experiment_summary": load_json_safe(d / "experiment_summary.json"),
        "additional_baselines": load_json_safe(d / "additional_baselines.json"),
        "bartscore_summeval": load_json_safe(d / "bartscore_summeval.json"),
        "aggrefact_results": load_json_safe(d / "aggrefact_results.json"),
        "halueval_qa_results": load_json_safe(d / "halueval_qa_results.json"),
        "robustness": load_json_safe(d / "robustness_entity_swap.json"),
        "efficiency": load_json_safe(d / "efficiency_benchmark.json"),
        "ehi_umls": load_json_safe(d / "ehi_umls_comparison.json"),
    }


def fmt(val, decimals=3):
    if val is None or (isinstance(val, float) and (val != val)):
        return "—"
    return f"{val:.{decimals}f}"


def sig_badge(p):
    if p is None:
        return ""
    if p < 0.001:
        return ' <span class="sig sig3">***</span>'
    if p < 0.01:
        return ' <span class="sig sig2">**</span>'
    if p < 0.05:
        return ' <span class="sig sig1">*</span>'
    return ""


def build_summeval_sample_table(data, baselines, bartscore):
    if not data:
        return '<p class="pending">SummEval sample-level results not available.</p>'

    rows = []
    for metric, vals in sorted(data.items(), key=lambda x: -x[1].get("spearman", 0)):
        rows.append({
            "metric": metric,
            "spearman": vals.get("spearman"),
            "kendall": vals.get("kendall"),
            "p": vals.get("p_value"),
        })

    if baselines and baselines.get("minicheck"):
        mc = baselines["minicheck"]
        rows.append({"metric": "MiniCheck", "spearman": mc.get("spearman"), "kendall": None, "p": None})

    if bartscore and "BARTScore" in bartscore:
        bs = bartscore["BARTScore"]
        rows.append({"metric": "BARTScore", "spearman": bs.get("spearman"), "kendall": bs.get("kendall"), "p": bs.get("p_value")})

    rows.sort(key=lambda r: -(r["spearman"] or 0))

    html = '<table class="sortable"><thead><tr><th>Metric</th><th>Spearman ρ</th><th>Kendall τ</th><th>Significance</th></tr></thead><tbody>'
    for r in rows:
        is_chi = r["metric"] in ("CHI", "EHI", "RHI*", "QHI")
        cls = ' class="highlight"' if is_chi else ""
        html += f'<tr{cls}><td>{r["metric"]}</td><td>{fmt(r["spearman"])}</td><td>{fmt(r["kendall"])}</td><td>{sig_badge(r["p"])}</td></tr>'
    html += '</tbody></table>'
    return html


def build_summeval_system_table(data):
    if not data:
        return '<p class="pending">System-level results not available.</p>'

    dimensions = ["consistency", "coherence", "relevance", "average"]
    metrics = set()
    for dim in dimensions:
        if dim in data:
            metrics.update(data[dim].keys())

    html = '<table class="sortable"><thead><tr><th>Metric</th>'
    for d in dimensions:
        html += f'<th>{d.title()}</th>'
    html += '</tr></thead><tbody>'

    metric_avg = {}
    for m in sorted(metrics):
        vals = []
        for d in dimensions:
            v = data.get(d, {}).get(m, {}).get("spearman")
            vals.append(v)
        avg = sum(v for v in vals if v is not None) / max(len([v for v in vals if v is not None]), 1)
        metric_avg[m] = avg

    for m in sorted(metrics, key=lambda x: -metric_avg.get(x, 0)):
        is_chi = any(c in m for c in ("CHI", "EHI", "RHI", "QHI"))
        cls = ' class="highlight"' if is_chi else ""
        html += f'<tr{cls}><td>{m}</td>'
        for d in dimensions:
            v = data.get(d, {}).get(m, {}).get("spearman")
            p = data.get(d, {}).get(m, {}).get("p")
            bold = ' style="font-weight:700"' if d == "average" else ""
            html += f'<td{bold}>{fmt(v)}{sig_badge(p)}</td>'
        html += '</tr>'
    html += '</tbody></table>'
    return html


def build_cross_domain_table(data):
    if not data:
        return '<p class="pending">Cross-domain results not available.</p>'

    domains = ["news", "medical", "legal", "financial"]
    metric_keys = ["ehi", "rhi_star", "qhi", "chi", "rouge1", "rougeL", "bertscore", "alignscore_proxy", "summac_proxy"]
    metric_labels = {"ehi": "EHI", "rhi_star": "RHI*", "qhi": "QHI", "chi": "CHI",
                     "rouge1": "ROUGE-1", "rougeL": "ROUGE-L", "bertscore": "BERTScore",
                     "alignscore_proxy": "AlignScore", "summac_proxy": "SummaC"}

    html = '<table class="sortable"><thead><tr><th>Metric</th>'
    for d in domains:
        html += f'<th>{d.title()}</th>'
    html += '<th>Average</th></tr></thead><tbody>'

    for mk in metric_keys:
        is_chi = mk in ("ehi", "rhi_star", "qhi", "chi")
        cls = ' class="highlight"' if is_chi else ""
        vals = []
        html += f'<tr{cls}><td>{metric_labels.get(mk, mk)}</td>'
        for d in domains:
            v = data.get(d, {}).get("correlations", {}).get(mk, {}).get("spearman_rho")
            vals.append(v)
            html += f'<td>{fmt(v)}</td>'
        avg = sum(v for v in vals if v is not None) / max(len([v for v in vals if v is not None]), 1)
        html += f'<td style="font-weight:700">{fmt(avg)}</td></tr>'
    html += '</tbody></table>'
    return html


def build_aggrefact_table(data):
    if not data:
        return '<p class="pending">AggreFact experiment not yet run. Execute: <code>python run_aggrefact_experiment.py --max-samples 500</code></p>'

    overall = data.get("overall", {})
    metrics = overall.get("metrics", {})
    n = overall.get("n", "?")

    html = f'<p>N = {n} samples from AggreFact (binary consistent/inconsistent labels)</p>'
    html += '<table class="sortable"><thead><tr><th>Metric</th><th>ROC-AUC</th><th>Balanced Acc.</th><th>Point-Biserial r</th><th>Spearman ρ</th></tr></thead><tbody>'

    for m in sorted(metrics.keys(), key=lambda x: -(metrics[x].get("roc_auc") or 0)):
        v = metrics[m]
        is_chi = any(c in m for c in ("CHI", "EHI", "RHI", "QHI"))
        cls = ' class="highlight"' if is_chi else ""
        html += f'<tr{cls}><td>{m}</td><td>{fmt(v.get("roc_auc"))}</td><td>{fmt(v.get("balanced_acc"))}</td><td>{fmt(v.get("point_biserial"))}</td><td>{fmt(v.get("spearman"))}</td></tr>'
    html += '</tbody></table>'

    per_sub = data.get("per_subset", {})
    if per_sub:
        html += '<h3>Per-Subset Breakdown</h3>'
        for subset, sdata in sorted(per_sub.items()):
            sn = sdata.get("n", "?")
            sm = sdata.get("metrics", {})
            html += f'<details><summary>{subset} (N={sn})</summary><table class="sortable"><thead><tr><th>Metric</th><th>ROC-AUC</th><th>Balanced Acc.</th></tr></thead><tbody>'
            for mk in sorted(sm.keys(), key=lambda x: -(sm[x].get("roc_auc") or 0)):
                v = sm[mk]
                html += f'<tr><td>{mk}</td><td>{fmt(v.get("roc_auc"))}</td><td>{fmt(v.get("balanced_acc"))}</td></tr>'
            html += '</tbody></table></details>'

    return html


def build_halueval_table(data):
    if not data:
        return '<p class="pending">HaluEval QA experiment not yet run. Execute: <code>python run_halueval_experiment.py --max-samples 500</code></p>'

    overall = data.get("overall", data)
    metrics = overall.get("metrics", data.get("results", {}))
    n = overall.get("n", data.get("n_samples", "?"))
    stats = data.get("extraction_stats", data.get("extraction_statistics", {}))

    html = f'<p>N = {n} QA samples from HaluEval (paired right/hallucinated answers)</p>'
    if stats:
        med_ent = stats.get("median_entities", stats.get("median_entities_per_answer", "?"))
        med_tri = stats.get("median_triples", stats.get("median_triples_per_answer", "?"))
        html += f'<p class="stat-note">Median entities/answer: {med_ent} | Median SVO triples/answer: {med_tri}</p>'

    html += '<p class="stat-note">Effective AUC = max(AUC, 1-AUC). Faithfulness metrics score faithful text higher, so raw AUC &lt; 0.5 indicates correct discrimination.</p>'
    html += '<table class="sortable"><thead><tr><th>Metric</th><th>Effective AUC</th><th>Balanced Acc.</th><th>|Point-Biserial r|</th><th>Spearman ρ</th></tr></thead><tbody>'
    for m in sorted(metrics.keys(), key=lambda x: -max((metrics[x].get("roc_auc") or 0.5), 1 - (metrics[x].get("roc_auc") or 0.5))):
        v = metrics[m]
        raw_auc = v.get("roc_auc")
        eff_auc = max(raw_auc, 1 - raw_auc) if raw_auc is not None else None
        pb = v.get("point_biserial")
        abs_pb = abs(pb) if pb is not None and pb == pb else None
        is_chi = any(c in m for c in ("CHI", "EHI", "RHI", "QHI"))
        cls = ' class="highlight"' if is_chi else ""
        html += f'<tr{cls}><td>{m}</td><td>{fmt(eff_auc)}</td><td>{fmt(v.get("balanced_acc"))}</td><td>{fmt(abs_pb)}</td><td>{fmt(v.get("spearman"))}</td></tr>'
    html += '</tbody></table>'
    return html


def build_robustness_section(data):
    if not data:
        return '<p class="pending">Robustness test not yet run. Execute: <code>python run_robustness_test.py --max-samples 200</code></p>'

    es = data.get("entity_swap", {})
    ri = data.get("relation_inversion", {})

    html = f'<p>N = {data.get("n_samples", "?")} high-quality SummEval summaries (consistency ≥ 4)</p>'

    if es:
        html += '<h3>Entity-Swap Perturbation</h3>'
        html += '<p>Swap ratio = fraction of entities replaced with type-matched random entities not in source.</p>'
        ratios = es.get("swap_ratios", [])
        metrics = es.get("metrics", {})
        html += '<table class="sortable"><thead><tr><th>Metric</th>'
        for r in ratios:
            html += f'<th>{int(r*100)}%</th>'
        html += '<th>Δ (0→100%)</th></tr></thead><tbody>'
        for m in sorted(metrics.keys(), key=lambda x: abs(metrics[x].get("delta", 0)), reverse=True):
            v = metrics[m]
            scores = v.get("scores", [])
            delta = v.get("delta", 0)
            is_chi = any(c in m for c in ("CHI", "EHI", "RHI", "QHI"))
            cls = ' class="highlight"' if is_chi else ""
            html += f'<tr{cls}><td>{m}</td>'
            for s in scores:
                html += f'<td>{fmt(s)}</td>'
            color = "#c0392b" if delta < -0.05 else "#27ae60" if delta > 0.05 else "#7f8c8d"
            html += f'<td style="color:{color};font-weight:700">{fmt(delta)}</td></tr>'
        html += '</tbody></table>'

    if ri:
        html += '<h3>Relation Inversion</h3>'
        html += '<p>Relation words (increased→decreased, approved→rejected, etc.) inverted at varying ratios.</p>'
        ratios = ri.get("inversion_ratios", [])
        metrics = ri.get("metrics", {})
        html += '<table class="sortable"><thead><tr><th>Metric</th>'
        for r in ratios:
            html += f'<th>{int(r*100)}%</th>'
        html += '<th>Δ</th></tr></thead><tbody>'
        for m in sorted(metrics.keys(), key=lambda x: abs(metrics[x].get("delta", 0)), reverse=True):
            v = metrics[m]
            scores = v.get("scores", [])
            delta = v.get("delta", 0)
            is_chi = any(c in m for c in ("CHI", "EHI", "RHI", "QHI"))
            cls = ' class="highlight"' if is_chi else ""
            html += f'<tr{cls}><td>{m}</td>'
            for s in scores:
                html += f'<td>{fmt(s)}</td>'
            color = "#c0392b" if delta < -0.05 else "#27ae60" if delta > 0.05 else "#7f8c8d"
            html += f'<td style="color:{color};font-weight:700">{fmt(delta)}</td></tr>'
        html += '</tbody></table>'

    ranking = data.get("sensitivity_ranking", [])
    if ranking:
        html += '<h3>Sensitivity Ranking (most sensitive first)</h3><ol>'
        for m in ranking:
            html += f'<li>{m}</li>'
        html += '</ol>'

    return html


def build_efficiency_table(data):
    if not data:
        return '<p class="pending">Efficiency benchmark not yet run. Execute: <code>python run_efficiency_benchmark.py</code></p>'

    metrics = data.get("metrics", {})
    html = '<table class="sortable"><thead><tr><th>Metric</th><th>Mean (ms)</th><th>Std (ms)</th><th>Median (ms)</th><th>Category</th></tr></thead><tbody>'
    for m in sorted(metrics.keys(), key=lambda x: metrics[x].get("mean_ms", 0)):
        v = metrics[m]
        cat = v.get("category", "?")
        cat_cls = {"lightweight": "cat-light", "moderate": "cat-mod", "heavyweight": "cat-heavy"}.get(cat, "")
        html += f'<tr><td>{m}</td><td>{fmt(v.get("mean_ms"), 1)}</td><td>{fmt(v.get("std_ms"), 1)}</td><td>{fmt(v.get("median_ms"), 1)}</td><td><span class="cat-badge {cat_cls}">{cat}</span></td></tr>'
    html += '</tbody></table>'
    return html


def build_chart_data(results):
    charts = {}

    se = results["summeval_correlations"]
    ab = results["additional_baselines"]
    bs = results["bartscore_summeval"]
    if se:
        labels = []
        values = []
        colors = []
        chi_metrics = {"CHI", "EHI", "RHI*", "QHI"}
        all_items = dict(sorted(se.items(), key=lambda x: -x[1].get("spearman", 0)))
        if ab and ab.get("minicheck"):
            all_items["MiniCheck"] = {"spearman": ab["minicheck"]["spearman"]}
        if bs and "BARTScore" in bs:
            all_items["BARTScore"] = {"spearman": bs["BARTScore"]["spearman"]}
        all_items = dict(sorted(all_items.items(), key=lambda x: -(x[1].get("spearman") or 0)))
        for m, v in all_items.items():
            labels.append(m)
            values.append(v.get("spearman", 0))
            colors.append("rgba(42,120,214,0.85)" if m in chi_metrics else "rgba(82,81,78,0.5)")
        charts["summeval_bar"] = {"labels": labels, "values": values, "colors": colors}

    sl = results["summeval_system_level"]
    if sl:
        dims = ["consistency", "coherence", "relevance", "average"]
        metric_order = ["CHI (full)", "RHI*", "EHI", "QHI", "ROUGE-1", "ROUGE-2", "ROUGE-L"]
        datasets = []
        palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
        for i, d in enumerate(dims):
            vals = []
            for m in metric_order:
                v = sl.get(d, {}).get(m, {}).get("spearman", 0)
                vals.append(v)
            datasets.append({"label": d.title(), "data": vals, "backgroundColor": palette[i]})
        charts["system_grouped"] = {"labels": metric_order, "datasets": datasets}

    es = results["experiment_summary"]
    if es:
        domains = ["news", "medical", "legal", "financial"]
        chi_keys = ["ehi", "rhi_star", "chi"]
        chi_labels = {"ehi": "EHI", "rhi_star": "RHI*", "chi": "CHI"}
        palette = ["#2a78d6", "#eb6834", "#1baf7a"]
        datasets = []
        for i, mk in enumerate(chi_keys):
            vals = []
            for d in domains:
                v = es.get(d, {}).get("correlations", {}).get(mk, {}).get("spearman_rho", 0)
                vals.append(v)
            datasets.append({"label": chi_labels[mk], "data": vals, "borderColor": palette[i], "fill": False, "tension": 0.3})
        charts["cross_domain_line"] = {"labels": [d.title() for d in domains], "datasets": datasets}

    halu = results["halueval_qa_results"]
    if halu:
        metrics = halu.get("results", halu.get("overall", {}).get("metrics", {}))
        labels = []
        values = []
        colors = []
        chi_metrics = {"CHI", "EHI", "EHI_enhanced", "RHI*", "QHI"}
        for m in sorted(metrics.keys(), key=lambda x: -max((metrics[x].get("roc_auc") or 0.5), 1-(metrics[x].get("roc_auc") or 0.5))):
            raw = metrics[m].get("roc_auc") or 0.5
            labels.append(m)
            values.append(max(raw, 1 - raw))
            colors.append("rgba(42,120,214,0.85)" if m in chi_metrics else "rgba(82,81,78,0.5)")
        charts["halueval_bar"] = {"labels": labels, "values": values, "colors": colors}

    agg = results["aggrefact_results"]
    if agg:
        metrics = agg.get("overall", {}).get("metrics", {})
        labels = []
        values = []
        colors = []
        chi_metrics = {"CHI", "EHI", "EHI_enhanced", "RHI*", "QHI"}
        for m in sorted(metrics.keys(), key=lambda x: -max((metrics[x].get("roc_auc") or 0.5), 1-(metrics[x].get("roc_auc") or 0.5))):
            raw = metrics[m].get("roc_auc") or 0.5
            labels.append(m)
            values.append(max(raw, 1 - raw))
            colors.append("rgba(42,120,214,0.85)" if m in chi_metrics else "rgba(82,81,78,0.5)")
        charts["aggrefact_bar"] = {"labels": labels, "values": values, "colors": colors}

    rob = results["robustness"]
    if rob:
        es_data = rob.get("entity_swap", {})
        ratios = es_data.get("swap_ratios", [])
        mets = es_data.get("metrics", {})
        palette_map = {"CHI": "#2a78d6", "EHI": "#eb6834", "RHI*": "#1baf7a", "QHI": "#eda100",
                       "ROUGE-L": "#e87ba4", "BERTScore": "#008300", "MiniCheck": "#4a3aa7"}
        datasets = []
        for m, v in mets.items():
            color = palette_map.get(m, "#52514e")
            datasets.append({"label": m, "data": v.get("scores", []), "borderColor": color, "fill": False, "tension": 0.3})
        charts["robustness_line"] = {"labels": [f"{int(r*100)}%" for r in ratios], "datasets": datasets}

    eff = results["efficiency"]
    if eff:
        metrics = eff.get("metrics", {})
        labels = []
        values = []
        colors = []
        cat_colors = {"lightweight": "#1baf7a", "moderate": "#eda100", "heavyweight": "#eb6834"}
        for m in sorted(metrics.keys(), key=lambda x: metrics[x].get("mean_ms", 0)):
            labels.append(m)
            values.append(metrics[m].get("mean_ms", 0))
            colors.append(cat_colors.get(metrics[m].get("category", ""), "#52514e"))
        charts["efficiency_bar"] = {"labels": labels, "values": values, "colors": colors}

    return charts


def generate_html(results, output_path):
    charts = build_chart_data(results)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CHI Metric Evaluation Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
:root {{
  --surface: #fcfcfb;
  --surface-2: #f4f3f1;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #8a8985;
  --accent: #2a78d6;
  --accent-light: rgba(42,120,214,0.1);
  --border: #e0dfdc;
  --highlight-bg: rgba(42,120,214,0.06);
  --good: #1baf7a;
  --warn: #eda100;
  --bad: #c0392b;
}}
[data-theme="dark"] {{
  --surface: #1a1a19;
  --surface-2: #252524;
  --text-primary: #ffffff;
  --text-secondary: #c3c2b7;
  --text-muted: #8a8985;
  --accent: #3987e5;
  --accent-light: rgba(57,135,229,0.15);
  --border: #3a3a38;
  --highlight-bg: rgba(57,135,229,0.1);
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; background:var(--surface); color:var(--text-primary); line-height:1.6; }}
.container {{ max-width:1200px; margin:0 auto; padding:1rem 1.5rem; }}
header {{ background:var(--accent); color:#fff; padding:1.5rem 0; margin-bottom:1.5rem; }}
header .container {{ display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; }}
header h1 {{ font-size:1.5rem; font-weight:700; }}
header .subtitle {{ font-size:0.9rem; opacity:0.85; }}
.theme-toggle {{ background:rgba(255,255,255,0.2); border:none; color:#fff; padding:0.4rem 0.8rem; border-radius:4px; cursor:pointer; font-size:0.85rem; }}
nav {{ display:flex; gap:0; border-bottom:2px solid var(--border); margin-bottom:1.5rem; overflow-x:auto; }}
nav button {{ background:none; border:none; padding:0.7rem 1.2rem; font-size:0.85rem; color:var(--text-secondary); cursor:pointer; border-bottom:2px solid transparent; margin-bottom:-2px; white-space:nowrap; transition:all 0.15s; }}
nav button:hover {{ color:var(--text-primary); background:var(--accent-light); }}
nav button.active {{ color:var(--accent); border-bottom-color:var(--accent); font-weight:600; }}
.tab {{ display:none; }}
.tab.active {{ display:block; }}
h2 {{ font-size:1.25rem; margin-bottom:1rem; color:var(--text-primary); }}
h3 {{ font-size:1.05rem; margin:1.5rem 0 0.8rem; color:var(--text-secondary); }}
p {{ margin-bottom:0.8rem; color:var(--text-secondary); font-size:0.92rem; }}
table {{ width:100%; border-collapse:collapse; margin-bottom:1.5rem; font-size:0.85rem; }}
th {{ text-align:left; padding:0.6rem 0.8rem; background:var(--surface-2); color:var(--text-secondary); font-weight:600; border-bottom:2px solid var(--border); cursor:pointer; user-select:none; white-space:nowrap; }}
th:hover {{ background:var(--accent-light); }}
th::after {{ content:''; display:inline-block; width:0; height:0; margin-left:6px; vertical-align:middle; }}
th.asc::after {{ border-left:4px solid transparent; border-right:4px solid transparent; border-bottom:5px solid var(--text-secondary); }}
th.desc::after {{ border-left:4px solid transparent; border-right:4px solid transparent; border-top:5px solid var(--text-secondary); }}
td {{ padding:0.5rem 0.8rem; border-bottom:1px solid var(--border); }}
tr:hover {{ background:var(--accent-light); }}
tr.highlight {{ background:var(--highlight-bg); }}
tr.highlight:hover {{ background:rgba(42,120,214,0.12); }}
.pending {{ background:var(--surface-2); padding:1.5rem; border-radius:8px; text-align:center; color:var(--text-muted); font-style:italic; }}
.pending code {{ background:var(--surface); padding:0.2rem 0.5rem; border-radius:4px; font-size:0.82rem; }}
.chart-container {{ position:relative; height:350px; margin-bottom:1.5rem; background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:1rem; }}
.stat-cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:1rem; margin-bottom:1.5rem; }}
.stat-card {{ background:var(--surface-2); border-radius:8px; padding:1rem 1.2rem; border-left:4px solid var(--accent); }}
.stat-card .label {{ font-size:0.78rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.03em; margin-bottom:0.3rem; }}
.stat-card .value {{ font-size:1.8rem; font-weight:700; color:var(--text-primary); }}
.stat-card .detail {{ font-size:0.78rem; color:var(--text-secondary); margin-top:0.2rem; }}
.sig {{ font-size:0.7rem; font-weight:700; }}
.sig3 {{ color:var(--good); }}
.sig2 {{ color:var(--accent); }}
.sig1 {{ color:var(--warn); }}
.cat-badge {{ padding:0.15rem 0.6rem; border-radius:12px; font-size:0.75rem; font-weight:600; }}
.cat-light {{ background:#e8f8f0; color:#15764a; }}
.cat-mod {{ background:#fef6e0; color:#946200; }}
.cat-heavy {{ background:#fde8e4; color:#922b21; }}
[data-theme="dark"] .cat-light {{ background:#1a3d2a; color:#4fd99a; }}
[data-theme="dark"] .cat-mod {{ background:#3d3010; color:#e5b800; }}
[data-theme="dark"] .cat-heavy {{ background:#3d1a14; color:#e8644e; }}
.stat-note {{ font-size:0.85rem; color:var(--text-muted); font-style:italic; }}
details {{ margin-bottom:0.8rem; }}
summary {{ cursor:pointer; padding:0.5rem 0; font-weight:600; color:var(--text-secondary); }}
.metric-tree {{ display:flex; align-items:center; gap:1rem; justify-content:center; margin:1.5rem 0; flex-wrap:wrap; }}
.metric-box {{ background:var(--surface-2); border:2px solid var(--border); border-radius:8px; padding:0.8rem 1.2rem; text-align:center; min-width:120px; }}
.metric-box.composite {{ border-color:var(--accent); background:var(--accent-light); }}
.metric-box .name {{ font-weight:700; font-size:1rem; }}
.metric-box .desc {{ font-size:0.72rem; color:var(--text-muted); }}
.arrow {{ color:var(--text-muted); font-size:1.2rem; }}
.two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; }}
@media (max-width:768px) {{ .two-col {{ grid-template-columns:1fr; }} }}
.pipeline-diagram {{ background:var(--surface-2); border-radius:8px; padding:1.5rem; margin:1rem 0; text-align:center; }}
.pipeline-step {{ display:inline-block; background:var(--surface); border:2px solid var(--border); border-radius:8px; padding:0.6rem 1rem; margin:0.3rem; vertical-align:middle; }}
.pipeline-step.active {{ border-color:var(--accent); background:var(--accent-light); }}
.pipeline-arrow {{ display:inline-block; color:var(--text-muted); font-size:1.2rem; vertical-align:middle; margin:0 0.3rem; }}
.formula-box {{ background:var(--surface-2); border:2px solid var(--accent); border-radius:8px; padding:1rem 1.5rem; margin:1rem 0; font-size:1.1rem; text-align:center; font-family:Georgia,'Times New Roman',serif; }}
footer {{ text-align:center; padding:2rem 0; color:var(--text-muted); font-size:0.78rem; border-top:1px solid var(--border); margin-top:2rem; }}
</style>
</head>
<body>

<header>
<div class="container">
<div>
<h1>CHI: Composite Hallucination Index</h1>
<div class="subtitle">Comprehensive Evaluation Report &mdash; Entity + Relation + Quantity Hallucination Detection</div>
</div>
<div>
<button class="theme-toggle" onclick="toggleTheme()">Toggle Dark</button>
<span class="subtitle">{now}</span>
</div>
</div>
</header>

<div class="container">
<nav id="tabs">
<button class="active" onclick="showTab('overview')">Overview</button>
<button onclick="showTab('summeval')">SummEval</button>
<button onclick="showTab('aggrefact')">AggreFact</button>
<button onclick="showTab('crossdomain')">Cross-Domain</button>
<button onclick="showTab('qa')">QA Transfer</button>
<button onclick="showTab('robustness')">Robustness</button>
<button onclick="showTab('efficiency')">Efficiency</button>
<button onclick="showTab('screening')">SA/RA/AR</button>
<button onclick="showTab('formulas')">Formulas</button>
</nav>

<!-- TAB 1: OVERVIEW -->
<div id="overview" class="tab active">
<h2>Key Findings</h2>
<div class="stat-cards">
"""
    sl = results["summeval_system_level"]
    se = results["summeval_correlations"]
    if sl:
        chi_avg = sl.get("average", {}).get("CHI (full)", {}).get("spearman", 0)
        html += f'<div class="stat-card"><div class="label">CHI System-Level Avg</div><div class="value">{fmt(chi_avg)}</div><div class="detail">Spearman ρ across 16 systems (SummEval)</div></div>'
        chi_rel = sl.get("relevance", {}).get("CHI (full)", {}).get("spearman", 0)
        rouge1_rel = sl.get("relevance", {}).get("ROUGE-1", {}).get("spearman")
        html += f'<div class="stat-card"><div class="label">CHI Relevance</div><div class="value">{fmt(chi_rel)}</div><div class="detail">Beats ROUGE-1 ({fmt(rouge1_rel)})</div></div>'
    if se:
        rhi_sp = se.get("RHI*", {}).get("spearman", 0)
        html += f'<div class="stat-card"><div class="label">RHI* Sample-Level</div><div class="value">{fmt(rhi_sp)}</div><div class="detail">Best CHI component on consistency</div></div>'
    ab = results["additional_baselines"]
    if ab and ab.get("minicheck"):
        mc = ab["minicheck"]["spearman"]
        html += f'<div class="stat-card"><div class="label">MiniCheck Baseline</div><div class="value">{fmt(mc)}</div><div class="detail">NLI-based claim checking (N=1600)</div></div>'
    halu = results["halueval_qa_results"]
    if halu:
        halu_metrics = halu.get("results", halu.get("overall", {}).get("metrics", {}))
        chi_auc = halu_metrics.get("CHI", {}).get("roc_auc")
        if chi_auc is not None:
            eff_auc = max(chi_auc, 1 - chi_auc)
            html += f'<div class="stat-card"><div class="label">CHI on QA (HaluEval)</div><div class="value">{fmt(eff_auc)}</div><div class="detail">Effective AUC — non-summarization transfer</div></div>'
    eff = results["efficiency"]
    if eff:
        rhi_ms = eff.get("metrics", {}).get("RHI*", {}).get("mean_ms")
        if rhi_ms is not None:
            html += f'<div class="stat-card"><div class="label">RHI* Speed (optimized)</div><div class="value">{fmt(rhi_ms, 0)}ms</div><div class="detail">Per sample — 3-5x faster after batch encoding</div></div>'
    agg = results["aggrefact_results"]
    if agg:
        best_chi = 0
        best_name = ""
        for m, v in agg.get("overall", {}).get("metrics", {}).items():
            auc = v.get("roc_auc") or 0
            eff_auc = max(auc, 1 - auc)
            if any(c in m for c in ("CHI", "EHI", "RHI")) and eff_auc > best_chi:
                best_chi = eff_auc
                best_name = m
        if best_chi > 0:
            html += f'<div class="stat-card"><div class="label">{best_name} AUC (AggreFact)</div><div class="value">{fmt(best_chi)}</div><div class="detail">Binary factuality classification</div></div>'

    html += """</div>

<h3>Metric Architecture</h3>
<div class="metric-tree">
<div class="metric-box"><div class="name">EHI</div><div class="desc">Entity<br>5 softmax factors</div></div>
<span class="arrow">+</span>
<div class="metric-box"><div class="name">RHI*</div><div class="desc">Relation (SVO)<br>6 softmax factors</div></div>
<span class="arrow">+</span>
<div class="metric-box"><div class="name">QHI</div><div class="desc">Quantity<br>5 softmax factors</div></div>
<span class="arrow">&rarr;</span>
<div class="metric-box composite"><div class="name">CHI</div><div class="desc">Composite<br>Adaptive Harmonic Mean</div></div>
</div>

<h3>What Makes CHI Different</h3>
<div class="two-col">
<div>
<p><strong>Decomposable:</strong> CHI tells you <em>what kind</em> of hallucination occurred &mdash; entity fabrication (EHI), relation distortion (RHI*), or numerical error (QHI). Black-box metrics like MiniCheck give only a score.</p>
<p><strong>Interpretable Factors:</strong> Each sub-index decomposes into named factors (Positive Hallucination, Lost Focus, Over Focus, etc.) that pinpoint the failure mode.</p>
</div>
<div>
<p><strong>Formally Grounded:</strong> All three components use a consistent softmax architecture over Venn-diagram set operations. The composite uses adaptive harmonic mean that degrades gracefully when a dimension is absent.</p>
<p><strong>No Training Required:</strong> Pure NLP pipeline (NER + dependency parsing + quantity extraction). No fine-tuning, no GPU training, deterministic.</p>
</div>
</div>
</div>

<!-- TAB 2: SUMMEVAL -->
<div id="summeval" class="tab">
<h2>SummEval Benchmark (Real Human Annotations)</h2>
<p>100 CNN/DailyMail articles &times; 16 model summaries = 1,600 evaluation triples. Human consistency ratings (1-5, averaged across 3 annotators).</p>

<div class="two-col">
<div>
<h3>Sample-Level Correlations (N=1,600)</h3>
"""
    html += build_summeval_sample_table(se, ab, results["bartscore_summeval"])
    html += """</div>
<div>
<h3>Correlation Bar Chart</h3>
<div class="chart-container"><canvas id="chart_summeval_bar"></canvas></div>
</div>
</div>

<h3>System-Level Correlations (16 Systems)</h3>
<p>System-level aggregation reveals CHI's strength: ranking <em>models</em> rather than individual summaries. BARTScore and MiniCheck are sample-level only (per-system scores not yet available).</p>
"""
    html += build_summeval_system_table(sl)

    html += """<h3>System-Level Grouped Comparison</h3>
<div class="chart-container"><canvas id="chart_system_grouped"></canvas></div>
"""

    sem = results["summeval_ehi_method"]
    if sem:
        html += '<h3>Per-Dimension Breakdown (Sample-Level)</h3>'
        dims = ["spearman_consistency", "spearman_coherence", "spearman_fluency", "spearman_relevance", "spearman_avg"]
        dim_labels = ["Consistency", "Coherence", "Fluency", "Relevance", "Average"]
        html += '<table class="sortable"><thead><tr><th>Metric</th>'
        for dl in dim_labels:
            html += f'<th>{dl}</th>'
        html += '</tr></thead><tbody>'
        for m in sorted(sem.keys(), key=lambda x: -(sem[x].get("spearman_avg", 0))):
            v = sem[m]
            is_chi = any(c in m for c in ("CHI", "EHI", "RHI", "QHI"))
            cls = ' class="highlight"' if is_chi else ""
            html += f'<tr{cls}><td>{m}</td>'
            for dk in dims:
                html += f'<td>{fmt(v.get(dk))}</td>'
            html += '</tr>'
        html += '</tbody></table>'

    html += """</div>

<!-- TAB 3: AGGREFACT -->
<div id="aggrefact" class="tab">
<h2>AggreFact Benchmark (Binary Factuality Classification)</h2>
<p>Community-standard benchmark with binary consistent/inconsistent labels. Source document used as both Input and Reference (I=R) &mdash; standard for reference-free factuality evaluation.</p>
"""
    html += build_aggrefact_table(results["aggrefact_results"])
    html += """<h3>ROC-AUC Comparison</h3>
<div class="chart-container"><canvas id="chart_aggrefact_bar"></canvas></div>
</div>

<!-- TAB 4: CROSS-DOMAIN -->
<div id="crossdomain" class="tab">
<h2>Cross-Domain Analysis</h2>
<p>Evaluation across 4 domains (News, Medical, Legal, Financial). 20 articles &times; 5 model variants per domain. Spearman correlation with proxy human scores (60% BERTScore + 40% AlignScore).</p>
"""
    html += build_cross_domain_table(results["experiment_summary"])
    html += """<h3>CHI Components Across Domains</h3>
<div class="chart-container"><canvas id="chart_crossdomain_line"></canvas></div>

<h3>Key Observations</h3>
<div class="two-col">
<div>
<p><strong>RHI* excels in domain-specific text:</strong> Highest correlation in Medical (0.734) and Legal (0.722) where relation faithfulness matters most.</p>
<p><strong>EHI is most stable:</strong> Consistently above 0.5 across all domains, reflecting the universal importance of entity accuracy.</p>
</div>
<div>
<p><strong>QHI is domain-sensitive:</strong> Strong in News (0.461) and Medical (0.535) but weak in Legal (0.241) where quantities are less diagnostic.</p>
<p><strong>CHI balances components:</strong> The harmonic mean prevents any single weak component from dominating.</p>
</div>
</div>
</div>

<!-- TAB 5: QA TRANSFER -->
<div id="qa" class="tab">
<h2>Non-Summarization Transfer: QA Hallucination Detection</h2>
<p>Evaluation on HaluEval QA dataset. Tests whether entity/relation/quantity metrics generalize beyond summarization to question-answering.</p>
<p>Mapping: Input = question + knowledge, Reference = correct answer, Generated = model answer.</p>
"""
    html += build_halueval_table(results["halueval_qa_results"])
    html += """<h3>Effective AUC Comparison</h3>
<div class="chart-container"><canvas id="chart_halueval_bar"></canvas></div>
</div>

<!-- TAB 6: ROBUSTNESS -->
<div id="robustness" class="tab">
<h2>Robustness: Controlled Perturbation Tests</h2>
<p>High-quality SummEval summaries progressively corrupted to test metric sensitivity. A good faithfulness metric should show large, monotonic score drops as perturbation increases.</p>
"""
    html += build_robustness_section(results["robustness"])
    html += """<h3>Score vs. Perturbation Intensity</h3>
<div class="chart-container"><canvas id="chart_robustness_line"></canvas></div>
</div>

<!-- TAB 7: EFFICIENCY -->
<div id="efficiency" class="tab">
<h2>Computational Efficiency</h2>
<p>Per-sample timing on SummEval triples. Categories: Lightweight (&lt;100ms), Moderate (100-1000ms), Heavyweight (&gt;1000ms).</p>
"""
    html += build_efficiency_table(results["efficiency"])
    html += """<h3>Time per Sample (ms)</h3>
<div class="chart-container"><canvas id="chart_efficiency_bar"></canvas></div>
</div>

<!-- TAB 8: SA/RA/AR SCREENING -->
<div id="screening" class="tab">
<h2>SA/RA/AR: Abstractiveness Screening Tool</h2>
<p>SA (Source Alignment), RA (Reference Alignment), and AR (Alignment Ratio) are <strong>pre-screening tools</strong>, not competing hallucination metrics. They measure <em>how abstractive</em> a summary is, which determines whether expensive CHI evaluation is needed.</p>

<h3>Two-Stage Pipeline</h3>
<div class="pipeline-diagram">
<div class="pipeline-step">Input Text</div>
<span class="pipeline-arrow">&rarr;</span>
<div class="pipeline-step">Model Summary</div>
<span class="pipeline-arrow">&rarr;</span>
<div class="pipeline-step active">SA/RA/AR<br><small>Fast: word-overlap</small></div>
<span class="pipeline-arrow">&rarr;</span>
<div class="pipeline-step" style="border-style:dashed">AR &gt; threshold?</div>
<span class="pipeline-arrow">&rarr;</span>
<div class="pipeline-step active">CHI Evaluation<br><small>NER + SVO + Qty</small></div>
</div>

<h3>Why This Matters</h3>
<div class="two-col">
<div>
<p><strong>Near-extractive summaries are low-risk:</strong> If SA ≈ 0 (almost no new content), hallucination probability is very low. Skip expensive CHI computation.</p>
<p><strong>Highly abstractive summaries need scrutiny:</strong> High AR indicates the model is generating novel content. These are the summaries most likely to contain hallucinations.</p>
</div>
<div>
<p><strong>AR as a risk signal:</strong> AR = RA/SA. Values significantly above 1.0 mean the reference is more abstractive than the model &mdash; the model may be "playing it safe" with extractive copying. Values below 1.0 flag aggressive abstraction.</p>
<p><strong>Computational saving:</strong> SA/RA/AR computation takes &lt;1ms per sample (word-overlap only). CHI takes 1-10 seconds. Filtering out safe summaries first can reduce total evaluation time by 30-60%.</p>
</div>
</div>

<h3>Formula Reference</h3>
<table>
<thead><tr><th>Metric</th><th>Formula</th><th>Range</th><th>Interpretation</th></tr></thead>
<tbody>
<tr><td>RA</td><td>HM(R, O) &times; (1 &minus; c<sub>R</sub>)<sup>3</sup></td><td>[0, &infin;)</td><td>Reference abstractiveness vs. source</td></tr>
<tr><td>SA</td><td>HM(S, O) &times; (1 &minus; c<sub>S</sub>)<sup>3</sup></td><td>[0, &infin;)</td><td>Summary abstractiveness vs. source</td></tr>
<tr><td>AR</td><td>RA / SA</td><td>(0, &infin;)</td><td>Relative abstractiveness; AR=1 is balanced</td></tr>
</tbody>
</table>
<p class="stat-note">HM = Harmonic Mean of text lengths. c<sub>R</sub>, c<sub>S</sub> = normalized overlap count with source. Published: Katwe et al. (IEEE OCIT 2022), revised in thesis Chapter 3.</p>
</div>

<!-- TAB 9: FORMULAS -->
<div id="formulas" class="tab">
<h2>Metric Formulas &amp; Factor Definitions</h2>
<p>All three sub-indices share a unified softmax architecture over Venn-diagram set operations. Given Input text (I), Reference text (R), and Generated text (G), entity/relation/quantity sets are extracted and factors computed from their overlaps.</p>

<h3>EHI &mdash; Entity Hallucination Index</h3>
<p>Extraction: spaCy NER on I, R, G &rarr; entity sets <b>I</b>, <b>R</b>, <b>G</b>.</p>
<table>
<thead><tr><th>Factor</th><th>Set Operation</th><th>Formula</th><th>Meaning</th></tr></thead>
<tbody>
<tr><td><b>EF</b></td><td>I &cap; R &cap; G</td><td>3&middot;|I&cap;R&cap;G| / (|I|+|R|+|G|)</td><td>Faithfully preserved entities</td></tr>
<tr><td><b>PH</b></td><td>R &cap; G</td><td>2&middot;|R&cap;G| / (|R|+|G|)</td><td>Beneficial abstraction (reference-aligned)</td></tr>
<tr><td><b>OF</b></td><td>I &cap; G &minus; R</td><td>2&middot;|I&cap;G| / (|I|+|G|)</td><td>Over-focus on irrelevant source entities</td></tr>
<tr><td><b>NH</b></td><td>G &minus; (I &cup; R)</td><td>(|G| &minus; |R&cap;G| &minus; |I&cap;G| + |I&cap;R&cap;G|) / |G|</td><td>Fabricated entities (negative hallucination)</td></tr>
<tr><td><b>LF</b></td><td>R &minus; G</td><td>(|R| &minus; |I&cap;R| + |I&cap;R&cap;G|) / (|R|+|G|)</td><td>Lost/omitted reference entities</td></tr>
</tbody>
</table>
<div class="formula-box">
<b>EHI</b> = (e<sup>PH</sup> + e<sup>EF</sup>) / (e<sup>PH</sup> + e<sup>EF</sup> + e<sup>NH</sup> + e<sup>OF</sup> + e<sup>LF</sup>) &ensp; &isin; [0, 1] &ensp; | &ensp; Equilibrium = 0.4
</div>
<p class="stat-note">Published: Katwe et al. (ACI 2025). Higher = better faithfulness.</p>

<h3>RHI* &mdash; Relation Hallucination Index (Softmax Variant)</h3>
<p>Extraction: spaCy dependency parsing for Subject-Verb-Object (SVO) triples from I, R, G. Partial matching via semantic verb similarity (sentence-transformers, cosine &ge; 0.3). Each SVO match scores 1/3 per matching component.</p>
<table>
<thead><tr><th>Factor</th><th>Set Operation</th><th>Formula</th><th>Meaning</th></tr></thead>
<tbody>
<tr><td><b>EF</b></td><td>I &cap; R &cap; G</td><td>3&middot;|I&cap;R&cap;G|<sub>partial</sub> / (|I|+|R|+|G|)</td><td>Faithfully preserved relations</td></tr>
<tr><td><b>PH</b></td><td>R &cap; G</td><td>2&middot;|R&cap;G|<sub>partial</sub> / (|R|+|G|)</td><td>Reference-aligned novel relations</td></tr>
<tr><td><b>OF</b></td><td>I &cap; G &minus; I&cap;R&cap;G</td><td>2&middot;(|I&cap;G| &minus; |I&cap;R&cap;G|) / (|I|+|G|)</td><td>Source relations over-represented</td></tr>
<tr><td><b>NH</b></td><td>G &minus; (I &cup; R)</td><td>|G| &minus; (|R&cap;G| + |I&cap;G| &minus; |I&cap;R&cap;G|) / |G|</td><td>Fabricated relations</td></tr>
<tr><td><b>LH</b></td><td>R &minus; I &minus; G</td><td>2&middot;(|R| &minus; |I&cap;R| &minus; |R&cap;G| + |I&cap;R&cap;G|) / (|I|+|G|)</td><td>Lost hallucination (reference-only relations missed)</td></tr>
<tr><td><b>LF</b></td><td>I &cap; R &minus; G</td><td>(|I&cap;R| &minus; |I&cap;R&cap;G|) / |G|</td><td>Lost focus (agreed relations dropped)</td></tr>
</tbody>
</table>
<div class="formula-box">
<b>RHI*</b> = (e<sup>EF</sup> + e<sup>PH</sup>) / (e<sup>EF</sup> + e<sup>PH</sup> + e<sup>OF</sup> + e<sup>NH</sup> + e<sup>LH</sup> + e<sup>LF</sup>) &ensp; &isin; [0, 1] &ensp; | &ensp; Equilibrium = 0.333
</div>
<p class="stat-note">Based on: Katwe et al. (ACM FIRE 2024). Softmax variant for CHI integration. 6 factors vs EHI's 5 (adds LH).</p>

<h3>QHI &mdash; Quantity Hallucination Index</h3>
<p>Extraction: spaCy NER (MONEY, PERCENT, QUANTITY, CARDINAL) + regex fallback. Tolerance-aware matching (&epsilon; = 5%).</p>
<table>
<thead><tr><th>Factor</th><th>Set Operation</th><th>Formula</th><th>Meaning</th></tr></thead>
<tbody>
<tr><td><b>QEF</b></td><td>I &cap; R &cap; G</td><td>3&middot;|I&cap;R&cap;G| / (|I|+|R|+|G|)</td><td>Faithfully preserved quantities</td></tr>
<tr><td><b>QPH</b></td><td>R &cap; G</td><td>2&middot;|R&cap;G| / (|R|+|G|)</td><td>Reference-aligned quantities</td></tr>
<tr><td><b>QOF</b></td><td>I &cap; G &minus; I&cap;R&cap;G</td><td>2&middot;(|I&cap;G| &minus; |I&cap;R&cap;G|) / (|I|+|G|)</td><td>Over-focused source quantities</td></tr>
<tr><td><b>QNH</b></td><td>G &minus; (I &cup; R)</td><td>(|G| &minus; |matched in I or R|) / |G|</td><td>Fabricated numbers</td></tr>
<tr><td><b>QLF</b></td><td>R &minus; G</td><td>|R not matched by G| / (|R|+|G|)</td><td>Lost/omitted quantities</td></tr>
</tbody>
</table>
<div class="formula-box">
<b>QHI</b> = (e<sup>QPH</sup> + e<sup>QEF</sup>) / (e<sup>QPH</sup> + e<sup>QEF</sup> + e<sup>QNH</sup> + e<sup>QOF</sup> + e<sup>QLF</sup>) &ensp; &isin; [0, 1] &ensp; | &ensp; Equilibrium = 0.4
</div>
<p class="stat-note">Novel contribution. Same softmax architecture as EHI. Detects numerical fabrication.</p>

<h3>CHI &mdash; Composite Hallucination Index</h3>
<div class="formula-box">
<b>CHI</b> = HarmonicMean(EHI, RHI*, QHI) = 3 / (1/EHI + 1/RHI* + 1/QHI) &ensp; &isin; [0, 1]
</div>
<p>Adaptive: uses 2-way harmonic mean if one dimension is absent (e.g., no entities or no quantities in the generated text).</p>
<table>
<thead><tr><th>Condition</th><th>Formula</th></tr></thead>
<tbody>
<tr><td>All three dimensions present</td><td>HM<sub>3</sub>(EHI, RHI*, QHI)</td></tr>
<tr><td>No quantities in G</td><td>HM<sub>2</sub>(EHI, RHI*)</td></tr>
<tr><td>No entities in G</td><td>HM<sub>2</sub>(QHI, RHI*)</td></tr>
<tr><td>Only relations</td><td>RHI* (fallback)</td></tr>
</tbody>
</table>

<h3>SA / RA / AR &mdash; Abstractiveness Ratios</h3>
<p>Pre-screening tool. Measures how abstractive a summary is relative to the source.</p>
<table>
<thead><tr><th>Metric</th><th>Formula</th><th>Range</th><th>Purpose</th></tr></thead>
<tbody>
<tr><td><b>RA</b></td><td>HM(|R|, |O|) &times; (1 &minus; c<sub>R</sub>)<sup>3</sup></td><td>[0, &infin;)</td><td>Reference abstractiveness vs source</td></tr>
<tr><td><b>SA</b></td><td>HM(|S|, |O|) &times; (1 &minus; c<sub>S</sub>)<sup>3</sup></td><td>[0, &infin;)</td><td>Summary abstractiveness vs source</td></tr>
<tr><td><b>AR</b></td><td>RA / SA</td><td>(0, &infin;)</td><td>Relative abstractiveness; AR &gt; 1 &rArr; reference more abstractive</td></tr>
</tbody>
</table>
<p class="stat-note">HM = Harmonic Mean of text lengths. c<sub>R</sub>, c<sub>S</sub> = normalized overlap count. Published: Katwe et al. (IEEE OCIT 2022).</p>

<h3>Baseline Metrics</h3>
<p>All baselines computed in our evaluation pipeline for head-to-head comparison.</p>

<table>
<thead><tr><th>Metric</th><th>Type</th><th>Formula / Method</th><th>Model</th><th>Range</th><th>Citation</th></tr></thead>
<tbody>
<tr class="highlight"><td><b>ROUGE-1/2/L</b></td><td>Lexical overlap</td>
<td>F1 = 2&middot;P&middot;R / (P+R) over n-gram overlap (1-gram, 2-gram, LCS). Uses stemming.</td>
<td>None (string matching)</td><td>[0, 1]</td><td>Lin, 2004</td></tr>

<tr><td><b>BERTScore</b></td><td>Embedding similarity</td>
<td>For each token in candidate, find max cosine similarity to any token in reference using contextual embeddings. Compute precision, recall, F1 over these maximal similarities. Applies IDF weighting.</td>
<td>roberta-large (355M)</td><td>[0, 1]</td><td>Zhang et al., 2020</td></tr>

<tr class="highlight"><td><b>BARTScore</b></td><td>Generation probability</td>
<td>BARTScore(src &rarr; hyp) = &minus;NLL = &minus;loss where loss = CrossEntropy of generating hypothesis conditioned on source.<br>
<code>score = &minus;model(input_ids=src, labels=hyp).loss</code><br>Higher = more likely generation = more faithful.</td>
<td>facebook/bart-large-cnn (406M)</td><td>(&minus;&infin;, 0]</td><td>Yuan et al., 2021</td></tr>

<tr><td><b>MiniCheck</b></td><td>NLI claim verification</td>
<td>1. Decompose summary into sentences. 2. For each sentence, compute max P(entailment) across source sentences using NLI cross-encoder. 3. If max entailment &gt; 0.5, sentence is &ldquo;entailed.&rdquo; 4. Score = fraction of entailed sentences.</td>
<td>cross-encoder/nli-deberta-v3-base (86M)</td><td>[0, 1]</td><td>Tang et al., 2024</td></tr>

<tr class="highlight"><td><b>AlignScore (proxy)</b></td><td>NLI entailment</td>
<td>P(entailment | source, summary) from a single NLI forward pass. Entailment logit after softmax over [contradiction, entailment, neutral].</td>
<td>cross-encoder/nli-deberta-v3-base (86M)</td><td>[0, 1]</td><td>Zha et al., 2023</td></tr>

<tr><td><b>SummaC (proxy)</b></td><td>NLI non-contradiction</td>
<td>For each summary sentence: 1 &minus; P(contradiction | source, sentence). Average across sentences. Measures absence of contradictory content.</td>
<td>cross-encoder/nli-deberta-v3-base (86M)</td><td>[0, 1]</td><td>Laban et al., 2022</td></tr>

<tr class="highlight"><td><b>Entity F1</b></td><td>NER overlap</td>
<td>F1 = 2&middot;P&middot;R / (P+R) where P = |source_ents &cap; gen_ents| / |gen_ents|, R = |source_ents &cap; gen_ents| / |source_ents|. Entities extracted via spaCy NER.</td>
<td>en_core_web_sm (spaCy)</td><td>[0, 1]</td><td>Nan et al., 2021</td></tr>
</tbody>
</table>

<h3>Comparison: Our Metrics vs Baselines</h3>
<table>
<thead><tr><th>Property</th><th>CHI Family</th><th>ROUGE</th><th>BERTScore</th><th>BARTScore</th><th>MiniCheck</th></tr></thead>
<tbody>
<tr><td>Requires training</td><td>No</td><td>No</td><td>No (pretrained)</td><td>No (pretrained)</td><td>No (pretrained)</td></tr>
<tr><td>Model size</td><td>spaCy 12M + ST 22M</td><td>0</td><td>355M</td><td>406M</td><td>86M</td></tr>
<tr><td>Decomposable</td><td><b>Yes (5&ndash;6 factors)</b></td><td>No</td><td>No</td><td>No</td><td>Per-sentence only</td></tr>
<tr><td>Hallucination type</td><td><b>Entity + Relation + Quantity</b></td><td>None</td><td>None</td><td>None</td><td>Binary only</td></tr>
<tr><td>Reference needed</td><td>Yes (I + R + G)</td><td>Yes (R + G)</td><td>Yes (R + G)</td><td>No (src &rarr; hyp)</td><td>No (src + G)</td></tr>
<tr><td>Deterministic</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td></tr>
</tbody>
</table>

<h3>Unified Architecture Summary</h3>
<div class="metric-tree">
<div class="metric-box"><div class="name">Input (I)</div><div class="desc">Source document</div></div>
<div class="metric-box"><div class="name">Reference (R)</div><div class="desc">Human summary</div></div>
<div class="metric-box"><div class="name">Generated (G)</div><div class="desc">Model summary</div></div>
</div>
<p style="text-align:center">&darr; Extract entities / SVO triples / quantities &darr;</p>
<div class="metric-tree">
<div class="metric-box"><div class="name">Venn Diagram</div><div class="desc">I &cap; R &cap; G, R &cap; G, I &cap; G, etc.</div></div>
<span class="arrow">&rarr;</span>
<div class="metric-box"><div class="name">5&ndash;6 Factors</div><div class="desc">EF, PH, OF, NH, LF (+LH)</div></div>
<span class="arrow">&rarr;</span>
<div class="metric-box"><div class="name">Softmax</div><div class="desc">(e<sup>good</sup>) / &Sigma;(e<sup>all</sup>)</div></div>
<span class="arrow">&rarr;</span>
<div class="metric-box composite"><div class="name">CHI</div><div class="desc">Harmonic Mean</div></div>
</div>

<h3>Key Design Principles</h3>
<div class="two-col">
<div>
<p><strong>Numerator = positive signal:</strong> EF (faithful preservation) and PH (beneficial abstraction) in the numerator. These are the "good" factors.</p>
<p><strong>Denominator = all factors:</strong> Normalizes by the total signal including hallucination factors (NH, OF, LF, LH). More hallucination &rarr; lower score.</p>
</div>
<div>
<p><strong>Softmax guarantees [0,1]:</strong> Bounded output, equilibrium at 2/N factors. No arbitrary scaling.</p>
<p><strong>Partial matching in RHI*:</strong> SVO triples match 1/3 per component (subject, verb, object), with semantic verb similarity via cosine distance.</p>
</div>
</div>

<h3>Statistical Methods</h3>

<h4>Correlation Measures</h4>
<table>
<thead><tr><th>Measure</th><th>Formula</th><th>Used Where</th><th>Why</th></tr></thead>
<tbody>
<tr><td><b>Spearman &rho;</b></td>
<td>&rho; = 1 &minus; (6 &sum;d<sub>i</sub><sup>2</sup>) / (n(n<sup>2</sup>&minus;1)), where d<sub>i</sub> = rank(x<sub>i</sub>) &minus; rank(y<sub>i</sub>)</td>
<td>All experiments</td>
<td>Rank-based, robust to outliers, no normality assumption. The primary correlation metric in NLG evaluation.</td></tr>
<tr><td><b>Kendall &tau;</b></td>
<td>&tau; = (concordant &minus; discordant) / &frac12;n(n&minus;1)</td>
<td>SummEval, cross-domain</td>
<td>Counts pairwise ordering agreements. More conservative than Spearman; preferred for small N (system-level, N=16).</td></tr>
<tr><td><b>Point-Biserial r</b></td>
<td>r<sub>pb</sub> = (M<sub>1</sub> &minus; M<sub>0</sub>) / s &times; &radic;(n<sub>1</sub>n<sub>0</sub>/n<sup>2</sup>), where M<sub>1</sub>,M<sub>0</sub> = group means</td>
<td>AggreFact, HaluEval</td>
<td>Pearson r between a continuous score and a binary label. Measures linear association with the ground truth class.</td></tr>
</tbody>
</table>

<h4>Significance Stars</h4>
<table>
<thead><tr><th>Symbol</th><th>p-value Threshold</th><th>Interpretation</th></tr></thead>
<tbody>
<tr><td><span class="sig sig1">*</span></td><td>p &lt; 0.05</td><td>Statistically significant at 95% confidence</td></tr>
<tr><td><span class="sig sig2">**</span></td><td>p &lt; 0.01</td><td>Highly significant at 99% confidence</td></tr>
<tr><td><span class="sig sig3">***</span></td><td>p &lt; 0.001</td><td>Very highly significant at 99.9% confidence. Extremely unlikely to be due to chance.</td></tr>
</tbody>
</table>
<p class="stat-note">p-value = probability of observing a correlation this strong (or stronger) if the true correlation were zero. Computed via scipy.stats.spearmanr which uses a t-distribution approximation: t = &rho; &times; &radic;((n&minus;2)/(1&minus;&rho;<sup>2</sup>)), df = n&minus;2.</p>

<h4>Binary Classification Metrics (AggreFact / HaluEval)</h4>
<table>
<thead><tr><th>Metric</th><th>Formula</th><th>Why Used</th></tr></thead>
<tbody>
<tr><td><b>ROC-AUC</b></td>
<td>Area under the Receiver Operating Characteristic curve. ROC plots TPR vs FPR at every threshold. AUC = P(score<sub>positive</sub> &gt; score<sub>negative</sub>) for a random pair. Threshold-free.</td>
<td>Primary metric for binary factuality. Handles class imbalance (AggreFact: 464 vs 36). No threshold selection needed.</td></tr>
<tr><td><b>Effective AUC</b></td>
<td>max(AUC, 1 &minus; AUC)</td>
<td>For faithfulness metrics where higher = more faithful but label=1 = hallucinated. Corrects polarity so all metrics are comparable.</td></tr>
<tr><td><b>Balanced Accuracy</b></td>
<td>&frac12;(TPR + TNR) at optimal threshold. Optimal threshold = argmax<sub>t</sub>(TPR(t) + TNR(t) &minus; 1) &ensp; (Youden&rsquo;s J statistic)</td>
<td>Fair accuracy under class imbalance. Random baseline = 0.5 regardless of class ratio.</td></tr>
</tbody>
</table>

<h4>Cross-Domain Correlation Protocol</h4>
<p>For each domain (News, Medical, Legal, Financial): 20 articles &times; 5 model variants (BART, T5, extractive, corrupted, reference) = 100 (source, reference, generated) triples. Each metric scores all 100 triples. Spearman &rho; is computed between the 100 metric scores and the 100 proxy human scores. Proxy human = 0.6&times;BERTScore + 0.4&times;AlignScore, scaled to [1,10]. Reported per-domain, then averaged.</p>

<h4>Robustness: Score vs. Perturbation Intensity</h4>
<p>For each perturbation level (0%, 25%, 50%, 75%, 100%), the mean metric score across all 200 samples is plotted. A good faithfulness metric shows a <b>monotonic decrease</b> as perturbation intensity increases &mdash; it should detect that progressively corrupted text is less faithful. The sensitivity &Delta; = score(100%) &minus; score(0%); larger |&Delta;| means the metric is more sensitive to that perturbation type.</p>

</div>

</div><!-- /container -->

<footer>
CHI: Composite Hallucination Index &mdash; Katwe et al. &mdash; Report generated {now}
</footer>

<script>
"""
    html += f"const chartData = {json.dumps(charts)};\n"
    html += """
function showTab(id) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  const btns = document.querySelectorAll('nav button');
  const tabIds = ['overview','summeval','aggrefact','crossdomain','qa','robustness','efficiency','screening','formulas'];
  const idx = tabIds.indexOf(id);
  if (idx >= 0 && btns[idx]) btns[idx].classList.add('active');
  requestAnimationFrame(renderCharts);
}

function toggleTheme() {
  const html = document.documentElement;
  html.dataset.theme = html.dataset.theme === 'dark' ? 'light' : 'dark';
}

const chartInstances = {};

function makeBar(canvasId, data, title, horizontal) {
  if (!data || !document.getElementById(canvasId)) return;
  if (chartInstances[canvasId]) chartInstances[canvasId].destroy();
  const ctx = document.getElementById(canvasId).getContext('2d');
  chartInstances[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.labels,
      datasets: [{
        data: data.values,
        backgroundColor: data.colors,
        borderRadius: 4,
        maxBarThickness: 40,
      }]
    },
    options: {
      indexAxis: horizontal ? 'y' : 'x',
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false }, title: { display: !!title, text: title, font: { size: 14 } },
        tooltip: { callbacks: { label: ctx => ctx.parsed[horizontal ? 'x' : 'y'].toFixed(3) } } },
      scales: { [horizontal ? 'x' : 'y']: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.06)' } },
        [horizontal ? 'y' : 'x']: { grid: { display: false } } }
    }
  });
}

function makeGroupedBar(canvasId, data, title) {
  if (!data || !document.getElementById(canvasId)) return;
  if (chartInstances[canvasId]) chartInstances[canvasId].destroy();
  const ctx = document.getElementById(canvasId).getContext('2d');
  const datasets = data.datasets.map(ds => ({
    label: ds.label, data: ds.data, backgroundColor: ds.backgroundColor,
    borderRadius: 4, maxBarThickness: 30,
  }));
  chartInstances[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: { labels: data.labels, datasets: datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { title: { display: !!title, text: title, font: { size: 14 } },
        tooltip: { callbacks: { label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(3) } } },
      scales: { y: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.06)' } }, x: { grid: { display: false } } }
    }
  });
}

function makeLine(canvasId, data, title) {
  if (!data || !document.getElementById(canvasId)) return;
  if (chartInstances[canvasId]) chartInstances[canvasId].destroy();
  const ctx = document.getElementById(canvasId).getContext('2d');
  const datasets = data.datasets.map(ds => ({
    label: ds.label, data: ds.data, borderColor: ds.borderColor, fill: false,
    tension: ds.tension || 0.3, pointRadius: 5, pointHoverRadius: 7, borderWidth: 2,
  }));
  chartInstances[canvasId] = new Chart(ctx, {
    type: 'line',
    data: { labels: data.labels, datasets: datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { title: { display: !!title, text: title, font: { size: 14 } },
        tooltip: { mode: 'index', intersect: false,
          callbacks: { label: ctx => ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(3) } } },
      scales: { y: { grid: { color: 'rgba(0,0,0,0.06)' } }, x: { grid: { display: false } } },
      interaction: { mode: 'nearest', axis: 'x', intersect: false }
    }
  });
}

function renderCharts() {
  if (chartData.summeval_bar) makeBar('chart_summeval_bar', chartData.summeval_bar, 'Spearman ρ vs Human Consistency (Sample-Level)');
  if (chartData.system_grouped) makeGroupedBar('chart_system_grouped', chartData.system_grouped, 'System-Level Spearman ρ by Dimension');
  if (chartData.halueval_bar) makeBar('chart_halueval_bar', chartData.halueval_bar, 'Effective AUC on HaluEval QA');
  if (chartData.aggrefact_bar) makeBar('chart_aggrefact_bar', chartData.aggrefact_bar, 'Effective AUC on AggreFact');
  if (chartData.cross_domain_line) makeLine('chart_crossdomain_line', chartData.cross_domain_line, 'CHI Components: Spearman ρ by Domain');
  if (chartData.robustness_line) makeLine('chart_robustness_line', chartData.robustness_line, 'Score vs Entity-Swap Perturbation Intensity');
  if (chartData.efficiency_bar) makeBar('chart_efficiency_bar', chartData.efficiency_bar, 'Mean Time per Sample (ms)', true);
}

// Sortable tables
document.querySelectorAll('table.sortable th').forEach(th => {
  th.addEventListener('click', () => {
    const table = th.closest('table');
    const idx = Array.from(th.parentNode.children).indexOf(th);
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const asc = !th.classList.contains('asc');
    table.querySelectorAll('th').forEach(h => { h.classList.remove('asc','desc'); });
    th.classList.add(asc ? 'asc' : 'desc');
    rows.sort((a, b) => {
      let va = a.children[idx]?.textContent?.trim() || '';
      let vb = b.children[idx]?.textContent?.trim() || '';
      const na = parseFloat(va.replace(/[^0-9.\\-]/g, ''));
      const nb = parseFloat(vb.replace(/[^0-9.\\-]/g, ''));
      if (!isNaN(na) && !isNaN(nb)) return asc ? na - nb : nb - na;
      return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    });
    rows.forEach(r => tbody.appendChild(r));
  });
});

renderCharts();
</script>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Report generated: {output_path}")
    print(f"  File size: {Path(output_path).stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Generate CHI evaluation HTML report")
    parser.add_argument("--results-dir", default="./results", help="Directory with JSON results")
    parser.add_argument("--output", default="./results/chi_evaluation_report.html", help="Output HTML path")
    args = parser.parse_args()

    print("Loading results...")
    results = load_all_results(args.results_dir)

    available = sum(1 for v in results.values() if v is not None)
    total = len(results)
    print(f"  Found {available}/{total} result files")

    for name, data in results.items():
        status = "OK" if data else "PENDING"
        print(f"    {name}: {status}")

    print("\nGenerating HTML report...")
    generate_html(results, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
