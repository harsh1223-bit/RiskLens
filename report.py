from __future__ import annotations
from html import escape
from datetime import datetime, timezone

def build_html_report(performance, flags, source_note):
    def fmt(v):
        try:
            return f"{float(v):.3f}"
        except (TypeError, ValueError):
            return "N/A"
    rows=[]
    for flag in flags:
        status=escape(str(flag.get("status","Unavailable")))
        cls={"Green":"green","Amber":"amber","Red":"red"}.get(status,"gray")
        rows.append(f'<tr><td>{escape(str(flag.get("area","")))}</td><td><span class="badge {cls}">{status}</span></td><td>{escape(str(flag.get("finding","")))}</td></tr>')
    metrics = ["accuracy","precision","recall","f1","roc_auc"]
    metric_html="".join(f"<li><strong>{m.replace('_',' ').upper()}:</strong> {fmt(performance.get(m))}</li>" for m in metrics)
    timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RiskLens Risk Report</title><style>
body{{font-family:Arial,Helvetica,sans-serif;max-width:900px;margin:36px auto;padding:0 22px;color:#182b3b;line-height:1.5}}
h1,h2{{color:#0F2B46}} .sub{{color:#64748b}} .badge{{padding:3px 8px;border-radius:12px;font-weight:bold}}
.green{{background:#d9f5ef;color:#087b70}} .amber{{background:#fff0c2;color:#805900}} .red{{background:#ffe0df;color:#a52222}} .gray{{background:#e2e8f0;color:#475569}}
table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{text-align:left;border-bottom:1px solid #e2e8f0;padding:10px}}.panel{{background:#f6f8fb;border-radius:10px;padding:16px}}
</style></head><body>
<h1>RiskLens — AI Model Risk Report</h1><p class="sub">Generated {timestamp}</p>
<div class="panel"><strong>Data source / method:</strong><p>{escape(str(source_note))}</p><strong>Sample size evaluated:</strong> {performance.get("n","N/A")}</div>
<h2>Performance metrics</h2><ul>{metric_html}</ul>
<h2>Risk summary</h2><table><thead><tr><th>Area</th><th>Status</th><th>Finding</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<h2>Interpretation</h2><p>Traffic-light labels are heuristic indicators for review. They are not an overall certification or proof that a model is safe, fair, or compliant.</p>
<h2>Limitations</h2><ul>
<li>Small, imbalanced, or unrepresentative samples can produce unstable estimates.</li>
<li>Baseline mode uses a simple logistic regression model and does not tune it.</li>
<li>Group gaps are not proof of discrimination; context and additional analysis are necessary.</li>
<li>Drift thresholds are heuristics. Statistical significance is not the same as practical significance.</li>
<li>Drift alone does not establish that model performance has deteriorated.</li>
<li>This report is not legal advice and does not certify compliance with any regulation or standard.</li>
</ul><p class="sub">All displayed metrics are calculated from the data processed by RiskLens.</p></body></html>"""
