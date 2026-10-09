from __future__ import annotations

import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.inspection import permutation_importance

from metrics import evaluate_predictions, train_baseline, metric_definitions
from fairness import fairness_report
from drift import drift_report
from report import build_html_report
from sample_data import make_demo_data, make_demo_drift_data

st.set_page_config(page_title="RiskLens | AI Model Risk Checker", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
:root {
  --rl-bg: #e8edf4;
  --rl-navy: #17324d;
  --rl-teal: #078f88;
  --rl-muted: #607286;
  --rl-shadow-dark: #c4ccd7;
  --rl-shadow-light: #ffffff;
}
.stApp { background: var(--rl-bg); color: var(--rl-navy); }
[data-testid="stHeader"] { background: rgba(232,237,244,.9); }
[data-testid="stSidebar"], [data-testid="stSidebar"] > div { background: var(--rl-bg); }
[data-testid="stSidebar"] * { color: var(--rl-navy) !important; }
h1, h2, h3, h4, p, label, .stMarkdown, [data-testid="stCaptionContainer"] { color: var(--rl-navy); }
div[data-testid="stMetric"] {
  background: var(--rl-bg); padding: 18px 20px; border-radius: 18px;
  border: 1px solid rgba(255,255,255,.8);
  box-shadow: 7px 7px 15px var(--rl-shadow-dark), -7px -7px 15px var(--rl-shadow-light);
}
[data-testid="stMetricLabel"] { color: var(--rl-muted) !important; }
[data-testid="stMetricValue"] { color: var(--rl-navy) !important; }
.stTabs [data-baseweb="tab-list"] {
  gap: 8px; background: var(--rl-bg); padding: 8px; border-radius: 15px;
  box-shadow: inset 3px 3px 7px var(--rl-shadow-dark), inset -3px -3px 7px var(--rl-shadow-light);
}
.stTabs [data-baseweb="tab"] { color: var(--rl-muted) !important; border-radius: 10px; padding: 8px 14px; }
.stTabs [aria-selected="true"] {
  color: var(--rl-teal) !important; background: #edf2f8;
  box-shadow: 3px 3px 7px var(--rl-shadow-dark), -3px -3px 7px var(--rl-shadow-light);
}
.stButton > button, .stDownloadButton > button {
  background: var(--rl-bg); color: var(--rl-navy); border: 1px solid rgba(255,255,255,.8);
  border-radius: 12px; box-shadow: 5px 5px 10px var(--rl-shadow-dark), -5px -5px 10px var(--rl-shadow-light);
  font-weight: 650;
}
.stButton > button:hover, .stDownloadButton > button:hover { color: var(--rl-teal); border-color: #cbd5e1; }
.stButton > button:active, .stDownloadButton > button:active {
  box-shadow: inset 3px 3px 6px var(--rl-shadow-dark), inset -3px -3px 6px var(--rl-shadow-light);
}
input, textarea, [data-baseweb="select"] > div { background: var(--rl-bg) !important; color: var(--rl-navy) !important; border-radius: 10px !important; }
[data-testid="stFileUploader"] section { background: var(--rl-bg); border-radius: 14px; border: 1px dashed #b7c3d1; }
[data-testid="stDataFrame"], [data-testid="stTable"] { border-radius: 14px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 12px; }
hr { border-color: #d2dae4; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ RiskLens")
st.caption("AI Model Risk Checker · Evaluate performance, group fairness, and data drift.")
st.info("RiskLens provides diagnostic indicators, not a legal compliance determination or proof of discrimination.")

def status_color(level: str) -> str:
    return {"Green":"🟢", "Amber":"🟠", "Red":"🔴", "Unavailable":"⚪"}.get(level, "⚪")

def status_from_gap(gap, amber=0.05, red=0.10):
    if gap is None or pd.isna(gap):
        return "Unavailable"
    if gap < amber: return "Green"
    if gap <= red: return "Amber"
    return "Red"

with st.sidebar:
    st.header("Data & settings")
    mode = st.radio("Start with", ["Demo data", "Upload predictions CSV", "Upload dataset and train baseline"])
    pred_upload = None
    full_upload = None
    drift_upload = None
    if mode == "Upload predictions CSV":
        pred_upload = st.file_uploader("Predictions CSV", type=["csv"], key="pred")
    elif mode == "Upload dataset and train baseline":
        full_upload = st.file_uploader("Dataset CSV", type=["csv"], key="full")
    st.markdown("---")
    st.subheader("Fairness thresholds")
    amber_gap_pp = st.slider("Amber threshold (percentage points)", 1, 20, 5) / 100
    red_gap_pp = st.slider("Red threshold (percentage points)", 2, 40, 10) / 100
    if red_gap_pp < amber_gap_pp:
        st.warning("Red threshold is below amber threshold; thresholds will be swapped for interpretation.")
        amber_gap_pp, red_gap_pp = min(amber_gap_pp, red_gap_pp), max(amber_gap_pp, red_gap_pp)
    st.markdown("---")
    st.subheader("Drift reference / new data")
    drift_upload = st.file_uploader("Optional new data CSV (same feature columns)", type=["csv"], key="drift")

try:
    if mode == "Demo data":
        df = make_demo_data()
        y_true = df["y_true"]
        y_pred = df["y_pred"]
        y_prob = df["y_prob"]
        default_group = "gender"
        feature_df = df[["income", "age", "credit_history", "gender", "age_band"]]
        reference_df = feature_df.copy()
        new_df = make_demo_drift_data(reference_df)
        source_note = "Synthetic demo data generated locally."
    elif mode == "Upload predictions CSV":
        if pred_upload is None:
            st.warning("Upload a CSV with y_true and either y_pred or y_prob to begin.")
            st.stop()
        df = pd.read_csv(pred_upload)
        if "y_true" not in df.columns or not ({"y_pred", "y_prob"} & set(df.columns)):
            st.error("CSV must contain y_true and at least one of y_pred or y_prob.")
            st.stop()
        if "y_pred" not in df.columns and "y_prob" in df.columns:
            try:
                df["y_prob"] = pd.to_numeric(df["y_prob"], errors="raise")
                df["y_pred"] = (df["y_prob"] >= 0.5).astype(int)
            except Exception:
                st.error("y_prob must contain numeric probabilities between 0 and 1.")
                st.stop()
        y_true, y_pred = df["y_true"], df["y_pred"]
        y_prob = pd.to_numeric(df["y_prob"], errors="coerce") if "y_prob" in df.columns else None
        reserved = {"y_true", "y_pred", "y_prob"}
        feature_df = df[[c for c in df.columns if c not in reserved]].copy()
        reference_df = feature_df.copy()
        new_df = None
        default_group = feature_df.columns[0] if len(feature_df.columns) else None
        source_note = "Uploaded prediction data."
    else:
        if full_upload is None:
            st.warning("Upload a dataset to train and evaluate a baseline model.")
            st.stop()
        raw = pd.read_csv(full_upload)
        if raw.shape[0] < 20 or raw.shape[1] < 2:
            st.error("Please upload at least 20 rows and two columns for a meaningful baseline split.")
            st.stop()
        target = st.selectbox("Target column", raw.columns.tolist())
        sensitive_default = next((c for c in raw.columns if c.lower() in {"gender", "sex", "age_band", "race", "ethnicity"}), raw.columns[0])
        sensitive = st.selectbox("Sensitive/group column (optional)", ["None"] + raw.columns.tolist(), index=(["None"] + raw.columns.tolist()).index(sensitive_default) + 1 if sensitive_default in raw.columns else 0)
        y_true, y_pred, y_prob, model, X_test, y_test = train_baseline(raw, target)
        feature_df = X_test.copy()
        if sensitive != "None" and sensitive in raw.columns:
            # Attach held-out group values by the original test indices.
            feature_df[sensitive] = raw.loc[X_test.index, sensitive].astype(str)
            default_group = sensitive
        else:
            default_group = feature_df.columns[0] if len(feature_df.columns) else None
        reference_df = feature_df.copy()
        new_df = None
        source_note = f"Logistic regression baseline trained to predict '{target}' on a stratified 80/20 split. It is not tuned."
        st.caption("Baseline results are for a quick diagnostic only; they are not a substitute for model validation.")

    if len(y_true) < 2:
        st.error("At least two rows are required.")
        st.stop()
    perf = evaluate_predictions(y_true, y_pred, y_prob)
    st.caption(source_note)

    perf_tab, fair_tab, drift_tab, explain_tab, report_tab = st.tabs(["Performance", "Fairness", "Drift", "Explainability", "Report"])

    with perf_tab:
        st.subheader("Model performance")
        cols = st.columns(5)
        for col, name in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
            val = perf.get(name)
            col.metric(name.replace("_", " ").upper(), f"{val:.3f}" if val is not None and pd.notna(val) else "N/A")
        st.caption("Metrics summarize predictions against known labels. They do not establish that a model is suitable for a particular use.")
        cm = perf["confusion_matrix"]
        fig = go.Figure(data=go.Heatmap(z=cm, x=perf["labels"], y=perf["labels"], colorscale=[[0,"#e7f5f3"],[1,"#0F2B46"]], text=cm, texttemplate="%{text}", showscale=False))
        fig.update_layout(title="Confusion matrix", xaxis_title="Predicted label", yaxis_title="Actual label", margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Metric definitions"):
            for key, definition in metric_definitions().items():
                st.markdown(f"**{key.replace('_',' ').title()}** — {definition}")

    with fair_tab:
        st.subheader("Group fairness diagnostics")
        candidates = list(feature_df.columns)
        if not candidates:
            st.info("No group columns are available. Include sensitive/group columns in the CSV.")
            fair = {"available": False, "message": "No group columns supplied.", "rows": [], "gaps": {}}
        else:
            selected_group = st.selectbox("Choose a group column", candidates, index=candidates.index(default_group) if default_group in candidates else 0)
            fair = fairness_report(y_true, y_pred, feature_df[selected_group], amber_threshold=amber_gap_pp, red_threshold=red_gap_pp)
            if fair["available"]:
                st.caption("Group-level differences are descriptive. They are not, by themselves, proof of discrimination or a complete fairness assessment.")
                fdf = pd.DataFrame(fair["rows"])
                st.dataframe(fdf, use_container_width=True, hide_index=True)
                numeric_gap_keys = ["selection_rate", "tpr", "fpr", "precision"]
                gap_cols = st.columns(len(numeric_gap_keys))
                for col, metric in zip(gap_cols, numeric_gap_keys):
                    gap = fair["gaps"].get(metric)
                    value = f"{float(gap):.3f}" if isinstance(gap, (int, float)) and pd.notna(gap) else "N/A"
                    col.metric(f"{metric.replace('_', ' ').title()} gap", value,
                               help="Maximum group value minus minimum group value.")
                chart_df = fdf.melt(id_vars="group", value_vars=["selection_rate", "tpr", "fpr", "precision"], var_name="metric", value_name="value")
                fig = px.bar(chart_df, x="group", y="value", color="metric", barmode="group", range_y=[0,1], title="Metrics by group", color_discrete_sequence=["#0F2B46","#00B3A6","#6b8e9f","#a7b8c8"])
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("**How to read this:** compare the bars across groups. Large differences deserve investigation, but context, sample size, label quality, and the use case matter.")
            else:
                st.warning(fair["message"])

    with drift_tab:
        st.subheader("Feature drift")
        if drift_upload is not None:
            new_df = pd.read_csv(drift_upload)
        if new_df is None:
            st.info("Upload a new-data CSV in the sidebar to compare it with the reference features.")
            st.markdown("In demo mode, a synthetic comparison is available automatically.")
        if mode == "Demo data" or drift_upload is not None:
            ref = reference_df.copy()
            new = new_df.copy()
            common = [c for c in ref.columns if c in new.columns]
            if common:
                drift = drift_report(ref[common], new[common])
                ddf = pd.DataFrame(drift["rows"])
                if not ddf.empty:
                    st.dataframe(ddf, use_container_width=True, hide_index=True)
                    fig = px.bar(ddf, x="feature", y="psi", color="status", title="Population Stability Index by feature", color_discrete_map={"Green":"#00B3A6","Amber":"#e6a23c","Red":"#d9534f","Unavailable":"#94a3b8"})
                    st.plotly_chart(fig, use_container_width=True)
                    selected_feature = st.selectbox("Inspect feature distributions", common)
                    plot_df = pd.concat([ref[[selected_feature]].assign(dataset="Reference"), new[[selected_feature]].assign(dataset="New")], ignore_index=True)
                    if pd.api.types.is_numeric_dtype(ref[selected_feature]) and pd.api.types.is_numeric_dtype(new[selected_feature]):
                        fig = px.histogram(plot_df, x=selected_feature, color="dataset", barmode="overlay", histnorm="probability density", opacity=0.55, title=f"{selected_feature}: reference vs new")
                    else:
                        fig = px.histogram(plot_df, x=selected_feature, color="dataset", barmode="group", histnorm="probability", title=f"{selected_feature}: reference vs new")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Drift indicates that feature distributions changed. It does not automatically mean model performance worsened.")
                else:
                    st.info("No shared features were found.")
            else:
                st.warning("The new dataset has no feature columns in common with the reference data.")

    with explain_tab:
        st.subheader("Explainability")
        if mode == "Upload dataset and train baseline":
            try:
                result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, scoring="f1")
                imp = pd.DataFrame({"feature": X_test.columns, "importance": result.importances_mean}).sort_values("importance", ascending=True)
                fig = px.bar(imp, x="importance", y="feature", orientation="h", title="Permutation importance (F1 score)", color_discrete_sequence=["#00B3A6"])
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("**How to read this:** features with larger positive importance tend to contribute more to the model's F1 score in this evaluation. Correlated features and small test sets can make these estimates unstable.")
            except Exception as exc:
                st.warning(f"Permutation importance could not be computed for this dataset: {exc}")
        else:
            st.info("Permutation importance is available in dataset-upload baseline mode. For uploaded predictions, feature attribution requires the model or additional explanation data.")

    with report_tab:
        st.subheader("Risk report")
        flags = []
        flags.append({"area":"Performance","status":"Green" if perf.get("f1", 0) >= 0.8 else "Amber" if perf.get("f1", 0) >= 0.6 else "Red","finding":f"F1 score: {perf.get('f1'):.3f}" if perf.get("f1") is not None else "F1 score unavailable"})
        if "fair" in locals() and fair.get("available"):
            numeric_gaps_for_status = [v for k, v in fair["gaps"].items()
                                       if k in {"selection_rate", "tpr", "fpr", "precision"}
                                       and isinstance(v, (int, float)) and pd.notna(v)]
            status_rank = {"Unavailable": -1, "Green": 0, "Amber": 1, "Red": 2}
            worst = max((status_from_gap(v, amber_gap_pp, red_gap_pp) for v in numeric_gaps_for_status),
                        key=lambda x: status_rank.get(x, -1), default="Unavailable")
            numeric_gaps = {k: v for k, v in fair["gaps"].items()
                            if k in {"selection_rate", "tpr", "fpr", "precision"}
                            and isinstance(v, (int, float)) and pd.notna(v)}
            flags.append({"area":"Fairness","status":worst,"finding":"Largest observed gaps: " +
                          (", ".join(f"{k}={float(v):.3f}" for k, v in numeric_gaps.items()) or "not available")})
        if mode == "Demo data" or drift_upload is not None:
            if "drift" in locals() and drift.get("rows"):
                rank = {"Green":0,"Amber":1,"Red":2,"Unavailable":-1}
                worst = max((r["status"] for r in drift["rows"]), key=lambda s: rank.get(s,-1))
                flags.append({"area":"Drift","status":worst,"finding":f"{sum(r['status']=='Red' for r in drift['rows'])} feature(s) flagged red; {sum(r['status']=='Amber' for r in drift['rows'])} amber."})
        report_html = build_html_report(perf, flags, source_note)
        st.download_button("⬇️ Download self-contained HTML report", data=report_html, file_name="risklens_report.html", mime="text/html")
        st.markdown("**Limitations**")
        st.markdown("- Small or unrepresentative samples can produce unstable estimates.\n- Baseline mode uses a simple logistic regression model and does not tune it.\n- Group gaps are not proof of discrimination.\n- Drift thresholds are heuristics; statistical significance is not practical significance.\n- This tool does not certify legal or regulatory compliance.")

except (ValueError, KeyError, TypeError, pd.errors.ParserError) as exc:
    st.error(f"Could not evaluate this input: {exc}. Check the CSV columns and values, then try again.")
except Exception as exc:
    st.error(f"RiskLens encountered an issue while processing the data: {exc}")
