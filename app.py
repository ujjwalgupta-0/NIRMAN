
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="NIRMAN AI", page_icon="🏗️", layout="wide")

DATA = Path(__file__).parent / "data"
hist = pd.read_csv(DATA / "prototype_project_history.csv")
public = pd.read_csv(DATA / "paimana_public_snapshot.csv")

FEATURES = [
    "planned_progress_pct","actual_progress_pct","expenditure_ratio_pct",
    "schedule_delay_days","cost_escalation_pct","milestone_slips","open_issues"
]

X = hist[FEATURES]
y = hist["high_risk"]
model = RandomForestClassifier(
    n_estimators=220, max_depth=7, min_samples_leaf=2, random_state=42, class_weight="balanced"
)
model.fit(X, y)

def risk_label(p):
    if p >= .75: return "CRITICAL"
    if p >= .55: return "HIGH"
    if p >= .30: return "MODERATE"
    return "LOW"

def risk_factors(row):
    vals = {
        "Progress gap": max(0, row["planned_progress_pct"] - row["actual_progress_pct"]),
        "Schedule delay": row["schedule_delay_days"] / 30,
        "Cost escalation": row["cost_escalation_pct"],
        "Milestone slippage": row["milestone_slips"] * 3,
        "Open issues": row["open_issues"] * 2,
    }
    return sorted(vals.items(), key=lambda x: x[1], reverse=True)[:3]

st.title("🏗️ NIRMAN AI")
st.caption("Infrastructure Project Monitoring & Early-Warning Decision Support — Internal PoC")

st.info(
    "PoC note: project identities/cost/expenditure fields are based on a public PAIMANA snapshot. "
    "The monthly progress/risk training history is synthetic augmentation created for this prototype. "
    "It is NOT official OCMS historical data and must be replaced with authorized historical data for production."
)

# Sidebar
st.sidebar.header("Project / Scenario")
codes = public["project_code"].astype(str).tolist()
names = dict(zip(public["project_code"].astype(str), public["project_name"]))
code = st.sidebar.selectbox("Select project", codes, format_func=lambda x: f"{x} — {names[x][:48]}")
pbase = public[public.project_code.astype(str)==code].iloc[0]
latest = hist[hist.project_code.astype(str)==code].iloc[-1]

scenario = st.sidebar.radio("Mode", ["Current project snapshot", "What-if simulator"])

if scenario == "Current project snapshot":
    vals = {f: float(latest[f]) for f in FEATURES}
else:
    st.sidebar.markdown("### Change project conditions")
    vals = {}
    vals["planned_progress_pct"] = st.sidebar.slider("Planned progress (%)", 5.0, 95.0, float(latest.planned_progress_pct), 1.0)
    vals["actual_progress_pct"] = st.sidebar.slider("Actual progress (%)", 1.0, 95.0, float(latest.actual_progress_pct), 1.0)
    vals["expenditure_ratio_pct"] = st.sidebar.slider("Expenditure / planned cost (%)", 1.0, 100.0, float(latest.expenditure_ratio_pct), 1.0)
    vals["schedule_delay_days"] = st.sidebar.slider("Schedule delay (days)", 0, 720, int(latest.schedule_delay_days), 10)
    vals["cost_escalation_pct"] = st.sidebar.slider("Cost escalation (%)", 0.0, 80.0, float(latest.cost_escalation_pct), 1.0)
    vals["milestone_slips"] = st.sidebar.slider("Milestone slips", 0, 12, int(latest.milestone_slips))
    vals["open_issues"] = st.sidebar.slider("Open issues", 0, 20, int(latest.open_issues))

pred = model.predict_proba(pd.DataFrame([vals])[FEATURES])[0,1]
label = risk_label(pred)

# Metrics
c1,c2,c3,c4 = st.columns(4)
c1.metric("Risk score", f"{pred*100:.0f}%")
c2.metric("Risk level", label)
c3.metric("Original cost", f"₹{pbase.original_cost_cr:,.0f} Cr")
c4.metric("Expenditure", f"₹{pbase.expenditure_cr:,.2f} Cr")

st.markdown("---")

left, right = st.columns([1.35, 1])
with left:
    st.subheader("Project Risk Radar")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred*100,
        number={"suffix":"%"},
        gauge={"axis":{"range":[0,100]},
            "threshold":{"line":{"width":4},"thickness":0.8,"value":pred*100}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20,r=20,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Why is this project risky?")
    factors = risk_factors(pd.Series(vals))
    for name, score in factors:
        st.write(f"**{name}** — {score:.1f}")
    if label in ["HIGH","CRITICAL"]:
        st.error(f"🚨 Early warning: {label} risk threshold crossed.")
    else:
        st.success("No high-risk threshold crossed in this scenario.")

st.subheader("Planned vs Actual Progress")
trend = hist[hist.project_code.astype(str)==code].copy()
trend = trend.sort_values("month_index")
fig2 = px.line(trend, x="month_index", y=["planned_progress_pct","actual_progress_pct"],
               markers=True, labels={"value":"Progress (%)","month_index":"Monitoring period","variable":"Series"})
fig2.update_layout(height=330, legend_title_text="")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Portfolio Risk Ranking")
rank_rows = []
for c in public.project_code.astype(str):
    last = hist[hist.project_code.astype(str)==c].iloc[-1]
    pp = model.predict_proba(pd.DataFrame([last[FEATURES]])[FEATURES])[0,1]
    rank_rows.append([c, names[c], pp*100, risk_label(pp)])
rank = pd.DataFrame(rank_rows, columns=["Project Code","Project","Risk","Level"]).sort_values("Risk", ascending=False)
st.dataframe(rank.style.format({"Risk":"{:.0f}%"}), use_container_width=True, hide_index=True)

with st.expander("Prototype architecture / implementation"):
    st.markdown("""
**Data → Feature engineering → Random Forest risk model → Explainability → Early-warning threshold → Dashboard**

- Frontend: Streamlit + Plotly
- ML: scikit-learn Random Forest
- Data: public PAIMANA snapshot + clearly labelled synthetic time-series augmentation
- Production direction: replace prototype augmentation with authorized OCMS/PAIMANA historical monthly data; add SHAP, real milestone data, authenticated APIs and role-based workflows.
""")

st.caption("NIRMAN AI PoC • Built for SIH 2026 internal demonstration")
