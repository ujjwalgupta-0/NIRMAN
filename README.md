# NIRMAN AI — SIH 2026 Internal PoC

A small working proof-of-concept for infrastructure project risk monitoring.

## What this PoC demonstrates
1. Project data ingestion
2. Progress/cost/schedule risk features
3. ML-based project risk score
4. Risk classification: Low / Moderate / High / Critical
5. "Why risky?" contributing-factor view
6. Planned vs actual progress trend
7. Portfolio risk ranking
8. What-if simulator that changes project conditions and recomputes risk
9. Early-warning threshold

## Data honesty
- `data/paimana_public_snapshot.csv` contains a selected public snapshot of project records visible on the MoSPI PAIMANA public dashboard (May 2026).
- `data/prototype_project_history.csv` contains synthetic monthly augmentation generated from that public snapshot for a PoC. It is NOT official OCMS historical data.
- For the post-selection version, replace the synthetic history with authorized historical monthly OCMS/PAIMANA data.

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## 90-second demo flow
1. Open dashboard and show portfolio risk ranking.
2. Select a project.
3. Show current risk score and "Why is this project risky?"
4. Switch to What-if simulator.
5. Increase delay / progress gap / cost escalation.
6. Show risk rising and the early-warning alert.
7. End on portfolio ranking and architecture.

## Important
This is a PoC, not a production forecasting system. Do not claim that the synthetic model is trained on official historical project outcomes.
