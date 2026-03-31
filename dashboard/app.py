import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.db import Database

st.set_page_config(page_title="LLM Benchmark Dashboard", layout="wide")
db = Database("results.db")

page = st.sidebar.selectbox("View", ["Leaderboard", "Model Detail", "Compare", "History"])

if page == "Leaderboard":
    st.title("Model Leaderboard")
    runs = db.get_all_runs()
    if not runs:
        st.info("No benchmark runs yet. Run: python -m src --model 'your-model'")
    else:
        rows = []
        for run in runs:
            results = db.get_results_by_run(run["id"])
            cat_scores = {}
            for cat in ["tool_calling", "knowledge", "context"]:
                cat_results = [r for r in results if r["category"] == cat]
                if cat_results:
                    total = sum(r["score"] for r in cat_results)
                    max_total = sum(r["max_score"] for r in cat_results)
                    cat_scores[cat] = round(total / max_total * 100, 1) if max_total > 0 else 0
                else:
                    cat_scores[cat] = None
            rows.append({
                "Model": run["model_name"],
                "Tool Calling %": cat_scores.get("tool_calling"),
                "Knowledge %": cat_scores.get("knowledge"),
                "Context %": cat_scores.get("context"),
                "Composite %": run["composite_score"],
                "Date": run["timestamp"][:10],
                "run_id": run["id"],
            })
        df = pd.DataFrame(rows).sort_values("Composite %", ascending=False)
        st.dataframe(df.drop(columns=["run_id"]), use_container_width=True, hide_index=True)

elif page == "Model Detail":
    st.title("Model Detail")
    runs = db.get_all_runs()
    if not runs:
        st.info("No runs yet.")
    else:
        run_options = {"{} ({})".format(r["model_name"], r["timestamp"][:10]): r["id"] for r in runs}
        selected = st.selectbox("Select run", list(run_options.keys()))
        run_id = run_options[selected]
        results = db.get_results_by_run(run_id)

        # Score by tier
        tier_data = {}
        for r in results:
            tier = r["tier"]
            cat = r["category"]
            key = "{} T{}".format(cat, tier)
            if key not in tier_data:
                tier_data[key] = {"score": 0, "max": 0}
            tier_data[key]["score"] += r["score"]
            tier_data[key]["max"] += r["max_score"]

        if tier_data:
            tier_df = pd.DataFrame([
                {"Category": k, "Score %": round(v["score"] / v["max"] * 100, 1) if v["max"] > 0 else 0}
                for k, v in tier_data.items()
            ])
            fig = px.bar(tier_df, x="Category", y="Score %", title="Score by Category & Tier",
                        color="Score %", color_continuous_scale="RdYlGn", range_color=[0, 100])
            st.plotly_chart(fig, use_container_width=True)

        # Radar chart by category
        cat_scores = {}
        for cat in ["tool_calling", "knowledge", "context"]:
            cat_results = [r for r in results if r["category"] == cat]
            if cat_results:
                total = sum(r["score"] for r in cat_results)
                max_total = sum(r["max_score"] for r in cat_results)
                cat_scores[cat] = round(total / max_total * 100, 1) if max_total > 0 else 0

        if cat_scores:
            fig = go.Figure(data=go.Scatterpolar(
                r=list(cat_scores.values()),
                theta=list(cat_scores.keys()),
                fill="toself",
                name="Score %",
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                title="Category Radar",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Per-test breakdown
        st.subheader("Per-Test Results")
        for r in results:
            pct = round(r["score"] / r["max_score"] * 100, 1) if r["max_score"] > 0 else 0
            icon = "PASS" if pct >= 70 else "FAIL"
            with st.expander("[{}] {} - {:.1f}/{:.1f} ({:.1f}%)".format(icon, r["test_name"], r["score"], r["max_score"], pct)):
                try:
                    transcript = json.loads(r["transcript"]) if isinstance(r["transcript"], str) else r["transcript"]
                    st.json(transcript)
                except (json.JSONDecodeError, TypeError):
                    st.text(str(r["transcript"]))

elif page == "Compare":
    st.title("Compare Models")
    runs = db.get_all_runs()
    if len(runs) < 2:
        st.info("Need at least 2 runs to compare.")
    else:
        run_options = {"{} ({})".format(r["model_name"], r["timestamp"][:10]): r["id"] for r in runs}
        selected = st.multiselect("Select runs to compare (2-3)", list(run_options.keys()), max_selections=3)
        if len(selected) >= 2:
            compare_data = []
            for s in selected:
                rid = run_options[s]
                results = db.get_results_by_run(rid)
                for cat in ["tool_calling", "knowledge", "context"]:
                    cat_results = [r for r in results if r["category"] == cat]
                    if cat_results:
                        total = sum(r["score"] for r in cat_results)
                        max_total = sum(r["max_score"] for r in cat_results)
                        pct = round(total / max_total * 100, 1) if max_total > 0 else 0
                    else:
                        pct = 0
                    compare_data.append({"Model": s.split(" (")[0], "Category": cat, "Score %": pct})
            df = pd.DataFrame(compare_data)
            fig = px.bar(df, x="Category", y="Score %", color="Model", barmode="group", title="Model Comparison")
            st.plotly_chart(fig, use_container_width=True)

            # Delta table
            st.subheader("Score Differences")
            pivot = df.pivot(index="Category", columns="Model", values="Score %").reset_index()
            st.dataframe(pivot, use_container_width=True, hide_index=True)

elif page == "History":
    st.title("Model History")
    runs = db.get_all_runs()
    model_names = sorted(set(r["model_name"] for r in runs))
    if not model_names:
        st.info("No runs yet.")
    else:
        selected_model = st.selectbox("Select model", model_names)
        model_runs = [r for r in runs if r["model_name"] == selected_model]
        if len(model_runs) < 2:
            st.info("Need at least 2 runs of this model to show history.")
        else:
            history = [{"Date": r["timestamp"][:10], "Composite %": r["composite_score"]} for r in model_runs]
            df = pd.DataFrame(history)
            fig = px.line(df, x="Date", y="Composite %", title="{} - Score Over Time".format(selected_model), markers=True)
            st.plotly_chart(fig, use_container_width=True)

db.close()
