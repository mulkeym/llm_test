import json
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.db import Database
from src.loader import load_tests

st.set_page_config(page_title="LLM Benchmark Dashboard", layout="wide")
db = Database("results.db")


@st.cache_data
def get_test_definitions():
    tests = load_tests("tests")
    return {t["name"]: t for t in tests}


test_defs = get_test_definitions()


def make_run_label(r):
    ts = r["timestamp"][:19].replace("T", " ")
    score = r.get("composite_score")
    score_str = " | {:.1f}%".format(score) if score is not None else " | incomplete"
    return "{} ({}{})".format(r["model_name"], ts, score_str)


def get_status(r):
    pct = round(r["score"] / r["max_score"] * 100, 1) if r["max_score"] > 0 else 0
    transcript = r["transcript"]
    if isinstance(transcript, str):
        try:
            transcript = json.loads(transcript)
        except (json.JSONDecodeError, TypeError):
            transcript = []
    if isinstance(transcript, list) and len(transcript) > 0:
        last = transcript[-1] if transcript else {}
        if isinstance(last, dict) and "error" in last:
            return "error"
    if r["score"] == 0 and r["max_score"] > 0:
        return "error"
    if pct >= 70:
        return "passed"
    return "failed"


def render_test_detail(r, test_def, status):
    """Render the expanded detail for a single test result."""
    category = r["category"]
    description = test_def.get("description", "No description available")
    prompt = test_def.get("prompt", "N/A")

    st.markdown("#### Test Intent")
    st.markdown("**Description:** {}".format(description))
    st.markdown("**Prompt sent to model:**")
    st.code(prompt, language=None)

    # Expected result
    st.markdown("#### Expected Result")

    if category == "tool_calling":
        expected_calls = test_def.get("expected_tool_calls", [])
        if expected_calls:
            st.markdown("**Expected tool calls (in order):**")
            for i, ec in enumerate(expected_calls, 1):
                params_str = ", ".join("{}={}".format(k, v) for k, v in ec.get("params", {}).items())
                st.markdown("{}. `{}({})`".format(i, ec["tool"], params_str))
        else:
            st.markdown("No expected tool calls defined.")

    elif category == "knowledge":
        criteria = test_def.get("judge_criteria", [])
        if criteria:
            st.markdown("**Judge criteria (what the answer should cover):**")
            for c in criteria:
                st.markdown("- {}".format(c))

    elif category == "context":
        needles = test_def.get("needle", [])
        questions = test_def.get("questions", [])
        if needles:
            st.markdown("**Facts embedded in document (needles):**")
            for n in needles:
                st.markdown("- Position {:.0%}: _{}_".format(n["position"], n["text"]))
        if questions:
            st.markdown("**Questions & expected answers:**")
            for q in questions:
                if isinstance(q, dict):
                    st.markdown("- Q: {} → **{}**".format(
                        q.get("question", q.get("text", "")),
                        q.get("expected", "N/A")
                    ))

    # What the LLM actually did
    st.markdown("#### What the LLM Did")
    try:
        transcript = json.loads(r["transcript"]) if isinstance(r["transcript"], str) else r["transcript"]
    except (json.JSONDecodeError, TypeError):
        transcript = r["transcript"]

    if isinstance(transcript, list) and transcript:
        last = transcript[-1] if transcript else {}
        if isinstance(last, dict) and "error" in last:
            st.error("**Error:** {}".format(last["error"]))
        else:
            if category == "tool_calling":
                model_calls = []
                model_responses = []
                available = test_def.get("available_tools", [])
                expected_calls = test_def.get("expected_tool_calls", [])

                for msg in transcript:
                    if isinstance(msg, dict):
                        if msg.get("role") == "assistant" and msg.get("tool_calls"):
                            for tc in msg["tool_calls"]:
                                fn = tc.get("function", tc)
                                name = fn.get("name", "unknown")
                                args = fn.get("arguments", {})
                                if isinstance(args, str):
                                    try:
                                        args = json.loads(args)
                                    except json.JSONDecodeError:
                                        pass
                                model_calls.append({"tool": name, "params": args})
                        elif msg.get("role") == "assistant" and msg.get("content"):
                            model_responses.append(msg["content"])

                if not model_calls:
                    st.error("Model made **no tool calls** — expected {} call(s).".format(len(expected_calls)))
                else:
                    expected_names = [ec["tool"] for ec in expected_calls]
                    actual_names = [mc["tool"] for mc in model_calls]
                    actual_in_expected_order = [a for a in actual_names if a in expected_names]
                    order_correct = actual_in_expected_order == expected_names
                    hallucinated = [a for a in actual_names if a not in available]
                    missing_tools = [e for e in expected_names if e not in actual_names]
                    extra_tools = [a for a in actual_names if a not in expected_names and a in available]

                    issues = []
                    if missing_tools:
                        issues.append("Missing tools: **{}**".format(", ".join("`{}`".format(t) for t in missing_tools)))
                    if hallucinated:
                        issues.append("Hallucinated tools (not in available set): **{}**".format(", ".join("`{}`".format(t) for t in hallucinated)))
                    if extra_tools:
                        issues.append("Extra tools called (not expected): **{}**".format(", ".join("`{}`".format(t) for t in extra_tools)))
                    if not order_correct and len(expected_calls) > 1:
                        issues.append("Wrong ordering — expected: {} | got: {}".format(
                            " → ".join("`{}`".format(n) for n in expected_names),
                            " → ".join("`{}`".format(n) for n in actual_names),
                        ))

                    if issues:
                        st.markdown("**Issues found:**")
                        for issue in issues:
                            st.warning(issue)

                    st.markdown("**Step-by-step comparison:**")
                    max_steps = max(len(expected_calls), len(model_calls))
                    for i in range(max_steps):
                        exp = expected_calls[i] if i < len(expected_calls) else None
                        act = model_calls[i] if i < len(model_calls) else None

                        st.markdown("---")
                        st.markdown("**Step {}**".format(i + 1))
                        col_exp, col_act = st.columns(2)

                        with col_exp:
                            if exp:
                                st.markdown("*Expected:*")
                                params_str = ", ".join("{}={}".format(k, v) for k, v in exp.get("params", {}).items())
                                st.code("{}({})".format(exp["tool"], params_str), language=None)
                            else:
                                st.markdown("*Expected:*")
                                st.info("No more expected calls")

                        with col_act:
                            if act:
                                st.markdown("*Model called:*")
                                act_params = json.dumps(act["params"], indent=2) if isinstance(act["params"], dict) else str(act["params"])

                                if exp and act["tool"] == exp["tool"]:
                                    exp_params = exp.get("params", {})
                                    act_p = act["params"] if isinstance(act["params"], dict) else {}
                                    param_issues = []
                                    for pk, pv in exp_params.items():
                                        av = act_p.get(pk)
                                        if av is None:
                                            param_issues.append("missing `{}`".format(pk))
                                        elif str(av).lower() != str(pv).lower():
                                            param_issues.append("`{}`: expected `{}`, got `{}`".format(pk, pv, av))

                                    if param_issues:
                                        st.code("{}({})".format(act["tool"], act_params), language="json")
                                        for pi in param_issues:
                                            st.warning("Param mismatch: {}".format(pi))
                                    else:
                                        st.success("`{}` — correct".format(act["tool"]))
                                        st.code(act_params, language="json")
                                elif exp:
                                    st.error("`{}` — expected `{}`".format(act["tool"], exp["tool"]))
                                    st.code(act_params, language="json")
                                else:
                                    st.warning("`{}` — extra call (not expected)".format(act["tool"]))
                                    st.code(act_params, language="json")
                            else:
                                st.markdown("*Model called:*")
                                st.error("No call made — expected `{}`".format(exp["tool"] if exp else "?"))

                if model_responses:
                    st.markdown("**Model's final response:**")
                    st.markdown("> {}".format(model_responses[-1][:500]))

            elif category == "knowledge":
                model_answer = None
                for msg in transcript:
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        content = msg.get("content", "")
                        if content:
                            model_answer = content
                            break

                if model_answer:
                    st.markdown("**Model's answer:**")
                    st.markdown(model_answer)

                    criteria = test_def.get("judge_criteria", [])
                    if criteria and status != "passed":
                        st.markdown("**Criteria check (approximate):**")
                        answer_lower = model_answer.lower()
                        for c in criteria:
                            key_terms = []
                            paren_vals = re.findall(r'\(([^)]+)\)', c)
                            for val in paren_vals:
                                key_terms.append(val.strip().lower())
                            if key_terms:
                                found = any(term in answer_lower for term in key_terms)
                            else:
                                words = [w.lower() for w in c.split() if len(w) > 4]
                                found = sum(1 for w in words if w in answer_lower) > len(words) * 0.5
                            if found:
                                st.markdown("- {}".format(c))
                            else:
                                st.warning("Possibly missing: {}".format(c))
                else:
                    st.error("Model produced no response.")

            elif category == "context":
                questions = test_def.get("questions", [])
                if isinstance(transcript, list):
                    for idx, item in enumerate(transcript):
                        if isinstance(item, dict) and "question" in item:
                            q_text = item["question"]
                            response = item.get("response", "No response")

                            expected_answer = ""
                            if idx < len(questions):
                                q_def = questions[idx]
                                if isinstance(q_def, dict):
                                    expected_answer = q_def.get("expected", "")

                            st.markdown("**Q:** {}".format(q_text))
                            st.markdown("**Expected:** `{}`".format(expected_answer))
                            st.markdown("**Model answered:** {}".format(response[:500]))

                            if expected_answer:
                                if expected_answer.lower() in response.lower():
                                    st.success("Contains expected answer")
                                else:
                                    st.error("Expected answer `{}` NOT found in response".format(expected_answer))
                            st.markdown("---")

    elif isinstance(transcript, dict) and "error" in transcript:
        st.error("**Error:** {}".format(transcript["error"]))
    else:
        st.warning("No transcript data available.")

    with st.popover("View raw transcript"):
        st.json(transcript)


# ── Page Navigation ──
page = st.sidebar.selectbox("View", ["Dashboard", "Model Detail", "Compare", "History"])

if page == "Dashboard":
    st.title("LLM Benchmark Dashboard")
    runs = db.get_all_runs()
    if not runs:
        st.info("No benchmark runs yet. Run: `python -m src --model 'your-model'`")
    else:
        # ── Leaderboard ──
        st.subheader("Leaderboard")
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

        # ── Test Results ──
        st.markdown("---")
        st.subheader("Test Results")

        # Build run options sorted by composite score (top model first)
        sorted_runs = sorted(runs, key=lambda r: r.get("composite_score") or 0, reverse=True)
        run_options = {make_run_label(r): r["id"] for r in sorted_runs}
        run_labels = list(run_options.keys())

        selected = st.selectbox("Select run", run_labels, index=0)
        run_id = run_options[selected]
        results = db.get_results_by_run(run_id)

        # Summary metrics
        total_tests = len(results)
        passed_count = sum(1 for r in results if get_status(r) == "passed")
        failed_count = sum(1 for r in results if get_status(r) == "failed")
        error_count = sum(1 for r in results if get_status(r) == "error")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tests", total_tests)
        col2.metric("Passed (>=70%)", passed_count)
        col3.metric("Failed (<70%)", failed_count)
        col4.metric("Errors/Timeouts", error_count)

        # Filters — default to "Failed Only"
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_options = ["Failed Only", "All Tests", "Passed Only", "Errors Only"]
            show_filter = st.selectbox("Show", filter_options, index=0)
        with col2:
            categories = sorted(set(r["category"] for r in results))
            cat_filter = st.selectbox("Category", ["All"] + categories)
        with col3:
            tiers = sorted(set(r["tier"] for r in results))
            tier_filter = st.selectbox("Tier", ["All"] + [str(t) for t in tiers])

        # Apply filters
        filtered = results
        if cat_filter != "All":
            filtered = [r for r in filtered if r["category"] == cat_filter]
        if tier_filter != "All":
            filtered = [r for r in filtered if r["tier"] == int(tier_filter)]

        if show_filter == "Failed Only":
            filtered = [r for r in filtered if get_status(r) == "failed"]
        elif show_filter == "Passed Only":
            filtered = [r for r in filtered if get_status(r) == "passed"]
        elif show_filter == "Errors Only":
            filtered = [r for r in filtered if get_status(r) == "error"]

        st.markdown("**Showing {} of {} tests**".format(len(filtered), len(results)))

        if not filtered:
            if show_filter == "Failed Only":
                st.success("No failed tests! Try selecting 'All Tests' to see full results.")
            elif show_filter == "Errors Only":
                st.success("No errors! Try selecting 'All Tests' to see full results.")
            else:
                st.info("No tests match the current filters.")
        else:
            for r in filtered:
                pct = round(r["score"] / r["max_score"] * 100, 1) if r["max_score"] > 0 else 0
                status = get_status(r)
                test_name = r["test_name"]
                test_def = test_defs.get(test_name, {})

                if status == "error":
                    badge = "ERROR"
                    color = "red"
                elif status == "failed":
                    badge = "FAIL"
                    color = "orange"
                else:
                    badge = "PASS"
                    color = "green"

                header = ":{}: **[{}]** {} — {:.1f}/{:.1f} ({:.1f}%) | {} | Tier {}".format(
                    color, badge, test_name, r["score"], r["max_score"], pct, r["category"], r["tier"]
                )

                with st.expander(header, expanded=False):
                    render_test_detail(r, test_def, status)

elif page == "Model Detail":
    st.title("Model Detail")
    runs = db.get_all_runs()
    if not runs:
        st.info("No runs yet.")
    else:
        sorted_runs = sorted(runs, key=lambda r: r.get("composite_score") or 0, reverse=True)
        run_options = {make_run_label(r): r["id"] for r in sorted_runs}
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

        # Radar chart
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

        # Summary counts
        total_tests = len(results)
        passed = sum(1 for r in results if get_status(r) == "passed")
        failed = sum(1 for r in results if get_status(r) == "failed")
        errors = sum(1 for r in results if get_status(r) == "error")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tests", total_tests)
        col2.metric("Passed (>=70%)", passed)
        col3.metric("Failed (<70%)", failed)
        col4.metric("Errors/Timeouts", errors)

elif page == "Compare":
    st.title("Compare Models")
    runs = db.get_all_runs()
    if len(runs) < 2:
        st.info("Need at least 2 runs to compare.")
    else:
        run_options = {make_run_label(r): r["id"] for r in runs}
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
