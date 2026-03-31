import json
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

# Load test definitions for metadata (description, expected results)
@st.cache_data
def get_test_definitions():
    tests = load_tests("tests")
    return {t["name"]: t for t in tests}

test_defs = get_test_definitions()

def make_run_label(r):
    """Create unique label for each run using model name + full timestamp."""
    ts = r["timestamp"][:19].replace("T", " ")
    score = r.get("composite_score")
    score_str = " | {:.1f}%".format(score) if score is not None else " | incomplete"
    return "{} ({}{})".format(r["model_name"], ts, score_str)

page = st.sidebar.selectbox("View", ["Leaderboard", "Model Detail", "Test Results", "Compare", "History"])

if page == "Leaderboard":
    st.title("Model Leaderboard")
    runs = db.get_all_runs()
    if not runs:
        st.info("No benchmark runs yet. Run: `python -m src --model 'your-model'`")
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
        run_options = {make_run_label(r): r["id"] for r in runs}
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

        # Summary counts
        total_tests = len(results)
        passed = sum(1 for r in results if r["max_score"] > 0 and (r["score"] / r["max_score"] * 100) >= 70)
        failed = sum(1 for r in results if r["max_score"] > 0 and (r["score"] / r["max_score"] * 100) < 70)
        errors = sum(1 for r in results if r["max_score"] <= 0 or r["score"] == 0)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tests", total_tests)
        col2.metric("Passed (>=70%)", passed)
        col3.metric("Failed (<70%)", failed)
        col4.metric("Errors/Timeouts", errors)

elif page == "Test Results":
    st.title("Test Results Detail")
    runs = db.get_all_runs()
    if not runs:
        st.info("No runs yet.")
    else:
        run_options = {make_run_label(r): r["id"] for r in runs}
        selected = st.selectbox("Select run", list(run_options.keys()))
        run_id = run_options[selected]
        results = db.get_results_by_run(run_id)

        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            show_filter = st.selectbox("Show", ["All Tests", "Failed Only", "Passed Only", "Errors Only"])
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

        def get_status(r):
            pct = round(r["score"] / r["max_score"] * 100, 1) if r["max_score"] > 0 else 0
            transcript = r["transcript"]
            if isinstance(transcript, str):
                try:
                    transcript = json.loads(transcript)
                except (json.JSONDecodeError, TypeError):
                    transcript = []
            # Check for error/timeout
            if isinstance(transcript, list) and len(transcript) > 0:
                last = transcript[-1] if transcript else {}
                if isinstance(last, dict) and "error" in last:
                    return "error"
            if r["score"] == 0 and r["max_score"] > 0:
                return "error"
            if pct >= 70:
                return "passed"
            return "failed"

        if show_filter == "Failed Only":
            filtered = [r for r in filtered if get_status(r) == "failed"]
        elif show_filter == "Passed Only":
            filtered = [r for r in filtered if get_status(r) == "passed"]
        elif show_filter == "Errors Only":
            filtered = [r for r in filtered if get_status(r) == "error"]

        # Summary bar
        st.markdown("**Showing {} of {} tests**".format(len(filtered), len(results)))
        st.markdown("---")

        for r in filtered:
            pct = round(r["score"] / r["max_score"] * 100, 1) if r["max_score"] > 0 else 0
            status = get_status(r)
            test_name = r["test_name"]
            test_def = test_defs.get(test_name, {})

            # Status badge
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

            with st.expander(header, expanded=(status != "passed")):
                # Test intent / description
                description = test_def.get("description", "No description available")
                prompt = test_def.get("prompt", "N/A")

                st.markdown("#### Test Intent")
                st.markdown("**Description:** {}".format(description))
                st.markdown("**Prompt sent to model:**")
                st.code(prompt, language=None)

                # Expected result
                st.markdown("#### Expected Result")
                category = r["category"]

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

                # What the LLM actually did — with failure highlighting
                st.markdown("#### What the LLM Did")
                try:
                    transcript = json.loads(r["transcript"]) if isinstance(r["transcript"], str) else r["transcript"]
                except (json.JSONDecodeError, TypeError):
                    transcript = r["transcript"]

                if isinstance(transcript, list) and transcript:
                    # Check for error
                    last = transcript[-1] if transcript else {}
                    if isinstance(last, dict) and "error" in last:
                        st.error("**Error:** {}".format(last["error"]))
                    else:
                        if category == "tool_calling":
                            # Extract actual tool calls from transcript
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
                                # Compare expected vs actual side by side
                                expected_names = [ec["tool"] for ec in expected_calls]
                                actual_names = [mc["tool"] for mc in model_calls]

                                # Check ordering
                                actual_in_expected_order = [a for a in actual_names if a in expected_names]
                                order_correct = actual_in_expected_order == expected_names

                                # Hallucination check
                                hallucinated = [a for a in actual_names if a not in available]
                                missing_tools = [e for e in expected_names if e not in actual_names]
                                extra_tools = [a for a in actual_names if a not in expected_names and a in available]

                                # Summary issues
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

                                # Per-call comparison
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

                                            # Highlight match/mismatch
                                            if exp and act["tool"] == exp["tool"]:
                                                # Tool name matches — check params
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
                                                # Wrong tool
                                                st.error("`{}` — expected `{}`".format(act["tool"], exp["tool"]))
                                                st.code(act_params, language="json")
                                            else:
                                                # Extra call
                                                st.warning("`{}` — extra call (not expected)".format(act["tool"]))
                                                st.code(act_params, language="json")
                                        else:
                                            st.markdown("*Model called:*")
                                            st.error("No call made — expected `{}`".format(exp["tool"] if exp else "?"))

                            if model_responses:
                                st.markdown("**Model's final response:**")
                                st.markdown("> {}".format(model_responses[-1][:500]))

                        elif category == "knowledge":
                            # Show the model's answer
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

                                # Check judge criteria coverage
                                criteria = test_def.get("judge_criteria", [])
                                if criteria and status != "passed":
                                    st.markdown("**Criteria check (approximate):**")
                                    answer_lower = model_answer.lower()
                                    for c in criteria:
                                        # Extract key terms from the criterion
                                        # Simple heuristic: check if key words appear in response
                                        key_terms = []
                                        # Look for quoted or parenthesized values
                                        import re
                                        paren_vals = re.findall(r'\(([^)]+)\)', c)
                                        for val in paren_vals:
                                            key_terms.append(val.strip().lower())
                                        if key_terms:
                                            found = any(term in answer_lower for term in key_terms)
                                        else:
                                            # Fall back to checking a few significant words
                                            words = [w.lower() for w in c.split() if len(w) > 4]
                                            found = sum(1 for w in words if w in answer_lower) > len(words) * 0.5
                                        if found:
                                            st.markdown("- {} {}".format(c, ""))
                                        else:
                                            st.warning("Possibly missing: {}".format(c))
                            else:
                                st.error("Model produced no response.")

                        elif category == "context":
                            # Show question/response pairs with match highlighting
                            questions = test_def.get("questions", [])
                            if isinstance(transcript, list):
                                for idx, item in enumerate(transcript):
                                    if isinstance(item, dict) and "question" in item:
                                        q_text = item["question"]
                                        response = item.get("response", "No response")

                                        # Find matching expected answer
                                        expected_answer = ""
                                        if idx < len(questions):
                                            q_def = questions[idx]
                                            if isinstance(q_def, dict):
                                                expected_answer = q_def.get("expected", "")

                                        st.markdown("**Q:** {}".format(q_text))
                                        st.markdown("**Expected:** `{}`".format(expected_answer))
                                        st.markdown("**Model answered:** {}".format(response[:500]))

                                        # Highlight if expected answer appears in response
                                        if expected_answer:
                                            if expected_answer.lower() in response.lower():
                                                st.success("Contains expected answer")
                                            else:
                                                st.error("Expected answer `{}` NOT found in response".format(expected_answer))
                                        st.markdown("---")
                                        st.markdown("---")

                elif isinstance(transcript, dict) and "error" in transcript:
                    st.error("**Error:** {}".format(transcript["error"]))
                else:
                    st.warning("No transcript data available.")

                # Raw transcript toggle
                with st.popover("View raw transcript"):
                    st.json(transcript)

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
