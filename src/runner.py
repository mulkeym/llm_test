import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv

from .client import LLMClient
from .context import ContextGenerator
from .db import Database
from .judge import Judge
from .loader import load_tools, load_tests, filter_tests
from .scorer import ToolCallScorer


class BenchmarkRunner:
    def __init__(self, config_path: str = "config.yaml"):
        load_dotenv()
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        api_cfg = self.config["target_api"]
        self.client = LLMClient(
            base_url=api_cfg["base_url"],
            api_key=api_cfg.get("api_key", ""),
            timeout=api_cfg.get("timeout", 60),
            max_retries=api_cfg.get("max_retries", 2),
        )
        self.judge = Judge(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model=self.config["judge"].get("model", "claude-sonnet-4-6-20250514"),
        )
        self.db = Database()
        self.scorer = ToolCallScorer()
        self.context_gen = ContextGenerator(seed=42)
        self.tools = load_tools("tools")

    def run(self, model_name: str, export_format: Optional[str] = None):
        run_cfg = self.config["run"]
        test_cases = load_tests("tests")
        test_cases = filter_tests(
            test_cases,
            tiers=run_cfg.get("tiers"),
            categories=run_cfg.get("categories"),
            domains=run_cfg.get("domains") or None,
        )

        if not test_cases:
            print("No test cases found matching config filters.")
            return

        run_id = self.db.create_run(model_name, self.config)
        results_summary = {"tool_calling": [], "knowledge": [], "context": []}  # type: Dict[str, List[dict]]

        print("\n" + "=" * 60)
        print("  LLM Benchmark - Model: {}".format(model_name))
        print("  Tests: {}".format(len(test_cases)))
        print("=" * 60 + "\n")

        for i, tc in enumerate(test_cases, 1):
            category = tc["category"]
            name = tc["name"]
            print("[{}/{}] {}/{} (tier {})...".format(i, len(test_cases), category, name, tc.get("tier", "?")), end=" ", flush=True)

            try:
                if category == "tool_calling":
                    score, max_score, transcript = self._run_tool_test(tc, model_name)
                elif category == "knowledge":
                    score, max_score, transcript = self._run_knowledge_test(tc, model_name)
                elif category == "context":
                    score, max_score, transcript = self._run_context_test(tc, model_name)
                else:
                    print("SKIP (unknown category)")
                    continue

                pct = (score / max_score * 100) if max_score > 0 else 0
                print("{:.1f}/{:.1f} ({:.0f}%)".format(score, max_score, pct))

                self.db.save_result(
                    run_id, name, category, tc.get("tier", 0), score, max_score, transcript
                )
                results_summary[category].append({"score": score, "max_score": max_score})

            except Exception as e:
                print("ERROR: {}".format(e))
                self.db.save_result(run_id, name, category, tc.get("tier", 0), 0, 1, [{"error": str(e)}])
                results_summary[category].append({"score": 0, "max_score": 1})

        composite = self._compute_composite(results_summary)
        self.db.update_run_score(run_id, composite)
        self._print_summary(results_summary, composite)

        if export_format:
            from .export import export_run
            export_run(self.db, run_id, export_format)

    def _run_tool_test(self, tc: dict, model_name: str) -> Tuple[float, float, list]:
        available_tool_defs = [self.tools[t] for t in tc["available_tools"] if t in self.tools]
        messages = [
            {"role": "system", "content": "You are an IT operations assistant. Use the provided tools to help the user. Call tools when appropriate."},
            {"role": "user", "content": tc["prompt"]},
        ]
        transcript = self.client.chat_with_tools(
            messages=messages,
            tools=available_tool_defs,
            simulated_responses=tc.get("simulated_responses", {}),
            model=model_name,
        )

        # Extract actual tool calls from transcript
        actual_calls = []
        for msg in transcript:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc_call in msg["tool_calls"]:
                    fn = tc_call.get("function", {})
                    name = fn.get("name", tc_call.get("name", ""))
                    args_raw = fn.get("arguments", tc_call.get("arguments", {}))
                    if isinstance(args_raw, str):
                        try:
                            args = json.loads(args_raw)
                        except json.JSONDecodeError:
                            args = {}
                    else:
                        args = args_raw
                    actual_calls.append({"name": name, "arguments": args})

        score_result = self.scorer.score(
            expected=tc.get("expected_tool_calls", []),
            actual=actual_calls,
            available_tools=tc["available_tools"],
            tier=tc.get("tier", 1),
        )
        return score_result["total"], score_result["max_total"], transcript

    def _run_knowledge_test(self, tc: dict, model_name: str) -> Tuple[float, float, list]:
        messages = [
            {"role": "system", "content": "You are a knowledgeable IT professional. Answer technical questions accurately and thoroughly."},
            {"role": "user", "content": tc["prompt"]},
        ]
        result = self.client.chat(messages, model=model_name)
        transcript = messages + [{"role": "assistant", "content": result["content"]}]
        judgement = self.judge.judge_knowledge(
            question=tc["prompt"],
            response=result["content"] or "",
            criteria=tc.get("judge_criteria", []),
        )
        score = judgement.get("weighted_total", 0)
        return score, 10.0, transcript

    def _run_context_test(self, tc: dict, model_name: str) -> Tuple[float, float, list]:
        document = self.context_gen.build_context_document(
            filler_type=tc.get("filler", "runbook"),
            target_tokens=tc.get("context_size", 4000),
            needles=tc.get("needle", []),
        )
        questions = tc.get("questions", [])
        total_score = 0
        max_score = len(questions) * 10
        transcript = []

        for q in questions:
            if isinstance(q, dict):
                question_text = q.get("question", q.get("text", ""))
                expected = q.get("expected", "")
            else:
                question_text = str(q)
                expected = ""

            messages = [
                {"role": "system", "content": "Read the following document carefully and answer the question based only on the information in the document."},
                {"role": "user", "content": "DOCUMENT:\n{}\n\nQUESTION: {}".format(document, question_text)},
            ]
            result = self.client.chat(messages, model=model_name)
            response_text = result["content"] or ""
            transcript.append({"question": question_text, "response": response_text})

            judgement = self.judge.judge_context(question_text, response_text, expected)
            total_score += judgement.get("score", 0)

        return total_score, max_score if max_score > 0 else 1, transcript

    def _compute_composite(self, summary: dict) -> float:
        weights = self.config["scoring"]
        category_map = [
            ("tool_calling", "tool_weight"),
            ("knowledge", "knowledge_weight"),
            ("context", "context_weight"),
        ]
        composite = 0.0
        for cat, key in category_map:
            items = summary.get(cat, [])
            if items:
                total = sum(i["score"] for i in items)
                max_total = sum(i["max_score"] for i in items)
                pct = (total / max_total * 100) if max_total > 0 else 0
            else:
                pct = 0
            composite += pct * weights.get(key, 0)
        return round(composite, 2)

    def _print_summary(self, summary: dict, composite: float):
        print("\n" + "=" * 60)
        print("  RESULTS SUMMARY")
        print("=" * 60)
        for cat in ["tool_calling", "knowledge", "context"]:
            items = summary.get(cat, [])
            if items:
                total = sum(i["score"] for i in items)
                max_total = sum(i["max_score"] for i in items)
                pct = (total / max_total * 100) if max_total > 0 else 0
                print("  {:20s}: {:.1f}/{:.1f} ({:.1f}%)".format(cat, total, max_total, pct))
        print("  " + "-" * 38)
        print("  {:20s}: {:.1f}%".format("COMPOSITE", composite))
        print("=" * 60 + "\n")
