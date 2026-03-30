from difflib import SequenceMatcher
from typing import Dict, List


class ToolCallScorer:
    def score(
        self,
        expected: List[dict],
        actual: List[dict],
        available_tools: List[str],
        tier: int,
    ) -> Dict[str, float]:
        result = {
            "correct_tool": 0,
            "correct_params": 0,
            "no_hallucinated_tools": 0,
            "correct_ordering": 0,
            "parallel_calls": 0,
            "error_recovery": 0,
            "total": 0,
            "max_total": 0,
        }

        max_total = 10  # base: 5 (tool) + 3 (params) + 2 (hallucination)
        if tier >= 2:
            max_total += 3  # ordering
        if tier >= 3:
            max_total += 2  # parallel
        result["max_total"] = max_total

        if not actual:
            return result

        # Check hallucinated tools
        all_actual_names = [a["name"] for a in actual]
        hallucinated = any(name not in available_tools for name in all_actual_names)
        result["no_hallucinated_tools"] = 0 if hallucinated else 2

        # Match expected to actual tool calls
        tool_score = 0
        param_score = 0
        for i, exp in enumerate(expected):
            matched = [a for a in actual if a["name"] == exp["tool"]]
            if matched:
                tool_score += 5
                best_param = max(
                    (self._score_params(exp.get("params", {}), m.get("arguments", {})) for m in matched),
                    default=0,
                )
                param_score += best_param * 3

        if expected:
            result["correct_tool"] = round(tool_score / len(expected), 2)
            result["correct_params"] = round(param_score / len(expected), 2)

        # Ordering (tier 2+)
        if tier >= 2 and len(expected) > 1:
            expected_order = [e["tool"] for e in expected]
            actual_order = [a["name"] for a in actual if a["name"] in expected_order]
            if actual_order == expected_order:
                result["correct_ordering"] = 3

        result["total"] = round(sum(
            result[k] for k in ["correct_tool", "correct_params", "no_hallucinated_tools", "correct_ordering", "parallel_calls", "error_recovery"]
        ), 2)

        return result

    def _score_params(self, expected: dict, actual: dict) -> float:
        if not expected:
            return 1.0
        matches = 0
        for key, exp_val in expected.items():
            act_val = actual.get(key)
            if act_val is None:
                continue
            if str(exp_val).lower() == str(act_val).lower():
                matches += 1
            elif isinstance(exp_val, str) and isinstance(act_val, str):
                ratio = SequenceMatcher(None, exp_val.lower(), act_val.lower()).ratio()
                if ratio > 0.8:
                    matches += ratio
        return matches / len(expected)
