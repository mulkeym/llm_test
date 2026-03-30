import json
from typing import List, Dict, Optional
import anthropic


class Judge:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-6-20250514",
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def _build_knowledge_prompt(
        self, question: str, response: str, criteria: List[str]
    ) -> str:
        criteria_text = "\n".join(f"- {c}" for c in criteria)
        return f"""You are a technical knowledge judge. Score the following LLM response.

QUESTION:
{question}

LLM RESPONSE:
{response}

EVALUATION CRITERIA:
{criteria_text}

Score each dimension 1-10:
- accuracy (40% weight): Are the facts correct?
- completeness (25% weight): Does it cover all criteria?
- reasoning (20% weight): Is the logic sound?
- clarity (15% weight): Is it well-explained?

Respond with ONLY valid JSON:
{{"scores": {{"accuracy": N, "completeness": N, "reasoning": N, "clarity": N}}, "weighted_total": N.NN, "explanation": "brief explanation"}}

Calculate weighted_total as: accuracy*0.4 + completeness*0.25 + reasoning*0.2 + clarity*0.15"""

    def _build_context_prompt(
        self, question: str, response: str, expected: str
    ) -> str:
        return f"""You are judging whether an LLM correctly retrieved a specific fact from a long document.

QUESTION: {question}
EXPECTED ANSWER: {expected}
LLM RESPONSE: {response}

Does the LLM response contain the correct answer? The answer may be paraphrased but must be factually equivalent.

Respond with ONLY valid JSON:
{{"correct": true/false, "score": N, "explanation": "brief explanation"}}

Score 10 if exact match, 7-9 if correct but paraphrased, 1-6 if partially correct, 0 if wrong."""

    def _parse_judge_response(self, raw: str) -> dict:
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except (json.JSONDecodeError, ValueError):
            pass
        return {
            "scores": {"accuracy": 0, "completeness": 0, "reasoning": 0, "clarity": 0},
            "weighted_total": 0.0,
            "explanation": f"Failed to parse judge response: {raw[:200]}",
        }

    def judge_knowledge(
        self, question: str, response: str, criteria: List[str]
    ) -> dict:
        prompt = self._build_knowledge_prompt(question, response, criteria)
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._parse_judge_response(msg.content[0].text)

    def judge_context(
        self, question: str, response: str, expected: str
    ) -> dict:
        prompt = self._build_context_prompt(question, response, expected)
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return self._parse_judge_response(msg.content[0].text)
