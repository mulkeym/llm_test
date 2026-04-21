import json
from typing import List, Dict, Optional


class Judge:
    def __init__(
        self,
        provider: str = "anthropic",
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-6",
        base_url: Optional[str] = None,
        auth_type: str = "api_key",
        auth_token: Optional[str] = None,
        max_retries: int = 5,
    ):
        self.provider = provider
        self.model = model

        if provider == "anthropic":
            import anthropic
            if auth_type == "oauth":
                self.client = anthropic.Anthropic(auth_token=auth_token, max_retries=max_retries)
            else:
                self.client = anthropic.Anthropic(api_key=api_key, max_retries=max_retries)
        elif provider == "openai":
            from openai import OpenAI
            kwargs = {"api_key": api_key or "not-needed", "max_retries": max_retries}
            if base_url:
                kwargs["base_url"] = base_url if base_url.endswith("/v1") else base_url + "/v1"
            self.client = OpenAI(**kwargs)
        else:
            raise ValueError("Unknown judge provider: {}. Use 'anthropic' or 'openai'.".format(provider))

    def _build_knowledge_prompt(
        self, question: str, response: str, criteria: List[str]
    ) -> str:
        criteria_text = "\n".join("- {}".format(c) for c in criteria)
        return """You are a technical knowledge judge. Score the following LLM response.

QUESTION:
{question}

LLM RESPONSE:
{response}

EVALUATION CRITERIA:
{criteria}

Score each dimension 1-10:
- accuracy (40% weight): Are the facts correct?
- completeness (25% weight): Does it cover all criteria?
- reasoning (20% weight): Is the logic sound?
- clarity (15% weight): Is it well-explained?

Respond with ONLY valid JSON:
{{"scores": {{"accuracy": N, "completeness": N, "reasoning": N, "clarity": N}}, "weighted_total": N.NN, "explanation": "brief explanation"}}

Calculate weighted_total as: accuracy*0.4 + completeness*0.25 + reasoning*0.2 + clarity*0.15""".format(
            question=question, response=response, criteria=criteria_text
        )

    def _build_context_prompt(
        self, question: str, response: str, expected: str
    ) -> str:
        return """You are judging whether an LLM correctly retrieved a specific fact from a long document.

QUESTION: {question}
EXPECTED ANSWER: {expected}
LLM RESPONSE: {response}

Does the LLM response contain the correct answer? The answer may be paraphrased but must be factually equivalent.

Respond with ONLY valid JSON:
{{"correct": true/false, "score": N, "explanation": "brief explanation"}}

Score 10 if exact match, 7-9 if correct but paraphrased, 1-6 if partially correct, 0 if wrong.""".format(
            question=question, expected=expected, response=response
        )

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
            "explanation": "Failed to parse judge response: {}".format(raw[:200]),
        }

    def _call(self, prompt: str, max_tokens: int = 500) -> str:
        if self.provider == "anthropic":
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        else:
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""

    def judge_knowledge(
        self, question: str, response: str, criteria: List[str]
    ) -> dict:
        prompt = self._build_knowledge_prompt(question, response, criteria)
        raw = self._call(prompt, max_tokens=500)
        return self._parse_judge_response(raw)

    def judge_context(
        self, question: str, response: str, expected: str
    ) -> dict:
        prompt = self._build_context_prompt(question, response, expected)
        raw = self._call(prompt, max_tokens=300)
        return self._parse_judge_response(raw)
