import json
from typing import Dict, List, Optional
from openai import OpenAI


class LLMClient:
    def __init__(
        self,
        base_url: str = "http://192.168.1.181:8080",
        api_key: str = "not-needed",
        timeout: int = 60,
        max_retries: int = 2,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = OpenAI(
            base_url=f"{base_url}/v1",
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    def _build_tools_payload(self, tool_defs: List[Dict]) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {}),
                },
            }
            for t in tool_defs
        ]

    def _extract_tool_calls(self, message) -> List[Dict]:
        if not message.tool_calls:
            return []
        calls = []
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": args,
                }
            )
        return calls

    def chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        model: str = "default",
    ) -> Dict:
        kwargs = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = self._build_tools_payload(tools)
        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        return {
            "role": "assistant",
            "content": message.content,
            "tool_calls": self._extract_tool_calls(message),
            "raw": message,
        }

    def chat_with_tools(
        self,
        messages: List[Dict],
        tools: List[Dict],
        simulated_responses: Dict,
        model: str = "default",
        max_turns: int = 10,
    ) -> List[Dict]:
        """Multi-turn conversation with simulated tool execution."""
        transcript = list(messages)
        for _ in range(max_turns):
            result = self.chat(transcript, tools, model)
            transcript.append(self._to_message(result))
            if not result["tool_calls"]:
                break
            for tc in result["tool_calls"]:
                tool_name = tc["name"]
                sim = simulated_responses.get(
                    tool_name, {"error": f"Unknown tool: {tool_name}"}
                )
                transcript.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(sim),
                    }
                )
        return transcript

    def _to_message(self, result: Dict) -> Dict:
        msg = {"role": "assistant"}
        if result["content"]:
            msg["content"] = result["content"]
        if result["tool_calls"]:
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    },
                }
                for tc in result["tool_calls"]
            ]
        return msg
