import pytest
from src.judge import Judge


def test_build_knowledge_prompt():
    judge = Judge(api_key="test")
    prompt = judge._build_knowledge_prompt(
        question="What is CIDR?",
        response="CIDR is Classless Inter-Domain Routing...",
        criteria=["Correct definition", "Mentions prefix notation"],
    )
    assert "What is CIDR?" in prompt
    assert "Correct definition" in prompt


def test_build_context_prompt():
    judge = Judge(api_key="test")
    prompt = judge._build_context_prompt(
        question="What IP was the server migrated to?",
        response="The server was migrated to 10.22.7.45",
        expected="10.22.7.45",
    )
    assert "10.22.7.45" in prompt


def test_parse_judge_response_valid():
    judge = Judge(api_key="test")
    raw = '{"scores": {"accuracy": 8, "completeness": 7, "reasoning": 9, "clarity": 8}, "weighted_total": 8.05, "explanation": "Good"}'
    result = judge._parse_judge_response(raw)
    assert result["weighted_total"] == 8.05


def test_parse_judge_response_invalid():
    judge = Judge(api_key="test")
    result = judge._parse_judge_response("not json")
    assert result["weighted_total"] == 0.0
    assert "parse" in result["explanation"].lower()
