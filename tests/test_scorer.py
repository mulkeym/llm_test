import pytest
from src.scorer import ToolCallScorer


def test_correct_tool_and_params():
    scorer = ToolCallScorer()
    result = scorer.score(
        expected=[{"tool": "restart_service", "params": {"service_name": "apache2", "host": "web-prod-01"}}],
        actual=[{"name": "restart_service", "arguments": {"service_name": "apache2", "host": "web-prod-01"}}],
        available_tools=["restart_service", "check_logs"],
        tier=1,
    )
    assert result["correct_tool"] == 5
    assert result["correct_params"] == 3
    assert result["no_hallucinated_tools"] == 2


def test_wrong_tool():
    scorer = ToolCallScorer()
    result = scorer.score(
        expected=[{"tool": "restart_service", "params": {}}],
        actual=[{"name": "check_logs", "arguments": {}}],
        available_tools=["restart_service", "check_logs"],
        tier=1,
    )
    assert result["correct_tool"] == 0


def test_hallucinated_tool():
    scorer = ToolCallScorer()
    result = scorer.score(
        expected=[{"tool": "restart_service", "params": {}}],
        actual=[{"name": "nonexistent_tool", "arguments": {}}],
        available_tools=["restart_service", "check_logs"],
        tier=1,
    )
    assert result["no_hallucinated_tools"] == 0


def test_partial_params():
    scorer = ToolCallScorer()
    result = scorer.score(
        expected=[{"tool": "restart_service", "params": {"service_name": "apache2", "host": "web-prod-01"}}],
        actual=[{"name": "restart_service", "arguments": {"service_name": "apache2"}}],
        available_tools=["restart_service"],
        tier=1,
    )
    assert result["correct_tool"] == 5
    assert 0 < result["correct_params"] < 3


def test_ordering_tier2():
    scorer = ToolCallScorer()
    result = scorer.score(
        expected=[
            {"tool": "check_service_status", "params": {"service_name": "apache2"}},
            {"tool": "restart_service", "params": {"service_name": "apache2"}},
        ],
        actual=[
            {"name": "check_service_status", "arguments": {"service_name": "apache2"}},
            {"name": "restart_service", "arguments": {"service_name": "apache2"}},
        ],
        available_tools=["check_service_status", "restart_service"],
        tier=2,
    )
    assert result["correct_ordering"] == 3


def test_wrong_ordering_tier2():
    scorer = ToolCallScorer()
    result = scorer.score(
        expected=[
            {"tool": "check_service_status", "params": {}},
            {"tool": "restart_service", "params": {}},
        ],
        actual=[
            {"name": "restart_service", "arguments": {}},
            {"name": "check_service_status", "arguments": {}},
        ],
        available_tools=["check_service_status", "restart_service"],
        tier=2,
    )
    assert result["correct_ordering"] == 0


def test_no_tool_calls():
    scorer = ToolCallScorer()
    result = scorer.score(
        expected=[{"tool": "restart_service", "params": {}}],
        actual=[],
        available_tools=["restart_service"],
        tier=1,
    )
    assert result["correct_tool"] == 0
    assert result["total"] == 0
