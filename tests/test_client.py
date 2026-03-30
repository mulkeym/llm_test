import pytest
from unittest.mock import patch, MagicMock
from src.client import LLMClient


def test_client_init():
    client = LLMClient(base_url="http://localhost:8080", api_key="test")
    assert client.base_url == "http://localhost:8080"


def test_build_tools_payload():
    client = LLMClient(base_url="http://localhost:8080")
    tool_defs = [
        {
            "name": "restart_service",
            "description": "Restart a service",
            "parameters": {
                "type": "object",
                "properties": {
                    "service_name": {"type": "string"},
                },
                "required": ["service_name"],
            },
        }
    ]
    payload = client._build_tools_payload(tool_defs)
    assert len(payload) == 1
    assert payload[0]["type"] == "function"
    assert payload[0]["function"]["name"] == "restart_service"


def test_extract_tool_calls():
    client = LLMClient(base_url="http://localhost:8080")
    mock_message = MagicMock()
    mock_tc = MagicMock()
    mock_tc.id = "call_123"
    mock_tc.function.name = "restart_service"
    mock_tc.function.arguments = '{"service_name": "apache2"}'
    mock_message.tool_calls = [mock_tc]

    calls = client._extract_tool_calls(mock_message)
    assert len(calls) == 1
    assert calls[0]["name"] == "restart_service"
    assert calls[0]["arguments"]["service_name"] == "apache2"


def test_extract_tool_calls_none():
    client = LLMClient(base_url="http://localhost:8080")
    mock_message = MagicMock()
    mock_message.tool_calls = None

    calls = client._extract_tool_calls(mock_message)
    assert calls == []
