import pytest
import yaml
from src.loader import load_tools, load_tests, filter_tests


@pytest.fixture
def tools_dir(tmp_path):
    tool = {
        "name": "test_tool",
        "description": "A test tool",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }
    (tmp_path / "test_tool.yaml").write_text(yaml.dump(tool))
    return str(tmp_path)


@pytest.fixture
def tests_dir(tmp_path):
    tc_dir = tmp_path / "tool_calling" / "tier1_basic"
    tc_dir.mkdir(parents=True)
    tc = {"name": "test_case", "category": "tool_calling", "tier": 1, "prompt": "hi"}
    (tc_dir / "test_case.yaml").write_text(yaml.dump(tc))
    return str(tmp_path)


def test_load_tools(tools_dir):
    tools = load_tools(tools_dir)
    assert "test_tool" in tools


def test_load_tests(tests_dir):
    tests = load_tests(tests_dir)
    assert len(tests) == 1
    assert tests[0]["name"] == "test_case"


def test_filter_by_tier(tests_dir):
    tests = load_tests(tests_dir)
    assert len(filter_tests(tests, tiers=[1])) == 1
    assert len(filter_tests(tests, tiers=[2])) == 0


def test_filter_by_category(tests_dir):
    tests = load_tests(tests_dir)
    assert len(filter_tests(tests, categories=["tool_calling"])) == 1
    assert len(filter_tests(tests, categories=["knowledge"])) == 0
