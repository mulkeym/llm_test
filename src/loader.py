import yaml
from pathlib import Path
from typing import Dict, List, Optional


def load_tools(tools_dir: str = "tools") -> Dict[str, dict]:
    tools = {}
    for f in Path(tools_dir).glob("*.yaml"):
        with open(f) as fh:
            tool = yaml.safe_load(fh)
            tools[tool["name"]] = tool
    return tools


def load_tests(tests_dir: str = "tests") -> List[dict]:
    test_cases = []
    for f in Path(tests_dir).rglob("*.yaml"):
        with open(f) as fh:
            tc = yaml.safe_load(fh)
            if tc and isinstance(tc, dict) and "name" in tc and "category" in tc:
                tc["_source"] = str(f)
                test_cases.append(tc)
    return test_cases


def filter_tests(
    test_cases: List[dict],
    tiers: Optional[List[int]] = None,
    categories: Optional[List[str]] = None,
    domains: Optional[List[str]] = None,
) -> List[dict]:
    filtered = test_cases
    if tiers:
        filtered = [t for t in filtered if t.get("tier") in tiers]
    if categories:
        filtered = [t for t in filtered if t.get("category") in categories]
    if domains:
        filtered = [t for t in filtered if t.get("domain") in domains or t.get("category") != "knowledge"]
    return filtered
