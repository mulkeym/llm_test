# LLM Benchmark Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Python benchmarking tool that evaluates local LLMs on tool calling, knowledge depth, and context handling, with Claude as judge and a Streamlit dashboard.

**Architecture:** YAML-defined tests and tools, Python runner that talks to an OpenAI-compatible API, deterministic + Claude-judge scoring, SQLite persistence, Streamlit dashboard. Tests are data, not code.

**Tech Stack:** Python 3.10+, openai SDK, anthropic SDK, Streamlit, PyYAML, SQLite3, python-dotenv

---

### Task 1: Project Scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `config.yaml`
- Create: `src/__init__.py`
- Create: `dashboard/__init__.py`
- Create: `tests/` directory structure

**Step 1: Create requirements.txt**

```
openai>=1.0.0
anthropic>=0.40.0
streamlit>=1.30.0
pyyaml>=6.0
python-dotenv>=1.0.0
plotly>=5.18.0
pandas>=2.0.0
```

**Step 2: Create .gitignore**

```
results.db
.env
__pycache__/
*.pyc
.venv/
venv/
```

**Step 3: Create .env.example**

```
ANTHROPIC_API_KEY=your-key-here
```

**Step 4: Create config.yaml**

```yaml
target_api:
  base_url: "http://192.168.1.181:8080"
  api_key: ""
  timeout: 60
  max_retries: 2

judge:
  provider: anthropic
  model: claude-sonnet-4-6-20250514

scoring:
  tool_weight: 0.4
  knowledge_weight: 0.3
  context_weight: 0.3

run:
  tiers: [1, 2, 3]
  categories: [tool_calling, knowledge, context]
  domains: []
```

**Step 5: Create directory structure**

```bash
mkdir -p src tools tests/tool_calling/tier1_basic tests/tool_calling/tier2_intermediate tests/tool_calling/tier3_advanced tests/knowledge tests/context/tier1_short tests/context/tier2_medium tests/context/tier3_long dashboard
touch src/__init__.py dashboard/__init__.py
```

**Step 6: Commit**

```bash
git add -A && git commit -m "feat: project scaffolding — config, deps, directory structure"
```

---

### Task 2: SQLite Database Layer

**Files:**
- Create: `src/db.py`
- Create: `tests/test_db.py`

**Step 1: Write failing tests**

```python
# tests/test_db.py
import os
import tempfile
import pytest
from src.db import Database


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = Database(path)
    yield database
    database.close()
    os.unlink(path)


def test_create_run(db):
    run_id = db.create_run("test-model", {"tiers": [1, 2, 3]})
    assert run_id is not None
    assert isinstance(run_id, int)


def test_save_result(db):
    run_id = db.create_run("test-model", {})
    result_id = db.save_result(
        run_id=run_id,
        test_name="test_basic",
        category="tool_calling",
        tier=1,
        score=8.0,
        max_score=10.0,
        transcript=[{"role": "user", "content": "test"}],
    )
    assert result_id is not None


def test_save_judgement(db):
    run_id = db.create_run("test-model", {})
    result_id = db.save_result(run_id, "test_q", "knowledge", 1, 7.5, 10.0, [])
    db.save_judgement(result_id, "accuracy", 8, "Good answer")
    judgements = db.get_judgements(result_id)
    assert len(judgements) == 1
    assert judgements[0]["criterion"] == "accuracy"


def test_update_run_score(db):
    run_id = db.create_run("test-model", {})
    db.update_run_score(run_id, 85.5)
    run = db.get_run(run_id)
    assert run["composite_score"] == 85.5


def test_get_all_runs(db):
    db.create_run("model-a", {})
    db.create_run("model-b", {})
    runs = db.get_all_runs()
    assert len(runs) == 2


def test_get_results_by_run(db):
    run_id = db.create_run("test-model", {})
    db.save_result(run_id, "t1", "tool_calling", 1, 5.0, 10.0, [])
    db.save_result(run_id, "t2", "knowledge", 1, 8.0, 10.0, [])
    results = db.get_results_by_run(run_id)
    assert len(results) == 2


def test_get_results_by_category(db):
    run_id = db.create_run("test-model", {})
    db.save_result(run_id, "t1", "tool_calling", 1, 5.0, 10.0, [])
    db.save_result(run_id, "t2", "knowledge", 1, 8.0, 10.0, [])
    results = db.get_results_by_run(run_id, category="tool_calling")
    assert len(results) == 1
```

**Step 2: Run tests to verify they fail**

```bash
cd /Users/michaelmulkey/Documents/Repositories/llm_test
python -m pytest tests/test_db.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.db'`

**Step 3: Implement src/db.py**

```python
# src/db.py
import json
import sqlite3
from datetime import datetime, timezone


class Database:
    def __init__(self, path: str = "results.db"):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                config_snapshot TEXT,
                composite_score REAL
            );
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                test_name TEXT NOT NULL,
                category TEXT NOT NULL,
                tier INTEGER NOT NULL,
                score REAL NOT NULL,
                max_score REAL NOT NULL,
                transcript TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );
            CREATE TABLE IF NOT EXISTS judgements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                result_id INTEGER NOT NULL,
                criterion TEXT NOT NULL,
                score REAL NOT NULL,
                explanation TEXT,
                FOREIGN KEY (result_id) REFERENCES results(id)
            );
        """)

    def create_run(self, model_name: str, config: dict) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs (model_name, timestamp, config_snapshot) VALUES (?, ?, ?)",
            (model_name, datetime.now(timezone.utc).isoformat(), json.dumps(config)),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_run_score(self, run_id: int, composite_score: float):
        self.conn.execute(
            "UPDATE runs SET composite_score = ? WHERE id = ?",
            (composite_score, run_id),
        )
        self.conn.commit()

    def save_result(
        self,
        run_id: int,
        test_name: str,
        category: str,
        tier: int,
        score: float,
        max_score: float,
        transcript: list,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO results (run_id, test_name, category, tier, score, max_score, transcript) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, test_name, category, tier, score, max_score, json.dumps(transcript)),
        )
        self.conn.commit()
        return cur.lastrowid

    def save_judgement(
        self, result_id: int, criterion: str, score: float, explanation: str
    ):
        self.conn.execute(
            "INSERT INTO judgements (result_id, criterion, score, explanation) VALUES (?, ?, ?, ?)",
            (result_id, criterion, score, explanation),
        )
        self.conn.commit()

    def get_run(self, run_id: int) -> dict:
        row = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def get_all_runs(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM runs ORDER BY timestamp DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_results_by_run(
        self, run_id: int, category: str | None = None
    ) -> list[dict]:
        if category:
            rows = self.conn.execute(
                "SELECT * FROM results WHERE run_id = ? AND category = ?",
                (run_id, category),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM results WHERE run_id = ?", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_judgements(self, result_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM judgements WHERE result_id = ?", (result_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_db.py -v
```

Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add src/db.py tests/test_db.py && git commit -m "feat: SQLite database layer with full CRUD"
```

---

### Task 3: API Client

**Files:**
- Create: `src/client.py`
- Create: `tests/test_client.py`

**Step 1: Write failing tests**

```python
# tests/test_client.py
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
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_client.py -v
```

**Step 3: Implement src/client.py**

```python
# src/client.py
import json
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

    def _build_tools_payload(self, tool_defs: list[dict]) -> list[dict]:
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

    def _extract_tool_calls(self, message) -> list[dict]:
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
        messages: list[dict],
        tools: list[dict] | None = None,
        model: str = "default",
    ) -> dict:
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
        messages: list[dict],
        tools: list[dict],
        simulated_responses: dict,
        model: str = "default",
        max_turns: int = 10,
    ) -> list[dict]:
        """Multi-turn conversation with simulated tool execution."""
        transcript = list(messages)
        for _ in range(max_turns):
            result = self.chat(transcript, tools, model)
            transcript.append(self._to_message(result))
            if not result["tool_calls"]:
                break
            for tc in result["tool_calls"]:
                tool_name = tc["name"]
                sim = simulated_responses.get(tool_name, {"error": f"Unknown tool: {tool_name}"})
                transcript.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(sim),
                    }
                )
        return transcript

    def _to_message(self, result: dict) -> dict:
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
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_client.py -v
```

Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/client.py tests/test_client.py && git commit -m "feat: OpenAI-compatible API client with multi-turn tool support"
```

---

### Task 4: Tool Definitions (18 YAML files)

**Files:**
- Create: `tools/*.yaml` (18 files)

**Step 1: Create all 18 tool YAML files**

Create each tool file in `tools/` following the schema format from the design doc. Each file defines: name, description, parameters (type, properties, required).

Tools to create:
1. `query_database.yaml`
2. `restart_service.yaml`
3. `check_logs.yaml`
4. `create_ticket.yaml`
5. `run_script.yaml`
6. `lookup_user.yaml`
7. `check_service_status.yaml`
8. `send_notification.yaml`
9. `get_metrics.yaml`
10. `update_firewall_rule.yaml`
11. `dns_lookup.yaml`
12. `ping_host.yaml`
13. `traceroute.yaml`
14. `whois_lookup.yaml`
15. `subnet_calculator.yaml`
16. `manage_dns_record.yaml`
17. `check_port.yaml`
18. `get_arp_table.yaml`

**Step 2: Create tool loader utility**

```python
# src/loader.py
import os
import yaml
from pathlib import Path


def load_tools(tools_dir: str = "tools") -> dict[str, dict]:
    """Load all tool definitions from YAML files. Returns dict keyed by tool name."""
    tools = {}
    for f in Path(tools_dir).glob("*.yaml"):
        with open(f) as fh:
            tool = yaml.safe_load(fh)
            tools[tool["name"]] = tool
    return tools


def load_tests(tests_dir: str = "tests") -> list[dict]:
    """Load all test case YAML files from the tests directory tree."""
    test_cases = []
    for f in Path(tests_dir).rglob("*.yaml"):
        with open(f) as fh:
            tc = yaml.safe_load(fh)
            if tc and "name" in tc and "category" in tc:
                tc["_source"] = str(f)
                test_cases.append(tc)
    return test_cases


def filter_tests(
    test_cases: list[dict],
    tiers: list[int] | None = None,
    categories: list[str] | None = None,
    domains: list[str] | None = None,
) -> list[dict]:
    """Filter test cases by tier, category, and domain."""
    filtered = test_cases
    if tiers:
        filtered = [t for t in filtered if t.get("tier") in tiers]
    if categories:
        filtered = [t for t in filtered if t.get("category") in categories]
    if domains:
        filtered = [t for t in filtered if t.get("domain") in domains or t.get("category") != "knowledge"]
    return filtered
```

**Step 3: Write loader tests**

```python
# tests/test_loader.py
import tempfile
import os
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
    assert tools["test_tool"]["description"] == "A test tool"


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
```

**Step 4: Run tests**

```bash
python -m pytest tests/test_loader.py -v
```

**Step 5: Commit**

```bash
git add tools/ src/loader.py tests/test_loader.py && git commit -m "feat: 18 tool definitions + YAML loader with filtering"
```

---

### Task 5: Claude Judge

**Files:**
- Create: `src/judge.py`
- Create: `tests/test_judge.py`

**Step 1: Write failing tests**

```python
# tests/test_judge.py
import pytest
from unittest.mock import patch, MagicMock
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
```

**Step 2: Implement src/judge.py**

```python
# src/judge.py
import json
import anthropic


class Judge:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6-20250514",
    ):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def _build_knowledge_prompt(
        self, question: str, response: str, criteria: list[str]
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
        self, question: str, response: str, criteria: list[str]
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
```

**Step 3: Run tests**

```bash
python -m pytest tests/test_judge.py -v
```

**Step 4: Commit**

```bash
git add src/judge.py tests/test_judge.py && git commit -m "feat: Claude-as-judge for knowledge and context scoring"
```

---

### Task 6: Tool-Calling Scorer

**Files:**
- Create: `src/scorer.py`
- Create: `tests/test_scorer.py`

**Step 1: Write failing tests**

```python
# tests/test_scorer.py
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
```

**Step 2: Implement src/scorer.py**

```python
# src/scorer.py
from difflib import SequenceMatcher


class ToolCallScorer:
    def score(
        self,
        expected: list[dict],
        actual: list[dict],
        available_tools: list[str],
        tier: int,
    ) -> dict:
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
            else:
                result["correct_ordering"] = 0

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
```

**Step 3: Run tests**

```bash
python -m pytest tests/test_scorer.py -v
```

**Step 4: Commit**

```bash
git add src/scorer.py tests/test_scorer.py && git commit -m "feat: deterministic tool-calling scorer with tiered criteria"
```

---

### Task 7: Context Filler Generator

**Files:**
- Create: `src/context.py`
- Create: `tests/test_context.py`

**Step 1: Write failing tests**

```python
# tests/test_context.py
import pytest
from src.context import ContextGenerator


def test_generate_filler_runbook():
    gen = ContextGenerator()
    text = gen.generate_filler("runbook", target_tokens=500)
    assert len(text) > 100


def test_generate_filler_log_dump():
    gen = ContextGenerator()
    text = gen.generate_filler("log_dump", target_tokens=500)
    assert len(text) > 100


def test_insert_needles_single():
    gen = ContextGenerator()
    filler = "word " * 1000
    needles = [{"text": "SECRET_NEEDLE_123", "position": 0.5}]
    result = gen.insert_needles(filler, needles)
    assert "SECRET_NEEDLE_123" in result


def test_insert_needles_ordering():
    gen = ContextGenerator()
    filler = "word " * 1000
    needles = [
        {"text": "NEEDLE_A", "position": 0.25},
        {"text": "NEEDLE_B", "position": 0.75},
    ]
    result = gen.insert_needles(filler, needles)
    pos_a = result.find("NEEDLE_A")
    pos_b = result.find("NEEDLE_B")
    assert pos_a < pos_b


def test_build_context_document():
    gen = ContextGenerator()
    doc = gen.build_context_document(
        filler_type="runbook",
        target_tokens=500,
        needles=[{"text": "The secret port is 9999", "position": 0.5}],
    )
    assert "The secret port is 9999" in doc
```

**Step 2: Implement src/context.py**

```python
# src/context.py
import random


class ContextGenerator:
    RUNBOOK_LINES = [
        "## Server Maintenance Procedure",
        "1. Verify backup completion status on the NAS dashboard.",
        "2. Check disk utilization across all partitions using df -h.",
        "3. Review failed login attempts in /var/log/auth.log for the past 24 hours.",
        "4. Rotate application logs older than 7 days using logrotate.",
        "5. Confirm NTP synchronization on all production servers.",
        "6. Verify SSL certificate expiration dates for public-facing services.",
        "7. Check RAID array status using mdadm --detail /dev/md0.",
        "8. Review cron job execution logs for any failures.",
        "9. Test failover for the primary database cluster.",
        "10. Update package lists and apply security patches.",
        "11. Verify firewall rules match the approved baseline.",
        "12. Check active connections on load balancers.",
        "13. Review memory utilization trends over the past week.",
        "14. Validate DNS resolution for all critical internal services.",
        "15. Confirm automated backups completed for all PostgreSQL databases.",
        "16. Check Docker container health status on all hosts.",
        "17. Review and rotate API keys older than 90 days.",
        "18. Verify monitoring alert thresholds are correctly configured.",
        "19. Test disaster recovery runbook with a tabletop exercise.",
        "20. Document any changes made during this maintenance window.",
        "### Pre-Maintenance Checklist",
        "- Notify stakeholders at least 24 hours in advance.",
        "- Create a snapshot of all VMs before patching.",
        "- Ensure rollback procedures are documented and tested.",
        "- Verify that the on-call engineer is available.",
        "### Post-Maintenance Validation",
        "- Run smoke tests against all critical endpoints.",
        "- Verify all services are reporting healthy in monitoring.",
        "- Check that no alerts were triggered during the window.",
        "- Update the maintenance log with actions taken.",
        "### Network Checks",
        "- Verify BGP peering status with upstream providers.",
        "- Check interface error counters on core switches.",
        "- Validate VLAN configuration on trunk ports.",
        "- Test connectivity between all data center zones.",
    ]

    LOG_LINES = [
        "Mar 15 08:23:41 web-prod-01 sshd[12345]: Accepted publickey for admin from 10.0.1.50",
        "Mar 15 08:23:42 web-prod-01 systemd[1]: Starting Apache HTTP Server...",
        "Mar 15 08:23:43 web-prod-01 apache2[12400]: AH00558: Could not reliably determine FQDN",
        "Mar 15 08:24:01 db-prod-01 postgresql[5432]: LOG: checkpoint starting: time",
        "Mar 15 08:24:02 db-prod-01 postgresql[5432]: LOG: checkpoint complete: wrote 156 buffers",
        "Mar 15 08:24:15 lb-prod-01 haproxy[8080]: 10.0.2.15:43210 [15/Mar/2026:08:24:15] frontend~ backend/web-prod-01 0/0/1/15/16 200 2456",
        "Mar 15 08:25:00 mon-prod-01 prometheus[9090]: level=info msg=\"Scrape completed\" target=web-prod-01",
        "Mar 15 08:25:01 web-prod-02 nginx[1234]: 10.0.3.20 - - [15/Mar/2026:08:25:01] \"GET /api/health HTTP/1.1\" 200 15",
        "Mar 15 08:25:30 auth-prod-01 sshd[14000]: Failed password for invalid user test from 203.0.113.50",
        "Mar 15 08:26:00 db-prod-02 mysqld[3306]: InnoDB: Buffer pool hit rate 998 / 1000",
        "Mar 15 08:26:15 cache-prod-01 redis[6379]: DB saved on disk",
        "Mar 15 08:26:30 queue-prod-01 rabbitmq[5672]: accepting AMQP connection <0.1234.0> (10.0.4.10:51234 -> 10.0.4.20:5672)",
        "Mar 15 08:27:00 web-prod-01 kernel: [UFW BLOCK] IN=eth0 OUT= SRC=198.51.100.25 DST=10.0.1.10 PROTO=TCP DPT=22",
        "Mar 15 08:27:15 dns-prod-01 named[53]: client @0x7f8b3c query: internal.example.com IN A +",
        "Mar 15 08:27:30 mail-prod-01 postfix/smtp[2525]: connect to smtp.example.com[93.184.216.34]:25: Connection timed out",
        "Mar 15 08:28:00 vpn-prod-01 openvpn[1194]: client-01/10.8.0.6 PUSH: Received control message",
        "Mar 15 08:28:15 backup-prod-01 rsync[9999]: sent 1,234,567 bytes received 456 bytes total size 1,234,567",
        "Mar 15 08:28:30 ci-prod-01 jenkins[8080]: Build #1234 completed: SUCCESS",
        "Mar 15 08:29:00 k8s-master-01 kube-apiserver[6443]: I0315 08:29:00 handler.go:153] GET /api/v1/nodes: (1.234ms) 200",
    ]

    CONFIG_LINES = [
        "server {",
        "    listen 80;",
        "    server_name internal.example.com;",
        "    location / {",
        "        proxy_pass http://backend:8080;",
        "        proxy_set_header Host $host;",
        "        proxy_set_header X-Real-IP $remote_addr;",
        "    }",
        "    location /health {",
        "        return 200 'OK';",
        "    }",
        "}",
        "# Upstream configuration",
        "upstream backend {",
        "    server 10.0.1.10:8080 weight=5;",
        "    server 10.0.1.11:8080 weight=3;",
        "    server 10.0.1.12:8080 backup;",
        "}",
        "# Rate limiting zone",
        "limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;",
        "# SSL configuration",
        "ssl_protocols TLSv1.2 TLSv1.3;",
        "ssl_ciphers HIGH:!aNULL:!MD5;",
        "ssl_prefer_server_ciphers on;",
        "# Logging",
        "access_log /var/log/nginx/access.log combined;",
        "error_log /var/log/nginx/error.log warn;",
        "# Firewall rules (iptables export)",
        "-A INPUT -p tcp --dport 22 -s 10.0.0.0/8 -j ACCEPT",
        "-A INPUT -p tcp --dport 80 -j ACCEPT",
        "-A INPUT -p tcp --dport 443 -j ACCEPT",
        "-A INPUT -p tcp --dport 5432 -s 10.0.1.0/24 -j ACCEPT",
        "-A INPUT -j DROP",
    ]

    EMAIL_LINES = [
        "From: admin@example.com",
        "To: ops-team@example.com",
        "Subject: Re: Production deployment planned for Friday",
        "",
        "Team,",
        "",
        "Quick update on the deployment plan:",
        "- We'll start the rolling restart at 02:00 UTC.",
        "- The canary instance will get traffic first for 15 minutes.",
        "- If error rates stay below 0.1%, we proceed to full rollout.",
        "- Rollback plan: revert to tag v2.3.1 and restart all pods.",
        "",
        "Let me know if anyone has concerns.",
        "",
        "---",
        "From: devops@example.com",
        "To: ops-team@example.com",
        "Subject: Re: Database migration status",
        "",
        "The migration script ran for 47 minutes. All 12 million rows migrated.",
        "Spot-checked 1000 records, all look correct.",
        "Index rebuild completed. Query performance is back to baseline.",
        "",
        "---",
        "From: security@example.com",
        "To: ops-team@example.com",
        "Subject: Vulnerability scan results - March",
        "",
        "Scan completed. 3 medium findings:",
        "1. OpenSSH 8.2 on jump-host-01 (upgrade to 9.x recommended)",
        "2. TLS 1.0 still enabled on legacy-app-01",
        "3. Default SNMP community string on switch-floor3",
        "",
        "No critical findings. Remediation tickets created.",
    ]

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def generate_filler(self, filler_type: str, target_tokens: int) -> str:
        """Generate filler text of approximately target_tokens tokens."""
        lines_map = {
            "runbook": self.RUNBOOK_LINES,
            "log_dump": self.LOG_LINES,
            "config_dump": self.CONFIG_LINES,
            "email_thread": self.EMAIL_LINES,
        }
        source = lines_map.get(filler_type, self.RUNBOOK_LINES)
        # Rough estimate: 1 token ≈ 4 chars
        target_chars = target_tokens * 4
        result_lines = []
        current_chars = 0
        while current_chars < target_chars:
            line = self.rng.choice(source)
            result_lines.append(line)
            current_chars += len(line) + 1
        return "\n".join(result_lines)

    def insert_needles(self, filler: str, needles: list[dict]) -> str:
        """Insert needle texts at specified positions (0.0-1.0) in the filler."""
        lines = filler.split("\n")
        sorted_needles = sorted(needles, key=lambda n: n["position"], reverse=True)
        for needle in sorted_needles:
            pos = int(needle["position"] * len(lines))
            pos = max(0, min(pos, len(lines)))
            lines.insert(pos, needle["text"])
        return "\n".join(lines)

    def build_context_document(
        self,
        filler_type: str,
        target_tokens: int,
        needles: list[dict],
    ) -> str:
        filler = self.generate_filler(filler_type, target_tokens)
        return self.insert_needles(filler, needles)
```

**Step 3: Run tests**

```bash
python -m pytest tests/test_context.py -v
```

**Step 4: Commit**

```bash
git add src/context.py tests/test_context.py && git commit -m "feat: context filler generator with needle insertion"
```

---

### Task 8: Test Runner

**Files:**
- Create: `src/runner.py`
- Create: `src/__main__.py`

**Step 1: Implement src/runner.py**

```python
# src/runner.py
import argparse
import json
import os
import sys
from pathlib import Path

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

    def run(self, model_name: str, export_format: str | None = None):
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
        results_summary = {"tool_calling": [], "knowledge": [], "context": []}

        print(f"\n{'='*60}")
        print(f"  LLM Benchmark — Model: {model_name}")
        print(f"  Tests: {len(test_cases)}")
        print(f"{'='*60}\n")

        for i, tc in enumerate(test_cases, 1):
            category = tc["category"]
            name = tc["name"]
            print(f"[{i}/{len(test_cases)}] {category}/{name} (tier {tc.get('tier', '?')})...", end=" ", flush=True)

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
                print(f"{score:.1f}/{max_score:.1f} ({pct:.0f}%)")

                result_id = self.db.save_result(
                    run_id, name, category, tc.get("tier", 0), score, max_score, transcript
                )
                results_summary[category].append({"score": score, "max_score": max_score})

            except Exception as e:
                print(f"ERROR: {e}")
                self.db.save_result(run_id, name, category, tc.get("tier", 0), 0, 1, [{"error": str(e)}])
                results_summary[category].append({"score": 0, "max_score": 1})

        composite = self._compute_composite(results_summary)
        self.db.update_run_score(run_id, composite)
        self._print_summary(results_summary, composite)

        if export_format:
            self._export(run_id, export_format)

    def _run_tool_test(self, tc: dict, model_name: str) -> tuple[float, float, list]:
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
        actual_calls = []
        for msg in transcript:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc_call in msg["tool_calls"]:
                    fn = tc_call.get("function", {})
                    actual_calls.append({
                        "name": fn.get("name", tc_call.get("name", "")),
                        "arguments": json.loads(fn["arguments"]) if isinstance(fn.get("arguments"), str) else fn.get("arguments", tc_call.get("arguments", {})),
                    })

        score_result = self.scorer.score(
            expected=tc.get("expected_tool_calls", []),
            actual=actual_calls,
            available_tools=tc["available_tools"],
            tier=tc.get("tier", 1),
        )
        return score_result["total"], score_result["max_total"], transcript

    def _run_knowledge_test(self, tc: dict, model_name: str) -> tuple[float, float, list]:
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

    def _run_context_test(self, tc: dict, model_name: str) -> tuple[float, float, list]:
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
            question_text = q if isinstance(q, str) else q.get("question", "")
            expected = q.get("expected", "") if isinstance(q, dict) else ""

            messages = [
                {"role": "system", "content": "Read the following document carefully and answer the question based only on the information in the document."},
                {"role": "user", "content": f"DOCUMENT:\n{document}\n\nQUESTION: {question_text}"},
            ]
            result = self.client.chat(messages, model=model_name)
            response_text = result["content"] or ""
            transcript.append({"question": question_text, "response": response_text})

            judgement = self.judge.judge_context(question_text, response_text, expected)
            total_score += judgement.get("score", 0)

        return total_score, max_score, transcript

    def _compute_composite(self, summary: dict) -> float:
        weights = self.config["scoring"]
        scores = {}
        for cat, key in [("tool_calling", "tool_weight"), ("knowledge", "knowledge_weight"), ("context", "context_weight")]:
            items = summary.get(cat, [])
            if items:
                total = sum(i["score"] for i in items)
                max_total = sum(i["max_score"] for i in items)
                scores[cat] = (total / max_total * 100) if max_total > 0 else 0
            else:
                scores[cat] = 0

        composite = sum(scores.get(cat, 0) * weights.get(key, 0)
                        for cat, key in [("tool_calling", "tool_weight"), ("knowledge", "knowledge_weight"), ("context", "context_weight")])
        return round(composite, 2)

    def _print_summary(self, summary: dict, composite: float):
        print(f"\n{'='*60}")
        print("  RESULTS SUMMARY")
        print(f"{'='*60}")
        for cat in ["tool_calling", "knowledge", "context"]:
            items = summary.get(cat, [])
            if items:
                total = sum(i["score"] for i in items)
                max_total = sum(i["max_score"] for i in items)
                pct = (total / max_total * 100) if max_total > 0 else 0
                print(f"  {cat:20s}: {total:.1f}/{max_total:.1f} ({pct:.1f}%)")
        print(f"{'  ':-<40}")
        print(f"  {'COMPOSITE':20s}: {composite:.1f}%")
        print(f"{'='*60}\n")

    def _export(self, run_id: int, fmt: str):
        from .export import export_run
        export_run(self.db, run_id, fmt)
```

**Step 2: Create src/__main__.py**

```python
# src/__main__.py
import argparse
from .runner import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="LLM Benchmark Suite")
    parser.add_argument("--model", required=True, help="Model name for this run")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--export", choices=["json", "csv"], help="Export results")
    args = parser.parse_args()

    runner = BenchmarkRunner(config_path=args.config)
    runner.run(model_name=args.model, export_format=args.export)


if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add src/runner.py src/__main__.py && git commit -m "feat: benchmark runner with multi-turn tool execution and scoring"
```

---

### Task 9: Export Module

**Files:**
- Create: `src/export.py`
- Create: `tests/test_export.py`

**Step 1: Write failing tests**

```python
# tests/test_export.py
import json
import os
import tempfile
import pytest
from src.db import Database
from src.export import export_run, export_all


@pytest.fixture
def db_with_data():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = Database(path)
    run_id = db.create_run("test-model", {"tiers": [1]})
    db.save_result(run_id, "test1", "tool_calling", 1, 8.0, 10.0, [{"role": "user", "content": "hi"}])
    db.save_result(run_id, "test2", "knowledge", 1, 7.5, 10.0, [])
    db.update_run_score(run_id, 77.5)
    yield db, run_id, path
    db.close()
    os.unlink(path)


def test_export_json(db_with_data, tmp_path):
    db, run_id, _ = db_with_data
    outfile = str(tmp_path / "results.json")
    export_run(db, run_id, "json", outfile)
    with open(outfile) as f:
        data = json.load(f)
    assert data["model_name"] == "test-model"
    assert len(data["results"]) == 2


def test_export_csv(db_with_data, tmp_path):
    db, run_id, _ = db_with_data
    outfile = str(tmp_path / "results.csv")
    export_run(db, run_id, "csv", outfile)
    with open(outfile) as f:
        lines = f.readlines()
    assert len(lines) == 3  # header + 2 rows
```

**Step 2: Implement src/export.py**

```python
# src/export.py
import csv
import json

from .db import Database


def export_run(db: Database, run_id: int, fmt: str, output: str | None = None):
    run = db.get_run(run_id)
    results = db.get_results_by_run(run_id)

    if fmt == "json":
        data = {
            "model_name": run["model_name"],
            "timestamp": run["timestamp"],
            "composite_score": run["composite_score"],
            "results": [
                {
                    "test_name": r["test_name"],
                    "category": r["category"],
                    "tier": r["tier"],
                    "score": r["score"],
                    "max_score": r["max_score"],
                    "percentage": round(r["score"] / r["max_score"] * 100, 1) if r["max_score"] > 0 else 0,
                }
                for r in results
            ],
        }
        path = output or f"export_{run['model_name']}_{run_id}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Exported to {path}")

    elif fmt == "csv":
        path = output or f"export_{run['model_name']}_{run_id}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "test_name", "category", "tier", "score", "max_score", "percentage"])
            for r in results:
                pct = round(r["score"] / r["max_score"] * 100, 1) if r["max_score"] > 0 else 0
                writer.writerow([run["model_name"], r["test_name"], r["category"], r["tier"], r["score"], r["max_score"], pct])
        print(f"Exported to {path}")


def export_all(db: Database, fmt: str, output: str | None = None):
    runs = db.get_all_runs()
    if fmt == "json":
        data = []
        for run in runs:
            results = db.get_results_by_run(run["id"])
            data.append({
                "model_name": run["model_name"],
                "timestamp": run["timestamp"],
                "composite_score": run["composite_score"],
                "results": [dict(r) for r in results],
            })
        path = output or "export_all.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Exported to {path}")

    elif fmt == "csv":
        path = output or "export_all.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "timestamp", "composite_score", "test_name", "category", "tier", "score", "max_score", "percentage"])
            for run in runs:
                results = db.get_results_by_run(run["id"])
                for r in results:
                    pct = round(r["score"] / r["max_score"] * 100, 1) if r["max_score"] > 0 else 0
                    writer.writerow([run["model_name"], run["timestamp"], run["composite_score"], r["test_name"], r["category"], r["tier"], r["score"], r["max_score"], pct])
        print(f"Exported to {path}")
```

**Step 3: Run tests**

```bash
python -m pytest tests/test_export.py -v
```

**Step 4: Commit**

```bash
git add src/export.py tests/test_export.py && git commit -m "feat: JSON and CSV export for benchmark results"
```

---

### Task 10: Tool YAML Files (18 tools)

**Files:**
- Create: `tools/*.yaml` (18 files, all tool definitions per design doc)

Create all 18 tool definitions with complete OpenAI-compatible function schemas. Each file must have: name, description, parameters (with type, properties, required).

**Step 1: Create all files** (see design doc for full parameter specs)

**Step 2: Verify all tools load**

```bash
python -c "from src.loader import load_tools; tools = load_tools('tools'); print(f'{len(tools)} tools loaded: {sorted(tools.keys())}')"
```

Expected: `18 tools loaded: [...]`

**Step 3: Commit**

```bash
git add tools/ && git commit -m "feat: 18 IT/ops and network tool definitions"
```

---

### Task 11: Test YAML Files — Tool Calling

**Files:**
- Create: `tests/tool_calling/tier1_basic/*.yaml` (~15 tests)
- Create: `tests/tool_calling/tier2_intermediate/*.yaml` (~15 tests)
- Create: `tests/tool_calling/tier3_advanced/*.yaml` (~10 tests)

Write realistic IT/ops scenarios for each tier. Each test case follows the format from the design doc with: name, tier, category, description, prompt, available_tools, expected_tool_calls, simulated_responses, scoring.

**Tier 1 examples:** restart a service, check logs for errors, look up a user, ping a host, create a ticket
**Tier 2 examples:** diagnose a slow service (check status → check logs → restart), resolve a DNS issue with ambiguous symptoms, handle a tool returning an error
**Tier 3 examples:** full network outage diagnosis (ping + dns + traceroute + logs + ticket), parallel health checks across multiple hosts, firewall + DNS + notification chain

**Step 1: Create test files for each tier**

**Step 2: Verify all tests load**

```bash
python -c "from src.loader import load_tests; tests = load_tests('tests'); tc = [t for t in tests if t['category']=='tool_calling']; print(f'{len(tc)} tool-calling tests loaded')"
```

**Step 3: Commit**

```bash
git add tests/tool_calling/ && git commit -m "feat: ~40 tool-calling test cases across 3 tiers"
```

---

### Task 12: Test YAML Files — Knowledge

**Files:**
- Create: `tests/knowledge/networking.yaml` (~10 questions)
- Create: `tests/knowledge/linux_admin.yaml` (~10 questions)
- Create: `tests/knowledge/scripting.yaml` (~10 questions)
- Create: `tests/knowledge/cloud_infra.yaml` (~10 questions)
- Create: `tests/knowledge/security.yaml` (~10 questions)

Each file contains a list of test cases (or one per file in subdirectories). Questions should range from basic to advanced within each domain. Include judge_criteria with specific expected answers.

**Networking examples:** subnetting, BGP basics, TCP handshake, DNS record types, VLAN configuration
**Linux admin examples:** file permissions, systemd management, LVM, iptables, process management
**Scripting examples:** bash loops, cron syntax, regex patterns, Python automation, sed/awk
**Cloud infra examples:** VPC design, load balancer types, container orchestration, IaC concepts, HA patterns
**Security examples:** TLS handshake, SSH hardening, OWASP top 10, firewall zones, certificate management

**Step 1: Create all knowledge test files**

**Step 2: Verify**

```bash
python -c "from src.loader import load_tests; tests = load_tests('tests'); kn = [t for t in tests if t['category']=='knowledge']; print(f'{len(kn)} knowledge tests loaded')"
```

**Step 3: Commit**

```bash
git add tests/knowledge/ && git commit -m "feat: ~50 knowledge depth test cases across 5 IT domains"
```

---

### Task 13: Test YAML Files — Context

**Files:**
- Create: `tests/context/tier1_short/*.yaml` (~5 tests)
- Create: `tests/context/tier2_medium/*.yaml` (~5 tests)
- Create: `tests/context/tier3_long/*.yaml` (~5 tests)

Each test defines: name, tier, category, context_size, needle(s) with positions, filler type, questions with expected answers.

**Step 1: Create context test files**

**Step 2: Verify**

```bash
python -c "from src.loader import load_tests; tests = load_tests('tests'); ctx = [t for t in tests if t['category']=='context']; print(f'{len(ctx)} context tests loaded')"
```

**Step 3: Commit**

```bash
git add tests/context/ && git commit -m "feat: ~15 needle-in-haystack context length test cases"
```

---

### Task 14: Streamlit Dashboard

**Files:**
- Create: `dashboard/app.py`

**Step 1: Implement dashboard**

```python
# dashboard/app.py
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.db import Database

st.set_page_config(page_title="LLM Benchmark Dashboard", layout="wide")
db = Database("results.db")

# Sidebar navigation
page = st.sidebar.selectbox("View", ["Leaderboard", "Model Detail", "Compare", "History"])

if page == "Leaderboard":
    st.title("Model Leaderboard")
    runs = db.get_all_runs()
    if not runs:
        st.info("No benchmark runs yet. Run: python -m src.runner --model 'your-model'")
    else:
        rows = []
        for run in runs:
            results = db.get_results_by_run(run["id"])
            cat_scores = {}
            for cat in ["tool_calling", "knowledge", "context"]:
                cat_results = [r for r in results if r["category"] == cat]
                if cat_results:
                    total = sum(r["score"] for r in cat_results)
                    max_total = sum(r["max_score"] for r in cat_results)
                    cat_scores[cat] = round(total / max_total * 100, 1) if max_total > 0 else 0
                else:
                    cat_scores[cat] = None
            rows.append({
                "Model": run["model_name"],
                "Tool Calling %": cat_scores.get("tool_calling"),
                "Knowledge %": cat_scores.get("knowledge"),
                "Context %": cat_scores.get("context"),
                "Composite %": run["composite_score"],
                "Date": run["timestamp"][:10],
                "run_id": run["id"],
            })
        df = pd.DataFrame(rows).sort_values("Composite %", ascending=False)
        st.dataframe(df.drop(columns=["run_id"]), use_container_width=True, hide_index=True)

elif page == "Model Detail":
    st.title("Model Detail")
    runs = db.get_all_runs()
    if not runs:
        st.info("No runs yet.")
    else:
        run_options = {f"{r['model_name']} ({r['timestamp'][:10]})": r["id"] for r in runs}
        selected = st.selectbox("Select run", list(run_options.keys()))
        run_id = run_options[selected]
        results = db.get_results_by_run(run_id)

        # Score by tier
        tier_data = {}
        for r in results:
            tier = r["tier"]
            cat = r["category"]
            key = f"{cat} T{tier}"
            if key not in tier_data:
                tier_data[key] = {"score": 0, "max": 0}
            tier_data[key]["score"] += r["score"]
            tier_data[key]["max"] += r["max_score"]

        if tier_data:
            tier_df = pd.DataFrame([
                {"Category": k, "Score %": round(v["score"] / v["max"] * 100, 1) if v["max"] > 0 else 0}
                for k, v in tier_data.items()
            ])
            fig = px.bar(tier_df, x="Category", y="Score %", title="Score by Category & Tier")
            st.plotly_chart(fig, use_container_width=True)

        # Per-test breakdown
        st.subheader("Per-Test Results")
        for r in results:
            pct = round(r["score"] / r["max_score"] * 100, 1) if r["max_score"] > 0 else 0
            icon = "PASS" if pct >= 70 else "FAIL"
            with st.expander(f"[{icon}] {r['test_name']} — {r['score']:.1f}/{r['max_score']:.1f} ({pct}%)"):
                st.json(json.loads(r["transcript"]) if isinstance(r["transcript"], str) else r["transcript"])

elif page == "Compare":
    st.title("Compare Models")
    runs = db.get_all_runs()
    if len(runs) < 2:
        st.info("Need at least 2 runs to compare.")
    else:
        run_options = {f"{r['model_name']} ({r['timestamp'][:10]})": r["id"] for r in runs}
        selected = st.multiselect("Select runs to compare", list(run_options.keys()), max_selections=3)
        if len(selected) >= 2:
            compare_data = []
            for s in selected:
                rid = run_options[s]
                results = db.get_results_by_run(rid)
                for cat in ["tool_calling", "knowledge", "context"]:
                    cat_results = [r for r in results if r["category"] == cat]
                    if cat_results:
                        total = sum(r["score"] for r in cat_results)
                        max_total = sum(r["max_score"] for r in cat_results)
                        pct = round(total / max_total * 100, 1) if max_total > 0 else 0
                    else:
                        pct = 0
                    compare_data.append({"Model": s, "Category": cat, "Score %": pct})
            df = pd.DataFrame(compare_data)
            fig = px.bar(df, x="Category", y="Score %", color="Model", barmode="group", title="Model Comparison")
            st.plotly_chart(fig, use_container_width=True)

elif page == "History":
    st.title("Model History")
    runs = db.get_all_runs()
    model_names = sorted(set(r["model_name"] for r in runs))
    if not model_names:
        st.info("No runs yet.")
    else:
        selected_model = st.selectbox("Select model", model_names)
        model_runs = [r for r in runs if r["model_name"] == selected_model]
        if len(model_runs) < 2:
            st.info("Need at least 2 runs of this model to show history.")
        else:
            history = [{"Date": r["timestamp"][:10], "Composite %": r["composite_score"]} for r in model_runs]
            df = pd.DataFrame(history)
            fig = px.line(df, x="Date", y="Composite %", title=f"{selected_model} — Score Over Time", markers=True)
            st.plotly_chart(fig, use_container_width=True)

db.close()
```

**Step 2: Test locally**

```bash
streamlit run dashboard/app.py
```

Verify: opens browser, shows "No benchmark runs yet" on leaderboard.

**Step 3: Commit**

```bash
git add dashboard/app.py && git commit -m "feat: Streamlit dashboard with leaderboard, detail, compare, history views"
```

---

### Task 15: Integration Test & Polish

**Files:**
- Modify: various files for any bugs found

**Step 1: Run full test suite**

```bash
python -m pytest tests/ -v
```

**Step 2: Dry-run against actual API (if available)**

```bash
python -m src.runner --model "test-model"
```

**Step 3: Fix any issues found**

**Step 4: Final commit**

```bash
git add -A && git commit -m "fix: integration test fixes and polish"
```
