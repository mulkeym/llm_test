# LLM Benchmark Suite

A Python benchmarking tool for evaluating local LLMs served via OpenAI-compatible APIs (e.g., llama.cpp server). Measures three capabilities:

- **Tool Calling** — Can the model select and invoke the right tools with correct parameters? (3 tiers: basic → advanced)
- **Knowledge Depth** — How well does the model answer technical IT questions? (5 domains: networking, linux admin, scripting, cloud infra, security)
- **Context Handling** — Can the model find specific facts buried in long documents? (needle-in-a-haystack, 2K–50K tokens)

Results are scored using Claude as a judge, persisted in SQLite, and visualized in a Streamlit dashboard.

## Quick Start

### 1. Install Dependencies

```bash
cd llm_test
pip install -r requirements.txt
```

### 2. Configure

Copy the example environment file and add your Anthropic API key (used for the Claude judge):

```bash
cp .env.example .env
```

Edit `.env`:
```
ANTHROPIC_API_KEY=your-key-here
```

Edit `config.yaml` to point to your model server:

```yaml
target_api:
  base_url: "http://192.168.1.181:8080"  # your llama.cpp server
  api_key: ""
  timeout: 60
  max_retries: 2
```

### 3. Run a Benchmark

```bash
python -m src --model "your-model-name"
```

The model name is a label for tracking — it's stored in the results database so you can compare runs later.

#### Options

```bash
# Run and export results to JSON
python -m src --model "llama-3-8b-q4" --export json

# Run and export to CSV
python -m src --model "llama-3-8b-q4" --export csv

# Use a custom config file
python -m src --model "my-model" --config my-config.yaml
```

### 4. View Results

Launch the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard provides four views:

- **Leaderboard** — All models ranked by composite score
- **Model Detail** — Per-tier bar charts, category radar chart, and per-test breakdowns with transcripts
- **Compare** — Side-by-side comparison of 2–3 models
- **History** — Track the same model's score over time

## Test Suite

### Tool Calling (40 tests)

| Tier | Tests | Description |
|---|---|---|
| Tier 1 (Basic) | 15 | Single tool, clear instructions, 2–4 available tools |
| Tier 2 (Intermediate) | 15 | Multi-step chains, ambiguity, error handling, 5–10 tools |
| Tier 3 (Advanced) | 10 | All 18 tools, parallel calls, long chains, error recovery |

### Available Tools (18)

**IT Operations:** query_database, restart_service, check_logs, create_ticket, run_script, lookup_user, check_service_status, send_notification, get_metrics, update_firewall_rule

**Network:** dns_lookup, ping_host, traceroute, whois_lookup, subnet_calculator, manage_dns_record, check_port, get_arp_table

### Knowledge Depth (50 tests)

10 questions per domain across 5 IT domains:
- Networking (TCP, DNS, BGP, VLANs, subnetting)
- Linux Administration (systemd, LVM, iptables, processes)
- Scripting (bash, cron, regex, Python, Ansible)
- Cloud Infrastructure (VPC, Kubernetes, CI/CD, IaC)
- Security (TLS, SSH hardening, OWASP, RBAC, zero trust)

### Context Handling (15 tests)

| Tier | Context Size | Needles | Description |
|---|---|---|---|
| Tier 1 (Short) | 2K–4K tokens | 1 | Single fact in a short IT document |
| Tier 2 (Medium) | 8K–16K tokens | 2–3 | Multiple facts across a medium document |
| Tier 3 (Long) | 32K–50K tokens | 3–5 | Many facts, some contradicting earlier statements |

## Scoring

### Tool Calling (Deterministic)
- Correct tool selected (5 pts)
- Correct parameters (3 pts)
- No hallucinated tools (2 pts)
- Correct ordering — tier 2+ (3 pts)
- Parallel calls — tier 3 (2 pts)

### Knowledge (Claude Judge)
- Factual accuracy (40%)
- Completeness (25%)
- Reasoning quality (20%)
- Clarity (15%)

### Context (Deterministic + Claude Judge)
- Retrieval accuracy (7 pts)
- Position agnostic (3 pts)

### Composite Score
Default weighting: 40% tool calling, 30% knowledge, 30% context. Configurable in `config.yaml`.

## Adding Custom Tests

Tests are YAML files — add new ones by dropping files into the appropriate directory:

```
tests/
├── tool_calling/
│   ├── tier1_basic/your_test.yaml
│   ├── tier2_intermediate/your_test.yaml
│   └── tier3_advanced/your_test.yaml
├── knowledge/
│   └── your_domain/your_test.yaml
└── context/
    ├── tier1_short/your_test.yaml
    ├── tier2_medium/your_test.yaml
    └── tier3_long/your_test.yaml
```

See `docs/plans/2026-03-30-llm-benchmark-design.md` for YAML formats and examples.

## Running Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
llm_test/
├── config.yaml          # API endpoints, judge config, scoring weights
├── src/
│   ├── runner.py        # Test orchestration
│   ├── client.py        # OpenAI-compatible API client
│   ├── judge.py         # Claude-as-judge scoring
│   ├── scorer.py        # Deterministic tool-calling scorer
│   ├── context.py       # Filler generator + needle insertion
│   ├── loader.py        # YAML test/tool loader
│   ├── db.py            # SQLite persistence
│   └── export.py        # JSON/CSV export
├── dashboard/
│   └── app.py           # Streamlit dashboard
├── tools/               # 18 tool definitions (YAML)
└── tests/               # 105 test cases (YAML) + unit tests
```
