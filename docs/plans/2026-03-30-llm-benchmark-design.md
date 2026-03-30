# LLM Benchmark Suite — Design Document

**Date:** 2026-03-30
**Status:** Approved

## Overview

A Python benchmarking tool for evaluating local LLMs served via llama.cpp's OpenAI-compatible API. Measures three capabilities: tool calling, knowledge depth, and long-context retrieval. Claude acts as judge for subjective scoring. Results persist in SQLite with a Streamlit dashboard for comparison.

## Target Environment

- **Model server:** llama.cpp server at `http://192.168.1.181:8080` (OpenAI-compatible API)
- **Domain context:** Healthcare IT environment — models evaluated on IT/ops automation and technical tasks
- **Judge:** Claude via Anthropic API (key via `ANTHROPIC_API_KEY` env var)
- **Usage pattern:** Single model per run, accumulate historical results for comparison

## Project Structure

```
llm_test/
├── config.yaml                  # API endpoints, judge config, defaults
├── tools/                       # Tool definitions (JSON schemas)
│   ├── query_database.yaml
│   ├── restart_service.yaml
│   ├── check_logs.yaml
│   └── ...
├── tests/
│   ├── tool_calling/
│   │   ├── tier1_basic/
│   │   ├── tier2_intermediate/
│   │   └── tier3_advanced/
│   ├── knowledge/
│   │   ├── networking.yaml
│   │   ├── linux_admin.yaml
│   │   ├── scripting.yaml
│   │   ├── cloud_infra.yaml
│   │   └── security.yaml
│   └── context/
│       ├── tier1_short/
│       ├── tier2_medium/
│       └── tier3_long/
├── src/
│   ├── runner.py                # Test orchestration
│   ├── client.py                # OpenAI-compatible API client
│   ├── judge.py                 # Claude-as-judge scoring
│   ├── db.py                    # SQLite persistence
│   └── export.py                # JSON/CSV export
├── dashboard/
│   └── app.py                   # Streamlit web dashboard
├── results.db                   # SQLite (gitignored)
├── .env                         # API keys (gitignored)
└── requirements.txt
```

Key principle: **tools and tests are data, not code.** New benchmarks are added by writing YAML files, not Python.

## Tool Definitions

18 pre-built IT/ops tools defined as YAML files with OpenAI-compatible function schemas.

### IT Operations Tools

| Tool | Description | Key Parameters |
|---|---|---|
| `query_database` | Run SQL against a named database | query, database, timeout |
| `restart_service` | Restart a system service | service_name, host, force |
| `check_logs` | Search logs by service/severity/time | service, severity, time_range |
| `create_ticket` | Create a helpdesk/incident ticket | title, description, priority, assignee |
| `run_script` | Execute a shell command on target host | command, host, sudo |
| `lookup_user` | Look up user in Active Directory | username, fields |
| `check_service_status` | Get health/status of a service | service_name, host |
| `send_notification` | Send alert via email/Slack/Teams | channel, recipient, message, urgency |
| `get_metrics` | Pull system metrics (CPU, mem, disk) | host, metric_type, time_range |
| `update_firewall_rule` | Add/modify/remove firewall rule | action, source, destination, port, protocol |

### Network Tools

| Tool | Description | Key Parameters |
|---|---|---|
| `dns_lookup` | Resolve hostname or reverse lookup | target, record_type (A/AAAA/MX/CNAME/PTR) |
| `ping_host` | Ping host, return latency/packet loss | host, count, timeout |
| `traceroute` | Trace network path to destination | host, max_hops, protocol |
| `whois_lookup` | WHOIS registration info for domain/IP | target |
| `subnet_calculator` | Calculate network/broadcast/usable range | cidr, operation (info/contains/split) |
| `manage_dns_record` | Create/update/delete DNS record | action, zone, name, record_type, value, ttl |
| `check_port` | Check if port is open on host | host, port, protocol (tcp/udp), timeout |
| `get_arp_table` | Get ARP entries for host/interface | host, interface, filter_ip |

### Example Tool Definition

```yaml
# tools/restart_service.yaml
name: restart_service
description: Restart a system service on the specified host
parameters:
  type: object
  properties:
    service_name:
      type: string
      description: Name of the service to restart
    host:
      type: string
      description: Target hostname or IP
    force:
      type: boolean
      description: Force restart even if service has active connections
      default: false
  required: [service_name, host]
```

Tools do not execute anything — the runner intercepts tool calls and returns simulated responses defined in test cases.

## Test Categories

### 1. Tool Calling Tests (~40 tests)

#### Tier 1 — Basic (single tool, clear intent, ~15 tests)
- Pick the right tool from 2-4 options
- Extract parameters from a straightforward prompt

#### Tier 2 — Intermediate (multi-step, ambiguity, ~15 tests)
- Pick from 5-10 tools
- Handle missing info (ask for clarification or infer)
- Multi-step: check status → diagnose → act
- Ambiguous prompts where multiple tools could apply
- Simulated tool errors requiring follow-up

#### Tier 3 — Advanced (complex orchestration, ~10 tests)
- All 18 tools available
- Parallel tool calls (e.g., ping + dns_lookup simultaneously)
- Long chains: diagnose a network outage across multiple hosts
- Error recovery: tool returns failure, model must adapt
- Complex schemas with optional/nested params

#### Test Case Format

```yaml
name: restart_unresponsive_service
tier: 1
category: tool_calling
description: Service is unresponsive, user asks to restart it

prompt: "The apache2 service on web-prod-01 is not responding. Can you restart it?"

available_tools:
  - restart_service
  - check_service_status
  - check_logs

expected_tool_calls:
  - tool: restart_service
    params:
      service_name: apache2
      host: web-prod-01

simulated_responses:
  restart_service:
    success: true
    message: "Service apache2 restarted successfully on web-prod-01"

scoring:
  correct_tool: 5
  correct_params: 3
  no_hallucinated_tools: 2
```

### 2. Knowledge Depth Tests (~50 tests)

5 domains, ~10 questions each: networking, linux admin, scripting/automation, cloud infrastructure, security.

```yaml
name: subnet_masking
tier: 2
category: knowledge
domain: networking

prompt: "A host has IP 10.45.12.130/21. What is the network address, broadcast address, and how many usable hosts are in this subnet?"

judge_criteria:
  - Correct network address (10.45.8.0)
  - Correct broadcast address (10.45.15.255)
  - Correct usable host count (2046)
  - Clear explanation of the math
```

### 3. Context Length Tests (~15 tests)

Needle-in-a-haystack tests with IT-relevant content.

| Tier | Context Size | Needles | Description |
|---|---|---|---|
| 1 | 2K-4K tokens | 1 | Single fact in a short IT doc |
| 2 | 8K-16K tokens | 2-3 | Multiple facts in a medium document |
| 3 | 32K+ tokens | 3-5 | Many facts in a long document, some contradicting earlier statements |

#### Filler Types
- **runbook** — Server maintenance procedures
- **log_dump** — Syslog/application log entries
- **config_dump** — Nginx/Apache/firewall configs
- **email_thread** — IT team discussion threads

#### Test Case Format

```yaml
name: find_server_ip_in_runbook
tier: 2
category: context
context_size: 12000

needle:
  - text: "The backup DNS server was migrated to 10.22.7.45 on March 3rd"
    position: 0.25
  - text: "Port 8443 was opened for the monitoring agent on db-prod-03"
    position: 0.75

filler: runbook

questions:
  - "What IP was the backup DNS server migrated to?"
    expected: "10.22.7.45"
  - "What port was opened on db-prod-03 and why?"
    expected: "Port 8443 for the monitoring agent"

scoring:
  retrieval_accuracy: 7
  position_agnostic: 3
```

## Scoring System

### Tool Calling (Deterministic)

| Criterion | Points | Measurement |
|---|---|---|
| Correct tool selected | 5 | Exact match against expected |
| Correct parameters | 3 | Key-by-key comparison, fuzzy match on strings |
| No hallucinated tools | 2 | Called only tools from available set |
| Correct ordering (tier 2-3) | 3 | Steps in the right sequence |
| Parallel calls (tier 3) | 2 | Multiple tools in single response |
| Error recovery (tier 2-3) | 3 | Adapted after simulated failure |

### Knowledge Depth (Claude Judge)

| Criterion | Weight |
|---|---|
| Factual accuracy | 40% |
| Completeness | 25% |
| Reasoning quality | 20% |
| Clarity | 15% |

Judge returns structured JSON:
```json
{
  "scores": {
    "accuracy": 8,
    "completeness": 7,
    "reasoning": 9,
    "clarity": 8
  },
  "weighted_total": 8.05,
  "explanation": "..."
}
```

### Context (Deterministic + Claude for paraphrased answers)

Scored on whether the model extracted the correct fact, regardless of needle position.

### Composite Score

Default weighting: **40% tool calling, 30% knowledge, 30% context.** Configurable in `config.yaml`.

## Runner Flow

```bash
$ python -m src.runner --model "my-model-name"
```

1. Load config from `config.yaml`, resolve env vars from `.env`
2. Discover tests — scan `tests/` directory, filter by config (tiers, categories, domains)
3. For each test case:
   - Build messages array with system prompt + available tools + user prompt
   - Send to target API via OpenAI-compatible client
   - If model returns tool calls → match against expected, return simulated response, continue if multi-step
   - Collect full conversation transcript
4. Score tool-calling tests deterministically
5. Score knowledge tests via Claude judge
6. Score context tests deterministically (Claude for paraphrase edge cases)
7. Persist all results + transcripts to SQLite
8. Print summary table to terminal
9. Export JSON/CSV if requested (`--export json`)

### Multi-Turn Conversation Loop

For tier 2-3 tool-calling tests, the runner acts as the tool executor:

```
Model says: call restart_service(service_name="apache2", host="web-prod-01")
Runner:     looks up simulated_responses → returns {"success": true, ...}
Model says: "The service has been restarted successfully."
Runner:     scores the full sequence
```

## Configuration

```yaml
# config.yaml
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
  domains: []  # empty = all domains
```

## Dashboard (Streamlit)

### Views

- **Leaderboard** — all models sorted by composite score (model name, tool %, knowledge %, context %, composite %, date)
- **Model Detail** — score breakdown by tier (bar charts), by domain (radar chart), per-test results with pass/fail, expandable transcripts
- **Compare** — select 2-3 models, overlay scores on same charts, highlight deltas
- **History** — track same model over time (different quantizations, prompt formats)

### Launch

```bash
streamlit run dashboard/app.py
```

## Export

```bash
python -m src.runner --model "my-model" --export json
python -m src.export --format csv --output results.csv
python -m src.export --model "my-model" --format json
```

## SQLite Schema

```sql
runs       (id, model_name, timestamp, config_snapshot, composite_score)
results    (id, run_id, test_name, category, tier, score, max_score, transcript)
judgements (id, result_id, criterion, score, explanation)
```

## Dependencies

- `openai` — API client for llama.cpp endpoint
- `anthropic` — Claude judge
- `streamlit` — dashboard
- `pyyaml` — test definitions
- `sqlite3` — built-in persistence
