# ORBIT — Multi-Agent AI Orchestration Platform

ORBIT routes tasks to specialized agents behind a single API:

| Agent | Responsibility | Engine |
|---|---|---|
| **Orchestrator** | Deterministic task routing | rule-based (no LLM) |
| **AI Agent** | General questions, explanations, summarization, reasoning, text generation, file-context tasks | Freebuff LLM |
| **Data Agent** | CSV profiling: rows/cols/dtypes, missing values, duplicates, mean/median/min/max/std, correlations, deterministic insights | pandas + NumPy (local) |
| **ML Agent** | Dataset validation, regression & classification, model training + metrics | pandas + scikit-learn (local) |
| **Research Agent** | Query generation, source discovery, article extraction, evidence validation, grounded synthesis | httpx + BeautifulSoup + Freebuff |

All agent responses share one envelope:

```json
{ "agent": "data", "status": "completed", "message": "...", "data": { }, "metadata": { } }
```

Statuses: `queued`, `running`, `completed`, `failed`, `partial`, `insufficient_data`.

**The LLM provider is Freebuff only.** ORBIT uses Freebuff Desktop's real programmatic
integration — its **BYOK (bring-your-own-key) connection system** — and speaks the
OpenAI-compatible `/chat/completions` protocol to the connection's endpoint. No OpenAI
SDKs are used and no keys are hardcoded, printed, or copied.

### How ORBIT connects to Freebuff

Provider settings resolve in this order:

1. **Explicit env vars** (optional override): `FREEBUFF_API_KEY` (+ optional
   `FREEBUFF_BASE_URL`, `FREEBUFF_MODEL`).
2. **Freebuff Desktop's BYOK store** (automatic, recommended):
   - `~/.config/freebuff/byok/connections.json` — connection metadata (provider,
     base URL, model, `credentialRef`),
   - the credential itself stays in the **OS keychain** under service
     `com.freebuff.byok.v1` and is read via `Bun.secrets` using the **Bun runtime
     bundled with Freebuff Desktop** — the exact mechanism Freebuff's own
     orchestrator uses. The key never leaves the process.

Verify the chain at any time:

```bash
./.venv/Scripts/python.exe scripts/freebuff_connectivity_test.py
```

It reports the resolved source (`env` or `byok`), connection name, base URL and
model (key hidden), then performs one minimal completion. Exit code 0 = pass.

Optional env toggles: `FREEBUFF_BYOK_CONNECTION_ID` (pick a specific connection),
`FREEBUFF_BUN_PATH` (custom bun.exe), `FREEBUFF_BYOK_CONFIG_DIR` (alternate store).

## Layout

```
orbit-backend/        FastAPI + agents + tests
  app/
    config.py         env-driven Freebuff settings (no hardcoded secrets)
    llm/client.py     shared Freebuff LLM client (async + sync)
    core/             results envelope, file handling, routing, task registry
    agents/           base + ai + data + ml + research + orchestrator
    api/              schemas + routes
    main.py           app factory (CORS + optional static frontend)
  scripts/freebuff_connectivity_test.py   Phase 1 gate
  tests/            pytest suite (46 tests)
orbit-frontend/       React + Vite + TypeScript "ORBIT Workspace"
```

## Setup

### Backend

```bash
cd orbit-backend
py -3 -m venv .venv                       # first time only
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

No key configuration is required if Freebuff Desktop already has a BYOK
connection (see above). To override explicitly, copy `.env.example` to
`.data/.env` and set `FREEBUFF_API_KEY` etc. — env vars always win.

### Run the API

```bash
./.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Interactive docs: http://127.0.0.1:8000/docs

### Frontend

```bash
cd orbit-frontend
npm install        # first time only
npm run dev        # http://localhost:5173  (proxies /api -> 127.0.0.1:8000)
```

`npm run build` typechecks and emits `dist/`; when `dist/` exists the backend
also serves it at `/` so one process can host everything.

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Platform descriptor (standard envelope) |
| `/health` | GET | Liveness + secret-free config summary |
| `/run` | POST | Submit a task (JSON: `{task, agent?, history?}`) → `202` + `task_id` |
| `/run/file` | POST | Submit with a CSV upload (multipart: `task`, `agent?`, `file`) |
| `/status/{task_id}` | GET | Poll the task's `AgentResult` |

Example:

```bash
curl -s -X POST http://127.0.0.1:8000/run/file \
  -F "task=Train a model to predict price from this CSV" \
  -F "file=@houses.csv;type=text/csv"

curl -s http://127.0.0.1:8000/status/<task_id>
```

## Deterministic routing

Priority order (first match wins): **research → ML → data → AI**, with an
optional explicit `agent` override (`ai | data | ml | research`). A file alone
does not force the Data Agent — a general AI question with a file attached
still reaches the AI Agent (with the file preview as context); only
file-referencing or vague tasks default to Data. Rules live in
`app/core/routing.py`.

## Tests

```bash
cd orbit-backend
./.venv/Scripts/python.exe -m pytest
```

46 tests cover the routing matrix, agent envelopes and error mapping, Data
Agent local statistics, ML Agent training/evaluation on synthetic datasets,
Research Agent evidence validation (fixtures only — no live network), the task
registry, and the full API flow.

## How to add an agent

1. Create `app/agents/my_agent.py`:

   ```python
   from app.agents.base import BaseAgent
   from app.core.results import AgentResult

   class MyAgent(BaseAgent):
       name = "my"

       def _run(self, task, file_context=None, history=None) -> AgentResult:
           return AgentResult.create("my", "completed", "done",
                                     data={"response": "hello"})
   ```

2. Register it in `Orchestrator.__init__` (`app/agents/orchestrator.py`) and add
   `"my"` to `SUPPORTED_AGENTS` + `AGENT_LABELS` (`app/core/routing.py`,
   `app/core/results.py`).
3. Add a `RoutingRule` in `app/core/routing.py` so tasks reach it.
4. Add tests following `tests/test_agents.py`.

## Design notes

- **Numbers are never invented.** Data/ML results are computed locally with
  pandas/NumPy/scikit-learn; the LLM only narrates already-computed statistics,
  and the metadata marks `computed_locally`.
- **Freebuff stays the only provider.** The `FreebuffLLMClient` interface
  (`chat`/`chat_sync`) is provider-agnostic internally but only ever resolves
  through Freebuff: env overrides or the Freebuff BYOK store. Agents never see
  endpoints or keys.
- **Research evidence discipline.** Only full fetched article text can become
  `verified` evidence; RSS/search metadata stays `unverified`/`rejected` and is
  never synthesized as fact. Every source reports title, publisher, date, URL,
  and evidence status. Networking is bounded: 10s timeouts, ≤1 fetch retry,
  ≤6 sources, concurrency ≤3, wall-clock budget.
- **No persistence in v1.** The in-memory task registry (bounded to 100
  terminal tasks) keeps the architecture simple; durable storage can be added
  later without touching the agents.
