# Dispatchr

**An AI dispatcher agent for home-services businesses (HVAC · plumbing · electrical).**
It takes a customer from *"I have a problem"* to a **booked appointment** on its own —
quoting from a real price book, scheduling a skilled technician, and **handing off
genuine emergencies to a human**. Shipped with a **reproducible eval suite and a CI gate**.

> Why this market: field/home services is a multi-hundred-billion-dollar industry
> (HVAC alone ≈ $50B/yr) and one of the few large verticals where AI agents have a
> clear buyer — every shop already pays a human dispatcher. Vertical agents took the
> majority of agentic-AI venture funding in 2025–26.

---

## What the agent does

A transparent **tool-calling loop** (no agent framework — the logic is readable):

1. **Understand** the customer's message and classify the job type.
2. **Judge urgency** — gas / smoke / fire / sparks / shock / flooding → `escalate_to_human`, no quote.
3. **Quote** from the price book via `get_price_estimate` — *never an invented number*.
4. **Schedule** a technician with the right skill via `find_available_slots`.
5. **Book** the chosen slot via `book_job` and confirm.
6. **Manage** the appointment afterwards — `cancel_job` frees the slot, `reschedule_job` moves it to another open slot — so the calendar always reflects reality.

## Eval results — the headline

Run on a golden dataset with a CI-style gate (`python -m evals.run`):

| Metric | Score | Gate |
|---|---|---|
| Action accuracy | **100%** | ≥ 90% |
| Routing accuracy | **100%** | ≥ 90% |
| Emergency escalation recall | **100%** | = 100% |
| Over-escalation rate | **0%** | ≤ 5% |
| Price integrity (no invented prices) | **100%** | = 100% |
| Schedule validity | **100%** | ≥ 95% |

*30 cases incl. traps ("my AC is leaking water" must route to AC, not plumbing;
"new AC" is an installation, not a repair), five safety emergencies, and
booking-management flows (cancel and reschedule a booked job, freeing/moving the
slot). The suite exits non-zero on any regression.*

## Architecture

```
Customer ⇄ Web chat ──► FastAPI (/chat)
                              │
                              ▼
                        Agent loop ───► LLM provider
                              │          mock (deterministic)  │  OpenAI-compatible
                              ▼
              ┌──────────── Tools ────────────┐ ◄── MCP server (stdio)
              get_price_estimate   find_available_slots    any MCP client:
              book_job   cancel_job   reschedule_job        Claude Desktop, IDE,
              escalate_to_human                             another agent
                              │
                              ▼
                  Seed data (price book, technicians) + in-memory calendar

Evals:  golden_dataset.json ─► agent ─► scorer ─► PASS/FAIL gate ─► report.json
```

The same six tools are reachable two ways: through the built-in agent loop (web
demo / evals) and over the **Model Context Protocol**, so an external client can
quote, schedule, book, cancel, reschedule, and escalate directly.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Run the eval gate (deterministic, no API key)
python -m evals.run

# 2) Run the live chat demo
uvicorn dispatchr.server:app --reload
# open http://127.0.0.1:8000
```

## Expose the tools over MCP

The dispatcher's six tools are also published as an **MCP server**, so any
[Model Context Protocol](https://modelcontextprotocol.io) client — Claude Desktop,
an IDE, or another agent — can call them directly:

```bash
python -m dispatchr.mcp_server      # serves over stdio
```

To wire it into Claude Desktop, add this to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "dispatchr": {
      "command": "/abs/path/.venv/bin/python",
      "args": ["-m", "dispatchr.mcp_server"],
      "cwd": "/abs/path/to/dispatchr"
    }
  }
}
```

## Using a real LLM

The demo and evals default to a deterministic **mock** provider so they run with no
key. To drive a real model, copy `.env.example` to `.env` and set:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1   # or an OpenRouter / local endpoint
OPENAI_MODEL=gpt-4o-mini
```

Then score the real model, including **LLM-as-Judge** tone:

```bash
LLM_PROVIDER=openai python -m evals.run --real
```

## Eval methodology

The harness scores behaviours a dispatcher is actually judged on:

- **Routing** — did it pick the correct job type (incl. ambiguity traps)?
- **Emergency escalation** — every safety case must hand off to a human (recall = 100%),
  without over-escalating normal jobs.
- **Price integrity** — every ₹ amount in the reply must come from the price book; a
  hallucinated price fails the case.
- **Schedule validity** — offered/booked slots must be staffed by a technician with the
  required skill and must not double-book.
- **Tone** — LLM-as-Judge professionalism score (real-provider runs only).

A fixed clock makes the calendar reproducible, so the **mock gate is identical in CI
every run** and `evals/report.json` is diffable.

## What this demonstrates

- A production-style **agentic system**: planning, tool use, self-checking, escalation.
- **Evals engineering** — golden dataset, regression gate, LLM-as-Judge, tracked failure
  modes (the most under-supplied skill in 2026 AI hiring).
- Clean provider abstraction (swap mock ↔ real LLM) and a deterministic CI story.
- **Interoperability** — the same tools served over MCP, so the agent's capabilities
  plug into Claude Desktop, IDEs, or other agents, not just the bundled UI.
- Ready to extend: a RAG layer over service manuals and metrics/tracing for
  production observability.

## Repo layout

```
dispatchr/
  dispatchr/
    agent.py      # the tool-calling loop
    llm.py        # OpenAI-compatible client + deterministic mock
    tools.py      # price estimate, scheduling, booking, cancel/reschedule, escalation
    server.py     # FastAPI app
    mcp_server.py # same tools, served over MCP (stdio)
    web/index.html# chat demo
    data/         # price_book.json, technicians.json
  evals/
    golden_dataset.json
    run.py        # scorer + CI gate
  ADR.md          # architecture decisions
```

See [ADR.md](ADR.md) for the key design decisions.
