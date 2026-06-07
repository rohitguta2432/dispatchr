# Architecture Decision Records

## ADR-001 — A framework-light agent loop

**Context.** Agent frameworks (LangChain, CrewAI, AutoGen) add abstraction layers that
hide the control flow. For a single-purpose dispatcher — and for a portfolio piece meant
to *show* how an agent works — that indirection is a cost, not a benefit.

**Decision.** Implement the agent as a small, explicit tool-calling loop
([`agent.py`](dispatchr/agent.py)) over the standard OpenAI tool-calling interface.

**Consequences.**
- A reviewer can read the entire decision path in one short file.
- Works with any OpenAI-compatible endpoint (OpenAI, OpenRouter, local) with no adapter.
- We own retries, max-step bounds, and tool dispatch — no framework upgrade surprises.
- Trade-off: multi-agent orchestration would need more code, but it's out of scope.

## ADR-002 — A deterministic mock LLM provider

**Context.** Evals must be the project's backbone, but LLM calls are non-deterministic,
cost money, and need network + keys. That makes them a poor CI gate and a poor
out-of-the-box demo.

**Decision.** Define one LLM interface with two implementations ([`llm.py`](dispatchr/llm.py)):
`OpenAILLM` (real tool-calling) and `MockLLM` (a deterministic, rule-based policy that
returns the same tool-call/finish decisions). The agent loop, server, and eval harness
are identical across both.

**Consequences.**
- `python -m evals.run` produces the **same scores every run** → a trustworthy regression
  gate and a diffable `report.json`.
- The web demo and evals run with **zero setup** (no key, no network).
- The real model is one env var away (`LLM_PROVIDER=openai`), and `--real` adds
  LLM-as-Judge tone scoring.
- Trade-off: the mock encodes expected behaviour, so it validates the *agent loop and
  tools*, not the model's reasoning. Real-provider eval runs cover the latter.

## ADR-003 — Expose the tools over MCP, sharing one tool layer

**Context.** The dispatcher's value is its *tools* — priced quoting, skill-aware
scheduling, safe booking, emergency escalation. Locking those behind the bundled
agent loop limits reuse: a shop already running Claude Desktop, an IDE assistant, or a
larger orchestrator can't reach them. The Model Context Protocol is the emerging
standard for exactly this hand-off.

**Decision.** Add a thin [`mcp_server.py`](dispatchr/mcp_server.py) (FastMCP) that
publishes the **same four functions** from [`tools.py`](dispatchr/tools.py) over MCP.
The MCP layer is a presentation adapter only — it owns no business logic and reuses the
identical `Tools` implementation and `JOB_TYPES` the agent loop and evals already use.

**Consequences.**
- One source of truth: a fix to pricing, scheduling, or escalation is reflected in the
  web demo, the eval gate, *and* every MCP client at once — no drift.
- The agent's capabilities compose into any MCP host (Claude Desktop, IDEs, other
  agents), turning a single-purpose demo into reusable infrastructure.
- The eval gate still exercises the canonical tools, so MCP consumers inherit the same
  verified behaviour without a separate test surface.
- Trade-off: MCP runs the tools directly, bypassing the agent's system prompt and
  escalation-first ordering — so the *client's* model owns that judgement. The tool
  docstrings carry the safety guidance (e.g. "always call get_price_estimate before
  quoting") to mitigate this.
