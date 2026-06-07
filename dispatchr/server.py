"""FastAPI server: the live Dispatchr demo.

Holds one in-memory session per chat (its own message buffer + calendar) so
bookings persist across turns. Runs the mock provider by default; set
LLM_PROVIDER=openai in .env to drive a real model.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .agent import Agent, build_messages
from .llm import get_llm
from .tools import Tools

load_dotenv()

app = FastAPI(title="Dispatchr")
_WEB = Path(__file__).parent / "web" / "index.html"
SESSIONS: dict[str, dict] = {}


class ChatIn(BaseModel):
    message: str
    session_id: str | None = None


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _WEB.read_text()


@app.get("/health")
def health() -> dict:
    return {"ok": True, "sessions": len(SESSIONS)}


@app.post("/chat")
def chat(body: ChatIn) -> dict:
    sid = body.session_id or uuid.uuid4().hex[:12]
    sess = SESSIONS.setdefault(sid, {"messages": build_messages([]), "tools": Tools()})
    sess["messages"].append({"role": "user", "content": body.message})

    agent = Agent(llm=get_llm(), tools=sess["tools"])
    result = agent.run(sess["messages"])

    return {
        "session_id": sid,
        "reply": result.reply,
        "tools": [t.name for t in result.tool_calls],
        "escalated": result.escalated,
    }
