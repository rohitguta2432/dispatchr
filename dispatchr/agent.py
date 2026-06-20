"""The Dispatchr agent: a transparent tool-calling loop.

No agent framework — just a readable loop so reviewers can see exactly how the
agent decides. The agent understands the request, judges urgency, then quotes,
schedules, books, or escalates by calling tools.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .llm import AssistantTurn, Message, get_llm
from .tools import Tools, tool_schemas

SYSTEM_PROMPT = """\
You are Dispatchr, the dispatcher for a home-services company (HVAC, plumbing, \
electrical). You answer customer messages and take them from "I have a problem" \
to a booked appointment, on your own.

Policy:
- SAFETY FIRST. If the message suggests a gas smell, smoke, fire, sparks, \
electric shock, or flooding, call escalate_to_human immediately — do not quote \
or book.
- Classify the problem into one job type, then call get_price_estimate. NEVER \
state a price you did not get from that tool.
- Call find_available_slots and offer the customer real open slots only.
- When the customer picks a slot and gives a name + address, call book_job.
- If the customer later wants to cancel, call cancel_job with their booking_id. \
If they want a different time, call reschedule_job with the booking_id and a new \
slot from find_available_slots.
- Be warm, concise, and professional. Show money with the ₹ symbol (e.g. ₹800).
"""


@dataclass
class StepRecord:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass
class Result:
    reply: str
    tool_calls: list[StepRecord] = field(default_factory=list)
    escalated: bool = False
    messages: list[Message] = field(default_factory=list)

    def called(self, name: str) -> StepRecord | None:
        for rec in self.tool_calls:
            if rec.name == name:
                return rec
        return None


def _assistant_message(turn: AssistantTurn) -> Message:
    return {
        "role": "assistant",
        "content": turn.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in turn.tool_calls
        ],
    }


class Agent:
    def __init__(self, llm: Any | None = None, tools: Tools | None = None, max_steps: int = 6) -> None:
        self.llm = llm or get_llm()
        self.tools = tools or Tools()
        self.max_steps = max_steps

    def run(self, messages: list[Message]) -> Result:
        """Drive the loop until the agent produces a final message (no tool calls)."""
        records: list[StepRecord] = []
        for _ in range(self.max_steps):
            turn = self.llm.complete(messages, tool_schemas())
            if turn.tool_calls:
                messages.append(_assistant_message(turn))
                for tc in turn.tool_calls:
                    result = self.tools.call(tc.name, tc.arguments)
                    records.append(StepRecord(tc.name, tc.arguments, result))
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": json.dumps(result),
                    })
                continue
            reply = turn.content or ""
            messages.append({"role": "assistant", "content": reply})
            return Result(reply, records, any(r.name == "escalate_to_human" for r in records), messages)

        return Result("Sorry, I'm having trouble — let me get a human to help.", records, False, messages)


def build_messages(user_messages: list[str]) -> list[Message]:
    msgs: list[Message] = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs += [{"role": "user", "content": m} for m in user_messages]
    return msgs


def run_conversation(user_messages: list[str], llm: Any | None = None, now: datetime | None = None) -> Result:
    """One-shot helper used by the eval harness."""
    agent = Agent(llm=llm, tools=Tools(now=now))
    return agent.run(build_messages(user_messages))
