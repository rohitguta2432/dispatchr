"""LLM providers.

Two interchangeable implementations behind one interface:

* ``OpenAILLM`` — real tool-calling via any OpenAI-compatible API (OpenAI,
  OpenRouter, a local server).
* ``MockLLM`` — a deterministic, rule-based stand-in. It implements the exact
  same decision surface (return tool calls or a final message) so the agent
  loop, the web demo, and the eval suite all run with no API key and no network.

The mock is what makes the eval gate reproducible in CI: same input -> same
score, every time.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

Message = dict[str, Any]


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:8])


@dataclass
class AssistantTurn:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class OpenAILLM:
    def __init__(self, model: str, api_key: str, base_url: str) -> None:
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> AssistantTurn:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = resp.choices[0].message
        calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments or "{}"))
            for tc in (msg.tool_calls or [])
        ]
        return AssistantTurn(content=msg.content, tool_calls=calls)


# --------------------------------------------------------------------- mock

_EMERGENCY = [
    "gas", "smell of gas", "smoke", "fire", "burning", "burnt", "spark",
    "carbon monoxide", "flood", "shock", "electrocut", "explosion",
    "can't breathe", "cant breathe",
]

# Ordered: an explicit "install" intent wins; AC is checked before plumbing so
# "my AC is leaking water" routes to ac_repair, not plumbing.
_CLASSIFY: list[tuple[str, list[str]]] = [
    ("installation", ["install", "new ac", "set up", "setup", "mount", "fit a new"]),
    ("ac_repair", ["air condition", "aircon", "a/c", " ac ", "cooling", "not cooling", "won't cool", "ac is", "ac isn"]),
    ("heating_repair", ["heat", "heater", "furnace", "boiler", "radiator", "geyser not", "no hot water"]),
    ("electrical", ["outlet", "socket", "wiring", "breaker", "fuse", "switchboard", "no power", "lights not", "light not", "short circuit", "tripping", "mcb"]),
    ("plumbing", ["leak", "pipe", "drain", "clog", "blockage", "tap", "faucet", "toilet", "sink", "overflow", "sewage", "geyser"]),
]


def _is_emergency(text: str) -> bool:
    return any(k in text for k in _EMERGENCY)


def _classify(text: str) -> str | None:
    padded = f" {text} "
    for job_type, keys in _CLASSIFY:
        if any(k in padded for k in keys):
            return job_type
    return None


_CANCEL = [
    "cancel", "call it off", "call off", "scrap the booking", "scrap the appointment",
    "don't need it", "do not need it", "no longer need",
]
_RESCHEDULE = [
    "reschedule", "resched", "move it", "move my", "move the appointment",
    "change the time", "change my appointment", "different time", "another time",
    "push it", "switch me", "switch to a", "rebook",
]


def _wants_cancel(text: str) -> bool:
    return any(k in text for k in _CANCEL)


def _wants_reschedule(text: str) -> bool:
    return any(k in text for k in _RESCHEDULE)


def _money(n: int) -> str:
    return f"₹{n:,}"


def _label_hour(label: str) -> tuple[int, str] | None:
    m = re.search(r"(\d{1,2}):00\s*(AM|PM)", label)
    return (int(m.group(1)), m.group(2).lower()) if m else None


def _pick_slot(text: str, slots: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not slots:
        return None
    t = text.lower()
    if any(w in t for w in ["second", "2nd", "slot 2", " 2 "]):
        return slots[1] if len(slots) > 1 else None
    if any(w in t for w in ["third", "3rd", "slot 3", " 3 "]):
        return slots[2] if len(slots) > 2 else None
    tm = re.search(r"(\d{1,2})\s*(am|pm)", t)
    if tm:
        want = (int(tm.group(1)), tm.group(2).lower())
        for s in slots:
            if _label_hour(s["label"]) == want:
                return s
    if any(w in t for w in ["first", "1st", "earliest", "yes", "book", "sure", "that works", "ok", "okay", "sounds good"]):
        return slots[0]
    return None


def _extract_contact(text: str) -> tuple[str, str]:
    name_m = re.search(r"\b(?:i'm|i am|my name is|this is)\s+([A-Z][a-zA-Z]+)", text)
    addr_m = re.search(r"\bat\s+(.+?)(?:[.;]|$)", text) or re.search(r"\b(\d+\s+[A-Za-z][\w\s,]+)", text)
    name = name_m.group(1) if name_m else "Customer"
    address = addr_m.group(1).strip() if addr_m else "(address to confirm)"
    return name, address


def _booking_message(user_msgs: list[str], slots: list[dict[str, Any]]) -> str | None:
    """The customer turn that chose a slot to BOOK — the latest message with a slot
    cue that isn't itself a cancel/reschedule request. Lets the agent remember an
    earlier booking choice when a later turn asks to cancel or move it."""
    chosen: str | None = None
    for m in user_msgs:
        low = m.lower()
        if _wants_cancel(low) or _wants_reschedule(low):
            continue
        if _pick_slot(m, slots) is not None:
            chosen = m
    return chosen


def _reschedule_message(user_msgs: list[str]) -> str | None:
    for m in reversed(user_msgs):
        if _wants_reschedule(m.lower()):
            return m
    return None


class MockLLM:
    """Deterministic policy mirroring the agent's intended behaviour."""

    def complete(self, messages: list[Message], tools: list[dict[str, Any]]) -> AssistantTurn:
        user_msgs = [m.get("content") or "" for m in messages if m.get("role") == "user"]
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        done = {m.get("name") for m in tool_msgs}
        results: dict[str, Any] = {}
        for m in tool_msgs:
            try:
                results[m.get("name")] = json.loads(m.get("content") or "{}")
            except json.JSONDecodeError:
                results[m.get("name")] = {}

        all_text = " ".join(user_msgs).lower()
        first_user = user_msgs[0] if user_msgs else ""
        last_user = user_msgs[-1] if user_msgs else ""

        # Once handed off, the agent stops — it does not keep quoting/booking.
        if "escalate_to_human" in done:
            return AssistantTurn(content=(
                "I'm connecting you to a human dispatcher right now for your safety. "
                "If anyone is in danger, please call your local emergency number immediately."
            ))

        if _is_emergency(all_text) and "escalate_to_human" not in done:
            return AssistantTurn(tool_calls=[ToolCall(
                name="escalate_to_human",
                arguments={"reason": "Possible safety emergency in customer message", "summary": first_user[:200]},
            )])

        job_type = _classify(all_text)

        if job_type is None and "get_price_estimate" not in done:
            return AssistantTurn(content=(
                "Happy to help! Could you tell me a bit more about the problem — "
                "is it your AC, heating, plumbing, or electrical?"
            ))

        if "get_price_estimate" not in done:
            return AssistantTurn(tool_calls=[ToolCall(name="get_price_estimate", arguments={"job_type": job_type})])

        if "find_available_slots" not in done:
            return AssistantTurn(tool_calls=[ToolCall(name="find_available_slots", arguments={"job_type": job_type})])

        slots = results.get("find_available_slots", {}).get("slots", [])

        if "book_job" not in done:
            bmsg = _booking_message(user_msgs, slots)
            chosen = _pick_slot(bmsg, slots) if bmsg else None
            if chosen is not None:
                name, address = _extract_contact(bmsg)
                return AssistantTurn(tool_calls=[ToolCall(name="book_job", arguments={
                    "slot_id": chosen["slot_id"],
                    "customer_name": name,
                    "address": address,
                    "problem": first_user[:200],
                    "job_type": job_type,
                })])
            return AssistantTurn(content=_compose_offer(results.get("get_price_estimate", {}), slots))

        booking = results.get("book_job", {})

        # Booking exists — honour a cancel/reschedule request before confirming.
        if booking.get("status") == "booked":
            booking_id = booking.get("booking_id")
            if _wants_cancel(all_text) and "cancel_job" not in done:
                return AssistantTurn(tool_calls=[ToolCall(
                    name="cancel_job", arguments={"booking_id": booking_id})])
            if _wants_reschedule(all_text) and "reschedule_job" not in done:
                rmsg = _reschedule_message(user_msgs) or last_user
                target = _pick_slot(rmsg, slots)
                if target is not None and target["slot_id"] != booking.get("slot_id"):
                    return AssistantTurn(tool_calls=[ToolCall(name="reschedule_job", arguments={
                        "booking_id": booking_id, "new_slot_id": target["slot_id"]})])
                return AssistantTurn(content=(
                    "Happy to move it — which of the times I offered would you prefer?"))

        cancelled = results.get("cancel_job", {})
        if cancelled.get("status") == "cancelled":
            return AssistantTurn(content=(
                f"Done — booking {cancelled['booking_id']} is cancelled and that slot is freed up. "
                "If you'd like to rebook later, just let me know."
            ))

        rescheduled = results.get("reschedule_job", {})
        if rescheduled.get("status") == "rescheduled":
            return AssistantTurn(content=(
                f"All set — I've moved booking {rescheduled['booking_id']} to "
                f"{rescheduled['window']} with {rescheduled['technician_name']}. Anything else?"
            ))

        if booking.get("status") == "booked":
            return AssistantTurn(content=(
                f"Booked! {booking['technician_name']} will arrive {booking['window']}. "
                f"Your reference is {booking['booking_id']}. You'll get a reminder text before arrival. "
                "Anything else I can help with?"
            ))
        return AssistantTurn(content="That slot just filled up — would another time work for you?")


def _compose_offer(price: dict[str, Any], slots: list[dict[str, Any]]) -> str:
    label = price.get("label", "the job").lower()
    fee = price.get("diagnostic_fee", 0)
    low, high = price.get("repair_low", 0), price.get("repair_high", 0)
    fee_part = f"The diagnostic visit is {_money(fee)}, and " if fee else "There's no separate diagnostic fee, and "
    quote = f"{fee_part}most {label} jobs run {_money(low)}–{_money(high)}."
    if not slots:
        return quote + " I don't have an open slot in the next two days — want me to put you on the waitlist?"
    lines = "\n".join(f"  {i + 1}) {s['label']} with {s['technician_name']}" for i, s in enumerate(slots))
    return f"{quote}\nI can offer:\n{lines}\nWhich works for you?"


def get_llm() -> Any:
    provider = os.getenv("LLM_PROVIDER", "mock").lower()
    if provider == "openai":
        return OpenAILLM(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
    return MockLLM()
