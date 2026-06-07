"""Tools the Dispatchr agent can call.

Each tool is a plain function over seed data plus an in-memory calendar. The
`Tools` object owns the mutable state (bookings) so every conversation — and
every eval case — gets an isolated, reproducible world.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

_DATA = Path(__file__).parent / "data"

with (_DATA / "price_book.json").open() as f:
    PRICE_BOOK: dict[str, dict[str, Any]] = json.load(f)
with (_DATA / "technicians.json").open() as f:
    TECHNICIANS: list[dict[str, Any]] = json.load(f)

# Canonical job types — the only values the agent is allowed to route to.
JOB_TYPES: list[str] = list(PRICE_BOOK.keys())

# Working windows, 24h start hours -> (start, end).
_WINDOWS = [(9, 11), (11, 13), (14, 16), (16, 18)]


def _fmt_hour(h: int) -> str:
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12}:00 {suffix}"


def _day_label(day: datetime, now: datetime) -> str:
    delta = (day.date() - now.date()).days
    if delta == 0:
        return "Today"
    if delta == 1:
        return "Tomorrow"
    return day.strftime("%a %d %b")


class Tools:
    def __init__(self, now: datetime | None = None) -> None:
        self.now = now or datetime.now()
        self._booked: set[str] = set()
        self._bookings: dict[str, dict[str, Any]] = {}
        self._slots: dict[str, dict[str, Any]] = self._build_slots()

    # ------------------------------------------------------------------ slots
    def _build_slots(self) -> dict[str, dict[str, Any]]:
        slots: dict[str, dict[str, Any]] = {}
        base = self.now.replace(minute=0, second=0, microsecond=0)
        for day_off in (0, 1):
            day = base + timedelta(days=day_off)
            for w_idx, (sh, eh) in enumerate(_WINDOWS):
                start = day.replace(hour=sh)
                if start <= self.now:  # window already started/passed today
                    continue
                end = day.replace(hour=eh)
                for tech in TECHNICIANS:
                    sid = f"{tech['id']}-{day_off}-{w_idx}"
                    slots[sid] = {
                        "slot_id": sid,
                        "technician_id": tech["id"],
                        "technician_name": tech["name"],
                        "skills": tech["skills"],
                        "start": start,
                        "end": end,
                        "label": f"{_day_label(start, self.now)} {_fmt_hour(sh)}–{_fmt_hour(eh)}",
                    }
        return slots

    # ------------------------------------------------------------------ tools
    def get_price_estimate(self, job_type: str) -> dict[str, Any]:
        entry = PRICE_BOOK.get(job_type)
        if entry is None:
            return {"error": f"unknown job_type '{job_type}'", "known_job_types": JOB_TYPES}
        return {"job_type": job_type, **entry}

    def find_available_slots(self, job_type: str, limit: int = 3) -> dict[str, Any]:
        if job_type not in PRICE_BOOK:
            return {"error": f"unknown job_type '{job_type}'", "known_job_types": JOB_TYPES}
        matches: list[dict[str, Any]] = []
        seen_windows: set[datetime] = set()
        for s in sorted(self._slots.values(), key=lambda s: s["start"]):
            if job_type not in s["skills"] or s["slot_id"] in self._booked:
                continue
            if s["start"] in seen_windows:  # one technician per time window
                continue
            seen_windows.add(s["start"])
            matches.append({"slot_id": s["slot_id"], "technician_name": s["technician_name"], "label": s["label"]})
            if len(matches) >= limit:
                break
        return {"job_type": job_type, "slots": matches}

    def book_job(self, slot_id: str, customer_name: str, address: str, problem: str) -> dict[str, Any]:
        slot = self._slots.get(slot_id)
        if slot is None:
            return {"error": f"unknown slot_id '{slot_id}'"}
        if slot_id in self._booked:
            return {"error": f"slot '{slot_id}' is already booked"}
        self._booked.add(slot_id)
        booking_id = f"BK-{1000 + len(self._bookings)}"
        booking = {
            "booking_id": booking_id,
            "technician_name": slot["technician_name"],
            "window": slot["label"],
            "customer_name": customer_name,
            "address": address,
            "problem": problem,
        }
        self._bookings[booking_id] = booking
        return {"status": "booked", **booking}

    def escalate_to_human(self, reason: str, summary: str = "") -> dict[str, Any]:
        return {
            "status": "escalated",
            "reason": reason,
            "message": "Flagged for a human dispatcher to take over immediately.",
            "summary": summary,
        }

    # --------------------------------------------------------------- dispatch
    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        fn = {
            "get_price_estimate": self.get_price_estimate,
            "find_available_slots": self.find_available_slots,
            "book_job": self.book_job,
            "escalate_to_human": self.escalate_to_human,
        }.get(name)
        if fn is None:
            return {"error": f"unknown tool '{name}'"}
        try:
            return fn(**arguments)
        except TypeError as exc:
            return {"error": f"bad arguments for '{name}': {exc}"}


def tool_schemas() -> list[dict[str, Any]]:
    """OpenAI-style function schemas exposed to a real LLM."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_price_estimate",
                "description": "Look up the diagnostic fee and repair price range for a job type. "
                "Always call this before quoting any price — never invent prices.",
                "parameters": {
                    "type": "object",
                    "properties": {"job_type": {"type": "string", "enum": JOB_TYPES}},
                    "required": ["job_type"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "find_available_slots",
                "description": "Find open appointment slots staffed by a technician who has the skill for this job type.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_type": {"type": "string", "enum": JOB_TYPES},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                    "required": ["job_type"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "book_job",
                "description": "Book a specific available slot for the customer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "slot_id": {"type": "string"},
                        "customer_name": {"type": "string"},
                        "address": {"type": "string"},
                        "problem": {"type": "string"},
                    },
                    "required": ["slot_id", "customer_name", "address", "problem"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "escalate_to_human",
                "description": "Hand the conversation to a human dispatcher immediately. Use for safety "
                "emergencies (gas, fire, smoke, sparks, flooding, electric shock) or anything out of scope.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string"},
                        "summary": {"type": "string"},
                    },
                    "required": ["reason"],
                },
            },
        },
    ]
