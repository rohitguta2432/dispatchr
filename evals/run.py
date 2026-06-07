"""Dispatchr eval suite.

Runs the agent against a golden dataset and scores the behaviours that matter
for a dispatcher: correct routing, mandatory emergency escalation, price
integrity (no invented prices), and valid scheduling. A CI-style gate exits
non-zero if any metric regresses below threshold.

    python -m evals.run            # deterministic mock provider (CI gate)
    LLM_PROVIDER=openai \\
        python -m evals.run --real # score a real LLM, incl. LLM-as-Judge tone

Each metric is one of:
  * a MIN gate (score must be >= threshold), or
  * a MAX gate (rate must be <= threshold, e.g. over-escalation).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from dispatchr.agent import Result, run_conversation
from dispatchr.llm import MockLLM, OpenAILLM, get_llm
from dispatchr.tools import PRICE_BOOK, TECHNICIANS

NOW = datetime(2026, 6, 8, 8, 0, 0)  # fixed clock -> reproducible slots
DATASET = Path(__file__).parent / "golden_dataset.json"
REPORT = Path(__file__).parent / "report.json"

TECH_SKILLS = {t["id"]: set(t["skills"]) for t in TECHNICIANS}

# name -> (direction, threshold). "min": score >= t to pass; "max": rate <= t.
GATES = {
    "action_accuracy": ("min", 0.90),
    "routing_accuracy": ("min", 0.90),
    "emergency_escalation_recall": ("min", 1.0),
    "over_escalation_rate": ("max", 0.05),
    "price_integrity": ("min", 1.0),
    "schedule_validity": ("min", 0.95),
}


def tech_skills_for_slot(slot_id: str) -> set[str]:
    return TECH_SKILLS.get(slot_id.split("-")[0], set())


def allowed_prices(job_type: str) -> set[int]:
    e = PRICE_BOOK[job_type]
    return {e["diagnostic_fee"], e["repair_low"], e["repair_high"]}


def extract_prices(text: str) -> set[int]:
    return {int(x.replace(",", "")) for x in re.findall(r"₹([\d,]+)", text)}


def predict(result: Result) -> tuple[str, str | None]:
    if result.escalated:
        action = "escalate"
    elif (bk := result.called("book_job")) and bk.result.get("status") == "booked":
        action = "book"
    elif result.called("get_price_estimate"):
        action = "quote"
    else:
        action = "clarify"
    rec = result.called("get_price_estimate") or result.called("find_available_slots")
    job_type = rec.arguments.get("job_type") if rec else None
    return action, job_type


def price_ok(result: Result, job_type: str | None) -> bool:
    if not job_type:
        return True
    return extract_prices(result.reply) <= allowed_prices(job_type)


def schedule_ok(result: Result, job_type: str | None, action: str) -> bool:
    if action == "quote":
        rec = result.called("find_available_slots")
        slots = rec.result.get("slots", []) if rec else []
        return bool(slots) and all(job_type in tech_skills_for_slot(s["slot_id"]) for s in slots)
    if action == "book":
        rec = result.called("book_job")
        if not rec or rec.result.get("status") != "booked":
            return False
        return job_type in tech_skills_for_slot(rec.arguments.get("slot_id", ""))
    return True


def judge_tone(reply: str) -> int | None:
    """LLM-as-Judge professionalism score (1-5). Real provider only."""
    if reply.strip() == "":
        return None
    client = OpenAILLM(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )._client
    rubric = (
        "Rate the professionalism and warmth of this home-services dispatcher reply "
        "on an integer scale 1-5 (5=excellent). Reply with only the number.\n\n" + reply
    )
    out = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": rubric}],
        temperature=0,
    )
    m = re.search(r"[1-5]", out.choices[0].message.content or "")
    return int(m.group()) if m else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Dispatchr eval suite.")
    parser.add_argument("--real", action="store_true", help="use the configured real LLM + LLM-as-Judge tone")
    args = parser.parse_args()

    llm = get_llm() if args.real else MockLLM()
    judge = args.real and isinstance(llm, OpenAILLM)
    cases = json.loads(DATASET.read_text())

    rows, failures = [], []
    tone_scores: list[int] = []
    for c in cases:
        result = run_conversation(c["messages"], llm=llm, now=NOW)
        action, job_type = predict(result)
        row = {
            "id": c["id"],
            "expected_action": c["expected_action"],
            "got_action": action,
            "expected_job_type": c["expected_job_type"],
            "got_job_type": job_type,
            "action_ok": action == c["expected_action"],
            "routing_ok": job_type == c["expected_job_type"],
            "price_ok": price_ok(result, job_type),
            "schedule_ok": schedule_ok(result, job_type, action),
            "emergency": c["emergency"],
        }
        if judge:
            score = judge_tone(result.reply)
            row["tone"] = score
            if score is not None:
                tone_scores.append(score)
        rows.append(row)
        if not row["action_ok"]:
            failures.append(f"{c['id']}: expected {c['expected_action']}, got {action}")

    routing = [r for r in rows if r["expected_action"] in ("quote", "book")]
    emergencies = [r for r in rows if r["emergency"]]
    non_emerg = [r for r in rows if not r["emergency"]]
    quotes = [r for r in rows if r["got_action"] in ("quote", "book")]

    def mean(xs: list[bool]) -> float:
        return sum(xs) / len(xs) if xs else 1.0

    metrics = {
        "action_accuracy": mean([r["action_ok"] for r in rows]),
        "routing_accuracy": mean([r["routing_ok"] for r in routing]),
        "emergency_escalation_recall": mean([r["got_action"] == "escalate" for r in emergencies]),
        "over_escalation_rate": (sum(r["got_action"] == "escalate" for r in non_emerg) / len(non_emerg)) if non_emerg else 0.0,
        "price_integrity": mean([r["price_ok"] for r in quotes]),
        "schedule_validity": mean([r["schedule_ok"] for r in quotes]),
    }

    print(f"\nDispatchr evals — {len(cases)} cases — provider: {'real' if args.real else 'mock'}\n")
    all_pass = True
    for name, (direction, thr) in GATES.items():
        score = metrics[name]
        ok = score <= thr if direction == "max" else score >= thr
        all_pass &= ok
        cmp = "<=" if direction == "max" else ">="
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<28} {score:6.1%}  ({cmp} {thr:.0%})")
    if tone_scores:
        print(f"  ----  {'tone (LLM-as-Judge)':<28} {sum(tone_scores) / len(tone_scores):.2f} / 5")

    if failures:
        print("\n  tracked failures:")
        for f in failures:
            print(f"    - {f}")

    REPORT.write_text(json.dumps({"metrics": metrics, "rows": rows, "passed": all_pass}, indent=2))
    print(f"\n  report -> {REPORT}")
    print(f"\n{'GATE PASSED' if all_pass else 'GATE FAILED'}\n")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
