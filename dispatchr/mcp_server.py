"""Dispatchr as an MCP server.

Exposes the same six dispatcher tools over the Model Context Protocol, so any
MCP client (Claude Desktop, an IDE, another agent) can quote, schedule, book,
cancel, reschedule, and escalate — not just the built-in agent loop. The tools
share one in-memory calendar for the life of the process.

Run over stdio:

    python -m dispatchr.mcp_server

Claude Desktop config (claude_desktop_config.json):

    {
      "mcpServers": {
        "dispatchr": {
          "command": "/abs/path/.venv/bin/python",
          "args": ["-m", "dispatchr.mcp_server"],
          "cwd": "/abs/path/to/dispatchr"
        }
      }
    }
"""
from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from .tools import JOB_TYPES, Tools

mcp = FastMCP("dispatchr")
_tools = Tools()

_JOBS = ", ".join(JOB_TYPES)


@mcp.tool()
def get_price_estimate(job_type: str) -> dict[str, Any]:
    """Look up the diagnostic fee and repair price range for a job type.

    Always call this before quoting a price — never invent one.
    Valid job_type values: {jobs}.
    """.format(jobs=_JOBS)
    return _tools.get_price_estimate(job_type)


@mcp.tool()
def find_available_slots(job_type: str, limit: int = 3) -> dict[str, Any]:
    """Find open appointment slots staffed by a technician with the right skill for this job type."""
    return _tools.find_available_slots(job_type, limit)


@mcp.tool()
def book_job(slot_id: str, customer_name: str, address: str, problem: str,
             job_type: str | None = None) -> dict[str, Any]:
    """Book a specific available slot (from find_available_slots) for the customer.

    Pass job_type (the classified job) so the booking can be rescheduled safely later.
    """
    return _tools.book_job(slot_id, customer_name, address, problem, job_type)


@mcp.tool()
def cancel_job(booking_id: str) -> dict[str, Any]:
    """Cancel an existing booking by its booking_id (e.g. BK-1000) and free the slot."""
    return _tools.cancel_job(booking_id)


@mcp.tool()
def reschedule_job(booking_id: str, new_slot_id: str) -> dict[str, Any]:
    """Move an existing booking to a different open slot (from find_available_slots), freeing the old one."""
    return _tools.reschedule_job(booking_id, new_slot_id)


@mcp.tool()
def escalate_to_human(reason: str, summary: str = "") -> dict[str, Any]:
    """Hand the conversation to a human dispatcher immediately.

    Use for safety emergencies (gas, fire, smoke, sparks, electric shock,
    flooding) or anything outside home-services dispatch.
    """
    return _tools.escalate_to_human(reason, summary)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
