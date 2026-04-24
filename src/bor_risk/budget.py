"""Passive resource tracking for pipeline runs.

BudgetTracker records LLM calls, web queries, and hazard scores for
logging and experiment accounting only.  There are no hard caps —
the pipeline always runs to completion.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class BudgetTracker:
    """Track resource consumption across a run (passive counters only)."""

    llm_calls: int = 0
    web_queries: int = 0
    hazard_scores: int = 0  # tracked, not capped
    wall_clock_start: float = field(default_factory=time.time)
    call_log: list[dict] = field(default_factory=list)

    def record_llm_call(self, purpose: str, **kwargs: object) -> None:
        """Record one LLM API call."""
        self.llm_calls += 1
        self.call_log.append(
            {"type": "llm", "purpose": purpose, "call_num": self.llm_calls, **kwargs}
        )

    def record_web_query(self, query: str, **kwargs: object) -> None:
        """Record one web search query."""
        self.web_queries += 1
        self.call_log.append(
            {"type": "web", "query": query, "call_num": self.web_queries, **kwargs}
        )

    def record_hazard_score(self, supplier: str, hazard: str) -> None:
        """Record one hazard scoring call."""
        self.hazard_scores += 1
        self.call_log.append(
            {"type": "hazard", "supplier": supplier, "hazard": hazard}
        )

    def summary(self) -> dict:
        """Return a JSON-serialisable summary of resource consumption."""
        return {
            "llm_calls": self.llm_calls,
            "web_queries": self.web_queries,
            "hazard_scores": self.hazard_scores,
            "wall_clock_seconds": round(time.time() - self.wall_clock_start, 2),
        }
