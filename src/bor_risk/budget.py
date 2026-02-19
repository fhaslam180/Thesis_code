"""Budget tracking for LLM calls and web queries.

Budget caps apply to LLM calls and web queries (the expensive,
non-deterministic resources). Hazard scoring is deterministic
and tracked for reporting but never capped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class BudgetTracker:
    """Track resource consumption across a run.

    Attributes
    ----------
    max_llm_calls : int
        Maximum LLM API calls allowed (discovery, mitigations, etc.).
    max_web_queries : int
        Maximum web search queries allowed.
    """

    max_llm_calls: int = 20
    max_web_queries: int = 30

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
        """Record one hazard scoring call (tracked, never capped)."""
        self.hazard_scores += 1
        self.call_log.append(
            {"type": "hazard", "supplier": supplier, "hazard": hazard}
        )

    @property
    def llm_budget_remaining(self) -> int:
        """How many LLM calls are left."""
        return max(0, self.max_llm_calls - self.llm_calls)

    @property
    def web_budget_remaining(self) -> int:
        """How many web queries are left."""
        return max(0, self.max_web_queries - self.web_queries)

    @property
    def budget_exhausted(self) -> bool:
        """True when both LLM and web budgets are fully spent."""
        return (
            self.llm_calls >= self.max_llm_calls
            and self.web_queries >= self.max_web_queries
        )

    def summary(self) -> dict:
        """Return a JSON-serialisable summary of resource consumption."""
        return {
            "llm_calls": self.llm_calls,
            "web_queries": self.web_queries,
            "hazard_scores": self.hazard_scores,
            "max_llm_calls": self.max_llm_calls,
            "max_web_queries": self.max_web_queries,
            "wall_clock_seconds": round(time.time() - self.wall_clock_start, 2),
        }
