"""Tests for BudgetTracker."""

from bor_risk.budget import BudgetTracker


class TestBudgetTracker:
    def test_initial_state(self):
        bt = BudgetTracker(max_llm_calls=10, max_web_queries=15)
        assert bt.llm_calls == 0
        assert bt.web_queries == 0
        assert bt.hazard_scores == 0
        assert bt.llm_budget_remaining == 10
        assert bt.web_budget_remaining == 15
        assert not bt.budget_exhausted

    def test_record_llm_call(self):
        bt = BudgetTracker(max_llm_calls=2)
        bt.record_llm_call(purpose="test")
        assert bt.llm_calls == 1
        assert bt.llm_budget_remaining == 1
        assert len(bt.call_log) == 1
        assert bt.call_log[0]["type"] == "llm"
        assert bt.call_log[0]["purpose"] == "test"

    def test_record_web_query(self):
        bt = BudgetTracker(max_web_queries=3)
        bt.record_web_query(query="test query")
        assert bt.web_queries == 1
        assert bt.web_budget_remaining == 2
        assert bt.call_log[0]["type"] == "web"

    def test_record_hazard_score(self):
        bt = BudgetTracker()
        bt.record_hazard_score(supplier="Acme", hazard="earthquake")
        assert bt.hazard_scores == 1
        # Hazard scores are never capped.
        assert not bt.budget_exhausted

    def test_budget_exhausted(self):
        bt = BudgetTracker(max_llm_calls=1, max_web_queries=1)
        assert not bt.budget_exhausted
        bt.record_llm_call(purpose="a")
        assert not bt.budget_exhausted  # web still available
        bt.record_web_query(query="b")
        assert bt.budget_exhausted

    def test_remaining_never_negative(self):
        bt = BudgetTracker(max_llm_calls=0, max_web_queries=0)
        assert bt.llm_budget_remaining == 0
        assert bt.web_budget_remaining == 0
        bt.record_llm_call(purpose="over")
        assert bt.llm_budget_remaining == 0

    def test_summary(self):
        bt = BudgetTracker(max_llm_calls=5, max_web_queries=10)
        bt.record_llm_call(purpose="x")
        bt.record_web_query(query="y")
        bt.record_hazard_score(supplier="A", hazard="flood")
        s = bt.summary()
        assert s["llm_calls"] == 1
        assert s["web_queries"] == 1
        assert s["hazard_scores"] == 1
        assert s["max_llm_calls"] == 5
        assert s["max_web_queries"] == 10
        assert "wall_clock_seconds" in s
