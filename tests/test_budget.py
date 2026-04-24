"""Tests for BudgetTracker (passive counters only — no hard caps)."""

from bor_risk.budget import BudgetTracker


class TestBudgetTracker:
    def test_record_llm_call(self):
        bt = BudgetTracker()
        bt.record_llm_call(purpose="test")
        assert bt.llm_calls == 1
        assert len(bt.call_log) == 1
        assert bt.call_log[0]["type"] == "llm"
        assert bt.call_log[0]["purpose"] == "test"

    def test_record_web_query(self):
        bt = BudgetTracker()
        bt.record_web_query(query="test query")
        assert bt.web_queries == 1
        assert bt.call_log[0]["type"] == "web"

    def test_record_hazard_score(self):
        bt = BudgetTracker()
        bt.record_hazard_score(supplier="Acme", hazard="earthquake")
        assert bt.hazard_scores == 1

    def test_summary(self):
        bt = BudgetTracker()
        bt.record_llm_call(purpose="x")
        bt.record_web_query(query="y")
        bt.record_hazard_score(supplier="A", hazard="flood")
        s = bt.summary()
        assert s["llm_calls"] == 1
        assert s["web_queries"] == 1
        assert s["hazard_scores"] == 1
        assert "wall_clock_seconds" in s
        assert "max_llm_calls" not in s
        assert "max_web_queries" not in s
