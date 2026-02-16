"""Tests for tier-expansion graph and CLI output (mock-data path)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from bor_risk.graph import run_graph
from bor_risk.cli import main as cli_main


class TestRunGraph:
    """Tests that exercise the JSON (non-LLM) code path."""

    def test_returns_suppliers(self) -> None:
        state = run_graph("ACME", tier_depth=2)
        assert len(state["suppliers"]) > 0

    def test_returns_edges_with_evidence(self) -> None:
        state = run_graph("ACME", tier_depth=2)
        assert len(state["edges"]) > 0
        for edge in state["edges"]:
            assert len(edge["evidence_ids"]) >= 1, (
                f"Edge {edge['parent']} -> {edge['child']} has no evidence"
            )

    def test_returns_evidence(self) -> None:
        state = run_graph("ACME", tier_depth=2)
        assert len(state["evidence"]) > 0

    def test_report_text_present(self) -> None:
        state = run_graph("ACME", tier_depth=2)
        assert "Supply-Chain Risk Report" in state["report_text"]

    def test_different_company(self) -> None:
        state = run_graph("GlobalMfg", tier_depth=1)
        names = {s["name"] for s in state["suppliers"]}
        assert "PlastiCo" in names
        assert "SteelCorp" not in names

    def test_unknown_company_raises(self) -> None:
        import pytest

        with pytest.raises(KeyError, match="NonExistent"):
            run_graph("NonExistent", tier_depth=2)

    def test_tier_depth_filters(self) -> None:
        state = run_graph("ACME", tier_depth=1)
        names = {s["name"] for s in state["suppliers"]}
        assert "SteelCorp" in names
        assert "RareMinerals Ltd" not in names


class TestCLI:
    """CLI integration tests — patched to avoid real LLM calls."""

    def _run_cli(self, tmp_output_dir: Path) -> None:
        """Run CLI with LLM disabled via patch."""
        out_flag = str(tmp_output_dir / "acme.txt")
        with patch("bor_risk.cli.run_graph", wraps=run_graph) as mock_rg:
            # Intercept to force use_llm=False
            def _no_llm(company, tier_depth, **kw):
                return run_graph(company, tier_depth, use_llm=False)

            mock_rg.side_effect = _no_llm
            cli_main(["--company", "ACME", "--tier-depth", "2", "--out", out_flag])

    def test_creates_output_files(self, tmp_output_dir: Path) -> None:
        self._run_cli(tmp_output_dir)
        assert (tmp_output_dir / "acme_report.txt").exists()
        assert (tmp_output_dir / "acme_graph.json").exists()
        assert (tmp_output_dir / "acme_evidence.jsonl").exists()

    def test_graph_json_has_edges(self, tmp_output_dir: Path) -> None:
        self._run_cli(tmp_output_dir)
        data = json.loads((tmp_output_dir / "acme_graph.json").read_text())
        assert "edges" in data
        for edge in data["edges"]:
            assert len(edge["evidence_ids"]) >= 1

    def test_evidence_jsonl_valid(self, tmp_output_dir: Path) -> None:
        self._run_cli(tmp_output_dir)
        lines = (tmp_output_dir / "acme_evidence.jsonl").read_text().strip().splitlines()
        assert len(lines) > 0
        for line in lines:
            item = json.loads(line)
            assert "evidence_id" in item
