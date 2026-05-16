"""Tests for typed RAG discovery query prompt (Phase D)."""
from __future__ import annotations

from datetime import datetime, timezone


def _load_prompt() -> str:
    from bor_risk.utils import load_prompts
    return load_prompts()["discovery_queries_prompt"]


class TestDiscoveryQueriesPrompt:

    def test_renders_without_key_error(self):
        template = _load_prompt()
        rendered = template.format(
            focal_company="Apple",
            target_company="Apple",
            n_queries=6,
            industry="Consumer Electronics",
            products="iPhone, Mac, iPad",
            tier=1,
            year=2024,
        )
        assert "Apple" in rendered

    def test_no_target_domain_or_target_slug_variable(self):
        template = _load_prompt()
        assert "{target_domain}" not in template
        assert "{target_slug}" not in template

    def test_year_resolves_to_previous_year(self):
        expected_year = datetime.now(timezone.utc).year - 1
        template = _load_prompt()
        rendered = template.format(
            focal_company="Apple",
            target_company="Apple",
            n_queries=6,
            industry="Consumer Electronics",
            products="iPhone",
            tier=1,
            year=expected_year,
        )
        assert str(expected_year) in rendered

    def test_tier2_prompt_references_target_company_not_focal(self):
        template = _load_prompt()
        rendered = template.format(
            focal_company="Apple",
            target_company="TSMC",
            n_queries=4,
            industry="Semiconductor",
            products="Logic chips",
            tier=2,
            year=2024,
        )
        # The prompt should mention target_company for tier-2
        assert "TSMC" in rendered

    def test_generate_discovery_queries_passes_year(self):
        from unittest.mock import patch, MagicMock
        from bor_risk.tools import generate_discovery_queries

        expected_year = datetime.now(timezone.utc).year - 1
        captured_prompts: list[str] = []

        def fake_invoke(messages):
            captured_prompts.append(messages[0].content)
            mock_resp = MagicMock()
            mock_resp.content = "query one\nquery two\n"
            return mock_resp

        with patch("bor_risk.tools._make_llm") as mock_llm:
            mock_llm.return_value.invoke.side_effect = fake_invoke
            generate_discovery_queries(
                focal_company="Apple",
                target_company="Apple",
                tier=1,
                industry="Consumer Electronics",
                products_str="iPhone, Mac",
            )

        assert captured_prompts, "LLM was not invoked"
        assert str(expected_year) in captured_prompts[0]
