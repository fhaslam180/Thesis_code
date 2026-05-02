"""Tests for claim verification and evidence-source weighting."""

import pytest
from unittest.mock import patch

from bor_risk.budget import BudgetTracker
from bor_risk.models import Claim
from bor_risk.tools import compute_risk_summary, verify_claim_against_evidence, verify_suppliers_batch


# ---------------------------------------------------------------------------
# Existing pipeline verification tests (Condition B)
# ---------------------------------------------------------------------------


class TestVerifySuppliersBatch:
    def _make_suppliers(self):
        return [
            {"name": "HighConf", "lat": 0, "lon": 0, "tier": 1,
             "confidence": 0.9, "evidence_source": "llm_only"},
            {"name": "MedConf", "lat": 0, "lon": 0, "tier": 1,
             "confidence": 0.7, "evidence_source": "llm_only"},
            {"name": "LowConf", "lat": 0, "lon": 0, "tier": 1,
             "confidence": 0.3, "evidence_source": "llm_only"},
        ]

    def test_verifies_in_descending_confidence(self):
        """Pipeline verifies highest-confidence suppliers first."""
        suppliers = self._make_suppliers()
        budget = BudgetTracker()
        call_order = []

        def mock_search(query, **kwargs):
            call_order.append(query)
            return []

        with patch("bor_risk.search.search_web_snapshot", side_effect=mock_search):
            updated, evidence = verify_suppliers_batch(
                suppliers, "TestCo", budget, snapshot_mode=True
            )

        # Should have tried to verify HighConf first (highest confidence).
        assert len(call_order) >= 1
        assert "HighConf" in call_order[0]

    def test_upgrades_on_co_mention(self):
        """Verification upgrades evidence_source when co-mention found."""
        suppliers = [
            {"name": "TSMC", "lat": 24.8, "lon": 121.0, "tier": 1,
             "confidence": 0.8, "evidence_source": "llm_only"},
        ]
        budget = BudgetTracker()

        mock_results = [
            {"title": "Apple suppliers",
             "url": "https://example.com/article",
             "content": "Apple relies on TSMC as a key supplier for its A-series chips.",
             "retrieved_at": "2025-01-01T00:00:00Z"},
        ]

        with patch("bor_risk.search.search_web_snapshot", return_value=mock_results):
            updated, evidence = verify_suppliers_batch(
                suppliers, "Apple", budget, snapshot_mode=True
            )

        tsmc = next(s for s in updated if s["name"] == "TSMC")
        assert tsmc["evidence_source"] == "web_verified"
        assert tsmc["verification_url"] == "https://example.com/article"
        assert tsmc["confidence"] == 0.9  # boosted by 0.1
        assert len(evidence) == 1

    def test_no_upgrade_without_co_mention(self):
        """No upgrade if only supplier name appears (no company co-mention)."""
        suppliers = [
            {"name": "TSMC", "lat": 24.8, "lon": 121.0, "tier": 1,
             "confidence": 0.8, "evidence_source": "llm_only"},
        ]
        budget = BudgetTracker()

        # Content mentions TSMC but not Apple.
        mock_results = [
            {"title": "TSMC news",
             "url": "https://example.com/article",
             "content": "TSMC is the world's largest semiconductor supplier.",
             "retrieved_at": "2025-01-01T00:00:00Z"},
        ]

        with patch("bor_risk.search.search_web_snapshot", return_value=mock_results):
            updated, evidence = verify_suppliers_batch(
                suppliers, "Apple", budget, snapshot_mode=True
            )

        tsmc = next(s for s in updated if s["name"] == "TSMC")
        assert tsmc["evidence_source"] == "llm_only"


class TestEvidenceSourceWeighting:
    """SMHEI does not apply confidence/evidence penalties to exposure_index.

    Evidence source is preserved as informational metadata only. Both
    llm_only and fixture suppliers get the same exposure_index for equal
    hazard scores, which is the intentional SMHEI design (purely physical).
    """
    def _make_hazard_scores_6(self, supplier_name: str, score: float) -> list[dict]:
        return [
            {"supplier_name": supplier_name, "hazard_type": h, "score": score,
             "score_100": round(score * 100), "level": "Medium", "dataset_metadata": {}}
            for h in ("earthquake", "flood", "wildfire", "cyclone", "heat_stress", "drought")
        ]

    def test_llm_only_and_verified_get_same_exposure_index(self):
        """SMHEI does not penalize llm_only suppliers — purely physical aggregation."""
        hazard_scores = (
            self._make_hazard_scores_6("Verified", 0.5)
            + self._make_hazard_scores_6("Unverified", 0.5)
        )
        suppliers = [
            {"name": "Verified", "tier": 1, "confidence": 0.8, "evidence_source": "web_verified"},
            {"name": "Unverified", "tier": 1, "confidence": 0.8, "evidence_source": "llm_only"},
        ]
        summary = compute_risk_summary(hazard_scores, suppliers)
        risks = {r["supplier_name"]: r for r in summary["supplier_risks"]}
        assert risks["Verified"]["exposure_index"] == risks["Unverified"]["exposure_index"]

    def test_fixture_sources_pass_through(self):
        """Fixture suppliers pass through with correct exposure_index."""
        hazard_scores = self._make_hazard_scores_6("Fixture", 0.5)
        suppliers = [{"name": "Fixture", "tier": 1, "confidence": 0.8, "evidence_source": "fixture"}]
        summary = compute_risk_summary(hazard_scores, suppliers)
        risk = summary["supplier_risks"][0]
        assert risk["exposure_index"] == pytest.approx(0.5, abs=1e-4)
        assert "exposure_band" in risk
        assert "dominant_hazard" in risk


# ---------------------------------------------------------------------------
# New: verify_claim_against_evidence unit tests
# ---------------------------------------------------------------------------


def _make_claim(company: str, supplier: str, tier: int = 1) -> Claim:
    import hashlib
    cid = hashlib.sha256(
        f"{company}|{supplier}|SUPPLIES_TO|{tier}".encode()
    ).hexdigest()[:12]
    return Claim(
        claim_id=cid,
        display_id="C001",
        subject_entity_id=company,
        object_entity_id=supplier,
        normalized_name=supplier.lower(),
        confidence=0.8,
    )


class TestVerifyClaimAgainstEvidence:
    """Unit tests for verify_claim_against_evidence (heuristic path)."""

    def test_co_mention_missing_returns_unknown(self):
        """Company not in evidence text → UNKNOWN verdict (Stage 1)."""
        claim = _make_claim("Apple", "TSMC")
        # Text mentions TSMC but not Apple.
        evidence = "TSMC is a major semiconductor manufacturer based in Taiwan."

        result = verify_claim_against_evidence(claim, evidence, budget=None)
        assert result.verdict == "UNKNOWN"

    def test_supplier_missing_returns_unknown(self):
        """Supplier not in evidence text → UNKNOWN verdict (Stage 1)."""
        claim = _make_claim("Apple", "TSMC")
        # Text mentions Apple but not TSMC.
        evidence = "Apple is a major technology company headquartered in Cupertino."

        result = verify_claim_against_evidence(claim, evidence, budget=None)
        assert result.verdict == "UNKNOWN"

    def test_termination_language_stage1(self):
        """Termination/dispute language → DISPUTED verdict (Stage 1)."""
        claim = _make_claim("Apple", "TSMC")
        evidence = (
            "Apple has no longer relied on TSMC as a supplier and "
            "has terminated the relationship."
        )

        result = verify_claim_against_evidence(claim, evidence, budget=None)
        assert result.verdict == "DISPUTED"

    def test_alias_variant_allows_brand_name_co_mention(self):
        """Supplier variants should match evidence that uses a shorter brand name."""
        claim = _make_claim("Apple", "Pegatron Corporation")
        evidence = "Pegatron, which is a supplier to Apple, opened a new facility."

        with patch("bor_risk.tools.load_prompts", return_value={}):
            result = verify_claim_against_evidence(claim, evidence, budget=None)
        assert result.verdict in {"SUPPORTED", "WEAK"}

    def test_unrelated_negation_does_not_force_disputed(self):
        """A negative word elsewhere on the page should not override support."""
        claim = _make_claim("Apple", "TSMC")
        evidence = (
            "TSMC is a key supplier for Apple silicon chips. "
            "A sidebar says another retailer dropped the price of a laptop."
        )

        with patch("bor_risk.tools.load_prompts", return_value={}):
            result = verify_claim_against_evidence(claim, evidence, budget=None)
        assert result.verdict in {"SUPPORTED", "WEAK"}

    def test_substring_guard_with_mocked_llm(self):
        """Stage 3: LLM returns SUPPORTED with non-verbatim quote → downgrade to WEAK."""
        from unittest.mock import patch, MagicMock
        from bor_risk.models import VerificationVerdict

        claim = _make_claim("Apple", "TSMC")
        evidence = "Apple relies on TSMC as a key supplier for semiconductors."
        budget = BudgetTracker()

        # LLM returns SUPPORTED but with a quote NOT in the evidence text
        fake_verdict = VerificationVerdict(
            verdict="SUPPORTED",
            rationale="Apple and TSMC have a supplier relationship.",
            supporting_quote="TSMC manufactures chips exclusively for Apple",  # NOT verbatim
            direction="supplier_to_company",
        )

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = fake_verdict

        with patch("bor_risk.tools.ChatOpenAI", return_value=mock_llm):
            result = verify_claim_against_evidence(claim, evidence, budget)

        # Quote not in evidence → SUPPORTED downgraded to WEAK
        assert result.verdict == "WEAK"
        assert result.supporting_quote == ""

    def test_substring_guard_keeps_valid_quote(self):
        """Stage 3: LLM returns SUPPORTED with verbatim quote → stays SUPPORTED."""
        from unittest.mock import patch, MagicMock
        from bor_risk.models import VerificationVerdict

        claim = _make_claim("Apple", "TSMC")
        evidence = "Apple relies on TSMC as a key supplier for semiconductors."
        budget = BudgetTracker()

        # LLM returns SUPPORTED with a quote that IS verbatim in the evidence
        verbatim_quote = "TSMC as a key supplier"
        fake_verdict = VerificationVerdict(
            verdict="SUPPORTED",
            rationale="TSMC is Apple's supplier.",
            supporting_quote=verbatim_quote,
            direction="supplier_to_company",
        )

        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = fake_verdict

        with patch("bor_risk.tools.ChatOpenAI", return_value=mock_llm):
            result = verify_claim_against_evidence(claim, evidence, budget)

        assert result.verdict == "SUPPORTED"
        assert result.supporting_quote == verbatim_quote
