"""Tests for proximity-filtered location resolution (Step 1 fixes)."""

from __future__ import annotations

from bor_risk.models import Claim, EvidenceSpan, LocationCandidate
from bor_risk.tools import LOCATION_ACCEPT_THRESHOLD, resolve_locations_for_claims


def _make_claim(
    name: str = "TestSupplier",
    evidence_refs: list[str] | None = None,
    location_candidates: list[LocationCandidate] | None = None,
    verdict: str = "UNKNOWN",
    supporting_spans: list[EvidenceSpan] | None = None,
    place_name: str = "",
) -> Claim:
    llm_loc = LocationCandidate(
        lat=37.338,
        lon=-121.886,
        confidence=0.5,
        source="llm",
        place_name=place_name,
    )
    return Claim(
        claim_id="abc123",
        display_id="C001",
        subject_entity_id="Apple",
        object_entity_id=name,
        normalized_name=name.lower(),
        confidence=0.7,
        evidence_refs=evidence_refs or [],
        location_candidates=location_candidates or [llm_loc],
        verdict=verdict,
        supporting_spans=supporting_spans or [],
    )


class TestSingleMentionUnverified:
    """A single proximate city mention in unverified evidence must not displace coords."""

    def test_single_mention_unverified_rejected(self):
        # "Paris" appears right next to "TestSupplier" — but verdict is UNKNOWN
        ev = "TestSupplier headquarters in Paris for European ops."
        claim = _make_claim(evidence_refs=["E1"])
        result = resolve_locations_for_claims([claim], {"E1": ev})
        best = max(result[0].location_candidates, key=lambda lc: lc.confidence)
        assert best.source == "llm", "single unverified mention must not displace LLM coords"
        assert best.confidence == 0.5

    def test_proximity_check_rejects_distant_city(self):
        # "Paris" is 500+ chars from any mention of "Broadcom"
        padding = "A" * 600
        ev = f"Broadcom Inc operates in San Jose. {padding} Paris is in France."
        claim = _make_claim(name="Broadcom", evidence_refs=["E1"])
        result = resolve_locations_for_claims([claim], {"E1": ev})
        best = max(result[0].location_candidates, key=lambda lc: lc.confidence)
        assert best.source == "llm", "Broadcom should keep San Jose coords, not Paris"


class TestSingleMentionExceptions:
    """A single mention is promoted when evidence quality is high or it confirms LLM."""

    def test_single_mention_supported_verdict_accepted(self):
        ev = "TestSupplier manufactures components in Seoul."
        claim = _make_claim(evidence_refs=["E1"], verdict="SUPPORTED")
        result = resolve_locations_for_claims([claim], {"E1": ev})
        best = max(result[0].location_candidates, key=lambda lc: lc.confidence)
        assert best.source == "evidence"
        assert best.confidence == 0.75

    def test_single_mention_with_supporting_span_accepted(self):
        ev = "TestSupplier has its fab in Seoul."
        span = EvidenceSpan(evidence_id="E1", quote="TestSupplier has its fab in Seoul")
        claim = _make_claim(evidence_refs=["E1"], supporting_spans=[span])
        result = resolve_locations_for_claims([claim], {"E1": ev})
        best = max(result[0].location_candidates, key=lambda lc: lc.confidence)
        assert best.source == "evidence"
        assert best.confidence == 0.75

    def test_single_mention_confirms_llm_city_accepted(self):
        # City in evidence matches LLM-supplied place_name
        ev = "TestSupplier main site in Seoul."
        claim = _make_claim(evidence_refs=["E1"], place_name="Seoul")
        result = resolve_locations_for_claims([claim], {"E1": ev})
        best = max(result[0].location_candidates, key=lambda lc: lc.confidence)
        assert best.source == "evidence"
        assert best.confidence == 0.75


class TestMultipleMentions:
    """Multiple proximate mentions produce higher confidence."""

    def _ev_with_n_mentions(self, name: str, city: str, n: int) -> str:
        segment = f"{name} facility in {city}. "
        return segment * n

    def test_two_nearby_mentions_accepted(self):
        ev = self._ev_with_n_mentions("TestSupplier", "seoul", 2)
        claim = _make_claim(evidence_refs=["E1"])
        result = resolve_locations_for_claims([claim], {"E1": ev})
        best = max(result[0].location_candidates, key=lambda lc: lc.confidence)
        assert best.source == "evidence"
        assert best.confidence == 0.85

    def test_three_mentions_max_confidence(self):
        ev = self._ev_with_n_mentions("TestSupplier", "seoul", 3)
        claim = _make_claim(evidence_refs=["E1"])
        result = resolve_locations_for_claims([claim], {"E1": ev})
        best = max(result[0].location_candidates, key=lambda lc: lc.confidence)
        assert best.confidence == 0.95

    def test_most_mentioned_city_wins(self):
        # Seoul mentioned twice, Tokyo once — Seoul should win
        ev = (
            "TestSupplier plant in seoul. TestSupplier expansion in seoul. "
            "TestSupplier visiting tokyo."
        )
        claim = _make_claim(evidence_refs=["E1"])
        result = resolve_locations_for_claims([claim], {"E1": ev})
        best = max(result[0].location_candidates, key=lambda lc: lc.confidence)
        assert best.place_name.lower() == "seoul"


class TestGraphNodeThresholdGuard:
    """resolve_locations_node must not overwrite coords when confidence < threshold."""

    def test_threshold_constant_is_accessible(self):
        assert LOCATION_ACCEPT_THRESHOLD == 0.70

    def test_below_threshold_does_not_displace(self):
        # Verify that a confidence-0.65 candidate (1 unverified mention) is not
        # applied. We test this by calling resolve_locations_for_claims with a
        # single unverified mention and confirming the LLM candidate survives.
        ev = "TestSupplier office in paris."  # single mention, UNKNOWN verdict
        claim = _make_claim(evidence_refs=["E1"])
        result = resolve_locations_for_claims([claim], {"E1": ev})
        # Should still have the LLM candidate as best (confidence 0.5, not replaced)
        best = max(result[0].location_candidates, key=lambda lc: lc.confidence)
        assert best.source == "llm"


class TestRegressionCases:
    """Specific regression: Samsung stays Seoul, Broadcom stays San Jose."""

    def test_samsung_stays_seoul_not_shenzhen(self):
        # Shenzhen mentioned far from Samsung Electronics
        padding = "X" * 500
        ev = f"Samsung Electronics key fab in Seoul. {padding} shenzhen is a major tech hub."
        llm_loc = LocationCandidate(lat=37.266, lon=127.005, confidence=0.5, source="llm", place_name="Seoul")
        claim = _make_claim(name="Samsung Electronics", evidence_refs=["E1"],
                            location_candidates=[llm_loc])
        result = resolve_locations_for_claims([claim], {"E1": ev})
        best = max(result[0].location_candidates, key=lambda lc: lc.confidence)
        # Either the LLM Seoul is kept, or "seoul" evidence is found — in both
        # cases the final place should not be Shenzhen.
        assert best.place_name.lower() != "shenzhen"

    def test_broadcom_stays_san_jose_not_paris(self):
        padding = "Y" * 500
        ev = f"Broadcom Inc R&D in san jose. {padding} paris hosts a conference."
        claim = _make_claim(name="Broadcom", evidence_refs=["E1"])
        result = resolve_locations_for_claims([claim], {"E1": ev})
        best = max(result[0].location_candidates, key=lambda lc: lc.confidence)
        assert best.place_name.lower() != "paris"
