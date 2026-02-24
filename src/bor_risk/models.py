"""Pydantic models and LangGraph state definition."""

from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field, field_validator


class Supplier(BaseModel):
    """A supplier node in the supply-chain graph."""

    name: str
    lat: float
    lon: float
    tier: int = Field(ge=1)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_ids: list[str] = Field(default_factory=list)
    industry: str = ""
    product_category: str = ""
    location_description: str = ""
    relationship_type: str = ""
    evidence_source: str = "llm_only"  # "llm_only" | "web_verified" | "fixture"
    verification_url: str = ""
    verification_snippet: str = ""


class Evidence(BaseModel):
    """A single piece of evidence backing a claim."""

    evidence_id: str
    source: str
    description: str
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)


class SupplierEdge(BaseModel):
    """Directed edge between two suppliers (parent → child).

    Every edge must cite at least one evidence item.
    """

    parent: str
    child: str
    evidence_ids: list[str] = Field(min_length=1)

    @field_validator("evidence_ids")
    @classmethod
    def at_least_one(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("Every supplier edge must have >= 1 evidence_id")
        return v


class HazardScore(BaseModel):
    """Deterministic hazard score for one supplier."""

    supplier_name: str
    hazard_type: str
    score: float = Field(ge=0.0, le=1.0)
    score_100: int = Field(ge=0, le=100)
    level: str = Field(pattern=r"^(Low|Medium|High)$")
    dataset_metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM structured-output models
# ---------------------------------------------------------------------------


class CompanyProfile(BaseModel):
    """LLM structured output for company entity resolution."""

    canonical_name: str = Field(description="Full official company name")
    industry: str = Field(description="Primary industry sector")
    products: list[str] = Field(description="Main product lines or services")
    headquarters: str = Field(description="HQ city, country")
    description: str = Field(description="One-sentence company description")


class LLMSupplier(BaseModel):
    """A supplier identified by the LLM for one tier."""

    name: str
    lat: float = Field(ge=-90, le=90, description="Latitude of primary facility")
    lon: float = Field(ge=-180, le=180, description="Longitude of primary facility")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in this supplier relationship (0-1)",
    )
    rationale: str = Field(
        description="Brief explanation of why this company is a plausible supplier",
    )
    industry: str = Field(
        default="",
        description="Industry sector, e.g. 'Semiconductor Manufacturing'",
    )
    product_category: str = Field(
        default="",
        description="What they supply, e.g. 'Silicon wafers'",
    )
    location_description: str = Field(
        default="",
        description="City, country of primary facility",
    )
    relationship_type: str = Field(
        default="",
        description="Type: raw_material, component, service, logistics",
    )


class TierResponse(BaseModel):
    """LLM structured output for one tier of suppliers."""

    parent_company: str
    tier: int
    suppliers: list[LLMSupplier]


class MitigationItem(BaseModel):
    """A single mitigation recommendation from the LLM."""

    supplier_name: str
    hazard_type: str
    level: str = Field(pattern=r"^(Medium|High)$")
    actions: list[str] = Field(
        description="2-3 concrete, implementable mitigation steps",
    )
    priority: str = Field(
        pattern=r"^(P1|P2|P3)$",
        description="P1=immediate, P2=within 2 weeks, P3=ongoing",
    )
    rationale: str = Field(
        description="Why this mitigation is needed, citing specific score data",
    )


class MitigationResponse(BaseModel):
    """LLM structured output for mitigation strategies."""

    company: str
    mitigations: list[MitigationItem]


class AlternativeCandidate(BaseModel):
    """A single alternative supplier candidate with location and rationale."""

    name: str
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(
        description="Why this location has lower hazard exposure",
    )


class AlternativeSuggestion(BaseModel):
    """Suggestion to replace one high-risk supplier."""

    alternative_for: str = Field(description="Name of the high-risk supplier")
    hazard_type: str = Field(description="Dominant hazard driving the suggestion")
    candidates: list[AlternativeCandidate]
    rationale: str = Field(description="Overall reasoning for the suggestion")


class AlternativeResponse(BaseModel):
    """LLM structured output for alternative supplier suggestions."""

    company: str
    alternatives: list[AlternativeSuggestion]


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes.

    Fields with ``Annotated[..., operator.add]`` use a *reducer*: when
    multiple parallel nodes each return a partial list, LangGraph
    concatenates them automatically (fan-in merge).
    """

    company: str
    tier_depth: int
    suppliers_path: str
    use_llm: bool
    interactive: bool
    company_profile: dict
    suppliers: list[dict]
    edges: list[dict]
    evidence: list[dict]
    hazard_scores: Annotated[list[dict], operator.add]
    company_risk_summary: dict
    workflow_decision: dict
    workflow_actions: list[dict]
    llm_mitigations: list[dict]
    suggested_alternatives: list[dict]
    workflow_trace: Annotated[list[str], operator.add]
    report_text: str
    enable_web: bool
    snapshot_mode: bool
    agent_trace: list[dict]
    budget_summary: dict
    _max_web_queries: int
    _budget_tracker: object  # BudgetTracker instance, not serialised
