"""Supplier-name normalization and alias helpers."""

from __future__ import annotations

import re

_LEGAL_SUFFIXES_RE = re.compile(
    r"[,.]?\s*\b(inc|incorporated|corp|ltd|se|ag|gmbh|co|plc|llc|nv|sa|sas|kk|spa|bv|oy|ab"
    r"|corporation|company|limited)\b\.?\s*$",
    re.IGNORECASE,
)

_PUNCT_RE = re.compile(r"[.,\-'\"]")
_PARENS_RE = re.compile(r"\((.*?)\)")
_CORPORATE_DESCRIPTOR_TOKENS = frozenset({
    "electronics",
    "manufacturing",
    "technology",
    "technologies",
    "group",
})


def normalize_spacing(value: str) -> str:
    """Lowercase, remove most punctuation, and collapse whitespace."""
    value = _PUNCT_RE.sub(" ", value.strip().lower())
    value = re.sub(r"[()]", " ", value)
    return " ".join(value.split())


def strip_legal_suffixes(name: str) -> str:
    """Lowercase a supplier name and strip trailing legal/generic suffixes."""
    current = normalize_spacing(name)
    previous = ""
    while current and current != previous:
        previous = current
        current = _LEGAL_SUFFIXES_RE.sub("", current).strip()
    return current or normalize_spacing(name)


def name_variants(name: str, aliases: list[str] | None = None) -> set[str]:
    """Return normalized aliases for supplier/entity matching.

    The variants intentionally stay conservative. They cover common legal
    suffixes, parenthetical acronyms such as ``TSMC (...)``, and brand names
    followed only by generic descriptors such as ``Samsung Electronics``.
    """
    raw = name.strip()
    variants: set[str] = set()

    def _add(value: str) -> None:
        normalized = normalize_spacing(value)
        if normalized:
            variants.add(normalized)
            stripped = strip_legal_suffixes(normalized)
            if stripped:
                variants.add(stripped)

    _add(raw)

    paren_matches = _PARENS_RE.findall(raw)
    for match in paren_matches:
        _add(match)

    without_parens = _PARENS_RE.sub(" ", raw)
    _add(without_parens)

    base = strip_legal_suffixes(without_parens)
    tokens = base.split()
    if len(tokens) >= 2 and all(t in _CORPORATE_DESCRIPTOR_TOKENS for t in tokens[1:]):
        if len(tokens[0]) > 3:
            variants.add(tokens[0])

    # Acronym from capitalized words, useful for names like Taiwan Semiconductor
    # Manufacturing Company. Avoid generating two-letter variants such as LG.
    words = re.findall(r"\b[A-Z][A-Za-z&]+\b", raw)
    if len(words) >= 3:
        acronym = "".join(w[0] for w in words).lower()
        if len(acronym) >= 3:
            variants.add(acronym)

    for alias in aliases or []:
        _add(alias)

    return variants


def canonical_name_key(name: str) -> str:
    """Return the primary normalized key used for de-duplicated evaluation."""
    variants = name_variants(name)
    if not variants:
        return normalize_spacing(name)
    return min(variants, key=lambda v: (len(v.split()), len(v), v))


def names_match(left: str, right: str) -> bool:
    """Conservative supplier-name match using normalized aliases."""
    left_variants = name_variants(left)
    right_variants = name_variants(right)
    if left_variants & right_variants:
        return True

    left_key = strip_legal_suffixes(left)
    right_key = strip_legal_suffixes(right)
    if not left_key or not right_key:
        return False

    left_tokens = set(left_key.split())
    right_tokens = set(right_key.split())
    if len(left_tokens) >= 2 and len(right_tokens) >= 2:
        overlap = left_tokens & right_tokens
        return overlap == left_tokens or overlap == right_tokens

    return False
