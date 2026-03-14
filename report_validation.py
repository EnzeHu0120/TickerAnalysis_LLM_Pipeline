from __future__ import annotations

"""
report_validation.py

Schema-level validation helpers for the final LLM report.
These functions are intentionally conservative: they do not mutate the report,
they only describe what is wrong. llm_pipeline is responsible for deciding
whether to repair via code, reprompt the LLM, or fail fast.
"""

from typing import Any, Dict, List


CANONICAL_RATINGS = {"Overweight", "Equal-weight", "Hold", "Underweight", "Reduce"}
SCENARIOS = ["Bear", "Consensus", "Bull"]


def validate_rating(report: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    rating = report.get("rating")
    if rating not in CANONICAL_RATINGS:
        errs.append(f"rating must be one of {sorted(CANONICAL_RATINGS)}, got {rating!r}")
    return errs


def validate_price_target_matrix(report: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    matrix = report.get("price_target_matrix")
    if not isinstance(matrix, list):
        return ["price_target_matrix must be a list"]
    if len(matrix) != 3:
        errs.append(f"price_target_matrix must contain exactly 3 entries, got {len(matrix)}")

    seen = []
    for row in matrix:
        if not isinstance(row, dict):
            errs.append("each price_target_matrix row must be an object")
            continue
        scen = row.get("scenario")
        if scen not in SCENARIOS:
            errs.append(f"invalid scenario {scen!r}; expected one of {SCENARIOS}")
        else:
            seen.append(scen)
        ptr = row.get("price_target_range")
        if not isinstance(ptr, dict) or not all(k in ptr for k in ("low", "high")):
            errs.append("price_target_range must be an object with 'low' and 'high'")

    if sorted(seen) != sorted(SCENARIOS):
        errs.append(f"price_target_matrix must contain scenarios {SCENARIOS}, got {seen}")

    return errs


def validate_structured_catalysts(report: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    cats = report.get("structured_catalysts")
    if cats is None:
        # Optional but recommended
        return []
    if not isinstance(cats, list):
        return ["structured_catalysts must be a list if present"]

    allowed_category = {"company", "industry", "speculative"}
    allowed_direction = {"positive", "negative", "neutral"}
    allowed_source_type = {"reported", "inferred", "market_implied", "speculative"}
    allowed_time_horizon = {"short_term", "medium_term", "long_term"}
    allowed_level = {"Low", "Medium", "High"}
    allowed_priced = {"Yes", "No", "Partially"}

    for idx, cat in enumerate(cats):
        if not isinstance(cat, dict):
            errs.append(f"structured_catalysts[{idx}] must be an object")
            continue
        c = cat.get("category")
        if c not in allowed_category:
            errs.append(f"structured_catalysts[{idx}].category invalid: {c!r}")
        d = cat.get("direction")
        if d not in allowed_direction:
            errs.append(f"structured_catalysts[{idx}].direction invalid: {d!r}")
        st = cat.get("source_type")
        if st not in allowed_source_type:
            errs.append(f"structured_catalysts[{idx}].source_type invalid: {st!r}")
        th = cat.get("time_horizon")
        if th not in allowed_time_horizon:
            errs.append(f"structured_catalysts[{idx}].time_horizon invalid: {th!r}")
        conf = cat.get("confidence")
        if conf not in allowed_level:
            errs.append(f"structured_catalysts[{idx}].confidence invalid: {conf!r}")
        imp = cat.get("impact_level")
        if imp not in allowed_level:
            errs.append(f"structured_catalysts[{idx}].impact_level invalid: {imp!r}")
        priced = cat.get("is_already_priced")
        if priced not in allowed_priced:
            errs.append(f"structured_catalysts[{idx}].is_already_priced invalid: {priced!r}")
        if not isinstance(cat.get("description"), str):
            errs.append(f"structured_catalysts[{idx}].description must be string")
        if not isinstance(cat.get("monitoring_trigger"), str):
            errs.append(f"structured_catalysts[{idx}].monitoring_trigger must be string")
        if not isinstance(cat.get("evidence_summary"), str):
            errs.append(f"structured_catalysts[{idx}].evidence_summary must be string")
    return errs


def validate_signals(report: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    sigs = report.get("signals")
    if sigs is None:
        return []
    if not isinstance(sigs, list):
        return ["signals must be a list if present"]

    allowed_types = {
        "growth",
        "profitability",
        "valuation",
        "balance_sheet",
        "momentum",
        "industry_catalyst",
        "company_catalyst",
        "speculative_catalyst",
        "risk",
        "management_execution",
    }
    allowed_strength = {"strong_positive", "positive", "cautious", "strong_negative"}

    for idx, sig in enumerate(sigs):
        if not isinstance(sig, dict):
            errs.append(f"signals[{idx}] must be an object")
            continue
        t = sig.get("type")
        if t not in allowed_types:
            errs.append(f"signals[{idx}].type invalid: {t!r}")
        s = sig.get("strength")
        if s not in allowed_strength:
            errs.append(f"signals[{idx}].strength invalid: {s!r}")
        if not isinstance(sig.get("reason"), str):
            errs.append(f"signals[{idx}].reason must be string")
    return errs


def validate_report(report: Dict[str, Any]) -> List[str]:
    """Run all validators and return a flat list of human-readable errors."""
    if not isinstance(report, dict):
        return ["report must be a JSON object at top level"]
    errs: List[str] = []
    errs.extend(validate_rating(report))
    errs.extend(validate_price_target_matrix(report))
    errs.extend(validate_structured_catalysts(report))
    errs.extend(validate_signals(report))
    return errs

