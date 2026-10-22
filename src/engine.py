"""
src/engine.py
-------------
4-Stage Multi-Criteria Decision Pipeline for Shipment Route Optimisation.

Pipeline stages (executed in order):
  Stage 1  –  Hard constraints + structured infeasibility fallback
  Stage 2  –  Pareto dominance filter
  Stage 3  –  Risk-adjusted expected-value metrics
  Stage 4  –  Min-max normalisation + profile-weighted penalty scoring

Public API
~~~~~~~~~~
    result = run_pipeline(shipment: dict) -> PipelineResult

All intermediate state is stored on the returned PipelineResult dataclass so
that explain.py and evaluate.py can inspect every decision without re-running
the pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DELAY_BUFFER_DAYS    = 1.0      # hard deadline safety buffer
ALPHA                = 0.0005   # 0.05 % of cargo value per delayed day
BETA                 = 0.20     # cap: 20 % of base freight rate
LOW_CONFIDENCE_MARGIN = 0.05    # penalty-score proximity threshold

PROFILE_WEIGHTS: dict[str, dict[str, float]] = {
    "urgent":         {"cost": 0.15, "time": 0.50, "risk": 0.35},
    "standard":       {"cost": 0.35, "time": 0.30, "risk": 0.35},
    "cost_sensitive": {"cost": 0.60, "time": 0.20, "risk": 0.20},
}


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------
@dataclass
class RouteMetrics:
    """All computed metrics for a single candidate route."""
    route_id:                str
    mode:                    str
    archetype:               str
    base_freight_cost_usd:   float
    base_transit_days:       float
    delay_probability:       float
    delay_duration_days:     float
    congestion_score:        float
    geopolitical_risk_score: float

    # Stage-3 derived
    expected_delay_days:    float = 0.0
    expected_transit_days:  float = 0.0
    delay_cost_usd:         float = 0.0
    expected_landed_cost:   float = 0.0
    risk_index:             float = 0.0

    # Stage-4 derived
    norm_cost:     float = 0.0
    norm_time:     float = 0.0
    norm_risk:     float = 0.0
    cost_penalty:  float = 0.0
    time_penalty:  float = 0.0
    risk_penalty:  float = 0.0
    penalty_score: float = 0.0

    # Structural / descriptive (optional; empty by default)
    hubs: list[str]       = field(default_factory=list)
    legs: list[dict]      = field(default_factory=list)


@dataclass
class FallbackResult:
    """Structured fallback record when no fully feasible route exists."""
    reason:               str        # "budget_short" | "deadline_short"
    recommended_route_id: str
    extend_deadline_days: float = 0.0
    raise_budget_usd:     float = 0.0


@dataclass
class PipelineResult:
    """Complete artefact returned by run_pipeline()."""
    shipment_id:        str
    profile:            str
    cargo_value_usd:    float
    max_budget_usd:     float
    deadline_days:      float
    scoring_mode:       str = "minmax"

    # Shipment-level geography
    origin:      str = ""
    destination: str = ""

    # All routes with computed metrics
    all_metrics: list[RouteMetrics] = field(default_factory=list)

    # Stage-1
    feasible_ids:       list[str]         = field(default_factory=list)
    disqualified:       dict[str, str]    = field(default_factory=dict)   # route_id → reason

    # Stage-2
    pareto_ids:         list[str]         = field(default_factory=list)
    dominated_ids:      dict[str, str]    = field(default_factory=dict)   # route_id → reason

    # Stage-4
    scored_routes:      list[RouteMetrics] = field(default_factory=list)  # sorted asc by penalty
    winner_id:          Optional[str]      = None
    low_confidence:     bool               = False

    # Fallback (populated only when no feasible route)
    fallback:           Optional[FallbackResult] = None

    # ── Convenience look-ups ──────────────────────────────────────────────
    def get_metric(self, route_id: str) -> Optional[RouteMetrics]:
        for m in self.all_metrics:
            if m.route_id == route_id:
                return m
        return None

    @property
    def winner(self) -> Optional[RouteMetrics]:
        return self.get_metric(self.winner_id) if self.winner_id else None

    @property
    def cheapest(self) -> Optional[RouteMetrics]:
        if not self.scored_routes:
            return None
        return min(self.scored_routes, key=lambda m: m.expected_landed_cost)

    @property
    def fastest(self) -> Optional[RouteMetrics]:
        if not self.scored_routes:
            return None
        return min(self.scored_routes, key=lambda m: m.expected_transit_days)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _build_metrics(route: dict[str, Any], cargo_value: float) -> RouteMetrics:
    """Stage 3: compute all risk-adjusted expected-value metrics."""
    p_d   = route["delay_probability"]
    d     = route["delay_duration_days"]
    C_r   = route["base_freight_cost_usd"]
    T_b   = route["base_transit_days"]
    cong  = route["congestion_score"]
    geo   = route["geopolitical_risk_score"]

    E_D   = p_d * d
    E_T   = T_b + E_D
    C_delay = min(E_D * ALPHA * cargo_value, BETA * C_r)
    E_C   = C_r + C_delay
    risk  = (cong + geo) / 2.0

    return RouteMetrics(
        route_id                 = route["route_id"],
        mode                     = route["mode"],
        archetype                = route["archetype"],
        base_freight_cost_usd    = C_r,
        base_transit_days        = T_b,
        delay_probability        = p_d,
        delay_duration_days      = d,
        congestion_score         = cong,
        geopolitical_risk_score  = geo,
        expected_delay_days      = round(E_D, 4),
        expected_transit_days    = round(E_T, 4),
        delay_cost_usd           = round(C_delay, 4),
        expected_landed_cost     = round(E_C, 4),
        risk_index               = round(risk, 4),
        hubs                     = list(route.get("hubs", [])),
        legs                     = list(route.get("legs", [])),
    )


def _is_dominated(candidate: RouteMetrics, others: list[RouteMetrics]) -> tuple[bool, str]:
    """
    Return (True, reason_string) if *candidate* is Pareto-dominated by any
    route in *others*; otherwise (False, "").
    """
    for other in others:
        if other.route_id == candidate.route_id:
            continue
        # Dominator must be ≤ in all objectives
        cost_le = other.expected_landed_cost  <= candidate.expected_landed_cost
        time_le = other.expected_transit_days <= candidate.expected_transit_days
        risk_le = other.risk_index            <= candidate.risk_index
        # … and strictly better in at least one
        cost_lt = other.expected_landed_cost  <  candidate.expected_landed_cost
        time_lt = other.expected_transit_days <  candidate.expected_transit_days
        risk_lt = other.risk_index            <  candidate.risk_index

        if cost_le and time_le and risk_le and (cost_lt or time_lt or risk_lt):
            better_dims: list[str] = []
            if cost_lt:
                better_dims.append(
                    f"cost ${other.expected_landed_cost:.2f} < ${candidate.expected_landed_cost:.2f}"
                )
            if time_lt:
                better_dims.append(
                    f"time {other.expected_transit_days:.2f}d < {candidate.expected_transit_days:.2f}d"
                )
            if risk_lt:
                better_dims.append(
                    f"risk {other.risk_index:.4f} < {candidate.risk_index:.4f}"
                )
            reason = (
                f"dominated by {other.route_id} "
                f"({'; '.join(better_dims)})"
            )
            return True, reason

    return False, ""


def _minmax_normalize(values: list[float]) -> list[float]:
    """
    Min-max normalise a list of floats onto [0, 1].
    Zero-variance → all values become 0.0 (no ZeroDivisionError).
    """
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi, rel_tol=1e-12, abs_tol=1e-12):
        return [0.0] * len(values)
    span = hi - lo
    return [(v - lo) / span for v in values]


# ---------------------------------------------------------------------------
# Public API: run_pipeline
# ---------------------------------------------------------------------------
def run_pipeline(
    shipment: dict[str, Any],
    scoring:  str = "minmax",
) -> PipelineResult:
    """
    Execute the full 4-stage pipeline for a single shipment dict.

    Parameters
    ----------
    shipment : dict
        Must contain keys: shipment_id, profile, cargo_value_usd,
        max_budget_usd, deadline_days, routes (list of route dicts).
    scoring : str, default "minmax"
        Stage 4 scoring mode: "minmax" (relative min-max normalisation) or
        "reference" (absolute normalization against shipment budget and deadline).

    Returns
    -------
    PipelineResult
        Fully populated result object with all intermediate state.
    """
    sid          = shipment["shipment_id"]
    profile      = shipment["profile"].lower()
    cargo_value  = shipment["cargo_value_usd"]
    max_budget   = shipment["max_budget_usd"]
    deadline     = shipment["deadline_days"]
    routes       = shipment["routes"]

    result = PipelineResult(
        shipment_id     = sid,
        profile         = profile,
        cargo_value_usd = cargo_value,
        max_budget_usd  = max_budget,
        deadline_days   = deadline,
        scoring_mode    = scoring,
        origin          = shipment.get("origin", ""),
        destination     = shipment.get("destination", ""),
    )

    # ── Stage 3 (metrics computed first; needed by all stages) ────────────
    all_metrics: list[RouteMetrics] = [_build_metrics(r, cargo_value) for r in routes]
    result.all_metrics = all_metrics

    # ── Stage 1: Hard-constraint filtering ────────────────────────────────
    feasible:      list[RouteMetrics] = []
    disqualified:  dict[str, str]     = {}

    for m in all_metrics:
        reasons: list[str] = []
        if m.base_freight_cost_usd > max_budget:
            reasons.append(
                f"cost ${m.base_freight_cost_usd:,.2f} > budget ${max_budget:,.2f}"
            )
        if m.expected_transit_days + DELAY_BUFFER_DAYS > deadline:
            reasons.append(
                f"expected time + buffer "
                f"{m.expected_transit_days + DELAY_BUFFER_DAYS:.2f}d > deadline {deadline:.1f}d"
            )
        if reasons:
            disqualified[m.route_id] = "; ".join(reasons)
        else:
            feasible.append(m)

    result.feasible_ids  = [m.route_id for m in feasible]
    result.disqualified  = disqualified

    # ── Structured fallback when zero feasible routes ─────────────────────
    if not feasible:
        # Try: any routes within budget?
        budget_ok = [m for m in all_metrics if m.base_freight_cost_usd <= max_budget]
        if budget_ok:
            # Pick fastest among budget-satisfying routes
            best   = min(budget_ok, key=lambda m: m.expected_transit_days)
            reason = "deadline_short"
        else:
            # Zero routes within budget; pick overall cheapest
            best   = min(all_metrics, key=lambda m: m.base_freight_cost_usd)
            reason = "budget_short"

        # Always compute BOTH gap fields for the chosen fallback route
        raise_by = round(max(0.0, best.base_freight_cost_usd - max_budget), 2)
        extend   = round(max(0.0, best.expected_transit_days + DELAY_BUFFER_DAYS - deadline), 2)

        result.fallback = FallbackResult(
            reason               = reason,
            recommended_route_id = best.route_id,
            raise_budget_usd     = raise_by,
            extend_deadline_days = extend,
        )
        # No further pipeline stages possible
        return result

    # ── Stage 2: Pareto dominance filter ──────────────────────────────────
    pareto:    list[RouteMetrics] = []
    dominated: dict[str, str]    = {}

    for m in feasible:
        is_dom, reason = _is_dominated(m, feasible)
        if is_dom:
            dominated[m.route_id] = reason
        else:
            pareto.append(m)

    # Safety: if Pareto filter removes everything (numerical tie), keep all
    if not pareto:
        pareto = list(feasible)

    result.pareto_ids   = [m.route_id for m in pareto]
    result.dominated_ids = dominated

    # ── Stage 4: Normalisation & profile-weighted scoring ─────────────────
    weights = PROFILE_WEIGHTS.get(profile, PROFILE_WEIGHTS["standard"])

    if scoring == "minmax":
        costs  = [m.expected_landed_cost  for m in pareto]
        times  = [m.expected_transit_days for m in pareto]
        risks  = [m.risk_index            for m in pareto]

        norm_costs = _minmax_normalize(costs)
        norm_times = _minmax_normalize(times)
        norm_risks = _minmax_normalize(risks)

        for m, nc, nt, nr in zip(pareto, norm_costs, norm_times, norm_risks):
            m.norm_cost     = round(nc, 6)
            m.norm_time     = round(nt, 6)
            m.norm_risk     = round(nr, 6)
            m.cost_penalty  = m.norm_cost
            m.time_penalty  = m.norm_time
            m.risk_penalty  = m.norm_risk
            m.penalty_score = round(
                weights["cost"] * nc + weights["time"] * nt + weights["risk"] * nr,
                6,
            )

    elif scoring == "reference":
        for m in pareto:
            cost_pen = m.expected_landed_cost / max_budget
            time_pen = m.expected_transit_days / deadline
            risk_pen = m.risk_index

            m.cost_penalty  = round(cost_pen, 6)
            m.time_penalty  = round(time_pen, 6)
            m.risk_penalty  = round(risk_pen, 6)
            m.norm_cost     = m.cost_penalty
            m.norm_time     = m.time_penalty
            m.norm_risk     = m.risk_penalty
            m.penalty_score = round(
                weights["cost"] * cost_pen + weights["time"] * time_pen + weights["risk"] * risk_pen,
                6,
            )
    else:
        raise ValueError(
            f"Unknown scoring mode: {scoring!r}. Expected 'minmax' or 'reference'."
        )

    # Sort ascending by penalty score (lower = better)
    scored = sorted(pareto, key=lambda m: m.penalty_score)
    result.scored_routes = scored
    result.winner_id     = scored[0].route_id

    # Low-confidence flag
    if len(scored) >= 2:
        margin = scored[1].penalty_score - scored[0].penalty_score
        result.low_confidence = margin < LOW_CONFIDENCE_MARGIN

    return result
