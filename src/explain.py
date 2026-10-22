"""
src/explain.py
--------------
Deterministic, traceable explanation engine for logistics route decisions.

Generates operator-ready plain-text reports covering:
  • Recommended route with penalty score and profile weights
  • Delta analysis vs. cheapest feasible baseline (cost premium → risk/time saved)
  • Delta analysis vs. fastest feasible baseline (time conceded → freight saved)
  • Disqualified route summary with explicit per-route reasons
  • Dominated route summary
  • Low-confidence warning when top-2 score margin < 0.05
  • Fallback advisory when no fully feasible route exists

Public API
~~~~~~~~~~
    text = build_explanation(result: PipelineResult) -> str
    print_explanation(result: PipelineResult) -> None
"""

from __future__ import annotations

from typing import Optional

from .engine import PipelineResult, RouteMetrics

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
_SEP_MAJOR = "=" * 72
_SEP_MINOR = "-" * 72
_PROFILE_LABELS = {
    "urgent":         "Urgent        (w_cost=0.15, w_time=0.50, w_risk=0.35)",
    "standard":       "Standard      (w_cost=0.35, w_time=0.30, w_risk=0.35)",
    "cost_sensitive": "Cost-Sensitive (w_cost=0.60, w_time=0.20, w_risk=0.20)",
}


def _fmt_legs(legs: list) -> str:
    """Return e.g. 'Shanghai --[Sea]--> Port Klang --[Air]--> Rotterdam'."""
    if not legs:
        return ""
    parts = [legs[0]["from"]]
    for leg in legs:
        parts.append(f"--[{leg['mode']}]-->")
        parts.append(leg["to"])
    return " ".join(parts)


def _fmt_route_summary(m: RouteMetrics, scoring: str = "minmax") -> str:
    if scoring == "reference":
        pen_line = f"    Penalties(c/t/r): {m.cost_penalty:.4f} / {m.time_penalty:.4f} / {m.risk_penalty:.4f}  (c=E[C]/budget, t=E[T]/deadline, r=risk)\n"
    else:
        pen_line = f"    Norm (c/t/r)  : {m.norm_cost:.4f} / {m.norm_time:.4f} / {m.norm_risk:.4f}\n"

    base = (
        f"  Route {m.route_id} [{m.mode} | {m.archetype}]\n"
        f"    Base Freight  : ${m.base_freight_cost_usd:>10,.2f}\n"
        f"    Expected Cost : ${m.expected_landed_cost:>10,.2f}  "
        f"(delay cost ${m.delay_cost_usd:,.2f})\n"
        f"    Expected Time : {m.expected_transit_days:>7.2f} days  "
        f"(base {m.base_transit_days:.1f}d + E[delay] {m.expected_delay_days:.2f}d)\n"
        f"    Risk Index    : {m.risk_index:>7.4f}  "
        f"(congestion {m.congestion_score:.4f}, geo {m.geopolitical_risk_score:.4f})\n"
        f"    Delay Prob    : {m.delay_probability*100:.1f}%  "
        f"\u2192 {m.delay_duration_days:.1f}d if delayed\n"
        f"{pen_line}"
        f"    Penalty Score : {m.penalty_score:.6f}"
    )
    extras: list[str] = []
    if getattr(m, "hubs", None):
        extras.append(f"    Hubs          : {', '.join(m.hubs)}")
    if getattr(m, "legs", None):
        extras.append(f"    Legs          : {_fmt_legs(m.legs)}")
    if extras:
        return base + "\n" + "\n".join(extras)
    return base



def _delta_block(
    label:     str,
    winner:    RouteMetrics,
    baseline:  Optional[RouteMetrics],
) -> str:
    """Return a delta-analysis block comparing winner vs. a baseline route."""
    if baseline is None or baseline.route_id == winner.route_id:
        return f"  ✓ Winner IS the {label} route – no delta."

    cost_delta = winner.expected_landed_cost  - baseline.expected_landed_cost
    time_delta = winner.expected_transit_days - baseline.expected_transit_days
    risk_delta = winner.risk_index            - baseline.risk_index

    lines = [f"  Versus {label} ({baseline.route_id}):"]

    if label == "Cheapest":
        # Winner costs more; show what was gained in time/risk
        lines.append(
            f"    Cost premium  : +${cost_delta:,.2f}  "
            f"({cost_delta/baseline.expected_landed_cost*100:+.1f}% over cheapest)"
        )
        if time_delta < 0:
            lines.append(f"    Time saved     : {abs(time_delta):.2f} days faster")
        else:
            lines.append(f"    Time trade-off : {time_delta:+.2f} days slower")
        if risk_delta < 0:
            lines.append(f"    Risk reduced   : {abs(risk_delta):.4f} lower risk index")
        else:
            lines.append(f"    Risk trade-off : {risk_delta:+.4f} higher risk index")

    else:  # Fastest
        # Winner is slower; show what was saved in cost/risk
        lines.append(
            f"    Time conceded  : {time_delta:+.2f} days slower than fastest"
        )
        if cost_delta < 0:
            lines.append(
                f"    Cost saved     : ${abs(cost_delta):,.2f}  "
                f"({abs(cost_delta)/baseline.expected_landed_cost*100:.1f}% below fastest)"
            )
        else:
            lines.append(f"    Cost trade-off : +${cost_delta:,.2f} more expensive")
        if risk_delta < 0:
            lines.append(f"    Risk reduced   : {abs(risk_delta):.4f} lower risk index")
        else:
            lines.append(f"    Risk trade-off : {risk_delta:+.4f} higher risk index")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_explanation(result: PipelineResult) -> str:
    """
    Build a deterministic, operator-ready explanation string for *result*.

    Returns
    -------
    str
        Multi-line text report.
    """
    lines: list[str] = []
    add = lines.append

    add(_SEP_MAJOR)
    add(f"  LOGISTICS ROUTE DECISION REPORT")
    add(f"  Shipment : {result.shipment_id}")
    if result.origin or result.destination:
        add(f"  Route    : {result.origin} -> {result.destination}")
    add(f"  Profile  : {_PROFILE_LABELS.get(result.profile, result.profile)}")
    add(f"  Scoring  : {getattr(result, 'scoring_mode', 'minmax').upper()} mode")
    add(f"  Budget   : ${result.max_budget_usd:,.2f}   Deadline: {result.deadline_days:.1f} days")
    add(f"  Cargo $  : ${result.cargo_value_usd:,.2f}")
    add(_SEP_MAJOR)

    # ── FALLBACK PATH ──────────────────────────────────────────────────────
    if result.fallback:
        fb = result.fallback
        add("")
        add("  ⚠  INFEASIBILITY FALLBACK TRIGGERED")
        add(_SEP_MINOR)

        # Dual-gap: check each gap independently (both are always computed)
        need_budget   = fb.raise_budget_usd    > 0
        need_deadline = fb.extend_deadline_days > 0

        if need_budget and need_deadline:
            add(
                f"  Routes fail on BOTH budget and deadline.\n"
                f"  ADVISORY: Raise budget by at least ${fb.raise_budget_usd:,.2f} "
                f"AND extend deadline by at least {fb.extend_deadline_days:.2f} days "
                f"to make Route {fb.recommended_route_id} feasible."
            )
        elif need_budget:
            add(
                f"  No route fits within the budget of ${result.max_budget_usd:,.2f}.\n"
                f"  ADVISORY: Raise budget by at least ${fb.raise_budget_usd:,.2f} "
                f"to unlock Route {fb.recommended_route_id} (cheapest available)."
            )
        else:
            add(
                f"  Budget is sufficient but all routes miss the "
                f"{result.deadline_days:.1f}-day deadline (incl. {1.0:.1f}d buffer).\n"
                f"  ADVISORY: Extend deadline by at least {fb.extend_deadline_days:.2f} days "
                f"to unlock Route {fb.recommended_route_id} (fastest within budget)."
            )

        add("")
        add("  Fallback Recommended Route:")
        fb_metric = result.get_metric(fb.recommended_route_id)
        if fb_metric:
            add(_fmt_route_summary(fb_metric, getattr(result, "scoring_mode", "minmax")))
        add("")
        add("  [No penalty scoring performed – feasibility constraints not met]")
        add("")
        _append_disqualified_section(result, lines)
        add(_SEP_MAJOR)
        return "\n".join(lines)

    # ── NORMAL PATH: feasible winner ──────────────────────────────────────
    winner = result.winner
    if winner is None:
        add("  ERROR: Pipeline produced no winner. Please inspect input data.")
        add(_SEP_MAJOR)
        return "\n".join(lines)

    add("")
    add("  ✅  RECOMMENDATION")
    add(_SEP_MINOR)
    add(_fmt_route_summary(winner, getattr(result, "scoring_mode", "minmax")))
    add("")

    # ── Low-confidence alert ───────────────────────────────────────────────
    if result.low_confidence and len(result.scored_routes) >= 2:
        runner_up = result.scored_routes[1]
        margin    = runner_up.penalty_score - winner.penalty_score
        add("  ⚠  LOW-CONFIDENCE ALERT")
        add(
            f"     Top-2 penalty scores are very close: "
            f"{winner.penalty_score:.6f} vs {runner_up.penalty_score:.6f} "
            f"(margin {margin:.6f} < 0.05).\n"
            f"     Operator review recommended before committing to {winner.route_id}."
        )
        add("")

    # ── Delta analysis ─────────────────────────────────────────────────────
    add("  TRADE-OFF ANALYSIS")
    add(_SEP_MINOR)

    # Cheapest baseline from scored (Pareto-surviving) routes
    cheapest = result.cheapest
    fastest  = result.fastest

    add(_delta_block("Cheapest", winner, cheapest))
    add("")
    add(_delta_block("Fastest",  winner, fastest))
    add("")

    # ── All scored candidates ──────────────────────────────────────────────
    add("  ALL PARETO-SURVIVING CANDIDATES (ranked by penalty score)")
    add(_SEP_MINOR)
    is_ref = getattr(result, "scoring_mode", "minmax") == "reference"
    for rank, m in enumerate(result.scored_routes, start=1):
        marker = "★" if m.route_id == winner.route_id else " "
        if is_ref:
            add(
                f"  {marker} #{rank:02d}  {m.route_id:20s}  "
                f"score={m.penalty_score:.6f}  "
                f"penalties(c/t/r)={m.cost_penalty:.4f}/{m.time_penalty:.4f}/{m.risk_penalty:.4f}  "
                f"E[C]=${m.expected_landed_cost:>9,.2f}  "
                f"E[T]={m.expected_transit_days:>6.2f}d  "
                f"Risk={m.risk_index:.4f}"
            )
        else:
            add(
                f"  {marker} #{rank:02d}  {m.route_id:20s}  "
                f"score={m.penalty_score:.6f}  "
                f"E[C]=${m.expected_landed_cost:>9,.2f}  "
                f"E[T]={m.expected_transit_days:>6.2f}d  "
                f"Risk={m.risk_index:.4f}"
            )
    add("")

    # ── Dominated routes ──────────────────────────────────────────────────
    if result.dominated_ids:
        add("  DOMINATED ROUTES (filtered before scoring)")
        add(_SEP_MINOR)
        for rid, reason in result.dominated_ids.items():
            add(f"  ✗ {rid:20s}  → {reason}")
        add("")

    # ── Disqualified routes ───────────────────────────────────────────────
    _append_disqualified_section(result, lines)
    add(_SEP_MAJOR)

    return "\n".join(lines)


def _append_disqualified_section(result: PipelineResult, lines: list[str]) -> None:
    """Append the disqualified-routes block to *lines* in-place."""
    if not result.disqualified:
        return
    lines.append("  DISQUALIFIED ROUTES (hard-constraint violations)")
    lines.append(_SEP_MINOR)
    for rid, reason in result.disqualified.items():
        lines.append(f"  ✗ {rid:20s}  → {reason}")
    lines.append("")


def print_explanation(result: PipelineResult) -> None:
    """Print the explanation report to stdout."""
    text = build_explanation(result)
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding))

