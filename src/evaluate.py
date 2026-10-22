"""
src/evaluate.py
---------------
Benchmarking & comparative evaluation for the Aiqon logistics optimisation pipeline.

Three strategies, all operating on the SAME Stage-1 feasible set per shipment:
  * Cheapest-First   -- lowest E[C] among feasible routes
  * Fastest-First    -- lowest E[T] among feasible routes
  * Proposed Engine  -- Pareto filter + profile-weighted scoring on feasible set

CRN simulation (1,000 draws per shipment):
  For each shipment and trial k one uniform U_k is drawn (shared seed).
  A route is delayed in trial k iff U_k < p_d (route's own probability); if
  delayed it adds its own delay_duration_days.  Miss = arrival > actual deadline
  (the 1-day buffer is a HARD GATE, NOT added here).

Tables printed:
  Table A -- feasible shipments only
  Table B -- all 35 shipments

Additional analyses:
  * Frequency the engine differs from Cheapest-First and Fastest-First
  * Alpha sensitivity: re-run with alpha=0.0005 and alpha=0.001; count changes
  * Cost-weight sweep on every shipment with >=2 Pareto survivors under its own
    UNRELAXED constraints; print flip counts and flip points; save one plot from
    a shipment where a flip actually occurs (or report none if none flip).

Usage
-----
    python -m src.evaluate                          # uses data/shipments.json
    python -m src.evaluate --data path/to/file.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .engine import (
    RouteMetrics,
    FallbackResult,
    PipelineResult,
    _build_metrics,
    _is_dominated,
    _minmax_normalize,
    run_pipeline,
    PROFILE_WEIGHTS,
    DELAY_BUFFER_DAYS,
    ALPHA as DEFAULT_ALPHA,
    BETA,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CRN_DRAWS   = 1_000
CRN_SEED    = 2024
SWEEP_STEPS = 50

# ---------------------------------------------------------------------------
# Archetype cosmetics (for plot)
# ---------------------------------------------------------------------------
_ARCHETYPE_COLOURS: dict[str, str] = {
    "fast_expensive": "#F97316",
    "balanced":       "#3B82F6",
    "cheap_risky":    "#22C55E",
    "dominated":      "#A855F7",
}
_ARCHETYPE_LABELS: dict[str, str] = {
    "fast_expensive": "Fast & Expensive",
    "balanced":       "Balanced",
    "cheap_risky":    "Cheap & Risky",
    "dominated":      "Dominated",
}


# ---------------------------------------------------------------------------
# Shared Stage-1 feasibility helper (reuses engine formula)
# ---------------------------------------------------------------------------
def _get_feasible_metrics(
    shipment:  dict[str, Any],
    alpha:     float = DEFAULT_ALPHA,
) -> list[RouteMetrics]:
    """
    Return Stage-1 feasible RouteMetrics for *shipment* using *alpha*.
    Mirrors engine Stage 1 exactly: base_freight_cost_usd <= max_budget AND
    E[T] + DELAY_BUFFER_DAYS <= deadline_days.
    """
    cargo_value = shipment["cargo_value_usd"]
    max_budget  = shipment["max_budget_usd"]
    deadline    = shipment["deadline_days"]

    feasible: list[RouteMetrics] = []
    for r in shipment["routes"]:
        m = _build_metrics_with_alpha(r, cargo_value, alpha)
        cost_ok = m.base_freight_cost_usd <= max_budget
        time_ok = m.expected_transit_days + DELAY_BUFFER_DAYS <= deadline
        if cost_ok and time_ok:
            feasible.append(m)
    return feasible


def _build_metrics_with_alpha(
    route:       dict[str, Any],
    cargo_value: float,
    alpha:       float,
) -> RouteMetrics:
    """_build_metrics but with a parametric alpha (avoids monkey-patching engine)."""
    p_d  = route["delay_probability"]
    d    = route["delay_duration_days"]
    C_r  = route["base_freight_cost_usd"]
    T_b  = route["base_transit_days"]
    cong = route["congestion_score"]
    geo  = route["geopolitical_risk_score"]

    E_D     = p_d * d
    E_T     = T_b + E_D
    C_delay = min(E_D * alpha * cargo_value, BETA * C_r)
    E_C     = C_r + C_delay
    risk    = (cong + geo) / 2.0

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
    )


def _pareto_filter(metrics: list[RouteMetrics]) -> list[RouteMetrics]:
    """Return Pareto non-dominated routes from *metrics* (mirrors engine Stage 2)."""
    pareto: list[RouteMetrics] = []
    for m in metrics:
        is_dom, _ = _is_dominated(m, metrics)
        if not is_dom:
            pareto.append(m)
    return pareto if pareto else list(metrics)


def _engine_pick(
    shipment: dict[str, Any],
    feasible: list[RouteMetrics],
    scoring:  str = "minmax",
) -> RouteMetrics | None:
    """
    Engine pick from the already-computed feasible set: Pareto filter + scoring.
    Supports scoring='minmax' and scoring='reference'.
    Falls back to cheapest-overall if feasible is empty.
    """
    if not feasible:
        return None
    pareto  = _pareto_filter(feasible)
    profile = shipment["profile"].lower()
    weights = PROFILE_WEIGHTS.get(profile, PROFILE_WEIGHTS["standard"])

    if scoring == "minmax":
        costs = [m.expected_landed_cost  for m in pareto]
        times = [m.expected_transit_days for m in pareto]
        risks = [m.risk_index            for m in pareto]

        nc = _minmax_normalize(costs)
        nt = _minmax_normalize(times)
        nr = _minmax_normalize(risks)

        best_score = float("inf")
        best_m     = pareto[0]
        for m, c_, t_, r_ in zip(pareto, nc, nt, nr):
            s = weights["cost"] * c_ + weights["time"] * t_ + weights["risk"] * r_
            if s < best_score:
                best_score = s
                best_m     = m
        return best_m

    elif scoring == "reference":
        max_budget = shipment["max_budget_usd"]
        deadline   = shipment["deadline_days"]
        best_score = float("inf")
        best_m     = pareto[0]
        for m in pareto:
            cost_pen = m.expected_landed_cost / max_budget
            time_pen = m.expected_transit_days / deadline
            risk_pen = m.risk_index
            s = weights["cost"] * cost_pen + weights["time"] * time_pen + weights["risk"] * risk_pen
            if s < best_score:
                best_score = s
                best_m     = m
        return best_m
    else:
        raise ValueError(f"Unknown scoring mode: {scoring!r}")


def _fallback_pick(shipment: dict[str, Any]) -> RouteMetrics:
    """
    Shared fallback (mirrors engine): fastest within budget, else overall cheapest.
    Used by ALL strategies when no route is feasible.
    """
    cargo_value = shipment["cargo_value_usd"]
    max_budget  = shipment["max_budget_usd"]
    all_m       = [_build_metrics(r, cargo_value) for r in shipment["routes"]]

    budget_ok = [m for m in all_m if m.base_freight_cost_usd <= max_budget]
    if budget_ok:
        return min(budget_ok, key=lambda m: m.expected_transit_days)
    return min(all_m, key=lambda m: m.base_freight_cost_usd)


# ---------------------------------------------------------------------------
# CRN simulation
# ---------------------------------------------------------------------------
def _crn_miss_rate(
    metrics:  RouteMetrics,
    deadline: float,
    U:        np.ndarray,      # pre-drawn (CRN_DRAWS,) array in [0,1)
) -> float:
    """
    Deadline-miss rate for one route using pre-drawn CRN uniforms.
    Miss = actual arrival > deadline  (no 1-day buffer here).
    """
    delayed  = U < metrics.delay_probability
    arrivals = metrics.base_transit_days + delayed.astype(float) * metrics.delay_duration_days
    return float((arrivals > deadline).mean())


# ---------------------------------------------------------------------------
# Per-strategy selection
# ---------------------------------------------------------------------------
def _pick_for_strategy(
    strategy: str,
    shipment: dict[str, Any],
    feasible: list[RouteMetrics],
) -> RouteMetrics:
    """
    Select one route for *strategy* from the Stage-1 feasible set.
    All strategies use the same *feasible* set and the same *_fallback_pick*
    when feasible is empty.
    """
    if not feasible:
        return _fallback_pick(shipment)

    if strategy == "Cheapest-First":
        return min(feasible, key=lambda m: m.expected_landed_cost)
    elif strategy == "Fastest-First":
        return min(feasible, key=lambda m: m.expected_transit_days)
    elif strategy in ("Engine (minmax)", "Proposed Engine"):
        result = _engine_pick(shipment, feasible, scoring="minmax")
        return result if result is not None else _fallback_pick(shipment)
    elif strategy == "Engine (reference)":
        result = _engine_pick(shipment, feasible, scoring="reference")
        return result if result is not None else _fallback_pick(shipment)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Dataset-level categorisation
# ---------------------------------------------------------------------------
def _categorise(shipments: list[dict[str, Any]]) -> dict[str, Any]:
    """
    For each shipment determine:
      n_pareto = number of Pareto survivors after Stage 1+2
      feasible_count = number of Stage-1 feasible routes
    Returns dict with shipment_id -> info, and summary counts.
    """
    ge2: list[str] = []
    eq1: list[str] = []
    eq0: list[str] = []

    per_shp: dict[str, dict] = {}
    for shp in shipments:
        feasible = _get_feasible_metrics(shp)
        if not feasible:
            eq0.append(shp["shipment_id"])
            n_p = 0
        else:
            pareto = _pareto_filter(feasible)
            n_p    = len(pareto)
            if n_p >= 2:
                ge2.append(shp["shipment_id"])
            else:
                eq1.append(shp["shipment_id"])

        per_shp[shp["shipment_id"]] = {
            "feasible_count": len(feasible),
            "pareto_count":   n_p,
        }

    return {"ge2": ge2, "eq1": eq1, "eq0": eq0, "per_shp": per_shp}


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def _run_benchmark(
    shipments: list[dict[str, Any]],
    cat:       dict[str, Any],
    alpha:     float = DEFAULT_ALPHA,
) -> tuple[dict[str, list[dict]], list[np.ndarray]]:
    """
    Run all strategies and collect per-shipment results.

    Returns
    -------
    (results_dict, crn_arrays)
    """
    rng = np.random.default_rng(CRN_SEED)

    # Pre-draw one CRN array per shipment (shared across strategies)
    crn_arrays: list[np.ndarray] = [
        rng.random(CRN_DRAWS) for _ in shipments
    ]

    strategies = [
        "Cheapest-First",
        "Fastest-First",
        "Engine (minmax)",
        "Engine (reference)",
    ]
    results: dict[str, list[dict]] = {s: [] for s in strategies}

    for shp_idx, shp in enumerate(shipments):
        sid      = shp["shipment_id"]
        deadline = shp["deadline_days"]
        U        = crn_arrays[shp_idx]

        feasible = _get_feasible_metrics(shp, alpha=alpha)

        for strat in strategies:
            pick     = _pick_for_strategy(strat, shp, feasible)
            miss     = _crn_miss_rate(pick, deadline, U)
            n_pareto = cat["per_shp"][sid]["pareto_count"]
            results[strat].append({
                "shipment_id":      sid,
                "feasible":         len(feasible) > 0,
                "n_pareto":         n_pareto,
                "route_id":         pick.route_id,
                "expected_cost":    pick.expected_landed_cost,
                "expected_time":    pick.expected_transit_days,
                "risk_index":       pick.risk_index,
                "miss_rate":        miss,
            })

    return results, crn_arrays


def _aggregate(rows: list[dict], feasible_only: bool) -> dict[str, float]:
    subset = [r for r in rows if r["feasible"]] if feasible_only else rows
    if not subset:
        return {"avg_cost": 0.0, "avg_time": 0.0, "avg_risk": 0.0, "miss_pct": 0.0, "n": 0}
    n = len(subset)
    return {
        "avg_cost": sum(r["expected_cost"] for r in subset) / n,
        "avg_time": sum(r["expected_time"] for r in subset) / n,
        "avg_risk": sum(r["risk_index"]    for r in subset) / n,
        "miss_pct": sum(r["miss_rate"]     for r in subset) / n * 100,
        "n":        n,
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------
_COL = 78

def _hdr(title: str) -> None:
    print(f"\n{'='*_COL}")
    print(f"  {title}")
    print(f"{'='*_COL}")


def _print_table(label: str, all_results: dict[str, list[dict]], feasible_only: bool) -> None:
    aggs = {s: _aggregate(rows, feasible_only) for s, rows in all_results.items()}
    n    = next(iter(aggs.values()))["n"]
    _hdr(f"{label}  (N={n})")
    print(
        f"  {'Strategy':<22}  {'Avg E[C]':>13}  {'Miss Rate':>11}  "
        f"{'Avg E[T]':>10}  {'Avg Risk':>9}"
    )
    print(f"  {'-'*22}  {'-'*13}  {'-'*11}  {'-'*10}  {'-'*9}")
    for strat, ag in aggs.items():
        print(
            f"  {strat:<22}  ${ag['avg_cost']:>12,.2f}  "
            f"{ag['miss_pct']:>10.1f}%  "
            f"{ag['avg_time']:>9.2f}d  "
            f"{ag['avg_risk']:>9.4f}"
        )


def _print_diff_analysis(
    all_results: dict[str, list[dict]],
    shipments:   list[dict[str, Any]],
) -> None:
    _hdr("Engine vs. Baselines & Decision Characteristics")
    cheap_rows = all_results["Cheapest-First"]
    fast_rows  = all_results["Fastest-First"]

    for strat_key, mode_name in [
        ("Engine (minmax)", "minmax"),
        ("Engine (reference)", "reference"),
    ]:
        engine_rows = all_results[strat_key]
        n_feas = sum(1 for r in engine_rows if r["feasible"])

        diff_vs_cheap = sum(
            1 for c, e in zip(cheap_rows, engine_rows)
            if e["feasible"] and c["route_id"] != e["route_id"]
        )
        diff_vs_fast = sum(
            1 for f, e in zip(fast_rows, engine_rows)
            if e["feasible"] and f["route_id"] != e["route_id"]
        )
        neither = sum(
            1 for c, f, e in zip(cheap_rows, fast_rows, engine_rows)
            if e["feasible"]
            and e["route_id"] != c["route_id"]
            and e["route_id"] != f["route_id"]
        )

        low_conf_sids = [
            s["shipment_id"] for s in shipments
            if run_pipeline(s, scoring=mode_name).low_confidence
        ]

        print(f"\n  --- {strat_key} ---")
        print(f"  Feasible shipments evaluated        : {n_feas}")
        print(
            f"  Engine differs from Cheapest-First  : {diff_vs_cheap} / {n_feas}  "
            f"({100*diff_vs_cheap/max(n_feas,1):.1f}%)"
        )
        print(
            f"  Engine differs from Fastest-First   : {diff_vs_fast} / {n_feas}  "
            f"({100*diff_vs_fast/max(n_feas,1):.1f}%)"
        )
        print(
            f"  Picks neither cheapest nor fastest : {neither} / {n_feas}  "
            f"({100*neither/max(n_feas,1):.1f}%)"
        )
        print(
            f"  Low-confidence shipments (margin<0.05): {len(low_conf_sids)} / {len(shipments)}  "
            f"({100*len(low_conf_sids)/len(shipments):.1f}%)  {low_conf_sids}"
        )


def _print_alpha_sensitivity(
    shipments: list[dict[str, Any]],
    cat:       dict[str, Any],
) -> None:
    _hdr("Alpha sensitivity: alpha=0.0005 vs alpha=0.001")
    res_a, _ = _run_benchmark(shipments, cat, alpha=0.0005)
    res_b, _ = _run_benchmark(shipments, cat, alpha=0.001)

    for strat in ("Engine (minmax)", "Engine (reference)"):
        engine_a = res_a[strat]
        engine_b = res_b[strat]

        changes = sum(
            1 for a, b in zip(engine_a, engine_b)
            if a["feasible"] and a["route_id"] != b["route_id"]
        )
        n_feas = sum(1 for r in engine_a if r["feasible"])
        print(
            f"  {strat:<22} : {changes} / {n_feas} recommendation changes "
            f"({100*changes/max(n_feas,1):.1f}%)"
        )


# ---------------------------------------------------------------------------
# Detailed comparison: SHP-001 under both modes
# ---------------------------------------------------------------------------
def _print_shp001_comparison(shipments: list[dict[str, Any]]) -> None:
    _hdr("SHP-001 Detailed Evaluation: Min-Max vs. Reference Scoring")
    shp = next((s for s in shipments if s["shipment_id"] == "SHP-001"), None)
    if not shp:
        print("  SHP-001 not found.")
        return

    res_mm  = run_pipeline(shp, scoring="minmax")
    res_ref = run_pipeline(shp, scoring="reference")

    print(
        f"  Shipment ID : SHP-001\n"
        f"  Origin/Dest : {shp.get('origin', 'N/A')} -> {shp.get('destination', 'N/A')}\n"
        f"  Profile     : {shp['profile']} (weights: cost=0.15, time=0.50, risk=0.35)\n"
        f"  Budget      : ${shp['max_budget_usd']:,.2f}   Deadline: {shp['deadline_days']:.1f} days\n"
    )

    for mode_name, res in [("MIN-MAX MODE", res_mm), ("REFERENCE MODE", res_ref)]:
        w = res.winner
        print(f"  [{mode_name}]")
        print(
            f"    Selected Winner   : {res.winner_id} [{w.mode} | {w.archetype}]\n"
            f"    Penalty Score     : {w.penalty_score:.6f}\n"
            f"    Expected Landed $ : ${w.expected_landed_cost:>10,.2f}\n"
            f"    Expected Transit  : {w.expected_transit_days:>7.2f} days\n"
            f"    Risk Index        : {w.risk_index:>7.4f}"
        )
        if mode_name.startswith("REFERENCE"):
            print(
                f"    Penalties (c/t/r) : {w.cost_penalty:.4f} / {w.time_penalty:.4f} / {w.risk_penalty:.4f}  "
                f"(cost=E[C]/budget, time=E[T]/deadline, risk=index)"
            )
        else:
            print(
                f"    Norm (c/t/r)      : {w.norm_cost:.4f} / {w.norm_time:.4f} / {w.norm_risk:.4f}"
            )

        if len(res.scored_routes) >= 2:
            runner = res.scored_routes[1]
            margin = runner.penalty_score - w.penalty_score
            print(f"    Runner-up         : {runner.route_id} (margin = {margin:.6f}, low_conf = {res.low_confidence})")

        print("    All Candidates Ranked:")
        for rank, m in enumerate(res.scored_routes, start=1):
            star = "*" if m.route_id == res.winner_id else " "
            if mode_name.startswith("REFERENCE"):
                pen_str = f"penalties(c/t/r)={m.cost_penalty:.4f}/{m.time_penalty:.4f}/{m.risk_penalty:.4f}"
            else:
                pen_str = f"norm(c/t/r)={m.norm_cost:.4f}/{m.norm_time:.4f}/{m.norm_risk:.4f}"
            print(
                f"      {star} #{rank:02d} {m.route_id:15s} score={m.penalty_score:.6f}  "
                f"{pen_str}  E[C]=${m.expected_landed_cost:>9,.2f}  E[T]={m.expected_transit_days:>5.2f}d  Risk={m.risk_index:.4f}"
            )
        print()


# ---------------------------------------------------------------------------
# Simulated miss rate for the 3 infeasible shipments
# ---------------------------------------------------------------------------
def _print_infeasible_simulation(
    shipments:   list[dict[str, Any]],
    cat:         dict[str, Any],
    crn_arrays:  list[np.ndarray],
) -> None:
    _hdr("Simulated Miss Rates for Infeasible Shipments (N=3)")
    infeas_sids = cat["eq0"]
    shp_map = {s["shipment_id"]: (idx, s) for idx, s in enumerate(shipments)}

    print(
        f"  {'Shipment':<10}  {'Reason':<16}  {'Fallback Pick':<15}  "
        f"{'Budget':>11}  {'Deadline':>9}  {'E[C]':>11}  {'E[T]':>8}  {'Miss Rate':>10}"
    )
    print(f"  {'-'*10}  {'-'*16}  {'-'*15}  {'-'*11}  {'-'*9}  {'-'*11}  {'-'*8}  {'-'*10}")

    for sid in infeas_sids:
        idx, shp = shp_map[sid]
        U = crn_arrays[idx]
        res = run_pipeline(shp)
        fb = res.fallback
        fb_metric = res.get_metric(fb.recommended_route_id)
        miss = _crn_miss_rate(fb_metric, shp["deadline_days"], U)

        print(
            f"  {sid:<10}  {fb.reason:<16}  {fb.recommended_route_id:<15}  "
            f"${shp['max_budget_usd']:>10,.2f}  {shp['deadline_days']:>8.1f}d  "
            f"${fb_metric.expected_landed_cost:>10,.2f}  {fb_metric.expected_transit_days:>7.2f}d  "
            f"{miss*100:>9.1f}%"
        )


# ---------------------------------------------------------------------------
# Cost-weight sweep (per-shipment, unrelaxed constraints)
# ---------------------------------------------------------------------------
def _sweep_one_shipment(
    shp:        dict[str, Any],
    pareto:     list[RouteMetrics],
    w_costs:    list[float],
    scoring:    str = "minmax",
) -> list[str]:
    """Return winner archetype label at each w_cost step for a single shipment."""
    W_T_BASE, W_R_BASE = 0.30, 0.35
    denom = W_T_BASE + W_R_BASE

    if scoring == "minmax":
        costs = [m.expected_landed_cost  for m in pareto]
        times = [m.expected_transit_days for m in pareto]
        risks = [m.risk_index            for m in pareto]

        c_vals = _minmax_normalize(costs)
        t_vals = _minmax_normalize(times)
        r_vals = _minmax_normalize(risks)
    else:  # reference
        max_budget = shp["max_budget_usd"]
        deadline   = shp["deadline_days"]
        c_vals = [m.expected_landed_cost / max_budget for m in pareto]
        t_vals = [m.expected_transit_days / deadline for m in pareto]
        r_vals = [m.risk_index for m in pareto]

    labels: list[str] = []
    for w_c in w_costs:
        remaining = 1.0 - w_c
        w_t = remaining * (W_T_BASE / denom) if remaining > 0 else 0.0
        w_r = remaining * (W_R_BASE / denom) if remaining > 0 else 0.0

        best_score = float("inf")
        best_label = pareto[0].archetype
        for m, c_, t_, r_ in zip(pareto, c_vals, t_vals, r_vals):
            s = w_c * c_ + w_t * t_ + w_r * r_
            if s < best_score:
                best_score = s
                best_label = m.archetype
        labels.append(best_label)

    return labels


def _run_sweep(
    shipments: list[dict[str, Any]],
    cat:       dict[str, Any],
    scoring:   str = "minmax",
) -> dict:
    """
    Run cost-weight sweep on every shipment with >=2 Pareto survivors under
    its own UNRELAXED constraints.

    Returns summary dict with flip counts and first flipping shipment info.
    """
    w_costs = list(np.linspace(0.0, 1.0, SWEEP_STEPS))

    n_swept        = 0
    n_flipping     = 0
    first_flip_sid = None
    first_flip_pareto: list[RouteMetrics] = []
    first_flip_labels: list[str] = []
    first_flip_points: list[float] = []
    all_shipment_flips: dict[str, list[float]] = {}

    _hdr(f"Cost-Weight Sweep [{scoring.upper()} mode] (unrelaxed constraints, >=2 Pareto survivors)")
    print(f"  Candidates: {len(cat['ge2'])} shipments with >=2 Pareto survivors\n")

    for shp in shipments:
        sid = shp["shipment_id"]
        if sid not in cat["ge2"]:
            continue

        feasible = _get_feasible_metrics(shp)
        pareto   = _pareto_filter(feasible)
        if len(pareto) < 2:
            continue

        n_swept += 1
        labels   = _sweep_one_shipment(shp, pareto, w_costs, scoring=scoring)
        flips    = [round(float(w_costs[k]), 3) for k in range(1, len(w_costs)) if labels[k] != labels[k-1]]

        if flips:
            n_flipping += 1
            all_shipment_flips[sid] = flips
            if first_flip_sid is None:
                first_flip_sid    = sid
                first_flip_pareto = pareto
                first_flip_labels = labels
                first_flip_points = flips
            print(
                f"  {sid}: {len(flips)} flip point(s) at "
                f"w_cost = {flips}"
            )

    distinct_flips = sorted(list(set(f for fl in all_shipment_flips.values() for f in fl)))

    if n_flipping == 0:
        print("  No shipments showed a decision flip under unrelaxed constraints.")

    print(
        f"\n  Summary [{scoring.upper()} mode]:\n"
        f"    Swept shipments          : {n_swept}\n"
        f"    Shipments with >=1 flip  : {n_flipping} ({100*n_flipping/max(n_swept,1):.1f}%)\n"
        f"    Shipments with 0 flips   : {n_swept - n_flipping}\n"
        f"    Distinct flip points ({len(distinct_flips)}) : {distinct_flips}"
    )

    return {
        "n_swept":         n_swept,
        "n_flipping":      n_flipping,
        "first_flip_sid":  first_flip_sid,
        "pareto":          first_flip_pareto,
        "labels":          first_flip_labels,
        "flip_points":     first_flip_points,
        "distinct_flips":  distinct_flips,
        "w_costs":         w_costs,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def _save_decision_flip_plot(
    shipment_id:   str,
    w_costs:       list[float],
    winner_labels: list[str],
    out_path:      Path,
) -> None:
    seen: list[str] = []
    for lbl in winner_labels:
        if lbl not in seen:
            seen.append(lbl)
    unique_archetypes = seen

    y_idx = [unique_archetypes.index(lbl) for lbl in winner_labels]

    fig, ax = plt.subplots(figsize=(10, max(4, len(unique_archetypes) * 1.5)))

    ax.step(w_costs, y_idx, where="post", color="#1D4ED8", linewidth=2.8, zorder=3)

    # Shade regime bands
    prev_i = 0
    for k in range(1, len(w_costs)):
        if y_idx[k] != y_idx[prev_i] or k == len(w_costs) - 1:
            end_k  = k if y_idx[k] != y_idx[prev_i] else k + 1
            arch   = unique_archetypes[y_idx[prev_i]]
            colour = _ARCHETYPE_COLOURS.get(arch, "#94A3B8")
            ax.axhspan(
                y_idx[prev_i] - 0.45, y_idx[prev_i] + 0.45,
                xmin=w_costs[prev_i],
                xmax=w_costs[min(end_k, len(w_costs) - 1)],
                alpha=0.13, color=colour, zorder=1,
            )
            prev_i = k

    flip_x = [w_costs[k] for k in range(1, len(w_costs)) if y_idx[k] != y_idx[k-1]]
    flip_y = [y_idx[k]   for k in range(1, len(w_costs)) if y_idx[k] != y_idx[k-1]]
    if flip_x:
        ax.scatter(flip_x, flip_y, s=90, color="#EF4444", zorder=5,
                   label=f"{len(flip_x)} decision flip(s)")

    ax.set_yticks(range(len(unique_archetypes)))
    ax.set_yticklabels(
        [_ARCHETYPE_LABELS.get(a, a) for a in unique_archetypes], fontsize=11
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.6, len(unique_archetypes) - 0.4)
    ax.set_xlabel(r"Cost Weight  ($w_{\mathrm{cost}}$)", fontsize=12)
    ax.set_ylabel("Selected Route Archetype", fontsize=12)
    ax.set_title(
        f"Decision Flip Analysis  \u00b7  {shipment_id}  (unrelaxed constraints)\n"
        r"$w_{\mathrm{time}}$ & $w_{\mathrm{risk}}$ rescaled proportionally (30:35)",
        fontsize=12, fontweight="bold",
    )
    ax.axvline(0.35, color="#6B7280", linestyle="--", linewidth=1.5,
               label="Standard profile (w_cost=0.35)")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[evaluate] Saved plot -> {out_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------
def run_evaluation(
    shipments:     list[dict[str, Any]],
    artifacts_dir: Path,
) -> None:
    """Run the full benchmark suite for both scoring modes."""

    # 0. Categorise dataset
    cat = _categorise(shipments)

    _hdr("Dataset overview")
    print(
        f"  Total shipments               : {len(shipments)}\n"
        f"  >=2 Pareto survivors          : {len(cat['ge2'])}\n"
        f"  Exactly 1 Pareto survivor     : {len(cat['eq1'])}\n"
        f"  Infeasible (0 routes)         : {len(cat['eq0'])}  {cat['eq0']}"
    )

    # 1. Benchmark (both modes)
    all_results, crn_arrays = _run_benchmark(shipments, cat)

    _print_table("Table A: Feasible shipments only", all_results, feasible_only=True)
    _print_table("Table B: All 35 shipments",        all_results, feasible_only=False)

    # 2. Diff & decision characteristics analysis
    _print_diff_analysis(all_results, shipments)

    # 3. Detailed evaluation of SHP-001 under both modes
    _print_shp001_comparison(shipments)

    # 4. Infeasible shipments simulated miss rate
    _print_infeasible_simulation(shipments, cat, crn_arrays)

    # 5. Alpha sensitivity
    _print_alpha_sensitivity(shipments, cat)

    # 6. Cost-weight sweep: Minmax & Reference
    sweep_mm  = _run_sweep(shipments, cat, scoring="minmax")
    sweep_ref = _run_sweep(shipments, cat, scoring="reference")

    if sweep_mm["first_flip_sid"]:
        _save_decision_flip_plot(
            shipment_id   = sweep_mm["first_flip_sid"],
            w_costs       = sweep_mm["w_costs"],
            winner_labels = sweep_mm["labels"],
            out_path      = artifacts_dir / "decision_flip.png",
        )
    else:
        print(
            "\n[evaluate] No shipment flipped; decision_flip.png not saved.",
            file=sys.stderr,
        )

    # 7. SHP-001 Reference Decision-Flip Plot
    _save_shp001_reference_flip_plot(
        shipments = shipments,
        out_path  = artifacts_dir / "decision_flip_reference.png",
        step      = 0.005,
    )


# ---------------------------------------------------------------------------
# SHP-001 Reference Scoring Decision-Flip Plot (Compact)
# ---------------------------------------------------------------------------
def _save_shp001_reference_flip_plot(
    shipments: list[dict[str, Any]],
    out_path:  Path,
    step:      float = 0.005,
) -> float:
    """
    Generate decision-flip plot for SHP-001 under reference scoring.
    - Uses SHP-001's own profile: urgent (w_cost=0.15, w_time=0.50, w_risk=0.35)
    - Proportional rescaling of w_time and w_risk as w_cost varies in [0, 1]
    - Dashed vertical line labelled 'urgent profile (w_cost = 0.15)'
    - Real route labels with values from data (e.g. 'R0 Air ($11,805, 5.9 d)')
    - Fine grid (step 0.005)
    - Compact: figsize=(6.5, 2.4), dpi=200, readable fonts
    - Title: 'SHP-001: recommended route vs cost weight (reference scoring)'
    - Returns exact flip value
    """
    shp = next((s for s in shipments if s["shipment_id"] == "SHP-001"), None)
    if shp is None:
        raise ValueError("SHP-001 not found in dataset")

    res = run_pipeline(shp, scoring="reference")
    pareto = res.scored_routes  # [SHP-001_R0, SHP-001_R3]

    max_budget = shp["max_budget_usd"]
    deadline   = shp["deadline_days"]

    c_pen = [m.expected_landed_cost / max_budget for m in pareto]
    t_pen = [m.expected_transit_days / deadline for m in pareto]
    r_pen = [m.risk_index for m in pareto]

    # SHP-001 profile is urgent: (w_cost=0.15, w_time=0.50, w_risk=0.35)
    pw = PROFILE_WEIGHTS[shp["profile"]]
    w_c_actual = pw["cost"]
    W_T_BASE   = pw["time"]
    W_R_BASE   = pw["risk"]
    denom      = W_T_BASE + W_R_BASE

    # Analytical exact flip point between the two surviving candidates
    # Score(w_c) = w_c * c + (1 - w_c) * [(W_T_BASE/denom)*t + (W_R_BASE/denom)*r]
    tr = [(W_T_BASE / denom) * t + (W_R_BASE / denom) * r for t, r in zip(t_pen, r_pen)]
    exact_flip = (tr[1] - tr[0]) / ((c_pen[0] - c_pen[1]) + (tr[1] - tr[0]))

    # Fine grid sweep
    w_costs = np.arange(0.0, 1.0 + step / 2.0, step)
    winner_idx = []
    for wc in w_costs:
        rem = 1.0 - wc
        wt = rem * (W_T_BASE / denom) if rem > 0 else 0.0
        wr = rem * (W_R_BASE / denom) if rem > 0 else 0.0
        scores = [wc * c + wt * t + wr * r for c, t, r in zip(c_pen, t_pen, r_pen)]
        winner_idx.append(int(np.argmin(scores)))

    # Real route labels from data: e.g. "R0 Air ($11,805, 5.9 d)"
    route_labels = []
    for m in pareto:
        short_id = m.route_id.split("_")[-1]
        lbl = f"{short_id} {m.mode} (${m.base_freight_cost_usd:,.0f}, {m.expected_transit_days:.1f} d)"
        route_labels.append(lbl)

    # Plot creation with required compact dimensions
    fig, ax = plt.subplots(figsize=(6.5, 2.4), dpi=200)

    ax.step(w_costs, winner_idx, where="post", color="#1D4ED8", linewidth=2.0, zorder=3)

    # Regime shading
    ax.axvspan(0.0, exact_flip, alpha=0.08, color="#F97316")
    ax.axvspan(exact_flip, 1.0, alpha=0.08, color="#3B82F6")

    # Urgent profile dashed vertical line
    ax.axvline(
        w_c_actual,
        color="#DC2626",
        linestyle="--",
        linewidth=1.3,
        zorder=4,
        label=f"urgent profile (w_cost = {w_c_actual:.2f})",
    )

    # Mark flip point
    ax.scatter([exact_flip], [1], color="#B91C1C", s=30, zorder=5, label=f"flip point (w_cost = {exact_flip:.3f})")

    ax.set_yticks([0, 1])
    ax.set_yticklabels(route_labels, fontsize=8.5)
    ax.set_ylim(-0.4, 1.4)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"Cost Weight  ($w_{\mathrm{cost}}$)", fontsize=9)
    ax.set_title("SHP-001: recommended route vs cost weight (reference scoring)", fontsize=9.5, fontweight="bold", pad=6)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.92)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    _hdr("SHP-001 Reference Decision-Flip Plot (Fine Grid)")
    print(f"  Exact analytical flip point : w_cost = {exact_flip:.6f} ({exact_flip:.4f})")
    print(f"  Fine grid step              : {step} ({len(w_costs)} evaluation points)")
    print(f"  Candidate routes            : {route_labels[0]} -> {route_labels[1]}")
    print(f"  Saved plot                  : {out_path}")

    return exact_flip


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the Aiqon logistics optimisation pipeline."
    )
    parser.add_argument(
        "--data", default=None,
        help="Path to shipments.json (default: data/shipments.json relative to repo root)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args      = _parse_args()
    repo_root = Path(__file__).resolve().parent.parent

    data_path = Path(args.data) if args.data else repo_root / "data" / "shipments.json"
    if not data_path.exists():
        print(f"[evaluate] ERROR: data file not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    with open(data_path, "r", encoding="utf-8") as fh:
        shipments = json.load(fh)

    print(f"[evaluate] Loaded {len(shipments)} shipments from {data_path}", file=sys.stderr)
    run_evaluation(shipments, artifacts_dir=repo_root / "artifacts")
