"""
src/generator.py
----------------
Synthetic dataset generator for the Aiqon logistics optimisation assignment.

Generates 35 shipments (seed 42) each with 3-5 route options drawn from four
archetypes:
  * Fast & Expensive
  * Balanced
  * Cheap & Risky
  * Dominated  (strictly dominated in all three objectives)

Deadline / budget sizing
------------------------
For each non-forced shipment, deadline and budget are drawn from E[T]-based
ranges so that Stage 4 weighted scoring is genuinely exercised:

  urgent:         deadline = fastest_ET + 1.0 + U(0.45,0.70)*(slowest_ET-fastest_ET)
                  budget   = cheapest_cost + U(0.85,1.10)*(priciest_cost-cheapest_cost)
  standard:       deadline = fastest_ET + 1.0 + U(0.65,0.90)*(slowest_ET-fastest_ET)
                  budget   = cheapest_cost + U(0.65,0.95)*(priciest_cost-cheapest_cost)
  cost_sensitive: deadline = fastest_ET + 1.0 + U(0.85,1.15)*(slowest_ET-fastest_ET)
                  budget   = cheapest_cost + U(0.45,0.75)*(priciest_cost-cheapest_cost)

A redraw loop (max 50 attempts) guarantees every non-forced shipment has
>=2 Pareto-surviving feasible routes.

Forced infeasible shipments (indices 10, 20, 30 → SHP-011, SHP-021, SHP-031):
  SHP-011: budget < cheapest_cost (budget-short), deadline generous
  SHP-021: deadline < fastest_ET + 1.0 (deadline-short), budget generous
  SHP-031: both budget-short AND deadline-short

All randomness uses one np.random.default_rng(42) passed through functions.

Run standalone:
    python -m src.generator
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility — ONE generator, passed explicitly everywhere
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODES    = ["Air", "Ocean", "Rail", "Road", "Ocean+Air"]  # kept for rng stream compat
ORIGINS  = ["Shanghai", "Hamburg", "Los Angeles", "Singapore", "Dubai", "Rotterdam"]
DESTS    = ["New York", "Frankfurt", "Chicago", "London", "Tokyo", "Sydney", "Mumbai"]
PROFILES = ["urgent", "standard", "cost_sensitive"]

# Separate seed for structural fields (hubs, legs, mode).
# MUST NOT advance the main numpy rng (seed 42).
STRUCT_SEED = 999

N_SHIPMENTS       = 35
CARGO_VALUE_RANGE = (20_000, 500_000)   # USD

# Route raw-parameter ranges per archetype
ARCHETYPE_RANGES: dict[str, dict] = {
    "fast_expensive": dict(
        cost=(4_000, 12_000), base_days=(2,  7),
        delay_prob=(0.05, 0.15), delay_days=(0.5, 2.0),
        congestion=(0.10, 0.35), geo_risk=(0.05, 0.25),
    ),
    "balanced": dict(
        cost=(2_500, 7_000), base_days=(7, 18),
        delay_prob=(0.10, 0.25), delay_days=(1.0, 4.0),
        congestion=(0.20, 0.50), geo_risk=(0.10, 0.35),
    ),
    "cheap_risky": dict(
        cost=(1_200, 4_000), base_days=(15, 35),
        delay_prob=(0.25, 0.55), delay_days=(3.0, 9.0),
        congestion=(0.40, 0.80), geo_risk=(0.25, 0.65),
    ),
    "dominated": dict(
        cost=(5_000, 11_000), base_days=(14, 30),
        delay_prob=(0.30, 0.60), delay_days=(3.0, 8.0),
        congestion=(0.50, 0.85), geo_risk=(0.30, 0.70),
    ),
}

# Indices (0-based) of the three forced-infeasible shipments
INFEASIBLE_BUDGET_IDX   = 10   # SHP-011 — budget-short only
INFEASIBLE_DEADLINE_IDX = 20   # SHP-021 — deadline-short only
INFEASIBLE_BOTH_IDX     = 30   # SHP-031 — both short

FORCED_INFEASIBLE_INDICES = {
    INFEASIBLE_BUDGET_IDX,
    INFEASIBLE_DEADLINE_IDX,
    INFEASIBLE_BOTH_IDX,
}

MAX_REDRAW_ATTEMPTS = 50

# ---------------------------------------------------------------------------
# Corridor table: geographically realistic sea hubs, air hubs, and arrival ports
# ---------------------------------------------------------------------------

CORRIDOR_TABLE: dict[tuple[str, str], dict[str, Any]] = {
    ("Mumbai", "Hamburg"): {
        "sea_hubs": ["Jebel Ali", "Port Said", "Piraeus"],
        "air_hubs": ["Dubai"],
        "arrival_port": "Rotterdam",
    },
    ("Dubai", "Tokyo"): {
        "sea_hubs": ["Colombo", "Singapore", "Busan"],
        "air_hubs": ["Hong Kong"],
        "arrival_port": "Yokohama",
    },
    ("Shanghai", "Rotterdam"): {
        "sea_hubs": ["Singapore", "Port Said", "Algeciras"],
        "air_hubs": ["Dubai"],
        "arrival_port": "Antwerp",
    },
    ("Santos", "Los Angeles"): {
        "sea_hubs": ["Cartagena", "Panama City", "Manzanillo"],
        "air_hubs": ["Miami"],
        "arrival_port": "Long Beach",
    },
    ("Singapore", "Chicago"): {
        "sea_hubs": ["Hong Kong", "Busan", "Yokohama"],
        "air_hubs": ["Anchorage"],
        "arrival_port": "Los Angeles",
    },
    ("Busan", "Antwerp"): {
        "sea_hubs": ["Singapore", "Port Said", "Algeciras"],
        "air_hubs": ["Dubai"],
        "arrival_port": "Rotterdam",
    },
    ("Guangzhou", "New York"): {
        "sea_hubs": ["Busan", "Panama City", "Kingston"],
        "air_hubs": ["Anchorage"],
        "arrival_port": "Newark",
    },
    ("Hong Kong", "Felixstowe"): {
        "sea_hubs": ["Singapore", "Port Said", "Rotterdam"],
        "air_hubs": ["Dubai"],
        "arrival_port": "Southampton",
    },
    ("Colombo", "Barcelona"): {
        "sea_hubs": ["Salalah", "Port Said", "Malta Freeport"],
        "air_hubs": ["Dubai"],
        "arrival_port": "Valencia",
    },
    ("Jakarta", "Hamburg"): {
        "sea_hubs": ["Singapore", "Port Said", "Piraeus"],
        "air_hubs": ["Dubai"],
        "arrival_port": "Rotterdam",
    },
}

PORT_PAIRS: list[tuple[str, str]] = list(CORRIDOR_TABLE.keys())

SHIPMENT_PORT_PAIRS: list[tuple[str, str]] = [
    ("Dubai", "Tokyo"),
    ("Mumbai", "Hamburg"),
    ("Santos", "Los Angeles"),
    ("Mumbai", "Hamburg"),
    ("Guangzhou", "New York"),
    ("Dubai", "Tokyo"),
    ("Shanghai", "Rotterdam"),
    ("Hong Kong", "Felixstowe"),
    ("Shanghai", "Rotterdam"),
    ("Jakarta", "Hamburg"),
    ("Dubai", "Tokyo"),
    ("Mumbai", "Hamburg"),
    ("Dubai", "Tokyo"),
    ("Jakarta", "Hamburg"),
    ("Mumbai", "Hamburg"),
    ("Hong Kong", "Felixstowe"),
    ("Shanghai", "Rotterdam"),
    ("Santos", "Los Angeles"),
    ("Mumbai", "Hamburg"),
    ("Guangzhou", "New York"),
    ("Singapore", "Chicago"),
    ("Guangzhou", "New York"),
    ("Jakarta", "Hamburg"),
    ("Santos", "Los Angeles"),
    ("Busan", "Antwerp"),
    ("Colombo", "Barcelona"),
    ("Hong Kong", "Felixstowe"),
    ("Busan", "Antwerp"),
    ("Singapore", "Chicago"),
    ("Jakarta", "Hamburg"),
    ("Santos", "Los Angeles"),
    ("Jakarta", "Hamburg"),
    ("Hong Kong", "Felixstowe"),
    ("Busan", "Antwerp"),
    ("Santos", "Los Angeles"),
]


def _get_corridor(origin: str, dest: str) -> dict[str, Any]:
    """Retrieve corridor entry for (origin, dest) with a safe fallback."""
    if (origin, dest) in CORRIDOR_TABLE:
        return CORRIDOR_TABLE[(origin, dest)]
    return {
        "sea_hubs": ["Singapore", "Port Said"],
        "air_hubs": ["Dubai"],
        "arrival_port": dest,
    }


# ---------------------------------------------------------------------------
# Helpers: route sampling (all use the passed rng)
# ---------------------------------------------------------------------------

# Module-level structural rng (seed 999, completely separate from main rng-42)
import random as _random
_STRUCT_RNG = _random.Random(STRUCT_SEED)


def _build_legs_and_mode(
    archetype:  str,
    origin:     str,
    dest:       str,
    rng999:     _random.Random,
) -> tuple[list[str], list[dict[str, str]], str]:
    """
    Build (hubs, legs, mode_string) for one route using the structural rng.
    The main numpy rng (seed 42) is NOT touched here.

    Geographic realism rules
    ------------------------
    - Hubs come only from the pair's corridor entry.
    - Hubs never equal origin, destination, or origin's port.
    - Hubs appear in geographic order along the route.
    - Road/Rail legs are allowed only from the arrival port (or a hub in the
      destination's own region) to the destination city, never across an ocean.
    - Air legs may connect any two points; Sea legs connect ports only.
    - Keep mode = legs joined with "+", de-duplicated.
    """
    corridor = _get_corridor(origin, dest)
    available_sea_hubs = [h for h in corridor["sea_hubs"] if h not in (origin, dest)]
    available_air_hubs = [h for h in corridor["air_hubs"] if h not in (origin, dest)]
    arrival_port       = corridor["arrival_port"]

    legs: list[dict[str, str]] = []
    hubs: list[str] = []

    if archetype == "fast_expensive":
        # Air is the main haul: direct Air, via air hub, or Air to arrival port + Road last-mile
        draw = rng999.random()
        if draw < 0.40 and available_air_hubs:
            air_hub = rng999.choice(available_air_hubs)
            hubs.append(air_hub)
            legs = [
                {"from": origin,  "to": air_hub, "mode": "Air"},
                {"from": air_hub, "to": dest,    "mode": "Air"},
            ]
        elif draw < 0.70 and arrival_port not in (origin, dest):
            hubs.append(arrival_port)
            legs = [
                {"from": origin,       "to": arrival_port, "mode": "Air"},
                {"from": arrival_port, "to": dest,         "mode": "Road"},
            ]
        else:
            legs = [{"from": origin, "to": dest, "mode": "Air"}]

    elif archetype == "balanced":
        # Balanced multimodal: Sea+Air hybrid or Sea+Rail/Road from arrival port
        use_sea_air = rng999.random() < 0.50
        if use_sea_air and available_sea_hubs:
            hub = rng999.choice(available_sea_hubs)
            hubs.append(hub)
            legs = [
                {"from": origin, "to": hub,  "mode": "Sea"},
                {"from": hub,    "to": dest, "mode": "Air"},
            ]
        elif arrival_port not in (origin, dest):
            last_mode = rng999.choice(["Rail", "Road"])
            hubs.append(arrival_port)
            legs = [
                {"from": origin,       "to": arrival_port, "mode": "Sea"},
                {"from": arrival_port, "to": dest,         "mode": last_mode},
            ]
        elif available_sea_hubs:
            hub = rng999.choice(available_sea_hubs)
            hubs.append(hub)
            legs = [
                {"from": origin, "to": hub,  "mode": "Sea"},
                {"from": hub,    "to": dest, "mode": "Sea"},
            ]
        else:
            legs = [{"from": origin, "to": dest, "mode": "Sea"}]

    elif archetype == "cheap_risky":
        # Sea via 1-2 transshipment hubs in geographic order, then Road/Rail from arrival port
        n_sea_hubs = rng999.choice([1, 2])
        if len(available_sea_hubs) >= 2 and n_sea_hubs == 2:
            indices = sorted(rng999.sample(range(len(available_sea_hubs)), 2))
            chosen_sea_hubs = [available_sea_hubs[idx] for idx in indices]
        elif available_sea_hubs:
            idx = rng999.randrange(len(available_sea_hubs))
            chosen_sea_hubs = [available_sea_hubs[idx]]
        else:
            chosen_sea_hubs = []

        hubs.extend(chosen_sea_hubs)

        if arrival_port not in (origin, dest):
            last_mode = rng999.choice(["Road", "Rail"])
            hubs.append(arrival_port)
            waypoints = [origin] + chosen_sea_hubs + [arrival_port]
            legs = []
            for k in range(len(waypoints) - 1):
                legs.append({
                    "from": waypoints[k],
                    "to":   waypoints[k + 1],
                    "mode": "Sea",
                })
            legs.append({"from": arrival_port, "to": dest, "mode": last_mode})
        else:
            waypoints = [origin] + chosen_sea_hubs + [dest]
            legs = []
            for k in range(len(waypoints) - 1):
                legs.append({
                    "from": waypoints[k],
                    "to":   waypoints[k + 1],
                    "mode": "Sea",
                })

    else:  # dominated
        # Slow and suboptimal: Sea to arrival port (or via a sea hub), then overland Road to destination
        if arrival_port not in (origin, dest):
            if available_sea_hubs and rng999.random() < 0.50:
                hub = rng999.choice(available_sea_hubs)
                hubs.extend([hub, arrival_port])
                legs = [
                    {"from": origin,       "to": hub,          "mode": "Sea"},
                    {"from": hub,          "to": arrival_port, "mode": "Sea"},
                    {"from": arrival_port, "to": dest,         "mode": "Road"},
                ]
            else:
                hubs.append(arrival_port)
                legs = [
                    {"from": origin,       "to": arrival_port, "mode": "Sea"},
                    {"from": arrival_port, "to": dest,         "mode": "Road"},
                ]
        elif available_sea_hubs:
            # Regional hub overland trucking to dest (e.g. Algeciras -> Rotterdam)
            hub = available_sea_hubs[-1]
            hubs.append(hub)
            legs = [
                {"from": origin, "to": hub,  "mode": "Sea"},
                {"from": hub,    "to": dest, "mode": "Road"},
            ]
        else:
            legs = [{"from": origin, "to": dest, "mode": "Sea"}]

    # Derive mode string: de-duplicated leg modes in sequence order
    seen_modes: list[str] = []
    for leg in legs:
        m = leg["mode"]
        if m not in seen_modes:
            seen_modes.append(m)
    mode_str = "+".join(seen_modes)

    return hubs, legs, mode_str


def _sample_route(
    archetype:   str,
    shipment_id: str,
    route_idx:   int,
    rng:         np.random.Generator,
    origin:      str = "",
    dest:        str = "",
    struct_rng:  _random.Random | None = None,
) -> dict[str, Any]:
    """Sample one route dict from the given archetype using *rng*."""
    r = ARCHETYPE_RANGES[archetype]

    cost       = round(float(rng.uniform(*r["cost"])),       2)
    base_days  = round(float(rng.uniform(*r["base_days"])),  1)
    delay_prob = round(float(rng.uniform(*r["delay_prob"])), 4)
    delay_days = round(float(rng.uniform(*r["delay_days"])), 2)
    congestion = round(float(rng.uniform(*r["congestion"])), 4)
    geo_risk   = round(float(rng.uniform(*r["geo_risk"])),   4)
    # Keep this rng call to preserve the rng-42 stream; result unused for mode
    _raw_mode_draw = int(rng.integers(0, len(MODES)))

    # Build structural fields using the separate rng (seed 999)
    if struct_rng is None:
        struct_rng = _STRUCT_RNG
    hubs, legs, mode_str = _build_legs_and_mode(
        archetype, origin or "Port A", dest or "Port B", struct_rng
    )

    return {
        "route_id":                f"{shipment_id}_R{route_idx}",
        "mode":                    mode_str,
        "archetype":               archetype,
        "base_freight_cost_usd":   cost,
        "base_transit_days":       base_days,
        "delay_probability":       delay_prob,
        "delay_duration_days":     delay_days,
        "congestion_score":        congestion,
        "geopolitical_risk_score": geo_risk,
        "hubs":                    hubs,
        "legs":                    legs,
    }


def _expected_time(route: dict[str, Any]) -> float:
    """E[T] = base_transit_days + delay_probability * delay_duration_days."""
    return (
        route["base_transit_days"]
        + route["delay_probability"] * route["delay_duration_days"]
    )


# ---------------------------------------------------------------------------
# Pareto check (mirrors engine logic without importing it, to avoid circular
# dep at generation time; uses only E[C]=base_cost, E[T], Risk)
# ---------------------------------------------------------------------------
def _is_pareto_survivor(
    idx:     int,
    routes:  list[dict[str, Any]],
    ets:     list[float],
    costs:   list[float],
    risks:   list[float],
) -> bool:
    """Return True if route[idx] is NOT dominated by any other route in the list."""
    ec_i   = costs[idx]
    et_i   = ets[idx]
    risk_i = risks[idx]

    for j, r in enumerate(routes):
        if j == idx:
            continue
        ec_j   = costs[j]
        et_j   = ets[j]
        risk_j = risks[j]
        # j dominates i if j <= i in all dims AND strictly < in at least one
        if (ec_j <= ec_i and et_j <= et_i and risk_j <= risk_i
                and (ec_j < ec_i or et_j < et_i or risk_j < risk_i)):
            return False
    return True


def _pareto_surviving_count(
    routes:      list[dict[str, Any]],
    max_budget:  float,
    deadline:    float,
) -> int:
    """
    Count routes that pass Stage 1 (budget + deadline+1d buffer) and
    are not Pareto-dominated within the feasible set.
    Uses base_freight_cost_usd as proxy for E[C] (same relative ordering).
    """
    feasible_idx = []
    feasible_routes = []
    for i, r in enumerate(routes):
        et = _expected_time(r)
        if (r["base_freight_cost_usd"] <= max_budget
                and et + 1.0 <= deadline):
            feasible_idx.append(i)
            feasible_routes.append(r)

    if len(feasible_routes) < 2:
        return len(feasible_routes)

    ets   = [_expected_time(r) for r in feasible_routes]
    costs = [r["base_freight_cost_usd"] for r in feasible_routes]
    risks = [(r["congestion_score"] + r["geopolitical_risk_score"]) / 2.0
             for r in feasible_routes]

    count = sum(
        1 for i in range(len(feasible_routes))
        if _is_pareto_survivor(i, feasible_routes, ets, costs, risks)
    )
    return count


# ---------------------------------------------------------------------------
# Budget / deadline drawing
# ---------------------------------------------------------------------------
def _draw_budget_deadline(
    profile:        str,
    fastest_et:     float,
    slowest_et:     float,
    cheapest_cost:  float,
    priciest_cost:  float,
    rng:            np.random.Generator,
) -> tuple[float, float]:
    """
    Draw (max_budget, deadline) from profile-specific E[T]-anchored ranges.
    Returns (max_budget_usd, deadline_days).
    """
    et_span   = slowest_et - fastest_et      # >= 0
    cost_span = priciest_cost - cheapest_cost  # >= 0

    if profile == "urgent":
        dl_frac = float(rng.uniform(0.45, 0.70))
        bd_frac = float(rng.uniform(0.85, 1.10))
    elif profile == "standard":
        dl_frac = float(rng.uniform(0.65, 0.90))
        bd_frac = float(rng.uniform(0.65, 0.95))
    else:  # cost_sensitive
        dl_frac = float(rng.uniform(0.85, 1.15))
        bd_frac = float(rng.uniform(0.55, 0.90))   # was 0.45-0.75; widened

    deadline   = round(fastest_et + 1.0 + dl_frac * et_span, 2)
    max_budget = round(cheapest_cost + bd_frac * cost_span,   2)
    return max_budget, deadline


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------
def generate_shipments(
    n:   int                        = N_SHIPMENTS,
    rng: np.random.Generator | None = None,
) -> list[dict[str, Any]]:
    """
    Return a list of *n* synthetic shipment dicts.

    Parameters
    ----------
    n   : number of shipments to generate (default 35)
    rng : seeded Generator; if None, np.random.default_rng(SEED) is used.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    struct_rng = _random.Random(STRUCT_SEED)
    shipments: list[dict[str, Any]] = []

    for i in range(n):
        shipment_id = f"SHP-{i+1:03d}"
        profile_idx = int(rng.integers(0, len(PROFILES)))
        profile     = PROFILES[profile_idx]
        origin_idx  = int(rng.integers(0, len(ORIGINS)))
        dest_idx    = int(rng.integers(0, len(DESTS)))
        origin      = ORIGINS[origin_idx]
        dest        = DESTS[dest_idx]
        cargo_value = round(float(rng.uniform(*CARGO_VALUE_RANGE)), 2)

        # ── Route count: 3-5 ─────────────────────────────────────────────
        n_routes = int(rng.integers(3, 6))   # randint(3,6) → 3,4,5

        archetypes: list[str] = [
            "fast_expensive", "balanced", "cheap_risky", "dominated"
        ]
        while len(archetypes) < n_routes:
            extra_idx  = int(rng.integers(0, len(ARCHETYPE_RANGES)))
            archetypes.append(list(ARCHETYPE_RANGES.keys())[extra_idx])

        # Shuffle, then trim
        perm       = rng.permutation(len(archetypes))
        archetypes = [archetypes[k] for k in perm[:n_routes]]

        # Pick a port-pair for this shipment
        port_pair = SHIPMENT_PORT_PAIRS[i % len(SHIPMENT_PORT_PAIRS)]
        shp_origin, shp_dest = port_pair

        routes = [
            _sample_route(a, shipment_id, idx, rng, shp_origin, shp_dest, struct_rng)
            for idx, a in enumerate(archetypes)
        ]

        # ── Aggregate stats over ALL routes ───────────────────────────────
        ets   = [_expected_time(r) for r in routes]
        costs = [r["base_freight_cost_usd"] for r in routes]

        fastest_et    = min(ets)
        slowest_et    = max(ets)
        cheapest_cost = min(costs)
        priciest_cost = max(costs)

        # ── Forced-infeasible shipments ───────────────────────────────────
        if i == INFEASIBLE_BUDGET_IDX:
            # Budget-short only: budget < cheapest_cost, deadline generous
            max_budget = round(cheapest_cost * 0.55, 2)
            deadline   = round(slowest_et + 2.0,     2)

        elif i == INFEASIBLE_DEADLINE_IDX:
            # Deadline-short only: deadline < fastest_ET + 1.0, budget generous
            max_budget = round(priciest_cost * 1.20, 2)
            deadline   = round(fastest_et * 0.60,    2)   # < fastest_et + 1.0

        elif i == INFEASIBLE_BOTH_IDX:
            # Both short
            max_budget = round(cheapest_cost * 0.55, 2)
            deadline   = round(fastest_et * 0.60,    2)

        else:
            # ── Normal shipments: redraw until >=2 Pareto survivors ────────
            max_budget, deadline = _draw_budget_deadline(
                profile, fastest_et, slowest_et,
                cheapest_cost, priciest_cost, rng,
            )

            redrawn = False
            for attempt in range(1, MAX_REDRAW_ATTEMPTS):
                survivors = _pareto_surviving_count(routes, max_budget, deadline)
                if survivors >= 2:
                    redrawn = True
                    break
                max_budget, deadline = _draw_budget_deadline(
                    profile, fastest_et, slowest_et,
                    cheapest_cost, priciest_cost, rng,
                )

            if not redrawn:
                # Final check with widened constraints
                survivors = _pareto_surviving_count(routes, max_budget, deadline)
                if survivors < 2:
                    # Widen to ensure >=1 feasible route (avoid accidental 0-feasible)
                    max_budget = round(priciest_cost * 1.05, 2)
                    deadline   = round(slowest_et + 3.0,     2)
                    survivors  = _pareto_surviving_count(routes, max_budget, deadline)
                    print(
                        f"[generator] WARNING: {shipment_id} has only {survivors} "
                        f"Pareto survivor(s) after {MAX_REDRAW_ATTEMPTS} attempts "
                        f"(widened constraints applied).",
                        file=sys.stderr,
                    )

        shipments.append({
            "shipment_id":     shipment_id,
            "origin":          shp_origin,
            "destination":     shp_dest,
            "profile":         profile,
            "cargo_value_usd": cargo_value,
            "max_budget_usd":  max_budget,
            "deadline_days":   deadline,
            "routes":          routes,
        })

    return shipments


# ---------------------------------------------------------------------------
# Persist / load
# ---------------------------------------------------------------------------
def save_shipments(
    shipments: list[dict[str, Any]],
    path:      str | Path = "data/shipments.json",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(shipments, fh, indent=2)
    print(f"[generator] Wrote {len(shipments)} shipments -> {path}", file=sys.stderr)


def load_shipments(path: str | Path = "data/shipments.json") -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Standalone entry-point  (python -m src.generator)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.engine import run_pipeline  # lazy import to avoid circular at module level

    repo_root   = Path(__file__).resolve().parent.parent
    output_path = repo_root / "data" / "shipments.json"

    rng       = np.random.default_rng(SEED)
    shipments = generate_shipments(rng=rng)
    save_shipments(shipments, output_path)

    # ── Assertion block ───────────────────────────────────────────────────
    n_ge2    = 0
    n_eq1    = 0
    n_eq0    = 0
    infeas   = []

    for shp in shipments:
        result = run_pipeline(shp)
        n_pareto = len(result.scored_routes)
        if result.fallback is not None:
            n_eq0 += 1
            infeas.append(shp["shipment_id"])
        elif n_pareto == 1:
            n_eq1 += 1
        else:
            n_ge2 += 1

    print(f"\nDataset summary ({len(shipments)} shipments):")
    print(f"  >=2 Pareto survivors : {n_ge2}")
    print(f"  Exactly 1 survivor   : {n_eq1}")
    print(f"  Infeasible (0 routes): {n_eq0}  {infeas}")

    assert n_eq0 == 3, f"Expected exactly 3 infeasible, got {n_eq0}"
    assert n_ge2 >= 24, f"Expected >=24 with >=2 Pareto survivors, got {n_ge2}"

    print("\nAll assertions passed.")
    print(f"Generated {len(shipments)} shipments.")
