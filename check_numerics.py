"""
check_numerics.py
-----------------
Verify that every numeric field in data/shipments.json is IDENTICAL to
data/shipments_before.json. Only origin, destination, mode, hubs, legs may differ.

Run:
    python check_numerics.py
"""
import json
from pathlib import Path

NUMERIC_ROUTE_FIELDS = [
    "base_freight_cost_usd",
    "base_transit_days",
    "delay_probability",
    "delay_duration_days",
    "congestion_score",
    "geopolitical_risk_score",
]
NUMERIC_SHP_FIELDS = ["cargo_value_usd", "max_budget_usd", "deadline_days"]

repo = Path(__file__).resolve().parent
before = json.loads((repo / "data" / "shipments_before.json").read_text())
after  = json.loads((repo / "data" / "shipments.json").read_text())

diffs: list[str] = []

if len(before) != len(after):
    diffs.append(f"Shipment count: {len(before)} vs {len(after)}")
else:
    for b_shp, a_shp in zip(before, after):
        sid = b_shp["shipment_id"]
        for f in NUMERIC_SHP_FIELDS:
            bv, av = b_shp.get(f), a_shp.get(f)
            if bv != av:
                diffs.append(f"{sid}.{f}: {bv!r} -> {av!r}")

        b_routes = {r["route_id"]: r for r in b_shp["routes"]}
        a_routes = {r["route_id"]: r for r in a_shp["routes"]}

        if set(b_routes) != set(a_routes):
            diffs.append(
                f"{sid} route IDs changed: {set(b_routes)} vs {set(a_routes)}"
            )
            continue

        for rid in b_routes:
            br, ar = b_routes[rid], a_routes[rid]
            for f in NUMERIC_ROUTE_FIELDS:
                bv, av = br.get(f), ar.get(f)
                if bv != av:
                    diffs.append(f"{sid}.{rid}.{f}: {bv!r} -> {av!r}")

if diffs:
    print(f"DIFFERENCES FOUND ({len(diffs)}):")
    for d in diffs:
        print(f"  {d}")
else:
    print(f"OK: all {len(NUMERIC_ROUTE_FIELDS)} route numeric fields and "
          f"{len(NUMERIC_SHP_FIELDS)} shipment numeric fields are IDENTICAL "
          f"across all {len(before)} shipments.")

# ---------------------------------------------------------------------------
# Geographic realism check
# ---------------------------------------------------------------------------
geo_diffs: list[str] = []
for shp in after:
    sid = shp["shipment_id"]
    orig = shp["origin"]
    dest = shp["destination"]
    for r in shp["routes"]:
        rid = r["route_id"]
        hubs = r.get("hubs", [])
        legs = r.get("legs", [])
        mode = r.get("mode", "")

        # Check hubs don't equal origin or destination
        for h in hubs:
            if h in (orig, dest):
                geo_diffs.append(f"{sid}.{rid}: hub {h!r} equals origin or destination")

        # Check mode matches de-duplicated leg modes
        seen = []
        for leg in legs:
            m = leg["mode"]
            if m not in seen:
                seen.append(m)
        expected_mode = "+".join(seen)
        if mode != expected_mode:
            geo_diffs.append(f"{sid}.{rid}: mode {mode!r} != legs mode {expected_mode!r}")

        # Check Road/Rail legs only lead into destination
        for leg in legs:
            if leg["mode"] in ("Road", "Rail"):
                if leg["to"] != dest:
                    geo_diffs.append(
                        f"{sid}.{rid}: {leg['mode']} leg {leg['from']} -> {leg['to']} "
                        f"does not terminate at destination {dest}"
                    )

if geo_diffs:
    print(f"GEOGRAPHIC VIOLATIONS FOUND ({len(geo_diffs)}):")
    for gd in geo_diffs:
        print(f"  {gd}")
else:
    print("OK: all routes satisfy geographic realism rules (corridor hubs, "
          "overland-only Road/Rail to destination, and de-duplicated mode).")

