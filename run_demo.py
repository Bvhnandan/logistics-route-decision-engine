"""
run_demo.py
-----------
One-command CLI demonstration of the Aiqon logistics optimisation pipeline.

Default auto-selection:
  Demo 1 -- feasible shipment with >=2 Pareto survivors where winner != cheapest
             (so trade-off analysis shows non-zero deltas).
  Demo 2 -- infeasible shipment with budget-only or dual advisory (SHP-011 or SHP-031).
  Demo 3 -- infeasible shipment with deadline-only advisory (SHP-021).

Usage
-----
    python run_demo.py
    python run_demo.py --ids SHP-003 SHP-011
    python run_demo.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.engine  import run_pipeline
from src.explain import print_explanation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _find_shipment(
    shipments: list[dict[str, Any]], sid: str
) -> dict[str, Any] | None:
    target = sid.strip().upper()
    return next((s for s in shipments if s["shipment_id"].strip().upper() == target), None)


def _list_shipments(shipments: list[dict[str, Any]], scoring: str = "reference") -> None:
    """Print a clean 1-line summary table of all shipments."""
    print("=" * 102)
    print("  AIQON LOGISTICS OPTIMISATION PIPELINE — SHIPMENT CATALOG (35 SHIPMENTS)")
    print("=" * 102)
    print(
        f"  {'ID':<8}  {'Corridor (Origin -> Destination)':<28}  "
        f"{'Profile':<15}  {'Budget':>12}  {'Deadline':>9}  {'Pareto Routes':>15}  {'Status':<10}"
    )
    print(f"  {'-'*8}  {'-'*28}  {'-'*15}  {'-'*12}  {'-'*9}  {'-'*15}  {'-'*10}")

    for s in shipments:
        sid = s["shipment_id"]
        corridor = f"{s.get('origin', 'N/A')} -> {s.get('destination', 'N/A')}"
        prof = s.get("profile", "N/A")
        budget = s.get("max_budget_usd", 0.0)
        deadline = s.get("deadline_days", 0.0)

        result = run_pipeline(s, scoring=scoring)
        n_pareto = len(result.scored_routes)

        if result.fallback is not None:
            status = "Infeasible"
            pareto_str = "0 (Infeasible)"
        else:
            status = "Low-Conf" if result.low_confidence else "Feasible"
            pareto_str = f"{n_pareto} route{'s' if n_pareto != 1 else ''}"

        print(
            f"  {sid:<8}  {corridor:<28}  {prof:<15}  "
            f"${budget:>11,.2f}  {deadline:>8.1f}d  {pareto_str:>15}  {status:<10}"
        )

    print("=" * 102)
    print(f"  Total: {len(shipments)} shipments | Scoring: {scoring.upper()} mode\n")


def _find_trade_off_demo(
    shipments: list[dict[str, Any]], scoring: str = "reference"
) -> dict[str, Any] | None:
    """
    Return the first feasible shipment where:
      - At least 2 Pareto survivors exist
      - The engine winner is NOT the cheapest Pareto survivor
        (ensures non-zero delta in trade-off analysis)
    """
    for s in shipments:
        result = run_pipeline(s, scoring=scoring)
        if result.fallback is not None or result.winner_id is None:
            continue
        if len(result.scored_routes) < 2:
            continue
        cheapest_id = min(
            result.scored_routes, key=lambda m: m.expected_landed_cost
        ).route_id
        if result.winner_id != cheapest_id:
            return s

    # Fallback: any feasible shipment with >=2 Pareto survivors
    for s in shipments:
        result = run_pipeline(s, scoring=scoring)
        if result.fallback is None and len(result.scored_routes) >= 2:
            return s

    print(
        "[run_demo] WARNING: No feasible shipment with >=2 Pareto survivors found.",
        file=sys.stderr,
    )
    return None


def _run_and_print(shipment: dict[str, Any], scoring: str = "reference") -> None:
    result = run_pipeline(shipment, scoring=scoring)
    print_explanation(result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo the Aiqon logistics route optimisation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_demo.py                      # Run default 3 showcase demo shipments\n"
            "  python run_demo.py --list               # Summary table of all 35 shipments\n"
            "  python run_demo.py SHP-001              # Run specific shipment (case-insensitive)\n"
            "  python run_demo.py shp-001              # Case-insensitive also works\n"
            "  python run_demo.py SHP-003 SHP-031      # Run multiple specific shipments\n"
            "  python run_demo.py --scoring minmax     # Run with minmax scoring mode\n"
            "  python run_demo.py --all                # Run all 35 shipments\n"
        ),
    )
    parser.add_argument("shipment_ids", nargs="*", metavar="SHIPMENT_ID",
                        help="One or more shipment IDs to evaluate (case-insensitive, e.g. SHP-001)")
    parser.add_argument("--list", action="store_true",
                        help="Print a clean 1-line summary table of all 35 shipments and exit")
    parser.add_argument("--data", default=None, metavar="PATH",
                        help="Path to shipments.json (default: data/shipments.json)")
    parser.add_argument("--scoring", default="reference", choices=["minmax", "reference"],
                        help="Stage 4 scoring mode: 'reference' (default) or 'minmax'")
    parser.add_argument("--ids", nargs="+", metavar="ID", default=[],
                        help="Alternative flag to specify one or more shipment IDs")
    parser.add_argument("--all", action="store_true",
                        help="Run the pipeline on every shipment in the dataset")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    data_path = (
        Path(args.data) if args.data
        else _REPO_ROOT / "data" / "shipments.json"
    )
    if not data_path.exists():
        print(
            f"[run_demo] ERROR: Dataset not found at {data_path}\n"
            "  Run: python -m src.generator  to generate it first.",
            file=sys.stderr,
        )
        return 1

    with open(data_path, "r", encoding="utf-8") as fh:
        shipments: list[dict[str, Any]] = json.load(fh)

    # ── Handle --list ─────────────────────────────────────────────────────
    if args.list:
        _list_shipments(shipments, scoring=args.scoring)
        return 0

    print(f"[run_demo] Loaded {len(shipments)} shipments from {data_path}\n")

    # Combine positional shipment_ids and --ids
    requested_ids = list(args.shipment_ids) + [i for i in args.ids if i not in args.shipment_ids]

    # ── Select targets ────────────────────────────────────────────────────
    if args.all:
        targets = shipments

    elif requested_ids:
        targets = []
        for sid in requested_ids:
            s = _find_shipment(shipments, sid)
            if s is None:
                print(f"[run_demo] WARNING: Shipment ID {sid!r} not found – skipping.",
                      file=sys.stderr)
            else:
                targets.append(s)

    else:
        targets = []

        # Demo 1: feasible with non-trivial winner
        demo1 = _find_trade_off_demo(shipments, scoring=args.scoring)
        if demo1:
            targets.append(demo1)
            print(
                f"[run_demo] Demo 1 — Feasible Trade-Off: {demo1['shipment_id']}",
                file=sys.stderr,
            )

        # Demo 2: SHP-031 (both-short → dual advisory)
        #         or SHP-011 (budget-only) as backup
        demo2 = _find_shipment(shipments, "SHP-031") or _find_shipment(shipments, "SHP-011")
        if demo2:
            targets.append(demo2)
            print(
                f"[run_demo] Demo 2 — Budget/Dual Infeasible: {demo2['shipment_id']}",
                file=sys.stderr,
            )

        # Demo 3: SHP-021 (deadline-only advisory)
        demo3 = _find_shipment(shipments, "SHP-021")
        if demo3:
            targets.append(demo3)
            print(
                "[run_demo] Demo 3 — Deadline Infeasible: SHP-021",
                file=sys.stderr,
            )

    if not targets:
        print("[run_demo] No shipments to evaluate.", file=sys.stderr)
        return 1

    print(f"\n{'#'*72}")
    print(f"  Aiqon Logistics Route Optimisation  |  {len(targets)} shipment(s)")
    print(f"{'#'*72}\n")

    for i, shp in enumerate(targets, start=1):
        print(f"\n[{i}/{len(targets)}] Processing {shp['shipment_id']} ...")
        _run_and_print(shp, scoring=args.scoring)

    print("\n[run_demo] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
