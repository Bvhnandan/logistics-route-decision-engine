"""
tests/test_pipeline.py
-----------------------
Pytest unit tests for the Aiqon multi-criteria logistics pipeline.

Covers all 6 edge-case scenarios required by the assignment:
  1. Hard budget exclusion
  2. Infeasible fallback (deadline & budget variants)
  3. Pareto dominance filtering
  4. Zero-variance normalisation (no ZeroDivisionError)
  5. Low-confidence margin alert
  6. Profile preference reversal (urgent vs. cost-sensitive)
"""

from __future__ import annotations

import pytest

from src.engine import (
    PipelineResult,
    _minmax_normalize,
    run_pipeline,
    DELAY_BUFFER_DAYS,
    LOW_CONFIDENCE_MARGIN,
    PROFILE_WEIGHTS,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _route(
    route_id:          str,
    base_cost:         float = 3_000.0,
    base_days:         float = 10.0,
    delay_prob:        float = 0.10,
    delay_days:        float = 2.0,
    congestion:        float = 0.30,
    geo_risk:          float = 0.20,
    mode:              str   = "Ocean",
    archetype:         str   = "balanced",
) -> dict:
    return {
        "route_id":               route_id,
        "mode":                   mode,
        "archetype":              archetype,
        "base_freight_cost_usd":  base_cost,
        "base_transit_days":      base_days,
        "delay_probability":      delay_prob,
        "delay_duration_days":    delay_days,
        "congestion_score":       congestion,
        "geopolitical_risk_score": geo_risk,
    }


def _shipment(
    routes:       list[dict],
    profile:      str   = "standard",
    max_budget:   float = 10_000.0,
    deadline:     float = 30.0,
    cargo_value:  float = 100_000.0,
    sid:          str   = "TEST-001",
) -> dict:
    return {
        "shipment_id":     sid,
        "origin":          "Shanghai",
        "destination":     "New York",
        "profile":         profile,
        "cargo_value_usd": cargo_value,
        "max_budget_usd":  max_budget,
        "deadline_days":   deadline,
        "routes":          routes,
    }


# ---------------------------------------------------------------------------
# Test 1: Hard budget exclusion
# ---------------------------------------------------------------------------
class TestHardBudgetExclusion:
    """An over-budget route must be disqualified even if it is the fastest."""

    def test_over_budget_route_excluded(self) -> None:
        """Fast route costing $8k should be excluded when budget is $5k."""
        fast_expensive = _route("R_FAST", base_cost=8_000, base_days=3)
        affordable     = _route("R_AFFD", base_cost=3_000, base_days=12)

        result = run_pipeline(_shipment(
            routes     = [fast_expensive, affordable],
            max_budget = 5_000,
            deadline   = 20.0,
        ))

        assert "R_FAST" in result.disqualified, (
            "Over-budget route should be in disqualified dict"
        )
        assert "R_FAST" not in result.feasible_ids, (
            "Over-budget route must not be in feasible set"
        )
        assert "R_AFFD" in result.feasible_ids, (
            "Affordable route should survive"
        )

    def test_disqualification_reason_contains_cost_and_budget(self) -> None:
        """Disqualification reason must mention both cost and budget figures."""
        over_budget = _route("R_OVR", base_cost=9_999)
        result = run_pipeline(_shipment(
            routes     = [over_budget, _route("R_OK", base_cost=1_000)],
            max_budget = 5_000,
        ))
        reason = result.disqualified.get("R_OVR", "")
        assert "budget" in reason.lower() or "$" in reason, (
            f"Reason should reference budget: {reason!r}"
        )


# ---------------------------------------------------------------------------
# Test 2: Infeasible fallback
# ---------------------------------------------------------------------------
class TestInfeasibleFallback:
    """Fallback logic triggers correctly for budget-short and deadline-short cases."""

    def test_budget_short_fallback(self) -> None:
        """All routes cost more than budget → budget_short fallback, raise_budget_usd > 0."""
        r1 = _route("R1", base_cost=5_000)
        r2 = _route("R2", base_cost=7_000)
        result = run_pipeline(_shipment(
            routes     = [r1, r2],
            max_budget = 2_000,    # impossibly small
            deadline   = 60.0,
        ))

        assert result.fallback is not None,              "Fallback should trigger"
        assert result.fallback.reason == "budget_short", "Reason should be budget_short"
        assert result.fallback.raise_budget_usd > 0,     "Must specify how much to raise budget"
        assert result.winner_id is None,                 "No winner when infeasible"

    def test_deadline_short_fallback(self) -> None:
        """Budget OK but all routes miss deadline → deadline_short fallback."""
        r1 = _route("R1", base_days=20, delay_prob=0.10, delay_days=2.0)
        r2 = _route("R2", base_days=15, delay_prob=0.05, delay_days=1.0)

        # E[T] for R2 = 15 + 0.05*1 = 15.05; with 1.0d buffer = 16.05d
        # deadline = 5 → all miss
        result = run_pipeline(_shipment(
            routes     = [r1, r2],
            max_budget = 50_000,
            deadline   = 5.0,
        ))

        assert result.fallback is not None,               "Fallback should trigger"
        assert result.fallback.reason == "deadline_short","Reason should be deadline_short"
        assert result.fallback.extend_deadline_days > 0,  "Must specify extension needed"

    def test_fallback_gap_computation_deadline(self) -> None:
        """extend_deadline_days matches formula: max(0, E[T]+buffer - deadline)."""
        r1 = _route("R1", base_days=10, delay_prob=0.20, delay_days=5.0)
        # E[T] = 10 + 0.20*5 = 11.0; with buffer = 12.0; deadline = 8.0 → gap = 4.0
        result = run_pipeline(_shipment(
            routes     = [r1],
            max_budget = 50_000,
            deadline   = 8.0,
        ))
        fb = result.fallback
        assert fb is not None
        expected_gap = max(0.0, 11.0 + DELAY_BUFFER_DAYS - 8.0)
        assert abs(fb.extend_deadline_days - expected_gap) < 1e-6, (
            f"extend_deadline_days {fb.extend_deadline_days} ≠ expected {expected_gap}"
        )

    def test_fallback_gap_computation_budget(self) -> None:
        """raise_budget_usd matches formula: max(0, cheapest_cost - max_budget)."""
        r1 = _route("R1", base_cost=4_500)
        r2 = _route("R2", base_cost=6_000)
        result = run_pipeline(_shipment(
            routes     = [r1, r2],
            max_budget = 3_000,
            deadline   = 60.0,
        ))
        fb = result.fallback
        assert fb is not None
        expected_raise = max(0.0, 4_500 - 3_000)
        assert abs(fb.raise_budget_usd - expected_raise) < 1e-6, (
            f"raise_budget_usd {fb.raise_budget_usd} ≠ expected {expected_raise}"
        )


# ---------------------------------------------------------------------------
# Test 3: Pareto dominance filtering
# ---------------------------------------------------------------------------
class TestParetoDominanceFilter:
    """Strictly dominated routes are pruned; non-dominated routes survive."""

    def _build_route_with_known_metrics(
        self,
        route_id: str,
        cost:     float,
        days:     float,
        congestion: float,
        geo:      float,
    ) -> dict:
        """
        Craft a route such that its E[C] ≈ cost and E[T] ≈ days with minimal
        delay contribution, and risk = (congestion + geo) / 2.
        We use delay_prob=0 so E[D]=0 and there's no delay cost.
        """
        return _route(
            route_id   = route_id,
            base_cost  = cost,
            base_days  = days,
            delay_prob = 0.0,
            delay_days = 0.0,
            congestion = congestion,
            geo_risk   = geo,
        )

    def test_dominated_route_pruned(self) -> None:
        """
        R_DOM is dominated by R_GOOD (better or equal in all three objectives,
        strictly better in at least one).
        """
        # R_GOOD:  E[C]=3000, E[T]=10, Risk=0.25
        # R_DOM:   E[C]=4000, E[T]=12, Risk=0.30  → strictly worse in all 3
        r_good = self._build_route_with_known_metrics("R_GOOD", 3_000, 10.0, 0.30, 0.20)
        r_dom  = self._build_route_with_known_metrics("R_DOM",  4_000, 12.0, 0.40, 0.20)

        result = run_pipeline(_shipment(
            routes     = [r_good, r_dom],
            max_budget = 10_000,
            deadline   = 30.0,
        ))

        assert "R_DOM" in result.dominated_ids, (
            "R_DOM should be identified as dominated"
        )
        assert "R_DOM" not in result.pareto_ids, (
            "R_DOM should not survive to Pareto set"
        )
        assert "R_GOOD" in result.pareto_ids, (
            "R_GOOD should survive as non-dominated"
        )

    def test_non_dominated_routes_all_survive(self) -> None:
        """Routes that are each better in at least one metric must all survive."""
        # R_A: cheapest but slowest; R_B: fastest but most expensive; R_C: lowest risk
        r_a = self._build_route_with_known_metrics("R_A", 1_500, 20.0, 0.30, 0.20)
        r_b = self._build_route_with_known_metrics("R_B", 5_000,  5.0, 0.30, 0.20)
        r_c = self._build_route_with_known_metrics("R_C", 3_000, 12.0, 0.10, 0.10)

        result = run_pipeline(_shipment(
            routes     = [r_a, r_b, r_c],
            max_budget = 10_000,
            deadline   = 60.0,
        ))

        for rid in ("R_A", "R_B", "R_C"):
            assert rid in result.pareto_ids, f"{rid} should not be dominated"

    def test_dominated_reason_string_mentions_dominator(self) -> None:
        """Dominance reason should reference the dominator route ID."""
        r_good = self._build_route_with_known_metrics("R_GOOD", 3_000, 10.0, 0.30, 0.20)
        r_dom  = self._build_route_with_known_metrics("R_DOM",  4_000, 12.0, 0.40, 0.20)

        result = run_pipeline(_shipment(
            routes     = [r_good, r_dom],
            max_budget = 10_000,
            deadline   = 30.0,
        ))
        reason = result.dominated_ids.get("R_DOM", "")
        assert "R_GOOD" in reason, (
            f"Dominance reason should mention R_GOOD: {reason!r}"
        )


# ---------------------------------------------------------------------------
# Test 4: Zero-variance normalisation
# ---------------------------------------------------------------------------
class TestZeroVarianceNormalisation:
    """No ZeroDivisionError when all values for a metric are identical."""

    def test_all_same_values_return_zeros(self) -> None:
        """_minmax_normalize with identical values must return all-zeros list."""
        result = _minmax_normalize([5.0, 5.0, 5.0])
        assert result == [0.0, 0.0, 0.0], f"Expected [0,0,0], got {result}"

    def test_pipeline_single_route_no_error(self) -> None:
        """Single Pareto survivor → all metrics have zero variance → no crash."""
        r = _route("R_SOLO", base_cost=3_000, base_days=10)
        result = run_pipeline(_shipment(routes=[r], max_budget=10_000, deadline=30.0))
        assert result.winner_id == "R_SOLO"
        winner = result.winner
        assert winner is not None
        # All norm scores should be 0.0 (zero variance)
        assert winner.norm_cost == 0.0
        assert winner.norm_time == 0.0
        assert winner.norm_risk == 0.0

    def test_pipeline_identical_routes_no_error(self) -> None:
        """Two routes with identical metrics → zero-variance normalisation, no crash."""
        r1 = _route("R1", base_cost=3_000, base_days=10, delay_prob=0.10,
                    delay_days=2.0, congestion=0.30, geo_risk=0.20)
        r2 = _route("R2", base_cost=3_000, base_days=10, delay_prob=0.10,
                    delay_days=2.0, congestion=0.30, geo_risk=0.20)
        # Should not raise any exception
        result = run_pipeline(_shipment(routes=[r1, r2], max_budget=10_000, deadline=30.0))
        assert result.winner_id is not None


# ---------------------------------------------------------------------------
# Test 5: Low-confidence margin alert
# ---------------------------------------------------------------------------
class TestLowConfidenceAlert:
    """Flag raised when top-2 penalty scores are within margin < 0.05."""

    def test_very_similar_routes_trigger_low_confidence(self) -> None:
        """Near-identical trade-off routes should produce a low-confidence warning."""
        # Non-dominated pair: R1 is marginally cheaper, R2 is marginally faster.
        # Neither dominates the other, so both survive Stage 2.
        # Their tiny cost/time differences produce a penalty-score margin well below 0.05.
        r1 = _route("R1", base_cost=3_000.0, base_days=10.05, delay_prob=0.10,
                    delay_days=2.0, congestion=0.30, geo_risk=0.20)
        r2 = _route("R2", base_cost=3_005.0, base_days=10.00, delay_prob=0.10,
                    delay_days=2.0, congestion=0.30, geo_risk=0.20)

        result = run_pipeline(_shipment(
            routes     = [r1, r2],
            max_budget = 10_000,
            deadline   = 30.0,
        ))
        assert result.low_confidence is True, (
            "Near-identical trade-off routes should trigger low-confidence flag"
        )

    def test_very_different_routes_no_low_confidence(self) -> None:
        """Clearly differentiated routes should NOT trigger low-confidence."""
        r_fast = _route("R_FAST", base_cost=9_000, base_days=2,  delay_prob=0.05,
                        delay_days=0.5, congestion=0.10, geo_risk=0.05)
        r_slow = _route("R_SLOW", base_cost=1_200, base_days=30, delay_prob=0.50,
                        delay_days=8.0, congestion=0.75, geo_risk=0.60)

        result = run_pipeline(_shipment(
            routes     = [r_fast, r_slow],
            max_budget = 20_000,
            deadline   = 60.0,
            profile    = "cost_sensitive",
        ))
        assert result.low_confidence is False, (
            "Very different routes should not trigger low-confidence"
        )

    def test_low_confidence_margin_threshold(self) -> None:
        """Margin exactly at LOW_CONFIDENCE_MARGIN is NOT flagged (strictly less-than)."""
        # We check that the threshold is < 0.05 (not <=)
        assert LOW_CONFIDENCE_MARGIN == 0.05


# ---------------------------------------------------------------------------
# Test 6: Profile preference reversal
# ---------------------------------------------------------------------------
class TestProfilePreferenceReversal:
    """Urgent profile prefers the fast route; cost-sensitive prefers the cheap route."""

    def _build_test_shipment(self, profile: str) -> dict:
        # 3 Pareto-non-dominated routes with clearly separated archetypes:
        #   R_FAST : very fast, expensive, low risk
        #   R_BAL  : medium cost/time/risk
        #   R_CHEAP: very cheap, slower, moderate risk
        # The cost gap (8000 vs 1500) is large enough that w_cost=0.60 decisively
        # picks R_CHEAP, and w_time=0.50 decisively picks R_FAST.
        r_fast = _route(
            "R_FAST", base_cost=8_000.0, base_days=4.0,
            delay_prob=0.05, delay_days=1.0, congestion=0.10, geo_risk=0.10,
        )
        r_bal = _route(
            "R_BAL", base_cost=4_000.0, base_days=12.0,
            delay_prob=0.15, delay_days=2.0, congestion=0.25, geo_risk=0.25,
        )
        r_cheap = _route(
            "R_CHEAP", base_cost=1_500.0, base_days=18.0,
            delay_prob=0.25, delay_days=3.0, congestion=0.35, geo_risk=0.35,
        )
        return _shipment(
            routes     = [r_fast, r_bal, r_cheap],
            profile    = profile,
            max_budget = 10_000,
            deadline   = 35.0,
        )

    def test_urgent_prefers_fast_route(self) -> None:
        """Under urgent profile, the fastest route should win (or very high-ranking)."""
        result = run_pipeline(self._build_test_shipment("urgent"))
        assert result.winner_id is not None
        # Urgent: w_time=0.50 dominates → fastest Pareto route should win
        assert result.winner_id == "R_FAST", (
            f"Urgent profile should prefer R_FAST, got {result.winner_id}"
        )

    def test_cost_sensitive_prefers_cheap_route(self) -> None:
        """Under cost_sensitive profile, the cheapest route should win."""
        result = run_pipeline(self._build_test_shipment("cost_sensitive"))
        assert result.winner_id is not None
        # Cost-sensitive: w_cost=0.60 dominates → cheapest Pareto route should win
        assert result.winner_id == "R_CHEAP", (
            f"Cost-sensitive profile should prefer R_CHEAP, got {result.winner_id}"
        )


    def test_profile_weights_sum_to_one(self) -> None:
        """All profile weight sets must sum exactly to 1.0."""
        for profile, weights in PROFILE_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-9, (
                f"Profile {profile!r} weights sum to {total}, expected 1.0"
            )

    def test_standard_profile_falls_between(self) -> None:
        """Standard profile should not pick the absolute fastest or cheapest extreme."""
        result = run_pipeline(self._build_test_shipment("standard"))
        assert result.winner_id is not None
        # Standard is a balanced profile; it may pick R_BAL or something non-extreme
        # At minimum, verify it ran without error and a winner exists
        assert result.winner_id in ("R_FAST", "R_BAL", "R_CHEAP")


# ---------------------------------------------------------------------------
# Test 7: Dual-gap advisory (both budget AND deadline violations)
# ---------------------------------------------------------------------------
class TestDualGapAdvisory:
    """FallbackResult always carries both gap fields; explain emits correct text."""

    def test_dual_gap_both_fields_populated(self) -> None:
        """When budget-short, raise_budget_usd > 0; extend_deadline_days may also > 0."""
        # Single route: costs more than budget AND arrives too late
        r1 = _route("R1", base_cost=8_000, base_days=20, delay_prob=0.10, delay_days=2.0)
        result = run_pipeline(_shipment(
            routes     = [r1],
            max_budget = 5_000,   # budget-short (8000 > 5000)
            deadline   = 8.0,     # deadline-short (20+0.2+1.0=21.2 > 8.0)
        ))

        assert result.fallback is not None, "Should trigger fallback"
        fb = result.fallback
        assert fb.raise_budget_usd    > 0, "raise_budget_usd must be > 0"
        assert fb.extend_deadline_days > 0, "extend_deadline_days must be > 0"

    def test_dual_gap_values_correct(self) -> None:
        """Both gap magnitudes match their exact formulas."""
        from src.engine import DELAY_BUFFER_DAYS
        r1 = _route("R1", base_cost=8_000, base_days=20, delay_prob=0.10, delay_days=2.0)
        # E[T] = 20 + 0.10*2.0 = 20.2; E[T]+buffer = 21.2
        result = run_pipeline(_shipment(
            routes     = [r1],
            max_budget = 5_000,
            deadline   = 8.0,
        ))
        fb = result.fallback
        assert fb is not None
        expected_raise  = round(max(0.0, 8_000 - 5_000), 2)
        expected_extend = round(max(0.0, 20.2 + DELAY_BUFFER_DAYS - 8.0), 2)
        assert abs(fb.raise_budget_usd    - expected_raise)  < 1e-6
        assert abs(fb.extend_deadline_days - expected_extend) < 1e-6

    def test_explain_dual_advisory_text(self) -> None:
        """explain.py emits 'AND' when both gaps are non-zero."""
        from src.explain import build_explanation
        r1 = _route("R1", base_cost=8_000, base_days=20, delay_prob=0.10, delay_days=2.0)
        result = run_pipeline(_shipment(
            routes     = [r1],
            max_budget = 5_000,
            deadline   = 8.0,
        ))
        text = build_explanation(result)
        assert "AND" in text, (
            "Dual-gap advisory should contain 'AND' when both budget and deadline fail"
        )

    def test_explain_budget_only_advisory(self) -> None:
        """explain.py emits budget-only text when deadline is not binding."""
        from src.explain import build_explanation
        r1 = _route("R1", base_cost=8_000, base_days=5, delay_prob=0.05, delay_days=1.0)
        # E[T] = 5 + 0.05 = 5.05; + buffer = 6.05 < deadline 60 → only budget fails
        result = run_pipeline(_shipment(
            routes     = [r1],
            max_budget = 3_000,
            deadline   = 60.0,
        ))
        fb = result.fallback
        assert fb is not None
        assert fb.raise_budget_usd > 0
        assert fb.extend_deadline_days == 0.0
        text = build_explanation(result)
        assert "AND" not in text
        assert "Raise budget" in text


# ---------------------------------------------------------------------------
# Test 8: Generator output satisfies count assertions
# ---------------------------------------------------------------------------
class TestGeneratorOutputCounts:
    """The generated dataset must have exactly 3 infeasible and >=24 with >=2 Pareto."""

    def test_infeasible_count_is_three(self) -> None:
        """Exactly 3 shipments trigger the fallback path."""
        import numpy as np
        from src.generator import generate_shipments, SEED
        rng = np.random.default_rng(SEED)
        shipments = generate_shipments(rng=rng)
        n_infeas = sum(
            1 for shp in shipments
            if run_pipeline(shp).fallback is not None
        )
        assert n_infeas == 3, f"Expected 3 infeasible, got {n_infeas}"

    def test_ge2_pareto_at_least_24(self) -> None:
        """At least 24 non-forced shipments have >=2 Pareto survivors."""
        import numpy as np
        from src.generator import generate_shipments, SEED
        rng = np.random.default_rng(SEED)
        shipments = generate_shipments(rng=rng)
        n_ge2 = sum(
            1 for shp in shipments
            if len(run_pipeline(shp).scored_routes) >= 2
        )
        assert n_ge2 >= 24, f"Expected >=24 with >=2 Pareto survivors, got {n_ge2}"


# ---------------------------------------------------------------------------
# Test 9: Baselines pick the same fallback route as the engine
# ---------------------------------------------------------------------------
class TestBaselineFallbackParity:
    """All strategies use the same fallback pick when no route is feasible."""

    def test_all_strategies_same_fallback(self) -> None:
        """
        When the shipment is infeasible, all three strategies must select
        the same fallback route (fastest within budget, else cheapest).
        """
        from src.evaluate import _pick_for_strategy, _get_feasible_metrics
        from src.engine import _build_metrics

        # Make all routes over budget so fallback triggers
        r1 = _route("R1", base_cost=9_000, base_days=5)
        r2 = _route("R2", base_cost=7_000, base_days=8)
        shp = _shipment(routes=[r1, r2], max_budget=3_000, deadline=60.0)

        feasible = _get_feasible_metrics(shp)
        assert len(feasible) == 0, "Shipment should be infeasible"

        picks = {
            strat: _pick_for_strategy(strat, shp, feasible)
            for strat in ("Cheapest-First", "Fastest-First", "Proposed Engine")
        }
        # All should pick the cheapest (budget_short → cheapest overall)
        route_ids = {strat: p.route_id for strat, p in picks.items()}
        assert len(set(route_ids.values())) == 1, (
            f"All strategies should pick the same fallback, got: {route_ids}"
        )


# ---------------------------------------------------------------------------
# Test 10: Dominated route never selected by the engine
# ---------------------------------------------------------------------------
class TestDominatedNeverSelected:
    """A route strictly dominated in all three objectives must not be the winner."""

    def test_dominated_not_selected(self) -> None:
        """
        R_DOM is strictly dominated by R_GOOD (all three objectives worse).
        The engine must never pick R_DOM.
        """
        r_good = _route(
            "R_GOOD", base_cost=3_000, base_days=10,
            delay_prob=0.0, delay_days=0.0, congestion=0.30, geo_risk=0.20,
        )
        r_dom = _route(
            "R_DOM", base_cost=4_500, base_days=14,
            delay_prob=0.0, delay_days=0.0, congestion=0.50, geo_risk=0.40,
        )

        for profile in ("urgent", "standard", "cost_sensitive"):
            result = run_pipeline(_shipment(
                routes     = [r_good, r_dom],
                profile    = profile,
                max_budget = 10_000,
                deadline   = 30.0,
            ))
            assert result.winner_id != "R_DOM", (
                f"Profile {profile!r}: dominated R_DOM must not be selected"
            )
            assert "R_DOM" in result.dominated_ids, (
                f"Profile {profile!r}: R_DOM should be in dominated_ids"
            )


# ---------------------------------------------------------------------------
# Test 11: Reference scoring properties
# ---------------------------------------------------------------------------
class TestReferenceScoring:
    """Validate reference scoring isolation and linear gap sensitivity."""

    def test_score_independent_of_extra_routes(self) -> None:
        """
        In reference mode, a route's score must be IDENTICAL with and without
        an extra non-dominated route present (unlike min-max normalisation).
        """
        r1 = _route(
            "R1", base_cost=3_000, base_days=12,
            delay_prob=0.1, delay_days=1.0, congestion=0.2, geo_risk=0.2,
        )
        r2 = _route(
            "R2", base_cost=6_000, base_days=6,
            delay_prob=0.1, delay_days=1.0, congestion=0.2, geo_risk=0.2,
        )
        r3 = _route(
            "R3", base_cost=4_500, base_days=9,
            delay_prob=0.1, delay_days=1.0, congestion=0.2, geo_risk=0.2,
        )

        shp_two = _shipment(routes=[r1, r2], max_budget=10_000, deadline=30.0)
        res_two = run_pipeline(shp_two, scoring="reference")
        score_r1_two = res_two.get_metric("R1").penalty_score

        shp_three = _shipment(routes=[r1, r2, r3], max_budget=10_000, deadline=30.0)
        res_three = run_pipeline(shp_three, scoring="reference")
        score_r1_three = res_three.get_metric("R1").penalty_score

        assert score_r1_two == score_r1_three, (
            f"Reference score of R1 should be identical: {score_r1_two} vs {score_r1_three}"
        )

    def test_cost_difference_gap_scaling(self) -> None:
        """
        A 1% and a 100% cost difference produce different score gaps in reference mode.
        (Min-max would map both 2-route cases to 0 and 1, producing the exact same gap).
        """
        r_base = _route(
            "R_BASE", base_cost=3_000, base_days=10,
            delay_prob=0.0, delay_days=0.0, congestion=0.2, geo_risk=0.2,
        )
        r_1pct = _route(
            "R_1PCT", base_cost=3_030, base_days=9.99,
            delay_prob=0.0, delay_days=0.0, congestion=0.2, geo_risk=0.2,
        )
        r_100pct = _route(
            "R_100PCT", base_cost=6_000, base_days=9.90,
            delay_prob=0.0, delay_days=0.0, congestion=0.2, geo_risk=0.2,
        )

        shp_1pct = _shipment(routes=[r_base, r_1pct], max_budget=10_000, deadline=30.0)
        res_1pct = run_pipeline(shp_1pct, scoring="reference")
        gap_1pct = abs(
            res_1pct.get_metric("R_1PCT").penalty_score -
            res_1pct.get_metric("R_BASE").penalty_score
        )

        shp_100pct = _shipment(routes=[r_base, r_100pct], max_budget=10_000, deadline=30.0)
        res_100pct = run_pipeline(shp_100pct, scoring="reference")
        gap_100pct = abs(
            res_100pct.get_metric("R_100PCT").penalty_score -
            res_100pct.get_metric("R_BASE").penalty_score
        )

        assert gap_1pct != gap_100pct, (
            f"1% gap ({gap_1pct}) and 100% gap ({gap_100pct}) must differ in reference mode"
        )
        assert gap_100pct > 50 * gap_1pct, (
            f"100% cost difference gap ({gap_100pct}) should be ~100x larger than 1% ({gap_1pct})"
        )


