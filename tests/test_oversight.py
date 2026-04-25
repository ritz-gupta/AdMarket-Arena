"""
Plan 2 — OversightAgent + violation_injector + F1 rubric tests.

Coverage targets:
  - F1 scorer correctness (toy cases, edge cases for empty sets).
  - HeuristicOversightAgent detects each of the 3 violation types.
  - LLMOversightAgent JSON parsing is robust to noisy completions.
  - ViolationInjector samples deterministically and produces non-empty
    plans at the configured rate.
  - OversightF1Rubric computes the documented daily and weekly rewards
    with the false-positive penalty applied.
  - End-to-end smoke: synthetic episode -> heuristic flags -> F1 > 0.

Run:
    pytest tests/test_oversight.py -v
"""

from __future__ import annotations

import math
import statistics

import pytest

from meta_ad_optimizer.models import (
    AuctionRecord,
    CampaignStateSummary,
    GroundTruthViolation,
    OversightObservation,
    ViolationFlag,
)
from meta_ad_optimizer.oversight import (
    HeuristicOversightAgent,
    LLMOversightAgent,
    parse_llm_flags,
    score_episode,
    score_flags,
)
from meta_ad_optimizer.server.arena_rubrics import (
    OversightF1Rubric,
    build_arena_rubrics,
)
from meta_ad_optimizer.violation_injector import (
    ViolationInjector,
    make_synthetic_user_id_pool,
)


# ---------------------------------------------------------------------------
# F1 scorer
# ---------------------------------------------------------------------------

class TestF1Scorer:

    def test_perfect_match(self):
        pred = [
            ViolationFlag(advertiser_id=0, violation_type="frequency_cap"),
            ViolationFlag(advertiser_id=1, violation_type="budget_overspend"),
        ]
        truth = [
            GroundTruthViolation(advertiser_id=0, violation_type="frequency_cap", day=0),
            GroundTruthViolation(advertiser_id=1, violation_type="budget_overspend", day=0),
        ]
        result = score_flags(pred, truth)
        assert result.f1 == 1.0
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.true_positives == 2
        assert result.false_positives == 0
        assert result.false_negatives == 0

    def test_complete_miss(self):
        pred = [ViolationFlag(advertiser_id=2, violation_type="shill_bidding")]
        truth = [
            GroundTruthViolation(advertiser_id=0, violation_type="frequency_cap", day=0),
        ]
        result = score_flags(pred, truth)
        assert result.f1 == 0.0
        assert result.true_positives == 0
        assert result.false_positives == 1
        assert result.false_negatives == 1

    def test_empty_both(self):
        result = score_flags([], [])
        assert result.f1 == 1.0
        assert result.precision == 1.0
        assert result.recall == 1.0

    def test_no_truth_some_pred(self):
        pred = [ViolationFlag(advertiser_id=0, violation_type="frequency_cap")]
        result = score_flags(pred, [])
        assert result.f1 == 0.0
        assert result.false_positives == 1

    def test_some_truth_no_pred(self):
        truth = [GroundTruthViolation(advertiser_id=0, violation_type="frequency_cap", day=0)]
        result = score_flags([], truth)
        assert result.f1 == 0.0
        assert result.false_negatives == 1

    def test_partial_match(self):
        pred = [
            ViolationFlag(advertiser_id=0, violation_type="frequency_cap"),
            ViolationFlag(advertiser_id=1, violation_type="frequency_cap"),
            ViolationFlag(advertiser_id=2, violation_type="shill_bidding"),
        ]
        truth = [
            GroundTruthViolation(advertiser_id=0, violation_type="frequency_cap", day=0),
            GroundTruthViolation(advertiser_id=3, violation_type="budget_overspend", day=0),
        ]
        result = score_flags(pred, truth)
        # tp=1 (advertiser 0), fp=2 (1 freq_cap, 2 shill), fn=1 (advertiser 3 budget)
        assert result.true_positives == 1
        assert result.false_positives == 2
        assert result.false_negatives == 1
        assert result.precision == 1 / 3
        assert result.recall == 0.5
        assert math.isclose(result.f1, 2 * (1/3) * 0.5 / ((1/3) + 0.5), rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Heuristic oversight detection
# ---------------------------------------------------------------------------

def _make_obs(
    auction_log: list[AuctionRecord],
    campaign_states: list[CampaignStateSummary],
    floor_price: float = 0.5,
    cap: int = 3,
    day: int = 0,
) -> OversightObservation:
    return OversightObservation(
        day=day,
        auction_log=auction_log,
        campaign_states=campaign_states,
        floor_price=floor_price,
        frequency_cap_per_user=cap,
    )


class TestHeuristicOversight:

    def test_detects_frequency_cap_violation(self):
        # Advertiser 0 wins 4 times against user u001 on day 0; cap is 3.
        log = [
            AuctionRecord(
                step=i, day=0, step_in_day=i, user_id="u001", user_segment="seg",
                advertiser_id=0, bid=1.5, won=True, clearing_price=1.0,
                floor_price=0.5, no_contest=False,
            )
            for i in range(4)
        ]
        obs = _make_obs(log, campaign_states=[], cap=3)
        agent = HeuristicOversightAgent()
        flags = agent.flag_day(obs)
        assert any(f.advertiser_id == 0 and f.violation_type == "frequency_cap" for f in flags)

    def test_does_not_flag_within_cap(self):
        log = [
            AuctionRecord(
                step=i, day=0, step_in_day=i, user_id="u001", user_segment="seg",
                advertiser_id=0, bid=1.0, won=True, clearing_price=0.7,
                floor_price=0.5, no_contest=False,
            )
            for i in range(2)
        ]
        obs = _make_obs(log, campaign_states=[], cap=3)
        agent = HeuristicOversightAgent()
        flags = agent.flag_day(obs)
        assert not any(f.violation_type == "frequency_cap" for f in flags)

    def test_detects_budget_overspend(self):
        states = [
            CampaignStateSummary(
                advertiser_id=1, advertiser_name="Big", spent_today=120.0,
                daily_budget_cap=100.0, spent_total=120.0,
                weekly_budget_cap=700.0, impressions_today=10, clicks_today=2,
            ),
            CampaignStateSummary(
                advertiser_id=2, advertiser_name="Small", spent_today=80.0,
                daily_budget_cap=100.0, spent_total=80.0,
                weekly_budget_cap=700.0, impressions_today=8, clicks_today=1,
            ),
        ]
        obs = _make_obs(auction_log=[], campaign_states=states)
        flags = HeuristicOversightAgent().flag_day(obs)
        assert any(f.advertiser_id == 1 and f.violation_type == "budget_overspend" for f in flags)
        assert not any(f.advertiser_id == 2 and f.violation_type == "budget_overspend" for f in flags)

    def test_detects_shill_bidding(self):
        # Advertiser 3 places 4 near-floor bids in no-contest auctions.
        log = []
        for i in range(4):
            log.append(
                AuctionRecord(
                    step=i, day=0, step_in_day=i, user_id=f"u{i:03d}", user_segment="seg",
                    advertiser_id=3, bid=0.51, won=False, clearing_price=0.0,
                    floor_price=0.50, no_contest=True,
                )
            )
        obs = _make_obs(auction_log=log, campaign_states=[])
        flags = HeuristicOversightAgent().flag_day(obs)
        assert any(f.advertiser_id == 3 and f.violation_type == "shill_bidding" for f in flags)

    def test_does_not_flag_high_bids_in_no_contest(self):
        # Far-from-floor bids in no-contest aren't shill suspects.
        log = [
            AuctionRecord(
                step=i, day=0, step_in_day=i, user_id=f"u{i}", user_segment="seg",
                advertiser_id=3, bid=2.0, won=False, clearing_price=0.0,
                floor_price=0.5, no_contest=True,
            )
            for i in range(4)
        ]
        obs = _make_obs(auction_log=log, campaign_states=[])
        flags = HeuristicOversightAgent().flag_day(obs)
        assert not any(f.violation_type == "shill_bidding" for f in flags)


# ---------------------------------------------------------------------------
# LLM JSON parser robustness
# ---------------------------------------------------------------------------

class TestLLMJsonParser:

    def test_valid_clean_json(self):
        raw = '{"flags": [{"advertiser_id": 0, "violation_type": "frequency_cap", "confidence": 0.9}]}'
        flags = parse_llm_flags(raw)
        assert len(flags) == 1
        assert flags[0].advertiser_id == 0
        assert flags[0].violation_type == "frequency_cap"

    def test_extracts_from_markdown_wrapper(self):
        raw = "Sure, here are the violations:\n```json\n{\"flags\": [{\"advertiser_id\": 2, \"violation_type\": \"shill_bidding\"}]}\n```"
        flags = parse_llm_flags(raw)
        assert len(flags) == 1
        assert flags[0].violation_type == "shill_bidding"

    def test_invalid_json_returns_empty(self):
        raw = "I think advertiser 0 is shilling"
        flags = parse_llm_flags(raw)
        assert flags == []

    def test_drops_unknown_violation_type(self):
        raw = '{"flags": [{"advertiser_id": 0, "violation_type": "click_fraud", "confidence": 0.5}]}'
        flags = parse_llm_flags(raw)
        assert flags == []

    def test_dedupes_same_pair(self):
        raw = (
            '{"flags": ['
            '{"advertiser_id": 0, "violation_type": "frequency_cap"},'
            '{"advertiser_id": 0, "violation_type": "frequency_cap", "confidence": 0.4}'
            ']}'
        )
        flags = parse_llm_flags(raw)
        assert len(flags) == 1

    def test_clamps_confidence(self):
        raw = '{"flags": [{"advertiser_id": 0, "violation_type": "frequency_cap", "confidence": 5.0}]}'
        flags = parse_llm_flags(raw)
        assert 0.01 <= flags[0].confidence <= 0.99

    def test_empty_flags_list(self):
        flags = parse_llm_flags('{"flags": []}')
        assert flags == []


# ---------------------------------------------------------------------------
# Violation injector
# ---------------------------------------------------------------------------

class TestViolationInjector:

    def test_deterministic_for_same_seed(self):
        injector = ViolationInjector()
        pool = make_synthetic_user_id_pool(20)
        plan_a = injector.sample_episode_plan(
            n_advertisers=5, n_days=7, impressions_per_day=50,
            frequency_cap_per_user=3, daily_budget_cap=143.0,
            floor_price_base=0.5, floor_price_daily_increment=0.1,
            candidate_user_ids=pool, seed=123,
        )
        plan_b = injector.sample_episode_plan(
            n_advertisers=5, n_days=7, impressions_per_day=50,
            frequency_cap_per_user=3, daily_budget_cap=143.0,
            floor_price_base=0.5, floor_price_daily_increment=0.1,
            candidate_user_ids=pool, seed=123,
        )
        assert len(plan_a.freq_cap_plans) == len(plan_b.freq_cap_plans)
        assert len(plan_a.budget_plans) == len(plan_b.budget_plans)
        assert len(plan_a.shill_plans) == len(plan_b.shill_plans)

    def test_high_probability_yields_violations(self):
        injector = ViolationInjector(persona_violation_probability=1.0)
        pool = make_synthetic_user_id_pool(20)
        plan = injector.sample_episode_plan(
            n_advertisers=5, n_days=7, impressions_per_day=50,
            frequency_cap_per_user=3, daily_budget_cap=143.0,
            floor_price_base=0.5, floor_price_daily_increment=0.1,
            candidate_user_ids=pool, seed=42,
        )
        # With p=1.0 every persona has a chance to inject; expect at least 1 across all 3 types.
        total = len(plan.freq_cap_plans) + len(plan.budget_plans) + len(plan.shill_plans)
        assert total >= 1

    def test_zero_probability_yields_no_violations(self):
        injector = ViolationInjector(persona_violation_probability=0.0)
        pool = make_synthetic_user_id_pool(20)
        plan = injector.sample_episode_plan(
            n_advertisers=5, n_days=7, impressions_per_day=50,
            frequency_cap_per_user=3, daily_budget_cap=143.0,
            floor_price_base=0.5, floor_price_daily_increment=0.1,
            candidate_user_ids=pool, seed=42,
        )
        assert plan.freq_cap_plans == []
        assert plan.budget_plans == []
        assert plan.shill_plans == []

    def test_floor_price_progression(self):
        injector = ViolationInjector()
        pool = make_synthetic_user_id_pool(20)
        plan = injector.sample_episode_plan(
            n_advertisers=5, n_days=7, impressions_per_day=50,
            frequency_cap_per_user=3, daily_budget_cap=143.0,
            floor_price_base=0.5, floor_price_daily_increment=0.1,
            candidate_user_ids=pool, seed=0,
        )
        assert plan.floor_price_for_day(0) == pytest.approx(0.5)
        assert plan.floor_price_for_day(6) == pytest.approx(0.5 + 6 * 0.1)


# ---------------------------------------------------------------------------
# OversightF1Rubric
# ---------------------------------------------------------------------------

class TestOversightF1Rubric:

    def test_perfect_day_reward_is_daily_weight(self):
        rubric = OversightF1Rubric(daily_weight=1.0, weekly_weight=3.0, fp_penalty=0.5)
        pred = [ViolationFlag(advertiser_id=0, violation_type="frequency_cap")]
        truth = [GroundTruthViolation(advertiser_id=0, violation_type="frequency_cap", day=0)]
        result = rubric.score_day(pred, truth)
        assert result["reward"] == pytest.approx(1.0)  # f1=1, fp=0

    def test_false_positive_penalty(self):
        rubric = OversightF1Rubric(daily_weight=1.0, weekly_weight=3.0, fp_penalty=0.5)
        pred = [
            ViolationFlag(advertiser_id=0, violation_type="frequency_cap"),
            ViolationFlag(advertiser_id=99, violation_type="shill_bidding"),  # FP
        ]
        truth = [GroundTruthViolation(advertiser_id=0, violation_type="frequency_cap", day=0)]
        result = rubric.score_day(pred, truth)
        # f1 = 2 * 0.5 * 1 / 1.5 = 0.667; reward = 0.667 - 0.5 = 0.167
        assert result["reward"] == pytest.approx(2 / 3 - 0.5, abs=1e-3)

    def test_weekly_weight_dominates_daily(self):
        rubric = OversightF1Rubric(daily_weight=1.0, weekly_weight=3.0, fp_penalty=0.5)
        pred = [ViolationFlag(advertiser_id=0, violation_type="frequency_cap")]
        truth = [GroundTruthViolation(advertiser_id=0, violation_type="frequency_cap", day=0)]
        weekly_result = rubric.score_week(pred, truth)
        assert weekly_result["reward"] == pytest.approx(3.0)

    def test_build_arena_rubrics_returns_oversight(self):
        rubrics = build_arena_rubrics()
        assert "oversight_f1" in rubrics
        rubrics_subset = build_arena_rubrics(enabled=["oversight_f1"])
        assert set(rubrics_subset.keys()) == {"oversight_f1"}


# ---------------------------------------------------------------------------
# LLMOversightAgent integration
# ---------------------------------------------------------------------------

class TestLLMOversightAgent:

    def test_completion_fn_returning_valid_json(self):
        def fake_completion(system: str, user: str) -> str:
            return '{"flags": [{"advertiser_id": 1, "violation_type": "budget_overspend", "confidence": 0.8}]}'
        agent = LLMOversightAgent(completion_fn=fake_completion)
        obs = _make_obs(auction_log=[], campaign_states=[])
        flags = agent.flag_day(obs)
        assert len(flags) == 1
        assert flags[0].advertiser_id == 1

    def test_completion_fn_raising_returns_empty(self):
        def broken_completion(system: str, user: str) -> str:
            raise RuntimeError("provider down")
        agent = LLMOversightAgent(completion_fn=broken_completion)
        obs = _make_obs(auction_log=[], campaign_states=[])
        flags = agent.flag_day(obs)
        assert flags == []

    def test_completion_fn_returning_garbage_returns_empty(self):
        def garbage_completion(system: str, user: str) -> str:
            return "I think advertiser 0 maybe broke a rule but I'm not sure"
        agent = LLMOversightAgent(completion_fn=garbage_completion)
        obs = _make_obs(auction_log=[], campaign_states=[])
        flags = agent.flag_day(obs)
        assert flags == []


# ---------------------------------------------------------------------------
# End-to-end smoke (Plan 2 acceptance criterion: heuristic F1 >= 0.7 on injected violations)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 7, 42, 99])
def test_smoke_heuristic_f1_on_synthetic_episode(seed: int):
    """End-to-end: synthetic episode -> heuristic flags -> F1 score.

    Acceptance: across seeds, mean heuristic F1 >= 0.5 (looser than the
    plan target 0.7 because the synthetic generator has noise — real
    env trajectories will be cleaner). Confirms the pipeline holds.
    """
    from scripts.collect_oversight_trajectories import _synthetic_episode

    import random as _r
    rng = _r.Random(seed)
    rows = _synthetic_episode(
        rng=rng,
        n_advertisers=5,
        n_days=3,
        impressions_per_day=20,
        frequency_cap=3,
        daily_budget=50.0,
        floor_price_base=0.5,
        floor_price_daily_increment=0.1,
        injection_probability=1.0,
    )

    per_day_truth = {row["day"]: [GroundTruthViolation.model_validate(t) for t in row["ground_truth"]] for row in rows}
    per_day_pred = {row["day"]: [ViolationFlag.model_validate(p) for p in row["heuristic_predictions"]] for row in rows}

    n_truth = sum(len(v) for v in per_day_truth.values())
    if n_truth == 0:
        pytest.skip("no ground-truth violations for this seed")

    score = score_episode(per_day_pred, per_day_truth)
    weekly_f1 = score["weekly"]["f1"]
    assert weekly_f1 >= 0.5, f"heuristic weekly F1 {weekly_f1:.3f} below 0.5 floor for seed={seed}"


def test_smoke_heuristic_f1_aggregate_across_seeds():
    """Aggregate test: heuristic F1 averaged across many episodes ≥ 0.7."""
    from scripts.collect_oversight_trajectories import _synthetic_episode
    import random as _r

    f1s: list[float] = []
    for seed in range(20):
        rng = _r.Random(seed)
        rows = _synthetic_episode(
            rng=rng, n_advertisers=5, n_days=3, impressions_per_day=20,
            frequency_cap=3, daily_budget=50.0,
            floor_price_base=0.5, floor_price_daily_increment=0.1,
            injection_probability=1.0,
        )
        per_day_truth = {row["day"]: [GroundTruthViolation.model_validate(t) for t in row["ground_truth"]] for row in rows}
        per_day_pred = {row["day"]: [ViolationFlag.model_validate(p) for p in row["heuristic_predictions"]] for row in rows}
        if sum(len(v) for v in per_day_truth.values()) == 0:
            continue
        f1s.append(score_episode(per_day_pred, per_day_truth)["weekly"]["f1"])
    assert f1s, "no episodes produced ground-truth violations"
    mean_f1 = statistics.mean(f1s)
    assert mean_f1 >= 0.7, f"mean heuristic F1 {mean_f1:.3f} below Plan 2 acceptance floor 0.7"
