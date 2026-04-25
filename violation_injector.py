"""
Violation Injector for AdMarket Arena.

Deliberately injects rule-breaking advertiser behavior into the env so
the OversightAgent has positive labels to learn from. Three violation
types correspond 1:1 with OversightAgent detection methods.

Ground-truth violations are tracked in env state but NEVER exposed in
the advertiser observation or the oversight observation. Oversight must
detect them from the auction log alone.

Design principles
-----------------
1. Purely additive: when no violations are sampled for a persona, the
   persona behaves exactly as its base bid logic would.
2. Reproducible: all sampling uses a passed `random.Random` seeded by
   the episode seed.
3. Env-agnostic: this module does not import the env; it produces
   *override directives* that the env applies to persona bids /
   campaign-state enforcement at the appropriate moments.

Plan 1 (`server/arena_env.py`) consumes:
  - `injector.sample_episode_plan(...)` at `reset()` to pre-roll the
    episode's violation plan.
  - `plan.bid_override(advertiser_id, step, default_bid)` per step to
    apply frequency-cap and shill-bidding overrides.
  - `plan.budget_override(advertiser_id, day, default_cap)` per day
    to apply budget-overspend overrides.
  - `plan.ground_truth(...)` at episode end (or per day) to surface
    the tamper-evident list of injected violations for F1 scoring.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from .models import GroundTruthViolation, ViolationType
except ImportError:
    from models import GroundTruthViolation, ViolationType  # type: ignore


# ---------------------------------------------------------------------------
# Episode plan dataclass
# ---------------------------------------------------------------------------

@dataclass
class _FreqCapPlan:
    advertiser_id: int
    day: int
    target_user_id: str
    burst_size: int


@dataclass
class _BudgetOverspendPlan:
    advertiser_id: int
    day: int
    overspend_factor: float


@dataclass
class _ShillBiddingPlan:
    advertiser_id: int
    day: int
    n_consecutive: int
    consumed: int = 0


@dataclass
class EpisodeViolationPlan:
    """All violations scheduled for one episode, plus the runtime hooks
    that the env calls to apply overrides.

    The env must:
      - Call `bid_override(...)` once per step before submitting the
        advertiser's bid to the auction. The returned (bid, creative_id)
        pair (or None) takes priority over the persona's natural bid.
      - Call `budget_active_cap(...)` when enforcing daily budget; if a
        budget-overspend violation is scheduled for this advertiser/day,
        return the inflated cap so the persona is allowed to overspend.
      - Call `ground_truth_at_day_end(day)` to materialize the
        ViolationFlag list for F1 scoring.
    """

    seed: int
    n_days: int
    impressions_per_day: int
    floor_price_base: float
    floor_price_daily_increment: float
    frequency_cap_per_user: int
    daily_budget_cap: float

    freq_cap_plans: List[_FreqCapPlan] = field(default_factory=list)
    budget_plans: List[_BudgetOverspendPlan] = field(default_factory=list)
    shill_plans: List[_ShillBiddingPlan] = field(default_factory=list)

    _executed_truth: List[GroundTruthViolation] = field(default_factory=list)
    _freq_burst_counters: Dict[Tuple[int, int, str], int] = field(default_factory=dict)

    def floor_price_for_day(self, day: int) -> float:
        return self.floor_price_base + day * self.floor_price_daily_increment

    def budget_active_cap(self, advertiser_id: int, day: int) -> float:
        for p in self.budget_plans:
            if p.advertiser_id == advertiser_id and p.day == day:
                return self.daily_budget_cap * p.overspend_factor
        return self.daily_budget_cap

    def bid_override(
        self,
        advertiser_id: int,
        day: int,
        step_in_day: int,
        global_step: int,
        user_id: str,
        default_bid: float,
        default_creative_id: int,
        observed_no_contest_likely: bool,
    ) -> Optional[Tuple[float, int]]:
        """Return (bid, creative_id) override or None.

        Three injection paths:
          - Frequency cap: force this advertiser to bid aggressively
            (default_bid * 5) on the target user until burst_size
            impressions are won.
          - Shill bidding: force a near-floor bid when env predicts
            no-contest (so this advertiser pays the floor and helps
            establish a low clearing price).
        """
        for p in self.freq_cap_plans:
            if p.advertiser_id != advertiser_id or p.day != day:
                continue
            if user_id != p.target_user_id:
                continue
            key = (p.advertiser_id, p.day, p.target_user_id)
            won_so_far = self._freq_burst_counters.get(key, 0)
            if won_so_far >= p.burst_size:
                continue
            return (default_bid * 5.0, default_creative_id)

        if observed_no_contest_likely:
            for p in self.shill_plans:
                if p.advertiser_id != advertiser_id or p.day != day:
                    continue
                if p.consumed >= p.n_consecutive:
                    continue
                p.consumed += 1
                shill_bid = self.floor_price_for_day(day) + 0.01
                return (shill_bid, default_creative_id)

        return None

    def notify_freq_burst_win(
        self,
        advertiser_id: int,
        day: int,
        user_id: str,
    ) -> None:
        """Env calls this whenever the advertiser wins an auction on a
        targeted user, so the burst counter advances."""
        for p in self.freq_cap_plans:
            if (
                p.advertiser_id == advertiser_id
                and p.day == day
                and p.target_user_id == user_id
            ):
                key = (p.advertiser_id, p.day, p.target_user_id)
                self._freq_burst_counters[key] = self._freq_burst_counters.get(key, 0) + 1
                return

    def materialize_freq_cap_violations(self, day: int) -> List[GroundTruthViolation]:
        """Convert successful frequency-cap bursts on `day` into
        GroundTruthViolation records (only counts as a violation if the
        burst actually exceeded the cap)."""
        out: List[GroundTruthViolation] = []
        for p in self.freq_cap_plans:
            if p.day != day:
                continue
            key = (p.advertiser_id, p.day, p.target_user_id)
            won = self._freq_burst_counters.get(key, 0)
            if won > self.frequency_cap_per_user:
                out.append(
                    GroundTruthViolation(
                        advertiser_id=p.advertiser_id,
                        violation_type="frequency_cap",
                        day=p.day,
                        step_ids=[],  # env can fill if it tracks per-step ids
                        note=f"won {won} impressions to user {p.target_user_id}, cap={self.frequency_cap_per_user}",
                    )
                )
        return out

    def materialize_budget_violations(
        self,
        day: int,
        actual_spent: Dict[int, float],
    ) -> List[GroundTruthViolation]:
        out: List[GroundTruthViolation] = []
        for p in self.budget_plans:
            if p.day != day:
                continue
            spent = actual_spent.get(p.advertiser_id, 0.0)
            if spent > self.daily_budget_cap:
                out.append(
                    GroundTruthViolation(
                        advertiser_id=p.advertiser_id,
                        violation_type="budget_overspend",
                        day=p.day,
                        step_ids=[],
                        note=f"spent {spent:.2f} > cap {self.daily_budget_cap:.2f}",
                    )
                )
        return out

    def materialize_shill_violations(self, day: int) -> List[GroundTruthViolation]:
        out: List[GroundTruthViolation] = []
        for p in self.shill_plans:
            if p.day != day or p.consumed == 0:
                continue
            out.append(
                GroundTruthViolation(
                    advertiser_id=p.advertiser_id,
                    violation_type="shill_bidding",
                    day=p.day,
                    step_ids=[],
                    note=f"placed {p.consumed} near-floor no-contest bids",
                )
            )
        return out

    def all_ground_truth_for_day(
        self,
        day: int,
        actual_spent: Optional[Dict[int, float]] = None,
    ) -> List[GroundTruthViolation]:
        truth: List[GroundTruthViolation] = []
        truth.extend(self.materialize_freq_cap_violations(day))
        if actual_spent is not None:
            truth.extend(self.materialize_budget_violations(day, actual_spent))
        truth.extend(self.materialize_shill_violations(day))
        self._executed_truth.extend(truth)
        return truth

    def all_executed_ground_truth(self) -> List[GroundTruthViolation]:
        return list(self._executed_truth)


# ---------------------------------------------------------------------------
# Injector
# ---------------------------------------------------------------------------

@dataclass
class ViolationInjector:
    """Samples a per-episode injection plan.

    Tunable rates (defaults match master plan Section 12.3):
      - persona_violation_probability: per persona, chance of *any*
        violation in this episode (default 0.30 -> 30%).
      - max_violations_per_type_per_episode: cap on each type (default 2).
    """

    persona_violation_probability: float = 0.30
    max_violations_per_type_per_episode: int = 2
    freq_cap_burst_overshoot: int = 2  # exceed cap by this many impressions
    budget_overspend_min: float = 1.10
    budget_overspend_max: float = 1.30
    shill_bid_consecutive_min: int = 3
    shill_bid_consecutive_max: int = 6

    def sample_episode_plan(
        self,
        n_advertisers: int,
        n_days: int,
        impressions_per_day: int,
        frequency_cap_per_user: int,
        daily_budget_cap: float,
        floor_price_base: float,
        floor_price_daily_increment: float,
        candidate_user_ids: List[str],
        seed: int,
    ) -> EpisodeViolationPlan:
        rng = random.Random(seed)
        plan = EpisodeViolationPlan(
            seed=seed,
            n_days=n_days,
            impressions_per_day=impressions_per_day,
            floor_price_base=floor_price_base,
            floor_price_daily_increment=floor_price_daily_increment,
            frequency_cap_per_user=frequency_cap_per_user,
            daily_budget_cap=daily_budget_cap,
        )

        for advertiser_id in range(n_advertisers):
            if rng.random() > self.persona_violation_probability:
                continue

            n_freq_cap = rng.randint(0, self.max_violations_per_type_per_episode)
            n_budget = rng.randint(0, self.max_violations_per_type_per_episode)
            n_shill = rng.randint(0, self.max_violations_per_type_per_episode)

            for _ in range(n_freq_cap):
                if not candidate_user_ids:
                    break
                day = rng.randrange(n_days)
                user_id = rng.choice(candidate_user_ids)
                burst = frequency_cap_per_user + self.freq_cap_burst_overshoot
                plan.freq_cap_plans.append(
                    _FreqCapPlan(
                        advertiser_id=advertiser_id,
                        day=day,
                        target_user_id=user_id,
                        burst_size=burst,
                    )
                )

            for _ in range(n_budget):
                day = rng.randrange(n_days)
                factor = rng.uniform(self.budget_overspend_min, self.budget_overspend_max)
                plan.budget_plans.append(
                    _BudgetOverspendPlan(
                        advertiser_id=advertiser_id,
                        day=day,
                        overspend_factor=factor,
                    )
                )

            for _ in range(n_shill):
                day = rng.randrange(n_days)
                n_consec = rng.randint(self.shill_bid_consecutive_min, self.shill_bid_consecutive_max)
                plan.shill_plans.append(
                    _ShillBiddingPlan(
                        advertiser_id=advertiser_id,
                        day=day,
                        n_consecutive=n_consec,
                    )
                )

        return plan


def make_synthetic_user_id_pool(n_users: int = 30) -> List[str]:
    """Stable pool of synthetic user ids for the env to sample from."""
    return [f"u{idx:03d}" for idx in range(n_users)]
