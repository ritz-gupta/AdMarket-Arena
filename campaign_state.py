"""
Per-advertiser campaign state for AdMarket Arena.

One AdvertiserCampaignState instance exists per advertiser per episode.
Tracks budget, spend, engagement outcomes, per-segment fatigue, and
the signals needed to compute objective progress for all three KPI types.

No Pydantic / OpenEnv dependency — plain dataclasses so this module
can be imported by both server code and offline analysis scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set

# Revenue value per click, used by the per-step reward formula.
# Intentionally simple: a click is worth $2.00 of simulated revenue.
# This is higher than the max floor price (≈$1.20 on day 7), making
# clicks genuinely profitable and ensuring the net margin reward fires
# positively for high-CTR impressions.
REVENUE_PER_CLICK = 2.0


@dataclass
class AdvertiserCampaignState:
    """Tracks one advertiser's budget, KPIs, and fatigue for the full episode.

    Resettable fields are zeroed at each day boundary (every 50 steps).
    Cumulative fields accumulate across all 7 days.
    """

    advertiser_id: str
    objective_type: str     # "awareness" | "conversion" | "retention"
    weekly_budget: float    # total spend cap for the 7-day campaign
    daily_budget: float     # soft daily spend target (weekly_budget / 7)

    # ----------------------------------------------------------------
    # Daily accumulators — reset each day boundary
    # ----------------------------------------------------------------
    spent_today: float = 0.0
    clicks_today: int = 0
    revenue_today: float = 0.0
    wins_today: int = 0         # auction wins today
    impressions_today: int = 0  # same as wins_today (one imp per win)

    # ----------------------------------------------------------------
    # Cumulative weekly accumulators — never reset mid-episode
    # ----------------------------------------------------------------
    spent_total: float = 0.0
    clicks_total: int = 0
    revenue_total: float = 0.0
    wins_total: int = 0
    impressions_total: int = 0

    # ----------------------------------------------------------------
    # Frequency-cap enforcement: per-user impression count today
    # ----------------------------------------------------------------
    per_user_impressions_today: Dict[str, int] = field(default_factory=dict)

    # ----------------------------------------------------------------
    # Fatigue tracking — per-segment, separate from the global fatigue
    # in the old single-agent env. Each advertiser accumulates their own
    # fatigue on each user segment independently.
    # ----------------------------------------------------------------
    per_segment_fatigue: Dict[str, float] = field(default_factory=dict)

    # ----------------------------------------------------------------
    # Objective-specific signals
    # ----------------------------------------------------------------
    # Awareness: set of unique user_ids reached this episode
    unique_users_reached: Set[str] = field(default_factory=set)

    # Retention: how many times each user clicked one of our ads
    # user_id → click count; repeat_engagements = users with count >= 2
    user_click_counts: Dict[str, int] = field(default_factory=dict)

    day_number: int = 1

    # ------------------------------------------------------------------
    # Day-boundary management
    # ------------------------------------------------------------------

    def reset_day(self) -> None:
        """Zero daily accumulators and advance the day counter.

        Called by arena_env.py at every 50-step boundary.
        Per-segment fatigue carries forward intentionally — fatigue
        built on day 3 still affects engagement on day 4.
        """
        self.spent_today = 0.0
        self.clicks_today = 0
        self.revenue_today = 0.0
        self.wins_today = 0
        self.impressions_today = 0
        self.per_user_impressions_today = {}
        self.day_number += 1

    # ------------------------------------------------------------------
    # State mutation helpers (called by arena_env after each auction step)
    # ------------------------------------------------------------------

    def record_win(self, clearing_price: float, user_id: str) -> None:
        """Deduct clearing price and increment win counters.

        Called immediately after the auction resolves in favour of this
        advertiser, before engagement is computed.
        """
        self.spent_today += clearing_price
        self.spent_total += clearing_price
        self.wins_today += 1
        self.wins_total += 1
        self.impressions_today += 1
        self.impressions_total += 1
        self.per_user_impressions_today[user_id] = (
            self.per_user_impressions_today.get(user_id, 0) + 1
        )
        self.unique_users_reached.add(user_id)

    def record_engagement(
        self,
        clicked: bool,
        user_id: str,
        segment: str,
        fatigue_increment: float = 0.06,
    ) -> None:
        """Update click counts, revenue, and per-segment fatigue.

        Called after compute_engagement() returns results for the
        winning advertiser's creative. The fatigue_increment is the
        same constant used by the old single-agent environment.
        """
        revenue = REVENUE_PER_CLICK if clicked else 0.0
        if clicked:
            self.clicks_today += 1
            self.clicks_total += 1
            self.revenue_today += revenue
            self.revenue_total += revenue
            self.user_click_counts[user_id] = (
                self.user_click_counts.get(user_id, 0) + 1
            )

        # Fatigue increment: showing an ad always builds fatigue
        current = self.per_segment_fatigue.get(segment, 0.0)
        self.per_segment_fatigue[segment] = round(
            min(1.0, current + fatigue_increment), 5
        )

    def recover_fatigue(self, segment: str, fatigue_recovery: float = 0.04) -> None:
        """Slight fatigue recovery on segments where we skip this step.

        Called by arena_env when this advertiser loses or skips an
        auction for a user in a given segment.
        """
        current = self.per_segment_fatigue.get(segment, 0.0)
        if current > 0.0:
            self.per_segment_fatigue[segment] = round(
                max(0.0, current - fatigue_recovery), 5
            )

    # ------------------------------------------------------------------
    # Derived KPI properties
    # ------------------------------------------------------------------

    @property
    def daily_roas(self) -> float:
        return self.revenue_today / max(0.01, self.spent_today)

    @property
    def weekly_roas(self) -> float:
        return self.revenue_total / max(0.01, self.spent_total)

    @property
    def repeat_engagements(self) -> int:
        """Count of users who clicked at least twice — retention signal."""
        return sum(1 for c in self.user_click_counts.values() if c >= 2)

    @property
    def budget_exhausted(self) -> bool:
        return self.spent_total >= self.weekly_budget

    @property
    def daily_budget_exhausted(self) -> bool:
        return self.spent_today >= self.daily_budget

    @property
    def objective_progress(self) -> float:
        """Normalised [0, 1] progress toward this advertiser's KPI.

        Used by the per-step reward formula inside arena_rubrics.py.
        """
        if self.objective_type == "awareness":
            # Target: reach 100 unique users over the week
            return min(1.0, len(self.unique_users_reached) / 100.0)
        elif self.objective_type == "conversion":
            if self.impressions_total == 0:
                return 0.0
            return min(1.0, self.clicks_total / max(1, self.impressions_total))
        elif self.objective_type == "retention":
            if self.impressions_total == 0:
                return 0.0
            return min(1.0, self.repeat_engagements / max(1, self.impressions_total))
        return 0.0
