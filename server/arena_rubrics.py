"""
Composable reward rubrics for AdMarket Arena.

Three rubrics fire at different time horizons:
  - PerStepEngagementRubric  — every step (dense)
  - DailyPacingRubric        — every 50 steps at day boundary (medium)
  - WeeklyROASRubric         — step 350 only (sparse, dominant signal)

A fourth stub (OversightF1Rubric) is left for Plan 2 to implement.

Design rationale:
  The 5.0× weekly weight is intentionally larger than the sum of all
  per-step rewards (~350 × 0.1 = 35). This forces the agent to treat
  weekly ROAS as the primary objective and use per-step/daily rewards
  only as shaping signals — not as exploitable local maxima.

  Keeping these as separate composable classes (not one monolithic
  reward function) satisfies the OpenEnv "stand out" criterion for
  composable rubric systems and makes reward ablations trivial.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from ..campaign_state import AdvertiserCampaignState, REVENUE_PER_CLICK
except ImportError:
    from campaign_state import AdvertiserCampaignState, REVENUE_PER_CLICK


class PerStepEngagementRubric:
    """Dense per-step reward fired after every auction resolution.

    Reward decomposition:
      Won + clicked:    +(REVENUE_PER_CLICK - clearing_price)   net margin
      Won + no click:   -(clearing_price × 0.1)                 wasted spend
      Skipped:          +0.02                                    budget nudge
      Over budget bid:  -0.50                                    hard penalty
      No contest win:   treated same as regular win (still counts)
    """

    # Configurable weights kept as class attributes for easy ablation
    WASTED_SPEND_FRACTION = 0.10   # fraction of clearing price charged on no-click win
    SKIP_BONUS = 0.02              # tiny positive to avoid always-skip degeneration
    INVALID_BID_PENALTY = -0.50   # fires when agent bids over budget

    def score(
        self,
        won_auction: bool,
        clicked: bool,
        clearing_price: float,
        skipped: bool,
        over_budget: bool,
    ) -> float:
        """Compute the per-step reward scalar.

        Args:
            won_auction: True if this advertiser won the auction.
            clicked: True if the user clicked (only valid when won_auction=True).
            clearing_price: what the winner paid (0 if lost/skipped).
            skipped: True if the advertiser passed (skip=True action).
            over_budget: True if the bid would have exceeded weekly_budget.
        """
        if over_budget:
            return self.INVALID_BID_PENALTY

        if skipped:
            return self.SKIP_BONUS

        if not won_auction:
            # Lost the auction — no spend, no reward
            return 0.0

        if clicked:
            # Net margin: revenue value minus what we paid
            return round(REVENUE_PER_CLICK - clearing_price, 5)
        else:
            # Showed the ad but user didn't click — small wasted-spend charge
            return round(-(clearing_price * self.WASTED_SPEND_FRACTION), 5)


class DailyPacingRubric:
    """Medium-density reward fired once per day (every 50 steps).

    Scores two things:
      1. Pacing quality: how close was actual spend to the ideal daily target?
      2. Daily ROAS: was today's revenue-to-spend ratio on track?

    Combined score is bounded in [0, 1] then multiplied by 0.5 so a
    perfect day adds +0.50 — visible signal but dwarfed by the weekly bonus.
    """

    MAX_DAILY_BONUS = 0.50     # upper bound on a single day's bonus
    TARGET_ROAS = 2.0          # reference ROAS used for normalisation

    def score(self, state: AdvertiserCampaignState) -> float:
        """Compute the end-of-day bonus.

        Must be called *before* state.reset_day() so daily accumulators
        are still populated.
        """
        daily_target = state.daily_budget

        # Pacing score: 1.0 = perfectly on target, 0.0 = wildly over/under
        if daily_target > 0:
            pacing_score = max(
                0.0,
                1.0 - abs(state.spent_today - daily_target) / daily_target,
            )
        else:
            pacing_score = 0.0

        # ROAS score: normalised against target, capped at 1.0
        roas_score = min(1.0, state.daily_roas / self.TARGET_ROAS)

        combined = 0.5 * pacing_score + 0.5 * roas_score
        return round(self.MAX_DAILY_BONUS * combined, 5)


class WeeklyROASRubric:
    """Sparse terminal reward fired once at episode end (step 350).

    This is the dominant signal (weight 5.0). It ensures the agent
    cannot ignore multi-day planning — per-step rewards total at most
    ~350 × 0.1 = 35, while a perfect weekly ROAS adds 5.0 × 1.5 = 7.5.
    More realistically a 2.0x ROAS gives 5.0 × 1.0 = 5.0.

    Penalty structure:
      Overspend (> weekly_budget):    -2.0  (hard constraint violation)
      Underspend (< 50% of budget):   -2.0  (budget not deployed)
    """

    WEEKLY_BONUS_WEIGHT = 5.0     # multiplier on achievement fraction
    TARGET_ROAS = 2.0             # the KPI the agent is trying to hit
    ACHIEVEMENT_CAP = 1.5         # cap at 150% of target to limit reward hacking
    OVERSPEND_PENALTY = -2.0
    UNDERSPEND_PENALTY = -2.0

    def score(self, state: AdvertiserCampaignState) -> float:
        """Compute the terminal weekly bonus.

        Should be called once when done=True (step 350).
        """
        # Core ROAS achievement
        achievement = min(
            self.ACHIEVEMENT_CAP,
            state.weekly_roas / self.TARGET_ROAS,
        )
        base_reward = self.WEEKLY_BONUS_WEIGHT * achievement

        # Budget utilisation penalties
        penalties = 0.0
        if state.spent_total > state.weekly_budget:
            penalties += self.OVERSPEND_PENALTY
        elif state.spent_total < 0.5 * state.weekly_budget:
            # Hoarding budget is almost as bad as overspending
            penalties += self.UNDERSPEND_PENALTY

        return round(base_reward + penalties, 5)


class OversightF1Rubric:
    """STUB — Plan 2 fills the full implementation.

    The OversightAgent is trained separately to flag advertiser
    violations. This rubric provides:
      - Dense daily F1 signal (fires at day boundaries)
      - Sparse weekly F1 bonus (3.0× weight, fires at episode end)
      - False-positive penalty (-0.5 per FP flag)

    Plan 2 must:
      1. Import ViolationFlag and ground_truth from env state.
      2. Implement score_daily(predicted, ground_truth) -> float.
      3. Implement score_weekly(predicted, ground_truth) -> float.
      4. Wire calls into arena_env's day-boundary and episode-end hooks.
    """

    WEEKLY_BONUS_WEIGHT = 3.0
    FALSE_POSITIVE_PENALTY = -0.50

    def score_daily(
        self,
        predicted_flags: list,
        ground_truth_flags: list,
    ) -> float:
        raise NotImplementedError("Plan 2 implements OversightF1Rubric.score_daily()")

    def score_weekly(
        self,
        predicted_flags: list,
        ground_truth_flags: list,
    ) -> float:
        raise NotImplementedError("Plan 2 implements OversightF1Rubric.score_weekly()")
