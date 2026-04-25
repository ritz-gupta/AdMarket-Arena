"""
Composable reward rubrics for AdMarket Arena.

Four rubrics, each fired at a different time horizon and consumed by
``server/arena_env.py`` (advertiser reward) or by Plan 2's
``train_oversight.ipynb`` (oversight reward):

  - ``PerStepEngagementRubric`` — every step (dense advertiser reward).
  - ``DailyPacingRubric``       — every day boundary (medium advertiser).
  - ``WeeklyROASRubric``        — episode end (sparse advertiser).
  - ``OversightF1Rubric``       — daily + weekly oversight F1, with FP
                                   penalty (Fleet AI bonus, master 12.4).

Plan 1 owns the first three; Plan 2 owns the fourth. Keeping these as
separate composable classes (not one monolithic reward function)
satisfies the OpenEnv "stand out" criterion for composable rubric
systems and makes reward ablations trivial via
``build_arena_rubrics(enabled=[...])``.

Design rationale for the advertiser weights:
  The 5.0× weekly weight is intentionally larger than the sum of all
  per-step rewards (~350 × 0.1 = 35). This forces the agent to treat
  weekly ROAS as the primary objective and use per-step/daily rewards
  only as shaping signals — not as exploitable local maxima.

Design rationale for the oversight weights:
  3.0 weekly bonus dominates the ~7 daily F1 signals so end-of-week
  credit assignment is non-trivial; -0.5 per FP keeps precision and
  recall in tension so the agent can't just flag everyone.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol

try:
    from ..campaign_state import AdvertiserCampaignState, REVENUE_PER_CLICK
    from ..models import GroundTruthViolation, ViolationFlag
    from ..oversight import score_flags
except ImportError:
    from campaign_state import AdvertiserCampaignState, REVENUE_PER_CLICK  # type: ignore
    from models import GroundTruthViolation, ViolationFlag  # type: ignore
    from oversight import score_flags  # type: ignore


# ---------------------------------------------------------------------------
# Rubric protocol
# ---------------------------------------------------------------------------

class ArenaRubric(Protocol):
    """All arena rubrics expose ``score(...) -> float`` and a ``name`` attr.

    Each rubric is independent, testable in isolation, and can be
    enabled/disabled per ``reset(enabled_rubrics=...)`` for ablations.
    """

    name: str

    def score(self, *args: Any, **kwargs: Any) -> float:
        ...


# ---------------------------------------------------------------------------
# Plan 1 — advertiser rubrics (full implementations)
# ---------------------------------------------------------------------------

class PerStepEngagementRubric:
    """Dense per-step reward fired after every auction resolution.

    Reward decomposition:
      Won + clicked:    +(REVENUE_PER_CLICK - clearing_price)   net margin
      Won + no click:   -(clearing_price × 0.1)                 wasted spend
      Skipped:          +0.02                                    budget nudge
      Over budget bid:  -0.50                                    hard penalty
      No contest win:   treated same as regular win (still counts)
    """

    name = "per_step_engagement"

    WASTED_SPEND_FRACTION = 0.10
    SKIP_BONUS = 0.02
    INVALID_BID_PENALTY = -0.50

    def score(
        self,
        won_auction: bool = False,
        clicked: bool = False,
        clearing_price: float = 0.0,
        skipped: bool = False,
        over_budget: bool = False,
        **_: Any,
    ) -> float:
        if over_budget:
            return self.INVALID_BID_PENALTY
        if skipped:
            return self.SKIP_BONUS
        if not won_auction:
            return 0.0
        if clicked:
            return round(REVENUE_PER_CLICK - clearing_price, 5)
        return round(-(clearing_price * self.WASTED_SPEND_FRACTION), 5)


class DailyPacingRubric:
    """Medium-density reward fired once per day.

    Scores pacing-quality (was actual spend close to ideal daily target?)
    and daily ROAS, blended 50/50, then capped at MAX_DAILY_BONUS.
    """

    name = "daily_pacing"

    MAX_DAILY_BONUS = 0.50
    TARGET_ROAS = 2.0

    def score(self, state: AdvertiserCampaignState, **_: Any) -> float:
        daily_target = state.daily_budget
        if daily_target > 0:
            pacing_score = max(
                0.0,
                1.0 - abs(state.spent_today - daily_target) / daily_target,
            )
        else:
            pacing_score = 0.0
        roas_score = min(1.0, state.daily_roas / self.TARGET_ROAS)
        combined = 0.5 * pacing_score + 0.5 * roas_score
        return round(self.MAX_DAILY_BONUS * combined, 5)


class WeeklyROASRubric:
    """Sparse terminal reward fired once at episode end (master Section 4).

    The dominant signal (weight 5.0). At a 2.0× target ROAS, a perfectly
    paced episode adds ~5.0; max ~7.5 at 1.5× cap. Penalties handle the
    two failure modes:
      - Overspend (> weekly_budget):    -2.0
      - Underspend (< 50% of budget):   -2.0
    """

    name = "weekly_roas"

    WEEKLY_BONUS_WEIGHT = 5.0
    TARGET_ROAS = 2.0
    ACHIEVEMENT_CAP = 1.5
    OVERSPEND_PENALTY = -2.0
    UNDERSPEND_PENALTY = -2.0

    def score(self, state: AdvertiserCampaignState, **_: Any) -> float:
        achievement = min(
            self.ACHIEVEMENT_CAP,
            state.weekly_roas / self.TARGET_ROAS,
        )
        base_reward = self.WEEKLY_BONUS_WEIGHT * achievement
        penalties = 0.0
        if state.spent_total > state.weekly_budget:
            penalties += self.OVERSPEND_PENALTY
        elif state.spent_total < 0.5 * state.weekly_budget:
            penalties += self.UNDERSPEND_PENALTY
        return round(base_reward + penalties, 5)


# ---------------------------------------------------------------------------
# Plan 2 — Oversight F1 reward (Fleet AI bonus)
# ---------------------------------------------------------------------------

class OversightF1Rubric:
    """Reward signal for OversightAgent training (Fleet AI bonus).

    Fires at the end of each day with the daily F1 component, and at
    the end of the week with a weekly F1 bonus. False positives carry
    a fixed penalty so the rubric cannot be gamed by flagging
    everyone (precision matters as much as recall).

    Design (master Section 12.4):
        daily F1   -> +1.0 * f1_today
        weekly F1  -> +3.0 * f1_week  (sparse, fires at episode end)
        FP penalty -> -0.5 per false positive
    """

    name = "oversight_f1"

    # Plan 1 module-level constants (kept for back-compat with
    # arena_env.py's reward-summing code).
    WEEKLY_BONUS_WEIGHT = 3.0
    FALSE_POSITIVE_PENALTY = -0.50

    def __init__(
        self,
        daily_weight: float = 1.0,
        weekly_weight: float = 3.0,
        fp_penalty: float = 0.5,
    ) -> None:
        self.daily_weight = daily_weight
        self.weekly_weight = weekly_weight
        self.fp_penalty = fp_penalty

    # --- Plan 2 method names ---

    def score_day(
        self,
        predicted: List[ViolationFlag],
        ground_truth: List[GroundTruthViolation],
    ) -> Dict[str, float]:
        f1 = score_flags(predicted, ground_truth)
        reward = self.daily_weight * f1.f1 - self.fp_penalty * f1.false_positives
        return {
            "reward": reward,
            "f1": f1.f1,
            "precision": f1.precision,
            "recall": f1.recall,
            "true_positives": f1.true_positives,
            "false_positives": f1.false_positives,
            "false_negatives": f1.false_negatives,
        }

    def score_week(
        self,
        predicted_all: List[ViolationFlag],
        ground_truth_all: List[GroundTruthViolation],
    ) -> Dict[str, float]:
        f1 = score_flags(predicted_all, ground_truth_all)
        reward = self.weekly_weight * f1.f1 - self.fp_penalty * f1.false_positives
        return {
            "reward": reward,
            "f1": f1.f1,
            "precision": f1.precision,
            "recall": f1.recall,
            "true_positives": f1.true_positives,
            "false_positives": f1.false_positives,
            "false_negatives": f1.false_negatives,
        }

    # --- Plan 1 method-name aliases (back-compat for arena_env hooks) ---

    def score_daily(
        self,
        predicted_flags: List[ViolationFlag],
        ground_truth_flags: List[GroundTruthViolation],
    ) -> float:
        return self.score_day(predicted_flags, ground_truth_flags)["reward"]

    def score_weekly(
        self,
        predicted_flags: List[ViolationFlag],
        ground_truth_flags: List[GroundTruthViolation],
    ) -> float:
        return self.score_week(predicted_flags, ground_truth_flags)["reward"]

    # Generic Protocol-compatible entry point. Decides between day or
    # week scoring based on whether ``kind="week"`` is passed.
    def score(
        self,
        predicted: List[ViolationFlag],
        ground_truth: List[GroundTruthViolation],
        kind: str = "day",
    ) -> float:
        if kind == "week":
            return self.score_week(predicted, ground_truth)["reward"]
        return self.score_day(predicted, ground_truth)["reward"]


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------

def build_arena_rubrics(
    enabled: Optional[List[str]] = None,
) -> Dict[str, ArenaRubric]:
    """Factory used by ``arena_env.py``. Returns a dict keyed by rubric
    name. If ``enabled`` is None, returns all four; otherwise filters to
    the named subset (used for ablation experiments).
    """
    all_rubrics: Dict[str, ArenaRubric] = {
        "per_step_engagement": PerStepEngagementRubric(),
        "daily_pacing": DailyPacingRubric(),
        "weekly_roas": WeeklyROASRubric(),
        "oversight_f1": OversightF1Rubric(),
    }
    if enabled is None:
        return all_rubrics
    return {name: rubric for name, rubric in all_rubrics.items() if name in enabled}
