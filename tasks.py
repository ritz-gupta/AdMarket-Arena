"""
Task definitions and grader functions for the Meta Ad Optimizer.

Three Round 1 difficulty tiers (creative_matcher / placement_optimizer /
campaign_optimizer) plus three AdMarket Arena tiers (arena_easy /
arena_medium / arena_hard). The arena tiers follow master plan
Section 3.1.1 verbatim and are consumed by:

  - ``server/arena_env.py`` (Plan 1) — reads ``initial_budget``,
    ``daily_budget_cap``, ``total_steps``, ``fatigue_increment``,
    ``fatigue_recovery``, ``frequency_cap_per_user``.
  - ``scripts/advertiser_eval.py`` + ``train_grpo.ipynb`` (Plan 3) —
    read ``weekly_budget``, ``daily_budget``, ``steps_per_episode``,
    ``target_weekly_roas``.

To keep both consumers happy, ``ArenaTaskConfig`` carries Plan 1's field
names as canonical and exposes Plan 3's names as ``@property`` aliases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from .models import AdState
except ImportError:
    from models import AdState


@dataclass(frozen=True)
class TaskConfig:
    """Immutable configuration for a single Round 1 task difficulty level."""

    name: str
    steps_per_episode: int
    creatives_per_episode: int
    platforms: List[str]
    allow_skip: bool              # whether show_ad=False is permitted
    fatigue_enabled: bool
    fatigue_increment: float
    fatigue_recovery: float
    surface_transitions: bool     # whether user drifts between surfaces

    fixed_platform: Optional[str] = None
    fixed_surface: Optional[str] = None
    fixed_format: Optional[str] = None


TASKS: Dict[str, TaskConfig] = {
    "creative_matcher": TaskConfig(
        name="creative_matcher",
        steps_per_episode=10,
        creatives_per_episode=4,
        platforms=["instagram"],
        allow_skip=False,
        fatigue_enabled=False,
        fatigue_increment=0.0,
        fatigue_recovery=0.0,
        surface_transitions=False,
        fixed_platform="instagram",
        fixed_surface="feed",
        fixed_format="image",
    ),
    "placement_optimizer": TaskConfig(
        name="placement_optimizer",
        steps_per_episode=15,
        creatives_per_episode=8,
        platforms=["instagram"],
        allow_skip=True,
        fatigue_enabled=True,
        fatigue_increment=0.03,
        fatigue_recovery=0.05,
        surface_transitions=False,
        fixed_platform="instagram",
    ),
    "campaign_optimizer": TaskConfig(
        name="campaign_optimizer",
        steps_per_episode=20,
        creatives_per_episode=12,
        platforms=["instagram", "facebook"],
        allow_skip=True,
        fatigue_enabled=True,
        fatigue_increment=0.06,
        fatigue_recovery=0.04,
        surface_transitions=True,
    ),
}

DEFAULT_TASK = "campaign_optimizer"


# ---------------------------------------------------------------------------
# AdMarket Arena (Round 2) — multi-advertiser auction tasks
# ---------------------------------------------------------------------------
#
# Per master plan Section 3.1.1. Curriculum scheduler
# (``curriculum_scheduler.py``) promotes advertiser training
# arena_easy -> arena_medium -> arena_hard when mean episode reward
# > 0.30 for 10 consecutive rollouts. arena_easy doubles as the fast
# smoke-test target (~2 min/episode on T4 vs ~5 min for arena_hard) so
# the GRPO pipeline can be validated in 30 min before committing 4 hours
# to arena_hard training.

@dataclass(frozen=True)
class ArenaTaskConfig:
    """Immutable configuration for one AdMarket Arena difficulty tier.

    Field naming follows Plan 1 (``initial_budget``, ``daily_budget_cap``,
    ``total_steps``) because ``server/arena_env.py`` constructs the env
    against these names. Plan 3 modules access the same values through
    aliased ``@property`` accessors (``weekly_budget``, ``daily_budget``,
    ``steps_per_episode``).
    """

    name: str
    days: int                                  # episode length in days (3 / 5 / 7)
    impressions_per_day: int                   # auction slots per day (20 / 30 / 50)
    n_personas: int                            # number of scripted PersonaBot competitors
    initial_budget: float                      # total weekly spend cap for trained agent
    daily_budget_cap: float                    # soft daily budget (reset each day)
    floor_price_base: float = 0.50             # base floor price on day 1
    floor_price_daily_increment: float = 0.10  # floor rises by this each day
    frequency_cap_per_user: int = 3            # max wins per user per day before exclusion
    fatigue_increment: float = 0.06            # per-impression fatigue growth
    fatigue_recovery: float = 0.04             # per-skipped-slot fatigue recovery
    target_weekly_roas: float = 2.0            # KPI used by reward + Plan 3 eval
    persona_jitter: bool = True                # per-episode trait jitter on opponents

    @property
    def total_steps(self) -> int:
        return self.days * self.impressions_per_day

    # --- Plan 3 aliases (keep both naming conventions working) ---

    @property
    def steps_per_episode(self) -> int:
        return self.total_steps

    @property
    def weekly_budget(self) -> float:
        return self.initial_budget

    @property
    def daily_budget(self) -> float:
        return self.daily_budget_cap


ARENA_TASKS: Dict[str, ArenaTaskConfig] = {
    "arena_easy": ArenaTaskConfig(
        name="arena_easy",
        days=3,
        impressions_per_day=20,
        n_personas=3,
        initial_budget=300.0,
        daily_budget_cap=100.0,
        floor_price_base=0.0,
        floor_price_daily_increment=0.0,
        frequency_cap_per_user=999,
        fatigue_increment=0.06,
        fatigue_recovery=0.04,
        target_weekly_roas=1.5,
        persona_jitter=False,
    ),
    "arena_medium": ArenaTaskConfig(
        name="arena_medium",
        days=5,
        impressions_per_day=30,
        n_personas=4,
        initial_budget=500.0,
        daily_budget_cap=100.0,
        floor_price_base=0.25,
        floor_price_daily_increment=0.05,
        frequency_cap_per_user=5,
        fatigue_increment=0.06,
        fatigue_recovery=0.04,
        target_weekly_roas=1.75,
        persona_jitter=True,
    ),
    "arena_hard": ArenaTaskConfig(
        name="arena_hard",
        days=7,
        impressions_per_day=50,
        n_personas=5,
        initial_budget=1000.0,
        daily_budget_cap=143.0,
        floor_price_base=0.50,
        floor_price_daily_increment=0.10,
        frequency_cap_per_user=3,
        fatigue_increment=0.06,
        fatigue_recovery=0.04,
        target_weekly_roas=2.0,
        persona_jitter=True,
    ),
}

DEFAULT_ARENA_TASK = "arena_easy"


# ---------------------------------------------------------------------------
# Grader functions  (each returns 0.0 – 1.0)
# ---------------------------------------------------------------------------

_SCORE_EPS = 1e-6  # scores must be strictly in (0, 1)


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def _clamp(score: float) -> float:
    """Clamp to strictly open interval (0, 1) as required by the validator."""
    return max(_SCORE_EPS, min(1.0 - _SCORE_EPS, score))


def score_creative_matcher(state: AdState) -> float:
    """Easy task grader: pure session CTR."""
    if state.total_impressions_shown == 0:
        return _clamp(0.0)
    return _clamp(min(1.0, state.total_clicks / state.total_impressions_shown))


def score_placement_optimizer(state: AdState, max_view_time: float = 15.0) -> float:
    """Medium task grader: validity + engagement blend."""
    total_actions = state.valid_actions + state.invalid_actions
    validity = _safe_div(state.valid_actions, total_actions) if total_actions > 0 else 0.0

    ctr = _safe_div(state.total_clicks, state.total_impressions_shown) if state.total_impressions_shown > 0 else 0.0
    norm_view = min(1.0, _safe_div(state.total_view_time, max_view_time * state.step_count))
    engagement = 0.5 * min(1.0, ctr / 0.5) + 0.5 * norm_view

    return _clamp(min(1.0, 0.3 * validity + 0.7 * engagement))


def score_campaign_optimizer(
    state: AdState,
    max_view_time: float = 15.0,
    max_satisfaction: float | None = None,
) -> float:
    """Hard task grader: multi-objective score."""
    total_actions = state.valid_actions + state.invalid_actions
    validity = _safe_div(state.valid_actions, total_actions) if total_actions > 0 else 0.0

    ctr = _safe_div(state.total_clicks, state.total_impressions_shown) if state.total_impressions_shown > 0 else 0.0
    ctr_score = min(1.0, ctr / 0.5)

    view_score = min(1.0, _safe_div(state.total_view_time, max_view_time * max(1, state.step_count)))

    if max_satisfaction is None:
        max_satisfaction = float(state.step_count) * 1.0
    sat_score = min(1.0, _safe_div(state.cumulative_satisfaction, max(1.0, max_satisfaction)))

    fatigue_score = 1.0 - state.fatigue_level

    return _clamp(min(1.0, (
        0.15 * validity
        + 0.25 * ctr_score
        + 0.20 * view_score
        + 0.25 * sat_score
        + 0.15 * fatigue_score
    )))


GRADERS = {
    "creative_matcher": score_creative_matcher,
    "placement_optimizer": score_placement_optimizer,
    "campaign_optimizer": score_campaign_optimizer,
}


_SCORE_LOW = 0.001
_SCORE_HIGH = 0.999


def grade_episode(state: AdState) -> float:
    """Score a completed episode using the task-appropriate grader."""
    grader = GRADERS.get(state.task, score_campaign_optimizer)
    raw = grader(state)
    clamped = max(_SCORE_LOW, min(_SCORE_HIGH, raw))
    return round(clamped, 4)
