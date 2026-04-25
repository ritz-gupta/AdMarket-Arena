"""
Data models for the Meta Ad Optimizer Environment.

Defines typed Action, Observation, and State for ad optimization
across Instagram and Facebook surfaces.

Also contains the AdMarket Arena multi-agent models:
  AuctionAction, AuctionObservation, AuctionResult, ArenaState,
  OversightObservation (stub), ViolationFlag (stub).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


VALID_PLATFORMS = ("instagram", "facebook")
VALID_PLACEMENTS_IG = ("feed", "reels", "stories", "explore", "search")
VALID_PLACEMENTS_FB = ("feed", "reels", "stories", "marketplace", "search", "right_column")
VALID_FORMATS = ("image", "video", "carousel", "reel", "collection")


class AdAction(Action):
    """Agent decision for a single impression opportunity.

    Attributes:
        show_ad: False to skip this slot and let fatigue recover.
        creative_id: Index into the per-episode creative pool.
        platform: Target platform for the ad.
        placement: Target surface within the platform.
        ad_format: Creative format to render.
    """

    show_ad: bool = Field(..., description="Whether to show an ad or skip")
    creative_id: int = Field(default=0, description="Index into available_creatives pool (0 to N-1)")
    platform: str = Field(default="instagram", description="'instagram' | 'facebook'")
    placement: str = Field(default="feed", description="Surface to place the ad on")
    ad_format: str = Field(default="image", description="'image' | 'video' | 'carousel' | 'reel' | 'collection'")


class AdObservation(Observation):
    """What the agent sees at each step.

    Includes user context, available creatives, fatigue state,
    and cumulative session metrics.
    """

    task: str = Field(default="campaign_optimizer", description="Active task name")
    user_segment: str = Field(default="", description="User segment archetype")
    user_interests: List[str] = Field(default_factory=list)
    user_device: str = Field(default="mobile")
    current_platform: str = Field(default="instagram")
    current_surface: str = Field(default="feed")
    available_creatives: List[Dict[str, Any]] = Field(default_factory=list)
    impression_count: int = Field(default=0, description="Ads shown so far")
    fatigue_level: float = Field(default=0.0, ge=0.0, le=1.0)
    step: int = Field(default=0)
    total_steps: int = Field(default=20)
    last_action_metrics: Dict[str, Any] = Field(default_factory=dict)
    session_metrics: Dict[str, Any] = Field(default_factory=dict)


class AdState(State):
    """Internal episode state tracked by the environment."""

    total_impressions_shown: int = 0
    total_clicks: int = 0
    total_view_time: float = 0.0
    cumulative_satisfaction: float = 0.0
    fatigue_level: float = 0.0
    task: str = "campaign_optimizer"
    valid_actions: int = 0
    invalid_actions: int = 0


# ---------------------------------------------------------------------------
# AdMarket Arena — Multi-Agent Long-Horizon Ad Auction (Plan 2)
# ---------------------------------------------------------------------------

class AuctionAction(Action):
    """Per-step decision for the trained advertiser in AdMarket Arena.

    One of these is produced by the LLM agent at every impression slot.
    Scripted PersonaBots use the same interface (see competitors.py).
    """

    skip: bool = Field(default=False, description="Pass on this slot; tiny positive reward, no spend")
    bid_amount: float = Field(default=0.0, ge=0.0, le=5.0, description="CPM bid in [0.0, 5.0]")
    creative_id: int = Field(default=0, description="Index into available_creatives pool")


class AuctionResult(BaseModel):
    """Pydantic mirror of the auction.AuctionResult dataclass.

    Embedded in AuctionObservation so the agent knows last step's outcome.
    The plain-Python dataclass in auction.py is converted to this before
    being serialised into the HTTP response.
    """

    winner_id: Optional[str] = None
    clearing_price: float = 0.0
    no_contest: bool = False
    all_bids: Dict[str, float] = Field(default_factory=dict)


class AuctionObservation(Observation):
    """What the trained advertiser sees at the start of each auction step.

    Theme 2 long-horizon mechanism: yesterday_recap is a ~200-token
    natural-language summary injected at each new day boundary, letting
    the agent plan across the full 7-day episode despite context limits.
    """

    task: str = Field(default="arena_hard")
    advertiser_id: str = Field(default="trained_advertiser")
    campaign_objective: str = Field(default="conversion", description="'awareness' | 'conversion' | 'retention'")

    # Time
    day_number: int = Field(default=1, description="Current day (1-7)")
    step_in_day: int = Field(default=0, description="Impression slot within today (0-49)")
    step: int = Field(default=0, description="Global step counter (0-349)")
    total_steps: int = Field(default=350)

    # User context for this slot
    user_segment: str = Field(default="")
    user_id: str = Field(default="")
    available_creatives: List[Dict[str, Any]] = Field(default_factory=list)

    # Budget
    weekly_budget: float = Field(default=1000.0)
    budget_remaining: float = Field(default=1000.0, description="Remaining weekly budget")
    daily_budget_remaining: float = Field(default=142.86, description="Remaining daily soft cap")

    # Auction market context
    recent_clearing_prices: List[float] = Field(
        default_factory=list,
        description="Last ≤5 clearing prices (all advertisers), oldest first",
    )
    floor_price: float = Field(default=0.50, description="Current floor price (rises each day)")
    last_auction_result: Optional[AuctionResult] = Field(
        default=None,
        description="Outcome of the previous step's auction (None on step 0)",
    )

    # Own KPI signals
    wins_today: int = Field(default=0)
    win_rate_today: float = Field(default=0.0, description="wins_today / max(1, step_in_day)")
    daily_roas: float = Field(default=0.0)
    weekly_roas: float = Field(default=0.0)
    per_segment_fatigue: Dict[str, float] = Field(
        default_factory=dict,
        description="Our own per-segment fatigue (0=fresh, 1=fully fatigued)",
    )

    # Competitor context
    persona_names: List[str] = Field(default_factory=list, description="Names of scripted opponents this episode")

    # Theme 2: long-horizon text summary
    yesterday_recap: str = Field(
        default="",
        description="~200-token day recap injected at each day boundary for cross-day planning",
    )


class ArenaState(State):
    """Internal episode state for AdMarket Arena, returned by /arena/state."""

    task: str = "arena_hard"
    day_number: int = 1
    step_in_day: int = 0
    weekly_budget: float = 1000.0
    spent_total: float = 0.0
    clicks_total: int = 0
    wins_total: int = 0
    weekly_roas: float = 0.0
    persona_names: List[str] = Field(default_factory=list)
    auction_log_length: int = 0  # number of steps recorded so far


# ---------------------------------------------------------------------------
# Plan 2 stubs — OversightAgent fills these
# ---------------------------------------------------------------------------

class ViolationFlag(BaseModel):
    """A single detected policy violation (Plan 2 fills detection logic).

    Ground-truth instances are injected by the test harness;
    OversightAgent's predictions are scored by F1 vs ground truth.
    """

    advertiser_id: str = ""
    violation_type: str = Field(
        default="",
        description="'freq_cap_violation' | 'budget_overspend' | 'shill_bidding'",
    )
    step: int = 0
    day: int = 0
    evidence: Dict[str, Any] = Field(default_factory=dict)


class OversightObservation(Observation):
    """STUB — Plan 2 fills the full implementation.

    Passed to the OversightAgent (separate from the trained advertiser).
    Contains the full auction log for the current day so the oversight
    model can detect violations before the day boundary reward fires.
    """

    day_number: int = 1
    auction_log: List[Dict[str, Any]] = Field(default_factory=list)
    predicted_flags: List[ViolationFlag] = Field(default_factory=list)
    ground_truth_flags: List[ViolationFlag] = Field(default_factory=list)
