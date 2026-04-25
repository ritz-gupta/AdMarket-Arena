"""
Data models for the Meta Ad Optimizer Environment.

Defines typed Action, Observation, and State for both:
  - The single-agent ad optimizer (Round 1: AdAction / AdObservation / AdState)
  - The multi-agent AdMarket Arena (Round 2):
      - AuctionAction, AuctionObservation, AuctionResult, ArenaState
        (Plan 1: trained-advertiser-facing schemas, populated by arena_env)
      - OversightObservation, OversightAction, ViolationFlag,
        GroundTruthViolation, AuctionRecord, CampaignStateSummary
        (Plan 2: oversight-agent schemas, populated by violation_injector
         + arena_env at day boundaries)

Merged-branch note: ``AuctionObservation`` carries the union of fields
from both branches. The env (Plan 1) populates the KPI/state fields it
tracks; the synthetic loop in ``scripts/advertiser_eval.py`` and the
training notebook (Plan 3) populate the trained-agent-facing fields they
need. Defaults make every field optional so consumers can ignore the
fields they don't care about.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


VALID_PLATFORMS = ("instagram", "facebook")
VALID_PLACEMENTS_IG = ("feed", "reels", "stories", "explore", "search")
VALID_PLACEMENTS_FB = ("feed", "reels", "stories", "marketplace", "search", "right_column")
VALID_FORMATS = ("image", "video", "carousel", "reel", "collection")


class AdAction(Action):
    """Agent decision for a single impression opportunity (Round 1)."""

    show_ad: bool = Field(..., description="Whether to show an ad or skip")
    creative_id: int = Field(default=0, description="Index into available_creatives pool (0 to N-1)")
    platform: str = Field(default="instagram", description="'instagram' | 'facebook'")
    placement: str = Field(default="feed", description="Surface to place the ad on")
    ad_format: str = Field(default="image", description="'image' | 'video' | 'carousel' | 'reel' | 'collection'")


class AdObservation(Observation):
    """What the Round 1 agent sees at each step."""

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
    """Internal episode state tracked by the Round 1 environment."""

    total_impressions_shown: int = 0
    total_clicks: int = 0
    total_view_time: float = 0.0
    cumulative_satisfaction: float = 0.0
    fatigue_level: float = 0.0
    task: str = "campaign_optimizer"
    valid_actions: int = 0
    invalid_actions: int = 0


# ---------------------------------------------------------------------------
# AdMarket Arena (Round 2) — multi-agent auction schemas
# ---------------------------------------------------------------------------

VIOLATION_TYPES = ("frequency_cap", "budget_overspend", "shill_bidding")
ViolationType = Literal["frequency_cap", "budget_overspend", "shill_bidding"]


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
    rejected_below_floor: List[str] = Field(default_factory=list)
    rejected_freq_cap: List[str] = Field(default_factory=list)


class AuctionObservation(Observation):
    """What the trained advertiser sees at the start of each auction step.

    Theme 2 long-horizon mechanism: yesterday_recap is a ~200-token
    natural-language summary injected at each new day boundary, letting
    the agent plan across the full 7-day episode despite context limits.

    The schema is the union of fields populated by Plan 1's
    ``server/arena_env.py`` (KPI signals: wins_today, daily_roas,
    weekly_roas, last_auction_result, persona_names) and Plan 3's
    ``scripts/advertiser_eval.py`` synthetic loop (spent counters,
    clicks, target_weekly_roas, frequency_cap_per_user).
    """

    task: str = Field(default="arena_hard")
    advertiser_id: str = Field(default="trained_advertiser")
    campaign_objective: str = Field(
        default="conversion",
        description="'awareness' | 'conversion' | 'retention'",
    )

    # Time
    day_number: int = Field(default=1, description="Current day (1-7), Plan 1 convention")
    day_of_week: int = Field(default=0, ge=0, le=6, description="Current day (0-6), Plan 3 convention")
    step_in_day: int = Field(default=0, description="Impression slot within today")
    step: int = Field(default=0, description="Global step counter")
    total_steps: int = Field(default=350)

    # User context for this slot
    user_segment: str = Field(default="")
    user_interests: List[str] = Field(default_factory=list)
    user_id: str = Field(default="")
    current_surface: str = Field(default="feed")
    available_creatives: List[Dict[str, Any]] = Field(default_factory=list)

    # Budget — Plan 1 names + Plan 3 aliases (both populated by the env
    # so consumers can use either convention).
    weekly_budget: float = Field(default=1000.0, description="Total weekly cap (Plan 1)")
    budget_remaining: float = Field(default=1000.0, description="Weekly remaining (Plan 1)")
    weekly_budget_remaining: float = Field(default=1000.0, description="Weekly remaining (Plan 3 alias)")
    daily_budget_remaining: float = Field(default=142.86, description="Daily soft-cap remaining")
    spent_so_far_today: float = Field(default=0.0)
    spent_so_far_week: float = Field(default=0.0)
    clicks_today: int = Field(default=0)
    clicks_week: int = Field(default=0)
    target_weekly_roas: float = Field(default=2.0, description="Episode KPI target")
    frequency_cap_per_user: int = Field(default=3)

    # Auction market context
    recent_clearing_prices: List[float] = Field(
        default_factory=list,
        description="Last <=5 clearing prices (all advertisers), oldest first",
    )
    floor_price: float = Field(default=0.50, description="Current floor price (rises each day)")
    last_auction_result: Optional[AuctionResult] = Field(
        default=None,
        description="Outcome of the previous step's auction (None on step 0)",
    )

    # Own KPI signals
    wins_today: int = Field(default=0)
    win_rate_today: float = Field(default=0.0)
    daily_roas: float = Field(default=0.0)
    weekly_roas: float = Field(default=0.0)
    per_segment_fatigue: Dict[str, float] = Field(
        default_factory=dict,
        description="Our own per-segment fatigue (0=fresh, 1=fully fatigued)",
    )

    # Competitor context
    persona_names: List[str] = Field(
        default_factory=list,
        description="Names of scripted opponents this episode",
    )

    # Theme 2: long-horizon text summary
    yesterday_recap: str = Field(
        default="",
        description="~200-token day recap injected at each day boundary",
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
    auction_log_length: int = 0


# ---------------------------------------------------------------------------
# Oversight Agent (Fleet AI bonus / Plan 2) — schemas
# ---------------------------------------------------------------------------

class AuctionRecord(BaseModel):
    """Single auction outcome as exposed to the OversightAgent.

    Redacted view: enough info to detect violations from auction-log
    analysis, no privileged ground-truth fields. Used by Plan 2's
    ``oversight.py`` (heuristic + LLM agents) and Plan 2's
    ``violation_injector.py`` ground-truth tracking.
    """

    step: int
    day: int
    step_in_day: int
    user_id: str = Field(description="Synthetic per-user ID for tracking impressions")
    user_segment: str
    advertiser_id: int = Field(description="Bidder identity (numeric for oversight)")
    bid: float
    won: bool
    clearing_price: float = Field(description="Price the winner paid (0 if no winner)")
    floor_price: float
    no_contest: bool = Field(description="True if all bids were below floor")
    creative_id: Optional[int] = None


class CampaignStateSummary(BaseModel):
    """Per-advertiser snapshot at a day boundary, exposed to OversightAgent."""

    advertiser_id: int
    advertiser_name: str = ""
    spent_today: float
    daily_budget_cap: float
    spent_total: float
    weekly_budget_cap: float
    impressions_today: int
    clicks_today: int


class ViolationFlag(BaseModel):
    """One predicted violation emitted by the OversightAgent.

    Plan 2 finalized schema (Plan 1 shipped a stub). Note
    ``advertiser_id`` is ``int`` (the numeric oversight-side identifier),
    not the string ``trained_advertiser``-style id used elsewhere — this
    matches how Plan 2's F1 scorer compares (advertiser_id, type) pairs
    against ground-truth.
    """

    advertiser_id: int
    violation_type: ViolationType
    confidence: float = Field(default=0.5, ge=0.01, le=0.99)
    evidence_step_ids: List[int] = Field(default_factory=list)


class OversightObservation(Observation):
    """What the OversightAgent sees at each day boundary.

    Plan 2 finalized schema. Plan 1 shipped a stub with auction_log as
    ``List[Dict]`` and ``predicted_flags``/``ground_truth_flags`` fields;
    those have been merged into typed ``AuctionRecord`` and
    ``CampaignStateSummary`` lists here. Ground-truth and predicted flags
    are tracked in env state separately (oversight never sees them in
    the observation — that's the whole point of the role).
    """

    day: int = Field(default=0)
    auction_log: List[AuctionRecord] = Field(default_factory=list)
    campaign_states: List[CampaignStateSummary] = Field(default_factory=list)
    floor_price: float = Field(default=0.0)
    frequency_cap_per_user: int = Field(default=999)
    advertiser_names: Dict[int, str] = Field(default_factory=dict)


class OversightAction(Action):
    """OversightAgent emits a list of violation flags per day boundary."""

    flags: List[ViolationFlag] = Field(default_factory=list)


class GroundTruthViolation(BaseModel):
    """Ground-truth violation tracked in env state for F1 scoring.

    Never exposed in any observation. Only the env's reward/scoring
    layer sees this (Plan 2's ``violation_injector.py`` writes it,
    ``OversightF1Rubric`` reads it).
    """

    advertiser_id: int
    violation_type: ViolationType
    day: int
    step_ids: List[int] = Field(default_factory=list)
    note: str = Field(default="", description="Optional human-readable detail")
