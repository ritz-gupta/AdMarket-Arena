"""
Meta Ad Optimizer Environment Client.

WebSocket-based client for interacting with the AdOptimizer
environment server using the standard EnvClient API.
"""

from __future__ import annotations

from typing import Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import AdAction, AdObservation, AdState, AuctionAction, AuctionObservation, AuctionResult, ArenaState


class AdEnv(EnvClient[AdAction, AdObservation, AdState]):
    """Client for the Meta Ad Optimizer Environment.

    Example (sync):
        >>> with AdEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset(task="creative_matcher")
        ...     print(result.observation.user_segment)
        ...     result = env.step(AdAction(show_ad=True, creative_id=0))
        ...     print(result.observation.last_action_metrics)

    Example (async):
        >>> async with AdEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset(task="campaign_optimizer")
        ...     result = await env.step(AdAction(
        ...         show_ad=True, creative_id=2,
        ...         platform="instagram", placement="reels", ad_format="reel",
        ...     ))
    """

    def _step_payload(self, action: AdAction) -> Dict:
        return {
            "show_ad": action.show_ad,
            "creative_id": action.creative_id,
            "platform": action.platform,
            "placement": action.placement,
            "ad_format": action.ad_format,
        }

    def _parse_result(self, payload: Dict) -> StepResult[AdObservation]:
        obs_data = payload.get("observation", {})
        observation = AdObservation(
            task=obs_data.get("task", ""),
            user_segment=obs_data.get("user_segment", ""),
            user_interests=obs_data.get("user_interests", []),
            user_device=obs_data.get("user_device", "mobile"),
            current_platform=obs_data.get("current_platform", "instagram"),
            current_surface=obs_data.get("current_surface", "feed"),
            available_creatives=obs_data.get("available_creatives", []),
            impression_count=obs_data.get("impression_count", 0),
            fatigue_level=obs_data.get("fatigue_level", 0.0),
            step=obs_data.get("step", 0),
            total_steps=obs_data.get("total_steps", 20),
            last_action_metrics=obs_data.get("last_action_metrics", {}),
            session_metrics=obs_data.get("session_metrics", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> AdState:
        return AdState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            total_impressions_shown=payload.get("total_impressions_shown", 0),
            total_clicks=payload.get("total_clicks", 0),
            total_view_time=payload.get("total_view_time", 0.0),
            cumulative_satisfaction=payload.get("cumulative_satisfaction", 0.0),
            fatigue_level=payload.get("fatigue_level", 0.0),
            task=payload.get("task", ""),
            valid_actions=payload.get("valid_actions", 0),
            invalid_actions=payload.get("invalid_actions", 0),
        )


class AdMarketArenaEnv(EnvClient[AuctionAction, AuctionObservation, ArenaState]):
    """Client for AdMarket Arena — the multi-agent long-horizon ad auction env.

    Connects to the /arena route served by server/arena_env.py.
    Trained advertiser submits one AuctionAction per step; the env runs
    the full Vickrey auction against 4 scripted PersonaBots and returns
    an AuctionObservation with the outcome and next slot's user context.

    Example (async):
        >>> async with AdMarketArenaEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset(task="arena_easy")
        ...     obs = result.observation
        ...     print(obs.user_segment, obs.budget_remaining)
        ...     result = await env.step(AuctionAction(skip=False, bid_amount=1.5, creative_id=0))
        ...     print(result.observation.last_auction_result)
    """

    def _step_payload(self, action: AuctionAction) -> Dict:
        return {
            "skip": action.skip,
            "bid_amount": action.bid_amount,
            "creative_id": action.creative_id,
        }

    def _parse_result(self, payload: Dict) -> StepResult[AuctionObservation]:
        obs_data = payload.get("observation", {})

        # Deserialise last_auction_result sub-object if present
        raw_result = obs_data.get("last_auction_result")
        last_auction_result: Optional[AuctionResult] = None
        if raw_result:
            last_auction_result = AuctionResult(
                winner_id=raw_result.get("winner_id"),
                clearing_price=raw_result.get("clearing_price", 0.0),
                no_contest=raw_result.get("no_contest", False),
                all_bids=raw_result.get("all_bids", {}),
            )

        observation = AuctionObservation(
            task=obs_data.get("task", "arena_hard"),
            advertiser_id=obs_data.get("advertiser_id", "trained_advertiser"),
            campaign_objective=obs_data.get("campaign_objective", "conversion"),
            day_number=obs_data.get("day_number", 1),
            step_in_day=obs_data.get("step_in_day", 0),
            step=obs_data.get("step", 0),
            total_steps=obs_data.get("total_steps", 350),
            user_segment=obs_data.get("user_segment", ""),
            user_id=obs_data.get("user_id", ""),
            available_creatives=obs_data.get("available_creatives", []),
            weekly_budget=obs_data.get("weekly_budget", 1000.0),
            budget_remaining=obs_data.get("budget_remaining", 1000.0),
            daily_budget_remaining=obs_data.get("daily_budget_remaining", 142.86),
            recent_clearing_prices=obs_data.get("recent_clearing_prices", []),
            floor_price=obs_data.get("floor_price", 0.50),
            last_auction_result=last_auction_result,
            wins_today=obs_data.get("wins_today", 0),
            win_rate_today=obs_data.get("win_rate_today", 0.0),
            daily_roas=obs_data.get("daily_roas", 0.0),
            weekly_roas=obs_data.get("weekly_roas", 0.0),
            per_segment_fatigue=obs_data.get("per_segment_fatigue", {}),
            persona_names=obs_data.get("persona_names", []),
            yesterday_recap=obs_data.get("yesterday_recap", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> ArenaState:
        return ArenaState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            task=payload.get("task", "arena_hard"),
            day_number=payload.get("day_number", 1),
            step_in_day=payload.get("step_in_day", 0),
            weekly_budget=payload.get("weekly_budget", 1000.0),
            spent_total=payload.get("spent_total", 0.0),
            clicks_total=payload.get("clicks_total", 0),
            wins_total=payload.get("wins_total", 0),
            weekly_roas=payload.get("weekly_roas", 0.0),
            persona_names=payload.get("persona_names", []),
            auction_log_length=payload.get("auction_log_length", 0),
        )
