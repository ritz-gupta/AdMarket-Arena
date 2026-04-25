"""
Advertiser 3-mode robustness eval (master Section 7.2).

Owns the eval table that drives the "trained advertiser" pitch slide.

Three modes:

  - **standard**: 5 jittered personas at default ranges, N=20 episodes.
  - **edge**: 3 sub-conditions (extreme-jitter, maxed-personas,
              held-out-OpportunisticArbitrageur) x N=10 episodes each.
  - **selfplay**: replace one persona slot (default
                  ``CautiousChallenger``) with a frozen *earlier* trained
                  checkpoint via ``LLMPolicyBot``. N=10 episodes.

For each mode the script computes the four advertiser metrics in the
master Section 7.2 table:
    weekly_roas, bid_precision, budget_depletion_day, fatigue_sensitivity

and writes everything to ``results/advertiser_eval.json`` for Plan 4 to
consume in the final eval table.

Plan 1 dependency: this module would normally drive
``server/arena_env.py`` end-to-end. Until Plan 1 lands, the module
falls back to the same synthetic auction loop used by
``scripts/collect_oversight_trajectories.py``: noisy pacing baselines
+ the trained agent (or a mock policy when ``--checkpoint`` is omitted),
so the harness, JSON schema, and behavioral diagnostics are testable
end-to-end without the real env. The notebook + ``inference.py``
wrapper invoke the same entry points, so swapping in the real env later
is a one-line change.

Usage:
    python -m scripts.advertiser_eval \
        --checkpoint checkpoints/advertiser_run/best/ \
        --opponent-checkpoint checkpoints/advertiser_run/checkpoint-10/ \
        --task arena_hard \
        --out results/advertiser_eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from competitors import (  # noqa: E402
    HELD_OUT_PERSONA,
    LLMPolicyBot,
    PERSONAS,
    PersonaBot,
    PersonaSpec,
    build_opponent_slate,
    jitter_persona,
    maxed_persona,
)
from models import AuctionAction, AuctionObservation  # noqa: E402
from tasks import ARENA_TASKS, ArenaTaskConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic auction loop (Plan 1 fallback)
# ---------------------------------------------------------------------------
#
# Same auction mechanics the env will use:
#   - second-price + reserve floor + frequency-cap filtering
#   - winner pays max(second-highest, reserve)
#   - per-day budget tracked, weekly budget tracked
#   - clearing prices recorded for the recent_clearing_prices window
#
# When Plan 1's `arena_env.py` lands, this synthetic loop should be
# swapped for `arena_env.AdMarketArenaEnvironment(...)` reset/step
# calls; the surrounding eval scaffolding (mode dispatch, metric
# aggregation, JSON output) is env-agnostic.

_SEGMENTS = [
    "gen_z_creator", "millennial_parent", "business_pro",
    "casual_scroller", "fitness_enthusiast", "bargain_hunter",
]


def _sample_creatives(rng: random.Random, n: int = 8) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx in range(n):
        out.append({
            "pool_index": idx,
            "category": rng.choice(["fashion", "fitness", "finance", "food", "travel", "gaming"]),
            "tone": rng.choice(["energetic", "calm", "playful", "premium"]),
            "target_segment": rng.choice(_SEGMENTS),
            "base_ctr": round(rng.uniform(0.02, 0.18), 3),
            "base_view_time": round(rng.uniform(2.0, 9.0), 2),
        })
    return out


@dataclass
class _AdvertiserState:
    advertiser_id: int
    name: str
    spent_today: float = 0.0
    spent_week: float = 0.0
    clicks_today: int = 0
    clicks_week: int = 0
    revenue_today: float = 0.0
    revenue_week: float = 0.0
    impressions_today: int = 0
    impressions_week: int = 0
    per_segment_fatigue: Dict[str, float] = field(default_factory=lambda: {s: 0.0 for s in _SEGMENTS})
    last_creative_id: Optional[int] = None
    creative_switches_under_fatigue: int = 0
    creative_decisions_under_fatigue: int = 0
    weekly_budget: float = 0.0
    daily_budget: float = 0.0
    budget_depletion_day: Optional[int] = None
    won_auctions_with_clearing: List[Tuple[float, float]] = field(default_factory=list)


def _build_observation(
    advertiser_id: int,
    advertiser_state: _AdvertiserState,
    cfg: ArenaTaskConfig,
    day: int,
    step_in_day: int,
    user_segment: str,
    user_interests: List[str],
    current_surface: str,
    creatives: List[Dict[str, Any]],
    recent_clearing_prices: List[float],
    floor_price: float,
    yesterday_recap: str,
) -> AuctionObservation:
    weekly_remaining = max(0.0, advertiser_state.weekly_budget - advertiser_state.spent_week)
    daily_remaining = max(0.0, advertiser_state.daily_budget - advertiser_state.spent_today)
    return AuctionObservation(
        user_segment=user_segment,
        user_interests=user_interests,
        current_surface=current_surface,
        day_of_week=day,
        step_in_day=step_in_day,
        weekly_budget_remaining=weekly_remaining,
        daily_budget_remaining=daily_remaining,
        spent_so_far_today=advertiser_state.spent_today,
        spent_so_far_week=advertiser_state.spent_week,
        clicks_today=advertiser_state.clicks_today,
        clicks_week=advertiser_state.clicks_week,
        per_segment_fatigue=dict(advertiser_state.per_segment_fatigue),
        available_creatives=creatives,
        recent_clearing_prices=recent_clearing_prices[-5:],
        yesterday_recap=yesterday_recap,
        target_weekly_roas=cfg.target_weekly_roas,
        floor_price=floor_price,
        frequency_cap_per_user=cfg.frequency_cap_per_user,
    )


def _resolve_auction(
    bids: List[Tuple[int, float, int]],
    floor_price: float,
    impressions_to_user_today: Dict[Tuple[int, str], int],
    user_id: str,
    cap: int,
) -> Tuple[Optional[int], float, bool]:
    """Second-price auction with reserve floor + frequency-cap filter.

    bids: list of (advertiser_id, bid_amount, creative_id).
    Returns (winner_id, clearing_price, no_contest).
    """
    valid: List[Tuple[int, float, int]] = []
    for adv_id, bid, c_id in bids:
        if bid < floor_price:
            continue
        if impressions_to_user_today.get((adv_id, user_id), 0) >= cap:
            continue
        valid.append((adv_id, bid, c_id))

    if not valid:
        return None, 0.0, True

    valid.sort(key=lambda t: t[1], reverse=True)
    winner_id, top_bid, _creative = valid[0]
    if len(valid) == 1:
        return winner_id, max(floor_price, top_bid * 0.9), True
    second = valid[1][1]
    clearing = max(floor_price, second)
    return winner_id, clearing, False


def _engagement_simulator(
    rng: random.Random,
    creative: Dict[str, Any],
    user_segment: str,
    fatigue: float,
) -> Tuple[bool, float]:
    """Returns (clicked, revenue_value)."""
    base_ctr = float(creative.get("base_ctr", 0.05))
    if creative.get("target_segment") == user_segment:
        base_ctr *= 2.0
    base_ctr *= max(0.1, 1.0 - 0.7 * fatigue)
    clicked = rng.random() < min(0.5, base_ctr)
    revenue_value = 1.5 if clicked else 0.0
    return clicked, revenue_value


def _update_fatigue(per_segment_fatigue: Dict[str, float], user_segment: str, served: bool) -> None:
    increment = 0.05
    recovery = 0.02
    for seg in per_segment_fatigue:
        if seg == user_segment and served:
            per_segment_fatigue[seg] = min(1.0, per_segment_fatigue[seg] + increment)
        else:
            per_segment_fatigue[seg] = max(0.0, per_segment_fatigue[seg] - recovery)


# ---------------------------------------------------------------------------
# Trained-agent stand-in (when no checkpoint is provided)
# ---------------------------------------------------------------------------

@dataclass
class _MockTrainedPolicy:
    """Reasonable pacing-aware agent so the eval harness produces
    sensible numbers when no real checkpoint is provided. Used by
    --eval-mode smoke tests + by inference.py when no checkpoint is
    passed."""

    name: str = "mock_trained"
    target_daily_pace: float = 1.0
    skip_under_fatigue: float = 0.5

    def bid(
        self,
        observation: AuctionObservation,
        state: Optional[Dict[str, Any]] = None,
    ) -> AuctionAction:
        fatigue = float(observation.per_segment_fatigue.get(observation.user_segment, 0.0))
        if fatigue > self.skip_under_fatigue:
            return AuctionAction(skip=True, bid_amount=0.0, creative_id=0)

        recent = [p for p in observation.recent_clearing_prices if p > 0]
        market = sum(recent) / len(recent) if recent else max(observation.floor_price, 0.5)
        # pace: scale down as we use up daily budget
        used_ratio = 0.0
        budget_today = observation.daily_budget_remaining + observation.spent_so_far_today
        if budget_today > 0:
            used_ratio = observation.spent_so_far_today / budget_today
        pace = max(0.4, 1.2 - used_ratio)
        bid_amount = max(observation.floor_price + 0.05, min(5.0, market * 1.05 * pace))

        creative_id = 0
        for idx, c in enumerate(observation.available_creatives):
            if c.get("target_segment") == observation.user_segment:
                creative_id = idx
                break
        return AuctionAction(skip=False, bid_amount=round(bid_amount, 4), creative_id=creative_id)


PolicyBidder = Callable[[AuctionObservation, Optional[Dict[str, Any]]], AuctionAction]


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    weekly_roas: float
    bid_precision: float
    budget_depletion_day: float
    fatigue_sensitivity: float
    weekly_revenue: float
    weekly_spend: float
    weekly_clicks: int
    weekly_impressions: int
    win_rate_per_persona: Dict[str, float]


def run_episode(
    seed: int,
    cfg: ArenaTaskConfig,
    trained_policy: PolicyBidder,
    opponents: List[PersonaBot],
) -> EpisodeResult:
    """Run one full episode against the supplied persona slate.

    The trained advertiser is always advertiser_id=0; opponents are
    advertiser_id 1..n. Same auction mechanics are used in
    standard/edge/selfplay modes — only the opponent slate composition
    changes.
    """
    rng = random.Random(seed)

    advertiser_states: List[_AdvertiserState] = []
    advertiser_states.append(_AdvertiserState(
        advertiser_id=0, name="trained",
        weekly_budget=cfg.weekly_budget,
        daily_budget=cfg.daily_budget,
    ))
    for idx, p in enumerate(opponents, start=1):
        advertiser_states.append(_AdvertiserState(
            advertiser_id=idx, name=p.spec.name,
            weekly_budget=cfg.weekly_budget,
            daily_budget=cfg.daily_budget,
        ))

    creatives = _sample_creatives(rng, n=8)
    recent_clearing_prices: List[float] = []
    yesterday_recap = ""
    impressions_to_user_today: Dict[Tuple[int, str], int] = {}
    user_pool = [f"u{idx:03d}" for idx in range(30)]
    win_counts: Dict[str, int] = {p.spec.name: 0 for p in opponents}
    win_counts["trained"] = 0

    for day in range(cfg.days):
        for s in advertiser_states:
            s.spent_today = 0.0
            s.clicks_today = 0
            s.revenue_today = 0.0
            s.impressions_today = 0
        impressions_to_user_today.clear()
        floor_price = cfg.floor_price_base + day * cfg.floor_price_daily_increment

        for step_in_day in range(cfg.impressions_per_day):
            user_id = rng.choice(user_pool)
            user_segment = rng.choice(_SEGMENTS)
            user_interests = rng.sample(_SEGMENTS, k=2)
            current_surface = rng.choice(["feed", "reels", "stories"])

            bids: List[Tuple[int, float, int]] = []

            obs0 = _build_observation(
                advertiser_id=0,
                advertiser_state=advertiser_states[0],
                cfg=cfg, day=day, step_in_day=step_in_day,
                user_segment=user_segment, user_interests=user_interests,
                current_surface=current_surface, creatives=creatives,
                recent_clearing_prices=recent_clearing_prices, floor_price=floor_price,
                yesterday_recap=yesterday_recap,
            )
            action0 = trained_policy(obs0, None)
            if not action0.skip and action0.bid_amount > 0.0 and advertiser_states[0].spent_today < advertiser_states[0].daily_budget:
                bids.append((0, action0.bid_amount, action0.creative_id))

            for idx, persona in enumerate(opponents, start=1):
                state = advertiser_states[idx]
                obs = _build_observation(
                    advertiser_id=idx,
                    advertiser_state=state,
                    cfg=cfg, day=day, step_in_day=step_in_day,
                    user_segment=user_segment, user_interests=user_interests,
                    current_surface=current_surface, creatives=creatives,
                    recent_clearing_prices=recent_clearing_prices, floor_price=floor_price,
                    yesterday_recap=yesterday_recap,
                )
                action = persona.bid(obs, {"spent_today": state.spent_today, "daily_target": cfg.daily_budget})
                if not action.skip and action.bid_amount > 0.0 and state.spent_today < state.daily_budget:
                    bids.append((idx, action.bid_amount, action.creative_id))

            winner_id, clearing_price, no_contest = _resolve_auction(
                bids=bids, floor_price=floor_price,
                impressions_to_user_today=impressions_to_user_today,
                user_id=user_id, cap=cfg.frequency_cap_per_user,
            )

            if winner_id is not None:
                impressions_to_user_today[(winner_id, user_id)] = (
                    impressions_to_user_today.get((winner_id, user_id), 0) + 1
                )
                state = advertiser_states[winner_id]
                state.spent_today += clearing_price
                state.spent_week += clearing_price

                # Record clearing for trained agent's bid_precision
                if winner_id == 0:
                    actual_bid = action0.bid_amount
                    state.won_auctions_with_clearing.append((actual_bid, clearing_price))

                winner_action = action0 if winner_id == 0 else next(
                    (b for b in bids if b[0] == winner_id), None
                )
                if winner_action is not None:
                    creative_id = winner_action.creative_id if winner_id == 0 else winner_action[2]
                    creative = creatives[max(0, min(creative_id, len(creatives) - 1))]
                else:
                    creative = creatives[0]

                # Track fatigue-sensitivity for the trained agent
                if winner_id == 0:
                    fatigue = state.per_segment_fatigue.get(user_segment, 0.0)
                    if fatigue > 0.4:
                        state.creative_decisions_under_fatigue += 1
                        if state.last_creative_id is not None and state.last_creative_id != creative_id:
                            state.creative_switches_under_fatigue += 1
                    state.last_creative_id = creative_id

                clicked, revenue = _engagement_simulator(rng, creative, user_segment, state.per_segment_fatigue.get(user_segment, 0.0))
                if clicked:
                    state.clicks_today += 1
                    state.clicks_week += 1
                state.revenue_today += revenue
                state.revenue_week += revenue
                state.impressions_today += 1
                state.impressions_week += 1
                _update_fatigue(state.per_segment_fatigue, user_segment, served=True)

                win_counts[state.name] = win_counts.get(state.name, 0) + 1
            else:
                _update_fatigue(advertiser_states[0].per_segment_fatigue, user_segment, served=False)

            if not no_contest and clearing_price > 0:
                recent_clearing_prices.append(clearing_price)

        # End-of-day budget tracking
        for s in advertiser_states:
            if s.budget_depletion_day is None and s.spent_week >= s.weekly_budget * 0.99:
                s.budget_depletion_day = day + 1
        yesterday_recap = (
            f"Day {day+1}: spent={advertiser_states[0].spent_today:.2f} "
            f"clicks={advertiser_states[0].clicks_today} "
            f"revenue={advertiser_states[0].revenue_today:.2f}"
        )

    # --- Compute episode metrics for the trained agent ---
    trained = advertiser_states[0]
    weekly_roas = trained.revenue_week / trained.spent_week if trained.spent_week > 0 else 0.0

    if trained.won_auctions_with_clearing:
        bid_precision = statistics.mean(
            (bid - clearing) / max(clearing, 1e-6)
            for bid, clearing in trained.won_auctions_with_clearing
        )
    else:
        bid_precision = 0.0

    budget_depletion_day = float(trained.budget_depletion_day or cfg.days)

    if trained.creative_decisions_under_fatigue > 0:
        switch_rate = trained.creative_switches_under_fatigue / trained.creative_decisions_under_fatigue
        # Negative correlation proxy: higher switch_rate means stronger
        # avoidance, so we report -switch_rate to match the master plan
        # convention (lower / more negative = better).
        fatigue_sensitivity = float(-switch_rate)
    else:
        fatigue_sensitivity = 0.0

    total_wins = sum(win_counts.values()) or 1
    win_rate_per_persona = {
        f"win_rate_vs_{name}": (
            count / total_wins if name != "trained" else win_counts.get("trained", 0) / total_wins
        )
        for name, count in win_counts.items()
        if name != "trained"
    }

    return EpisodeResult(
        weekly_roas=float(weekly_roas),
        bid_precision=float(bid_precision),
        budget_depletion_day=float(budget_depletion_day),
        fatigue_sensitivity=float(fatigue_sensitivity),
        weekly_revenue=float(trained.revenue_week),
        weekly_spend=float(trained.spent_week),
        weekly_clicks=int(trained.clicks_week),
        weekly_impressions=int(trained.impressions_week),
        win_rate_per_persona=win_rate_per_persona,
    )


# ---------------------------------------------------------------------------
# Mode dispatchers
# ---------------------------------------------------------------------------

def _sample_persona_slate(
    rng: random.Random,
    persona_names: Sequence[str],
    n: int,
    jitter_scale: float = 1.0,
    valuation_anchor: float = 1.0,
) -> List[PersonaBot]:
    pool = list(persona_names)
    if n <= len(pool):
        chosen = pool[:n]
    else:
        chosen = pool + [pool[i % len(pool)] for i in range(n - len(pool))]
    return build_opponent_slate(
        chosen, rng=rng, jitter_enabled=True,
        jitter_scale=jitter_scale, valuation_anchor=valuation_anchor,
    )


def run_standard_mode(
    cfg: ArenaTaskConfig,
    trained_policy: PolicyBidder,
    n_episodes: int,
    base_seed: int = 1000,
) -> Dict[str, Any]:
    persona_names = list(PERSONAS.keys())
    episodes: List[EpisodeResult] = []
    for ep in range(n_episodes):
        ep_rng = random.Random(base_seed + ep)
        opponents = _sample_persona_slate(ep_rng, persona_names, n=cfg.n_personas, jitter_scale=1.0)
        episodes.append(run_episode(seed=base_seed + ep, cfg=cfg,
                                    trained_policy=trained_policy, opponents=opponents))
    return _aggregate(episodes, mode="standard")


def run_edge_mode(
    cfg: ArenaTaskConfig,
    trained_policy: PolicyBidder,
    n_per_subcondition: int = 10,
    base_seed: int = 2000,
) -> Dict[str, Any]:
    persona_names = list(PERSONAS.keys())
    sub_results: Dict[str, Dict[str, Any]] = {}

    extreme_eps: List[EpisodeResult] = []
    for ep in range(n_per_subcondition):
        ep_rng = random.Random(base_seed + ep)
        opponents = _sample_persona_slate(ep_rng, persona_names, n=cfg.n_personas, jitter_scale=2.0)
        extreme_eps.append(run_episode(seed=base_seed + ep, cfg=cfg,
                                       trained_policy=trained_policy, opponents=opponents))
    sub_results["extreme_jitter"] = _aggregate(extreme_eps, mode="edge.extreme_jitter")

    maxed_eps: List[EpisodeResult] = []
    for ep in range(n_per_subcondition):
        ep_rng = random.Random(base_seed + 1000 + ep)
        chosen = persona_names[: cfg.n_personas]
        opponents: List[PersonaBot] = []
        for name in chosen:
            spec = PERSONAS[name]
            opponents.append(PersonaBot(spec=spec, traits=maxed_persona(spec)))
        maxed_eps.append(run_episode(seed=base_seed + 1000 + ep, cfg=cfg,
                                     trained_policy=trained_policy, opponents=opponents))
    sub_results["maxed_personas"] = _aggregate(maxed_eps, mode="edge.maxed_personas")

    held_out_eps: List[EpisodeResult] = []
    for ep in range(n_per_subcondition):
        ep_rng = random.Random(base_seed + 2000 + ep)
        chosen = persona_names[: max(0, cfg.n_personas - 1)] + [HELD_OUT_PERSONA.name]
        opponents = build_opponent_slate(
            chosen, rng=ep_rng, jitter_enabled=True, jitter_scale=1.0,
        )
        held_out_eps.append(run_episode(seed=base_seed + 2000 + ep, cfg=cfg,
                                        trained_policy=trained_policy, opponents=opponents))
    sub_results["held_out_persona"] = _aggregate(held_out_eps, mode="edge.held_out_persona")

    aggregate_eps = extreme_eps + maxed_eps + held_out_eps
    overall = _aggregate(aggregate_eps, mode="edge")
    overall["sub_conditions"] = sub_results
    return overall


def run_selfplay_mode(
    cfg: ArenaTaskConfig,
    trained_policy: PolicyBidder,
    opponent_policy: PolicyBidder,
    n_episodes: int = 10,
    base_seed: int = 3000,
    replace_persona: str = "CautiousChallenger",
) -> Dict[str, Any]:
    """Replace one persona slot with a frozen earlier checkpoint
    (``opponent_policy``) wrapped in an ``LLMPolicyBot``-shaped adapter.

    This eval is runnable at inference time with two checkpoints (e.g.
    a step-10 snapshot + the final trained model); does not require the
    Section 6 self-play training stretch to have landed.
    """

    class _PolicyAsPersona:
        def __init__(self, name: str, policy: PolicyBidder) -> None:
            self.spec = PersonaSpec(
                name=name, aggression=0.5, pacing_strength=0.5, segment_focus=0.5,
                fatigue_awareness=0.5, price_elasticity=0.5, jitter=0.0,
            )
            self.policy = policy

        def bid(self, observation: AuctionObservation, state: Optional[Dict[str, Any]] = None) -> AuctionAction:
            return self.policy(observation, state)

    persona_names = [n for n in PERSONAS if n != replace_persona][: cfg.n_personas - 1]
    episodes: List[EpisodeResult] = []
    for ep in range(n_episodes):
        ep_rng = random.Random(base_seed + ep)
        opponents = build_opponent_slate(persona_names, rng=ep_rng, jitter_enabled=True)
        opponents.append(_PolicyAsPersona("frozen_v1", opponent_policy))  # type: ignore[arg-type]
        episodes.append(run_episode(seed=base_seed + ep, cfg=cfg,
                                    trained_policy=trained_policy, opponents=opponents))
    return _aggregate(episodes, mode="selfplay")


def _aggregate(results: List[EpisodeResult], mode: str) -> Dict[str, Any]:
    if not results:
        return {
            "mode": mode,
            "n_episodes": 0,
            "weekly_roas_mean": 0.0,
            "bid_precision_mean": 0.0,
            "budget_depletion_day_mean": 0.0,
            "fatigue_sensitivity_mean": 0.0,
            "weekly_roas_episodes": [],
            "bid_precision_episodes": [],
            "budget_depletion_day_episodes": [],
            "fatigue_sensitivity_episodes": [],
        }

    return {
        "mode": mode,
        "n_episodes": len(results),
        "weekly_roas_mean": statistics.mean(r.weekly_roas for r in results),
        "weekly_roas_stdev": statistics.stdev([r.weekly_roas for r in results]) if len(results) > 1 else 0.0,
        "bid_precision_mean": statistics.mean(r.bid_precision for r in results),
        "budget_depletion_day_mean": statistics.mean(r.budget_depletion_day for r in results),
        "fatigue_sensitivity_mean": statistics.mean(r.fatigue_sensitivity for r in results),
        "weekly_revenue_mean": statistics.mean(r.weekly_revenue for r in results),
        "weekly_spend_mean": statistics.mean(r.weekly_spend for r in results),
        "weekly_clicks_mean": statistics.mean(r.weekly_clicks for r in results),
        "weekly_roas_episodes": [r.weekly_roas for r in results],
        "bid_precision_episodes": [r.bid_precision for r in results],
        "budget_depletion_day_episodes": [r.budget_depletion_day for r in results],
        "fatigue_sensitivity_episodes": [r.fatigue_sensitivity for r in results],
    }


# ---------------------------------------------------------------------------
# Policy loaders
# ---------------------------------------------------------------------------

def _load_policy(checkpoint_path: Optional[str], fallback_name: str = "mock_trained") -> PolicyBidder:
    if checkpoint_path:
        from competitors import (  # local import keeps top-level light
            LLMPolicyBot,
            make_unsloth_advertiser_completion_fn,
        )
        completion_fn = make_unsloth_advertiser_completion_fn(checkpoint_path)
        bot = LLMPolicyBot(completion_fn=completion_fn, name=Path(checkpoint_path).name)
        return bot.bid
    mock = _MockTrainedPolicy(name=fallback_name)
    return mock.bid


def _baselines_for_eval(cfg: ArenaTaskConfig) -> Dict[str, Dict[str, Any]]:
    """Compute a small basket of baselines (random + pacing) so the
    eval table has reference rows to anchor the trained-agent column."""

    class _Random:
        def bid(self, obs: AuctionObservation, state: Optional[Dict[str, Any]] = None) -> AuctionAction:
            r = random.Random(int(obs.spent_so_far_today * 1000) + obs.step_in_day)
            return AuctionAction(
                skip=r.random() < 0.15,
                bid_amount=round(r.uniform(obs.floor_price + 0.05, max(obs.floor_price + 0.2, 1.5)), 4),
                creative_id=r.randrange(max(1, len(obs.available_creatives))),
            )

    class _Pacing:
        def bid(self, obs: AuctionObservation, state: Optional[Dict[str, Any]] = None) -> AuctionAction:
            steps_per_day = max(1, cfg.impressions_per_day)
            target_per_step = cfg.daily_budget / steps_per_day
            return AuctionAction(
                skip=False,
                bid_amount=round(max(obs.floor_price + 0.05, target_per_step * 1.5), 4),
                creative_id=0,
            )

    persona_names = list(PERSONAS.keys())
    out: Dict[str, Dict[str, Any]] = {}
    for label, policy_obj in (("random", _Random()), ("pacing", _Pacing())):
        episodes: List[EpisodeResult] = []
        for ep in range(5):
            ep_rng = random.Random(9000 + ep)
            opponents = _sample_persona_slate(ep_rng, persona_names, n=cfg.n_personas, jitter_scale=1.0)
            episodes.append(run_episode(seed=9000 + ep, cfg=cfg,
                                        trained_policy=policy_obj.bid, opponents=opponents))
        out[label] = _aggregate(episodes, mode=f"baseline.{label}")
    return out


# ---------------------------------------------------------------------------
# Top-level orchestrator (callable from inference.py + script CLI)
# ---------------------------------------------------------------------------

def run_advertiser_eval(
    *,
    task_name: str = "arena_hard",
    checkpoint: Optional[str] = None,
    opponent_checkpoint: Optional[str] = None,
    n_standard: int = 20,
    n_edge_per_sub: int = 10,
    n_selfplay: int = 10,
    out_path: Path = Path("results/advertiser_eval.json"),
    include_baselines: bool = True,
    only_mode: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = ARENA_TASKS[task_name]
    trained_policy = _load_policy(checkpoint, fallback_name="mock_trained")

    per_mode: Dict[str, Any] = {}
    if only_mode in (None, "standard"):
        per_mode["standard"] = run_standard_mode(cfg, trained_policy, n_episodes=n_standard)
    if only_mode in (None, "edge"):
        per_mode["edge"] = run_edge_mode(cfg, trained_policy, n_per_subcondition=n_edge_per_sub)
    if only_mode in (None, "selfplay"):
        opponent_policy = _load_policy(opponent_checkpoint, fallback_name="mock_v1")
        per_mode["selfplay"] = run_selfplay_mode(
            cfg, trained_policy=trained_policy,
            opponent_policy=opponent_policy, n_episodes=n_selfplay,
        )

    payload: Dict[str, Any] = {
        "task": task_name,
        "checkpoint": checkpoint or "<mock>",
        "opponent_checkpoint": opponent_checkpoint or "<mock>",
        "config": {
            "days": cfg.days,
            "impressions_per_day": cfg.impressions_per_day,
            "weekly_budget": cfg.weekly_budget,
            "daily_budget": cfg.daily_budget,
            "n_personas": cfg.n_personas,
            "target_weekly_roas": cfg.target_weekly_roas,
            "frequency_cap_per_user": cfg.frequency_cap_per_user,
        },
        "per_mode": per_mode,
    }

    if include_baselines:
        payload["baselines"] = _baselines_for_eval(cfg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    parser.add_argument("--task", type=str, default="arena_hard",
                        choices=list(ARENA_TASKS.keys()))
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to the trained advertiser LoRA checkpoint. "
                             "Omit to use the built-in mock trained policy.")
    parser.add_argument("--opponent-checkpoint", type=str, default=None,
                        help="Path to an earlier advertiser checkpoint (selfplay 'v1').")
    parser.add_argument("--n-standard", type=int, default=20)
    parser.add_argument("--n-edge-per-sub", type=int, default=10)
    parser.add_argument("--n-selfplay", type=int, default=10)
    parser.add_argument("--only-mode", choices=["standard", "edge", "selfplay"], default=None)
    parser.add_argument("--no-baselines", action="store_true",
                        help="Skip the random + pacing baseline runs (faster).")
    parser.add_argument("--out", type=str, default="results/advertiser_eval.json")
    args = parser.parse_args()

    payload = run_advertiser_eval(
        task_name=args.task,
        checkpoint=args.checkpoint,
        opponent_checkpoint=args.opponent_checkpoint,
        n_standard=args.n_standard,
        n_edge_per_sub=args.n_edge_per_sub,
        n_selfplay=args.n_selfplay,
        out_path=Path(args.out),
        include_baselines=not args.no_baselines,
        only_mode=args.only_mode,
    )

    print(f"[eval] wrote {args.out}")
    for mode, payload_mode in payload.get("per_mode", {}).items():
        if not isinstance(payload_mode, dict):
            continue
        print(
            f"[eval] {mode:>10s}  weekly_roas={payload_mode.get('weekly_roas_mean', 0.0):.3f}  "
            f"bid_precision={payload_mode.get('bid_precision_mean', 0.0):.3f}  "
            f"depl_day={payload_mode.get('budget_depletion_day_mean', 0.0):.2f}  "
            f"fatigue_sens={payload_mode.get('fatigue_sensitivity_mean', 0.0):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
