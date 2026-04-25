"""
Day-end summariser for AdMarket Arena.

summarize_day() produces a ~200-token natural-language recap that is
injected into the AuctionObservation.yesterday_recap field at the start
of each new day. This is the Theme 2 long-horizon mechanism: the agent
reads a compact summary of yesterday instead of replaying raw history,
letting it plan across days even though the full episode exceeds its
context window.

No Pydantic / OpenEnv dependency — pure Python so this can be used
in offline analysis scripts as well as the live server.
"""

from __future__ import annotations

from typing import Dict, List, Optional

try:
    from campaign_state import AdvertiserCampaignState
except ImportError:
    from .campaign_state import AdvertiserCampaignState


def summarize_day(
    state: AdvertiserCampaignState,
    auction_log: List[Dict],
    total_steps: int = 350,
) -> str:
    """Generate a ~200-token natural-language day recap for the trained agent.

    Args:
        state: the trained advertiser's AdvertiserCampaignState *before*
               reset_day() is called (so daily accumulators are still set).
        auction_log: list of per-step dicts from the current day, each
                     containing keys:
                       step, winner_id, clearing_price, no_contest,
                       user_segment, clicked (optional, winner only)
        total_steps: total episode length (used to compute days remaining).

    Returns:
        A compact plain-text string suitable for inclusion in a system
        or user message. Targets ~180–220 tokens.
    """
    day = state.day_number
    days_remaining = 7 - day  # days left after this one

    # ----------------------------------------------------------------
    # Budget summary
    # ----------------------------------------------------------------
    budget_pct = 100.0 * state.spent_today / max(0.01, state.daily_budget)
    weekly_remaining = max(0.0, state.weekly_budget - state.spent_total)
    daily_remaining = max(0.0, state.daily_budget - state.spent_today)

    # ----------------------------------------------------------------
    # Auction outcomes for this day
    # ----------------------------------------------------------------
    day_auctions = len(auction_log)
    day_wins = state.wins_today
    win_rate = 100.0 * day_wins / max(1, day_auctions)

    # CTR
    ctr = 100.0 * state.clicks_today / max(1, day_wins) if day_wins > 0 else 0.0

    # Daily ROAS
    daily_roas = state.daily_roas

    # Average clearing price we paid
    our_prices = [
        e["clearing_price"]
        for e in auction_log
        if e.get("winner_id") == state.advertiser_id
    ]
    avg_clearing = sum(our_prices) / len(our_prices) if our_prices else 0.0

    # ----------------------------------------------------------------
    # Competitor price context (all auction clearing prices)
    # ----------------------------------------------------------------
    all_prices = [e["clearing_price"] for e in auction_log if e.get("clearing_price", 0) > 0]
    if all_prices:
        market_min = min(all_prices)
        market_max = max(all_prices)
        market_avg = sum(all_prices) / len(all_prices)
    else:
        market_min = market_max = market_avg = 0.0

    # ----------------------------------------------------------------
    # Fatigue snapshot — top 3 most fatigued segments
    # ----------------------------------------------------------------
    fatigue_items = sorted(
        state.per_segment_fatigue.items(), key=lambda x: -x[1]
    )[:3]
    fatigue_str = ", ".join(
        f"{seg}={v:.2f}" for seg, v in fatigue_items
    ) if fatigue_items else "none"

    # ----------------------------------------------------------------
    # Objective progress note
    # ----------------------------------------------------------------
    obj_notes = _objective_note(state)

    # ----------------------------------------------------------------
    # Compose the recap string
    # ----------------------------------------------------------------
    recap = (
        f"=== Day {day} recap ===\n"
        f"Budget: spent ${state.spent_today:.2f}/${state.daily_budget:.2f} "
        f"({budget_pct:.0f}%). Weekly remaining: ${weekly_remaining:.2f} "
        f"over {days_remaining} day(s).\n"
        f"Auctions: won {day_wins}/{day_auctions} ({win_rate:.0f}% win rate). "
        f"CTR {ctr:.1f}%. Daily ROAS {daily_roas:.2f}x.\n"
        f"Avg price paid: ${avg_clearing:.2f}. "
        f"Market range: ${market_min:.2f}–${market_max:.2f} (avg ${market_avg:.2f}).\n"
        f"Fatigue: {fatigue_str}.\n"
        f"{obj_notes}"
        f"=== End Day {day} ==="
    )
    return recap


def _objective_note(state: AdvertiserCampaignState) -> str:
    """One-line KPI progress note tailored to the advertiser's objective."""
    if state.objective_type == "awareness":
        reached = len(state.unique_users_reached)
        return f"Awareness: {reached} unique users reached so far.\n"
    elif state.objective_type == "conversion":
        total_ctr = (
            100.0 * state.clicks_total / max(1, state.impressions_total)
            if state.impressions_total > 0 else 0.0
        )
        return f"Conversion: weekly CTR {total_ctr:.1f}% ({state.clicks_total} clicks / {state.impressions_total} impressions).\n"
    elif state.objective_type == "retention":
        repeat = state.repeat_engagements
        return f"Retention: {repeat} users with ≥2 clicks so far.\n"
    return ""


def empty_recap(day_number: int) -> str:
    """Placeholder recap used on Day 1 when no prior day exists."""
    return f"=== Day 1 start === No prior day data. Weekly campaign begins now. ==="
