"""
Second-price (Vickrey) auction for AdMarket Arena.

Completely stateless — takes a dict of bids, applies filtering rules,
and returns the winner + clearing price. No Pydantic dependency so
this module can be imported anywhere without OpenEnv being installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class AuctionResult:
    """Outcome of one impression-slot auction."""

    winner_id: Optional[str]    # None if no valid bids cleared the floor
    clearing_price: float       # price the winner pays (second-highest bid or floor)
    no_contest: bool = False    # True when ≤1 valid bidder (winner pays floor)
    all_bids: Dict[str, float] = field(default_factory=dict)  # adv_id → raw bid amount


def run_auction(
    bids: Dict[str, Tuple[float, bool]],
    floor_price: float,
    freq_caps: Dict[str, Dict[str, int]],
    freq_cap_limit: int,
    user_id: str,
) -> AuctionResult:
    """Run a second-price sealed-bid (Vickrey) auction for one impression slot.

    Args:
        bids: mapping of advertiser_id → (bid_amount, skip).
              skip=True means the advertiser voluntarily passes this slot.
        floor_price: minimum bid required to enter the auction.
                     Rises each day: 0.50 + 0.10 * day_number.
        freq_caps: per-advertiser per-user impression counts today.
                   Shape: {advertiser_id: {user_id: count_today}}.
        freq_cap_limit: max impressions one advertiser can win for a
                        single user in one day before being excluded.
        user_id: the user being shown the ad this step.

    Returns:
        AuctionResult with winner_id=None if no valid bids.

    Filtering order (an advertiser is excluded if ANY of these apply):
        1. skip=True
        2. bid_amount < floor_price
        3. freq_caps[adv_id][user_id] >= freq_cap_limit
    """
    # Collect raw bids for logging before any filtering
    all_bids = {adv_id: amt for adv_id, (amt, _) in bids.items()}

    # Build the active (eligible) bid pool
    active: Dict[str, float] = {}
    for adv_id, (bid_amount, skip) in bids.items():
        if skip:
            continue
        if bid_amount < floor_price:
            continue
        user_impressions = freq_caps.get(adv_id, {}).get(user_id, 0)
        if user_impressions >= freq_cap_limit:
            continue
        active[adv_id] = bid_amount

    if not active:
        return AuctionResult(
            winner_id=None,
            clearing_price=0.0,
            no_contest=True,
            all_bids=all_bids,
        )

    if len(active) == 1:
        # Single bidder — wins but only pays the floor (no competitive pressure)
        winner_id = next(iter(active))
        return AuctionResult(
            winner_id=winner_id,
            clearing_price=floor_price,
            no_contest=True,
            all_bids=all_bids,
        )

    # Two or more bidders — sort descending; break ties deterministically by id
    sorted_bids = sorted(active.items(), key=lambda x: (-x[1], x[0]))
    winner_id = sorted_bids[0][0]
    second_price = sorted_bids[1][1]

    # Winner pays the second-highest bid, but never less than the floor
    clearing_price = max(floor_price, second_price)

    return AuctionResult(
        winner_id=winner_id,
        clearing_price=round(clearing_price, 4),
        no_contest=False,
        all_bids=all_bids,
    )
