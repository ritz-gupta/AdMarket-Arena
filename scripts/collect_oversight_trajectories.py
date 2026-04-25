"""
Collect oversight training trajectories.

Runs a *frozen advertiser policy* (rule-based pacing baseline) through
the AdMarket Arena env with `violation_injector` enabled, captures the
resulting auction logs + ground-truth violations + heuristic-oversight
predictions, and saves to a JSONL file.

Each line of the output JSONL is one episode-day with the schema:

    {
      "episode_id": str,
      "day": int,
      "observation": <OversightObservation as dict>,
      "ground_truth": [<GroundTruthViolation>, ...],
      "heuristic_predictions": [<ViolationFlag>, ...]
    }

Plan 2 phase 4a uses this dataset to train the LLM OversightAgent
without requiring the trained advertiser checkpoint to exist yet
(decouples oversight training from advertiser convergence).

Plan 1's `arena_env.py` is required to actually run this script. Until
Plan 1 is built, run the synthetic-data fallback (--synthetic) which
fabricates plausible auction logs from violation_injector alone for
use in tests and as a sanity dataset.

Usage:
    python -m scripts.collect_oversight_trajectories \\
        --episodes 200 \\
        --task arena_hard \\
        --seed 42 \\
        --out data/oversight_train_trajectories.jsonl

Synthetic mode (no env needed):
    python -m scripts.collect_oversight_trajectories \\
        --episodes 50 --synthetic --seed 42 \\
        --out data/oversight_train_trajectories.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Make the package importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models import (  # noqa: E402
    AuctionRecord,
    CampaignStateSummary,
    GroundTruthViolation,
    OversightObservation,
    ViolationFlag,
)
from oversight import HeuristicOversightAgent  # noqa: E402
from violation_injector import (  # noqa: E402
    EpisodeViolationPlan,
    ViolationInjector,
    make_synthetic_user_id_pool,
    _FreqCapPlan,
    _ShillBiddingPlan,
)


# ---------------------------------------------------------------------------
# Synthetic generator (used until Plan 1 env is ready)
# ---------------------------------------------------------------------------

_SEGMENTS = [
    "gen_z_creator", "millennial_parent", "business_pro",
    "casual_scroller", "fitness_enthusiast", "bargain_hunter",
]
_ADVERTISER_NAMES = [
    "PremiumBrand", "BargainReseller", "PerformanceMarketer",
    "SpamFlooder", "CautiousChallenger",
]


def _synthetic_episode(
    rng: random.Random,
    n_advertisers: int,
    n_days: int,
    impressions_per_day: int,
    frequency_cap: int,
    daily_budget: float,
    floor_price_base: float,
    floor_price_daily_increment: float,
    injection_probability: float,
) -> List[Dict]:
    """Generate one synthetic episode worth of per-day OversightObservations.

    Stages each of the three violation types deliberately so the
    oversight pipeline can be tested end-to-end without depending on
    Plan 1's env. Each step:

      1. Decide whether this step is "scripted" for an active freq_cap
         or shill_bidding plan; if so, force the user_id and at least
         one bidder's bid value accordingly.
      2. All other advertisers bid via a noisy pacing baseline.
      3. Auction resolves with second-price + reserve.
      4. `no_contest` is defined as `<= 1 valid bid above floor` (the
         standard ad-tech interpretation).
      5. Budget-overspend injection is realized by inflating clearing
         prices on injected days for the targeted advertiser, so
         spent_today actually exceeds daily_budget by episode end.

    NOT a substitute for the real env — Plan 1's `arena_env.py` will
    replace this with the real auction loop. Used here for tests and
    as a starter dataset.
    """
    seed = rng.randint(0, 1_000_000_000)
    inj = ViolationInjector(persona_violation_probability=injection_probability)
    user_pool = make_synthetic_user_id_pool(n_users=30)

    plan: EpisodeViolationPlan = inj.sample_episode_plan(
        n_advertisers=n_advertisers,
        n_days=n_days,
        impressions_per_day=impressions_per_day,
        frequency_cap_per_user=frequency_cap,
        daily_budget_cap=daily_budget,
        floor_price_base=floor_price_base,
        floor_price_daily_increment=floor_price_daily_increment,
        candidate_user_ids=user_pool,
        seed=seed,
    )

    episode_id = f"ep_{seed}"
    output: List[Dict] = []
    spent_today: Dict[int, float] = {a: 0.0 for a in range(n_advertisers)}
    spent_total: Dict[int, float] = {a: 0.0 for a in range(n_advertisers)}
    impressions_today: Dict[int, int] = {a: 0 for a in range(n_advertisers)}
    clicks_today: Dict[int, int] = {a: 0 for a in range(n_advertisers)}
    auction_log_global_step = 0

    for day in range(n_days):
        for a in spent_today:
            spent_today[a] = 0.0
            impressions_today[a] = 0
            clicks_today[a] = 0
        floor_price = plan.floor_price_for_day(day)
        day_log: List[AuctionRecord] = []

        active_freq_caps = [p for p in plan.freq_cap_plans if p.day == day]
        active_shill = [p for p in plan.shill_plans if p.day == day]
        active_budget = {p.advertiser_id for p in plan.budget_plans if p.day == day}

        # Pre-load budget-injected advertisers near the cap so accumulating
        # auctions (with 1.8x clearing inflation below) deterministically
        # push them past `daily_budget_cap`. Mirrors how the real env's
        # injector lifts the persona's enforcement: in the real env the
        # persona's own bids are unconstrained for that day; here we
        # short-circuit the integration and just pre-spend.
        for advertiser_id in active_budget:
            spent_today[advertiser_id] = daily_budget * 0.95

        # Reserve a fraction of steps for scripted freq_cap bursts and
        # shill scenarios so ground truth actually materializes.
        scripted_steps_freq: List[Tuple[int, _FreqCapPlan]] = []
        for fp in active_freq_caps:
            burst_target = fp.burst_size
            for offset in range(burst_target):
                if offset < impressions_per_day:
                    scripted_steps_freq.append((offset, fp))

        scripted_steps_shill: List[Tuple[int, _ShillBiddingPlan]] = []
        for sp in active_shill:
            for offset in range(sp.n_consecutive):
                step_idx = (impressions_per_day - 1) - offset
                if step_idx >= 0 and step_idx not in {s for s, _ in scripted_steps_freq}:
                    scripted_steps_shill.append((step_idx, sp))

        scripted_freq_by_step = {step_idx: fp for step_idx, fp in scripted_steps_freq}
        scripted_shill_by_step = {step_idx: sp for step_idx, sp in scripted_steps_shill}

        for step_in_day in range(impressions_per_day):
            forced_freq = scripted_freq_by_step.get(step_in_day)
            forced_shill = scripted_shill_by_step.get(step_in_day)

            if forced_freq is not None:
                user_id = forced_freq.target_user_id
            else:
                user_id = rng.choice(user_pool)
            user_segment = rng.choice(_SEGMENTS)

            bids: List[Dict] = []
            for advertiser_id in range(n_advertisers):
                pacing_remaining_fraction = max(
                    0.05,
                    1.0 - spent_today[advertiser_id] / max(daily_budget, 1e-6),
                )
                base_bid = rng.uniform(0.4, 1.4) * (0.5 + 0.5 * pacing_remaining_fraction)
                creative_id = rng.randrange(8)

                if forced_freq is not None and advertiser_id == forced_freq.advertiser_id:
                    bid_value = max(floor_price * 4.0, base_bid * 5.0)
                elif forced_shill is not None and advertiser_id == forced_shill.advertiser_id:
                    bid_value = floor_price + 0.01
                    forced_shill.consumed += 1
                elif forced_shill is not None and advertiser_id != forced_shill.advertiser_id:
                    bid_value = floor_price * 0.5
                else:
                    bid_value = base_bid

                bids.append({
                    "advertiser_id": advertiser_id,
                    "bid": float(bid_value),
                    "creative_id": creative_id,
                })

            valid_bids = [b for b in bids if b["bid"] >= floor_price]
            no_contest = len(valid_bids) <= 1

            winner_bid: Dict | None = None
            clearing_price = 0.0
            if valid_bids:
                valid_bids.sort(key=lambda b: b["bid"], reverse=True)
                winner_bid = valid_bids[0]
                second = valid_bids[1]["bid"] if len(valid_bids) > 1 else floor_price
                clearing_price = max(second, floor_price)
                if winner_bid["advertiser_id"] in active_budget:
                    clearing_price *= 1.8

            for b in bids:
                won = (winner_bid is not None) and (b["advertiser_id"] == winner_bid["advertiser_id"])
                day_log.append(
                    AuctionRecord(
                        step=auction_log_global_step,
                        day=day,
                        step_in_day=step_in_day,
                        user_id=user_id,
                        user_segment=user_segment,
                        advertiser_id=b["advertiser_id"],
                        bid=b["bid"],
                        won=won,
                        clearing_price=clearing_price if won else 0.0,
                        floor_price=floor_price,
                        no_contest=no_contest,
                        creative_id=b["creative_id"],
                    )
                )

            if winner_bid is not None:
                wid = winner_bid["advertiser_id"]
                spent_today[wid] += clearing_price
                spent_total[wid] += clearing_price
                impressions_today[wid] += 1
                if rng.random() < 0.18:
                    clicks_today[wid] += 1
                plan.notify_freq_burst_win(wid, day, user_id)

            auction_log_global_step += 1

        campaign_states = [
            CampaignStateSummary(
                advertiser_id=a,
                advertiser_name=_ADVERTISER_NAMES[a % len(_ADVERTISER_NAMES)],
                spent_today=round(spent_today[a], 2),
                daily_budget_cap=daily_budget,
                spent_total=round(spent_total[a], 2),
                weekly_budget_cap=daily_budget * n_days,
                impressions_today=impressions_today[a],
                clicks_today=clicks_today[a],
            )
            for a in range(n_advertisers)
        ]

        observation = OversightObservation(
            day=day,
            auction_log=day_log,
            campaign_states=campaign_states,
            floor_price=floor_price,
            frequency_cap_per_user=frequency_cap,
            advertiser_names={a: _ADVERTISER_NAMES[a % len(_ADVERTISER_NAMES)] for a in range(n_advertisers)},
        )

        ground_truth = plan.all_ground_truth_for_day(day, actual_spent=spent_today)

        heuristic = HeuristicOversightAgent()
        heuristic_predictions = heuristic.flag_day(observation)

        output.append({
            "episode_id": episode_id,
            "day": day,
            "observation": observation.model_dump(),
            "ground_truth": [v.model_dump() for v in ground_truth],
            "heuristic_predictions": [p.model_dump() for p in heuristic_predictions],
        })

    return output


# ---------------------------------------------------------------------------
# Real-env collector (Plan 1 dependency)
# ---------------------------------------------------------------------------

def _real_env_episode(*args, **kwargs):
    raise NotImplementedError(
        "Real-env trajectory collection requires Plan 1's "
        "server/arena_env.py to be implemented. Use --synthetic for now."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--task", type=str, default="arena_hard",
                        choices=["arena_easy", "arena_medium", "arena_hard"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/oversight_train_trajectories.jsonl")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data generator (Plan 1-independent).")
    parser.add_argument("--injection-probability", type=float, default=0.30,
                        help="Per-persona probability of any violations.")
    args = parser.parse_args()

    task_to_dims = {
        "arena_easy":   {"days": 3, "imp_per_day": 20, "n_adv": 3, "freq_cap": 999, "daily_budget": 100.0,
                          "floor_base": 0.0,  "floor_inc": 0.0},
        "arena_medium": {"days": 5, "imp_per_day": 30, "n_adv": 4, "freq_cap": 5,   "daily_budget": 100.0,
                          "floor_base": 0.25, "floor_inc": 0.05},
        "arena_hard":   {"days": 7, "imp_per_day": 50, "n_adv": 5, "freq_cap": 3,   "daily_budget": 143.0,
                          "floor_base": 0.50, "floor_inc": 0.10},
    }
    dims = task_to_dims[args.task]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    n_lines = 0
    n_truth = 0
    n_pred = 0
    print(f"[collect] task={args.task} episodes={args.episodes} synthetic={args.synthetic}")

    with out_path.open("w") as f:
        for ep_idx in range(args.episodes):
            if args.synthetic:
                day_records = _synthetic_episode(
                    rng=rng,
                    n_advertisers=dims["n_adv"],
                    n_days=dims["days"],
                    impressions_per_day=dims["imp_per_day"],
                    frequency_cap=dims["freq_cap"],
                    daily_budget=dims["daily_budget"],
                    floor_price_base=dims["floor_base"],
                    floor_price_daily_increment=dims["floor_inc"],
                    injection_probability=args.injection_probability,
                )
            else:
                day_records = _real_env_episode()

            for record in day_records:
                f.write(json.dumps(record))
                f.write("\n")
                n_lines += 1
                n_truth += len(record["ground_truth"])
                n_pred += len(record["heuristic_predictions"])

            if (ep_idx + 1) % 10 == 0:
                print(f"[collect] {ep_idx + 1}/{args.episodes} episodes "
                      f"({n_lines} day-records, "
                      f"{n_truth} truths, {n_pred} heuristic predictions)")

    print(f"[collect] DONE: wrote {n_lines} day-records to {out_path}")
    print(f"[collect] ground-truth violations: {n_truth}, heuristic predictions: {n_pred}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
