#!/usr/bin/env python3
"""
Recap ablation + hint-follower benchmark for AdMarket Arena.

Operationalises the doubt:
  *Is yesterday_recap doing the long-horizon work, or is it leaking the
  optimal pacing answer?*

Two complementary checks are produced:

PART A — Recap-content ablation
  Run the same agent (default: ArenaRecapFollowerBot, which only reads
  the recap text) under five recap-content variants:

    full              full production recap (baseline)
    no_recap          empty placeholder regardless of day (lower bound)
    stats_only        drop the two leaky lines (weekly_remaining + market avg)
    leak_only         keep ONLY the leaky lines
    numbers_shuffled  full template with numeric values perturbed +/-30%

  Interpretation:
    - If `full` ROAS >> `no_recap` ROAS → recap is informative.
    - If `leak_only` ROAS ~= `full` ROAS → the two leaky lines carry the
      bulk of the planning signal.
    - If `stats_only` ROAS ~= `full` ROAS → the agent is NOT relying on
      the leaky lines (best honest result for the long-horizon claim).
    - If `numbers_shuffled` ROAS ~= `full` ROAS → the agent isn't really
      reading the numbers (also informative, but for a different reason).

PART B — Hint-follower benchmark
  Compare ArenaRecapFollowerBot vs the existing baselines (random /
  greedy / pacing) under the production `full` recap. Establishes the
  "how much can a regex parser of the recap achieve?" floor that any
  trained LLM agent must convincingly beat to claim long-horizon
  reasoning. Plug a trained LLM bidder into ``--policy`` once it's
  ready (Plan 3 deliverable).

Usage:
  python scripts/recap_ablation.py --task arena_easy --episodes 5 --seed 42

Outputs:
  STDOUT  Two formatted tables
  JSON    results/recap_ablation.json (per-mode + per-agent stats)
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

# Allow running as `python scripts/recap_ablation.py` from the repo root.
# Package import path works because `pip install -e .` exposes
# ``meta_ad_optimizer`` as a top-level package mapped at the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from meta_ad_optimizer.baseline import (  # noqa: E402
    ArenaBaseAgent,
    ArenaGreedyAgent,
    ArenaPacingAgent,
    ArenaRandomAgent,
    ArenaRecapFollowerBot,
)
from meta_ad_optimizer.server.arena_env import AdMarketArenaEnvironment  # noqa: E402
from meta_ad_optimizer.summarizer import RECAP_MODES  # noqa: E402


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------

def _run_episodes(
    env: AdMarketArenaEnvironment,
    agent: ArenaBaseAgent,
    task: str,
    episodes: int,
    base_seed: int,
) -> Dict[str, List[float]]:
    """Run *episodes* of *agent* in *env* and return per-episode metrics."""
    roas: List[float] = []
    rewards: List[float] = []
    skip_rates: List[float] = []

    for ep in range(episodes):
        obs = env.reset(seed=base_seed + ep, task=task)
        ep_reward = 0.0
        skips = 0
        steps = 0
        while not obs.done:
            action = agent.act(obs)
            if action.skip:
                skips += 1
            obs = env.step(action)
            ep_reward += obs.reward or 0.0
            steps += 1
        roas.append(env.state.weekly_roas)
        rewards.append(round(ep_reward, 4))
        skip_rates.append(skips / max(1, steps))

    return {
        "weekly_roas": roas,
        "rewards": rewards,
        "skip_rate": skip_rates,
    }


def _summarise(label: str, results: Dict[str, List[float]]) -> Dict[str, float]:
    """Mean+std collapse of one (label, results) pair for the JSON dump."""
    roas = results["weekly_roas"]
    rew = results["rewards"]
    skp = results["skip_rate"]
    return {
        "label": label,
        "n_episodes": len(roas),
        "weekly_roas_mean": round(statistics.mean(roas), 4) if roas else 0.0,
        "weekly_roas_std": round(statistics.stdev(roas), 4) if len(roas) > 1 else 0.0,
        "reward_mean": round(statistics.mean(rew), 4) if rew else 0.0,
        "reward_std": round(statistics.stdev(rew), 4) if len(rew) > 1 else 0.0,
        "skip_rate_mean": round(statistics.mean(skp), 4) if skp else 0.0,
    }


# ---------------------------------------------------------------------------
# Policy registry (for --policy switch)
# ---------------------------------------------------------------------------

def _make_policy(name: str, seed: int) -> ArenaBaseAgent:
    """Construct a policy by name. Plug a trained LLM bidder in here later."""
    name = name.lower()
    if name == "recap_follower":
        return ArenaRecapFollowerBot()
    if name == "pacing":
        return ArenaPacingAgent()
    if name == "greedy":
        return ArenaGreedyAgent()
    if name == "random":
        return ArenaRandomAgent(random.Random(seed))
    raise ValueError(f"Unknown policy {name!r}. Choose recap_follower|pacing|greedy|random.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recap content ablation + hint-follower benchmark.",
    )
    parser.add_argument("--task", default="arena_easy",
                        choices=("arena_easy", "arena_medium", "arena_hard"))
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodes per recap variant / per baseline.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--policy", default="recap_follower",
        help="Policy under test for Part A. recap_follower (default) is the "
             "honest stand-in until a trained LLM bidder is plugged in.",
    )
    parser.add_argument("--out", default="results/recap_ablation.json",
                        help="JSON path for machine-readable results.")
    parser.add_argument("--skip-part-b", action="store_true",
                        help="Skip Part B (hint-follower benchmark).")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, object] = {
        "task": args.task,
        "episodes_per_cell": args.episodes,
        "seed": args.seed,
        "policy_under_test": args.policy,
        "recap_modes": list(RECAP_MODES),
        "wall_clock_seconds": 0.0,
    }
    t0 = time.time()

    # ----------------------------------------------------------------
    # Part A — recap-content ablation on the policy under test
    # ----------------------------------------------------------------
    print(f"\n=== Part A: recap-content ablation on '{args.policy}' "
          f"({args.task}, {args.episodes} eps/mode, seed={args.seed}) ===")
    print(f"  {'recap_mode':<18} {'weekly_roas':>14} "
          f"{'reward':>12} {'skip_rate':>10}")
    print("  " + "-" * 60)

    part_a: Dict[str, Dict[str, float]] = {}
    env = AdMarketArenaEnvironment()
    for mode in RECAP_MODES:
        env.set_recap_mode(mode)
        agent = _make_policy(args.policy, args.seed)
        results = _run_episodes(env, agent, args.task, args.episodes, args.seed)
        summary = _summarise(label=mode, results=results)
        part_a[mode] = summary
        print(f"  {mode:<18} "
              f"{summary['weekly_roas_mean']:>8.3f}+/-{summary['weekly_roas_std']:<5.3f}"
              f"  {summary['reward_mean']:>8.2f}+/-{summary['reward_std']:<3.2f}"
              f"  {summary['skip_rate_mean']:>9.2f}")

    # Quick interpretation hints
    full_roas = part_a["full"]["weekly_roas_mean"]
    no_roas = part_a["no_recap"]["weekly_roas_mean"]
    stats_roas = part_a["stats_only"]["weekly_roas_mean"]
    leak_roas = part_a["leak_only"]["weekly_roas_mean"]
    shuffled_roas = part_a["numbers_shuffled"]["weekly_roas_mean"]

    def _delta_pct(a: float, b: float) -> float:
        return 100.0 * (a - b) / max(1e-6, abs(b)) if b else 0.0

    print("\n  Interpretation:")
    print(f"    full vs no_recap         dROAS = {_delta_pct(full_roas, no_roas):+6.1f}%   "
          f"(positive => recap carries information)")
    print(f"    leak_only vs full        dROAS = {_delta_pct(leak_roas, full_roas):+6.1f}%   "
          f"(near-zero => the leaky lines ARE the recap)")
    print(f"    stats_only vs full       dROAS = {_delta_pct(stats_roas, full_roas):+6.1f}%   "
          f"(near-zero => agent doesn't lean on leaky lines  [good for L-H claim])")
    print(f"    shuffled vs full         dROAS = {_delta_pct(shuffled_roas, full_roas):+6.1f}%   "
          f"(near-zero => agent ignores the numbers)")

    # ----------------------------------------------------------------
    # Part B — hint-follower vs other baselines under full recap
    # ----------------------------------------------------------------
    part_b: Dict[str, Dict[str, float]] = {}
    if not args.skip_part_b:
        print(f"\n=== Part B: hint-follower benchmark "
              f"(full recap, {args.task}, {args.episodes} eps) ===")
        print(f"  {'agent':<22} {'weekly_roas':>14} {'reward':>12} {'skip_rate':>10}")
        print("  " + "-" * 64)

        env_b = AdMarketArenaEnvironment()
        env_b.set_recap_mode("full")
        agent_specs = [
            ("ArenaRandom", lambda s: ArenaRandomAgent(random.Random(s))),
            ("ArenaGreedy", lambda s: ArenaGreedyAgent()),
            ("ArenaPacing", lambda s: ArenaPacingAgent()),
            ("ArenaRecapFollower", lambda s: ArenaRecapFollowerBot()),
        ]
        for label, factory in agent_specs:
            agent = factory(args.seed)
            results = _run_episodes(env_b, agent, args.task, args.episodes, args.seed)
            summary = _summarise(label=label, results=results)
            part_b[label] = summary
            print(f"  {label:<22} "
                  f"{summary['weekly_roas_mean']:>8.3f}+/-{summary['weekly_roas_std']:<5.3f}"
                  f"  {summary['reward_mean']:>8.2f}+/-{summary['reward_std']:<3.2f}"
                  f"  {summary['skip_rate_mean']:>9.2f}")

        rf = part_b["ArenaRecapFollower"]["weekly_roas_mean"]
        pacing = part_b["ArenaPacing"]["weekly_roas_mean"]
        print("\n  Interpretation:")
        print(f"    A trained LLM bidder must beat ArenaRecapFollower (ROAS={rf:.3f}) "
              f"by a meaningful margin to claim it isn't just a regex.")
        print(f"    Reference: ArenaPacing structured-obs heuristic = {pacing:.3f}.")

    # ----------------------------------------------------------------
    # Persist
    # ----------------------------------------------------------------
    payload["wall_clock_seconds"] = round(time.time() - t0, 2)
    payload["part_a_recap_ablation"] = part_a
    payload["part_b_hint_follower_benchmark"] = part_b
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[recap-ablation] wrote {out_path}  ({payload['wall_clock_seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
