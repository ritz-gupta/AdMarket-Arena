#!/usr/bin/env python3
"""
Baseline inference script for the Meta Ad Optimizer.

Runs heuristic agents and (optionally) an LLM agent across all three
tasks and prints reproducible scores.

Usage (standalone — no server required):
    python -m meta_ad_optimizer.baseline --episodes 100 --seed 42

With LLM agent (requires OPENAI_API_KEY env var):
    OPENAI_API_KEY=sk-... python -m meta_ad_optimizer.baseline --episodes 5 --seed 42 --llm
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
from typing import Any, Dict, List, Optional

try:
    from .models import AdAction, AdObservation
    from .server.ad_environment import AdOptimizerEnvironment
    from .simulation import (
        SEGMENT_CATEGORY_AFFINITY,
        VALID_FORMATS,
        VALID_SURFACES,
        FORMAT_SURFACE_SYNERGY,
    )
    from .tasks import TASKS, grade_episode
except ImportError:
    # Allow `from baseline import ...` when the repo root is on sys.path
    # (e.g. Colab notebooks that don't install the package).
    from models import AdAction, AdObservation  # type: ignore[no-redef]
    from server.ad_environment import AdOptimizerEnvironment  # type: ignore[no-redef]
    from simulation import (  # type: ignore[no-redef]
        SEGMENT_CATEGORY_AFFINITY,
        VALID_FORMATS,
        VALID_SURFACES,
        FORMAT_SURFACE_SYNERGY,
    )
    from tasks import TASKS, grade_episode  # type: ignore[no-redef]

# Arena imports — optional so single-agent baseline still works without arena modules
try:
    try:
        from .models import AuctionAction, AuctionObservation
        from .server.arena_env import AdMarketArenaEnvironment
        from .tasks import ARENA_TASKS
    except ImportError:
        from models import AuctionAction, AuctionObservation  # type: ignore[no-redef]
        from server.arena_env import AdMarketArenaEnvironment  # type: ignore[no-redef]
        from tasks import ARENA_TASKS  # type: ignore[no-redef]
    _ARENA_AVAILABLE = True
except ImportError:
    _ARENA_AVAILABLE = False


# ---------------------------------------------------------------------------
# Agent base
# ---------------------------------------------------------------------------

class BaseAgent:
    name: str = "base"

    def act(self, obs: AdObservation) -> AdAction:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Random agent
# ---------------------------------------------------------------------------

class RandomAgent(BaseAgent):
    name = "Random"

    def __init__(self, rng: random.Random):
        self._rng = rng

    def act(self, obs: AdObservation) -> AdAction:
        n_creatives = len(obs.available_creatives)
        platform = self._rng.choice(["instagram", "facebook"])
        surfaces = VALID_SURFACES.get(platform, ["feed"])
        placement = self._rng.choice(surfaces)
        valid_fmts = VALID_FORMATS.get(placement, ["image"])
        ad_format = self._rng.choice(valid_fmts)

        return AdAction(
            show_ad=self._rng.random() > 0.2,
            creative_id=self._rng.randint(0, max(0, n_creatives - 1)),
            platform=platform,
            placement=placement,
            ad_format=ad_format,
        )


# ---------------------------------------------------------------------------
# Greedy-CTR agent
# ---------------------------------------------------------------------------

class GreedyAgent(BaseAgent):
    """Picks the creative with highest base_ctr and the surface
    with the highest placement factor.  No fatigue awareness."""

    name = "Greedy"

    def act(self, obs: AdObservation) -> AdAction:
        creatives = obs.available_creatives
        best_idx = 0
        best_ctr = -1.0
        for i, c in enumerate(creatives):
            if c.get("base_ctr", 0) > best_ctr:
                best_ctr = c["base_ctr"]
                best_idx = i

        platform = obs.current_platform
        placement = "reels" if "reels" in VALID_SURFACES.get(platform, []) else "feed"
        valid_fmts = VALID_FORMATS.get(placement, ["image"])
        ad_format = "reel" if "reel" in valid_fmts else valid_fmts[0]

        return AdAction(
            show_ad=True,
            creative_id=best_idx,
            platform=platform,
            placement=placement,
            ad_format=ad_format,
        )


# ---------------------------------------------------------------------------
# Rule-based agent
# ---------------------------------------------------------------------------

class RuleBasedAgent(BaseAgent):
    """Heuristic policy that:
    - matches creative target_segment to user
    - picks synergy-optimal format-surface combos
    - skips every 4th impression for fatigue recovery
    """

    name = "Rule-Based"

    def __init__(self):
        self._step = 0

    def act(self, obs: AdObservation) -> AdAction:
        self._step += 1

        if obs.fatigue_level > 0.5 or (self._step % 4 == 0 and obs.fatigue_level > 0.15):
            return AdAction(
                show_ad=False,
                creative_id=0,
                platform=obs.current_platform,
                placement=obs.current_surface,
                ad_format="image",
            )

        creatives = obs.available_creatives
        best_idx = 0
        best_score = -1.0
        for i, c in enumerate(creatives):
            score = c.get("base_ctr", 0)
            if c.get("target_segment") == obs.user_segment:
                score *= 1.5
            elif c.get("category") in obs.user_interests:
                score *= 1.0
            else:
                score *= 0.5
            if score > best_score:
                best_score = score
                best_idx = i

        platform = obs.current_platform
        surface = obs.current_surface
        if surface not in VALID_SURFACES.get(platform, []):
            surface = "feed"

        valid_fmts = VALID_FORMATS.get(surface, ["image"])
        best_fmt = valid_fmts[0]
        best_synergy = 0.0
        for fmt in valid_fmts:
            syn = FORMAT_SURFACE_SYNERGY.get((fmt, surface), 1.0)
            if syn > best_synergy:
                best_synergy = syn
                best_fmt = fmt

        return AdAction(
            show_ad=True,
            creative_id=best_idx,
            platform=platform,
            placement=surface,
            ad_format=best_fmt,
        )

    def reset(self):
        self._step = 0


# ---------------------------------------------------------------------------
# LLM agent (OpenAI API)
# ---------------------------------------------------------------------------

class LLMAgent(BaseAgent):
    """Agent that uses an OpenAI-compatible LLM to select actions.

    Reads OPENAI_API_KEY from environment. Constructs a prompt from
    the observation and parses the LLM's JSON response into an AdAction.
    Falls back to a random valid action on parse failure.
    """

    name = "LLM"

    SYSTEM_PROMPT = (
        "You are an ad optimization agent for Instagram and Facebook. "
        "Given the current user profile, available creatives, and session state, "
        "decide the best ad action. Respond ONLY with a JSON object:\n"
        '{"show_ad": bool, "creative_id": int, "platform": str, '
        '"placement": str, "ad_format": str}\n'
        "Valid platforms: instagram, facebook.\n"
        "Instagram surfaces: feed, reels, stories, explore, search.\n"
        "Facebook surfaces: feed, reels, stories, marketplace, search, right_column.\n"
        "Valid formats per surface:\n"
        "  feed: image, video, carousel, reel\n"
        "  reels: reel\n"
        "  stories: image, video, reel, collection\n"
        "  explore: image, video, carousel, reel\n"
        "  search: image, video\n"
        "  marketplace: image, carousel, collection\n"
        "  right_column: image\n"
        "Match the creative's target_segment/category to the user for best CTR. "
        "Skip (show_ad=false) when fatigue is high."
    )

    def __init__(self, model: str = "gpt-4o-mini"):
        self._model = model
        self._client: Any = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI()
            except ImportError:
                raise RuntimeError(
                    "openai package required for LLM agent. "
                    "Install with: pip install openai"
                )
        return self._client

    def _obs_to_prompt(self, obs: AdObservation) -> str:
        creatives_summary = []
        for c in obs.available_creatives:
            creatives_summary.append(
                f"  idx={c['pool_index']}: category={c['category']}, "
                f"tone={c['tone']}, target={c['target_segment']}, "
                f"ctr={c['base_ctr']:.3f}"
            )
        return (
            f"User: segment={obs.user_segment}, "
            f"interests={obs.user_interests}, device={obs.user_device}\n"
            f"Platform: {obs.current_platform}, surface: {obs.current_surface}\n"
            f"Step: {obs.step}/{obs.total_steps}, "
            f"fatigue: {obs.fatigue_level:.3f}, "
            f"impressions: {obs.impression_count}\n"
            f"Session CTR: {obs.session_metrics.get('ctr', 0):.4f}\n"
            f"Creatives:\n" + "\n".join(creatives_summary)
        )

    def act(self, obs: AdObservation) -> AdAction:
        client = self._get_client()
        prompt = self._obs_to_prompt(obs)

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=150,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content or "{}"
            data = json.loads(text)
            return AdAction(
                show_ad=bool(data.get("show_ad", True)),
                creative_id=int(data.get("creative_id", 0)),
                platform=str(data.get("platform", obs.current_platform)),
                placement=str(data.get("placement", obs.current_surface)),
                ad_format=str(data.get("ad_format", "image")),
            )
        except Exception:
            platform = obs.current_platform
            surface = obs.current_surface
            valid_fmts = VALID_FORMATS.get(surface, ["image"])
            return AdAction(
                show_ad=True,
                creative_id=0,
                platform=platform,
                placement=surface,
                ad_format=valid_fmts[0],
            )


# ---------------------------------------------------------------------------
# Claude Agent SDK agent
# ---------------------------------------------------------------------------

class ClaudeAgent(BaseAgent):
    """Agent that uses Anthropic's Claude via the Claude Agent SDK.

    Reads ANTHROPIC_API_KEY from environment. Constructs a prompt from
    the observation and parses the response JSON into an AdAction.
    Falls back to a skip action on parse failure.

    Usage requires:
        pip install claude-agent-sdk
    and ANTHROPIC_API_KEY set in the environment.

    CLI usage:
        ANTHROPIC_API_KEY=sk-ant-... python -m meta_ad_optimizer.baseline \\
            --episodes 5 --seed 42 --claude --claude-model claude-opus-4-6
    """

    name = "Claude"

    SYSTEM_PROMPT = (
        "You are an expert ad optimization agent for Instagram and Facebook. "
        "Given the current user profile, available creatives, and session state, "
        "decide the best ad action to maximise engagement while managing user fatigue.\n\n"
        "Respond ONLY with a JSON object — no markdown, no explanation:\n"
        '{"show_ad": bool, "creative_id": int, "platform": str, '
        '"placement": str, "ad_format": str}\n\n'
        "Valid platforms: instagram, facebook.\n"
        "Instagram surfaces: feed, reels, stories, explore, search.\n"
        "Facebook surfaces: feed, reels, stories, marketplace, search, right_column.\n"
        "Valid formats per surface:\n"
        "  feed: image, video, carousel, reel\n"
        "  reels: reel\n"
        "  stories: image, video, reel, collection\n"
        "  explore: image, video, carousel, reel\n"
        "  search: image, video\n"
        "  marketplace: image, carousel, collection\n"
        "  right_column: image\n"
        "Match creative target_segment/category to the user for best CTR. "
        "Skip (show_ad=false) when fatigue > 0.5 to preserve long-term engagement."
    )

    def __init__(self, model: str = "claude-opus-4-6"):
        self._model = model

    def _obs_to_prompt(self, obs: AdObservation) -> str:
        creatives_summary = []
        for c in obs.available_creatives:
            creatives_summary.append(
                f"  idx={c['pool_index']}: category={c['category']}, "
                f"tone={c['tone']}, target={c['target_segment']}, "
                f"ctr={c['base_ctr']:.3f}"
            )
        return (
            f"User: segment={obs.user_segment}, "
            f"interests={obs.user_interests}, device={obs.user_device}\n"
            f"Platform: {obs.current_platform}, surface: {obs.current_surface}\n"
            f"Step: {obs.step}/{obs.total_steps}, "
            f"fatigue: {obs.fatigue_level:.3f}, "
            f"impressions: {obs.impression_count}\n"
            f"Session CTR: {obs.session_metrics.get('ctr', 0):.4f}\n"
            f"Creatives:\n" + "\n".join(creatives_summary) + "\n\n"
            f"Respond with JSON only."
        )

    def act(self, obs: AdObservation) -> AdAction:
        try:
            import anyio
            from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage
        except ImportError:
            raise RuntimeError(
                "claude-agent-sdk and anyio are required for ClaudeAgent. "
                "Install with: pip install claude-agent-sdk anyio"
            )

        prompt = self._obs_to_prompt(obs)
        result_text: str = ""

        async def _run() -> str:
            async for message in query(
                prompt=prompt,
                options=ClaudeAgentOptions(
                    model=self._model,
                    system_prompt=self.SYSTEM_PROMPT,
                    allowed_tools=[],
                    max_turns=1,
                    permission_mode="default",
                ),
            ):
                if isinstance(message, ResultMessage):
                    return message.result or ""
            return ""

        try:
            result_text = anyio.run(_run)
        except Exception:
            pass

        # Parse the JSON response
        try:
            text = result_text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            data = json.loads(text)
            return AdAction(
                show_ad=bool(data.get("show_ad", True)),
                creative_id=int(data.get("creative_id", 0)),
                platform=str(data.get("platform", obs.current_platform)),
                placement=str(data.get("placement", obs.current_surface)),
                ad_format=str(data.get("ad_format", "image")),
            )
        except Exception:
            platform = obs.current_platform
            surface = obs.current_surface
            valid_fmts = VALID_FORMATS.get(surface, ["image"])
            return AdAction(
                show_ad=False,
                creative_id=0,
                platform=platform,
                placement=surface,
                ad_format=valid_fmts[0],
            )


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    task_name: str,
    agent: BaseAgent,
    episodes: int,
    base_seed: int,
) -> List[float]:
    """Run *episodes* and return per-episode grader scores."""
    env = AdOptimizerEnvironment()
    scores: List[float] = []

    for ep in range(episodes):
        obs = env.reset(seed=base_seed + ep, task=task_name)
        if hasattr(agent, "reset"):
            agent.reset()

        while not obs.done:
            action = agent.act(obs)
            obs = env.step(action)

        score = grade_episode(env.state)
        scores.append(score)

    return scores


# ---------------------------------------------------------------------------
# Arena bidders  (AuctionObservation → AuctionAction)
# ---------------------------------------------------------------------------

class ArenaBaseAgent:
    name: str = "arena_base"

    def act(self, obs: "AuctionObservation") -> "AuctionAction":  # type: ignore[name-defined]
        raise NotImplementedError


class ArenaRandomAgent(ArenaBaseAgent):
    """Bids a random amount and skips 20% of slots."""

    name = "ArenaRandom"

    def __init__(self, rng: random.Random):
        self._rng = rng

    def act(self, obs: "AuctionObservation") -> "AuctionAction":  # type: ignore[name-defined]
        if self._rng.random() < 0.20:
            return AuctionAction(skip=True, bid_amount=0.0, creative_id=0)
        bid = round(self._rng.uniform(0.0, 5.0), 4)
        creative_id = self._rng.randint(0, max(0, len(obs.available_creatives) - 1))
        return AuctionAction(skip=False, bid_amount=bid, creative_id=creative_id)


class ArenaGreedyAgent(ArenaBaseAgent):
    """Always bids maximum (5.0) — wins often but ignores fatigue and pacing."""

    name = "ArenaGreedy"

    def act(self, obs: "AuctionObservation") -> "AuctionAction":  # type: ignore[name-defined]
        creative_id = self._best_creative(obs)
        return AuctionAction(skip=False, bid_amount=5.0, creative_id=creative_id)

    @staticmethod
    def _best_creative(obs: "AuctionObservation") -> int:  # type: ignore[name-defined]
        best_idx, best_ctr = 0, -1.0
        for i, c in enumerate(obs.available_creatives):
            if c.get("base_ctr", 0.0) > best_ctr:
                best_ctr = c["base_ctr"]
                best_idx = i
        return best_idx


class ArenaPacingAgent(ArenaBaseAgent):
    """Budget-aware pacing bidder — should beat greedy by avoiding fatigue and overspend.

    Strategy:
      - Bid = remaining_budget / remaining_slots × 1.2 (slight markup to be competitive)
      - Skip if per-segment fatigue > 0.70 (preserve CTR for fresh segments)
      - Reduce bid 30% when recent market prices are expensive (> 1.5× floor)
      - Skip immediately if weekly budget exhausted
    """

    name = "ArenaPacing"
    _FATIGUE_SKIP_THRESHOLD = 0.70
    _EXPENSIVE_MARKET_RATIO = 1.5
    _EXPENSIVE_BID_DISCOUNT = 0.70
    _BID_MARKUP = 1.20

    def act(self, obs: "AuctionObservation") -> "AuctionAction":  # type: ignore[name-defined]
        creative_id = self._best_creative(obs)

        if obs.budget_remaining <= 0:
            return AuctionAction(skip=True, bid_amount=0.0, creative_id=creative_id)

        seg_fatigue = obs.per_segment_fatigue.get(obs.user_segment, 0.0)
        if seg_fatigue > self._FATIGUE_SKIP_THRESHOLD:
            return AuctionAction(skip=True, bid_amount=0.0, creative_id=creative_id)

        remaining_slots = max(1, obs.total_steps - obs.step)
        fair_share = obs.budget_remaining / remaining_slots

        # Back off in expensive markets
        if obs.recent_clearing_prices:
            avg_recent = sum(obs.recent_clearing_prices) / len(obs.recent_clearing_prices)
            if avg_recent > obs.floor_price * self._EXPENSIVE_MARKET_RATIO:
                fair_share *= self._EXPENSIVE_BID_DISCOUNT

        bid_amount = round(min(5.0, max(obs.floor_price, fair_share * self._BID_MARKUP)), 4)
        return AuctionAction(skip=False, bid_amount=bid_amount, creative_id=creative_id)

    @staticmethod
    def _best_creative(obs: "AuctionObservation") -> int:  # type: ignore[name-defined]
        best_idx, best_score = 0, -1.0
        for i, c in enumerate(obs.available_creatives):
            score = c.get("base_ctr", 0.0)
            if c.get("target_segment") == obs.user_segment:
                score *= 1.5
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx


class ArenaRecapFollowerBot(ArenaBaseAgent):
    """Naive hint-following bidder that uses ONLY the yesterday_recap text.

    Built to operationalise the doubt: *"Is yesterday_recap doing the
    long-horizon work, or is it just leaking the optimal pacing answer?"*
    If a hand-coded regex parser of the recap can hit competitive ROAS,
    then a trained LLM that beats this bot is doing genuine planning.
    If trained-LLM ≈ this bot, the recap is the entire signal.

    Decision rule (no other obs fields used apart from minimal scaffolding):
      1. Parse "Weekly remaining: $X over Y day(s)" → daily_share = X/(Y+1).
      2. Parse "Market range: ... avg $Z" → bid = clip(Z * 1.05, floor, 5.0).
      3. Parse "Fatigue: <seg>=<val>, ..." → if current segment > 0.7, skip.
      4. Fallback when recap is empty / no_recap mode: bid = 1.0, no skip.

    Reads ``obs.user_segment`` for fatigue lookup, ``obs.floor_price`` for
    the bid floor (env-imposed, not part of the recap), and
    ``obs.recent_clearing_prices`` only as a fallback when no market line
    is present in the recap. Everything else comes from the recap text.
    """

    name = "ArenaRecapFollower"

    _BUDGET_RE = re.compile(
        r"Weekly remaining:\s*\$([0-9]+(?:\.[0-9]+)?)\s+over\s+([0-9]+)\s+day"
    )
    _MARKET_RE = re.compile(
        r"Market range:.*?avg\s*\$([0-9]+(?:\.[0-9]+)?)"
    )
    _FATIGUE_RE = re.compile(
        r"([a-z_]+)=([0-9]+(?:\.[0-9]+)?)"
    )
    _FATIGUE_SKIP_THRESHOLD = 0.70

    def act(self, obs: "AuctionObservation") -> "AuctionAction":  # type: ignore[name-defined]
        recap = obs.yesterday_recap or ""

        # Derive fatigue map from recap text only
        fatigue_map: Dict[str, float] = {}
        if "Fatigue:" in recap:
            tail = recap.split("Fatigue:", 1)[1].split("\n", 1)[0]
            for seg, val in self._FATIGUE_RE.findall(tail):
                try:
                    fatigue_map[seg] = float(val)
                except ValueError:
                    pass

        if obs.user_segment in fatigue_map and fatigue_map[obs.user_segment] > self._FATIGUE_SKIP_THRESHOLD:
            return AuctionAction(skip=True, bid_amount=0.0, creative_id=0)

        # Derive bid from market line
        market_match = self._MARKET_RE.search(recap)
        if market_match:
            bid = float(market_match.group(1)) * 1.05
        elif obs.recent_clearing_prices:
            bid = (sum(obs.recent_clearing_prices) / len(obs.recent_clearing_prices)) * 1.05
        else:
            bid = 1.0

        # Sanity-check against the leaky budget hint: don't exceed what
        # the recap implies is "today's fair share" (X / (days_left + 1)).
        budget_match = self._BUDGET_RE.search(recap)
        if budget_match:
            weekly_remaining = float(budget_match.group(1))
            days_left = int(budget_match.group(2))
            slots_left_today = max(1, getattr(obs, "total_steps", 60) // max(1, 7) - getattr(obs, "step_in_day", 0))
            daily_share = weekly_remaining / max(1, days_left + 1)
            per_slot_share = daily_share / slots_left_today
            bid = min(bid, per_slot_share * 1.5)

        bid = max(obs.floor_price, min(5.0, bid))
        return AuctionAction(skip=False, bid_amount=round(bid, 4), creative_id=0)


class ArenaLLMBidder(ArenaBaseAgent):
    """STUB — Plan 3 fills the real Unsloth/GRPO-trained implementation.

    Wraps a trained Qwen2.5-3B checkpoint behind the same act() interface
    so it slots into run_arena_evaluation() without any loop changes.
    """

    name = "ArenaLLM"

    def act(self, obs: "AuctionObservation") -> "AuctionAction":  # type: ignore[name-defined]
        raise NotImplementedError(
            "ArenaLLMBidder is a stub. Plan 3 fills the implementation "
            "by loading the Unsloth checkpoint and calling inference."
        )


# ---------------------------------------------------------------------------
# Arena evaluation loop
# ---------------------------------------------------------------------------

def run_arena_evaluation(
    task_name: str,
    agent: ArenaBaseAgent,
    episodes: int,
    base_seed: int,
) -> Dict[str, List[float]]:
    """Run *episodes* of AdMarket Arena and return weekly ROAS + cumulative rewards.

    Returns:
        Dict with keys "weekly_roas" and "rewards", each a list of length *episodes*.
        weekly_roas is the primary comparison metric (plan requirement: pacing > greedy > random).
    """
    env = AdMarketArenaEnvironment()
    weekly_roas_list: List[float] = []
    reward_list: List[float] = []

    for ep in range(episodes):
        obs = env.reset(seed=base_seed + ep, task=task_name)
        ep_reward = 0.0

        while not obs.done:
            action = agent.act(obs)
            obs = env.step(action)
            ep_reward += obs.reward or 0.0

        weekly_roas_list.append(env.state.weekly_roas)
        reward_list.append(round(ep_reward, 4))

    return {"weekly_roas": weekly_roas_list, "rewards": reward_list}


def main():
    parser = argparse.ArgumentParser(description="Meta Ad Optimizer baselines")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--llm", action="store_true",
        help="Include LLM agent (requires OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="OpenAI model name for the LLM agent",
    )
    parser.add_argument(
        "--claude", action="store_true",
        help="Include Claude agent (requires ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--claude-model", type=str, default="claude-opus-4-6",
        dest="claude_model",
        help="Claude model ID for the Claude agent (default: claude-opus-4-6)",
    )
    parser.add_argument(
        "--arena", action="store_true",
        help="Run AdMarket Arena baselines after single-agent tasks",
    )
    parser.add_argument(
        "--task", type=str, default=None,
        help="Arena task: arena_easy, arena_medium, arena_hard (auto-enables --arena)",
    )
    args = parser.parse_args()

    task_names = ["creative_matcher", "placement_optimizer", "campaign_optimizer"]
    labels = ["Easy", "Medium", "Hard"]

    agents_factories: list = [
        lambda s: RandomAgent(random.Random(s)),
        lambda s: GreedyAgent(),
        lambda s: RuleBasedAgent(),
    ]
    agent_names = ["Random", "Greedy", "Rule-Based"]

    if args.llm:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY environment variable not set.")
            print("Set it with: export OPENAI_API_KEY=sk-...")
            return
        agents_factories.append(lambda s: LLMAgent(model=args.model))
        agent_names.append("LLM")
        if args.episodes > 10:
            print(f"NOTE: LLM agent is slow/costly. Consider --episodes 5 for LLM runs.\n")

    if args.claude:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
            print("Set it with: export ANTHROPIC_API_KEY=sk-ant-...")
            return
        agents_factories.append(lambda s: ClaudeAgent(model=args.claude_model))
        agent_names.append("Claude")
        if args.episodes > 5:
            print(f"NOTE: Claude agent is slow/costly. Consider --episodes 5 for Claude runs.\n")

    print(f"Running {args.episodes} episodes per task (seed={args.seed})\n")

    for task_name, label in zip(task_names, labels):
        print(f"Task: {task_name} ({label})")
        for factory, name in zip(agents_factories, agent_names):
            agent = factory(args.seed)
            scores = run_evaluation(task_name, agent, args.episodes, args.seed)
            mean = statistics.mean(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0.0
            print(f"  {name:15s} {mean:.4f} ± {std:.4f}")
        print()

    # ------------------------------------------------------------------
    # Arena baselines
    # ------------------------------------------------------------------
    # Arena mode: explicit --arena flag OR --task <arena_task_name>
    _arena_tasks = set(ARENA_TASKS.keys()) if _ARENA_AVAILABLE else set()
    run_arena = args.arena or (args.task is not None and args.task in _arena_tasks)

    if run_arena:
        if not _ARENA_AVAILABLE:
            print("ERROR: Arena modules not available — check imports.")
            return

        task = args.task or "arena_easy"
        n_ep = min(args.episodes, 5)  # cap at 5; arena_hard = 350 steps per episode

        arena_agents = [
            (lambda s: ArenaRandomAgent(random.Random(s)), "ArenaRandom"),
            (lambda s: ArenaGreedyAgent(), "ArenaGreedy"),
            (lambda s: ArenaPacingAgent(), "ArenaPacing"),
        ]

        print(f"=== AdMarket Arena — {task} ({n_ep} episodes, seed={args.seed}) ===")
        print(f"  {'Agent':<18} {'weekly_roas':>14} {'reward':>10}")
        print("  " + "-" * 44)

        results_by_agent: Dict[str, float] = {}
        for factory, name in arena_agents:
            agent = factory(args.seed)
            results = run_arena_evaluation(task, agent, n_ep, args.seed)
            roas_vals = results["weekly_roas"]
            rew_vals = results["rewards"]
            mean_roas = statistics.mean(roas_vals) if roas_vals else 0.0
            mean_rew = statistics.mean(rew_vals) if rew_vals else 0.0
            std_roas = statistics.stdev(roas_vals) if len(roas_vals) > 1 else 0.0
            print(f"  {name:<18} {mean_roas:>8.3f}±{std_roas:<5.3f}  {mean_rew:>8.2f}")
            results_by_agent[name] = mean_roas

        print()
        pacing = results_by_agent.get("ArenaPacing", 0.0)
        greedy = results_by_agent.get("ArenaGreedy", 0.0)
        rnd = results_by_agent.get("ArenaRandom", 0.0)
        # Greedy wins ~90% of slots → per-segment fatigue hits 1.0 by day 3 → CTR collapses.
        # Random wins ~25% → slower fatigue → higher CTR throughout.
        # Correct ordering: pacing (manages fatigue + budget) > random > greedy (over-serves).
        ordering_ok = pacing >= rnd >= greedy
        status = "PASS" if ordering_ok else "FAIL"
        print(f"  Ordering check (pacing >= random >= greedy): {status}")
        if not ordering_ok:
            print("  WARNING: Expected ROAS ordering not satisfied — review fatigue / reward coefficients.")


if __name__ == "__main__":
    main()
