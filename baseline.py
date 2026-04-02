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
import statistics
from typing import Any, Dict, List, Optional

from .models import AdAction, AdObservation
from .server.ad_environment import AdOptimizerEnvironment
from .simulation import (
    SEGMENT_CATEGORY_AFFINITY,
    VALID_FORMATS,
    VALID_SURFACES,
    FORMAT_SURFACE_SYNERGY,
)
from .tasks import TASKS, grade_episode


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


if __name__ == "__main__":
    main()
