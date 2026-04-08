"""
Inference Script — Meta Ad Optimizer
===================================
MANDATORY ENV VARS:
    API_BASE_URL       The API endpoint for the LLM.
    MODEL_NAME         The model identifier to use for inference.
    HF_TOKEN           Your Hugging Face / API key.
    LOCAL_IMAGE_NAME   (optional) Docker image name when using from_docker_image().

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

import json
import os
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from meta_ad_optimizer.models import AdAction, AdObservation
from meta_ad_optimizer.server.ad_environment import AdOptimizerEnvironment
from meta_ad_optimizer.simulation import VALID_FORMATS, VALID_SURFACES
from meta_ad_optimizer.tasks import TASKS, grade_episode

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = "meta_ad_optimizer"

TASK_NAMES = ["creative_matcher", "placement_optimizer", "campaign_optimizer"]
EPISODES_PER_TASK = int(os.getenv("EPISODES_PER_TASK", "3"))
SEED = 42

SYSTEM_PROMPT = textwrap.dedent("""\
You are an ad optimization agent for Instagram and Facebook.
Given the current user profile, available creatives, and session state,
decide the best ad action to maximize engagement (CTR + view time)
while managing user fatigue.

Respond with ONLY a JSON object (no markdown, no explanation):
{"show_ad": bool, "creative_id": int, "platform": str, "placement": str, "ad_format": str}

Rules:
- show_ad: set false to skip and recover fatigue when fatigue > 0.4
- creative_id: index into the available_creatives list (0-based)
- platform: "instagram" or "facebook" — must match the user's current platform
- placement: a surface on that platform
  Instagram surfaces: feed, reels, stories, explore, search
  Facebook surfaces: feed, reels, stories, marketplace, search, right_column
- ad_format: must be valid for the chosen surface
  feed: image, video, carousel, reel
  reels: reel
  stories: image, video, reel, collection
  explore: image, video, carousel, reel
  search: image, video
  marketplace: image, carousel, collection
  right_column: image

Strategy tips:
- Match creative target_segment to the user's segment for 2x CTR boost
- Match creative category to user interests for 1.2x boost
- Place ads on the surface the user is currently browsing for a context bonus
- Use reel format on reels surface for +35% synergy
- Use collection on marketplace for +40% synergy
- Skip impressions when fatigue > 0.4 to let it recover
""")


# ---------------------------------------------------------------------------
# Structured logging helpers
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Observation → prompt
# ---------------------------------------------------------------------------

def obs_to_prompt(obs: AdObservation) -> str:
    creatives_lines = []
    for c in obs.available_creatives:
        creatives_lines.append(
            f"  idx={c['pool_index']}: category={c['category']}, "
            f"tone={c['tone']}, target={c['target_segment']}, "
            f"base_ctr={c['base_ctr']:.3f}, base_view={c['base_view_time']:.1f}s"
        )
    valid_surfaces = VALID_SURFACES.get(obs.current_platform, ["feed"])
    surface_fmt_info = []
    for s in valid_surfaces:
        fmts = VALID_FORMATS.get(s, ["image"])
        surface_fmt_info.append(f"  {s}: {', '.join(fmts)}")

    return (
        f"Task: {obs.task}\n"
        f"User: segment={obs.user_segment}, interests={obs.user_interests}, "
        f"device={obs.user_device}\n"
        f"Platform: {obs.current_platform} (YOU MUST use this platform), "
        f"current surface: {obs.current_surface}\n"
        f"Valid surfaces for {obs.current_platform} and their formats:\n"
        + "\n".join(surface_fmt_info) + "\n"
        f"Step: {obs.step}/{obs.total_steps}, fatigue: {obs.fatigue_level:.3f}, "
        f"impressions: {obs.impression_count}\n"
        f"Session: CTR={obs.session_metrics.get('ctr', 0):.4f}, "
        f"view_time={obs.session_metrics.get('total_view_time', 0):.1f}s, "
        f"satisfaction={obs.session_metrics.get('cumulative_satisfaction', 0):.3f}\n"
        f"Available creatives:\n" + "\n".join(creatives_lines)
    )


# ---------------------------------------------------------------------------
# LLM action selection
# ---------------------------------------------------------------------------

def get_llm_action(
    client: OpenAI,
    obs: AdObservation,
    history: List[str],
) -> AdAction:
    """Ask the LLM for an action, with fallback on parse errors."""
    prompt = obs_to_prompt(obs)
    if history:
        prompt += "\n\nRecent history:\n" + "\n".join(history[-3:])

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=120,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        data = json.loads(text)
        raw_action = AdAction(
            show_ad=bool(data.get("show_ad", True)),
            creative_id=int(data.get("creative_id", 0)),
            platform=str(data.get("platform", obs.current_platform)),
            placement=str(data.get("placement", obs.current_surface)),
            ad_format=str(data.get("ad_format", "image")),
        )
        return _validate_action(raw_action, obs)
    except Exception:
        return _fallback_action(obs)


def _fallback_action(obs: AdObservation) -> AdAction:
    """Simple heuristic fallback when LLM fails."""
    if obs.fatigue_level > 0.45:
        return AdAction(
            show_ad=False, creative_id=0,
            platform=obs.current_platform,
            placement=obs.current_surface,
            ad_format="image",
        )
    best_idx, best_score = 0, -1.0
    for i, c in enumerate(obs.available_creatives):
        score = c.get("base_ctr", 0)
        if c.get("target_segment") == obs.user_segment:
            score *= 2.0
        if score > best_score:
            best_score = score
            best_idx = i

    surface = obs.current_surface
    valid = VALID_SURFACES.get(obs.current_platform, ["feed"])
    if surface not in valid:
        surface = "feed"
    fmts = VALID_FORMATS.get(surface, ["image"])
    fmt = "reel" if "reel" in fmts and surface in ("reels", "stories") else fmts[0]

    return AdAction(
        show_ad=True, creative_id=best_idx,
        platform=obs.current_platform,
        placement=surface, ad_format=fmt,
    )


def _validate_action(action: AdAction, obs: AdObservation) -> AdAction:
    """Correct invalid platform/surface/format combos from LLM output."""
    platform = action.platform
    placement = action.placement
    ad_format = action.ad_format

    valid_surfaces = VALID_SURFACES.get(platform, [])
    if not valid_surfaces:
        platform = obs.current_platform
        valid_surfaces = VALID_SURFACES.get(platform, ["feed"])

    if placement not in valid_surfaces:
        placement = obs.current_surface if obs.current_surface in valid_surfaces else valid_surfaces[0]

    valid_fmts = VALID_FORMATS.get(placement, ["image"])
    if ad_format not in valid_fmts:
        if "reel" in valid_fmts and placement in ("reels", "stories"):
            ad_format = "reel"
        elif "collection" in valid_fmts and placement == "marketplace":
            ad_format = "collection"
        else:
            ad_format = valid_fmts[0]

    creative_id = max(0, min(action.creative_id, len(obs.available_creatives) - 1))

    return AdAction(
        show_ad=action.show_ad,
        creative_id=creative_id,
        platform=platform,
        placement=placement,
        ad_format=ad_format,
    )


def action_to_str(action: AdAction) -> str:
    """Compact string representation of an action for logging."""
    if not action.show_ad:
        return "skip()"
    return (
        f"ad(creative={action.creative_id},"
        f"{action.platform}/{action.placement}/{action.ad_format})"
    )


# ---------------------------------------------------------------------------
# Run one episode
# ---------------------------------------------------------------------------

def run_episode(
    env: AdOptimizerEnvironment,
    client: OpenAI,
    task_name: str,
    seed: int,
) -> float:
    """Run a single episode, emitting [START]/[STEP]/[END] logs. Returns score."""
    cfg = TASKS[task_name]
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(seed=seed, task=task_name)

        for step in range(1, cfg.steps_per_episode + 1):
            if obs.done:
                break

            action = get_llm_action(client, obs, history)
            obs = env.step(action)

            reward = obs.reward if obs.reward is not None else 0.0
            done = obs.done
            error = None
            metrics = obs.last_action_metrics
            if isinstance(metrics, dict) and not metrics.get("valid_action", True):
                error = "invalid_format_surface_combo"

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_to_str(action),
                reward=reward,
                done=done,
                error=error,
            )

            history.append(
                f"Step {step}: {action_to_str(action)} -> r={reward:.2f}"
            )

            if done:
                break

        score = grade_episode(env.state)
        success = score >= 0.1

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, timeout=15.0)
    env = AdOptimizerEnvironment()

    all_scores: Dict[str, List[float]] = {}

    for task_name in TASK_NAMES:
        task_scores = []
        for ep in range(EPISODES_PER_TASK):
            score = run_episode(env, client, task_name, seed=SEED + ep)
            task_scores.append(score)
        all_scores[task_name] = task_scores

    print("\n=== Summary ===", flush=True)
    for task_name in TASK_NAMES:
        scores = all_scores[task_name]
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"  {task_name}: avg_score={avg:.4f} ({len(scores)} episodes)", flush=True)


if __name__ == "__main__":
    main()
