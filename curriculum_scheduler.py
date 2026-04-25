"""
Curriculum scheduler for advertiser GRPO training.

Owned by Plan 3, used by ``train_grpo.ipynb``. Promotes the advertiser
through three task tiers in sequence:

    arena_easy  ->  arena_medium  ->  arena_hard

Promotion rule (master Section 3.1.1 / 6): mean episode reward greater
than ``promotion_threshold`` (default 0.30) for ``required_streak``
consecutive rollouts (default 10). Demotion is *not* supported — a
stuck training run is killed by the operator, not auto-demoted, so
slip protocols stay deterministic.

Two ways to integrate (the notebook uses both):

  1. **Pull**: ``scheduler.current_tier`` query before every
     ``env.reset(task=...)`` call.
  2. **Push (TrainerCallback)**: ``scheduler.as_callback(...)`` returns
     a TRL TrainerCallback that registers an ``on_log`` hook to feed
     the latest mean reward in. Notebook only has to ``add_callback``
     once.

Both paths share the same internal state, so mixing is fine.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional


DEFAULT_TIER_ORDER: List[str] = ["arena_easy", "arena_medium", "arena_hard"]


@dataclass
class CurriculumScheduler:
    """Stateful scheduler that promotes through ``tier_order`` based on
    rollout-mean-reward.

    Args:
        tier_order: ordered list of task names; default
            ``["arena_easy", "arena_medium", "arena_hard"]``.
        promotion_threshold: mean reward above which a rollout
            "qualifies" for promotion; default 0.30 (master Section
            3.1.1).
        required_streak: how many consecutive qualifying rollouts
            trigger a tier bump; default 10.
        reward_metric: which key in the metrics dict contains the
            rollout-mean-reward; default ``episode_return_total``.
        on_promote: optional callback invoked as ``on_promote(old_tier,
            new_tier, step)`` so the notebook can ``env.reset(task=...)``
            and re-init opponent slates immediately.

    Stretch (Section 6, Framing D): the same scheduler can be reused for
    the *opponent* curriculum (Phase 1 fixed -> Phase 2 jittered ->
    Phase 3 selfplay) by passing a different ``tier_order`` and
    ``on_promote`` that swaps opponent slates instead of env tier.
    """

    tier_order: List[str] = field(default_factory=lambda: list(DEFAULT_TIER_ORDER))
    promotion_threshold: float = 0.30
    required_streak: int = 10
    reward_metric: str = "episode_return_total"
    on_promote: Optional[Callable[[str, str, int], None]] = None

    _tier_index: int = 0
    _streak: int = 0
    _history: Deque[float] = field(default_factory=lambda: deque(maxlen=64))
    _promotion_log: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def current_tier(self) -> str:
        return self.tier_order[self._tier_index]

    @property
    def current_streak(self) -> int:
        return self._streak

    @property
    def is_at_top(self) -> bool:
        return self._tier_index >= len(self.tier_order) - 1

    @property
    def promotion_log(self) -> List[Dict[str, Any]]:
        return list(self._promotion_log)

    # --- public update API ---

    def step(self, mean_reward: float, training_step: Optional[int] = None) -> Dict[str, Any]:
        """Feed one rollout's mean reward; possibly promote.

        Returns a small dict with the post-step state — useful for
        logging in the training notebook:

            {
              "current_tier": str,
              "promoted": bool,
              "previous_tier": Optional[str],
              "streak": int,
              "promotion_threshold": float,
            }
        """
        self._history.append(float(mean_reward))
        promoted = False
        previous_tier: Optional[str] = None

        if mean_reward >= self.promotion_threshold:
            self._streak += 1
        else:
            self._streak = 0

        if self._streak >= self.required_streak and not self.is_at_top:
            previous_tier = self.current_tier
            self._tier_index += 1
            self._streak = 0
            promoted = True
            self._promotion_log.append({
                "from": previous_tier,
                "to": self.current_tier,
                "training_step": training_step,
                "trigger_mean_reward": mean_reward,
            })
            if self.on_promote is not None:
                try:
                    self.on_promote(previous_tier, self.current_tier, int(training_step or 0))
                except Exception:
                    pass

        return {
            "current_tier": self.current_tier,
            "promoted": promoted,
            "previous_tier": previous_tier,
            "streak": self._streak,
            "promotion_threshold": self.promotion_threshold,
        }

    def update_from_metrics(self, metrics: Dict[str, Any], training_step: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Same as ``step`` but pulls the reward from a metrics dict by
        key. Returns None if the metric is missing (so the callback
        skips silently when training is in a non-rollout step)."""
        if self.reward_metric not in metrics:
            return None
        try:
            value = float(metrics[self.reward_metric])
        except (TypeError, ValueError):
            return None
        return self.step(value, training_step=training_step)

    # --- TrainerCallback bridge ---

    def as_callback(self) -> Any:
        """Return a TRL ``TrainerCallback`` that pipes the rollout
        reward into ``self.step`` on every ``on_log`` event.

        We import ``TrainerCallback`` lazily so this module remains
        importable in pytest without the huggingface stack."""
        try:  # pragma: no cover - exercised in training notebook only
            from transformers.trainer_callback import TrainerCallback
        except ImportError:  # pragma: no cover
            class TrainerCallback:  # type: ignore[no-redef]
                pass

        scheduler = self

        class _CurriculumTrainerCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
                if logs is None:
                    return control
                step = int(getattr(state, "global_step", 0)) if state else 0
                scheduler.update_from_metrics(logs, training_step=step)
                return control

        return _CurriculumTrainerCallback()


# ---------------------------------------------------------------------------
# Convenience factory used by the notebook
# ---------------------------------------------------------------------------

def make_advertiser_curriculum(
    on_promote: Optional[Callable[[str, str, int], None]] = None,
    promotion_threshold: float = 0.30,
    required_streak: int = 10,
    reward_metric: str = "episode_return_total",
) -> CurriculumScheduler:
    """Default advertiser curriculum: easy -> medium -> hard."""
    return CurriculumScheduler(
        tier_order=list(DEFAULT_TIER_ORDER),
        promotion_threshold=promotion_threshold,
        required_streak=required_streak,
        reward_metric=reward_metric,
        on_promote=on_promote,
    )
