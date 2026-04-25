"""
Persona-shell competitors for AdMarket Arena.

This module is the union of Plan 1's foundational ``PersonaBot`` /
``LLMPolicyBot`` stubs with Plan 3's filled implementation:

  - ``PersonaBot`` — five named advertiser archetypes (PremiumBrand,
    BargainReseller, PerformanceMarketer, SpamFlooder, CautiousChallenger),
    each parameterised by a 5-dim trait vector. Per-episode jitter,
    clipped to [0.01, 0.995]. Plan 1's ``bid(user_segment, user_id, ...)``
    signature is the canonical one used by ``server/arena_env.py``;
    Plan 3 adds ``bid_from_observation(observation, state)`` as an
    adapter for the synthetic loop in ``scripts/advertiser_eval.py``.
  - ``LLMPolicyBot`` — Plan 3 fills the real impl. Wraps any frozen LLM
    behind the same ``bid(...)`` interface so the env doesn't know
    whether it's facing a scripted persona or a frozen checkpoint.
  - ``HELD_OUT_PERSONA`` — sixth persona (OpportunisticArbitrageur)
    used only by ``--eval-mode edge`` to test generalization.

Both call shapes are exposed via the same module so:

  - ``arena_env.py`` keeps using ``bot.bid(user_segment, user_id, ...)``
    (Plan 1 contract).
  - ``advertiser_eval.py`` / training notebooks call
    ``bot.bid_from_observation(observation, state)`` and get an
    ``AuctionAction`` back (Plan 3 contract).
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from .models import AuctionAction, AuctionObservation
    from .simulation import (
        SEGMENT_NAMES,
        SEGMENT_CATEGORY_AFFINITY,
        USER_SEGMENTS,
    )
    from .campaign_state import AdvertiserCampaignState
except ImportError:
    from models import AuctionAction, AuctionObservation  # type: ignore
    from simulation import (  # type: ignore
        SEGMENT_NAMES,
        SEGMENT_CATEGORY_AFFINITY,
        USER_SEGMENTS,
    )
    from campaign_state import AdvertiserCampaignState  # type: ignore


# ---------------------------------------------------------------------------
# Trait vector + named personas
# ---------------------------------------------------------------------------

TRAIT_NAMES: Tuple[str, ...] = (
    "aggression",
    "pacing_strength",
    "segment_focus",
    "fatigue_awareness",
    "price_elasticity",
)

TRAIT_FLOOR = 0.01
TRAIT_CEILING = 0.995

# Aliases kept for back-compat with Plan 1 internal naming.
_CLIP_LOW = TRAIT_FLOOR
_CLIP_HIGH = TRAIT_CEILING


@dataclass
class PersonaTraits:
    """5-dim trait vector that drives the bid formula."""

    aggression: float
    pacing_strength: float
    segment_focus: float
    fatigue_awareness: float
    price_elasticity: float


@dataclass(frozen=True)
class PersonaSpec:
    """Immutable definition of a named persona archetype.

    Plan 3-style alias: same content as ``PersonaTraits`` + a ``name``
    and a ``jitter`` range. Used by ``from_persona_name`` constructor
    and by the ``maxed_persona`` / ``jitter_persona`` helpers.
    """

    name: str
    aggression: float
    pacing_strength: float
    segment_focus: float
    fatigue_awareness: float
    price_elasticity: float
    jitter: float

    def base_vector(self) -> Dict[str, float]:
        return {
            "aggression": self.aggression,
            "pacing_strength": self.pacing_strength,
            "segment_focus": self.segment_focus,
            "fatigue_awareness": self.fatigue_awareness,
            "price_elasticity": self.price_elasticity,
        }


_BASE_TRAITS: Dict[str, PersonaTraits] = {
    "PremiumBrand": PersonaTraits(
        aggression=0.85, pacing_strength=0.40, segment_focus=0.80,
        fatigue_awareness=0.60, price_elasticity=0.30,
    ),
    "BargainReseller": PersonaTraits(
        aggression=0.40, pacing_strength=0.85, segment_focus=0.30,
        fatigue_awareness=0.50, price_elasticity=0.85,
    ),
    "PerformanceMarketer": PersonaTraits(
        aggression=0.65, pacing_strength=0.65, segment_focus=0.70,
        fatigue_awareness=0.85, price_elasticity=0.55,
    ),
    "SpamFlooder": PersonaTraits(
        aggression=0.95, pacing_strength=0.10, segment_focus=0.10,
        fatigue_awareness=0.05, price_elasticity=0.20,
    ),
    "CautiousChallenger": PersonaTraits(
        aggression=0.35, pacing_strength=0.70, segment_focus=0.65,
        fatigue_awareness=0.70, price_elasticity=0.75,
    ),
}

_JITTER_RANGE: Dict[str, float] = {
    "PremiumBrand": 0.15,
    "BargainReseller": 0.15,
    "PerformanceMarketer": 0.10,
    "SpamFlooder": 0.10,
    "CautiousChallenger": 0.20,
}

# Plan 3-style PersonaSpec dict (same content, exposed alongside).
PERSONAS: Dict[str, PersonaSpec] = {
    name: PersonaSpec(
        name=name,
        aggression=traits.aggression,
        pacing_strength=traits.pacing_strength,
        segment_focus=traits.segment_focus,
        fatigue_awareness=traits.fatigue_awareness,
        price_elasticity=traits.price_elasticity,
        jitter=_JITTER_RANGE[name],
    )
    for name, traits in _BASE_TRAITS.items()
}

PERSONA_OBJECTIVES: Dict[str, str] = {
    "PremiumBrand": "awareness",
    "BargainReseller": "retention",
    "PerformanceMarketer": "conversion",
    "SpamFlooder": "conversion",
    "CautiousChallenger": "awareness",
}

PERSONA_NAMES: List[str] = [
    "PremiumBrand",
    "BargainReseller",
    "PerformanceMarketer",
    "SpamFlooder",
    "CautiousChallenger",
]

_PERSONA_TARGET_SEGMENTS: Dict[str, List[str]] = {
    "PremiumBrand": ["gen_z_creator", "fitness_enthusiast"],
    "BargainReseller": ["millennial_parent", "bargain_hunter"],
    "PerformanceMarketer": ["fitness_enthusiast", "business_pro"],
    "SpamFlooder": list(SEGMENT_NAMES),
    "CautiousChallenger": ["casual_scroller", "gen_z_creator"],
}

# Held-out persona for --eval-mode edge "held-out persona" sub-condition
# (master Section 7.2). Never used during training so trained-agent
# performance against it is genuine generalization, not intra-distribution
# jitter robustness.
HELD_OUT_PERSONA = PersonaSpec(
    name="OpportunisticArbitrageur",
    aggression=0.55, pacing_strength=0.45, segment_focus=0.80,
    fatigue_awareness=0.60, price_elasticity=0.90, jitter=0.10,
)


def _clip_trait(value: float) -> float:
    return max(TRAIT_FLOOR, min(TRAIT_CEILING, value))


def jitter_persona(spec: PersonaSpec, rng: random.Random, jitter_scale: float = 1.0) -> Dict[str, float]:
    """Sample a per-episode trait dict by jittering the spec.

    ``jitter_scale=2.0`` is used by --eval-mode edge "extreme jitter"
    sub-condition; default 1.0 matches the spec's declared range.
    Always clipped to [TRAIT_FLOOR, TRAIT_CEILING] post-jitter.
    """
    width = spec.jitter * jitter_scale
    return {
        trait: _clip_trait(getattr(spec, trait) + rng.uniform(-width, width))
        for trait in TRAIT_NAMES
    }


def maxed_persona(spec: PersonaSpec) -> Dict[str, float]:
    """Pin the persona's dominant trait at ``TRAIT_CEILING`` and the
    natural opposite at ``TRAIT_FLOOR``. Used by --eval-mode edge
    "maxed personas" sub-condition.
    """
    base = spec.base_vector()
    dominant_trait = max(base, key=base.get)
    opposing = {
        "aggression": "fatigue_awareness",
        "pacing_strength": "aggression",
        "segment_focus": "aggression",
        "fatigue_awareness": "aggression",
        "price_elasticity": "aggression",
    }[dominant_trait]
    out = dict(base)
    out[dominant_trait] = TRAIT_CEILING
    out[opposing] = TRAIT_FLOOR
    return out


# ---------------------------------------------------------------------------
# PersonaBot
# ---------------------------------------------------------------------------

class PersonaBot:
    """Scripted competitor with deterministic bid formula + per-episode jitter.

    Two calling conventions, sharing the same internal trait state:

      - Plan 1 / arena_env: ``bid(user_segment, user_id, step_in_day,
        state, recent_clearing_prices, creative_pool) ->
        (bid_amount, skip, creative_id)``. ``state`` must be an
        ``AdvertiserCampaignState`` instance.

      - Plan 3 / advertiser_eval / tests: ``bid_from_observation(
        observation, state=None) -> AuctionAction``. ``observation`` is
        an ``AuctionObservation``. The ``bid()`` overload also accepts
        an ``AuctionObservation`` as a positional first argument and
        dispatches to ``bid_from_observation``.

    Per-episode jitter is reproducible from ``persona_seed``.
    """

    def __init__(
        self,
        spec: Optional[PersonaSpec] = None,
        traits: Optional[Dict[str, float]] = None,
        valuation_anchor: float = 1.0,
        # Plan 1 constructor kwargs (kept for arena_env compat)
        name: Optional[str] = None,
        advertiser_id: Optional[str] = None,
        persona_seed: int = 0,
    ):
        # Plan 1 path: ``PersonaBot(name="PremiumBrand", advertiser_id=..., persona_seed=...)``.
        if spec is None and name is not None:
            if name == HELD_OUT_PERSONA.name:
                spec = HELD_OUT_PERSONA
            elif name in PERSONAS:
                spec = PERSONAS[name]
            else:
                raise ValueError(
                    f"Unknown persona: {name!r}. Choose from {list(PERSONAS)}"
                    f" or {HELD_OUT_PERSONA.name!r}"
                )
            if traits is None:
                if persona_seed:
                    rng = random.Random(persona_seed)
                    traits = jitter_persona(spec, rng)
                else:
                    traits = spec.base_vector()

        if spec is None:
            raise TypeError("PersonaBot requires either a `spec` or a `name`.")

        self.spec = spec
        self.name = spec.name
        self.advertiser_id = advertiser_id or f"persona_{spec.name.lower()}"
        self.objective_type = PERSONA_OBJECTIVES.get(spec.name, "conversion")
        self.traits: Dict[str, float] = dict(traits if traits is not None else spec.base_vector())
        self.valuation_anchor = valuation_anchor
        self._target_segments = _PERSONA_TARGET_SEGMENTS.get(
            spec.name, list(SEGMENT_NAMES) if SEGMENT_NAMES else []
        )

    # --- Plan 3 factory ---

    @classmethod
    def from_persona_name(
        cls,
        persona_name: str,
        rng: Optional[random.Random] = None,
        jitter_enabled: bool = True,
        jitter_scale: float = 1.0,
        valuation_anchor: float = 1.0,
    ) -> "PersonaBot":
        """Plan 3 / synthetic-loop constructor with optional per-episode jitter."""
        if persona_name == HELD_OUT_PERSONA.name:
            spec = HELD_OUT_PERSONA
        else:
            spec = PERSONAS[persona_name]
        if jitter_enabled and rng is not None:
            traits = jitter_persona(spec, rng, jitter_scale=jitter_scale)
        else:
            traits = spec.base_vector()
        return cls(spec=spec, traits=traits, valuation_anchor=valuation_anchor)

    # --- Plan 1 bid path (unchanged contract) ---

    def _bid_plan1(
        self,
        user_segment: str,
        user_id: str,
        step_in_day: int,
        state: AdvertiserCampaignState,
        recent_clearing_prices: List[float],
        creative_pool: List[dict],
    ) -> Tuple[float, bool, int]:
        t_agg = self.traits["aggression"]
        t_pacing = self.traits["pacing_strength"]
        t_segment = self.traits["segment_focus"]
        t_fatigue = self.traits["fatigue_awareness"]
        t_price = self.traits["price_elasticity"]

        if getattr(state, "budget_exhausted", False):
            return 0.0, True, 0

        valuation = self._valuation(user_segment, user_id, state)

        ideal_spent = (step_in_day / 50.0) * state.daily_budget
        if ideal_spent > 0:
            overspend_ratio = state.spent_today / ideal_spent
        else:
            overspend_ratio = 1.0
        pacing_factor = max(0.05, 1.0 - t_pacing * max(0.0, overspend_ratio - 1.0))

        segment_factor = self._segment_factor(user_segment, t_segment)

        seg_fatigue = state.per_segment_fatigue.get(user_segment, 0.0)
        fatigue_factor = max(0.05, 1.0 - t_fatigue * seg_fatigue)

        price_factor = self._price_factor(recent_clearing_prices, t_price)

        raw = t_agg * valuation * pacing_factor * segment_factor * fatigue_factor * price_factor
        bid_amount = round(min(5.0, max(0.0, raw * 3.0)), 4)

        skip = bid_amount < 0.10 or getattr(state, "daily_budget_exhausted", False)
        creative_id = self._pick_creative(user_segment, creative_pool)
        return bid_amount, skip, creative_id

    def bid(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Dual-shape entry point.

        - Plan 1: ``bid(user_segment, user_id, step_in_day, state,
          recent_clearing_prices, creative_pool)`` -> ``(bid_amount, skip, creative_id)``.
        - Plan 3: ``bid(observation, state=None)`` where ``observation``
          is an ``AuctionObservation`` -> ``AuctionAction``.

        Dispatches based on the type of the first positional argument.
        """
        if args and isinstance(args[0], AuctionObservation):
            observation = args[0]
            state = args[1] if len(args) > 1 else kwargs.get("state")
            return self.bid_from_observation(observation, state)
        return self._bid_plan1(*args, **kwargs)

    # --- Plan 3 bid path (operates on AuctionObservation) ---

    def bid_from_observation(
        self,
        observation: AuctionObservation,
        state: Optional[Dict[str, Any]] = None,
    ) -> AuctionAction:
        """Synthetic-loop adapter: produce an ``AuctionAction`` from
        an ``AuctionObservation`` + optional per-step state dict.

        The state dict accepts the keys ``spent_today``, ``daily_target``,
        and ``target_segment`` to mirror what the env would normally
        track for this advertiser.
        """
        state = state or {}
        spent_today = float(state.get("spent_today", observation.spent_so_far_today))
        daily_target = float(
            state.get(
                "daily_target",
                max(observation.daily_budget_remaining + spent_today, 1e-6),
            )
        )
        target_segment = state.get("target_segment")
        if target_segment is None and self._target_segments:
            target_segment = self._target_segments[0]

        v = self._valuation_estimate(observation)
        a = self.traits["aggression"]
        pacing = self._pacing_factor(spent_today, daily_target)
        segment = self._segment_factor_obs(observation.user_segment, target_segment)
        fatigue = self._fatigue_factor_obs(
            observation.per_segment_fatigue, observation.user_segment,
        )
        price = self._price_factor_obs(observation.recent_clearing_prices)

        bid_amount = max(0.0, a * v * pacing * segment * fatigue * price)
        skip = bid_amount < observation.floor_price * 0.5

        creative_id = 0
        if observation.available_creatives:
            for idx, c in enumerate(observation.available_creatives):
                if c.get("target_segment") == observation.user_segment:
                    creative_id = idx
                    break

        return AuctionAction(
            skip=skip,
            bid_amount=round(min(5.0, bid_amount), 4),
            creative_id=creative_id,
        )

    # --- Plan 1 factor helpers ---

    def _valuation(
        self,
        user_segment: str,
        user_id: str,
        state: AdvertiserCampaignState,
    ) -> float:
        if self.objective_type == "awareness":
            return 1.0 if user_id not in state.unique_users_reached else 0.15
        elif self.objective_type == "conversion":
            seg_data = USER_SEGMENTS.get(user_segment, {})
            ctr_mod = seg_data.get("base_ctr_modifier", 1.0)
            return min(1.5, ctr_mod)
        elif self.objective_type == "retention":
            clicks = state.user_click_counts.get(user_id, 0)
            return 1.3 if clicks >= 1 else 0.6
        return 0.5

    def _segment_factor(self, user_segment: str, segment_focus: float) -> float:
        if user_segment in self._target_segments:
            return 0.6 + 0.4 * segment_focus
        return max(0.1, 1.0 - 0.7 * segment_focus)

    def _price_factor(
        self, recent_prices: List[float], price_elasticity: float,
    ) -> float:
        if not recent_prices:
            return 1.0
        avg_price = sum(recent_prices[-5:]) / len(recent_prices[-5:])
        expensiveness = min(1.0, avg_price / 2.0)
        return max(0.1, 1.0 - price_elasticity * expensiveness)

    def _pick_creative(self, user_segment: str, creative_pool: List[dict]) -> int:
        if not creative_pool:
            return 0
        for i, c in enumerate(creative_pool):
            if c.get("target_segment") == user_segment:
                return i
        return 0

    # --- Plan 3 factor helpers (work off AuctionObservation) ---

    def _pacing_factor(self, spent_today: float, daily_target: float) -> float:
        if daily_target <= 0.0:
            return 1.0
        spent_ratio = spent_today / daily_target
        s = self.traits["pacing_strength"]
        return float(max(0.0, 1.0 - s * max(0.0, spent_ratio - 0.5)))

    def _segment_factor_obs(self, user_segment: str, target_segment: Optional[str]) -> float:
        if not target_segment:
            return 1.0
        f = self.traits["segment_focus"]
        if user_segment == target_segment:
            return 1.0
        return float(1.0 - f * 0.85)

    def _fatigue_factor_obs(
        self, per_segment_fatigue: Dict[str, float], user_segment: str,
    ) -> float:
        a = self.traits["fatigue_awareness"]
        fatigue = float(per_segment_fatigue.get(user_segment, 0.0))
        return float(max(0.05, 1.0 - a * fatigue * 0.85))

    def _price_factor_obs(self, recent_clearing_prices: List[float]) -> float:
        e = self.traits["price_elasticity"]
        if not recent_clearing_prices:
            return 1.0
        recent = [p for p in recent_clearing_prices if p > 0]
        if not recent:
            return 1.0
        mean_price = sum(recent) / len(recent)
        ratio = mean_price / max(self.valuation_anchor, 1e-6)
        return float(max(0.10, 1.0 - e * (ratio - 1.0)))

    def _valuation_estimate(self, observation: AuctionObservation) -> float:
        recent = observation.recent_clearing_prices or []
        if recent:
            mean_recent = sum(recent) / len(recent)
            return min(self.valuation_anchor * 1.5, max(self.valuation_anchor * 0.5, mean_recent))
        return self.valuation_anchor


# ---------------------------------------------------------------------------
# LLMPolicyBot — Plan 3 implementation
# ---------------------------------------------------------------------------

ADVERTISER_SYSTEM_PROMPT = """\
You are an LLM advertiser bidding in a multi-day second-price auction.
Each step you receive a user, your remaining budget, segment fatigue,
recent clearing prices, and a target weekly ROAS.

Decide whether to bid; if you bid, choose the bid amount and creative.
Reply with ONLY a JSON object (no markdown, no commentary):
{"skip": bool, "bid_amount": float, "creative_id": int}

Rules:
- skip=true means do not bid (saves budget, no impression served).
- bid_amount in [0.0, 5.0]. Pay second-highest price if you win.
- creative_id is the 0-based index into available_creatives.
- Pace spend across days; an underspend penalty applies at week end.
"""


def _format_observation_for_advertiser(obs: AuctionObservation) -> str:
    creatives_lines: List[str] = []
    for idx, c in enumerate(obs.available_creatives):
        creatives_lines.append(
            f"  idx={idx}: target={c.get('target_segment','-')}, "
            f"category={c.get('category','-')}, "
            f"base_ctr={float(c.get('base_ctr', 0.0)):.3f}"
        )

    fatigue_lines = ", ".join(f"{k}={v:.2f}" for k, v in obs.per_segment_fatigue.items()) or "<none>"
    recent_prices = ", ".join(f"{p:.3f}" for p in obs.recent_clearing_prices[-5:]) or "<none>"

    return (
        f"DAY {obs.day_of_week} step_in_day={obs.step_in_day} "
        f"floor_price={obs.floor_price:.3f} freq_cap={obs.frequency_cap_per_user}\n"
        f"User: segment={obs.user_segment} interests={obs.user_interests} surface={obs.current_surface}\n"
        f"Budget: weekly_remaining={obs.weekly_budget_remaining:.2f} "
        f"daily_remaining={obs.daily_budget_remaining:.2f} "
        f"spent_today={obs.spent_so_far_today:.2f} spent_week={obs.spent_so_far_week:.2f}\n"
        f"Clicks: today={obs.clicks_today} week={obs.clicks_week} target_roas={obs.target_weekly_roas:.2f}\n"
        f"Per-segment fatigue: {fatigue_lines}\n"
        f"Recent clearing prices: {recent_prices}\n"
        f"Yesterday recap: {obs.yesterday_recap or '<n/a>'}\n"
        f"Available creatives:\n" + ("\n".join(creatives_lines) or "  <none>")
    )


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def parse_llm_advertiser_action(raw_text: str, n_creatives: int) -> AuctionAction:
    """Robust JSON -> AuctionAction. Falls back to ``skip=True`` on parse failure."""
    candidates: List[str] = [raw_text]
    match = _JSON_BLOCK_RE.search(raw_text)
    if match:
        candidates.append(match.group(0))

    parsed: Optional[Dict[str, Any]] = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue

    if not isinstance(parsed, dict):
        return AuctionAction(skip=True, bid_amount=0.0, creative_id=0)

    try:
        skip = bool(parsed.get("skip", False))
        bid_amount = float(parsed.get("bid_amount", 0.0))
        creative_id = int(parsed.get("creative_id", 0))
    except (TypeError, ValueError):
        return AuctionAction(skip=True, bid_amount=0.0, creative_id=0)

    bid_amount = max(0.0, min(5.0, bid_amount))
    creative_id = max(0, min(n_creatives - 1, creative_id)) if n_creatives > 0 else 0
    if bid_amount == 0.0 and not skip:
        skip = True
    return AuctionAction(skip=skip, bid_amount=round(bid_amount, 4), creative_id=creative_id)


@dataclass
class LLMPolicyBot:
    """Adapter around a frozen LLM checkpoint that emits ``AuctionAction``s.

    The ``completion_fn`` is ``(system_prompt, user_prompt) -> str``,
    decoupling the bot from any specific runtime:

      - ``make_unsloth_advertiser_completion_fn(checkpoint)`` for self-play
        eval and in-env rollouts.
      - An OpenAI-style client for the legacy hosted-LLM baseline.

    The bot is *frozen* by contract — neither weights nor history persist
    across steps, so the env can call ``bid`` from many parallel auction
    loops without coordination.
    """

    completion_fn: Callable[[str, str], str]
    name: str = "LLMPolicyBot"
    advertiser_id: str = "llm_policy_bot"
    objective_type: str = "conversion"
    system_prompt: str = ADVERTISER_SYSTEM_PROMPT
    fallback: Optional[PersonaBot] = None

    def __post_init__(self) -> None:
        # Provide a synthetic spec so this bot is interchangeable with
        # PersonaBot in code that reads ``bot.spec.name`` (e.g. eval result
        # win_rate dictionaries).
        self.spec = PersonaSpec(
            name=self.name, aggression=0.5, pacing_strength=0.5,
            segment_focus=0.5, fatigue_awareness=0.5,
            price_elasticity=0.5, jitter=0.0,
        )

    def bid(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Dual-shape entry point matching ``PersonaBot.bid``.

        - Plan 1: ``bid(user_segment, user_id, step_in_day, state,
          recent_clearing_prices, creative_pool) ->
          (bid_amount, skip, creative_id)``.
        - Plan 3: ``bid(observation, state=None) -> AuctionAction``.
        """
        if args and isinstance(args[0], AuctionObservation):
            observation = args[0]
            state = args[1] if len(args) > 1 else kwargs.get("state")
            return self.bid_from_observation(observation, state)
        # Plan 1 caller — synthesize a minimal observation, run the LLM,
        # and unpack to (bid, skip, creative_id).
        return self._bid_plan1(*args, **kwargs)

    def _bid_plan1(
        self,
        user_segment: str,
        user_id: str,
        step_in_day: int,
        state: AdvertiserCampaignState,
        recent_clearing_prices: List[float],
        creative_pool: List[dict],
    ) -> Tuple[float, bool, int]:
        observation = AuctionObservation(
            user_segment=user_segment,
            user_id=user_id,
            step_in_day=step_in_day,
            recent_clearing_prices=list(recent_clearing_prices or []),
            available_creatives=list(creative_pool or []),
            spent_so_far_today=getattr(state, "spent_today", 0.0),
            daily_budget_remaining=max(
                0.0,
                getattr(state, "daily_budget", 0.0) - getattr(state, "spent_today", 0.0),
            ),
            per_segment_fatigue=dict(getattr(state, "per_segment_fatigue", {}) or {}),
        )
        action = self.bid_from_observation(observation)
        return action.bid_amount, action.skip, action.creative_id

    def bid_from_observation(
        self,
        observation: AuctionObservation,
        state: Optional[Dict[str, Any]] = None,
    ) -> AuctionAction:
        prompt = _format_observation_for_advertiser(observation)
        try:
            raw = self.completion_fn(self.system_prompt, prompt)
        except Exception:
            return self._fallback_action(observation, state)
        if not isinstance(raw, str):
            return self._fallback_action(observation, state)
        return parse_llm_advertiser_action(
            raw, n_creatives=len(observation.available_creatives),
        )

    def _fallback_action(
        self,
        observation: AuctionObservation,
        state: Optional[Dict[str, Any]],
    ) -> AuctionAction:
        if self.fallback is not None:
            return self.fallback.bid_from_observation(observation, state)
        return AuctionAction(skip=True, bid_amount=0.0, creative_id=0)


# ---------------------------------------------------------------------------
# Unsloth inference helper (lazy import)
# ---------------------------------------------------------------------------

def make_unsloth_advertiser_completion_fn(
    checkpoint_path: str,
    max_new_tokens: int = 64,
    max_seq_length: int = 4096,
) -> Callable[[str, str], str]:
    """Return a ``(system, user) -> str`` callable backed by a frozen
    Unsloth LoRA checkpoint with ``FastLanguageModel.for_inference()``
    enabled. Lazy import keeps the module usable without unsloth installed.
    """
    try:
        from unsloth import FastLanguageModel  # type: ignore
    except ImportError as e:
        raise ImportError(
            "unsloth is required for make_unsloth_advertiser_completion_fn. "
            "Install with `pip install unsloth`."
        ) from e

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    def completion(system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)
        output = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        text = tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True)
        return text

    return completion


# ---------------------------------------------------------------------------
# Convenience: build an opponent slate (used by eval + tests)
# ---------------------------------------------------------------------------

def build_opponent_slate(
    persona_names: List[str],
    rng: random.Random,
    jitter_enabled: bool = True,
    jitter_scale: float = 1.0,
    valuation_anchor: float = 1.0,
) -> List[PersonaBot]:
    """Construct one ``PersonaBot`` per name with per-episode jitter."""
    return [
        PersonaBot.from_persona_name(
            persona_name=name,
            rng=rng,
            jitter_enabled=jitter_enabled,
            jitter_scale=jitter_scale,
            valuation_anchor=valuation_anchor,
        )
        for name in persona_names
    ]
