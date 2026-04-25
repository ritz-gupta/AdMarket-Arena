"""
Scripted competitor personas for AdMarket Arena.

PersonaBot implements five named advertiser archetypes, each parameterised
by a 5-dimensional trait vector. Traits are jittered per episode so the
trained agent must generalise across the archetype rather than memorising
exact bid values.

LLMPolicyBot is a stub that exposes the same bid() interface but delegates
to a frozen Unsloth checkpoint — Plan 3 fills the real implementation so
self-play eval and Phase 3 opponent curriculum can swap personas for LLMs
without any env changes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from simulation import SEGMENT_NAMES, SEGMENT_CATEGORY_AFFINITY, USER_SEGMENTS
    from campaign_state import AdvertiserCampaignState
except ImportError:
    from .simulation import SEGMENT_NAMES, SEGMENT_CATEGORY_AFFINITY, USER_SEGMENTS
    from .campaign_state import AdvertiserCampaignState


# ---------------------------------------------------------------------------
# Trait clipping bounds — floor prevents zero-bid degeneration, ceiling
# prevents extreme single-trait domination
# ---------------------------------------------------------------------------
_CLIP_LOW = 0.01
_CLIP_HIGH = 0.995


@dataclass
class PersonaTraits:
    """5-dimensional trait vector that drives the bid formula."""

    aggression: float       # overall bid scale (0=never bids, 1=always max)
    pacing_strength: float  # how strictly daily budget is respected
    segment_focus: float    # penalty for off-target user segments
    fatigue_awareness: float  # bid reduction when per_segment_fatigue is high
    price_elasticity: float   # bid reduction when recent clearing prices are high


# ---------------------------------------------------------------------------
# Base trait vectors — chosen to create recognisable, distinct archetypes
# ---------------------------------------------------------------------------
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

# Each persona pursues a different campaign KPI — directly wired to the
# objective_type that AdvertiserCampaignState.objective_progress uses.
PERSONA_OBJECTIVES: Dict[str, str] = {
    "PremiumBrand": "awareness",
    "BargainReseller": "retention",
    "PerformanceMarketer": "conversion",
    "SpamFlooder": "conversion",
    "CautiousChallenger": "awareness",
}

# Canonical ordering: trained agent is always "trained_advertiser";
# persona slots 0-4 are filled from this list.
PERSONA_NAMES: List[str] = [
    "PremiumBrand",
    "BargainReseller",
    "PerformanceMarketer",
    "SpamFlooder",
    "CautiousChallenger",
]

# Segments each persona prioritises (for segment_factor computation).
# Derived from PERSONA_OBJECTIVES + domain knowledge about which segments
# over-index on awareness, conversion, and retention campaigns.
_PERSONA_TARGET_SEGMENTS: Dict[str, List[str]] = {
    "PremiumBrand": ["gen_z_creator", "fitness_enthusiast"],       # high-reach segments
    "BargainReseller": ["millennial_parent", "bargain_hunter"],     # repeat-purchase segments
    "PerformanceMarketer": ["fitness_enthusiast", "business_pro"],  # high-CTR segments
    "SpamFlooder": SEGMENT_NAMES,                                   # ignores segmentation
    "CautiousChallenger": ["casual_scroller", "gen_z_creator"],     # broad awareness
}


class PersonaBot:
    """Scripted competitor with a deterministic bid formula + per-episode jitter.

    All bid() calls are deterministic given the same persona seed, user, and
    campaign state — no stochastic sampling inside bid(). This makes reward
    variance from the perspective of the trained agent come only from
    engagement outcomes (clicks, view time), not from opponent unpredictability.

    The persona_seed changes each episode (passed from arena_env.reset(seed=...))
    so that trait jitter varies across episodes while remaining reproducible.
    """

    def __init__(self, name: str, advertiser_id: str, persona_seed: int = 0):
        if name not in _BASE_TRAITS:
            raise ValueError(f"Unknown persona: {name!r}. Choose from {list(_BASE_TRAITS)}")
        self.name = name
        self.advertiser_id = advertiser_id
        self.objective_type = PERSONA_OBJECTIVES[name]
        self.traits = self._apply_jitter(_BASE_TRAITS[name], _JITTER_RANGE[name], persona_seed)
        self._target_segments = _PERSONA_TARGET_SEGMENTS[name]

    # ------------------------------------------------------------------
    # Main interface — same signature as LLMPolicyBot.bid()
    # ------------------------------------------------------------------

    def bid(
        self,
        user_segment: str,
        user_id: str,
        step_in_day: int,
        state: AdvertiserCampaignState,
        recent_clearing_prices: List[float],
        creative_pool: List[dict],
    ) -> Tuple[float, bool, int]:
        """Return (bid_amount, skip, creative_id).

        bid_amount is in CPM units [0.0, 5.0].
        skip=True means this persona passes this impression slot.
        creative_id is the pool index of the creative to serve if won.
        """
        t = self.traits

        # Early exit: budget exhausted
        if state.budget_exhausted:
            return 0.0, True, 0

        # 1. Valuation: what is this user impression worth for our KPI?
        valuation = self._valuation(user_segment, user_id, state)

        # 2. Pacing: how aggressively should we spend right now?
        #    ideal_spent scales linearly across the day (step 0 → 25 → 50).
        ideal_spent = (step_in_day / 50.0) * state.daily_budget
        if ideal_spent > 0:
            overspend_ratio = state.spent_today / ideal_spent
        else:
            overspend_ratio = 1.0
        # pacing_strength=1 → strict: halve bid once 10% over budget
        # pacing_strength=0 → ignore pacing
        pacing_factor = max(
            0.05,
            1.0 - t.pacing_strength * max(0.0, overspend_ratio - 1.0),
        )

        # 3. Segment focus: does this user match our target segments?
        segment_factor = self._segment_factor(user_segment, t.segment_focus)

        # 4. Fatigue: has this segment been over-served by us?
        seg_fatigue = state.per_segment_fatigue.get(user_segment, 0.0)
        fatigue_factor = max(0.05, 1.0 - t.fatigue_awareness * seg_fatigue)

        # 5. Price sensitivity: is the market cheap right now?
        price_factor = self._price_factor(recent_clearing_prices, t.price_elasticity)

        # Combine — raw value is in abstract units, scale to CPM range [0, 5]
        raw = t.aggression * valuation * pacing_factor * segment_factor * fatigue_factor * price_factor
        bid_amount = round(min(5.0, max(0.0, raw * 3.0)), 4)

        # Skip if bid is trivially low or daily budget already blown
        skip = bid_amount < 0.10 or state.daily_budget_exhausted

        creative_id = self._pick_creative(user_segment, creative_pool)
        return bid_amount, skip, creative_id

    # ------------------------------------------------------------------
    # Factor helpers
    # ------------------------------------------------------------------

    def _valuation(
        self,
        user_segment: str,
        user_id: str,
        state: AdvertiserCampaignState,
    ) -> float:
        """How much is this specific impression worth for our objective?"""
        if self.objective_type == "awareness":
            # New user is far more valuable than already-reached user
            return 1.0 if user_id not in state.unique_users_reached else 0.15
        elif self.objective_type == "conversion":
            # Segments with high CTR modifier are worth more
            seg_data = USER_SEGMENTS.get(user_segment, {})
            ctr_mod = seg_data.get("base_ctr_modifier", 1.0)
            return min(1.5, ctr_mod)
        elif self.objective_type == "retention":
            # Users who already clicked are high-value repeat targets
            clicks = state.user_click_counts.get(user_id, 0)
            return 1.3 if clicks >= 1 else 0.6
        return 0.5

    def _segment_factor(self, user_segment: str, segment_focus: float) -> float:
        """High segment_focus → strong preference for target segments."""
        if user_segment in self._target_segments:
            # Bonus for matching: linearly scales with segment_focus
            return 0.6 + 0.4 * segment_focus
        else:
            # Penalty for mismatching: linearly scales with segment_focus
            return max(0.1, 1.0 - 0.7 * segment_focus)

    def _price_factor(
        self, recent_prices: List[float], price_elasticity: float
    ) -> float:
        """High price_elasticity → only enter cheap auctions.

        Uses the last 5 clearing prices to estimate market expensiveness.
        If no history yet, assume market is at a neutral price (factor=1.0).
        """
        if not recent_prices:
            return 1.0
        avg_price = sum(recent_prices[-5:]) / len(recent_prices[-5:])
        # Normalise: 2.0 CPM is considered "expensive" for this env
        expensiveness = min(1.0, avg_price / 2.0)
        return max(0.1, 1.0 - price_elasticity * expensiveness)

    def _pick_creative(self, user_segment: str, creative_pool: List[dict]) -> int:
        """Select the best matching creative pool index for this user.

        Prefers creatives that target the user's segment, falls back to
        the first creative if no match found.
        """
        if not creative_pool:
            return 0
        for i, c in enumerate(creative_pool):
            if c.get("target_segment") == user_segment:
                return i
        return 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_jitter(base: PersonaTraits, jitter: float, seed: int) -> PersonaTraits:
        """Add uniform noise to each trait and clip to [CLIP_LOW, CLIP_HIGH]."""
        rng = random.Random(seed)

        def j(val: float) -> float:
            noisy = val + rng.uniform(-jitter, jitter)
            return round(max(_CLIP_LOW, min(_CLIP_HIGH, noisy)), 5)

        return PersonaTraits(
            aggression=j(base.aggression),
            pacing_strength=j(base.pacing_strength),
            segment_focus=j(base.segment_focus),
            fatigue_awareness=j(base.fatigue_awareness),
            price_elasticity=j(base.price_elasticity),
        )


class LLMPolicyBot:
    """STUB — Plan 3 fills the real implementation.

    Wraps a frozen Unsloth checkpoint behind the same bid() interface as
    PersonaBot, so arena_env treats scripted and LLM opponents identically.

    Used for:
      - --eval-mode selfplay: replace CautiousChallenger slot with a frozen
        earlier checkpoint of the trained model ("v2 beats v1" slide).
      - Phase 3 opponent curriculum: self-play injection during training.

    Plan 3 must:
      1. Load the checkpoint with FastLanguageModel.from_pretrained().
      2. Call FastLanguageModel.for_inference(model) for speed.
      3. Format the observation dict into the same prompt the trained
         agent sees, run inference, parse the JSON output, and return
         (bid_amount, skip, creative_id).
    """

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self._model = None   # lazy-loaded by Plan 3's _load_model()
        self.advertiser_id = "llm_policy_bot"
        self.objective_type = "conversion"  # default; Plan 3 can override

    def bid(
        self,
        user_segment: str,
        user_id: str,
        step_in_day: int,
        state: AdvertiserCampaignState,
        recent_clearing_prices: List[float],
        creative_pool: List[dict],
    ) -> Tuple[float, bool, int]:
        raise NotImplementedError(
            "LLMPolicyBot.bid() is a stub. Plan 3 fills the real implementation "
            "in competitors.py by wrapping a frozen Unsloth checkpoint."
        )
