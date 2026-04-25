"""
AdMarket Arena — OpenEnv Environment implementation.

Wires together all Group A components into a running Environment:
  auction.py        — Vickrey second-price auction + freq-cap filtering
  campaign_state.py — per-advertiser budget / KPI / fatigue tracking
  competitors.py    — scripted PersonaBot opponents (bid() interface)
  summarizer.py     — Theme 2 day recap for long-horizon planning
  arena_rubrics.py  — composable per-step / daily / weekly reward rubrics

Episode structure:
  n_days × impressions_per_day auction steps.
  1 trained advertiser ("trained_advertiser") vs. n_personas PersonaBots.
  Reward flows ONLY to the trained advertiser — PersonaBots are env actors.

Usage:
  env = AdMarketArenaEnvironment()
  obs = env.reset(task="arena_easy", seed=42)
  obs = env.step(AuctionAction(skip=False, bid_amount=1.2, creative_id=0))
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        AuctionAction,
        AuctionObservation,
        AuctionResult as AuctionResultModel,
        ArenaState,
    )
    from ..campaign_state import AdvertiserCampaignState
    from ..auction import run_auction
    from ..competitors import PersonaBot, PERSONA_NAMES, PERSONA_OBJECTIVES
    from ..summarizer import summarize_day, empty_recap
    from ..simulation import (
        compute_engagement,
        generate_master_catalog,
        generate_user,
        sample_creatives,
        UserProfile,
    )
    from ..tasks import ARENA_TASKS, DEFAULT_ARENA_TASK, ArenaTaskConfig
    from .arena_rubrics import PerStepEngagementRubric, DailyPacingRubric, WeeklyROASRubric
except ImportError:
    from models import (
        AuctionAction,
        AuctionObservation,
        AuctionResult as AuctionResultModel,
        ArenaState,
    )
    from campaign_state import AdvertiserCampaignState
    from auction import run_auction
    from competitors import PersonaBot, PERSONA_NAMES, PERSONA_OBJECTIVES
    from summarizer import summarize_day, empty_recap
    from simulation import (
        compute_engagement,
        generate_master_catalog,
        generate_user,
        sample_creatives,
        UserProfile,
    )
    from tasks import ARENA_TASKS, DEFAULT_ARENA_TASK, ArenaTaskConfig
    from server.arena_rubrics import PerStepEngagementRubric, DailyPacingRubric, WeeklyROASRubric


_TRAINED_ID = "trained_advertiser"
_USER_POOL_SIZE = 100       # synthetic user pool per episode (repeat exposures → freq caps matter)
_CREATIVES_PER_EPISODE = 12
_RECENT_PRICE_WINDOW = 5    # how many clearing prices to surface in observation
_PERSONA_CLICK_RATE = 0.15  # simplified CTR used for scripted opponent state updates


class AdMarketArenaEnvironment(Environment):
    """Multi-agent long-horizon ad auction environment.

    Supports three difficulty tiers via reset(task=...):
      arena_easy   (3 days × 20 slots = 60 steps,  3 PersonaBots)
      arena_medium (5 days × 30 slots = 150 steps, 4 PersonaBots)
      arena_hard   (7 days × 50 slots = 350 steps, 5 PersonaBots)

    Reward decomposition (trained advertiser only):
      Per step:  PerStepEngagementRubric  (dense, ±0.5 range)
      Per day:   DailyPacingRubric        (max +0.50 per day)
      Per week:  WeeklyROASRubric         (max +7.50, dominant signal)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, catalog_size: int = 80, catalog_seed: int = 0):
        super().__init__()
        self._catalog = generate_master_catalog(n_creatives=catalog_size, seed=catalog_seed)

        # Rubric instances are stateless — safe to reuse across episodes
        self._per_step_rubric = PerStepEngagementRubric()
        self._daily_rubric = DailyPacingRubric()
        self._weekly_rubric = WeeklyROASRubric()

        # Episode-scoped state (reset on every reset() call)
        self._task_cfg: ArenaTaskConfig = ARENA_TASKS[DEFAULT_ARENA_TASK]
        self._trained_state: Optional[AdvertiserCampaignState] = None
        self._persona_bots: List[PersonaBot] = []
        self._persona_states: Dict[str, AdvertiserCampaignState] = {}
        self._creative_pool: List[dict] = []
        self._user_pool: List[Tuple[str, UserProfile]] = []

        self._rng: random.Random = random.Random()

        # Per-step tracking
        self._global_step: int = 0
        self._current_user_id: str = ""
        self._current_user_profile: Optional[UserProfile] = None
        self._recent_clearing_prices: List[float] = []
        self._day_auction_log: List[Dict] = []
        self._yesterday_recap: str = ""
        self._last_auction_result: Optional[AuctionResultModel] = None

        self._arena_state: ArenaState = ArenaState(episode_id=str(uuid4()), step_count=0)

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        task: Optional[str] = None,
        opponent_phase: int = 1,   # reserved for Plan 3 curriculum phases
        **kwargs: Any,
    ) -> AuctionObservation:
        """Initialise a new episode.

        Args:
            seed: RNG seed for full reproducibility.
            task: One of 'arena_easy', 'arena_medium', 'arena_hard'.
            opponent_phase: curriculum phase (1=scripted only; Plan 3 uses 2/3 for self-play).
        """
        task_name = task or DEFAULT_ARENA_TASK
        self._task_cfg = ARENA_TASKS.get(task_name, ARENA_TASKS[DEFAULT_ARENA_TASK])
        cfg = self._task_cfg

        self._rng = random.Random(seed)
        persona_seed = seed if seed is not None else self._rng.randint(0, 2 ** 31)

        # ---- trained advertiser ----
        self._trained_state = AdvertiserCampaignState(
            advertiser_id=_TRAINED_ID,
            objective_type="conversion",
            weekly_budget=cfg.initial_budget,
            daily_budget=cfg.daily_budget_cap,
        )

        # ---- scripted opponents ----
        n = min(cfg.n_personas, len(PERSONA_NAMES))
        self._persona_bots = [
            PersonaBot(
                name=PERSONA_NAMES[i],
                advertiser_id=f"persona_{i}",
                persona_seed=persona_seed + i,
            )
            for i in range(n)
        ]
        self._persona_states = {
            bot.advertiser_id: AdvertiserCampaignState(
                advertiser_id=bot.advertiser_id,
                objective_type=PERSONA_OBJECTIVES[bot.name],
                weekly_budget=cfg.initial_budget,
                daily_budget=cfg.daily_budget_cap,
            )
            for bot in self._persona_bots
        }

        # ---- shared creative pool ----
        self._creative_pool = sample_creatives(
            self._catalog, _CREATIVES_PER_EPISODE, self._rng
        )

        # ---- stable user pool for freq-cap realism ----
        # 100 users, each with a fixed profile; users repeat across slots
        self._user_pool = [
            (f"user_{i:04d}", generate_user(self._rng))
            for i in range(_USER_POOL_SIZE)
        ]

        # ---- reset counters ----
        self._global_step = 0
        self._recent_clearing_prices = []
        self._day_auction_log = []
        self._yesterday_recap = empty_recap(1)
        self._last_auction_result = None

        # Pre-sample the first user so reset() observation is fully populated
        self._current_user_id, self._current_user_profile = self._sample_user()

        self._arena_state = ArenaState(
            episode_id=str(uuid4()),
            step_count=0,
            task=cfg.name,
            day_number=1,
            weekly_budget=cfg.initial_budget,
            persona_names=[bot.name for bot in self._persona_bots],
        )

        return self._build_observation(done=False, reward=0.0)

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: AuctionAction) -> AuctionObservation:  # type: ignore[override]
        if self._trained_state is None:
            raise RuntimeError("Call reset() before step().")

        cfg = self._task_cfg
        trained = self._trained_state

        step_in_day = self._global_step % cfg.impressions_per_day
        day_number = trained.day_number
        floor_price = cfg.floor_price_base + cfg.floor_price_daily_increment * (day_number - 1)

        user_id = self._current_user_id
        user_profile = self._current_user_profile

        # ----------------------------------------------------------------
        # 1. Collect bids
        # ----------------------------------------------------------------
        # over_budget: agent bid when they had no remaining weekly budget
        over_budget = (not action.skip) and trained.budget_exhausted
        trained_bid = (0.0, True) if over_budget else (action.bid_amount, action.skip)

        bids: Dict[str, Tuple[float, bool]] = {_TRAINED_ID: trained_bid}

        for bot in self._persona_bots:
            p_state = self._persona_states[bot.advertiser_id]
            bid_amount, skip, _ = bot.bid(
                user_segment=user_profile.segment,
                user_id=user_id,
                step_in_day=step_in_day,
                state=p_state,
                recent_clearing_prices=self._recent_clearing_prices,
                creative_pool=self._creative_pool,
            )
            bids[bot.advertiser_id] = (bid_amount, skip)

        # Frequency caps: {adv_id: {user_id: impressions_today}}
        freq_caps: Dict[str, Dict[str, int]] = {
            _TRAINED_ID: trained.per_user_impressions_today,
            **{
                bot.advertiser_id: self._persona_states[bot.advertiser_id].per_user_impressions_today
                for bot in self._persona_bots
            },
        }

        # ----------------------------------------------------------------
        # 2. Run auction
        # ----------------------------------------------------------------
        auction_result = run_auction(
            bids=bids,
            floor_price=floor_price,
            freq_caps=freq_caps,
            freq_cap_limit=cfg.frequency_cap_per_user,
            user_id=user_id,
        )

        # ----------------------------------------------------------------
        # 3. Process winner
        # ----------------------------------------------------------------
        clicked = False
        clearing_price = auction_result.clearing_price

        if auction_result.winner_id == _TRAINED_ID:
            creative_id = max(0, min(action.creative_id, len(self._creative_pool) - 1))
            creative = self._creative_pool[creative_id]
            trained.record_win(clearing_price, user_id)

            engagement = compute_engagement(
                user=user_profile,
                creative=creative,
                platform=user_profile.platform,
                placement=user_profile.starting_surface,
                ad_format="image",
                fatigue_level=trained.per_segment_fatigue.get(user_profile.segment, 0.0),
                rng=self._rng,
            )
            clicked = engagement["clicked"]
            trained.record_engagement(
                clicked=clicked,
                user_id=user_id,
                segment=user_profile.segment,
                fatigue_increment=cfg.fatigue_increment,
            )

        elif auction_result.winner_id is not None:
            # A PersonaBot won — update their state for freq-cap / pacing accuracy
            p_state = self._persona_states[auction_result.winner_id]
            p_state.record_win(clearing_price, user_id)
            p_state.record_engagement(
                clicked=self._rng.random() < _PERSONA_CLICK_RATE,
                user_id=user_id,
                segment=user_profile.segment,
                fatigue_increment=cfg.fatigue_increment,
            )

        # Trained agent recovers fatigue on segments they didn't serve
        if auction_result.winner_id != _TRAINED_ID:
            trained.recover_fatigue(user_profile.segment, cfg.fatigue_recovery)

        # ----------------------------------------------------------------
        # 4. Per-step reward
        # ----------------------------------------------------------------
        step_reward = self._per_step_rubric.score(
            won_auction=(auction_result.winner_id == _TRAINED_ID),
            clicked=clicked,
            clearing_price=clearing_price,
            skipped=action.skip,
            over_budget=over_budget,
        )

        # ----------------------------------------------------------------
        # 5. Update market history + log
        # ----------------------------------------------------------------
        if clearing_price > 0:
            self._recent_clearing_prices.append(clearing_price)
            if len(self._recent_clearing_prices) > _RECENT_PRICE_WINDOW:
                self._recent_clearing_prices.pop(0)

        self._day_auction_log.append({
            "step": self._global_step,
            "step_in_day": step_in_day,
            "user_id": user_id,
            "user_segment": user_profile.segment,
            "winner_id": auction_result.winner_id,
            "clearing_price": clearing_price,
            "no_contest": auction_result.no_contest,
            "clicked": clicked if auction_result.winner_id == _TRAINED_ID else None,
        })

        self._last_auction_result = AuctionResultModel(
            winner_id=auction_result.winner_id,
            clearing_price=auction_result.clearing_price,
            no_contest=auction_result.no_contest,
            all_bids=auction_result.all_bids,
        )

        # ----------------------------------------------------------------
        # 6. Advance global step, check boundaries
        # ----------------------------------------------------------------
        self._global_step += 1
        is_last_step_of_day = (step_in_day == cfg.impressions_per_day - 1)
        is_done = (self._global_step >= cfg.total_steps)

        daily_reward = 0.0
        weekly_reward = 0.0

        if is_last_step_of_day and not is_done:
            # Day boundary: fire daily rubric BEFORE reset_day() clears accumulators
            daily_reward = self._daily_rubric.score(trained)

            self._yesterday_recap = summarize_day(
                state=trained,
                auction_log=self._day_auction_log,
                total_steps=cfg.total_steps,
            )

            trained.reset_day()
            for p_state in self._persona_states.values():
                p_state.reset_day()

            self._day_auction_log = []
            self._arena_state.day_number = trained.day_number

        if is_done:
            weekly_reward = self._weekly_rubric.score(trained)

        total_reward = round(step_reward + daily_reward + weekly_reward, 5)

        # ----------------------------------------------------------------
        # 7. Sync arena state + pre-sample next user
        # ----------------------------------------------------------------
        self._arena_state.step_count = self._global_step
        self._arena_state.step_in_day = self._global_step % cfg.impressions_per_day
        self._arena_state.spent_total = trained.spent_total
        self._arena_state.clicks_total = trained.clicks_total
        self._arena_state.wins_total = trained.wins_total
        self._arena_state.weekly_roas = round(trained.weekly_roas, 4)
        self._arena_state.auction_log_length = len(self._day_auction_log)

        if not is_done:
            self._current_user_id, self._current_user_profile = self._sample_user()

        return self._build_observation(done=is_done, reward=total_reward)

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------

    @property
    def state(self) -> ArenaState:
        return self._arena_state

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _sample_user(self) -> Tuple[str, UserProfile]:
        """Pick a random user from the episode's stable pool."""
        idx = self._rng.randrange(_USER_POOL_SIZE)
        return self._user_pool[idx]

    def _build_observation(self, done: bool, reward: float) -> AuctionObservation:
        """Construct the AuctionObservation for the UPCOMING auction slot."""
        cfg = self._task_cfg
        trained = self._trained_state

        if trained is None:
            return AuctionObservation(done=done, reward=reward)

        day_number = trained.day_number
        step_in_day = self._global_step % cfg.impressions_per_day
        floor_price = cfg.floor_price_base + cfg.floor_price_daily_increment * (day_number - 1)

        # win_rate_today: wins so far today / slots processed today
        wins_today = trained.wins_today
        win_rate_today = wins_today / max(1, step_in_day) if step_in_day > 0 else 0.0

        pool_dicts = [
            {
                "pool_index": c["pool_index"],
                "category": c["category"],
                "tone": c["tone"],
                "target_segment": c["target_segment"],
                "base_ctr": c["base_ctr"],
            }
            for c in self._creative_pool
        ]

        return AuctionObservation(
            task=cfg.name,
            advertiser_id=_TRAINED_ID,
            campaign_objective=trained.objective_type,
            day_number=day_number,
            step_in_day=step_in_day,
            step=self._global_step,
            total_steps=cfg.total_steps,
            # Show the user for the UPCOMING slot (pre-sampled at end of last step)
            user_segment=self._current_user_profile.segment if self._current_user_profile else "",
            user_id=self._current_user_id,
            available_creatives=pool_dicts,
            weekly_budget=cfg.initial_budget,
            budget_remaining=round(max(0.0, trained.weekly_budget - trained.spent_total), 4),
            daily_budget_remaining=round(max(0.0, trained.daily_budget - trained.spent_today), 4),
            recent_clearing_prices=list(self._recent_clearing_prices),
            floor_price=round(floor_price, 2),
            last_auction_result=self._last_auction_result,
            wins_today=wins_today,
            win_rate_today=round(win_rate_today, 4),
            daily_roas=round(trained.daily_roas, 4),
            weekly_roas=round(trained.weekly_roas, 4),
            per_segment_fatigue=dict(trained.per_segment_fatigue),
            persona_names=[bot.name for bot in self._persona_bots],
            yesterday_recap=self._yesterday_recap,
            done=done,
            reward=reward,
        )
