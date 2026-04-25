"""
Plan 3 — Advertiser training pipeline tests.

Coverage targets:

  - ``ARENA_TASKS`` configs match master Section 3.1.1 (steps + budgets
    + frequency caps).
  - ``PersonaBot`` produces valid ``AuctionAction``s; per-episode trait
    jitter never escapes [TRAIT_FLOOR, TRAIT_CEILING].
  - ``LLMPolicyBot`` swallows broken completions and returns a safe
    skip action; ``parse_llm_advertiser_action`` is robust to noise.
  - ``CurriculumScheduler`` promotes after the configured streak and
    not before; never demotes; tops out at the last tier.
  - ``ArenaTrainingCallback`` writes a CSV row, dumps a JSONL episode,
    and tracks the best metric.
  - ``scripts/make_plots`` emits valid PNG headers from a fake CSV.
  - ``scripts/advertiser_eval`` end-to-end on ``arena_easy`` with the
    mock trained policy: returns a finite weekly_roas and the four
    behavioral diagnostics.

These are smoke / unit tests — they don't load LLM weights, so they
run fast in CI alongside the Plan 2 oversight tests.

Run:
    pytest tests/test_advertiser_training.py -v
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from meta_ad_optimizer.competitors import (
    HELD_OUT_PERSONA,
    LLMPolicyBot,
    PERSONAS,
    PersonaBot,
    TRAIT_CEILING,
    TRAIT_FLOOR,
    TRAIT_NAMES,
    build_opponent_slate,
    jitter_persona,
    maxed_persona,
    parse_llm_advertiser_action,
)
from meta_ad_optimizer.curriculum_scheduler import (
    DEFAULT_TIER_ORDER,
    CurriculumScheduler,
    make_advertiser_curriculum,
)
from meta_ad_optimizer.models import AuctionAction, AuctionObservation
from meta_ad_optimizer.tasks import ARENA_TASKS, ArenaTaskConfig
from meta_ad_optimizer.training_callbacks import (
    ArenaTrainingCallback,
    BestCheckpointTracker,
    CSVMirror,
    EpisodeDumper,
    make_arena_callback,
)


# ---------------------------------------------------------------------------
# tasks.ARENA_TASKS
# ---------------------------------------------------------------------------

class TestArenaTasks:

    def test_three_tiers_present(self):
        assert set(ARENA_TASKS) == {"arena_easy", "arena_medium", "arena_hard"}

    def test_arena_easy_dimensions(self):
        cfg = ARENA_TASKS["arena_easy"]
        assert isinstance(cfg, ArenaTaskConfig)
        assert cfg.steps_per_episode == 60
        assert cfg.frequency_cap_per_user == 999  # effectively disabled
        assert cfg.persona_jitter is False

    def test_arena_hard_full_spec(self):
        cfg = ARENA_TASKS["arena_hard"]
        assert cfg.steps_per_episode == 350
        assert cfg.weekly_budget == pytest.approx(1000.0)
        assert cfg.daily_budget == pytest.approx(143.0)
        assert cfg.target_weekly_roas == pytest.approx(2.0)
        assert cfg.frequency_cap_per_user == 3

    def test_floor_price_progression(self):
        cfg = ARENA_TASKS["arena_medium"]
        assert cfg.floor_price_base == pytest.approx(0.25)
        assert cfg.floor_price_daily_increment == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# competitors.PersonaBot + jitter
# ---------------------------------------------------------------------------

class TestPersonaBot:

    def test_jitter_clipped_within_bounds(self):
        import random as _r
        rng = _r.Random(0)
        for _ in range(100):
            for spec in PERSONAS.values():
                traits = jitter_persona(spec, rng, jitter_scale=2.0)  # extreme
                for trait in TRAIT_NAMES:
                    assert TRAIT_FLOOR <= traits[trait] <= TRAIT_CEILING

    def test_persona_bid_returns_valid_action(self):
        bot = PersonaBot.from_persona_name("PerformanceMarketer", rng=None, jitter_enabled=False)
        obs = AuctionObservation(
            user_segment="gen_z_creator",
            user_interests=["fashion"],
            current_surface="feed",
            day_of_week=2,
            step_in_day=10,
            weekly_budget_remaining=400.0,
            daily_budget_remaining=80.0,
            spent_so_far_today=20.0,
            spent_so_far_week=120.0,
            recent_clearing_prices=[0.7, 0.6, 0.8],
            available_creatives=[{"target_segment": "gen_z_creator", "base_ctr": 0.1}],
            floor_price=0.5,
            frequency_cap_per_user=3,
        )
        action = bot.bid(obs, {"spent_today": 20.0, "daily_target": 100.0})
        assert isinstance(action, AuctionAction)
        assert action.bid_amount >= 0.0

    def test_spamflooder_stays_aggressive_under_fatigue(self):
        spam = PersonaBot.from_persona_name("SpamFlooder", rng=None, jitter_enabled=False)
        cautious = PersonaBot.from_persona_name("CautiousChallenger", rng=None, jitter_enabled=False)
        obs = AuctionObservation(
            user_segment="gen_z_creator",
            per_segment_fatigue={"gen_z_creator": 0.9},
            recent_clearing_prices=[0.5],
            available_creatives=[{"target_segment": "gen_z_creator", "base_ctr": 0.05}],
            daily_budget_remaining=100.0,
            floor_price=0.3,
        )
        spam_action = spam.bid(obs)
        cautious_action = cautious.bid(obs)
        assert spam_action.bid_amount > cautious_action.bid_amount, (
            "SpamFlooder ignores fatigue; CautiousChallenger should back off."
        )

    def test_held_out_persona_constructable(self):
        bot = PersonaBot.from_persona_name(HELD_OUT_PERSONA.name, rng=None, jitter_enabled=False)
        assert bot.spec.name == "OpportunisticArbitrageur"

    def test_maxed_persona_pins_dominant_trait(self):
        for spec in PERSONAS.values():
            traits = maxed_persona(spec)
            base = spec.base_vector()
            dominant = max(base, key=base.get)
            assert traits[dominant] == TRAIT_CEILING

    def test_build_opponent_slate_constructs_correct_count(self):
        import random as _r
        rng = _r.Random(0)
        slate = build_opponent_slate(["PremiumBrand", "BargainReseller"], rng=rng)
        assert len(slate) == 2
        assert all(isinstance(b, PersonaBot) for b in slate)


# ---------------------------------------------------------------------------
# competitors.LLMPolicyBot + parser
# ---------------------------------------------------------------------------

class TestLLMPolicyBot:

    def test_parse_clean_json(self):
        action = parse_llm_advertiser_action(
            '{"skip": false, "bid_amount": 1.25, "creative_id": 2}',
            n_creatives=4,
        )
        assert action.skip is False
        assert action.bid_amount == pytest.approx(1.25)
        assert action.creative_id == 2

    def test_parse_clamps_oversize_bid(self):
        action = parse_llm_advertiser_action(
            '{"skip": false, "bid_amount": 99.0, "creative_id": 0}',
            n_creatives=2,
        )
        assert action.bid_amount == pytest.approx(5.0)

    def test_parse_clamps_creative_index(self):
        action = parse_llm_advertiser_action(
            '{"skip": false, "bid_amount": 0.5, "creative_id": 999}',
            n_creatives=3,
        )
        assert action.creative_id == 2

    def test_parse_garbage_returns_skip(self):
        action = parse_llm_advertiser_action("I dunno", n_creatives=2)
        assert action.skip is True
        assert action.bid_amount == 0.0

    def test_zero_bid_falls_back_to_skip(self):
        action = parse_llm_advertiser_action(
            '{"skip": false, "bid_amount": 0.0, "creative_id": 0}',
            n_creatives=2,
        )
        assert action.skip is True

    def test_llm_bot_swallows_broken_completion(self):
        def broken(_sys, _user):
            raise RuntimeError("provider down")
        bot = LLMPolicyBot(completion_fn=broken)
        obs = AuctionObservation(available_creatives=[{"target_segment": "x", "base_ctr": 0.1}])
        action = bot.bid(obs)
        assert action.skip is True

    def test_llm_bot_uses_fallback_when_set(self):
        def broken(_sys, _user):
            raise RuntimeError("provider down")
        fallback = PersonaBot.from_persona_name("PremiumBrand", rng=None, jitter_enabled=False)
        bot = LLMPolicyBot(completion_fn=broken, fallback=fallback)
        obs = AuctionObservation(
            user_segment="gen_z_creator",
            available_creatives=[{"target_segment": "gen_z_creator", "base_ctr": 0.1}],
            floor_price=0.3,
        )
        action = bot.bid(obs)
        # Fallback returns the persona action; it may skip (low value) or not,
        # but the path executed successfully.
        assert isinstance(action, AuctionAction)


# ---------------------------------------------------------------------------
# curriculum_scheduler
# ---------------------------------------------------------------------------

class TestCurriculumScheduler:

    def test_initial_tier_is_easy(self):
        scheduler = CurriculumScheduler()
        assert scheduler.current_tier == "arena_easy"
        assert scheduler.current_streak == 0
        assert not scheduler.is_at_top

    def test_promotes_after_streak(self):
        scheduler = CurriculumScheduler(promotion_threshold=0.30, required_streak=3)
        for i in range(2):
            r = scheduler.step(0.4)
            assert r["promoted"] is False
            assert r["current_tier"] == "arena_easy"
        r = scheduler.step(0.4)
        assert r["promoted"] is True
        assert r["current_tier"] == "arena_medium"
        assert scheduler.current_streak == 0  # resets on promotion

    def test_streak_resets_on_low_reward(self):
        scheduler = CurriculumScheduler(promotion_threshold=0.30, required_streak=3)
        scheduler.step(0.5)
        scheduler.step(0.5)
        scheduler.step(0.1)
        assert scheduler.current_streak == 0
        scheduler.step(0.5)
        assert scheduler.current_streak == 1

    def test_does_not_promote_past_top_tier(self):
        scheduler = CurriculumScheduler(promotion_threshold=0.30, required_streak=2)
        for _ in range(2):
            scheduler.step(0.5)  # easy -> medium
        for _ in range(2):
            scheduler.step(0.5)  # medium -> hard
        assert scheduler.current_tier == "arena_hard"
        assert scheduler.is_at_top
        for _ in range(10):
            r = scheduler.step(0.99)
            assert r["promoted"] is False
            assert r["current_tier"] == "arena_hard"

    def test_on_promote_callback_fires(self):
        seen = []
        def cb(old, new, step):
            seen.append((old, new, step))
        scheduler = CurriculumScheduler(promotion_threshold=0.30, required_streak=2,
                                         on_promote=cb)
        scheduler.step(0.5, training_step=10)
        scheduler.step(0.5, training_step=20)
        assert seen == [("arena_easy", "arena_medium", 20)]

    def test_update_from_metrics_skips_when_missing(self):
        scheduler = CurriculumScheduler()
        result = scheduler.update_from_metrics({"loss": 0.5})
        assert result is None

    def test_make_advertiser_curriculum_defaults(self):
        scheduler = make_advertiser_curriculum()
        assert list(scheduler.tier_order) == DEFAULT_TIER_ORDER
        assert scheduler.promotion_threshold == 0.30
        assert scheduler.required_streak == 10


# ---------------------------------------------------------------------------
# training_callbacks
# ---------------------------------------------------------------------------

class TestTrainingCallbacks:

    def test_csv_mirror_writes_rows(self, tmp_path: Path):
        path = tmp_path / "run.csv"
        mirror = CSVMirror(path=path)
        mirror.log_row({"step": 0, "reward": 1.0})
        mirror.log_row({"step": 1, "reward": 1.5, "loss": 0.2})  # new column added later
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == 2
        assert rows[1]["loss"] == "0.2"

    def test_episode_dumper_emits_jsonl(self, tmp_path: Path):
        dumper = EpisodeDumper(out_dir=tmp_path)
        records = [{"step": 0, "obs": {"a": 1}}, {"step": 1, "obs": {"a": 2}}]
        path = dumper.dump(training_step=42, episode_records=records)
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["step"] == 0

    def test_best_tracker_records_higher_value(self, tmp_path: Path):
        tracker = BestCheckpointTracker(checkpoint_root=tmp_path, best_metric_name="weekly_roas")
        assert tracker.consider(0, {"weekly_roas": 1.0}) is True
        assert tracker.consider(1, {"weekly_roas": 0.5}) is False
        assert tracker.consider(2, {"weekly_roas": 1.5}) is True
        assert tracker.best_step == 2
        assert tracker.best_value == pytest.approx(1.5)

    def test_best_tracker_lower_is_better(self, tmp_path: Path):
        tracker = BestCheckpointTracker(
            checkpoint_root=tmp_path, best_metric_name="loss", higher_is_better=False,
        )
        tracker.consider(0, {"loss": 1.0})
        tracker.consider(1, {"loss": 1.2})
        tracker.consider(2, {"loss": 0.8})
        assert tracker.best_step == 2

    def test_best_tracker_snapshots_directory(self, tmp_path: Path):
        ckpt_root = tmp_path / "ckpts"
        tracker = BestCheckpointTracker(checkpoint_root=ckpt_root, best_metric_name="weekly_roas")
        step_dir = tmp_path / "checkpoint-1"
        step_dir.mkdir()
        (step_dir / "adapter.bin").write_text("payload")
        tracker.consider(1, {"weekly_roas": 1.0})
        result = tracker.snapshot_from(step_dir)
        assert result == ckpt_root / "best"
        assert (ckpt_root / "best" / "adapter.bin").read_text() == "payload"

    def test_callback_log_writes_csv(self, tmp_path: Path):
        cb = make_arena_callback(
            run_name="test_run",
            wandb_project="dummy",
            log_dir=str(tmp_path / "logs"),
            checkpoint_root=str(tmp_path / "ckpts"),
            use_wandb=False,
        )
        cb.on_log(args=None, state=None, control=None,
                  logs={"loss": 0.5, "weekly_roas": 1.2, "step": 0})
        rows = list(csv.DictReader(cb.csv_mirror.path.open()))
        assert len(rows) == 1
        assert float(rows[0]["weekly_roas"]) == pytest.approx(1.2)

    def test_callback_invokes_custom_metrics(self, tmp_path: Path):
        seen = []
        def custom(step):
            seen.append(step)
            return {"weekly_roas": 1.5}

        cb = make_arena_callback(
            run_name="t2",
            wandb_project="dummy",
            log_dir=str(tmp_path / "logs"),
            checkpoint_root=str(tmp_path / "ckpts"),
            use_wandb=False,
            custom_metrics_fn=custom,
        )
        cb.on_log(args=None, state=None, control=None, logs={"loss": 0.4})
        assert seen == [0]


# ---------------------------------------------------------------------------
# scripts.make_plots
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("matplotlib"),  # type: ignore[arg-type]
    reason="matplotlib not installed",
)
class TestMakePlots:

    def _write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r})
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    def test_plot_reward_curve_emits_png(self, tmp_path: Path):
        from meta_ad_optimizer.scripts.make_plots import plot_reward_curve  # type: ignore
        csv_path = tmp_path / "run.csv"
        self._write_csv(csv_path, [
            {"step": i, "episode_return_total": 0.1 * i} for i in range(20)
        ])
        out = tmp_path / "reward.png"
        result = plot_reward_curve(advertiser_csv=csv_path, baseline_csvs=None, out_path=out)
        assert result == out
        assert out.exists()
        assert out.stat().st_size > 100  # not empty
        with out.open("rb") as f:
            assert f.read(8) == b"\x89PNG\r\n\x1a\n"

    def test_plot_loss_curve_handles_missing_csv(self, tmp_path: Path):
        from meta_ad_optimizer.scripts.make_plots import plot_loss_curve  # type: ignore
        out = tmp_path / "loss.png"
        result = plot_loss_curve(advertiser_csv=tmp_path / "missing.csv", out_path=out)
        assert result.exists()  # placeholder PNG written

    def test_plot_budget_depletion_from_eval_json(self, tmp_path: Path):
        from meta_ad_optimizer.scripts.make_plots import plot_budget_depletion_comparison  # type: ignore
        eval_path = tmp_path / "eval.json"
        eval_path.write_text(json.dumps({
            "per_mode": {"standard": {"budget_depletion_day_mean": 6.5}},
            "baselines": {"random": {"budget_depletion_day_mean": 3.0},
                          "pacing": {"budget_depletion_day_mean": 7.0}},
        }))
        out = tmp_path / "depl.png"
        plot_budget_depletion_comparison(eval_json=eval_path, out_path=out)
        assert out.exists() and out.stat().st_size > 100


# ---------------------------------------------------------------------------
# scripts.advertiser_eval (full pipeline smoke)
# ---------------------------------------------------------------------------

class TestAdvertiserEvalSmoke:

    def test_run_advertiser_eval_arena_easy(self, tmp_path: Path):
        from meta_ad_optimizer.scripts.advertiser_eval import (  # type: ignore
            run_advertiser_eval,
        )
        out_path = tmp_path / "eval.json"
        payload = run_advertiser_eval(
            task_name="arena_easy",
            checkpoint=None,
            opponent_checkpoint=None,
            n_standard=2,
            n_edge_per_sub=1,
            n_selfplay=2,
            out_path=out_path,
            include_baselines=False,
        )
        assert out_path.exists()
        modes = payload.get("per_mode", {})
        assert set(modes.keys()) == {"standard", "edge", "selfplay"}
        for mode_name, m in modes.items():
            assert "weekly_roas_mean" in m, f"{mode_name} missing weekly_roas_mean"
            assert "bid_precision_mean" in m
            assert "budget_depletion_day_mean" in m
            assert "fatigue_sensitivity_mean" in m
            for v in (m["weekly_roas_mean"], m["bid_precision_mean"]):
                assert v == v  # not NaN

    def test_only_mode_runs_subset(self, tmp_path: Path):
        from meta_ad_optimizer.scripts.advertiser_eval import (  # type: ignore
            run_advertiser_eval,
        )
        out_path = tmp_path / "eval.json"
        payload = run_advertiser_eval(
            task_name="arena_easy",
            n_standard=2,
            out_path=out_path,
            include_baselines=False,
            only_mode="standard",
        )
        assert set(payload["per_mode"].keys()) == {"standard"}
