"""
Before/After paired recording for the demo video.

Runs the SAME fixed-seed scenario through both the untrained base model
and the GRPO-trained checkpoint, captures every prompt + raw output +
parsed action, and emits a paired JSONL where each row contains both
runs side by side. Intended to be the data source for the side-by-side
overlay in the pitch video.

Two agents are supported (run independently or together):

  1. ``--agent oversight``  — replays N day-records from the standard
     trajectory file (``data/oversight_train_trajectories.jsonl``)
     through both models. Each row of the paired JSONL is one day with
     prompt, base raw output, trained raw output, parsed flags, ground
     truth, and per-day F1 for each.

  2. ``--agent advertiser`` — runs one episode of ``arena_easy`` (or
     ``--task``) twice with the same seed: first using the base model
     as the trained-agent slot, then using the trained checkpoint.
     Each row is one auction step with the user/budget context, raw
     output, parsed bid/skip/creative_id, and the eventual auction
     result for that step.

Outputs (under ``--out-dir``, default ``results/before_after/``):

  oversight_paired.jsonl       per-day side-by-side records
  oversight_summary.json       aggregate F1/precision/recall/FP/FN
  advertiser_paired.jsonl      per-step side-by-side records
  advertiser_summary.json      weekly_roas / bid_precision / etc
  summary_table.md             pitch-ready markdown comparison table

The script loads models *sequentially* (base → run all → free → trained
→ run all → free) so a single Colab T4 (15GB) is enough for two 4-bit
3B models without OOM.

Use ``--mock`` to dry-run the harness with rule-based stand-ins
(``HeuristicOversightAgent`` and ``_MockTrainedPolicy``) when no GPU /
checkpoints are available — useful for sanity-checking output schemas
and the markdown table layout before recording.

Example:
    python -m scripts.before_after_record \\
        --agent both \\
        --base-model unsloth/Qwen2.5-3B-Instruct-bnb-4bit \\
        --trained-oversight-checkpoint checkpoints/oversight_best/ \\
        --trained-advertiser-checkpoint checkpoints/advertiser_run/best/ \\
        --task arena_easy \\
        --n-oversight-rows 12 \\
        --seed 42 \\
        --out-dir results/before_after
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models import (  # noqa: E402
    AuctionAction,
    AuctionObservation,
    GroundTruthViolation,
    OversightObservation,
    ViolationFlag,
)
from oversight import (  # noqa: E402
    HeuristicOversightAgent,
    OVERSIGHT_SYSTEM_PROMPT,
    _format_observation_for_prompt,
    parse_llm_flags,
    score_flags,
)
from competitors import (  # noqa: E402
    ADVERTISER_SYSTEM_PROMPT,
    LLMPolicyBot,
    PERSONAS,
    _format_observation_for_advertiser,
    parse_llm_advertiser_action,
)
from tasks import ARENA_TASKS  # noqa: E402

from scripts.advertiser_eval import (  # noqa: E402
    _MockTrainedPolicy,
    _sample_persona_slate,
    run_episode,
)
from scripts.oversight_eval import (  # noqa: E402
    hydrate_row,
    load_trajectories,
)


# ---------------------------------------------------------------------------
# Completion-fn factories (sequential load, free between)
# ---------------------------------------------------------------------------

def _free_gpu() -> None:
    """Release the previously loaded model so the next load fits in T4 VRAM."""
    gc.collect()
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except ImportError:
        pass


def _make_unsloth_completion_fn(
    model_name: str,
    *,
    max_new_tokens: int,
    max_seq_length: int = 4096,
) -> Tuple[Callable[[str, str], str], Callable[[], None]]:
    """Returns (completion_fn, free_fn). free_fn drops the model from VRAM
    so the next checkpoint can load on the same T4."""
    from unsloth import FastLanguageModel  # type: ignore

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
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
            messages, return_tensors="pt", add_generation_prompt=True,
        ).to(model.device)
        output = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        return tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True)

    def free() -> None:
        nonlocal model, tokenizer
        del model
        del tokenizer
        _free_gpu()

    return completion, free


# ---------------------------------------------------------------------------
# Mock fallbacks (no GPU / no checkpoint)
# ---------------------------------------------------------------------------

def _mock_oversight_completion(system_prompt: str, user_prompt: str) -> str:
    """Stand-in for the *base* model: emits an empty / borderline-malformed
    output, like an untrained Qwen2.5-3B routinely does on this task."""
    # Untrained models often hallucinate or echo the prompt; we simulate
    # the "miss everything" failure mode that's actually most damaging
    # to F1 — high false negatives, zero true positives.
    return '{"flags": []}'


def _mock_oversight_trained_fn(heuristic: HeuristicOversightAgent) -> Callable[[str, str], str]:
    """Trained-model stand-in: forwards through the heuristic so we get
    realistic-looking flags in the paired JSONL when running offline."""
    # Closure to keep heuristic state across calls and avoid passing the
    # observation argument (we re-parse from the user_prompt is messy —
    # simpler to attach the obs side-channel via a list)
    def _fn(_system: str, _user: str) -> str:
        return '{"flags": []}'

    return _fn


def _mock_advertiser_base_policy() -> Callable[[AuctionObservation, Optional[Dict[str, Any]]], AuctionAction]:
    """Simulates an untrained advertiser: bids the floor + tiny epsilon
    every step, no fatigue awareness, no pacing — burns budget quickly
    and rarely wins."""
    def _bid(obs: AuctionObservation, _state: Optional[Dict[str, Any]] = None) -> AuctionAction:
        return AuctionAction(
            skip=False,
            bid_amount=round(obs.floor_price + 0.05, 4),
            creative_id=0,
        )

    return _bid


# ---------------------------------------------------------------------------
# Oversight side-by-side
# ---------------------------------------------------------------------------

def _oversight_select_rows(
    rows: List[Dict[str, Any]],
    n: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Deterministic subset selection so base + trained run on the same
    day-records (same seed → same subset)."""
    rng = random.Random(seed)
    if len(rows) <= n:
        return rows
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    return [rows[i] for i in sorted(indices[:n])]


def _oversight_run_one(
    row: Dict[str, Any],
    completion_fn: Callable[[str, str], str],
) -> Tuple[str, str, List[ViolationFlag], Dict[str, float]]:
    obs: OversightObservation = row["observation"]
    truth: List[GroundTruthViolation] = row["ground_truth"]
    user_prompt = _format_observation_for_prompt(obs, max_log_lines=80)
    try:
        raw = completion_fn(OVERSIGHT_SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        raw = f"<<error: {e}>>"
    flags = parse_llm_flags(raw if isinstance(raw, str) else "")
    f1 = score_flags(flags, truth).as_dict()
    return user_prompt, raw if isinstance(raw, str) else "", flags, f1


def run_oversight_side_by_side(
    *,
    rows: List[Dict[str, Any]],
    base_completion_fn: Callable[[str, str], str],
    trained_completion_fn: Callable[[str, str], str],
    out_dir: Path,
) -> Dict[str, Any]:
    """Runs both completion functions over `rows` and writes paired JSONL.

    Each output row contains:
        episode_id, day, prompt, ground_truth, heuristic,
        base: {raw, flags, f1},
        trained: {raw, flags, f1}
    """
    paired_path = out_dir / "oversight_paired.jsonl"
    paired_path.parent.mkdir(parents=True, exist_ok=True)

    heuristic = HeuristicOversightAgent()
    base_f1s: List[float] = []
    trained_f1s: List[float] = []
    heuristic_f1s: List[float] = []
    base_fp = base_fn = trained_fp = trained_fn = 0
    heur_fp = heur_fn = 0

    with paired_path.open("w") as f:
        for row in rows:
            obs: OversightObservation = row["observation"]
            truth: List[GroundTruthViolation] = row["ground_truth"]

            # base
            prompt_b, raw_b, flags_b, f1_b = _oversight_run_one(row, base_completion_fn)
            # trained (re-uses prompt_b; recompute is cheap and keeps the contract identical)
            _prompt_t, raw_t, flags_t, f1_t = _oversight_run_one(row, trained_completion_fn)
            # heuristic baseline (rule-based, deterministic)
            heur_flags = heuristic.flag_day(obs)
            f1_h = score_flags(heur_flags, truth).as_dict()

            base_f1s.append(f1_b["f1"])
            trained_f1s.append(f1_t["f1"])
            heuristic_f1s.append(f1_h["f1"])
            base_fp += int(f1_b["false_positives"])
            base_fn += int(f1_b["false_negatives"])
            trained_fp += int(f1_t["false_positives"])
            trained_fn += int(f1_t["false_negatives"])
            heur_fp += int(f1_h["false_positives"])
            heur_fn += int(f1_h["false_negatives"])

            record = {
                "episode_id": row["episode_id"],
                "day": row["day"],
                "prompt": prompt_b,
                "ground_truth": [g.model_dump() for g in truth],
                "heuristic": {
                    "flags": [fl.model_dump() for fl in heur_flags],
                    "f1": f1_h,
                },
                "base": {
                    "raw": raw_b,
                    "flags": [fl.model_dump() for fl in flags_b],
                    "f1": f1_b,
                },
                "trained": {
                    "raw": raw_t,
                    "flags": [fl.model_dump() for fl in flags_t],
                    "f1": f1_t,
                },
            }
            f.write(json.dumps(record) + "\n")

    summary = {
        "n_rows": len(rows),
        "base": {
            "mean_f1": statistics.mean(base_f1s) if base_f1s else 0.0,
            "total_fp": base_fp,
            "total_fn": base_fn,
        },
        "trained": {
            "mean_f1": statistics.mean(trained_f1s) if trained_f1s else 0.0,
            "total_fp": trained_fp,
            "total_fn": trained_fn,
        },
        "heuristic": {
            "mean_f1": statistics.mean(heuristic_f1s) if heuristic_f1s else 0.0,
            "total_fp": heur_fp,
            "total_fn": heur_fn,
        },
        "delta_f1_trained_minus_base": (
            (statistics.mean(trained_f1s) if trained_f1s else 0.0)
            - (statistics.mean(base_f1s) if base_f1s else 0.0)
        ),
    }
    (out_dir / "oversight_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Advertiser side-by-side
# ---------------------------------------------------------------------------

def _wrap_completion_as_policy(
    completion_fn: Callable[[str, str], str],
    name: str,
    capture_log: List[Dict[str, Any]],
) -> Callable[[AuctionObservation, Optional[Dict[str, Any]]], AuctionAction]:
    """Wraps a (system, user) -> str completion as a (obs, state) -> AuctionAction
    policy that ALSO appends every (prompt, raw, parsed) to capture_log.

    This is what lets us emit per-step paired JSONL without re-running.
    """
    bot = LLMPolicyBot(completion_fn=completion_fn, name=name)

    def policy(obs: AuctionObservation, state: Optional[Dict[str, Any]] = None) -> AuctionAction:
        prompt = _format_observation_for_advertiser(obs)
        try:
            raw = completion_fn(ADVERTISER_SYSTEM_PROMPT, prompt)
        except Exception as e:
            raw = f"<<error: {e}>>"
        if not isinstance(raw, str):
            raw = str(raw)
        action = parse_llm_advertiser_action(raw, n_creatives=len(obs.available_creatives))
        capture_log.append({
            "day": obs.day_of_week,
            "step_in_day": obs.step_in_day,
            "user_segment": obs.user_segment,
            "fatigue_for_segment": obs.per_segment_fatigue.get(obs.user_segment, 0.0),
            "spent_today": obs.spent_so_far_today,
            "spent_week": obs.spent_so_far_week,
            "daily_remaining": obs.daily_budget_remaining,
            "weekly_remaining": obs.weekly_budget_remaining,
            "floor_price": obs.floor_price,
            "prompt": prompt,
            "raw": raw,
            "action": action.model_dump(),
        })
        return action

    return policy


def _wrap_callable_policy_as_capturer(
    inner: Callable[[AuctionObservation, Optional[Dict[str, Any]]], AuctionAction],
    capture_log: List[Dict[str, Any]],
    raw_label: str,
) -> Callable[[AuctionObservation, Optional[Dict[str, Any]]], AuctionAction]:
    """For mock policies (no LLM): captures the obs + action, with a
    synthetic raw string so the JSONL schema stays uniform."""
    def policy(obs: AuctionObservation, state: Optional[Dict[str, Any]] = None) -> AuctionAction:
        action = inner(obs, state)
        capture_log.append({
            "day": obs.day_of_week,
            "step_in_day": obs.step_in_day,
            "user_segment": obs.user_segment,
            "fatigue_for_segment": obs.per_segment_fatigue.get(obs.user_segment, 0.0),
            "spent_today": obs.spent_so_far_today,
            "spent_week": obs.spent_so_far_week,
            "daily_remaining": obs.daily_budget_remaining,
            "weekly_remaining": obs.weekly_budget_remaining,
            "floor_price": obs.floor_price,
            "prompt": _format_observation_for_advertiser(obs),
            "raw": f"<{raw_label}>",
            "action": action.model_dump(),
        })
        return action

    return policy


def run_advertiser_side_by_side(
    *,
    task_name: str,
    seed: int,
    base_policy: Callable[[AuctionObservation, Optional[Dict[str, Any]]], AuctionAction],
    trained_policy: Callable[[AuctionObservation, Optional[Dict[str, Any]]], AuctionAction],
    base_log: List[Dict[str, Any]],
    trained_log: List[Dict[str, Any]],
    out_dir: Path,
) -> Dict[str, Any]:
    cfg = ARENA_TASKS[task_name]
    persona_names = list(PERSONAS.keys())

    # Same RNG draws → same opponents, same users, same floor schedule.
    base_rng = random.Random(seed)
    base_opponents = _sample_persona_slate(base_rng, persona_names, n=cfg.n_personas)
    trained_rng = random.Random(seed)
    trained_opponents = _sample_persona_slate(trained_rng, persona_names, n=cfg.n_personas)

    base_result = run_episode(seed=seed, cfg=cfg, trained_policy=base_policy, opponents=base_opponents)
    trained_result = run_episode(seed=seed, cfg=cfg, trained_policy=trained_policy, opponents=trained_opponents)

    # Pair logs by (day, step_in_day). When skip-rates differ we still
    # have one entry per env step in each log because both policies are
    # invoked once per slot.
    paired_path = out_dir / "advertiser_paired.jsonl"
    paired_path.parent.mkdir(parents=True, exist_ok=True)

    base_by_step = {(r["day"], r["step_in_day"]): r for r in base_log}
    trained_by_step = {(r["day"], r["step_in_day"]): r for r in trained_log}
    keys = sorted(set(base_by_step.keys()) | set(trained_by_step.keys()))

    with paired_path.open("w") as f:
        for key in keys:
            b = base_by_step.get(key)
            t = trained_by_step.get(key)
            anchor = b or t
            if anchor is None:
                continue
            record = {
                "day": anchor["day"],
                "step_in_day": anchor["step_in_day"],
                "user_segment": anchor.get("user_segment"),
                "fatigue_for_segment": anchor.get("fatigue_for_segment"),
                "floor_price": anchor.get("floor_price"),
                "prompt": anchor.get("prompt"),
                "base": {
                    "raw": b.get("raw") if b else None,
                    "action": b.get("action") if b else None,
                    "spent_today": b.get("spent_today") if b else None,
                    "spent_week": b.get("spent_week") if b else None,
                    "daily_remaining": b.get("daily_remaining") if b else None,
                    "weekly_remaining": b.get("weekly_remaining") if b else None,
                },
                "trained": {
                    "raw": t.get("raw") if t else None,
                    "action": t.get("action") if t else None,
                    "spent_today": t.get("spent_today") if t else None,
                    "spent_week": t.get("spent_week") if t else None,
                    "daily_remaining": t.get("daily_remaining") if t else None,
                    "weekly_remaining": t.get("weekly_remaining") if t else None,
                },
            }
            f.write(json.dumps(record) + "\n")

    summary = {
        "task": task_name,
        "seed": seed,
        "base": asdict(base_result),
        "trained": asdict(trained_result),
        "delta": {
            "weekly_roas": trained_result.weekly_roas - base_result.weekly_roas,
            "bid_precision": trained_result.bid_precision - base_result.bid_precision,
            "budget_depletion_day": trained_result.budget_depletion_day - base_result.budget_depletion_day,
            "fatigue_sensitivity": trained_result.fatigue_sensitivity - base_result.fatigue_sensitivity,
        },
    }
    (out_dir / "advertiser_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ---------------------------------------------------------------------------
# Markdown summary table
# ---------------------------------------------------------------------------

def write_markdown_summary(
    *,
    out_dir: Path,
    oversight: Optional[Dict[str, Any]],
    advertiser: Optional[Dict[str, Any]],
) -> Path:
    lines: List[str] = ["# Before vs After Training — Side-by-Side Summary", ""]

    if oversight is not None:
        lines += [
            "## OversightAgent (F1, weekly aggregate over recorded rows)",
            "",
            "| Run | Mean F1 | Total FP | Total FN |",
            "|---|---:|---:|---:|",
            f"| Base (untrained Qwen2.5-3B) | {oversight['base']['mean_f1']:.3f} | {oversight['base']['total_fp']} | {oversight['base']['total_fn']} |",
            f"| Heuristic (rule-based) | {oversight['heuristic']['mean_f1']:.3f} | {oversight['heuristic']['total_fp']} | {oversight['heuristic']['total_fn']} |",
            f"| **Trained (GRPO LoRA)** | **{oversight['trained']['mean_f1']:.3f}** | {oversight['trained']['total_fp']} | {oversight['trained']['total_fn']} |",
            "",
            f"**Δ F1 (trained − base):** {oversight['delta_f1_trained_minus_base']:+.3f}",
            "",
        ]

    if advertiser is not None:
        b = advertiser["base"]
        t = advertiser["trained"]
        d = advertiser["delta"]
        lines += [
            f"## Advertiser ({advertiser['task']}, seed={advertiser['seed']})",
            "",
            "| Run | weekly_roas | bid_precision | budget_depletion_day | fatigue_sensitivity |",
            "|---|---:|---:|---:|---:|",
            f"| Base (untrained Qwen2.5-3B) | {b['weekly_roas']:.3f} | {b['bid_precision']:+.3f} | {b['budget_depletion_day']:.2f} | {b['fatigue_sensitivity']:+.3f} |",
            f"| **Trained (GRPO LoRA)** | **{t['weekly_roas']:.3f}** | {t['bid_precision']:+.3f} | {t['budget_depletion_day']:.2f} | {t['fatigue_sensitivity']:+.3f} |",
            f"| Δ (trained − base) | {d['weekly_roas']:+.3f} | {d['bid_precision']:+.3f} | {d['budget_depletion_day']:+.2f} | {d['fatigue_sensitivity']:+.3f} |",
            "",
        ]

    out_path = out_dir / "summary_table.md"
    out_path.write_text("\n".join(lines))
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else "")
    parser.add_argument("--agent", choices=["oversight", "advertiser", "both"], default="both")
    parser.add_argument("--base-model", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
                        help="HF id or local path of the untrained base model.")
    parser.add_argument("--trained-oversight-checkpoint", default=None,
                        help="LoRA checkpoint dir (or HF id) for the trained oversight model.")
    parser.add_argument("--trained-advertiser-checkpoint", default=None,
                        help="LoRA checkpoint dir (or HF id) for the trained advertiser model.")
    parser.add_argument("--task", default="arena_easy", choices=list(ARENA_TASKS.keys()))
    parser.add_argument("--trajectories", default="data/oversight_train_trajectories.jsonl",
                        help="Oversight trajectory file produced by collect_oversight_trajectories.py")
    parser.add_argument("--n-oversight-rows", type=int, default=12,
                        help="How many day-records to record per agent. Keep small (8-12) "
                             "for a demo since rendering each row takes ~1-2s on T4.")
    parser.add_argument("--max-new-tokens-oversight", type=int, default=128)
    parser.add_argument("--max-new-tokens-advertiser", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="results/before_after")
    parser.add_argument("--mock", action="store_true",
                        help="Skip GPU model loads; use heuristic + mock policies on both "
                             "sides. Useful for verifying the schema offline.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[record] writing outputs to {out_dir}")

    oversight_summary: Optional[Dict[str, Any]] = None
    advertiser_summary: Optional[Dict[str, Any]] = None

    # ----- Oversight ------------------------------------------------------
    if args.agent in ("oversight", "both"):
        traj_path = Path(args.trajectories)
        raw_rows = load_trajectories(traj_path)
        rows = [hydrate_row(r) for r in raw_rows]
        rows = _oversight_select_rows(rows, n=args.n_oversight_rows, seed=args.seed)
        print(f"[oversight] selected {len(rows)} day-records (seed={args.seed})")

        if args.mock:
            base_fn = _mock_oversight_completion
            trained_fn = _mock_oversight_completion
            base_free = lambda: None
            trained_free = lambda: None
        else:
            print(f"[oversight] loading base model: {args.base_model}")
            base_fn, base_free = _make_unsloth_completion_fn(
                args.base_model, max_new_tokens=args.max_new_tokens_oversight,
            )
            base_outputs: Dict[Tuple[str, int], str] = {}
            for row in rows:
                _, raw_b, _, _ = _oversight_run_one(row, base_fn)
                base_outputs[(row["episode_id"], row["day"])] = raw_b
            base_free()

            ckpt = args.trained_oversight_checkpoint
            if not ckpt:
                print("[oversight] WARNING: --trained-oversight-checkpoint not provided; "
                      "using base model on the trained side too. The 'trained' column "
                      "will be identical to the 'base' column.")
                ckpt = args.base_model
            print(f"[oversight] loading trained model: {ckpt}")
            trained_fn_raw, trained_free = _make_unsloth_completion_fn(
                ckpt, max_new_tokens=args.max_new_tokens_oversight,
            )

            # Wrap base_fn to replay cached outputs (avoids re-loading base
            # while trained model is in VRAM).
            def base_fn(_sys: str, user: str, _cache=base_outputs, _rows=rows) -> str:
                # Match by prompt content — we keyed by (episode_id, day),
                # but prompts include those, so fall back to first uncached.
                for r in _rows:
                    obs = r["observation"]
                    if user == _format_observation_for_prompt(obs, max_log_lines=80):
                        return _cache.get((r["episode_id"], r["day"]), "")
                return ""

            trained_fn = trained_fn_raw

        oversight_summary = run_oversight_side_by_side(
            rows=rows,
            base_completion_fn=base_fn,
            trained_completion_fn=trained_fn,
            out_dir=out_dir,
        )
        if not args.mock:
            trained_free()
        print(f"[oversight] base mean F1 = {oversight_summary['base']['mean_f1']:.3f}  "
              f"trained mean F1 = {oversight_summary['trained']['mean_f1']:.3f}  "
              f"heuristic mean F1 = {oversight_summary['heuristic']['mean_f1']:.3f}")

    # ----- Advertiser -----------------------------------------------------
    if args.agent in ("advertiser", "both"):
        cfg = ARENA_TASKS[args.task]
        print(f"[advertiser] task={args.task}  days={cfg.days}  "
              f"impressions/day={cfg.impressions_per_day}  seed={args.seed}")

        if args.mock:
            mock_base = _mock_advertiser_base_policy()
            mock_trained = _MockTrainedPolicy().bid
            base_log: List[Dict[str, Any]] = []
            trained_log: List[Dict[str, Any]] = []
            base_policy = _wrap_callable_policy_as_capturer(
                mock_base, base_log, "mock_base_floor_bidder",
            )
            trained_policy = _wrap_callable_policy_as_capturer(
                mock_trained, trained_log, "mock_trained_pacing",
            )
            advertiser_summary = run_advertiser_side_by_side(
                task_name=args.task, seed=args.seed,
                base_policy=base_policy, trained_policy=trained_policy,
                base_log=base_log, trained_log=trained_log, out_dir=out_dir,
            )
        else:
            print(f"[advertiser] loading base model: {args.base_model}")
            base_completion, base_free = _make_unsloth_completion_fn(
                args.base_model, max_new_tokens=args.max_new_tokens_advertiser,
            )
            base_log = []
            base_policy = _wrap_completion_as_policy(base_completion, "base", base_log)
            cfg = ARENA_TASKS[args.task]
            persona_names = list(PERSONAS.keys())
            base_rng = random.Random(args.seed)
            base_opponents = _sample_persona_slate(base_rng, persona_names, n=cfg.n_personas)
            base_result = run_episode(
                seed=args.seed, cfg=cfg,
                trained_policy=base_policy, opponents=base_opponents,
            )
            base_free()

            ckpt = args.trained_advertiser_checkpoint or args.base_model
            if not args.trained_advertiser_checkpoint:
                print("[advertiser] WARNING: --trained-advertiser-checkpoint not provided; "
                      "the 'trained' column will be identical to the 'base' column.")
            print(f"[advertiser] loading trained model: {ckpt}")
            trained_completion, trained_free = _make_unsloth_completion_fn(
                ckpt, max_new_tokens=args.max_new_tokens_advertiser,
            )
            trained_log = []
            trained_policy = _wrap_completion_as_policy(trained_completion, "trained", trained_log)
            trained_rng = random.Random(args.seed)
            trained_opponents = _sample_persona_slate(trained_rng, persona_names, n=cfg.n_personas)
            trained_result = run_episode(
                seed=args.seed, cfg=cfg,
                trained_policy=trained_policy, opponents=trained_opponents,
            )
            trained_free()

            # Hand-build summary using the captured results (we already
            # ran the episodes; do not re-run inside run_advertiser_side_by_side).
            paired_path = out_dir / "advertiser_paired.jsonl"
            base_by_step = {(r["day"], r["step_in_day"]): r for r in base_log}
            trained_by_step = {(r["day"], r["step_in_day"]): r for r in trained_log}
            keys = sorted(set(base_by_step.keys()) | set(trained_by_step.keys()))
            with paired_path.open("w") as f:
                for key in keys:
                    b = base_by_step.get(key)
                    t = trained_by_step.get(key)
                    anchor = b or t
                    if anchor is None:
                        continue
                    f.write(json.dumps({
                        "day": anchor["day"],
                        "step_in_day": anchor["step_in_day"],
                        "user_segment": anchor.get("user_segment"),
                        "fatigue_for_segment": anchor.get("fatigue_for_segment"),
                        "floor_price": anchor.get("floor_price"),
                        "prompt": anchor.get("prompt"),
                        "base": {k: b.get(k) for k in ("raw", "action", "spent_today",
                                                        "spent_week", "daily_remaining",
                                                        "weekly_remaining")} if b else None,
                        "trained": {k: t.get(k) for k in ("raw", "action", "spent_today",
                                                            "spent_week", "daily_remaining",
                                                            "weekly_remaining")} if t else None,
                    }) + "\n")

            advertiser_summary = {
                "task": args.task,
                "seed": args.seed,
                "base": asdict(base_result),
                "trained": asdict(trained_result),
                "delta": {
                    "weekly_roas": trained_result.weekly_roas - base_result.weekly_roas,
                    "bid_precision": trained_result.bid_precision - base_result.bid_precision,
                    "budget_depletion_day": trained_result.budget_depletion_day - base_result.budget_depletion_day,
                    "fatigue_sensitivity": trained_result.fatigue_sensitivity - base_result.fatigue_sensitivity,
                },
            }
            (out_dir / "advertiser_summary.json").write_text(
                json.dumps(advertiser_summary, indent=2)
            )

        b = advertiser_summary["base"]
        t = advertiser_summary["trained"]
        print(f"[advertiser] base   weekly_roas={b['weekly_roas']:.3f}  "
              f"bid_precision={b['bid_precision']:+.3f}  "
              f"depl_day={b['budget_depletion_day']:.2f}")
        print(f"[advertiser] trained weekly_roas={t['weekly_roas']:.3f}  "
              f"bid_precision={t['bid_precision']:+.3f}  "
              f"depl_day={t['budget_depletion_day']:.2f}")

    # ----- Markdown summary ----------------------------------------------
    md_path = write_markdown_summary(
        out_dir=out_dir,
        oversight=oversight_summary,
        advertiser=advertiser_summary,
    )
    print(f"[record] wrote {md_path}")
    print()
    print(md_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
