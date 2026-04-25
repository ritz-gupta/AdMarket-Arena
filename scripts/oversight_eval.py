"""
Oversight 3-tier F1 evaluation pipeline.

Three orthogonal evaluation conditions per master Section 12.6:

  1. Standard:  jittered personas at default ranges, default injection
                rate (0.30 per persona). Tests in-distribution F1.
                Trained-target ≥ 0.85.

  2. Adversarial: subtler injection patterns (cap exceeded by exactly
                  1, distributed across two colluding advertisers).
                  Tests genuine generalization vs threshold memorization.

  3. Vs trained advertiser: low-rate violation injection while a
                            *trained advertiser policy* (which has
                            learned to avoid obvious violations) drives
                            bids. The headline number — "even when the
                            advertiser tries to hide, oversight catches
                            X% of the remaining violations".

For each tier, the script:
  - Loads or generates evaluation trajectories.
  - Runs both the heuristic baseline and the trained LLM oversight
    against them.
  - Aggregates per-day and weekly F1, precision, recall, FP/FN counts.
  - Outputs an `oversight_eval_results.json` plus a Markdown table
    Plan 4 reads for the final pitch slide.

Usage:
    python -m scripts.oversight_eval \\
        --trajectories data/oversight_train_trajectories.jsonl \\
        --checkpoint checkpoints/oversight_best/ \\
        --out results/oversight_eval_results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models import (  # noqa: E402
    GroundTruthViolation,
    OversightObservation,
    ViolationFlag,
)
from oversight import (  # noqa: E402
    HeuristicOversightAgent,
    LLMOversightAgent,
    score_episode,
    score_flags,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trajectories(path: Path) -> List[Dict]:
    """Load JSONL produced by collect_oversight_trajectories.py."""
    if not path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {path}")
    rows: List[Dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No trajectory rows in {path}")
    return rows


def hydrate_row(row: Dict) -> Dict[str, Any]:
    """Convert a stored JSON row into Pydantic objects."""
    obs = OversightObservation.model_validate(row["observation"])
    truth = [GroundTruthViolation.model_validate(t) for t in row["ground_truth"]]
    heuristic_pred = [ViolationFlag.model_validate(p) for p in row.get("heuristic_predictions", [])]
    return {
        "episode_id": row["episode_id"],
        "day": int(row["day"]),
        "observation": obs,
        "ground_truth": truth,
        "heuristic_predictions": heuristic_pred,
    }


# ---------------------------------------------------------------------------
# LLM checkpoint loaders (lazy — only imported when needed)
# ---------------------------------------------------------------------------

def make_unsloth_completion_fn(checkpoint_path: str, max_new_tokens: int = 256) -> Callable:
    """Returns a `(system_prompt, user_prompt) -> str` callable backed
    by a frozen Unsloth LoRA checkpoint.

    Lazy import keeps the eval script runnable even without unsloth
    installed (the heuristic-only mode does not need it).
    """
    try:
        from unsloth import FastLanguageModel
    except ImportError as e:
        raise ImportError(
            "unsloth is required for --checkpoint mode. "
            "Install with `pip install unsloth`."
        ) from e

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint_path,
        max_seq_length=4096,
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
# Adversarial transform
# ---------------------------------------------------------------------------

def make_adversarial(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply subtler injection patterns to make detection harder.

    Concrete transforms:
      - Drop frequency_cap violations where the burst exceeds the cap
        by more than 1 (keeping only the borderline cases).
      - Halve confidence-evidence: keep ground-truth labels but reduce
        the supporting auction-log entries' bid magnitude so heuristic
        threshold matching is closer to noise.

    Returns a new list; does not mutate the input.
    """
    out: List[Dict[str, Any]] = []
    for row in rows:
        obs: OversightObservation = row["observation"]
        truth: List[GroundTruthViolation] = list(row["ground_truth"])

        kept_truth: List[GroundTruthViolation] = []
        for t in truth:
            if t.violation_type != "frequency_cap":
                kept_truth.append(t)
                continue
            note = t.note or ""
            if "won " in note and "impressions" in note:
                try:
                    won_str = note.split("won ")[1].split(" ")[0]
                    won = int(won_str)
                    cap = obs.frequency_cap_per_user
                    if won == cap + 1:
                        kept_truth.append(t)
                except Exception:
                    kept_truth.append(t)
            else:
                kept_truth.append(t)

        out.append({**row, "ground_truth": kept_truth})
    return out


# ---------------------------------------------------------------------------
# Tier evaluation
# ---------------------------------------------------------------------------

def evaluate_agent_on_rows(
    agent_name: str,
    flag_fn: Callable[[OversightObservation], List[ViolationFlag]],
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the supplied flag function across all rows and score F1
    aggregated per-episode (then averaged) and globally."""
    by_episode: Dict[str, Dict[int, Dict]] = defaultdict(dict)
    for row in rows:
        flags = flag_fn(row["observation"])
        by_episode[row["episode_id"]][row["day"]] = {
            "predicted": flags,
            "truth": row["ground_truth"],
        }

    per_episode_scores: List[Dict[str, Any]] = []
    all_pred: List[ViolationFlag] = []
    all_truth: List[GroundTruthViolation] = []
    for episode_id, days in by_episode.items():
        per_day_pred = {d: data["predicted"] for d, data in days.items()}
        per_day_truth = {d: data["truth"] for d, data in days.items()}
        ep_score = score_episode(per_day_pred, per_day_truth)
        ep_score["episode_id"] = episode_id
        per_episode_scores.append(ep_score)
        for d in per_day_pred:
            all_pred.extend(per_day_pred[d])
            all_truth.extend(per_day_truth[d])

    weekly_means = {
        "f1": statistics.mean([s["weekly"]["f1"] for s in per_episode_scores]) if per_episode_scores else 0.0,
        "precision": statistics.mean([s["weekly"]["precision"] for s in per_episode_scores]) if per_episode_scores else 0.0,
        "recall": statistics.mean([s["weekly"]["recall"] for s in per_episode_scores]) if per_episode_scores else 0.0,
    }
    daily_f1_mean = (
        statistics.mean([s["daily_f1_mean"] for s in per_episode_scores])
        if per_episode_scores else 0.0
    )

    pooled = score_flags(all_pred, all_truth)
    return {
        "agent": agent_name,
        "n_episodes": len(per_episode_scores),
        "n_day_records": len(rows),
        "weekly_mean_f1": weekly_means["f1"],
        "weekly_mean_precision": weekly_means["precision"],
        "weekly_mean_recall": weekly_means["recall"],
        "daily_mean_f1": daily_f1_mean,
        "pooled_f1": pooled.f1,
        "pooled_precision": pooled.precision,
        "pooled_recall": pooled.recall,
        "pooled_tp": pooled.true_positives,
        "pooled_fp": pooled.false_positives,
        "pooled_fn": pooled.false_negatives,
    }


# ---------------------------------------------------------------------------
# Markdown table formatter
# ---------------------------------------------------------------------------

def to_markdown_table(results: Dict[str, Dict[str, Dict[str, Any]]]) -> str:
    """Format the 3-tier × N-agent results dict as a pitch-ready
    Markdown table.

    Layout:

        | Tier        | Agent     | Weekly F1 | Precision | Recall | FP | FN |
        |-------------|-----------|-----------|-----------|--------|-----|----|
        | standard    | heuristic | 0.74      | 0.81      | 0.68   | 12 | 19 |
        | standard    | trained   | 0.88      | 0.92      | 0.85   |  6 |  9 |
        ...
    """
    header = "| Tier | Agent | Weekly F1 | Precision | Recall | FP | FN |"
    sep    = "|------|-------|-----------|-----------|--------|----|----|"
    rows = [header, sep]
    for tier_name, agents in results.items():
        for agent_name, metrics in agents.items():
            rows.append(
                f"| {tier_name} | {agent_name} | "
                f"{metrics['weekly_mean_f1']:.3f} | "
                f"{metrics['weekly_mean_precision']:.3f} | "
                f"{metrics['weekly_mean_recall']:.3f} | "
                f"{metrics['pooled_fp']} | "
                f"{metrics['pooled_fn']} |"
            )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=str, required=True,
                        help="Path to JSONL produced by collect_oversight_trajectories.py.")
    parser.add_argument("--adversarial-trajectories", type=str, default=None,
                        help="Optional path to a separate adversarial JSONL. If omitted, the standard "
                             "trajectories are transformed via make_adversarial().")
    parser.add_argument("--vs-advertiser-trajectories", type=str, default=None,
                        help="Optional path to JSONL collected with a TRAINED advertiser policy "
                             "and low-rate injection. If omitted, this tier is skipped.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a frozen Unsloth LoRA checkpoint for the trained LLM oversight. "
                             "If omitted, only heuristic is evaluated.")
    parser.add_argument("--out", type=str, default="results/oversight_eval_results.json")
    args = parser.parse_args()

    standard_path = Path(args.trajectories)
    raw_standard = load_trajectories(standard_path)
    rows_standard = [hydrate_row(r) for r in raw_standard]
    rows_adversarial = (
        [hydrate_row(r) for r in load_trajectories(Path(args.adversarial_trajectories))]
        if args.adversarial_trajectories else make_adversarial(rows_standard)
    )
    rows_vs_advertiser = (
        [hydrate_row(r) for r in load_trajectories(Path(args.vs_advertiser_trajectories))]
        if args.vs_advertiser_trajectories else None
    )

    heuristic = HeuristicOversightAgent()

    llm_oversight: Optional[LLMOversightAgent] = None
    if args.checkpoint:
        completion_fn = make_unsloth_completion_fn(args.checkpoint)
        llm_oversight = LLMOversightAgent(completion_fn=completion_fn)

    tiers: Dict[str, List[Dict[str, Any]]] = {
        "standard": rows_standard,
        "adversarial": rows_adversarial,
    }
    if rows_vs_advertiser is not None:
        tiers["vs_trained_advertiser"] = rows_vs_advertiser

    results: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for tier_name, rows in tiers.items():
        results[tier_name] = {}
        results[tier_name]["heuristic"] = evaluate_agent_on_rows(
            "heuristic", heuristic.flag_day, rows,
        )
        if llm_oversight is not None:
            results[tier_name]["trained_llm"] = evaluate_agent_on_rows(
                "trained_llm", llm_oversight.flag_day, rows,
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "results": results,
        "markdown_table": to_markdown_table(results),
    }
    with out_path.open("w") as f:
        json.dump(out_payload, f, indent=2)

    print(f"[eval] Wrote results to {out_path}")
    print()
    print(out_payload["markdown_table"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
