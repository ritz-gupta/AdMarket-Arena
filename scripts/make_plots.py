"""
CSV -> PNG plot generator (deterministic, re-runnable).

Owned by Plan 3, extended by Plan 2 with the two oversight plots. The
script is a *pure function* from CSV inputs to PNG outputs so judges
can regenerate every plot without re-running training (master Section
6.2 plot quality standards).

Five advertiser plots (Plan 3):

  1. ``reward_curve.png``           - mean episode return per GRPO step,
                                       trained vs untrained baseline.
  2. ``loss_curve.png``             - policy_loss + kl on shared axes.
  3. ``bid_precision_hist.png``     - per-validation-episode histogram
                                       of bid precision (overpayment ratio).
  4. ``budget_depletion_comparison.png`` - bar chart of budget
                                            depletion day per agent.
  5. ``fatigue_sensitivity_corr.png`` - correlation between fatigue
                                         spikes and creative switches
                                         over training.

Two oversight plots (Plan 2 hooks):

  6. ``oversight_f1_curve.png``     - F1 over GRPO steps (trained vs
                                       heuristic baseline).
  7. ``oversight_pr_scatter.png``   - precision-vs-recall scatter (one
                                       dot per GRPO step).

All plots:
  - both axes labeled with units (e.g. "GRPO step", "Mean weekly ROAS
    (x target)").
  - multiple runs on shared axes with distinct colors + a legend.
  - dpi 150 PNG output.
  - emit when the underlying CSV is missing (warn, don't crash) so the
    full plot pipeline can run incrementally during a slipping schedule.

Usage:
    python -m scripts.make_plots --advertiser-csv logs/training_run_advertiser.csv \
                                 --baseline-csv  logs/baseline_advertiser_random.csv \
                                 --eval-json     results/advertiser_eval.json \
                                 --plots-dir     assets/plots/ \
                                 --data-dir      assets/data/

Each plot function is independently importable so the training notebook
can call ``make_plots.plot_reward_curve(...)`` between checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Defer matplotlib import so this module is importable without it; the
# CLI checks for it at top-level.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False
    matplotlib = None  # type: ignore
    plt = None  # type: ignore


# ---------------------------------------------------------------------------
# CSV helpers (stdlib-only; no pandas dependency)
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def column(rows: List[Dict[str, str]], key: str) -> List[float]:
    out: List[float] = []
    for r in rows:
        val = r.get(key)
        if val is None or val == "":
            continue
        try:
            out.append(float(val))
        except ValueError:
            continue
    return out


def column_int(rows: List[Dict[str, str]], key: str) -> List[int]:
    return [int(round(v)) for v in column(rows, key)]


# ---------------------------------------------------------------------------
# Common figure helpers
# ---------------------------------------------------------------------------

def _require_mpl() -> None:
    if not _HAS_MPL:
        raise RuntimeError(
            "matplotlib is not installed. Install with `pip install matplotlib`."
        )


def _new_fig(figsize: Tuple[float, float] = (8.0, 5.0)) -> Tuple[Any, Any]:
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    return fig, ax


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _smooth(values: Sequence[float], window: int = 5) -> List[float]:
    if window <= 1 or len(values) <= window:
        return list(values)
    out: List[float] = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


# ---------------------------------------------------------------------------
# 1. reward_curve.png
# ---------------------------------------------------------------------------

def plot_reward_curve(
    advertiser_csv: Path,
    baseline_csvs: Optional[Dict[str, Path]] = None,
    out_path: Path = Path("assets/plots/reward_curve.png"),
    reward_key: str = "episode_return_total",
    step_key: str = "step",
) -> Path:
    """Mean episode return over GRPO step. Multiple runs on shared axes."""
    _require_mpl()
    fig, ax = _new_fig()

    series_drawn = 0
    rows = load_csv(advertiser_csv)
    if rows:
        steps = column(rows, step_key)
        rewards = column(rows, reward_key)
        if rewards:
            n = min(len(steps), len(rewards))
            ax.plot(steps[:n], _smooth(rewards[:n]),
                    label="trained advertiser", color="#0a8fdc", linewidth=2.0)
            series_drawn += 1

    palette = ["#d35400", "#27ae60", "#8e44ad", "#7f8c8d"]
    for idx, (label, path) in enumerate(sorted((baseline_csvs or {}).items())):
        rows = load_csv(path)
        if not rows:
            continue
        steps = column(rows, step_key)
        rewards = column(rows, reward_key)
        if not rewards:
            continue
        n = min(len(steps), len(rewards))
        ax.plot(steps[:n], _smooth(rewards[:n]),
                label=label, color=palette[idx % len(palette)],
                linestyle="--", linewidth=1.5)
        series_drawn += 1

    ax.set_xlabel("GRPO training step")
    ax.set_ylabel("Mean episode return (sum per episode)")
    ax.set_title("Advertiser reward curve")
    ax.grid(True, alpha=0.3)
    if series_drawn == 0:
        ax.text(0.5, 0.5, "no data — placeholder",
                transform=ax.transAxes, ha="center", va="center",
                color="#999", fontsize=12)
    else:
        ax.legend(loc="best")
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# 2. loss_curve.png
# ---------------------------------------------------------------------------

def plot_loss_curve(
    advertiser_csv: Path,
    out_path: Path = Path("assets/plots/loss_curve.png"),
    step_key: str = "step",
    loss_keys: Sequence[str] = ("policy_loss", "kl"),
) -> Path:
    """Policy loss and KL on shared axes (twin-y for KL if scales differ)."""
    _require_mpl()
    fig, ax_left = _new_fig()

    rows = load_csv(advertiser_csv)
    if not rows:
        ax_left.text(0.5, 0.5, "no data — placeholder",
                     transform=ax_left.transAxes, ha="center", va="center",
                     color="#999", fontsize=12)
        ax_left.set_xlabel("GRPO training step")
        ax_left.set_ylabel("Loss")
        return _save(fig, out_path)

    steps = column(rows, step_key)

    primary = column(rows, loss_keys[0])
    if primary:
        n = min(len(steps), len(primary))
        ax_left.plot(steps[:n], _smooth(primary[:n]),
                     label=loss_keys[0], color="#c0392b", linewidth=2.0)
    ax_left.set_xlabel("GRPO training step")
    ax_left.set_ylabel(f"{loss_keys[0]} (nats)", color="#c0392b")
    ax_left.tick_params(axis="y", labelcolor="#c0392b")
    ax_left.grid(True, alpha=0.3)

    if len(loss_keys) > 1:
        secondary = column(rows, loss_keys[1])
        if secondary:
            ax_right = ax_left.twinx()
            n = min(len(steps), len(secondary))
            ax_right.plot(steps[:n], _smooth(secondary[:n]),
                          label=loss_keys[1], color="#2c3e50", linewidth=1.5)
            ax_right.set_ylabel(f"{loss_keys[1]} (nats)", color="#2c3e50")
            ax_right.tick_params(axis="y", labelcolor="#2c3e50")
            lines, labels = ax_left.get_legend_handles_labels()
            lines2, labels2 = ax_right.get_legend_handles_labels()
            ax_left.legend(lines + lines2, labels + labels2, loc="upper right")
        else:
            ax_left.legend(loc="upper right")
    else:
        ax_left.legend(loc="upper right")

    ax_left.set_title("Advertiser GRPO loss curves")
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# 3. bid_precision_hist.png
# ---------------------------------------------------------------------------

def plot_bid_precision_hist(
    eval_json: Optional[Path] = None,
    advertiser_csv: Optional[Path] = None,
    out_path: Path = Path("assets/plots/bid_precision_hist.png"),
) -> Path:
    """Histogram of per-validation-episode bid precision (overpayment ratio).

    Bid precision = ``mean((bid_amount - clearing_price) / clearing_price)``
    across won auctions in an episode. Lower is better. Plots trained
    vs untrained on shared axes when both are present in the eval JSON.
    """
    _require_mpl()
    fig, ax = _new_fig()

    drawn = 0
    if eval_json is not None and eval_json.exists():
        with eval_json.open() as f:
            data = json.load(f)
        # Schema: {"per_mode": {<mode>: {"bid_precision_episodes": [...]}}}
        per_mode = data.get("per_mode", {}) if isinstance(data, dict) else {}
        palette = ["#0a8fdc", "#d35400", "#27ae60"]
        for idx, (mode, payload) in enumerate(per_mode.items()):
            if not isinstance(payload, dict):
                continue
            samples = payload.get("bid_precision_episodes", [])
            samples = [float(x) for x in samples if isinstance(x, (int, float))]
            if not samples:
                continue
            ax.hist(samples, bins=20, alpha=0.55, label=f"trained ({mode})",
                    color=palette[idx % len(palette)])
            drawn += 1

        baselines = data.get("baselines", {}) if isinstance(data, dict) else {}
        for idx, (label, payload) in enumerate(baselines.items()):
            if not isinstance(payload, dict):
                continue
            samples = payload.get("bid_precision_episodes", [])
            samples = [float(x) for x in samples if isinstance(x, (int, float))]
            if not samples:
                continue
            ax.hist(samples, bins=20, alpha=0.40, label=f"baseline ({label})",
                    color="#7f8c8d", histtype="step", linewidth=2.0)
            drawn += 1

    if drawn == 0 and advertiser_csv is not None:
        rows = load_csv(advertiser_csv)
        samples = column(rows, "bid_precision")
        if samples:
            ax.hist(samples, bins=20, alpha=0.6, label="training validation",
                    color="#0a8fdc")
            drawn += 1

    ax.set_xlabel("Bid precision (overpayment ratio = (bid - clearing) / clearing)")
    ax.set_ylabel("Validation episode count")
    ax.set_title("Bid precision distribution")
    ax.grid(True, alpha=0.3)
    if drawn == 0:
        ax.text(0.5, 0.5, "no data — placeholder",
                transform=ax.transAxes, ha="center", va="center",
                color="#999", fontsize=12)
    else:
        ax.legend(loc="best")
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# 4. budget_depletion_comparison.png
# ---------------------------------------------------------------------------

def plot_budget_depletion_comparison(
    eval_json: Path,
    out_path: Path = Path("assets/plots/budget_depletion_comparison.png"),
) -> Path:
    """Bar chart: which day each agent depletes its weekly budget.

    Higher day = better pacing. Greedy depletes early, pacing baseline
    lasts ~Sunday, trained agent should match or beat pacing.
    """
    _require_mpl()
    fig, ax = _new_fig()

    if not eval_json.exists():
        ax.text(0.5, 0.5, "no data — placeholder",
                transform=ax.transAxes, ha="center", va="center",
                color="#999", fontsize=12)
        ax.set_xlabel("Agent")
        ax.set_ylabel("Budget depletion day (1-7, higher = better pacing)")
        return _save(fig, out_path)

    with eval_json.open() as f:
        data = json.load(f)

    agents: List[str] = []
    days: List[float] = []
    palette: List[str] = []

    per_mode = data.get("per_mode", {}) if isinstance(data, dict) else {}
    for mode, payload in per_mode.items():
        if not isinstance(payload, dict):
            continue
        bdd = payload.get("budget_depletion_day_mean")
        if bdd is None:
            continue
        agents.append(f"trained\n({mode})")
        days.append(float(bdd))
        palette.append("#0a8fdc")

    baselines = data.get("baselines", {}) if isinstance(data, dict) else {}
    for label, payload in baselines.items():
        if not isinstance(payload, dict):
            continue
        bdd = payload.get("budget_depletion_day_mean")
        if bdd is None:
            continue
        agents.append(label)
        days.append(float(bdd))
        palette.append("#d35400")

    if agents:
        ax.bar(agents, days, color=palette, edgecolor="black", linewidth=0.5)
        for i, v in enumerate(days):
            ax.text(i, v + 0.05, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    else:
        ax.text(0.5, 0.5, "no data — placeholder",
                transform=ax.transAxes, ha="center", va="center",
                color="#999", fontsize=12)

    ax.set_xlabel("Agent")
    ax.set_ylabel("Budget depletion day (higher = better pacing)")
    ax.set_title("Budget depletion day comparison")
    ax.grid(True, alpha=0.3, axis="y")
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# 5. fatigue_sensitivity_corr.png
# ---------------------------------------------------------------------------

def plot_fatigue_sensitivity(
    advertiser_csv: Path,
    out_path: Path = Path("assets/plots/fatigue_sensitivity_corr.png"),
    step_key: str = "step",
    metric_key: str = "fatigue_sensitivity",
) -> Path:
    """Correlation between fatigue spikes and creative-switch behavior,
    plotted as a function of training step.

    Untrained policy ~ 0 correlation; trained should trend toward
    strongly negative (the agent learns to stop serving fatigued
    segments).
    """
    _require_mpl()
    fig, ax = _new_fig()

    rows = load_csv(advertiser_csv)
    drawn = False
    if rows:
        steps = column(rows, step_key)
        values = column(rows, metric_key)
        if values:
            n = min(len(steps), len(values))
            ax.plot(steps[:n], _smooth(values[:n]),
                    color="#0a8fdc", linewidth=2.0,
                    label="fatigue sensitivity (training)")
            ax.axhline(0.0, color="#aaa", linestyle="--", linewidth=1.0,
                       label="untrained baseline (~0)")
            drawn = True

    ax.set_xlabel("GRPO training step")
    ax.set_ylabel("Fatigue sensitivity (corr.; lower = stronger learned avoidance)")
    ax.set_title("Fatigue sensitivity over training")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.0, 1.0)
    if drawn:
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "no data — placeholder",
                transform=ax.transAxes, ha="center", va="center",
                color="#999", fontsize=12)
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# 6 + 7. Plan 2 oversight plot hooks
# ---------------------------------------------------------------------------

def plot_oversight_f1_curve(
    oversight_csv: Path,
    out_path: Path = Path("assets/plots/oversight_f1_curve.png"),
    step_key: str = "step",
    f1_keys: Sequence[str] = ("val_f1", "train_f1"),
    heuristic_baseline: Optional[float] = None,
) -> Path:
    """F1 over GRPO step (oversight). Plan 2 calls this with its own CSV."""
    _require_mpl()
    fig, ax = _new_fig()

    rows = load_csv(oversight_csv)
    drawn = 0
    palette = ["#27ae60", "#0a8fdc"]
    if rows:
        steps = column(rows, step_key)
        for idx, key in enumerate(f1_keys):
            values = column(rows, key)
            if not values:
                continue
            n = min(len(steps), len(values))
            ax.plot(steps[:n], _smooth(values[:n]),
                    label=key, color=palette[idx % len(palette)],
                    linewidth=2.0)
            drawn += 1

    if heuristic_baseline is not None:
        ax.axhline(heuristic_baseline, color="#7f8c8d", linestyle=":",
                   linewidth=1.5, label=f"heuristic baseline ({heuristic_baseline:.2f})")
        drawn += 1

    ax.set_xlabel("GRPO training step")
    ax.set_ylabel("F1 score")
    ax.set_title("Oversight F1 curve")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    if drawn:
        ax.legend(loc="best")
    else:
        ax.text(0.5, 0.5, "no data — placeholder",
                transform=ax.transAxes, ha="center", va="center",
                color="#999", fontsize=12)
    return _save(fig, out_path)


def plot_oversight_pr_scatter(
    oversight_csv: Path,
    out_path: Path = Path("assets/plots/oversight_pr_scatter.png"),
    precision_key: str = "val_precision",
    recall_key: str = "val_recall",
    step_key: str = "step",
) -> Path:
    """Scatter plot: one dot per GRPO step with precision on x, recall on y.
    Color encodes step so judges can see the agent walking out toward
    the (1, 1) corner over training.
    """
    _require_mpl()
    fig, ax = _new_fig()

    rows = load_csv(oversight_csv)
    if not rows:
        ax.text(0.5, 0.5, "no data — placeholder",
                transform=ax.transAxes, ha="center", va="center",
                color="#999", fontsize=12)
        ax.set_xlabel("Precision")
        ax.set_ylabel("Recall")
        return _save(fig, out_path)

    precision = column(rows, precision_key)
    recall = column(rows, recall_key)
    steps = column(rows, step_key)
    n = min(len(precision), len(recall), len(steps))
    if n == 0:
        ax.text(0.5, 0.5, "no data — placeholder",
                transform=ax.transAxes, ha="center", va="center",
                color="#999", fontsize=12)
        ax.set_xlabel("Precision")
        ax.set_ylabel("Recall")
        return _save(fig, out_path)

    sc = ax.scatter(precision[:n], recall[:n],
                    c=steps[:n], cmap="viridis", s=40, edgecolor="black", linewidth=0.4)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("GRPO step")
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_title("Oversight precision vs recall over training")
    ax.grid(True, alpha=0.3)
    ax.plot([0, 1], [0, 1], color="#aaa", linestyle=":", linewidth=1.0,
            label="y = x")
    ax.legend(loc="lower right")
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclass
class _PlotJob:
    name: str
    fn: Any
    kwargs: Dict[str, Any]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    parser.add_argument("--advertiser-csv", type=str, default="logs/training_run_advertiser.csv")
    parser.add_argument("--baseline-csv", type=str, default=None,
                        help="Optional baseline (e.g. random/pacing) reward CSV.")
    parser.add_argument("--baseline-label", type=str, default="baseline")
    parser.add_argument("--eval-json", type=str, default="results/advertiser_eval.json",
                        help="Path to eval_results.json from scripts/advertiser_eval.py.")
    parser.add_argument("--oversight-csv", type=str, default=None,
                        help="Plan 2 oversight CSV; if set, also generate the 2 oversight plots.")
    parser.add_argument("--oversight-heuristic-f1", type=float, default=None,
                        help="Optional heuristic F1 horizontal line on the oversight F1 plot.")
    parser.add_argument("--plots-dir", type=str, default="assets/plots")
    args = parser.parse_args()

    if not _HAS_MPL:
        print("[plots] matplotlib not installed; cannot generate PNGs.", file=sys.stderr)
        return 2

    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    advertiser_csv = Path(args.advertiser_csv)
    eval_json = Path(args.eval_json)
    baseline_csvs: Dict[str, Path] = {}
    if args.baseline_csv:
        baseline_csvs[args.baseline_label] = Path(args.baseline_csv)

    jobs: List[_PlotJob] = [
        _PlotJob("reward_curve", plot_reward_curve, {
            "advertiser_csv": advertiser_csv,
            "baseline_csvs": baseline_csvs,
            "out_path": plots_dir / "reward_curve.png",
        }),
        _PlotJob("loss_curve", plot_loss_curve, {
            "advertiser_csv": advertiser_csv,
            "out_path": plots_dir / "loss_curve.png",
        }),
        _PlotJob("bid_precision_hist", plot_bid_precision_hist, {
            "eval_json": eval_json if eval_json.exists() else None,
            "advertiser_csv": advertiser_csv,
            "out_path": plots_dir / "bid_precision_hist.png",
        }),
        _PlotJob("budget_depletion_comparison", plot_budget_depletion_comparison, {
            "eval_json": eval_json,
            "out_path": plots_dir / "budget_depletion_comparison.png",
        }),
        _PlotJob("fatigue_sensitivity", plot_fatigue_sensitivity, {
            "advertiser_csv": advertiser_csv,
            "out_path": plots_dir / "fatigue_sensitivity_corr.png",
        }),
    ]

    if args.oversight_csv:
        oversight_csv = Path(args.oversight_csv)
        jobs.append(_PlotJob("oversight_f1_curve", plot_oversight_f1_curve, {
            "oversight_csv": oversight_csv,
            "out_path": plots_dir / "oversight_f1_curve.png",
            "heuristic_baseline": args.oversight_heuristic_f1,
        }))
        jobs.append(_PlotJob("oversight_pr_scatter", plot_oversight_pr_scatter, {
            "oversight_csv": oversight_csv,
            "out_path": plots_dir / "oversight_pr_scatter.png",
        }))

    for job in jobs:
        try:
            out = job.fn(**job.kwargs)
            print(f"[plots] {job.name} -> {out}")
        except Exception as exc:
            print(f"[plots] {job.name} FAILED: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
