"""
Shared training observability for AdMarket Arena GRPO runs.

This module owns all observability code so the training notebooks
(``train_grpo.ipynb`` advertiser, ``train_oversight.ipynb`` oversight)
stay readable. It is deliberately neutral about *which* run is in
flight — the same callback works for both as long as the caller passes
a distinct ``wandb_project`` name.

Three orthogonal sinks (master Section 6.2):

  - **Weights & Biases** as the primary live dashboard. Public URL goes
    in the README + HF model cards. Free tier; no internet at venue is
    handled via the next two redundant sinks.
  - **Local CSV mirror** at ``logs/training_run_<timestamp>.csv`` with
    one row per logged step. Tamper-evident receipt of training; PNG
    plots are regenerable from this alone via ``scripts/make_plots.py``.
  - **JSONL episode dumps** at ``logs/episodes/ep_<step>.jsonl``: every
    Nth rollout's full sample (observations / actions / rewards /
    persona traits) for replay debugging and the demo before/after
    comparison.

Custom-metric injection is optional; pass a ``custom_metrics_fn``
callable returning a ``Dict[str, float]`` of business metrics computed
from the current rollout (weekly_roas, budget_utilization_pct, the
three behavioral diagnostics, etc.) and they'll be flushed alongside
the standard TRL/GRPO metrics.

Best-checkpoint tracking: pass a ``best_metric_name`` (default
``weekly_roas``) and the callback symlinks ``checkpoints/<run>/best/``
to the highest-scoring checkpoint observed. The advertiser eval +
HF Hub push pull from this path.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# TrainerCallback - wired into TRL via trainer.add_callback(...)
# ---------------------------------------------------------------------------
#
# We avoid a hard ``transformers.TrainerCallback`` import at module load
# time so the module stays importable in Plan 3's tests + ``inference.py``
# without dragging the huggingface stack into every entrypoint.

try:  # pragma: no cover - exercised in training notebook only
    from transformers.trainer_callback import TrainerCallback as _TrainerCallback
except ImportError:  # pragma: no cover
    class _TrainerCallback:  # type: ignore[no-redef]
        """Local fallback so this module imports without transformers."""

        def on_init_end(self, *args: Any, **kwargs: Any) -> None: ...
        def on_step_begin(self, *args: Any, **kwargs: Any) -> None: ...
        def on_step_end(self, *args: Any, **kwargs: Any) -> None: ...
        def on_log(self, *args: Any, **kwargs: Any) -> None: ...
        def on_save(self, *args: Any, **kwargs: Any) -> None: ...
        def on_train_end(self, *args: Any, **kwargs: Any) -> None: ...


# ---------------------------------------------------------------------------
# CSV mirror
# ---------------------------------------------------------------------------

@dataclass
class CSVMirror:
    """Write one row per step to a CSV. Columns are inferred and grow
    on first sight (the file is rewritten to add columns when a new
    metric appears so downstream readers always see a rectangular
    schema)."""

    path: Path
    columns: List[str] = field(default_factory=list)
    _rows: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_row(self, row: Dict[str, Any]) -> None:
        for k in row:
            if k not in self.columns:
                self.columns.append(k)
        self._rows.append(row)
        self._flush()

    def _flush(self) -> None:
        with self.path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._rows)


# ---------------------------------------------------------------------------
# JSONL episode dumper
# ---------------------------------------------------------------------------

@dataclass
class EpisodeDumper:
    """Write one JSONL file per dumped validation episode.

    File layout (one line per step in the episode):
        {"step": int, "observation": {...}, "action": {...},
         "reward": float, "info": {...}}
    """

    out_dir: Path

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def dump(self, training_step: int, episode_records: List[Dict[str, Any]]) -> Path:
        path = self.out_dir / f"ep_step{training_step:05d}.jsonl"
        with path.open("w") as f:
            for record in episode_records:
                f.write(json.dumps(record))
                f.write("\n")
        return path


# ---------------------------------------------------------------------------
# Best-checkpoint tracker
# ---------------------------------------------------------------------------

@dataclass
class BestCheckpointTracker:
    """Watches a metric stream and remembers the best step seen. The
    checkpoint dir is copied (not symlinked, to be Windows-friendly)
    into ``<checkpoint_root>/best/`` once we see a higher value.

    By default tracks ``weekly_roas`` (advertiser run) but the oversight
    run can pass ``best_metric_name='val_f1'`` for the same logic.
    """

    checkpoint_root: Path
    best_metric_name: str = "weekly_roas"
    higher_is_better: bool = True
    best_value: Optional[float] = None
    best_step: Optional[int] = None
    best_dir: Optional[Path] = None

    def __post_init__(self) -> None:
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.best_dir = self.checkpoint_root / "best"

    def consider(self, step: int, metrics: Dict[str, float]) -> bool:
        """Return True if this step is the new best."""
        if self.best_metric_name not in metrics:
            return False
        value = float(metrics[self.best_metric_name])
        if self.best_value is None:
            improved = True
        elif self.higher_is_better:
            improved = value > self.best_value
        else:
            improved = value < self.best_value
        if improved:
            self.best_value = value
            self.best_step = step
            return True
        return False

    def snapshot_from(self, step_dir: Path) -> Optional[Path]:
        """Copy the latest step checkpoint into ``best/`` if ``step_dir`` exists."""
        if self.best_dir is None or not step_dir.exists():
            return None
        if self.best_dir.exists():
            shutil.rmtree(self.best_dir)
        shutil.copytree(step_dir, self.best_dir)
        return self.best_dir


# ---------------------------------------------------------------------------
# The unified callback
# ---------------------------------------------------------------------------

@dataclass
class ArenaTrainingCallback(_TrainerCallback):
    """TRL TrainerCallback that wires W&B + CSV + JSONL + best-tracker.

    Reused by both training runs:

    Advertiser (Plan 3):
        callback = ArenaTrainingCallback(
            run_name="advertiser_grpo",
            wandb_project="admarket-arena-advertiser",
            log_dir=Path("logs"),
            checkpoint_root=Path("checkpoints/advertiser_run"),
            episode_dump_every=10,
            best_metric_name="weekly_roas",
            custom_metrics_fn=advertiser_validation_metrics,
        )

    Oversight (Plan 2):
        callback = ArenaTrainingCallback(
            run_name="oversight_grpo",
            wandb_project="admarket-arena-oversight",
            log_dir=Path("logs"),
            checkpoint_root=Path("checkpoints/oversight_run"),
            episode_dump_every=10,
            best_metric_name="val_f1",
            custom_metrics_fn=oversight_validation_metrics,
        )

    The callback never imports wandb at module load time — it imports
    lazily so a no-internet venue can drop the W&B sink and still keep
    CSV + JSONL working (matches Section 6.2 redundancy).
    """

    run_name: str
    wandb_project: str
    log_dir: Path
    checkpoint_root: Path
    episode_dump_every: int = 10
    save_checkpoint_every: int = 10
    best_metric_name: str = "weekly_roas"
    higher_is_better: bool = True
    custom_metrics_fn: Optional[Callable[[int], Dict[str, Any]]] = None
    use_wandb: bool = True
    wandb_config: Dict[str, Any] = field(default_factory=dict)

    csv_mirror: Optional[CSVMirror] = None
    episode_dumper: Optional[EpisodeDumper] = None
    best_tracker: Optional[BestCheckpointTracker] = None
    _wandb_initialized: bool = False
    _wandb_module: Any = None
    _step_counter: int = 0
    _start_time: float = 0.0

    def __post_init__(self) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        csv_path = self.log_dir / f"training_run_{self.run_name}_{timestamp}.csv"
        self.csv_mirror = CSVMirror(path=csv_path)

        ep_dir = self.log_dir / "episodes" / self.run_name
        self.episode_dumper = EpisodeDumper(out_dir=ep_dir)

        self.best_tracker = BestCheckpointTracker(
            checkpoint_root=self.checkpoint_root,
            best_metric_name=self.best_metric_name,
            higher_is_better=self.higher_is_better,
        )
        self._start_time = time.time()

    # --- W&B helpers ---
    def _maybe_init_wandb(self) -> None:
        if not self.use_wandb or self._wandb_initialized:
            return
        try:
            import wandb  # type: ignore
            self._wandb_module = wandb
        except ImportError:
            self.use_wandb = False
            return
        try:
            wandb.init(
                project=self.wandb_project,
                name=self.run_name,
                config={**self.wandb_config, "run_name": self.run_name},
                reinit=True,
            )
        except Exception:
            self.use_wandb = False
            return
        self._wandb_initialized = True

    def _log_to_wandb(self, metrics: Dict[str, Any]) -> None:
        if not self.use_wandb or self._wandb_module is None:
            return
        try:
            self._wandb_module.log(metrics)
        except Exception:
            pass

    # --- TrainerCallback hooks ---
    def on_train_begin(self, args: Any = None, state: Any = None, control: Any = None, **kwargs: Any) -> None:
        self._maybe_init_wandb()

    def on_log(
        self,
        args: Any = None,
        state: Any = None,
        control: Any = None,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if logs is None:
            return
        # Use TRL's reported global step if present, otherwise ours.
        step = int(getattr(state, "global_step", self._step_counter)) if state else self._step_counter
        self._step_counter = step

        merged: Dict[str, Any] = {
            "step": step,
            "wall_clock_min": round((time.time() - self._start_time) / 60.0, 3),
        }
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                merged[k] = float(v)

        if self.custom_metrics_fn is not None:
            try:
                custom = self.custom_metrics_fn(step) or {}
            except Exception as exc:  # pragma: no cover - defensive
                custom = {"custom_metrics_fn_error": str(exc)}
            for k, v in custom.items():
                if isinstance(v, (int, float)):
                    merged[k] = float(v)
                else:
                    merged[k] = v

        if self.csv_mirror is not None:
            self.csv_mirror.log_row({k: v for k, v in merged.items() if isinstance(v, (int, float))})

        self._log_to_wandb({k: v for k, v in merged.items() if isinstance(v, (int, float))})

        if self.best_tracker is not None:
            new_best = self.best_tracker.consider(step, {k: v for k, v in merged.items() if isinstance(v, (int, float))})
            if new_best:
                self._log_to_wandb({f"best/{self.best_metric_name}": self.best_tracker.best_value, "best/step": step})

    def on_save(
        self,
        args: Any = None,
        state: Any = None,
        control: Any = None,
        **kwargs: Any,
    ) -> None:
        if self.best_tracker is None:
            return
        if state is None:
            return
        step = int(getattr(state, "global_step", self._step_counter))
        # Heuristic: TRL writes step dirs as ``{output_dir}/checkpoint-{step}``.
        output_dir = getattr(args, "output_dir", None)
        if not output_dir:
            return
        candidate = Path(output_dir) / f"checkpoint-{step}"
        if (
            self.best_tracker.best_step == step
            and candidate.exists()
        ):
            self.best_tracker.snapshot_from(candidate)

    def on_train_end(
        self,
        args: Any = None,
        state: Any = None,
        control: Any = None,
        **kwargs: Any,
    ) -> None:
        if self.use_wandb and self._wandb_module is not None:
            try:
                self._wandb_module.finish()
            except Exception:
                pass

    # --- helpers used directly by the training notebook ---

    def dump_validation_episode(
        self,
        training_step: int,
        records: List[Dict[str, Any]],
    ) -> Optional[Path]:
        """Manual escape hatch for notebooks: dump a validation episode
        right now (the standard cadence is every ``episode_dump_every``
        steps but the notebook may want to dump on demand for the
        before/after demo)."""
        if self.episode_dumper is None:
            return None
        return self.episode_dumper.dump(training_step, records)


# ---------------------------------------------------------------------------
# Module-level convenience constructor (matches the master spec one-liner)
# ---------------------------------------------------------------------------

def make_arena_callback(
    run_name: str,
    wandb_project: str,
    *,
    log_dir: str = "logs",
    checkpoint_root: str = "checkpoints",
    episode_dump_every: int = 10,
    save_checkpoint_every: int = 10,
    best_metric_name: str = "weekly_roas",
    higher_is_better: bool = True,
    custom_metrics_fn: Optional[Callable[[int], Dict[str, Any]]] = None,
    use_wandb: bool = True,
    wandb_config: Optional[Dict[str, Any]] = None,
) -> ArenaTrainingCallback:
    return ArenaTrainingCallback(
        run_name=run_name,
        wandb_project=wandb_project,
        log_dir=Path(log_dir),
        checkpoint_root=Path(checkpoint_root) / run_name,
        episode_dump_every=episode_dump_every,
        save_checkpoint_every=save_checkpoint_every,
        best_metric_name=best_metric_name,
        higher_is_better=higher_is_better,
        custom_metrics_fn=custom_metrics_fn,
        use_wandb=use_wandb,
        wandb_config=wandb_config or {},
    )
