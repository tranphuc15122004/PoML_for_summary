"""Persistent metrics tracking for SFT/GRPO training.

Writes step-by-step metrics to dedicated CSV and JSONL files inside the
training output directory for easy monitoring and post-hoc analysis.

Usage:
    from metrics_logger import MetricsTracker

    tracker = MetricsTracker(output_dir="models/sft_lora")
    tracker.log_train(step=100, loss=1.23, lr=5e-5)
    tracker.log_eval(step=100, eval_loss=1.10)
    tracker.save_config({"learning_rate": 5e-5, ...})
"""

from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


# ==============================================================================
# MetricsTracker
# ==============================================================================

@dataclass
class MetricsTracker:
    """Write training and evaluation metrics to persistent files.

    Creates the following files under a ``metrics/`` subdirectory of
    ``output_dir``:

    * ``train_metrics.csv``   — step, loss, lr, epoch, grad_norm, gpu_mem (GB)
    * ``train_metrics.jsonl`` — same data, one JSON object per line
    * ``eval_metrics.csv``    — step, eval_loss
    * ``eval_metrics.jsonl``
    * ``config.json``         — snapshot of the training configuration
    """

    output_dir: str
    """Root output directory (e.g. ``models/sft_lora``)."""

    metrics_dir: str = ""
    """Sub-directory for metrics files. Set automatically in __post_init__."""

    _train_file_csv: str = ""
    _train_file_jsonl: str = ""
    _eval_file_csv: str = ""
    _eval_file_jsonl: str = ""
    _config_file: str = ""

    _train_csv_writer: Optional[csv.DictWriter] = None
    _train_csv_file: Optional[Any] = None
    _eval_csv_writer: Optional[csv.DictWriter] = None
    _eval_csv_file: Optional[Any] = None

    def __post_init__(self):
        self.metrics_dir = os.path.join(self.output_dir, "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)

        self._train_file_csv = os.path.join(self.metrics_dir, "train_metrics.csv")
        self._train_file_jsonl = os.path.join(self.metrics_dir, "train_metrics.jsonl")
        self._eval_file_csv = os.path.join(self.metrics_dir, "eval_metrics.csv")
        self._eval_file_jsonl = os.path.join(self.metrics_dir, "eval_metrics.jsonl")
        self._config_file = os.path.join(self.metrics_dir, "config.json")

    # ------------------------------------------------------------------
    # Train metrics
    # ------------------------------------------------------------------

    def log_train(self, **kwargs: Any) -> None:
        """Log one row of training metrics.

        Accepted keys (all optional):
            step, loss, lr, epoch, grad_norm, gpu_mem, total_steps, ...
        """
        row = {
            "step": kwargs.get("step"),
            "loss": kwargs.get("loss"),
            "lr": kwargs.get("lr"),
            "epoch": kwargs.get("epoch"),
            "grad_norm": kwargs.get("grad_norm"),
            "gpu_mem_gb": kwargs.get("gpu_mem_gb", self._get_gpu_mem()),
        }
        # Append extra fields caller may have passed
        for k, v in kwargs.items():
            if k not in row:
                row[k] = v

        # CSV
        self._init_train_csv(list(row.keys()))
        if self._train_csv_writer is not None:
            self._train_csv_writer.writerow(row)
            if self._train_csv_file is not None:
                self._train_csv_file.flush()
                os.fsync(self._train_csv_file.fileno())

        # JSONL (always append)
        with open(self._train_file_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def _init_train_csv(self, fieldnames: List[str]) -> None:
        """Open CSV file and write header on first call."""
        if self._train_csv_writer is not None:
            return
        file_exists = os.path.isfile(self._train_file_csv)
        self._train_csv_file = open(self._train_file_csv, "a", newline="")
        self._train_csv_writer = csv.DictWriter(
            self._train_csv_file, fieldnames=fieldnames
        )
        if not file_exists:
            self._train_csv_writer.writeheader()
            self._train_csv_file.flush()

    # ------------------------------------------------------------------
    # Eval metrics
    # ------------------------------------------------------------------

    def log_eval(self, **kwargs: Any) -> None:
        """Log one row of evaluation metrics.

        Accepted keys: step, eval_loss, ...
        """
        row = {"step": kwargs.get("step"), "eval_loss": kwargs.get("eval_loss")}
        for k, v in kwargs.items():
            if k not in row:
                row[k] = v

        self._init_eval_csv(list(row.keys()))
        if self._eval_csv_writer is not None:
            self._eval_csv_writer.writerow(row)
            if self._eval_csv_file is not None:
                self._eval_csv_file.flush()
                os.fsync(self._eval_csv_file.fileno())

        with open(self._eval_file_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def _init_eval_csv(self, fieldnames: List[str]) -> None:
        if self._eval_csv_writer is not None:
            return
        file_exists = os.path.isfile(self._eval_file_csv)
        self._eval_csv_file = open(self._eval_file_csv, "a", newline="")
        self._eval_csv_writer = csv.DictWriter(
            self._eval_csv_file, fieldnames=fieldnames
        )
        if not file_exists:
            self._eval_csv_writer.writeheader()
            self._eval_csv_file.flush()

    # ------------------------------------------------------------------
    # Config snapshot
    # ------------------------------------------------------------------

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save a snapshot of the training configuration."""
        with open(self._config_file, "w") as f:
            json.dump(config, f, default=str, indent=2)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _get_gpu_mem() -> Optional[float]:
        """Return current GPU memory allocated in GB, or None if no GPU."""
        if not torch.cuda.is_available():
            return None
        return round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)

    def close(self) -> None:
        """Close open file handles."""
        if self._train_csv_file is not None:
            self._train_csv_file.close()
        if self._eval_csv_file is not None:
            self._eval_csv_file.close()


# ==============================================================================
# MetricsCallback (Transformers Trainer callback)
# ==============================================================================

from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class MetricsCallback(TrainerCallback):
    """Transformers ``TrainerCallback`` that forwards logs to ``MetricsTracker``.

    Wire this into any ``transformers.Trainer`` (or ``trl.SFTTrainer``) to
    automatically log training and evaluation metrics to persistent files.
    """

    def __init__(self, tracker: MetricsTracker):
        self.tracker = tracker

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """Called after each logging step."""
        logs = state.log_history
        if not logs:
            return
        last = logs[-1]

        # Training metrics
        if "loss" in last:
            logger.info(
                "step=%d  loss=%.4f  lr=%.2e  epoch=%.4f  grad_norm=%.4f",
                state.global_step,
                last["loss"],
                last.get("learning_rate", args.learning_rate),
                round(state.epoch, 4) if state.epoch is not None else 0,
                last.get("grad_norm", 0),
            )
            self.tracker.log_train(
                step=state.global_step,
                loss=last["loss"],
                lr=last.get("learning_rate", args.learning_rate),
                epoch=round(state.epoch, 4) if state.epoch is not None else None,
                grad_norm=last.get("grad_norm"),
                samples_per_sec=last.get("train_samples_per_second"),
                steps_per_sec=last.get("train_steps_per_second"),
                total_steps=state.max_steps if state.max_steps else None,
            )

        # Evaluation metrics
        if "eval_loss" in last:
            logger.info(
                "step=%d  eval_loss=%.4f  eval_runtime=%.1fs",
                state.global_step,
                last["eval_loss"],
                last.get("eval_runtime", 0),
            )
            self.tracker.log_eval(
                step=state.global_step,
                eval_loss=last["eval_loss"],
                eval_runtime=last.get("eval_runtime"),
                eval_samples_per_sec=last.get("eval_samples_per_second"),
            )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.tracker.close()
