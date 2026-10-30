"""
experiment.py
=============
Experiment and run management utilities for ECG heartbeat classification.
Manages isolated run directories (results/runs/run1, run2, etc.) and records
configuration parameters, hyperparameters, and evaluation metrics in settings.txt.
"""

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pyhere import here

try:
  from src.constants import CLASS_NAMES, DEFAULT_RUNS_DIR
except ImportError:
  from constants import CLASS_NAMES, DEFAULT_RUNS_DIR

__all__ = [
  "append_run_evaluation",
  "get_latest_run_dir",
  "get_next_run_dir",
  "resolve_run_dir",
  "save_run_settings",
]


def get_next_run_dir(
  base_dir: str | Path = DEFAULT_RUNS_DIR, prefix: str = "run"
) -> tuple[str, Path]:
  """
  Find the next available run folder in base_dir by scanning for prefix + integer.

  Parameters
  ----------
  base_dir : str | Path
      Base directory where runs are kept (default: results/runs).
  prefix : str
      Folder name prefix (default: 'run').

  Returns
  -------
  tuple[str, Path]
      Run name and absolute path (e.g. ('run1', Path('.../results/runs/run1'))).
  """
  base_p = Path(base_dir) if Path(base_dir).is_absolute() else here(base_dir)
  base_p.mkdir(parents=True, exist_ok=True)

  existing: list[int] = []
  pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
  for entry in base_p.iterdir():
    if entry.is_dir():
      match = pattern.match(entry.name)
      if match:
        existing.append(int(match.group(1)))

  next_idx = max(existing) + 1 if existing else 1
  run_name = f"{prefix}{next_idx}"
  return run_name, base_p / run_name


def get_latest_run_dir(
  base_dir: str | Path = DEFAULT_RUNS_DIR, prefix: str = "run"
) -> tuple[str, Path]:
  """
  Locate the highest-numbered run folder in base_dir.

  Parameters
  ----------
  base_dir : str | Path
      Base directory where runs are kept.
  prefix : str
      Folder prefix to match.

  Returns
  -------
  tuple[str, Path]
      Run name and path for the latest run.
  """
  base_p = Path(base_dir) if Path(base_dir).is_absolute() else here(base_dir)
  if not base_p.exists():
    raise FileNotFoundError(f"Runs directory does not exist: {base_p}")

  existing: list[tuple[int, Path]] = []
  pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
  for entry in base_p.iterdir():
    if entry.is_dir():
      match = pattern.match(entry.name)
      if match:
        existing.append((int(match.group(1)), entry))

  if not existing:
    raise FileNotFoundError(f"No run folders matching '{prefix}*' found in {base_p}")

  existing.sort(key=lambda item: item[0])
  latest_idx, latest_path = existing[-1]
  return f"{prefix}{latest_idx}", latest_path


def resolve_run_dir(
  run_arg: str | None,
  base_dir: str | Path = DEFAULT_RUNS_DIR,
  prefix: str = "run",
  must_exist: bool = False,
) -> Path | None:
  """
  Resolve the --run argument into an experiment directory Path.

  Parameters
  ----------
  run_arg : str | None
      CLI argument:
      - None: returns None (runs feature disabled, standard paths used).
      - 'auto' / 'new' / 'next': finds next sequential directory (run1, run2, ...).
      - 'latest' / 'last': finds highest-numbered existing run directory.
      - '<name>': explicit directory name or path.
  base_dir : str | Path
      Root directory where runs are organized.
  prefix : str
      Folder prefix (default: 'run').
  must_exist : bool
      Whether to raise FileNotFoundError if the directory does not exist.

  Returns
  -------
  Path | None
      Resolved directory Path, or None if run_arg is empty.
  """
  if not run_arg:
    return None

  clean = run_arg.strip()
  match clean.lower():
    case "auto" | "new" | "next":
      _, run_p = get_next_run_dir(base_dir=base_dir, prefix=prefix)
      return run_p
    case "latest" | "last":
      _, run_p = get_latest_run_dir(base_dir=base_dir, prefix=prefix)
      return run_p
    case _:
      custom_p = Path(clean)
      if custom_p.is_absolute():
        resolved = custom_p
      else:
        base_p = Path(base_dir) if Path(base_dir).is_absolute() else here(base_dir)
        resolved = base_p / clean

      if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Run directory not found: {resolved}")
      return resolved


def save_run_settings(
  run_dir: str | Path,
  settings: dict[str, Any],
  filename: str = "settings.txt",
) -> Path:
  """
  Write full model, training, and data settings to a plain text file.

  Parameters
  ----------
  run_dir : str | Path
      Destination directory for the run.
  settings : dict[str, Any]
      Dictionary containing configuration key-value pairs.
  filename : str
      Output text filename (default: 'settings.txt').

  Returns
  -------
  Path
      Path to the written settings file.
  """
  run_p = Path(run_dir)
  run_p.mkdir(parents=True, exist_ok=True)
  out_path = run_p / filename

  now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

  lines = [
    "=" * 78,
    f"ECG Arrhythmia Classification - Run Settings: {run_p.name}",
    "=" * 78,
    f"Created:             {now_utc}",
    f"Run Directory:       {run_p.resolve()}",
    "",
    "[Model Architecture]",
    f"Model Name:          {settings.get('model_name', 'N/A')}",
    f"Model Class:         {settings.get('model_class', 'N/A')}",
    f"Parameters:          {settings.get('parameters', 'N/A')}",
    f"Random Seed:         {settings.get('seed', 'N/A')}",
    f"Quick Test:          {settings.get('quick_test', False)}",
  ]

  hyperparams = settings.get("hyperparameters", {})
  if hyperparams:
    lines.append("")
    lines.append("[Hyperparameters]")
    for key, val in hyperparams.items():
      lines.append(f"{key:<21}: {val}")

  lines.extend(
    [
      "",
      "[Imbalance & Sampling]",
      f"Class Weight:        {settings.get('class_weight', 'N/A')}",
      f"Resample Strategy:   {settings.get('resample', 'none')}",
      f"Augmentation:        {settings.get('augment', False)}",
      f"Balanced Sampling:   {settings.get('balanced_sampling', False)}",
    ]
  )

  lines.extend(
    [
      "",
      "[Data Splits & Samples]",
      f"Train Path:          {settings.get('train_path', 'N/A')}",
      f"Val Path:            {settings.get('val_path', 'N/A')}",
      f"Train Samples:       {settings.get('train_samples', 'N/A')}",
      f"Train Distribution:  {settings.get('train_distribution', {})}",
    ]
  )
  if "resampled_distribution" in settings:
    lines.extend(
      [
        f"Resampled Samples:   {settings.get('resampled_samples', 'N/A')}",
        f"Resampled Distrib:   {settings.get('resampled_distribution', {})}",
      ]
    )
  lines.extend(
    [
      f"Val Samples:         {settings.get('val_samples', 'N/A')}",
      f"Val Distribution:    {settings.get('val_distribution', {})}",
    ]
  )

  if "val_metrics" in settings:
    vm = settings["val_metrics"]
    lines.extend(
      [
        "",
        "[Validation Performance (DS1 Val)]",
        f"Val Accuracy:        {vm.get('accuracy', 0.0):.2f}%",
        f"Val Macro-F1:        {vm.get('macro_f1', 0.0):.4f}",
      ]
    )
    if "per_class_f1" in vm:
      lines.append(
        "Per-class F1 (N,S,V,F):      " + str([round(x, 4) for x in vm["per_class_f1"]])
      )
    if "per_class_recall" in vm:
      lines.append(
        "Per-class Recall (N,S,V,F):  "
        + str([round(x, 4) for x in vm["per_class_recall"]])
      )
    if "per_class_precision" in vm:
      lines.append(
        "Per-class Precision (N,S,V,F):"
        + str([round(x, 4) for x in vm["per_class_precision"]])
      )

  if settings.get("epoch_history"):
    lines.extend(
      [
        "",
        "[Epoch Training History]",
      ]
    )
    if settings.get("history_path"):
      lines.append(f"History CSV:         {settings['history_path']}")
    header = (
      f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc (%)':>13} | "
      f"{'Val Loss':>8} | {'Val Acc (%)':>11} | {'Val Macro-F1':>12}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for ep in settings["epoch_history"]:
      lines.append(
        f"{ep['epoch']:5d} | {ep['train_loss']:10.4f} | {ep['train_accuracy']:13.2f} | "
        f"{ep['val_loss']:8.4f} | {ep['val_accuracy']:11.2f} | {ep['val_macro_f1']:12.4f}"
      )

  lines.append("")
  lines.append("=" * 78)

  out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
  return out_path


def append_run_evaluation(
  run_dir: str | Path,
  results: dict[str, Any],
  filename: str = "settings.txt",
) -> None:
  """
  Append held-out test evaluation results to settings.txt.

  Parameters
  ----------
  run_dir : str | Path
      Run directory containing settings.txt.
  results : dict[str, Any]
      Output dictionary from evaluate_model.
  filename : str
      Target text filename (default: 'settings.txt').
  """
  run_p = Path(run_dir)
  out_path = run_p / filename
  now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

  lines = [
    "",
    "[Held-out Test Evaluation (DS2)]",
    f"Evaluated At:        {now_utc}",
    f"Test Accuracy:       {results['accuracy'] * 100:.2f}%",
    f"Test Macro-F1:       {results['macro_f1']:.4f}",
    f"Test Macro-Recall:   {results['macro_recall']:.4f}",
    f"Test Macro-Precision:{results['macro_precision']:.4f}",
    "",
    "Per-Class Performance Table:",
  ]

  if "per_class_table" in results:
    lines.append(results["per_class_table"].to_string(index=False))

  if "confusion_matrix" in results:
    lines.append("")
    lines.append("Confusion Matrix:")
    cm = results["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    lines.append(cm_df.to_string())

  lines.append("=" * 78)

  if out_path.exists():
    with open(out_path, "a", encoding="utf-8") as f:
      f.write("\n".join(lines) + "\n")
  else:
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
