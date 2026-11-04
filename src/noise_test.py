"""
noise_test.py
=============
Noise robustness evaluation pipeline for ECG heartbeat classification models.

Evaluates trained models (Baseline LogisticRegression, RandomForest, and the
Edge-AI MAX78002 1D CNN) under progressively increasing ECG noise levels:
    - Clean ECG (baseline test split, DS2)
    - 12 dB SNR (moderate noise)
    - 6 dB SNR (strong noise)
    - 0 dB SNR (severe noise)

Adheres to ANSI/AAMI EC57 and CRISP-DM Stage 5 (Evaluation, Section 5.4):
- Retains mathematical formulations from the colleagues' noise experiment.
- Evaluates the primary neural Edge model (1D CNN) and baselines on held-out DS2.
- Exports results to results/noise_results.csv, results/noise_results.png,
  and results/noise_per_class_metrics.csv matching course rubric Section 9.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pyhere import here
from sklearn.metrics import (
  accuracy_score,
  confusion_matrix,
  precision_recall_fscore_support,
)
from torch import nn

PROJECT_ROOT = here()
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

try:
  from src.constants import (
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODELS_DIR,
    DEFAULT_RANDOM_STATE,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TEST_PATH,
    NUM_CLASSES,
    WINDOW_SIZE,
    set_seed,
  )
except ImportError:
  from constants import (
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_MODELS_DIR,
    DEFAULT_RANDOM_STATE,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TEST_PATH,
    NUM_CLASSES,
    WINDOW_SIZE,
    set_seed,
  )

from src.dataset import ECGDataset
from src.model import (
  MODELS,
  BaseECGModel,
  ClassicalECGModel,
  NeuralECGModel,
  get_model,
)


def to_snake_case(name: str) -> str:
  """Convert CamelCase class name to snake_case."""
  s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
  return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s).lower()


# =====================================================================
# Noise Generation Utilities
# =====================================================================


def generate_gaussian_noise(
  signals: np.ndarray,
  snr_db: float,
  seed: int = DEFAULT_RANDOM_STATE,
  clip_range: tuple[float, float] = (-3.0, 3.0),
) -> np.ndarray:
  """
  Add calibrated Gaussian noise to normalized ECG heartbeats scaled to exact SNR.

  Formula matches colleagues' noise robustness experiment:
      P_target = P_signal / (10 ** (snr_db / 10))
      noise_scaled = noise_std * sqrt(P_target / P_noise)

  Parameters
  ----------
  signals : np.ndarray
      Input ECG signals of shape (N, window_size) or (N, 1, window_size).
  snr_db : float
      Target Signal-to-Noise Ratio in decibels (e.g. 12, 6, 0).
  seed : int
      Random seed for reproducible noise realizations.
  clip_range : tuple[float, float]
      Min and max clipping bounds representing Edge-AI sensor dynamic range.

  Returns
  -------
  np.ndarray
      Noisy ECG signals with same shape as input.
  """
  original_shape = signals.shape
  flat_signals = signals.reshape(signals.shape[0], -1).astype(np.float32)

  rng = np.random.default_rng(seed)
  standard_noise = rng.normal(size=flat_signals.shape).astype(np.float32)

  signal_power = np.mean(np.square(flat_signals), axis=1, keepdims=True)
  noise_power = np.mean(np.square(standard_noise), axis=1, keepdims=True)

  # Guard against zero-power division
  signal_power = np.where(signal_power <= 0, 1e-8, signal_power)
  noise_power = np.where(noise_power <= 0, 1e-8, noise_power)

  target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
  scaled_noise = standard_noise * np.sqrt(target_noise_power / noise_power)

  # Verify generated SNR
  measured_snr = 10.0 * np.log10(
    signal_power / np.mean(np.square(scaled_noise), axis=1, keepdims=True)
  )
  if not np.allclose(measured_snr, snr_db, atol=1e-3):
    avg_snr = float(np.mean(measured_snr))
    print(f"Warning: Mean SNR ({avg_snr:.2f} dB) slightly deviates from target {snr_db} dB")

  noisy = flat_signals + scaled_noise

  if clip_range is not None:
    c_min, c_max = clip_range
    noisy = np.clip(noisy, c_min, c_max)

  return noisy.reshape(original_shape).astype(np.float32)


def generate_physiological_noise(
  signals: np.ndarray,
  snr_db: float,
  fs: float = 360.0,
  seed: int = DEFAULT_RANDOM_STATE,
  clip_range: tuple[float, float] = (-3.0, 3.0),
) -> np.ndarray:
  """
  Generate composite physiological ECG noise (baseline wander + EMG tremor)
  calibrated to target SNR, simulating MIT-BIH Noise Stress Test conditions.
  """
  original_shape = signals.shape
  flat = signals.reshape(signals.shape[0], -1).astype(np.float32)
  n_samples = flat.shape[1]
  t = np.linspace(0, n_samples / fs, n_samples, endpoint=False, dtype=np.float32)

  rng = np.random.default_rng(seed)
  composite_noise = np.zeros_like(flat)

  for i in range(flat.shape[0]):
    # Low-frequency baseline wander (0.15 - 0.35 Hz respiratory modulation)
    bw_freq = rng.uniform(0.15, 0.35)
    bw_phase = rng.uniform(0, 2 * np.pi)
    bw = np.sin(2 * np.pi * bw_freq * t + bw_phase)

    # High-frequency EMG muscle tremor (stochastic jitter filtered)
    emg = rng.normal(0, 1, size=n_samples).astype(np.float32)

    # Combined physiological artifact
    combined = 0.6 * bw + 0.4 * emg
    composite_noise[i] = combined

  signal_power = np.mean(np.square(flat), axis=1, keepdims=True)
  noise_power = np.mean(np.square(composite_noise), axis=1, keepdims=True)

  signal_power = np.where(signal_power <= 0, 1e-8, signal_power)
  noise_power = np.where(noise_power <= 0, 1e-8, noise_power)

  target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
  scaled_noise = composite_noise * np.sqrt(target_noise_power / noise_power)

  noisy = flat + scaled_noise
  if clip_range is not None:
    noisy = np.clip(noisy, clip_range[0], clip_range[1])

  return noisy.reshape(original_shape).astype(np.float32)


def build_noisy_test_conditions(
  signals: np.ndarray,
  snr_levels_db: tuple[float, ...] = (12.0, 6.0, 0.0),
  noise_type: str = "gaussian",
  seed: int = DEFAULT_RANDOM_STATE,
) -> dict[str, np.ndarray]:
  """
  Build dictionary of paired test sets: Clean, and each requested SNR level.
  """
  conditions = {"Clean": signals.copy()}
  for snr in snr_levels_db:
    key = f"{int(snr) if snr.is_integer() else snr} dB"
    if noise_type.lower() == "physiological":
      conditions[key] = generate_physiological_noise(signals, snr_db=snr, seed=seed)
    else:
      conditions[key] = generate_gaussian_noise(signals, snr_db=snr, seed=seed)

  return conditions


# =====================================================================
# Model Loading & Checkpoint Resolution
# =====================================================================


def resolve_model_checkpoint(
  model_name: str,
  checkpoint_path: str | Path | None = None,
  results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> Path:
  """
  Find checkpoint file for the specified model in runs, results, or root.
  """
  if checkpoint_path is not None:
    p = Path(checkpoint_path)
    if p.exists():
      return p
    if here(checkpoint_path).exists():
      return here(checkpoint_path)

  slug = to_snake_case(model_name)
  res_p = Path(results_dir) if Path(results_dir).is_absolute() else here(results_dir)

  # Check root fallback (e.g. best_cnn_model.pth)
  if model_name.upper() == "CNN":
    root_pth = here("best_cnn_model.pth")
    if root_pth.exists():
      return root_pth

  # Check results directory
  candidates = [
    res_p / f"{slug}.pt",
    res_p / f"{slug}.pth",
    res_p / f"{slug}.joblib",
    here("results", f"{slug}.pt"),
    here("results", f"{slug}.pth"),
    here("results", f"{slug}.joblib"),
  ]

  # Check latest run directory
  runs_dir = here("results", "runs")
  if runs_dir.exists():
    run_folders = sorted(
      [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run")],
      key=lambda x: x.stat().st_mtime,
      reverse=True,
    )
    for rf in run_folders:
      for f in rf.glob("*.*"):
        if f.suffix in (".pt", ".pth", ".joblib") and (slug in f.stem.lower() or "best" in f.stem.lower()):
          candidates.insert(0, f)

  for cand in candidates:
    if cand.exists():
      return cand

  raise FileNotFoundError(
    f"Could not locate checkpoint for model '{model_name}'. "
    f"Checked paths: {[str(c) for c in candidates]}"
  )


def load_model_for_testing(
  model_name: str,
  checkpoint_path: Path,
  input_dim: int = WINDOW_SIZE,
  num_classes: int = NUM_CLASSES,
  seed: int = DEFAULT_RANDOM_STATE,
) -> BaseECGModel:
  """
  Instantiate model and load weights.
  """
  model = get_model(
    model_name,
    input_dim=input_dim,
    num_classes=num_classes,
    random_state=seed,
  )

  if isinstance(model, ClassicalECGModel):
    loaded = joblib.load(checkpoint_path)
    if hasattr(loaded, "predict"):
      return loaded
    model.classifier = loaded
    return model

  if isinstance(model, NeuralECGModel):
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
      model.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict):
      model.load_state_dict(ckpt)
    model.eval()
    return model

  return model


# =====================================================================
# Metrics & Batch Inference
# =====================================================================


def evaluate_predictions(
  y_true: np.ndarray,
  y_pred: np.ndarray,
  class_names: list[str] = CLASS_NAMES,
) -> dict[str, Any]:
  """
  Calculate full evaluation metrics matching calculate_metrics in colleagues notebook
  plus ANSI/AAMI EC57 specificity and per-class support.
  """
  labels = list(range(len(class_names)))
  acc = float(accuracy_score(y_true, y_pred))
  macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=labels, average="macro", zero_division=0
  )
  precisions, recalls, f1s, supports = precision_recall_fscore_support(
    y_true, y_pred, labels=labels, average=None, zero_division=0
  )
  cm = confusion_matrix(y_true, y_pred, labels=labels)

  specificities = []
  total_samples = len(y_true)
  for i in range(len(labels)):
    tp = cm[i, i]
    fn = cm[i, :].sum() - tp
    fp = cm[:, i].sum() - tp
    tn = total_samples - (tp + fn + fp)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    specificities.append(float(spec))

  return {
    "accuracy": acc,
    "macro_f1": float(macro_f1),
    "macro_recall": float(macro_rec),
    "macro_precision": float(macro_prec),
    "per_class_precision": [float(p) for p in precisions],
    "per_class_recall": [float(r) for r in recalls],
    "per_class_specificity": specificities,
    "per_class_f1": [float(f) for f in f1s],
    "support": [int(s) for s in supports],
    "confusion_matrix": cm,
  }


@torch.no_grad()
def predict_signals(
  model: BaseECGModel,
  signals: np.ndarray,
  batch_size: int = DEFAULT_BATCH_SIZE,
  device: torch.device | None = None,
) -> np.ndarray:
  """
  Run model predictions in batches.
  """
  if device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  if isinstance(model, nn.Module):
    model.to(device)
    model.eval()

    all_preds = []
    n = len(signals)
    for i in range(0, n, batch_size):
      batch = signals[i : i + batch_size]
      tensor_batch = torch.from_numpy(batch).float().to(device)
      logits = model.forward(tensor_batch)
      preds = torch.argmax(logits, dim=-1).cpu().numpy()
      all_preds.append(preds)
    return np.concatenate(all_preds)

  # Classical model (RandomForest, etc.)
  if hasattr(model, "predict"):
    flat = signals.reshape(signals.shape[0], -1)
    return model.predict(flat)

  raise TypeError(f"Unsupported model type: {type(model)}")


# =====================================================================
# Main Robustness Pipeline & Visualizations
# =====================================================================


def run_noise_robustness_experiment(
  models_to_test: list[str],
  test_path: str | Path = DEFAULT_TEST_PATH,
  snr_levels_db: tuple[float, ...] = (12.0, 6.0, 0.0),
  noise_type: str = "gaussian",
  batch_size: int = DEFAULT_BATCH_SIZE,
  output_dir: str | Path = DEFAULT_RESULTS_DIR,
  seed: int = DEFAULT_RANDOM_STATE,
  quick_test: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
  """
  Execute complete noise robustness experiment across models and SNR conditions.
  """
  set_seed(seed)
  out_p = Path(output_dir) if Path(output_dir).is_absolute() else here(output_dir)
  out_p.mkdir(parents=True, exist_ok=True)

  test_p = Path(test_path) if Path(test_path).is_absolute() else here(test_path)
  if not test_p.exists():
    raise FileNotFoundError(
      f"Held-out test split '{test_path}' not found. "
      "Run 'uv run python src/prepare_data.py' or 'task prepare-data' first."
    )

  print("=" * 80)
  print("ECG HEARTBEAT CLASSIFICATION — NOISE ROBUSTNESS BENCHMARK (CRISP-DM Stage 5.4)")
  print("=" * 80)
  print(f"Held-out Test Set : {test_p}")
  print(f"Noise Conditions  : Clean, {', '.join(f'{snr} dB' for snr in snr_levels_db)}")
  print(f"Noise Generator   : {noise_type.upper()}")
  print(f"Models to Test    : {', '.join(models_to_test)}")
  print(f"Random Seed       : {seed}")
  print("=" * 80)

  test_ds = ECGDataset.from_npz(test_p, augment=False, seed=seed)
  X_test = test_ds.X
  y_test = test_ds.y

  if quick_test:
    print("Running in --quick-test mode: evaluating on first 1000 test beats.")
    X_test = X_test[:1000]
    y_test = y_test[:1000]

  print(f"Total test beats  : {len(y_test):,}")
  print(f"Class distribution: {dict(zip(CLASS_NAMES, np.bincount(y_test, minlength=NUM_CLASSES), strict=False))}")
  print("-" * 80)

  # Generate paired noise conditions (shared realization across models for fair comparison)
  print("Generating calibrated SNR conditions...")
  test_conditions = build_noisy_test_conditions(
    X_test,
    snr_levels_db=snr_levels_db,
    noise_type=noise_type,
    seed=seed,
  )

  # Results structures
  summary_rows = []
  per_class_rows = []
  full_results: dict[str, dict[str, Any]] = {c: {} for c in test_conditions}

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  for model_name in models_to_test:
    print(f"\nEvaluating Model: {model_name}...")
    ckpt_path = resolve_model_checkpoint(model_name, results_dir=out_p)
    print(f"Restored weights from: {ckpt_path}")
    model = load_model_for_testing(model_name, ckpt_path, seed=seed)

    clean_metrics: dict[str, Any] | None = None

    for cond_name, noisy_X in test_conditions.items():
      preds = predict_signals(model, noisy_X, batch_size=batch_size, device=device)
      metrics = evaluate_predictions(y_test, preds)
      full_results[cond_name][model_name] = metrics

      if cond_name == "Clean":
        clean_metrics = metrics

      # Summary row (Table 5.4 in project instructions)
      summary_rows.append(
        {
          "Model": model_name,
          "Condition": cond_name,
          "Accuracy": metrics["accuracy"],
          "Macro_F1": metrics["macro_f1"],
          "Macro_Recall": metrics["macro_recall"],
          "Macro_Precision": metrics["macro_precision"],
          "N_F1": metrics["per_class_f1"][0],
          "S_F1": metrics["per_class_f1"][1],
          "V_F1": metrics["per_class_f1"][2],
          "F_F1": metrics["per_class_f1"][3],
        }
      )

      # Per-class row with deltas
      for cls_idx, cls_name in enumerate(CLASS_NAMES):
        recall = metrics["per_class_recall"][cls_idx]
        f1 = metrics["per_class_f1"][cls_idx]
        prec = metrics["per_class_precision"][cls_idx]
        spec = metrics["per_class_specificity"][cls_idx]

        clean_rec = clean_metrics["per_class_recall"][cls_idx] if clean_metrics else recall
        clean_f1 = clean_metrics["per_class_f1"][cls_idx] if clean_metrics else f1

        per_class_rows.append(
          {
            "Model": model_name,
            "Condition": cond_name,
            "Class": cls_name,
            "Precision": prec,
            "Recall_Sensitivity": recall,
            "Specificity": spec,
            "F1": f1,
            "Recall_Delta_From_Clean": recall - clean_rec,
            "F1_Delta_From_Clean": f1 - clean_f1,
            "Support": metrics["support"][cls_idx],
          }
        )

  summary_df = pd.DataFrame(summary_rows)
  per_class_df = pd.DataFrame(per_class_rows)

  # Export CSV artifacts required by Section 9
  csv_path = out_p / "noise_results.csv"
  summary_df.to_csv(csv_path, index=False)
  print(f"\n[Artifact Saved] Summary metrics -> {csv_path}")

  per_class_csv_path = out_p / "noise_per_class_metrics.csv"
  per_class_df.to_csv(per_class_csv_path, index=False)
  print(f"[Artifact Saved] Per-class metrics -> {per_class_csv_path}")

  # Print terminal summary matching Section 5.4 Table
  print("\n" + "=" * 80)
  print("SECTION 5.4 NOISE ROBUSTNESS TABLE (Clean vs 12 dB vs 6 dB vs 0 dB)")
  print("=" * 80)
  print(f"{'Model':<18} {'Condition':<12} {'Accuracy':>10} {'Macro F1':>10} {'Class V F1':>12} {'Class S F1':>12}")
  print("-" * 80)
  for _, r in summary_df.iterrows():
    print(
      f"{r['Model']:<18} {r['Condition']:<12} {r['Accuracy']:>10.4f} "
      f"{r['Macro_F1']:>10.4f} {r['V_F1']:>12.4f} {r['S_F1']:>12.4f}"
    )
  print("=" * 80)

  # Generate Visualizations (noise_results.png)
  fig_path = out_p / "noise_results.png"
  plot_noise_results(summary_df, full_results, test_conditions, models_to_test, fig_path)
  print(f"[Artifact Saved] Noise robustness visualization -> {fig_path}")

  return summary_df, per_class_df, full_results


def plot_noise_results(
  summary_df: pd.DataFrame,
  full_results: dict[str, dict[str, Any]],
  test_conditions: dict[str, np.ndarray],
  models: list[str],
  save_path: Path,
) -> None:
  """
  Render comprehensive 4-panel diagnostic figure matching colleagues' notebook
  and rubric Section 5.4:
    - Panel 1: Accuracy vs Noise Condition
    - Panel 2: Macro F1 vs Noise Condition
    - Panel 3: Per-Class F1 Degradation (N, S, V, F) across SNR
    - Panel 4: Test Confusion Matrix Grid across noise conditions
  """
  conditions = list(test_conditions.keys())
  primary_model = "CNN" if "CNN" in models else models[0]

  fig, axes = plt.subplots(2, 2, figsize=(16, 12))
  ax1 = axes[0, 0]
  ax2 = axes[0, 1]
  ax3 = axes[1, 0]
  ax4 = axes[1, 1]
  for m in models:
    m_sub = summary_df[summary_df["Model"] == m]
    ax1.plot(
      m_sub["Condition"],
      m_sub["Accuracy"] * 100.0,
      marker="o",
      linewidth=2.2,
      label=m,
    )
  ax1.set_title("Overall Accuracy vs. ECG Noise Condition", fontsize=13, fontweight="bold")
  ax1.set_xlabel("Noise Condition (SNR)", fontsize=11)
  ax1.set_ylabel("Accuracy (%)", fontsize=11)
  ax1.grid(True, linestyle="--", alpha=0.5)
  ax1.legend(loc="lower left", framealpha=0.9)

  # Panel 2: Macro F1 vs Noise Condition
  for m in models:
    m_sub = summary_df[summary_df["Model"] == m]
    ax2.plot(
      m_sub["Condition"],
      m_sub["Macro_F1"],
      marker="s",
      linewidth=2.2,
      label=m,
    )
  ax2.set_title("Macro F1-Score vs. ECG Noise Condition", fontsize=13, fontweight="bold")
  ax2.set_xlabel("Noise Condition (SNR)", fontsize=11)
  ax2.set_ylabel("Macro F1-Score", fontsize=11)
  ax2.grid(True, linestyle="--", alpha=0.5)
  ax2.legend(loc="lower left", framealpha=0.9)

  # Panel 3: Per-Class F1 Degradation for Primary Model
  class_keys = ["N_F1", "S_F1", "V_F1", "F_F1"]
  class_labels = ["N (Normal)", "S (Supraventricular)", "V (Ventricular)", "F (Fusion)"]
  markers = ["o", "^", "s", "D"]
  colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

  pri_sub = summary_df[summary_df["Model"] == primary_model]
  for k, lab, mk, col in zip(class_keys, class_labels, markers, colors, strict=False):
    ax3.plot(
      pri_sub["Condition"],
      pri_sub[k],
      marker=mk,
      color=col,
      linewidth=2.0,
      label=lab,
    )
  ax3.set_title(f"{primary_model}: Per-Class F1 Score vs. Noise", fontsize=13, fontweight="bold")
  ax3.set_xlabel("Noise Condition (SNR)", fontsize=11)
  ax3.set_ylabel("F1 Score", fontsize=11)
  ax3.set_ylim(-0.05, 1.05)
  ax3.grid(True, linestyle="--", alpha=0.5)
  ax3.legend(loc="best", framealpha=0.9)

  # Panel 4: Confusion Matrix at Strongest Noise (0 dB)
  cm_0db = full_results["0 dB"][primary_model]["confusion_matrix"]
  sns.heatmap(
    cm_0db,
    annot=True,
    fmt="d",
    cmap="Reds",
    xticklabels=["N", "S", "V", "F"],
    yticklabels=["N", "S", "V", "F"],
    ax=ax4,
    cbar=False,
  )
  ax4.set_title(f"{primary_model} Confusion Matrix at 0 dB (Severe Noise)", fontsize=13, fontweight="bold")
  ax4.set_xlabel("Predicted Class", fontsize=11)
  ax4.set_ylabel("True Class", fontsize=11)

  plt.tight_layout()
  plt.savefig(str(save_path), dpi=300)
  plt.close(fig)


# =====================================================================
# CLI Entry Point
# =====================================================================


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Evaluate ECG classification models under calibrated noise conditions (Clean, 12dB, 6dB, 0dB)."
  )
  parser.add_argument(
    "--model",
    "-m",
    default="CNN",
    help="Model to evaluate: 'CNN', 'LogisticRegression', 'RandomForest', or 'all' (default: CNN).",
  )
  parser.add_argument(
    "--test-path",
    default=str(DEFAULT_TEST_PATH),
    help="Path to held-out test split .npz (default: dataset/processed/test.npz).",
  )
  parser.add_argument(
    "--snr",
    nargs="+",
    type=float,
    default=[12.0, 6.0, 0.0],
    help="SNR levels in dB to test (default: 12.0 6.0 0.0).",
  )
  parser.add_argument(
    "--noise-type",
    choices=["gaussian", "physiological"],
    default="gaussian",
    help="Noise generation type: 'gaussian' (colleagues notebook match) or 'physiological' (default: gaussian).",
  )
  parser.add_argument(
    "--batch-size",
    type=int,
    default=DEFAULT_BATCH_SIZE,
    help=f"Inference batch size (default: {DEFAULT_BATCH_SIZE}).",
  )
  parser.add_argument(
    "--output-dir",
    default=str(DEFAULT_RESULTS_DIR),
    help=f"Directory to save noise results artifacts (default: {DEFAULT_RESULTS_DIR}).",
  )
  parser.add_argument(
    "--seed",
    type=int,
    default=DEFAULT_RANDOM_STATE,
    help=f"Random seed for reproducibility (default: {DEFAULT_RANDOM_STATE}).",
  )
  parser.add_argument(
    "--quick-test",
    action="store_true",
    help="Run on first 1000 test beats for quick verification.",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()

  if args.model.lower() == "all":
    models_to_test = ["LogisticRegression", "RandomForest", "CNN"]
  else:
    models_to_test = [args.model]

  run_noise_robustness_experiment(
    models_to_test=models_to_test,
    test_path=args.test_path,
    snr_levels_db=tuple(args.snr),
    noise_type=args.noise_type,
    batch_size=args.batch_size,
    output_dir=args.output_dir,
    seed=args.seed,
    quick_test=args.quick_test,
  )


if __name__ == "__main__":
  main()
