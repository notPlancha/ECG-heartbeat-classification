"""
evaluate.py
===========
Evaluation pipeline for trained ECG heartbeat classification models
on held-out test data (DS2) using scikit-learn metrics.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pyhere import here
from sklearn.metrics import (
  accuracy_score,
  classification_report,
  confusion_matrix,
  precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = here()
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

try:
  from src.constants import (
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_FIGURES_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_RANDOM_STATE,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TABLES_DIR,
    DEFAULT_TEST_PATH,
    NUM_CLASSES,
    WINDOW_SIZE,
    set_seed,
  )
except ImportError:
  from constants import (
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_FIGURES_DIR,
    DEFAULT_MODELS_DIR,
    DEFAULT_RANDOM_STATE,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TABLES_DIR,
    DEFAULT_TEST_PATH,
    NUM_CLASSES,
    WINDOW_SIZE,
    set_seed,
  )

from src.dataset import ECGDataset, get_data_loader
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


def compute_evaluation_metrics(
  y_true: np.ndarray,
  y_pred: np.ndarray,
  labels: list[int] | None = None,
  class_names: list[str] | None = None,
) -> dict[str, object]:
  """
  Compute overall and per-class performance metrics using scikit-learn.
  """
  if labels is None:
    labels = list(range(NUM_CLASSES))
  if class_names is None:
    class_names = CLASS_NAMES

  acc = accuracy_score(y_true, y_pred)
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
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    tn = total_samples - (tp + fp + fn)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    specificities.append(spec)

  report_text = classification_report(
    y_true, y_pred, labels=labels, target_names=class_names, digits=4, zero_division=0
  )

  per_class_df = pd.DataFrame(
    {
      "Class": class_names,
      "Precision": precisions,
      "Recall_Sensitivity": recalls,
      "Specificity": specificities,
      "F1": f1s,
      "Support": supports,
    }
  )

  return {
    "accuracy": acc,
    "macro_precision": macro_prec,
    "macro_recall": macro_rec,
    "macro_f1": macro_f1,
    "confusion_matrix": cm,
    "per_class_table": per_class_df,
    "classification_report": report_text,
  }


def plot_confusion_matrix(
  cm: np.ndarray,
  class_names: list[str],
  save_path: str | Path,
  title: str = "Confusion Matrix",
) -> None:
  """
  Save confusion matrix heatmap as an image.
  """
  save_p = Path(save_path)
  save_p.parent.mkdir(parents=True, exist_ok=True)

  fig, ax = plt.subplots(figsize=(6, 5))
  sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    ax=ax,
  )
  ax.set_xlabel("Predicted label")
  ax.set_ylabel("True label")
  ax.set_title(title)
  plt.tight_layout()
  plt.savefig(str(save_p), dpi=300)
  plt.close(fig)


@torch.no_grad()
def run_inference(
  model: BaseECGModel,
  loader: DataLoader,
  device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
  """
  Run model inference across a data loader and return predictions and ground truth.
  """
  if isinstance(model, nn.Module):
    model.eval()
  all_preds = []
  all_targets = []

  for beats, labels in loader:
    if isinstance(model, nn.Module):
      beats = beats.to(device)
    preds = model.predict(beats)

    if isinstance(preds, torch.Tensor):
      preds = preds.cpu().numpy()
    all_preds.append(preds)
    all_targets.append(labels.numpy())

  y_pred = np.concatenate(all_preds)
  y_true = np.concatenate(all_targets)
  return y_true, y_pred


def load_checkpoint(model: NeuralECGModel, checkpoint_path: str | Path) -> None:
  """
  Load weights into neural model from a checkpoint dictionary or state dict.
  """
  cp = Path(checkpoint_path)
  if not cp.exists():
    raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found.")

  ckpt = torch.load(str(cp), map_location="cpu", weights_only=False)
  match ckpt:
    case {"model_state_dict": state_dict}:
      model.load_state_dict(state_dict)
    case dict():
      model.load_state_dict(ckpt)
    case _:
      raise ValueError(f"Unrecognized checkpoint format at '{checkpoint_path}'")


def evaluate_model(
  model_name: str = "LogisticRegression",
  checkpoint_path: str | Path | None = None,
  data_path: str | Path = DEFAULT_TEST_PATH,
  batch_size: int = DEFAULT_BATCH_SIZE,
  results_dir: str | Path = DEFAULT_RESULTS_DIR,
  models_dir: str | Path = DEFAULT_MODELS_DIR,
  figures_dir: str | Path = DEFAULT_FIGURES_DIR,
  tables_dir: str | Path = DEFAULT_TABLES_DIR,
  output_dir: str | Path | None = None,
  seed: int = DEFAULT_RANDOM_STATE,
) -> dict[str, object]:
  """
  Full evaluation flow: load data, restore model, compute metrics, and export artifacts.
  """
  set_seed(seed)

  if output_dir is not None:
    results_dir = output_dir

  data_p = Path(data_path) if Path(data_path).exists() else here(data_path)
  if not data_p.exists():
    raise FileNotFoundError(f"Test data file not found: {data_path}")

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Device: {device}")
  print(f"Loading test split from {data_p}...")

  test_ds = ECGDataset.from_npz(data_p, augment=False, seed=seed)
  test_loader = get_data_loader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    balanced_sampling=False,
    seed=seed,
  )

  print(f"Loaded {len(test_ds)} test heartbeats.")
  print(f"Class distribution: {test_ds.get_class_counts()}")

  model = get_model(
    model_name,
    input_dim=WINDOW_SIZE,
    num_classes=NUM_CLASSES,
    random_state=seed,
  )
  slug = to_snake_case(type(model).__name__)

  if checkpoint_path is None:
    match model:
      case ClassicalECGModel():
        cand1 = Path(models_dir) / f"{slug}.joblib"
        cand_here = here("results", "models", f"{slug}.joblib")
        cand2 = Path(results_dir) / f"{slug}.joblib"
        if cand1.exists():
          checkpoint_path = str(cand1)
        elif cand_here.exists():
          checkpoint_path = str(cand_here)
        elif cand2.exists():
          checkpoint_path = str(cand2)
        else:
          checkpoint_path = str(cand_here)
      case NeuralECGModel():
        cand1 = Path(models_dir) / f"{slug}.pt"
        cand_here = here("results", "models", f"{slug}.pt")
        cand2 = Path(results_dir) / f"{slug}.pt"
        if cand1.exists():
          checkpoint_path = str(cand1)
        elif cand_here.exists():
          checkpoint_path = str(cand_here)
        elif cand2.exists():
          checkpoint_path = str(cand2)
        else:
          checkpoint_path = str(cand_here)
      case _:
        raise TypeError(f"Unsupported model type: {type(model)}")
  else:
    cp = Path(checkpoint_path)
    if not cp.exists() and here(checkpoint_path).exists():
      checkpoint_path = str(here(checkpoint_path))
    else:
      checkpoint_path = str(cp)

  if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(
      f"Checkpoint file not found: {checkpoint_path}. Train the model first."
    )

  match model:
    case ClassicalECGModel():
      print(f"Loading classical model from {checkpoint_path}...")
      model = joblib.load(checkpoint_path)
    case NeuralECGModel():
      print(f"Loading neural model from {checkpoint_path}...")
      load_checkpoint(model, checkpoint_path)
      model = model.to(device)
    case _:
      raise TypeError(f"Unsupported model type: {type(model)}")

  num_params = model.count_parameters()
  print(f"Loaded model '{type(model).__name__}' ({num_params} parameters / nodes).")
  print("Running test set inference...")

  y_true, y_pred = run_inference(model, test_loader, device)
  results = compute_evaluation_metrics(y_true, y_pred)

  print("\n" + "=" * 65)
  print("5.1 Clean Test Results")
  print("=" * 65)
  print(f"Accuracy:         {results['accuracy'] * 100:.2f}%")
  print(f"Macro F1:         {results['macro_f1']:.4f}")
  print(f"Macro Recall:     {results['macro_recall']:.4f}")
  print(f"Macro Precision:  {results['macro_precision']:.4f}")
  print(f"Model parameters: {num_params}")

  print("\n" + "=" * 65)
  print("5.2 Per-Class Performance")
  print("=" * 65)
  print(results["per_class_table"].to_string(index=False))

  print("\nConfusion Matrix:")
  cm = results["confusion_matrix"]
  cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
  print(cm_df)

  res_p = Path(results_dir) if Path(results_dir).is_absolute() else here(results_dir)
  fig_p = Path(figures_dir) if Path(figures_dir).is_absolute() else here(figures_dir)
  tbl_p = Path(tables_dir) if Path(tables_dir).is_absolute() else here(tables_dir)

  res_p.mkdir(parents=True, exist_ok=True)
  fig_p.mkdir(parents=True, exist_ok=True)
  tbl_p.mkdir(parents=True, exist_ok=True)

  model_tag = type(model).__name__

  row_data = {
    "Model": model_tag,
    "Parameters": num_params,
    "Accuracy": results["accuracy"],
    "Macro_F1": results["macro_f1"],
    "Macro_Recall": results["macro_recall"],
    "Macro_Precision": results["macro_precision"],
  }

  metrics_csv = res_p / "metrics.csv"
  if metrics_csv.exists():
    try:
      existing_df = pd.read_csv(metrics_csv)
      existing_df = existing_df[existing_df["Model"] != model_tag]
      summary_df = pd.concat([existing_df, pd.DataFrame([row_data])], ignore_index=True)
    except pd.errors.EmptyDataError, ValueError, KeyError:
      summary_df = pd.DataFrame([row_data])
  else:
    summary_df = pd.DataFrame([row_data])

  summary_df.to_csv(metrics_csv, index=False)
  print(f"\nUpdated overall metrics table at: {metrics_csv}")

  # Save detailed per-class metrics in tables directory and root results directory
  per_class_table_csv = tbl_p / f"{slug}_per_class_metrics.csv"
  results["per_class_table"].to_csv(per_class_table_csv, index=False)
  print(f"Saved per-class metrics to: {per_class_table_csv}")

  per_class_root_csv = res_p / "per_class_metrics.csv"
  results["per_class_table"].to_csv(per_class_root_csv, index=False)

  # Save confusion matrix CSV in tables directory and root results directory
  cm_table_csv = tbl_p / f"{slug}_confusion_matrix.csv"
  cm_df.to_csv(cm_table_csv)
  print(f"Saved confusion matrix CSV to: {cm_table_csv}")

  cm_root_csv = res_p / "confusion_matrix.csv"
  cm_df.to_csv(cm_root_csv)

  # Save confusion matrix figure in figures directory and root results directory
  cm_fig_png = fig_p / f"{slug}_confusion_matrix.png"
  plot_confusion_matrix(
    cm, CLASS_NAMES, cm_fig_png, title=f"Confusion Matrix - {model_tag}"
  )
  print(f"Saved confusion matrix figure to: {cm_fig_png}")

  cm_root_png = res_p / "confusion_matrix.png"
  plot_confusion_matrix(
    cm, CLASS_NAMES, cm_root_png, title=f"Confusion Matrix - {model_tag}"
  )

  return results


def main() -> None:
  model_choices = [cls.__name__ for cls in MODELS]
  parser = argparse.ArgumentParser(
    description="Evaluate trained ECG classification models using scikit-learn metrics."
  )
  parser.add_argument(
    "--model",
    type=str,
    default=model_choices[0],
    help=f"Model architecture. Choices: {model_choices}.",
  )
  parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to trained model checkpoint (auto-resolved if omitted).",
  )
  parser.add_argument(
    "--data",
    type=str,
    default=DEFAULT_TEST_PATH,
    help=f"Path to held-out test dataset (default: {DEFAULT_TEST_PATH}).",
  )
  parser.add_argument(
    "--batch-size",
    type=int,
    default=DEFAULT_BATCH_SIZE,
    help=f"Batch size for inference (default: {DEFAULT_BATCH_SIZE}).",
  )
  parser.add_argument(
    "--results-dir",
    type=str,
    default=DEFAULT_RESULTS_DIR,
    help=f"Root directory where metrics.csv is saved (default: {DEFAULT_RESULTS_DIR}).",
  )
  parser.add_argument(
    "--models-dir",
    type=str,
    default=DEFAULT_MODELS_DIR,
    help=f"Directory containing model checkpoints (default: {DEFAULT_MODELS_DIR}).",
  )
  parser.add_argument(
    "--figures-dir",
    type=str,
    default=DEFAULT_FIGURES_DIR,
    help=f"Directory where figure plots are saved (default: {DEFAULT_FIGURES_DIR}).",
  )
  parser.add_argument(
    "--tables-dir",
    type=str,
    default=DEFAULT_TABLES_DIR,
    help=f"Directory where table CSVs are saved (default: {DEFAULT_TABLES_DIR}).",
  )
  parser.add_argument(
    "--seed",
    type=int,
    default=DEFAULT_RANDOM_STATE,
    help=f"Random seed for evaluation (default: {DEFAULT_RANDOM_STATE}).",
  )
  args = parser.parse_args()

  evaluate_model(
    model_name=args.model,
    checkpoint_path=args.checkpoint,
    data_path=args.data,
    batch_size=args.batch_size,
    results_dir=args.results_dir,
    models_dir=args.models_dir,
    figures_dir=args.figures_dir,
    tables_dir=args.tables_dir,
    seed=args.seed,
  )


if __name__ == "__main__":
  main()
