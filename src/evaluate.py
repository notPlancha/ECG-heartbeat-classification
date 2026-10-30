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
from sklearn.metrics import (
  accuracy_score,
  classification_report,
  confusion_matrix,
  precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import CLASS_NAMES, NUM_CLASSES, WINDOW_SIZE
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
  save_path: str,
  title: str = "Confusion Matrix",
) -> None:
  """
  Save confusion matrix heatmap as an image.
  """
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
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  plt.savefig(save_path, dpi=300)
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


def load_checkpoint(model: NeuralECGModel, checkpoint_path: str) -> None:
  """
  Load weights into neural model from a checkpoint dictionary or state dict.
  """
  if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found.")

  ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    model.load_state_dict(ckpt["model_state_dict"])
  elif isinstance(ckpt, dict):
    model.load_state_dict(ckpt)
  else:
    raise ValueError(f"Unrecognized checkpoint format at '{checkpoint_path}'")


def evaluate_model(
  model_name: str = "LogisticRegression",
  checkpoint_path: str | None = None,
  data_path: str = "dataset/processed/test.npz",
  batch_size: int = 64,
  output_dir: str = "results",
) -> dict[str, object]:
  """
  Full evaluation flow: load data, restore model, compute metrics, and export artifacts.
  """
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Device: {device}")
  print(f"Loading test split from {data_path}...")

  test_ds = ECGDataset.from_npz(data_path, augment=False)
  test_loader = get_data_loader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    balanced_sampling=False,
  )

  print(f"Loaded {len(test_ds)} test heartbeats.")
  print(f"Class distribution: {test_ds.get_class_counts()}")

  model = get_model(model_name, input_dim=WINDOW_SIZE, num_classes=NUM_CLASSES)

  slug = to_snake_case(type(model).__name__)
  if checkpoint_path is None:
    if isinstance(model, ClassicalECGModel):
      checkpoint_path = f"results/{slug}.joblib"
    else:
      checkpoint_path = f"results/{slug}.pt"

  if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

  if isinstance(model, ClassicalECGModel):
    print(f"Loading classical model from {checkpoint_path}...")
    model = joblib.load(checkpoint_path)
  elif isinstance(model, NeuralECGModel):
    print(f"Loading neural model from {checkpoint_path}...")
    load_checkpoint(model, checkpoint_path)
    model = model.to(device)

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

  os.makedirs(output_dir, exist_ok=True)
  model_tag = type(model).__name__
  slug = to_snake_case(model_tag)

  row_data = {
    "Model": model_tag,
    "Parameters": num_params,
    "Accuracy": results["accuracy"],
    "Macro_F1": results["macro_f1"],
    "Macro_Recall": results["macro_recall"],
    "Macro_Precision": results["macro_precision"],
  }

  metrics_csv = os.path.join(output_dir, "metrics.csv")
  if os.path.exists(metrics_csv):
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

  per_class_csv = os.path.join(output_dir, f"{slug}_per_class_metrics.csv")
  results["per_class_table"].to_csv(per_class_csv, index=False)
  print(f"Saved per-class metrics to: {per_class_csv}")

  cm_csv = os.path.join(output_dir, f"{slug}_confusion_matrix.csv")
  cm_df.to_csv(cm_csv)
  print(f"Saved confusion matrix CSV to: {cm_csv}")

  cm_png = os.path.join(output_dir, f"{slug}_confusion_matrix.png")
  plot_confusion_matrix(
    cm, CLASS_NAMES, cm_png, title=f"Confusion Matrix - {model_tag}"
  )
  print(f"Saved confusion matrix figure to: {cm_png}")

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
    help="Path to trained model checkpoint.",
  )
  parser.add_argument(
    "--data",
    type=str,
    default="dataset/processed/test.npz",
    help="Path to held-out test dataset (default: dataset/processed/test.npz).",
  )
  parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    help="Batch size for inference (default: 64).",
  )
  parser.add_argument(
    "--output-dir",
    type=str,
    default="results",
    help="Directory where metrics and figures are saved (default: results).",
  )
  args = parser.parse_args()

  evaluate_model(
    model_name=args.model,
    checkpoint_path=args.checkpoint,
    data_path=args.data,
    batch_size=args.batch_size,
    output_dir=args.output_dir,
  )


if __name__ == "__main__":
  main()
