"""
train.py
========
Training pipeline for ECG heartbeat classification models using
preprocessed .npz splits.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import CLASS_NAMES, NUM_CLASSES, WINDOW_SIZE
from src.dataset import ECGDataset, get_data_loader
from src.model import MODELS, get_model


def compute_metrics(
  y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int = NUM_CLASSES
) -> dict[str, float | list[float]]:
  """
  Compute accuracy, per-class F1, and macro F1 using scikit-learn.
  """
  y_true_np = y_true.view(-1).cpu().numpy()
  y_pred_np = y_pred.view(-1).cpu().numpy()
  labels = list(range(num_classes))

  acc = accuracy_score(y_true_np, y_pred_np) * 100.0
  _, _, macro_f1, _ = precision_recall_fscore_support(
    y_true_np, y_pred_np, labels=labels, average="macro", zero_division=0
  )
  precisions, recalls, f1_scores, _ = precision_recall_fscore_support(
    y_true_np, y_pred_np, labels=labels, average=None, zero_division=0
  )

  return {
    "accuracy": acc,
    "macro_f1": float(macro_f1),
    "per_class_f1": [float(x) for x in f1_scores],
    "per_class_recall": [float(x) for x in recalls],
    "per_class_precision": [float(x) for x in precisions],
  }


def resolve_class_weights(
  dataset: ECGDataset, strategy: str | None
) -> tuple[torch.Tensor | None, np.ndarray | None, Any]:
  """
  Resolve class weighting strategy into PyTorch tensor, numpy array,
  and model-specific parameter (such as for scikit-learn).
  """
  if not strategy or strategy.lower() in ("none", "uniform"):
    return None, None, None

  strategy_clean = strategy.strip()
  if "," in strategy_clean:
    raw = [float(x.strip()) for x in strategy_clean.split(",")]
    if len(raw) != NUM_CLASSES:
      raise ValueError(
        f"Expected {NUM_CLASSES} class weights, got {len(raw)} in '{strategy}'"
      )
    weights_np = np.array(raw, dtype=np.float32)
    weights_torch = torch.from_numpy(weights_np).float()
    rf_param = {i: float(w) for i, w in enumerate(weights_np)}
    return weights_torch, weights_np, rf_param

  mode = strategy_clean.lower()
  if mode in ("balanced", "inverse"):
    weights_np = dataset.compute_class_weights("inverse")
    weights_torch = torch.from_numpy(weights_np).float()
    return weights_torch, weights_np, "balanced"
  elif mode == "sqrt_inverse":
    weights_np = dataset.compute_class_weights("sqrt_inverse")
    weights_torch = torch.from_numpy(weights_np).float()
    rf_param = {i: float(w) for i, w in enumerate(weights_np)}
    return weights_torch, weights_np, rf_param
  elif mode == "effective":
    weights_np = dataset.compute_class_weights("effective")
    weights_torch = torch.from_numpy(weights_np).float()
    rf_param = {i: float(w) for i, w in enumerate(weights_np)}
    return weights_torch, weights_np, rf_param
  elif mode == "balanced_subsample":
    weights_np = dataset.compute_class_weights("inverse")
    weights_torch = torch.from_numpy(weights_np).float()
    return weights_torch, weights_np, "balanced_subsample"
  else:
    valid = ["balanced", "inverse", "sqrt_inverse", "effective", "none"]
    raise ValueError(
      f"Unknown class weight strategy '{strategy}'. Choices: {valid} "
      "or comma-separated values (e.g. '1.0,10.0,5.0,20.0')."
    )


def load_data(
  train_path: str = "dataset/processed/train.npz",
  val_path: str = "dataset/processed/val.npz",
  batch_size: int = 64,
  augment: bool = False,
  balanced_sampling: bool = False,
  quick_test: bool = False,
  class_weight: str | None = "balanced",
) -> tuple[DataLoader, DataLoader, torch.Tensor | None]:
  """
  Load preprocessed .npz datasets and return train/val loaders with class weights.
  """
  if not os.path.exists(train_path):
    raise FileNotFoundError(
      f"Training file '{train_path}' not found. "
      "Run 'uv run python src/prepare_data.py' first."
    )
  if not os.path.exists(val_path):
    raise FileNotFoundError(
      f"Validation file '{val_path}' not found. "
      "Run 'uv run python src/prepare_data.py' first."
    )

  train_ds = ECGDataset.from_npz(train_path, augment=augment)
  val_ds = ECGDataset.from_npz(val_path, augment=False)

  if quick_test:
    train_ds.X = train_ds.X[:1000]
    train_ds.y = train_ds.y[:1000]
    val_ds.X = val_ds.X[:500]
    val_ds.y = val_ds.y[:500]

  class_weights_torch, _, _ = resolve_class_weights(train_ds, class_weight)

  train_loader = get_data_loader(
    train_ds,
    batch_size=batch_size,
    shuffle=not balanced_sampling,
    balanced_sampling=balanced_sampling,
  )
  val_loader = get_data_loader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    balanced_sampling=False,
  )

  print(f"Loaded {len(train_ds)} train beats and {len(val_ds)} validation beats.")
  print(f"Train distribution: {train_ds.get_class_counts()}")
  print(f"Val distribution:   {val_ds.get_class_counts()}")

  return train_loader, val_loader, class_weights_torch


def train_one_epoch(
  model: nn.Module,
  loader: DataLoader,
  criterion: nn.Module,
  optimizer: torch.optim.Optimizer,
  device: torch.device,
) -> tuple[float, float]:
  """
  Run one training epoch and report average loss and accuracy.
  """
  model.train()
  total_loss = 0.0
  correct = 0
  total_samples = 0

  for beats, labels in loader:
    beats = beats.to(device)
    labels = labels.to(device)

    optimizer.zero_grad()
    logits = model(beats)
    loss = criterion(logits, labels)

    loss.backward()
    optimizer.step()

    batch_size = labels.size(0)
    total_loss += loss.item() * batch_size
    preds = logits.argmax(dim=1)
    correct += (preds == labels).sum().item()
    total_samples += batch_size

  avg_loss = total_loss / max(total_samples, 1)
  accuracy = 100.0 * correct / max(total_samples, 1)
  return avg_loss, accuracy


@torch.no_grad()
def evaluate(
  model: nn.Module,
  loader: DataLoader,
  criterion: nn.Module,
  device: torch.device,
  num_classes: int = NUM_CLASSES,
) -> tuple[float, dict[str, float | list[float]]]:
  """
  Evaluate model on a dataset and return loss and classification metrics.
  """
  model.eval()
  total_loss = 0.0
  total_samples = 0
  all_preds = []
  all_targets = []

  for beats, labels in loader:
    beats = beats.to(device)
    labels = labels.to(device)

    logits = model(beats)
    loss = criterion(logits, labels)

    batch_size = labels.size(0)
    total_loss += loss.item() * batch_size
    preds = logits.argmax(dim=1)

    all_preds.append(preds.cpu())
    all_targets.append(labels.cpu())
    total_samples += batch_size

  avg_loss = total_loss / max(total_samples, 1)
  y_true = torch.cat(all_targets)
  y_pred = torch.cat(all_preds)
  metrics = compute_metrics(y_true, y_pred, num_classes=num_classes)

  return avg_loss, metrics


def train_random_forest(
  train_path: str = "dataset/processed/train.npz",
  val_path: str = "dataset/processed/val.npz",
  n_estimators: int = 100,
  max_depth: int | None = 20,
  min_samples_leaf: int = 2,
  class_weight: str | None = "balanced",
  random_state: int = 42,
  quick_test: bool = False,
  save_path: str = "results/random_forest.joblib",
) -> None:
  """
  Train a Random Forest classifier using preprocessed .npz splits.
  """
  if not os.path.exists(train_path):
    raise FileNotFoundError(f"Training file '{train_path}' not found.")
  if not os.path.exists(val_path):
    raise FileNotFoundError(f"Validation file '{val_path}' not found.")

  print(f"Loading training split from {train_path}...")
  train_ds = ECGDataset.from_npz(train_path, augment=False)
  val_ds = ECGDataset.from_npz(val_path, augment=False)

  X_train = train_ds.X
  y_train = train_ds.y
  X_val = val_ds.X
  y_val = val_ds.y

  if quick_test:
    X_train = X_train[:1000]
    y_train = y_train[:1000]
    X_val = X_val[:500]
    y_val = y_val[:500]

  print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)}")
  print(f"Train class distribution: {train_ds.get_class_counts()}")
  print(f"Val class distribution:   {val_ds.get_class_counts()}")

  _, weights_np, rf_param = resolve_class_weights(train_ds, class_weight)

  model = get_model(
    "RandomForest",
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_leaf=min_samples_leaf,
    class_weight=rf_param,
    random_state=random_state,
  )

  weight_desc = (
    rf_param
    if isinstance(rf_param, str)
    else (
      [round(float(w), 4) for w in weights_np] if weights_np is not None else "None"
    )
  )
  print(
    f"Training Random Forest (n_estimators={n_estimators}, max_depth={max_depth}, "
    f"class_weight={weight_desc})..."
  )
  model.fit(X_train, y_train)

  val_preds = model.predict(X_val)
  val_metrics = compute_metrics(torch.from_numpy(y_val), torch.from_numpy(val_preds))

  print("-" * 78)
  print(
    f"Validation Accuracy: {val_metrics['accuracy']:.2f}% | "
    f"Validation Macro-F1: {val_metrics['macro_f1']:.4f}"
  )
  print("Per-class Validation Performance:")
  for name, f1, rec, prec in zip(
    CLASS_NAMES,
    val_metrics["per_class_f1"],
    val_metrics["per_class_recall"],
    val_metrics["per_class_precision"],
    strict=False,
  ):
    print(f"  Class {name}: F1={f1:.4f} | Recall={rec:.4f} | Precision={prec:.4f}")

  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  joblib.dump(model, save_path)
  print(f"Saved trained Random Forest model to: {save_path}")


def run_training(
  model_name: str = "LogisticRegression",
  epochs: int = 15,
  lr: float = 0.01,
  optimizer_name: str = "sgd",
  batch_size: int = 64,
  augment: bool = False,
  balanced_sampling: bool = False,
  quick_test: bool = False,
  save_path: str = "results/best_model.pt",
  n_estimators: int = 100,
  max_depth: int | None = 20,
  class_weight: str | None = "balanced",
) -> None:
  """
  Dispatch training to neural or classical pipeline based on model name.
  """
  if model_name.lower() in ("randomforest", "rf"):
    if save_path in ("results/best_model.pt", "results/logistic_regression.pt"):
      save_path = "results/random_forest.joblib"
    train_random_forest(
      n_estimators=n_estimators,
      max_depth=max_depth,
      class_weight=class_weight,
      quick_test=quick_test,
      save_path=save_path,
    )
    return

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  train_loader, val_loader, class_weights = load_data(
    batch_size=batch_size,
    augment=augment,
    balanced_sampling=balanced_sampling,
    quick_test=quick_test,
    class_weight=class_weight,
  )

  model = get_model(model_name, input_dim=WINDOW_SIZE, num_classes=NUM_CLASSES)
  model = model.to(device)

  if class_weights is not None:
    class_weights = class_weights.to(device)
  criterion = nn.CrossEntropyLoss(weight=class_weights)

  if optimizer_name.lower() == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  elif optimizer_name.lower() == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
  else:
    raise ValueError(f"Unsupported optimizer: '{optimizer_name}'")

  param_count = sum(p.numel() for p in model.parameters())
  print(f"Model: {type(model).__name__} ({param_count} parameters)")
  if class_weights is not None:
    print(
      f"Class weights (N, S, V, F): {[round(w, 4) for w in class_weights.tolist()]}"
    )
  else:
    print("Class weights: None (unweighted cross-entropy loss)")
  print("-" * 78)

  best_val_f1 = 0.0
  best_val_acc = 0.0

  for epoch in range(1, epochs + 1):
    train_loss, train_acc = train_one_epoch(
      model, train_loader, criterion, optimizer, device
    )
    val_loss, val_metrics = evaluate(model, val_loader, criterion, device)

    val_acc = val_metrics["accuracy"]
    val_f1 = val_metrics["macro_f1"]

    print(
      f"Epoch {epoch:2d}/{epochs:2d} | "
      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:5.2f}% | "
      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.2f}% | Val Macro-F1: {val_f1:.4f}"
    )

    if val_f1 > best_val_f1:
      best_val_f1 = val_f1
      best_val_acc = val_acc
      os.makedirs(os.path.dirname(save_path), exist_ok=True)
      torch.save(
        {
          "epoch": epoch,
          "model_name": type(model).__name__,
          "model_state_dict": model.state_dict(),
          "val_macro_f1": val_f1,
          "val_accuracy": val_acc,
          "val_metrics": val_metrics,
        },
        save_path,
      )

  print("-" * 78)
  print(
    f"Training complete. Best Val Macro-F1: {best_val_f1:.4f} "
    f"(Validation Accuracy: {best_val_acc:.2f}%)"
  )
  print(f"Saved checkpoint to: {save_path}")

  if os.path.exists(save_path):
    ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
    metrics = ckpt["val_metrics"]
    print("\nPer-class Validation Performance:")
    for name, f1, rec, prec in zip(
      CLASS_NAMES,
      metrics["per_class_f1"],
      metrics["per_class_recall"],
      metrics["per_class_precision"],
      strict=False,
    ):
      print(f"  Class {name}: F1={f1:.4f} | Recall={rec:.4f} | Precision={prec:.4f}")


def main() -> None:
  model_choices = [cls.__name__ for cls in MODELS]
  parser = argparse.ArgumentParser(
    description="Train models for ECG heartbeat classification."
  )
  parser.add_argument(
    "--model",
    type=str,
    default=model_choices[0],
    help=f"Model to train. Choices: {model_choices}.",
  )
  parser.add_argument(
    "--epochs",
    type=int,
    default=15,
    help="Number of epochs to train (for neural models, default: 15).",
  )
  parser.add_argument(
    "--lr",
    type=float,
    default=0.01,
    help="Learning rate (default: 0.01).",
  )
  parser.add_argument(
    "--optimizer",
    type=str,
    choices=["sgd", "adam"],
    default="sgd",
    help="Optimizer: 'sgd' or 'adam' (default: sgd).",
  )
  parser.add_argument(
    "--batch-size",
    type=int,
    default=64,
    help="Batch size (default: 64).",
  )
  parser.add_argument(
    "--augment",
    action="store_true",
    help="Apply training augmentations.",
  )
  parser.add_argument(
    "--balanced-sampling",
    action="store_true",
    help="Use weighted random sampling in the training DataLoader.",
  )
  parser.add_argument(
    "--quick-test",
    action="store_true",
    help="Run on a small subset of samples for quick verification.",
  )
  parser.add_argument(
    "--save-path",
    type=str,
    default="results/best_model.pt",
    help="File path to save the best model weights.",
  )
  parser.add_argument(
    "--n-estimators",
    type=int,
    default=100,
    help="Number of trees in Random Forest (default: 100).",
  )
  parser.add_argument(
    "--max-depth",
    type=int,
    default=20,
    help="Max tree depth in Random Forest (default: 20, -1 for unconstrained).",
  )
  parser.add_argument(
    "--class-weight",
    type=str,
    default="balanced",
    help=(
      "Class weighting strategy across all models: "
      "'balanced' (inverse frequency), 'sqrt_inverse', 'effective', 'none', "
      "or comma-separated values (e.g. '1.0,10.0,5.0,20.0'). Default: balanced."
    ),
  )
  args = parser.parse_args()

  max_depth = None if args.max_depth <= 0 else args.max_depth

  run_training(
    model_name=args.model,
    epochs=args.epochs,
    lr=args.lr,
    optimizer_name=args.optimizer,
    batch_size=args.batch_size,
    augment=args.augment,
    balanced_sampling=args.balanced_sampling,
    quick_test=args.quick_test,
    save_path=args.save_path,
    n_estimators=args.n_estimators,
    max_depth=max_depth,
    class_weight=args.class_weight,
  )


if __name__ == "__main__":
  main()
