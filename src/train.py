"""
train.py
========
Training pipeline for ECG heartbeat classification models using
preprocessed .npz splits.
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from pyhere import here
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = here()
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

try:
  from src.constants import (
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLASS_WEIGHT,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_MODELS_DIR,
    DEFAULT_OPTIMIZER,
    DEFAULT_RANDOM_STATE,
    DEFAULT_RESAMPLING_STRATEGY,
    DEFAULT_RF_MAX_DEPTH,
    DEFAULT_RF_N_ESTIMATORS,
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VAL_PATH,
    NUM_CLASSES,
    RESAMPLING_STRATEGIES,
    WINDOW_SIZE,
    set_seed,
  )
except ImportError:
  from constants import (
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLASS_WEIGHT,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_MODELS_DIR,
    DEFAULT_OPTIMIZER,
    DEFAULT_RANDOM_STATE,
    DEFAULT_RESAMPLING_STRATEGY,
    DEFAULT_RF_MAX_DEPTH,
    DEFAULT_RF_N_ESTIMATORS,
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    DEFAULT_VAL_PATH,
    NUM_CLASSES,
    RESAMPLING_STRATEGIES,
    WINDOW_SIZE,
    set_seed,
  )


def to_snake_case(name: str) -> str:
  """Convert CamelCase class name to snake_case."""
  s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
  return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s).lower()


from src.dataset import ECGDataset, apply_resampling, get_data_loader
from src.experiment import resolve_run_dir, save_run_settings
from src.model import (
  MODELS,
  ClassicalECGModel,
  NeuralECGModel,
  get_model,
)


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

  match strategy_clean.lower():
    case "balanced" | "inverse":
      weights_np = dataset.compute_class_weights("inverse")
      weights_torch = torch.from_numpy(weights_np).float()
      return weights_torch, weights_np, "balanced"
    case "sqrt_inverse":
      weights_np = dataset.compute_class_weights("sqrt_inverse")
      weights_torch = torch.from_numpy(weights_np).float()
      rf_param = {i: float(w) for i, w in enumerate(weights_np)}
      return weights_torch, weights_np, rf_param
    case "effective":
      weights_np = dataset.compute_class_weights("effective")
      weights_torch = torch.from_numpy(weights_np).float()
      rf_param = {i: float(w) for i, w in enumerate(weights_np)}
      return weights_torch, weights_np, rf_param
    case "balanced_subsample":
      weights_np = dataset.compute_class_weights("inverse")
      weights_torch = torch.from_numpy(weights_np).float()
      return weights_torch, weights_np, "balanced_subsample"
    case _:
      valid = ["balanced", "inverse", "sqrt_inverse", "effective", "none"]
      raise ValueError(
        f"Unknown class weight strategy '{strategy}'. Choices: {valid} "
        "or comma-separated values (e.g. '1.0,10.0,5.0,20.0')."
      )


def load_data(
  train_path: str = DEFAULT_TRAIN_PATH,
  val_path: str = DEFAULT_VAL_PATH,
  batch_size: int = DEFAULT_BATCH_SIZE,
  augment: bool = False,
  balanced_sampling: bool = False,
  quick_test: bool = False,
  class_weight: str | None = DEFAULT_CLASS_WEIGHT,
  resample: str = DEFAULT_RESAMPLING_STRATEGY,
  seed: int | None = DEFAULT_RANDOM_STATE,
) -> tuple[DataLoader, DataLoader, torch.Tensor | None]:
  """
  Load preprocessed .npz datasets and return train/val loaders with class weights.
  """
  train_p = Path(train_path) if Path(train_path).exists() else here(train_path)
  val_p = Path(val_path) if Path(val_path).exists() else here(val_path)

  if not train_p.exists():
    raise FileNotFoundError(
      f"Training file '{train_path}' not found. "
      "Run 'uv run python src/prepare_data.py' first."
    )
  if not val_p.exists():
    raise FileNotFoundError(
      f"Validation file '{val_path}' not found. "
      "Run 'uv run python src/prepare_data.py' first."
    )

  train_ds = ECGDataset.from_npz(train_p, augment=augment, seed=seed)
  val_ds = ECGDataset.from_npz(val_p, augment=False, seed=seed)

  if quick_test:
    train_ds.X = train_ds.X[:1000]
    train_ds.y = train_ds.y[:1000]
    val_ds.X = val_ds.X[:500]
    val_ds.y = val_ds.y[:500]

  if resample and resample.lower() != "none":
    print(f"Applying '{resample}' resampling on training split (seed={seed})...")
    effective_seed = seed if seed is not None else DEFAULT_RANDOM_STATE
    x_res, y_res = apply_resampling(
      train_ds.X, train_ds.y, strategy=resample, seed=effective_seed
    )
    train_ds = ECGDataset(
      X=x_res,
      y=y_res,
      augment=augment,
      seed=seed,
    )

  class_weights_torch, _, _ = resolve_class_weights(train_ds, class_weight)

  train_loader = get_data_loader(
    train_ds,
    batch_size=batch_size,
    shuffle=not balanced_sampling,
    balanced_sampling=balanced_sampling,
    seed=seed,
  )
  val_loader = get_data_loader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    balanced_sampling=False,
    seed=seed,
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
def evaluate_neural(
  model: nn.Module,
  loader: DataLoader,
  criterion: nn.Module,
  device: torch.device,
  num_classes: int = NUM_CLASSES,
) -> tuple[float, dict[str, float | list[float]]]:
  """
  Evaluate neural model on a dataset and return loss and classification metrics.
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


def train_classical_model(
  model: ClassicalECGModel,
  train_path: str = DEFAULT_TRAIN_PATH,
  val_path: str = DEFAULT_VAL_PATH,
  quick_test: bool = False,
  save_path: str | None = None,
  resample: str = DEFAULT_RESAMPLING_STRATEGY,
  seed: int = DEFAULT_RANDOM_STATE,
) -> None:
  """
  Train any classical (non-neural) model adhering to ClassicalECGModel.
  """
  set_seed(seed)

  train_p = Path(train_path) if Path(train_path).exists() else here(train_path)
  val_p = Path(val_path) if Path(val_path).exists() else here(val_path)

  if not train_p.exists():
    raise FileNotFoundError(f"Training file '{train_path}' not found.")
  if not val_p.exists():
    raise FileNotFoundError(f"Validation file '{val_path}' not found.")

  slug = to_snake_case(type(model).__name__)
  if save_path is None:
    save_path = str(Path(DEFAULT_MODELS_DIR) / f"{slug}.joblib")
  elif not Path(save_path).is_absolute() and not Path(save_path).parent.exists():
    save_path = str(here(save_path))

  print(f"Loading training split from {train_p}...")
  train_ds = ECGDataset.from_npz(train_p, augment=False, seed=seed)
  val_ds = ECGDataset.from_npz(val_p, augment=False, seed=seed)

  X_train = train_ds.X
  y_train = train_ds.y
  X_val = val_ds.X
  y_val = val_ds.y

  if quick_test:
    X_train = X_train[:1000]
    y_train = y_train[:1000]
    X_val = X_val[:500]
    y_val = y_val[:500]

  if resample and resample.lower() != "none":
    print(f"Applying '{resample}' resampling on training split (seed={seed})...")
    X_train, y_train = apply_resampling(X_train, y_train, strategy=resample, seed=seed)
    res_counts = np.bincount(y_train, minlength=NUM_CLASSES)
    print(f"Resampled train samples: {len(X_train)}")
    print(f"Resampled distribution: {dict(zip(CLASS_NAMES, res_counts, strict=False))}")

  print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)}")
  print(
    f"Train class distribution: {dict(zip(CLASS_NAMES, np.bincount(y_train, minlength=NUM_CLASSES), strict=False))}"
  )
  print(f"Val class distribution:   {val_ds.get_class_counts()}")
  print(f"Random seed: {seed}")

  print(f"Fitting model: {type(model).__name__}...")
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
  print(f"Saved trained classical model to: {save_path}")

  return {
    "val_metrics": val_metrics,
    "train_samples": len(X_train),
    "train_distribution": {
      CLASS_NAMES[i]: int(c)
      for i, c in enumerate(np.bincount(y_train, minlength=NUM_CLASSES))
    },
    "val_samples": len(X_val),
    "val_distribution": val_ds.get_class_counts(),
  }


def train_neural_model(
  model: NeuralECGModel,
  train_path: str = DEFAULT_TRAIN_PATH,
  val_path: str = DEFAULT_VAL_PATH,
  epochs: int = DEFAULT_EPOCHS,
  lr: float = DEFAULT_LR,
  optimizer_name: str = DEFAULT_OPTIMIZER,
  batch_size: int = DEFAULT_BATCH_SIZE,
  augment: bool = False,
  balanced_sampling: bool = False,
  quick_test: bool = False,
  class_weight: str | None = DEFAULT_CLASS_WEIGHT,
  resample: str = DEFAULT_RESAMPLING_STRATEGY,
  save_path: str | None = None,
  seed: int = DEFAULT_RANDOM_STATE,
) -> None:
  """
  Train any neural network adhering to NeuralECGModel.
  """
  set_seed(seed)

  slug = to_snake_case(type(model).__name__)
  if save_path is None:
    save_path = str(Path(DEFAULT_MODELS_DIR) / f"{slug}.pt")
  elif not Path(save_path).is_absolute() and not Path(save_path).parent.exists():
    save_path = str(here(save_path))

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  train_loader, val_loader, class_weights = load_data(
    train_path=train_path,
    val_path=val_path,
    batch_size=batch_size,
    augment=augment,
    balanced_sampling=balanced_sampling,
    quick_test=quick_test,
    class_weight=class_weight,
    resample=resample,
    seed=seed,
  )

  model = model.to(device)

  if class_weights is not None:
    class_weights = class_weights.to(device)
  criterion = nn.CrossEntropyLoss(weight=class_weights)

  match optimizer_name.lower():
    case "adam":
      optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    case "sgd":
      optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    case _:
      raise ValueError(f"Unsupported optimizer: '{optimizer_name}'")

  param_count = model.count_parameters()
  print(f"Model: {type(model).__name__} ({param_count} parameters)")
  print(f"Random seed: {seed}")
  if class_weights is not None:
    print(
      f"Class weights (N, S, V, F): {[round(w, 4) for w in class_weights.tolist()]}"
    )
  else:
    print("Class weights: None (unweighted cross-entropy loss)")
  print("-" * 78)

  best_val_f1 = 0.0
  best_val_acc = 0.0
  best_val_metrics: dict[str, Any] = {}

  for epoch in range(1, epochs + 1):
    train_loss, train_acc = train_one_epoch(
      model, train_loader, criterion, optimizer, device
    )
    val_loss, val_metrics = evaluate_neural(model, val_loader, criterion, device)

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
      best_val_metrics = val_metrics
      os.makedirs(os.path.dirname(save_path), exist_ok=True)
      torch.save(
        {
          "epoch": epoch,
          "model_name": type(model).__name__,
          "model_state_dict": model.state_dict(),
          "val_macro_f1": val_f1,
          "val_accuracy": val_acc,
          "val_metrics": val_metrics,
          "random_seed": seed,
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

  return {
    "val_metrics": best_val_metrics,
    "train_samples": len(train_loader.dataset),
    "train_distribution": (
      train_loader.dataset.get_class_counts()
      if hasattr(train_loader.dataset, "get_class_counts")
      else {}
    ),
    "val_samples": len(val_loader.dataset),
    "val_distribution": (
      val_loader.dataset.get_class_counts()
      if hasattr(val_loader.dataset, "get_class_counts")
      else {}
    ),
  }


def run_training(
  model_name: str = "LogisticRegression",
  train_path: str = DEFAULT_TRAIN_PATH,
  val_path: str = DEFAULT_VAL_PATH,
  test_path: str = DEFAULT_TEST_PATH,
  epochs: int = DEFAULT_EPOCHS,
  lr: float = DEFAULT_LR,
  optimizer_name: str = DEFAULT_OPTIMIZER,
  batch_size: int = DEFAULT_BATCH_SIZE,
  augment: bool = False,
  balanced_sampling: bool = False,
  quick_test: bool = False,
  save_path: str | None = None,
  n_estimators: int = DEFAULT_RF_N_ESTIMATORS,
  max_depth: int | None = DEFAULT_RF_MAX_DEPTH,
  class_weight: str | None = DEFAULT_CLASS_WEIGHT,
  resample: str = DEFAULT_RESAMPLING_STRATEGY,
  run: str | None = None,
  evaluate: bool = False,
  seed: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Any]:
  """
  Instantiate model and dispatch to classical or neural pipeline via base class check.
  """
  set_seed(seed)

  model = get_model(
    model_name,
    input_dim=WINDOW_SIZE,
    num_classes=NUM_CLASSES,
    n_estimators=n_estimators,
    max_depth=max_depth,
    class_weight=class_weight,
    random_state=seed,
  )

  slug = to_snake_case(type(model).__name__)
  run_dir = resolve_run_dir(run)
  if run_dir is not None:
    run_dir.mkdir(parents=True, exist_ok=True)
    ext = ".joblib" if isinstance(model, ClassicalECGModel) else ".pt"
    if save_path is None:
      save_path = str(run_dir / f"{slug}{ext}")
    print(f"Isolated experiment run active: {run_dir.name} ({run_dir})")

  match model:
    case ClassicalECGModel():
      if save_path is None:
        save_path = str(Path(DEFAULT_MODELS_DIR) / f"{slug}.joblib")
      train_result = train_classical_model(
        model=model,
        train_path=train_path,
        val_path=val_path,
        quick_test=quick_test,
        save_path=save_path,
        resample=resample,
        seed=seed,
      )
    case NeuralECGModel():
      if save_path is None:
        save_path = str(Path(DEFAULT_MODELS_DIR) / f"{slug}.pt")
      train_result = train_neural_model(
        model=model,
        train_path=train_path,
        val_path=val_path,
        epochs=epochs,
        lr=lr,
        optimizer_name=optimizer_name,
        batch_size=batch_size,
        augment=augment,
        balanced_sampling=balanced_sampling,
        quick_test=quick_test,
        class_weight=class_weight,
        resample=resample,
        save_path=save_path,
        seed=seed,
      )
    case _:
      raise TypeError(
        f"Model '{type(model).__name__}' does not inherit from BaseECGModel."
      )

  if run_dir is not None:
    settings = {
      "model_name": model_name,
      "model_class": type(model).__name__,
      "parameters": model.count_parameters(),
      "seed": seed,
      "class_weight": class_weight,
      "resample": resample,
      "augment": augment,
      "balanced_sampling": balanced_sampling,
      "train_path": str(train_path),
      "val_path": str(val_path),
      "train_samples": train_result.get("train_samples"),
      "train_distribution": train_result.get("train_distribution"),
      "val_samples": train_result.get("val_samples"),
      "val_distribution": train_result.get("val_distribution"),
      "val_metrics": train_result.get("val_metrics"),
    }
    match model:
      case ClassicalECGModel():
        settings["hyperparameters"] = {
          "n_estimators": n_estimators,
          "max_depth": max_depth,
          "random_state": seed,
        }
      case NeuralECGModel():
        settings["hyperparameters"] = {
          "epochs": epochs,
          "lr": lr,
          "optimizer": optimizer_name,
          "batch_size": batch_size,
        }
    settings_file = save_run_settings(run_dir, settings)
    print(f"Saved run settings to: {settings_file}")

  if evaluate:
    from src.evaluate import evaluate_model

    print("-" * 78)
    print(f"Running automated test evaluation for {model_name} on DS2...")
    run_target = str(run_dir) if run_dir is not None else None
    eval_results = evaluate_model(
      model_name=model_name,
      checkpoint_path=save_path,
      data_path=test_path,
      batch_size=batch_size,
      run=run_target,
      seed=seed,
    )
    train_result["eval_results"] = eval_results

  return train_result


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
    "--train-path",
    type=str,
    default=DEFAULT_TRAIN_PATH,
    help=f"Path to training .npz file (default: {DEFAULT_TRAIN_PATH}).",
  )
  parser.add_argument(
    "--val-path",
    type=str,
    default=DEFAULT_VAL_PATH,
    help=f"Path to validation .npz file (default: {DEFAULT_VAL_PATH}).",
  )
  parser.add_argument(
    "--epochs",
    type=int,
    default=DEFAULT_EPOCHS,
    help=f"Number of epochs to train (for neural models, default: {DEFAULT_EPOCHS}).",
  )
  parser.add_argument(
    "--lr",
    type=float,
    default=DEFAULT_LR,
    help=f"Learning rate (default: {DEFAULT_LR}).",
  )
  parser.add_argument(
    "--optimizer",
    type=str,
    choices=["sgd", "adam"],
    default=DEFAULT_OPTIMIZER,
    help=f"Optimizer: 'sgd' or 'adam' (default: {DEFAULT_OPTIMIZER}).",
  )
  parser.add_argument(
    "--batch-size",
    type=int,
    default=DEFAULT_BATCH_SIZE,
    help=f"Batch size (default: {DEFAULT_BATCH_SIZE}).",
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
    default=None,
    help="File path to save the best model weights (default: results/models/<model_slug>.<ext>).",
  )
  parser.add_argument(
    "--n-estimators",
    type=int,
    default=DEFAULT_RF_N_ESTIMATORS,
    help=f"Number of trees in Random Forest (default: {DEFAULT_RF_N_ESTIMATORS}).",
  )
  parser.add_argument(
    "--max-depth",
    type=int,
    default=DEFAULT_RF_MAX_DEPTH,
    help=f"Max tree depth in Random Forest (default: {DEFAULT_RF_MAX_DEPTH}, -1 for unconstrained).",
  )
  parser.add_argument(
    "--class-weight",
    type=str,
    default=DEFAULT_CLASS_WEIGHT,
    help=(
      "Class weighting strategy across all models: "
      "'balanced', 'sqrt_inverse', 'effective', 'none', "
      "or comma-separated values (e.g. '1.0,10.0,5.0,20.0'). "
      f"Default: {DEFAULT_CLASS_WEIGHT}."
    ),
  )
  parser.add_argument(
    "--resample",
    type=str,
    choices=list(RESAMPLING_STRATEGIES),
    default=DEFAULT_RESAMPLING_STRATEGY,
    help=(
      "Resampling strategy using imblearn on training split: "
      "'none' (default), 'hybrid' (undersample N to 2*F + targeted oversampling), "
      "'oversample' (RandomOverSampler balancing all classes), "
      "or 'undersample' (RandomUnderSampler on N)."
    ),
  )
  parser.add_argument(
    "--run",
    nargs="?",
    const="auto",
    default=None,
    help=(
      "Save experiment outputs to an isolated run folder (e.g. --run or --run run1). "
      "Automatically increments (run1, run2, ...) if 'auto' or omitted."
    ),
  )
  parser.add_argument(
    "--eval",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="Automatically evaluate trained model on held-out test split (DS2) after training.",
  )
  parser.add_argument(
    "--test-path",
    type=str,
    default=DEFAULT_TEST_PATH,
    help=f"Path to held-out test .npz file for evaluation (default: {DEFAULT_TEST_PATH}).",
  )
  parser.add_argument(
    "--seed",
    "--random-state",
    dest="seed",
    type=int,
    default=DEFAULT_RANDOM_STATE,
    help=f"Random seed for reproducibility (default: {DEFAULT_RANDOM_STATE}).",
  )
  args = parser.parse_args()

  max_depth = None if args.max_depth <= 0 else args.max_depth

  run_training(
    model_name=args.model,
    train_path=args.train_path,
    val_path=args.val_path,
    test_path=args.test_path,
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
    resample=args.resample,
    run=args.run,
    evaluate=args.evaluate,
    seed=args.seed,
  )


if __name__ == "__main__":
  main()
