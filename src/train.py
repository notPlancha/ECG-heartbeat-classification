"""
train.py
========
Training pipeline for ECG heartbeat classification models.

Supports multiple models via src.model.get_model and two data sources:
1. Preprocessed .npz splits from dataset/processed (standard project pipeline)
2. Raw records loaded via ml-model/data_loader.py (classmate baseline loader)
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Add workspace root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
  sys.path.insert(0, str(PROJECT_ROOT))

# Add ml-model directory to sys.path for importing raw loader if needed
ML_MODEL_DIR = PROJECT_ROOT / "ml-model"
if str(ML_MODEL_DIR) not in sys.path:
  sys.path.insert(0, str(ML_MODEL_DIR))

from src.constants import CLASS_NAMES, NUM_CLASSES, WINDOW_SIZE
from src.dataset import ECGDataset, get_data_loader
from src.model import get_model, list_models


def calculate_class_weights(
  labels: torch.Tensor, num_classes: int = NUM_CLASSES
) -> torch.Tensor:
  """
  Calculate inverse-frequency class weights for cross-entropy loss.
  """
  labels_flat = torch.as_tensor(labels, dtype=torch.long).reshape(-1)
  counts = torch.bincount(labels_flat, minlength=num_classes).float()
  counts = torch.clamp(counts, min=1.0)
  total = counts.sum()
  return total / (num_classes * counts)


def compute_metrics(
  y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int = NUM_CLASSES
) -> dict[str, float | list[float]]:
  """
  Compute overall accuracy, per-class F1, and macro-averaged F1.
  """
  cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
  for t, p in zip(y_true.view(-1), y_pred.view(-1), strict=False):
    cm[t, p] += 1

  f1_scores = []
  recalls = []
  precisions = []

  for i in range(num_classes):
    tp = cm[i, i].item()
    fp = (cm[:, i].sum() - tp).item()
    fn = (cm[i, :].sum() - tp).item()

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    precisions.append(prec)
    recalls.append(rec)
    f1_scores.append(f1)

  total_samples = max(cm.sum().item(), 1)
  accuracy = 100.0 * cm.diag().sum().item() / total_samples
  macro_f1 = sum(f1_scores) / num_classes

  return {
    "accuracy": accuracy,
    "macro_f1": macro_f1,
    "per_class_f1": f1_scores,
    "per_class_recall": recalls,
    "per_class_precision": precisions,
  }


def load_data_from_classmates(
  quick_test: bool = False,
  batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, torch.Tensor]:
  """
  Load raw training and test data using the classmates' loader script.
  """
  from data_loader import _load_records, load_testing_data, load_training_data

  if quick_test:
    print("Quick test mode: loading sample records 101 and 100...")
    X_train, y_train = _load_records(["101"])
    X_test, y_test = _load_records(["100"])
  else:
    print("Loading full training data from raw WFDB records...")
    X_train, y_train = load_training_data()
    print("Loading full test data from raw WFDB records...")
    X_test, y_test = load_testing_data()

  X_train_t = torch.tensor(X_train, dtype=torch.float32)
  y_train_t = torch.tensor(y_train, dtype=torch.long)
  X_test_t = torch.tensor(X_test, dtype=torch.float32)
  y_test_t = torch.tensor(y_test, dtype=torch.long)

  class_weights = calculate_class_weights(y_train_t, NUM_CLASSES)

  train_ds = TensorDataset(X_train_t, y_train_t)
  val_ds = TensorDataset(X_test_t, y_test_t)

  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

  return train_loader, val_loader, class_weights


def load_data_from_npz(
  train_path: str = "dataset/processed/train.npz",
  val_path: str = "dataset/processed/val.npz",
  batch_size: int = 64,
  augment: bool = False,
  balanced_sampling: bool = False,
  quick_test: bool = False,
) -> tuple[DataLoader, DataLoader, torch.Tensor]:
  """
  Load preprocessed heartbeat splits (.npz) using ECGDataset.
  """
  if not os.path.exists(train_path):
    raise FileNotFoundError(
      f"Training split not found at '{train_path}'. "
      "Run 'uv run python src/prepare_data.py' first."
    )
  if not os.path.exists(val_path):
    raise FileNotFoundError(
      f"Validation split not found at '{val_path}'. "
      "Run 'uv run python src/prepare_data.py' first."
    )

  print(f"Loading preprocessed training data from {train_path}...")
  train_ds = ECGDataset.from_npz(train_path, augment=augment)
  print(f"Loading preprocessed validation data from {val_path}...")
  val_ds = ECGDataset.from_npz(val_path, augment=False)

  if quick_test:
    print("Quick test mode: truncating dataset splits...")
    train_ds.X = train_ds.X[:1000]
    train_ds.y = train_ds.y[:1000]
    val_ds.X = val_ds.X[:500]
    val_ds.y = val_ds.y[:500]

  # Compute inverse-frequency class weights from the training set
  weights_np = train_ds.compute_class_weights("inverse")
  class_weights = torch.from_numpy(weights_np).float()

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
  print(f"Train class distribution: {train_ds.get_class_counts()}")
  print(f"Val class distribution:   {val_ds.get_class_counts()}")

  return train_loader, val_loader, class_weights


def train_one_epoch(
  model: nn.Module,
  loader: DataLoader,
  criterion: nn.Module,
  optimizer: torch.optim.Optimizer,
  device: torch.device,
) -> tuple[float, float]:
  """
  Train model for one epoch and return average loss and accuracy.
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
  Evaluate model on a dataset and return loss and detailed metrics.
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


def run_training(
  model_name: str = "logistic_regression",
  loader_type: str = "npz",
  epochs: int = 15,
  lr: float = 0.001,
  optimizer_name: str = "sgd",
  batch_size: int = 64,
  augment: bool = False,
  balanced_sampling: bool = False,
  quick_test: bool = False,
  save_path: str = "results/best_model.pt",
) -> None:
  """
  Configure and run the full training loop with checkpointing.
  """
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")

  if loader_type == "classmates":
    print("Using classmates data loader (ml-model/data_loader.py)...")
    train_loader, val_loader, class_weights = load_data_from_classmates(
      quick_test=quick_test, batch_size=batch_size
    )
  elif loader_type == "npz":
    train_loader, val_loader, class_weights = load_data_from_npz(
      batch_size=batch_size,
      augment=augment,
      balanced_sampling=balanced_sampling,
      quick_test=quick_test,
    )
  else:
    raise ValueError(f"Unknown loader type: '{loader_type}'")

  model = get_model(model_name, in_features=WINDOW_SIZE, num_classes=NUM_CLASSES)
  model = model.to(device)

  class_weights = class_weights.to(device)
  criterion = nn.CrossEntropyLoss(weight=class_weights)

  if optimizer_name.lower() == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  elif optimizer_name.lower() == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
  else:
    raise ValueError(f"Unsupported optimizer: '{optimizer_name}'")

  print(
    f"Model: {type(model).__name__} ({sum(p.numel() for p in model.parameters())} params)"
  )
  print(f"Optimizer: {optimizer_name.upper()} (lr={lr})")
  print(f"Class weights (N, S, V, F): {[round(w, 4) for w in class_weights.tolist()]}")
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

    # Save model weights whenever validation Macro-F1 improves
    if val_f1 > best_val_f1:
      best_val_f1 = val_f1
      best_val_acc = val_acc
      os.makedirs(os.path.dirname(save_path), exist_ok=True)
      torch.save(
        {
          "epoch": epoch,
          "model_name": model_name,
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

  # Print per-class F1 breakdown for the best checkpoint
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
  available_models = list_models()
  parser = argparse.ArgumentParser(
    description="Train models for ECG heartbeat classification."
  )
  parser.add_argument(
    "--model",
    type=str,
    default="logistic_regression",
    help=f"Model architecture to train. Available: {', '.join(available_models)}.",
  )
  parser.add_argument(
    "--loader",
    type=str,
    choices=["npz", "classmates"],
    default="npz",
    help="Data loading source: 'npz' (preprocessed splits) or 'classmates' (raw WFDB files).",
  )
  parser.add_argument(
    "--epochs",
    type=int,
    default=15,
    help="Number of training epochs (default: 15).",
  )
  parser.add_argument(
    "--lr",
    type=float,
    default=0.001,
    help="Learning rate for the optimizer (default: 0.001).",
  )
  parser.add_argument(
    "--optimizer",
    type=str,
    choices=["sgd", "adam"],
    default="sgd",
    help="Optimizer to use: 'sgd' or 'adam' (default: sgd).",
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
    help="Apply training-only data augmentation (for npz loader).",
  )
  parser.add_argument(
    "--balanced-sampling",
    action="store_true",
    help="Use weighted random sampling in DataLoader (for npz loader).",
  )
  parser.add_argument(
    "--quick-test",
    action="store_true",
    help="Run quick sanity check on a small subset of samples.",
  )
  parser.add_argument(
    "--save-path",
    type=str,
    default="results/logistic_regression.pt",
    help="Path to save the best model weights.",
  )
  args = parser.parse_args()

  run_training(
    model_name=args.model,
    loader_type=args.loader,
    epochs=args.epochs,
    lr=args.lr,
    optimizer_name=args.optimizer,
    batch_size=args.batch_size,
    augment=args.augment,
    balanced_sampling=args.balanced_sampling,
    quick_test=args.quick_test,
    save_path=args.save_path,
  )


if __name__ == "__main__":
  main()
