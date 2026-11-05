"""
model.py
========
Model definitions and base classes for ECG heartbeat classification.
"""

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from pyhere import here
from sklearn.ensemble import RandomForestClassifier
from torch import nn

try:
  from src.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLASS_WEIGHT,
    DEFAULT_CNN_CHANNELS_1,
    DEFAULT_CNN_CHANNELS_2,
    DEFAULT_CNN_CHANNELS_3,
    DEFAULT_CNN_KERNEL_SIZE_1,
    DEFAULT_CNN_KERNEL_SIZE_2,
    DEFAULT_CNN_KERNEL_SIZE_3,
    DEFAULT_RANDOM_STATE,
    DEFAULT_RESULTS_DIR,
    DEFAULT_RF_MAX_DEPTH,
    DEFAULT_RF_MAX_FEATURES,
    DEFAULT_RF_MIN_SAMPLES_LEAF,
    DEFAULT_RF_N_ESTIMATORS,
    NUM_CLASSES,
    WINDOW_SIZE,
  )
except ImportError:
  from constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLASS_WEIGHT,
    DEFAULT_CNN_CHANNELS_1,
    DEFAULT_CNN_CHANNELS_2,
    DEFAULT_CNN_CHANNELS_3,
    DEFAULT_CNN_KERNEL_SIZE_1,
    DEFAULT_CNN_KERNEL_SIZE_2,
    DEFAULT_CNN_KERNEL_SIZE_3,
    DEFAULT_RANDOM_STATE,
    DEFAULT_RESULTS_DIR,
    DEFAULT_RF_MAX_DEPTH,
    DEFAULT_RF_MAX_FEATURES,
    DEFAULT_RF_MIN_SAMPLES_LEAF,
    DEFAULT_RF_N_ESTIMATORS,
    NUM_CLASSES,
    WINDOW_SIZE,
  )


class BaseECGModel(ABC):
  """
  Unified base class for all ECG classification models.
  """

  def __init__(
    self,
    input_dim: int = WINDOW_SIZE,
    num_classes: int = NUM_CLASSES,
  ) -> None:
    self.input_dim = input_dim
    self.num_classes = num_classes

  @abstractmethod
  def predict(
    self,
    x: np.ndarray | torch.Tensor,
    batch_size: int | None = DEFAULT_BATCH_SIZE,
  ) -> np.ndarray:
    """Predict discrete class labels for input heartbeats."""

  @abstractmethod
  def predict_proba(
    self,
    x: np.ndarray | torch.Tensor,
    batch_size: int | None = DEFAULT_BATCH_SIZE,
  ) -> np.ndarray:
    """Predict class probability distribution for input heartbeats."""

  @abstractmethod
  def count_parameters(self) -> int:
    """Return total number of parameters or tree nodes."""


class NeuralECGModel(BaseECGModel, nn.Module):
  """
  Base class for neural network models targeting PyTorch and MAX78002.
  """

  def __init__(
    self,
    input_dim: int = WINDOW_SIZE,
    num_classes: int = NUM_CLASSES,
  ) -> None:
    nn.Module.__init__(self)
    BaseECGModel.__init__(self, input_dim=input_dim, num_classes=num_classes)

  @abstractmethod
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Compute raw class logits."""

  @torch.no_grad()
  def predict(
    self,
    x: np.ndarray | torch.Tensor,
    batch_size: int | None = DEFAULT_BATCH_SIZE,
  ) -> np.ndarray:
    """
    Predict discrete class labels with memory-safe batched inference.
    """
    self.eval()
    try:
      device = next(self.parameters()).device
    except StopIteration:
      device = torch.device("cpu")

    num_samples = 1 if (hasattr(x, "ndim") and x.ndim == 1) else len(x)
    if batch_size is None or num_samples <= batch_size:
      if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float().to(device)
      elif isinstance(x, torch.Tensor) and x.device != device:
        x = x.to(device)
      logits = self.forward(x)
      return torch.argmax(logits, dim=-1).cpu().numpy()

    all_preds: list[np.ndarray] = []
    for i in range(0, num_samples, batch_size):
      chunk = x[i : i + batch_size]
      if isinstance(chunk, np.ndarray):
        chunk_t = torch.from_numpy(chunk).float().to(device)
      elif isinstance(chunk, torch.Tensor):
        chunk_t = chunk.to(device)
      else:
        chunk_t = torch.tensor(chunk, dtype=torch.float32, device=device)
      logits = self.forward(chunk_t)
      preds = torch.argmax(logits, dim=-1).cpu().numpy()
      all_preds.append(preds)

    return np.concatenate(all_preds)

  @torch.no_grad()
  def predict_proba(
    self,
    x: np.ndarray | torch.Tensor,
    batch_size: int | None = DEFAULT_BATCH_SIZE,
  ) -> np.ndarray:
    """
    Predict class probability distributions with memory-safe batched inference.
    """
    self.eval()
    try:
      device = next(self.parameters()).device
    except StopIteration:
      device = torch.device("cpu")

    num_samples = 1 if (hasattr(x, "ndim") and x.ndim == 1) else len(x)
    if batch_size is None or num_samples <= batch_size:
      if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float().to(device)
      elif isinstance(x, torch.Tensor) and x.device != device:
        x = x.to(device)
      logits = self.forward(x)
      return torch.softmax(logits, dim=-1).cpu().numpy()

    all_probs: list[np.ndarray] = []
    for i in range(0, num_samples, batch_size):
      chunk = x[i : i + batch_size]
      if isinstance(chunk, np.ndarray):
        chunk_t = torch.from_numpy(chunk).float().to(device)
      elif isinstance(chunk, torch.Tensor):
        chunk_t = chunk.to(device)
      else:
        chunk_t = torch.tensor(chunk, dtype=torch.float32, device=device)
      logits = self.forward(chunk_t)
      probs = torch.softmax(logits, dim=-1).cpu().numpy()
      all_probs.append(probs)

    return np.concatenate(all_probs)

  def count_parameters(self) -> int:
    return sum(p.numel() for p in self.parameters())


class ClassicalECGModel(BaseECGModel):
  """
  Base class for non-neural models such as trees or linear baselines.
  """

  def __init__(
    self,
    input_dim: int = WINDOW_SIZE,
    num_classes: int = NUM_CLASSES,
  ) -> None:
    super().__init__(input_dim=input_dim, num_classes=num_classes)

  @staticmethod
  def prepare_inputs(x: np.ndarray | torch.Tensor) -> np.ndarray:
    """Flatten or squeeze incoming heartbeat arrays into 2D matrices."""
    if isinstance(x, torch.Tensor):
      x = x.detach().cpu().numpy()
    if x.ndim == 1:
      return np.expand_dims(x, 0)
    match x.shape:
      case (_, 1, _):
        return x.squeeze(1)
      case (n, *_) if len(x.shape) > 2:
        return x.reshape(n, -1)
      case _:
        return x

  @abstractmethod
  def fit(
    self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor
  ) -> ClassicalECGModel:
    """Fit classical model on training inputs and target labels."""


class LogisticRegression(NeuralECGModel):
  """
  Multiclass linear classifier for 1D ECG windows.
  """

  def __init__(
    self,
    input_dim: int = WINDOW_SIZE,
    num_classes: int = NUM_CLASSES,
    bias: bool = True,
    random_state: int | None = DEFAULT_RANDOM_STATE,
    **kwargs: Any,
  ) -> None:
    if random_state is not None:
      torch.manual_seed(random_state)
    super().__init__(input_dim=input_dim, num_classes=num_classes)
    self.random_state = random_state
    self.linear = nn.Linear(input_dim, num_classes, bias=bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    match x.dim():
      case 1:
        x = x.unsqueeze(0)
      case d if d > 2:
        x = x.flatten(start_dim=1)
    return self.linear(x)


class RandomForest(ClassicalECGModel):
  """
  Random Forest baseline for ECG heartbeat classification.
  """

  def __init__(
    self,
    input_dim: int = WINDOW_SIZE,
    num_classes: int = NUM_CLASSES,
    n_estimators: int = DEFAULT_RF_N_ESTIMATORS,
    max_depth: int | None = DEFAULT_RF_MAX_DEPTH,
    min_samples_leaf: int = DEFAULT_RF_MIN_SAMPLES_LEAF,
    max_features: str | float = DEFAULT_RF_MAX_FEATURES,
    class_weight: str | dict[int, float] | None = DEFAULT_CLASS_WEIGHT,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = -1,
    **kwargs: Any,
  ) -> None:
    super().__init__(input_dim=input_dim, num_classes=num_classes)
    self.n_estimators = n_estimators
    self.max_depth = max_depth
    self.min_samples_leaf = min_samples_leaf
    self.max_features = max_features
    self.class_weight = class_weight
    self.random_state = random_state
    self.n_jobs = n_jobs
    rf_class_weight = (
      None
      if isinstance(class_weight, str) and class_weight.lower() in ("none", "uniform")
      else class_weight
    )
    self.classifier = RandomForestClassifier(
      n_estimators=n_estimators,
      max_depth=max_depth,
      min_samples_leaf=min_samples_leaf,
      max_features=max_features,
      class_weight=rf_class_weight,
      random_state=random_state,
      n_jobs=n_jobs,
      **kwargs,
    )

  def fit(
    self, x: np.ndarray | torch.Tensor, y: np.ndarray | torch.Tensor
  ) -> RandomForest:
    x_prep = self.prepare_inputs(x)
    if isinstance(y, torch.Tensor):
      y = y.detach().cpu().numpy()
    self.classifier.fit(x_prep, y)
    return self

  def predict(
    self,
    x: np.ndarray | torch.Tensor,
    batch_size: int | None = DEFAULT_BATCH_SIZE,
  ) -> np.ndarray:
    x_prep = self.prepare_inputs(x)
    return self.classifier.predict(x_prep)

  def predict_proba(
    self,
    x: np.ndarray | torch.Tensor,
    batch_size: int | None = DEFAULT_BATCH_SIZE,
  ) -> np.ndarray:
    x_prep = self.prepare_inputs(x)
    return self.classifier.predict_proba(x_prep)

  def count_parameters(self) -> int:
    if hasattr(self.classifier, "estimators_"):
      return sum(tree.tree_.node_count for tree in self.classifier.estimators_)
    return 0


class CNN(NeuralECGModel):
  """
  1D Convolutional Neural Network for ECG heartbeat classification.

  Matches the architecture developed in the ml-model branch and adheres
  to the Analog Devices MAX78002 CNN hardware accelerator constraints:
    - Layer 1: Conv1D(1, 16, kernel_size=5, padding=2) -> ReLU -> MaxPool1D(2)
    - Layer 2: Conv1D(16, 32, kernel_size=5, padding=2) -> ReLU -> MaxPool1D(2)
    - Layer 3: Conv1D(32, 64, kernel_size=3, padding=1) -> ReLU -> MaxPool1D(2)
    - Classification: Linear(2048, 4) for standard 256-sample window
  """

  def __init__(
    self,
    input_dim: int = WINDOW_SIZE,
    num_classes: int = NUM_CLASSES,
    in_channels: int = 1,
    c1: int = DEFAULT_CNN_CHANNELS_1,
    c2: int = DEFAULT_CNN_CHANNELS_2,
    c3: int = DEFAULT_CNN_CHANNELS_3,
    k1: int = DEFAULT_CNN_KERNEL_SIZE_1,
    k2: int = DEFAULT_CNN_KERNEL_SIZE_2,
    k3: int = DEFAULT_CNN_KERNEL_SIZE_3,
    random_state: int | None = DEFAULT_RANDOM_STATE,
    **kwargs: Any,
  ) -> None:
    if random_state is not None:
      torch.manual_seed(random_state)
    super().__init__(input_dim=input_dim, num_classes=num_classes)
    self.random_state = random_state
    self.in_channels = in_channels

    self.features = nn.Sequential(
      nn.Conv1d(in_channels, c1, kernel_size=k1, padding=k1 // 2),
      nn.ReLU(),
      nn.MaxPool1d(kernel_size=2),
      nn.Conv1d(c1, c2, kernel_size=k2, padding=k2 // 2),
      nn.ReLU(),
      nn.MaxPool1d(kernel_size=2),
      nn.Conv1d(c2, c3, kernel_size=k3, padding=k3 // 2),
      nn.ReLU(),
      nn.MaxPool1d(kernel_size=2),
    )

    reduced_dim = input_dim // 8
    self.classifier = nn.Linear(c3 * reduced_dim, num_classes)

  @property
  def conv_layers(self) -> list[nn.Conv1d]:
    """Return list of convolutional layers in sequence."""
    return [m for m in self.features if isinstance(m, nn.Conv1d)]

  @property
  def linear_layer(self) -> nn.Linear:
    """Return linear classification head."""
    return self.classifier

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    match x.dim():
      case 1:
        x = x.unsqueeze(0).unsqueeze(0)
      case 2:
        x = x.unsqueeze(1)
      case 3:
        pass
      case _:
        raise ValueError(f"Expected 1D, 2D, or 3D tensor, got shape {x.shape}")

    feat = self.features(x)
    flat = torch.flatten(feat, start_dim=1)
    return self.classifier(flat)


# Alias for explicit naming
Conv1DNet = CNN

# List of available model classes
MODELS: list[type[BaseECGModel]] = [
  LogisticRegression,
  RandomForest,
  CNN,
]


def get_model(name: str, **kwargs: Any) -> BaseECGModel:
  """
  Instantiate a model from MODELS by class or snake_case name.
  """
  normalized = name.lower().replace("_", "").replace("-", "")
  for cls in MODELS:
    if cls.__name__.lower() == normalized:
      return cls(**kwargs)
    if cls is CNN and normalized in ("conv1d", "conv1dnet", "cnn1d"):
      return cls(**kwargs)

  valid = [cls.__name__ for cls in MODELS] + ["Conv1DNet"]
  raise ValueError(f"Unknown model '{name}'. Available models: {valid}")


def to_snake_case(name: str) -> str:
  """Convert CamelCase class name to snake_case."""
  s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
  return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s).lower()


def infer_model_name_from_path(checkpoint_path: str | Path) -> str:
  """Infer model architecture name from a checkpoint filename or path."""
  p = str(checkpoint_path).lower()
  stem = Path(checkpoint_path).stem.lower()
  if "random_forest" in stem or stem == "rf":
    return "RandomForest"
  if "logistic_regression" in stem or stem == "lr":
    return "LogisticRegression"
  if "cnn" in stem or "conv" in stem:
    return "CNN"
  if "random_forest" in p:
    return "RandomForest"
  if "logistic_regression" in p:
    return "LogisticRegression"
  if "cnn" in p:
    return "CNN"
  raise ValueError(
    f"Could not infer model architecture from checkpoint path '{checkpoint_path}'. "
    f"Please pass model_name explicitly."
  )


def resolve_checkpoint_path(
  model_name: str | None = None,
  checkpoint_path: str | Path | None = None,
  results_dir: str | Path = DEFAULT_RESULTS_DIR,
) -> Path:
  """
  Resolve the file path to a model checkpoint on disk.

  Resolution strategy:
  1. If `checkpoint_path` is explicitly provided:
     - Check if it exists as an absolute or relative path.
     - Check if it exists relative to the project root (`pyhere.here()`).
     - Check if it exists within `results_dir`.
     - If not found, raise FileNotFoundError.
  2. If `checkpoint_path` is None:
     - Check canonical root locations (e.g. `best_cnn_model.pth` for CNN).
     - Check `results_dir` for `{slug}.pt`, `{slug}.pth`, or `{slug}.joblib`.
     - Check `results/` for `{slug}.pt`, `{slug}.pth`, or `{slug}.joblib`.
     - Check recent run directories under `results/runs/` for matching model files.
     - Return the first existing candidate or raise FileNotFoundError.
  """
  if checkpoint_path is not None:
    p = Path(checkpoint_path)
    if p.exists():
      return p.resolve()
    h = Path(here(checkpoint_path))
    if h.exists():
      return h.resolve()
    res_cand = Path(results_dir) / checkpoint_path
    if res_cand.exists():
      return res_cand.resolve()
    raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' does not exist.")

  if model_name is None:
    raise ValueError("Either model_name or checkpoint_path must be specified.")

  norm_name = model_name.lower().replace("_", "").replace("-", "")
  slug = to_snake_case(model_name)
  res_p = Path(results_dir) if Path(results_dir).is_absolute() else Path(here(results_dir))

  candidates: list[Path] = []

  # 1. Specialized root aliases (highest priority for CNN)
  if norm_name in ("cnn", "conv1d", "conv1dnet", "cnn1d"):
    candidates.append(Path(here("best_cnn_model.pth")))
    candidates.append(res_p / "best_cnn_model.pth")
    slug = "cnn"

  # 2. Results directory candidates
  candidates.extend([
    res_p / f"{slug}.pt",
    res_p / f"{slug}.pth",
    res_p / f"{slug}.joblib",
    Path(here("results", f"{slug}.pt")),
    Path(here("results", f"{slug}.pth")),
    Path(here("results", f"{slug}.joblib")),
  ])

  # 3. Experiment runs fallback (newest run first)
  runs_dir = Path(here("results", "runs"))
  if runs_dir.exists():
    run_folders = sorted(
      [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run")],
      key=lambda x: x.stat().st_mtime,
      reverse=True,
    )
    for rf in run_folders:
      for f in rf.glob("*.*"):
        if f.suffix in (".pt", ".pth", ".joblib") and (
          slug in f.stem.lower() or (slug == "cnn" and "best" in f.stem.lower())
        ):
          candidates.append(f)

  for cand in candidates:
    if cand.exists():
      return cand.resolve()

  raise FileNotFoundError(
    f"Could not locate checkpoint for model '{model_name}'. "
    f"Checked locations: {[str(c) for c in candidates]}"
  )


def load_trained_model(
  model_name: str | None = None,
  checkpoint_path: str | Path | None = None,
  results_dir: str | Path = DEFAULT_RESULTS_DIR,
  device: torch.device | str | None = None,
  input_dim: int = WINDOW_SIZE,
  num_classes: int = NUM_CLASSES,
  seed: int = DEFAULT_RANDOM_STATE,
  **model_kwargs: Any,
) -> BaseECGModel:
  """
  Instantiate and restore a trained ECG model from disk.

  Supports:
  - Classical scikit-learn models (joblib format).
  - Neural PyTorch models (both raw state dicts and training checkpoints with 'model_state_dict').
  - Automatic model name inference from checkpoint path if omitted.
  - Automatically moves neural models to the requested device and invokes `.eval()`.
  - Attaches `checkpoint_path` attribute to the returned model instance.
  """
  resolved_path = resolve_checkpoint_path(
    model_name=model_name,
    checkpoint_path=checkpoint_path,
    results_dir=results_dir,
  )

  if model_name is None:
    model_name = infer_model_name_from_path(resolved_path)

  model = get_model(
    model_name,
    input_dim=input_dim,
    num_classes=num_classes,
    random_state=seed,
    **model_kwargs,
  )

  if isinstance(model, ClassicalECGModel) or resolved_path.suffix == ".joblib":
    loaded = joblib.load(resolved_path)
    if isinstance(loaded, BaseECGModel):
      model = loaded
    elif hasattr(loaded, "predict"):
      model.classifier = loaded
    else:
      raise ValueError(f"Unrecognized object in classical checkpoint '{resolved_path}': {type(loaded)}")
    model.checkpoint_path = resolved_path
    return model

  if isinstance(model, NeuralECGModel):
    target_device = (
      torch.device(device)
      if device is not None
      else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    ckpt = torch.load(str(resolved_path), map_location="cpu", weights_only=False)
    match ckpt:
      case {"model_state_dict": state_dict}:
        model.load_state_dict(state_dict)
      case dict():
        model.load_state_dict(ckpt)
      case _:
        raise ValueError(f"Unrecognized checkpoint format at '{resolved_path}'")

    model.to(target_device)
    model.eval()
    model.checkpoint_path = resolved_path
    return model

  raise TypeError(f"Unsupported model type: {type(model)}")


__all__ = [
  "CNN",
  "MODELS",
  "BaseECGModel",
  "ClassicalECGModel",
  "Conv1DNet",
  "LogisticRegression",
  "NeuralECGModel",
  "RandomForest",
  "get_model",
  "infer_model_name_from_path",
  "load_trained_model",
  "resolve_checkpoint_path",
  "to_snake_case",
]
