"""
model.py
========
Model definitions and base classes for ECG heartbeat classification.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from torch import nn

try:
  from src.constants import (
    DEFAULT_CLASS_WEIGHT,
    DEFAULT_RANDOM_STATE,
    DEFAULT_RF_MAX_DEPTH,
    DEFAULT_RF_MAX_FEATURES,
    DEFAULT_RF_MIN_SAMPLES_LEAF,
    DEFAULT_RF_N_ESTIMATORS,
    NUM_CLASSES,
    WINDOW_SIZE,
  )
except ImportError:
  from constants import (
    DEFAULT_CLASS_WEIGHT,
    DEFAULT_RANDOM_STATE,
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
  def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
    """Predict discrete class labels for input heartbeats."""

  @abstractmethod
  def predict_proba(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
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
  def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
    try:
      device = next(self.parameters()).device
    except StopIteration:
      device = torch.device("cpu")
    if isinstance(x, np.ndarray):
      x = torch.from_numpy(x).float().to(device)
    elif isinstance(x, torch.Tensor) and x.device != device:
      x = x.to(device)
    logits = self.forward(x)
    return torch.argmax(logits, dim=-1).cpu().numpy()

  @torch.no_grad()
  def predict_proba(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
    try:
      device = next(self.parameters()).device
    except StopIteration:
      device = torch.device("cpu")
    if isinstance(x, np.ndarray):
      x = torch.from_numpy(x).float().to(device)
    elif isinstance(x, torch.Tensor) and x.device != device:
      x = x.to(device)
    logits = self.forward(x)
    return torch.softmax(logits, dim=-1).cpu().numpy()

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

  def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
    x_prep = self.prepare_inputs(x)
    return self.classifier.predict(x_prep)

  def predict_proba(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
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
    random_state: int | None = DEFAULT_RANDOM_STATE,
    **kwargs: Any,
  ) -> None:
    if random_state is not None:
      torch.manual_seed(random_state)
    super().__init__(input_dim=input_dim, num_classes=num_classes)
    self.random_state = random_state
    self.in_channels = in_channels

    self.features = nn.Sequential(
      nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.MaxPool1d(kernel_size=2),
      nn.Conv1d(16, 32, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.MaxPool1d(kernel_size=2),
      nn.Conv1d(32, 64, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool1d(kernel_size=2),
    )

    reduced_dim = input_dim // 8
    self.classifier = nn.Linear(64 * reduced_dim, num_classes)

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
]
