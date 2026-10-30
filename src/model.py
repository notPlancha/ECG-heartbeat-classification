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
  from src.constants import NUM_CLASSES, WINDOW_SIZE
except ImportError:
  from constants import NUM_CLASSES, WINDOW_SIZE


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
    if isinstance(x, np.ndarray):
      x = torch.from_numpy(x).float()
    logits = self.forward(x)
    return torch.argmax(logits, dim=-1).cpu().numpy()

  @torch.no_grad()
  def predict_proba(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, np.ndarray):
      x = torch.from_numpy(x).float()
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
    if x.ndim == 3 and x.shape[1] == 1:
      return x.squeeze(1)
    if x.ndim > 2:
      return x.reshape(len(x), -1)
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
    **kwargs: Any,
  ) -> None:
    super().__init__(input_dim=input_dim, num_classes=num_classes)
    self.linear = nn.Linear(input_dim, num_classes, bias=bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.ndim > 2:
      x = x.flatten(start_dim=1)
    elif x.ndim == 1:
      x = x.unsqueeze(0)
    return self.linear(x)


class RandomForest(ClassicalECGModel):
  """
  Random Forest baseline for ECG heartbeat classification.
  """

  def __init__(
    self,
    input_dim: int = WINDOW_SIZE,
    num_classes: int = NUM_CLASSES,
    n_estimators: int = 100,
    max_depth: int | None = 20,
    min_samples_leaf: int = 2,
    max_features: str | float = "sqrt",
    class_weight: str | dict[int, float] | None = "balanced",
    random_state: int = 42,
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
    self.classifier = RandomForestClassifier(
      n_estimators=n_estimators,
      max_depth=max_depth,
      min_samples_leaf=min_samples_leaf,
      max_features=max_features,
      class_weight=class_weight,
      random_state=random_state,
      n_jobs=n_jobs,
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


# List of available model classes
MODELS: list[type[BaseECGModel]] = [
  LogisticRegression,
  RandomForest,
]


def get_model(name: str, **kwargs: Any) -> BaseECGModel:
  """
  Instantiate a model from MODELS by class or snake_case name.
  """
  normalized = name.lower().replace("_", "").replace("-", "")
  for cls in MODELS:
    if cls.__name__.lower() == normalized:
      return cls(**kwargs)

  valid = [cls.__name__ for cls in MODELS]
  raise ValueError(f"Unknown model '{name}'. Available models: {valid}")


__all__ = [
  "MODELS",
  "BaseECGModel",
  "ClassicalECGModel",
  "LogisticRegression",
  "NeuralECGModel",
  "RandomForest",
  "get_model",
]
