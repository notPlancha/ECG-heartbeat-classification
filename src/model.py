"""
model.py
========
Model definitions for ECG heartbeat classification.
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


class BaseECGModel(nn.Module, ABC):
  """
  Abstract base class for neural ECG classification models.
  """

  def __init__(
    self,
    input_dim: int = WINDOW_SIZE,
    num_classes: int = NUM_CLASSES,
  ) -> None:
    super().__init__()
    self.input_dim = input_dim
    self.num_classes = num_classes

  @abstractmethod
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Compute raw class logits."""

  @torch.no_grad()
  def predict(self, x: torch.Tensor) -> torch.Tensor:
    logits = self.forward(x)
    return torch.argmax(logits, dim=-1)

  @torch.no_grad()
  def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
    logits = self.forward(x)
    return torch.softmax(logits, dim=-1)


class LogisticRegression(BaseECGModel):
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


class RandomForest:
  """
  Random Forest baseline for ECG heartbeat classification.
  """

  def __init__(
    self,
    n_estimators: int = 100,
    max_depth: int | None = 20,
    min_samples_leaf: int = 2,
    max_features: str | float = "sqrt",
    class_weight: str | dict[int, float] | None = "balanced",
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs: Any,
  ) -> None:
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

  def fit(self, x: np.ndarray, y: np.ndarray) -> RandomForest:
    if isinstance(x, torch.Tensor):
      x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
      y = y.detach().cpu().numpy()
    if x.ndim == 3 and x.shape[1] == 1:
      x = x.squeeze(1)
    elif x.ndim > 2:
      x = x.reshape(len(x), -1)
    self.classifier.fit(x, y)
    return self

  def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
      x = x.detach().cpu().numpy()
    if x.ndim == 3 and x.shape[1] == 1:
      x = x.squeeze(1)
    elif x.ndim > 2:
      x = x.reshape(len(x), -1)
    return self.classifier.predict(x)

  def predict_proba(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
      x = x.detach().cpu().numpy()
    if x.ndim == 3 and x.shape[1] == 1:
      x = x.squeeze(1)
    elif x.ndim > 2:
      x = x.reshape(len(x), -1)
    return self.classifier.predict_proba(x)

  def count_parameters(self) -> int:
    """Return total number of nodes across all decision trees."""
    if hasattr(self.classifier, "estimators_"):
      return sum(tree.tree_.node_count for tree in self.classifier.estimators_)
    return 0


# List of available model classes
MODELS: list[type] = [
  LogisticRegression,
  RandomForest,
]


def get_model(name: str, **kwargs: Any) -> Any:
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
  "LogisticRegression",
  "RandomForest",
  "get_model",
]
