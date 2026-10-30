"""
model.py
========
Model definitions for ECG heartbeat classification.
"""

from abc import ABC, abstractmethod

import torch
from torch import nn

try:
  from src.constants import NUM_CLASSES, WINDOW_SIZE
except ImportError:
  from constants import NUM_CLASSES, WINDOW_SIZE


class BaseECGModel(nn.Module, ABC):
  """
  Abstract base class for ECG classification models.
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
  ) -> None:
    super().__init__(input_dim=input_dim, num_classes=num_classes)
    self.linear = nn.Linear(input_dim, num_classes, bias=bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.ndim > 2:
      x = x.flatten(start_dim=1)
    elif x.ndim == 1:
      x = x.unsqueeze(0)
    return self.linear(x)


# List of available model classes
MODELS: list[type[BaseECGModel]] = [
  LogisticRegression,
]


def get_model(name: str, **kwargs) -> BaseECGModel:
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
  "get_model",
]
