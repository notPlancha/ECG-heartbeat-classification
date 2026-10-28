import torch
from torch import nn

try:
  from src.constants import NUM_CLASSES, WINDOW_SIZE
except ImportError:
  from constants import NUM_CLASSES, WINDOW_SIZE


class LogisticRegression(nn.Module):
  def __init__(
    self,
    input_dim: int = WINDOW_SIZE,
    num_classes: int = NUM_CLASSES,
    bias: bool = True,
  ) -> None:
    super().__init__()
    self.input_dim = input_dim
    self.num_classes = num_classes
    self.linear = nn.Linear(input_dim, num_classes, bias=bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:

    if x.ndim > 2:
      x = x.flatten(start_dim=1)

    return self.linear(x)


# Alias for backward compatibility or explicit naming
LogisticRegressionBaseline = LogisticRegression

__all__ = ["LogisticRegression", "LogisticRegressionBaseline"]
