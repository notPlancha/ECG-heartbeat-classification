import torch
from torch import nn

try:
  from src.constants import NUM_CLASSES, WINDOW_SIZE
except ImportError:
  from constants import NUM_CLASSES, WINDOW_SIZE


class LogisticRegression(nn.Module):
  """
  Multiclass Logistic Regression baseline model for ECG heartbeat classification.

  Maps a 1D ECG window (default: 256 samples) directly to class logits
  (default: 4 classes: N, S, V, F) via a single linear transformation.
  """

  def __init__(
    self,
    input_dim: int = WINDOW_SIZE,
    num_classes: int = NUM_CLASSES,
    bias: bool = True,
    in_features: int | None = None,
  ) -> None:
    super().__init__()
    if in_features is not None:
      input_dim = in_features
    self.input_dim = input_dim
    self.in_features = input_dim
    self.num_classes = num_classes
    self.linear = nn.Linear(input_dim, num_classes, bias=bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if x.ndim > 2:
      x = x.flatten(start_dim=1)
    elif x.ndim == 1:
      x = x.unsqueeze(0)
    return self.linear(x)

  @torch.no_grad()
  def predict(self, x: torch.Tensor) -> torch.Tensor:
    logits = self.forward(x)
    return torch.argmax(logits, dim=-1)

  @torch.no_grad()
  def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
    logits = self.forward(x)
    return torch.softmax(logits, dim=-1)


# Alias for backward compatibility or explicit naming
LogisticRegressionBaseline = LogisticRegression

__all__ = ["LogisticRegression", "LogisticRegressionBaseline"]
