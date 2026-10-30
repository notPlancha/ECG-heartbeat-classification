import torch
from torch import nn

try:
  from src.constants import NUM_CLASSES, WINDOW_SIZE
except ImportError:
  from constants import NUM_CLASSES, WINDOW_SIZE


class LogisticRegression(nn.Module):
  """
  Multiclass linear classifier for ECG heartbeat classification.

  Maps a 1D ECG window directly to class logits:
      logits = x * W^T + b
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


# Alias for backward compatibility
LogisticRegressionBaseline = LogisticRegression

# Registry for model lookup by name
MODEL_REGISTRY: dict[str, type[nn.Module]] = {
  "logistic_regression": LogisticRegression,
  "logistic_regression_baseline": LogisticRegression,
  "baseline": LogisticRegression,
  "linear": LogisticRegression,
}


def register_model(name: str, model_cls: type[nn.Module]) -> None:
  """Add a new model class to the registry."""
  key = name.lower().replace("-", "_")
  MODEL_REGISTRY[key] = model_cls


def list_models() -> list[str]:
  """Return unique registered model names."""
  seen = set()
  result = []
  for name, cls in MODEL_REGISTRY.items():
    if cls not in seen:
      seen.add(cls)
      result.append(name)
  return sorted(result)


def get_model(name: str, **kwargs) -> nn.Module:
  """
  Instantiate a model by name.

  Raises ValueError if the model name is not recognized.
  """
  key = name.lower().replace("-", "_")
  if key not in MODEL_REGISTRY:
    valid = ", ".join(sorted(MODEL_REGISTRY.keys()))
    raise ValueError(f"Unknown model '{name}'. Available models: {valid}")
  return MODEL_REGISTRY[key](**kwargs)


__all__ = [
  "MODEL_REGISTRY",
  "LogisticRegression",
  "LogisticRegressionBaseline",
  "get_model",
  "list_models",
  "register_model",
]
