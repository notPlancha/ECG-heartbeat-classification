"""
dataset.py
==========
Dataset loader and augmentation module for ECG Heartbeat Classification on Edge-AI.

Features:
1. Flexible `ECGDataset` class compatible with both pure NumPy and PyTorch.
2. Training-only data augmentations:
   - Random temporal jitter / shift (+/- 5 samples).
   - Amplitude scaling (alpha in [0.85, 1.15]).
   - Additive Gaussian noise (sigma = 0.02).
   - Low-frequency baseline wander simulation.
3. Class weight calculation utility to mitigate severe class imbalance.
4. Data loader utility with optional balanced batch sampling.

Author: Course Instructor Solution
Course: ECG Heartbeat Classification on Edge-AI
"""

from collections import Counter
from collections.abc import Callable

import numpy as np

# Try importing torch; if not present, graceful fallback to pure NumPy
try:
  import torch
  from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

  HAS_TORCH = True
except ImportError:
  HAS_TORCH = False
  Dataset = object  # type: ignore


# Class name definitions imported from centralized constants
try:
  from src.constants import CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES
except ImportError:
  from constants import CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES

__all__ = [
  "CLASS_NAMES",
  "CLASS_TO_IDX",
  "IDX_TO_CLASS",
  "NUM_CLASSES",
  "ECGDataset",
  "get_data_loader",
]


class ECGDataset(Dataset):
  """
  Dataset wrapper for ECG heartbeat windows.

  Parameters
  ----------
  X : np.ndarray
      Array of heartbeat waveforms, shape (N, 1, L) or (N, L), float32.
  y : np.ndarray
      Array of integer class labels (0..3).
  record_ids : np.ndarray, optional
      Source record IDs for each sample.
  r_peaks : np.ndarray, optional
      R-peak sample indices.
  symbols : np.ndarray, optional
      Original MIT-BIH symbols.
  augment : bool
      Whether to apply data augmentation (MUST only be True for training!).
  jitter_max : int
      Maximum temporal shift in samples for jitter augmentation (default: 5).
  scale_range : Tuple[float, float]
      Range of random amplitude scaling factor (default: (0.85, 1.15)).
  noise_std : float
      Standard deviation of additive Gaussian noise (default: 0.02).
  transform : callable, optional
      Custom transform function applied to waveform.
  to_torch : bool
      Whether to return PyTorch tensors if PyTorch is installed (default: True).
  """

  def __init__(
    self,
    X: np.ndarray,
    y: np.ndarray,
    record_ids: np.ndarray | None = None,
    r_peaks: np.ndarray | None = None,
    symbols: np.ndarray | None = None,
    augment: bool = False,
    jitter_max: int = 5,
    scale_range: tuple[float, float] = (0.85, 1.15),
    noise_std: float = 0.02,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    to_torch: bool = True,
  ) -> None:
    if X.ndim == 2:
      # Ensure shape is (N, 1, L)
      X = np.expand_dims(X, axis=1)

    self.X = X.astype(np.float32)
    self.y = y.astype(np.int64)
    self.record_ids = record_ids
    self.r_peaks = r_peaks
    self.symbols = symbols
    self.augment = augment
    self.jitter_max = jitter_max
    self.scale_range = scale_range
    self.noise_std = noise_std
    self.transform = transform
    self.to_torch = to_torch and HAS_TORCH

  def __len__(self) -> int:
    return len(self.y)

  def _apply_augmentation(self, beat: np.ndarray) -> np.ndarray:
    """
    Apply physics-grounded data augmentation to a single heartbeat.
    beat shape: (1, L)
    """
    augmented = beat.copy()
    length = augmented.shape[-1]

    # 1. Random temporal shift (jitter)
    if self.jitter_max > 0:
      shift = np.random.randint(-self.jitter_max, self.jitter_max + 1)
      if shift != 0:
        shifted = np.zeros_like(augmented)
        if shift > 0:
          shifted[:, shift:] = augmented[:, : length - shift]
          shifted[:, :shift] = augmented[:, 0:1]  # pad with edge
        else:
          s = abs(shift)
          shifted[:, : length - s] = augmented[:, s:]
          shifted[:, length - s :] = augmented[:, -1:]  # pad with edge
        augmented = shifted

    # 2. Random amplitude scaling
    if self.scale_range is not None:
      scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
      augmented = augmented * scale

    # 3. Additive Gaussian noise
    if self.noise_std > 0:
      noise = np.random.normal(0, self.noise_std, size=augmented.shape)
      augmented = augmented + noise

    # 4. Low-frequency baseline wander simulation
    if np.random.rand() > 0.5:
      # Generate a gentle half-sine wave baseline offset
      freq = np.random.uniform(0.2, 0.8)  # Hz relative
      t = np.linspace(0, 1, length)
      phase = np.random.uniform(0, 2 * np.pi)
      drift_amplitude = np.random.uniform(0.05, 0.15)
      drift = drift_amplitude * np.sin(2 * np.pi * freq * t + phase)
      augmented = augmented + drift

    return augmented.astype(np.float32)

  def __getitem__(
    self, idx: int
  ) -> tuple[np.ndarray | torch.Tensor, int | torch.Tensor]:
    beat = self.X[idx]  # shape: (1, L)
    label = self.y[idx]

    if self.augment:
      beat = self._apply_augmentation(beat)

    if self.transform is not None:
      beat = self.transform(beat)

    if self.to_torch:
      return torch.from_numpy(beat), torch.tensor(label, dtype=torch.long)
    return beat, label

  @classmethod
  def from_npz(
    cls,
    npz_path: str,
    augment: bool = False,
    to_torch: bool = True,
  ) -> ECGDataset:
    """Load an ECGDataset directly from a saved .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    return cls(
      X=data["X"],
      y=data["y"],
      record_ids=data["records"] if "records" in data else None,
      r_peaks=data["r_peaks"] if "r_peaks" in data else None,
      symbols=data["symbols"] if "symbols" in data else None,
      augment=augment,
      to_torch=to_torch,
    )

  def get_class_counts(self) -> dict[str, int]:
    """Return counts of each class in the dataset."""
    cnts = Counter(self.y)
    return {CLASS_NAMES[i]: cnts.get(i, 0) for i in range(4)}

  def compute_class_weights(self, method: str = "inverse") -> np.ndarray:
    """
    Compute class weights to compensate for severe class imbalance.

    Parameters
    ----------
    method : str
        'inverse' : weight = total_samples / (n_classes * class_count)
        'sqrt_inverse' : square root of inverse frequency
        'effective' : based on effective number of samples

    Returns
    -------
    np.ndarray
        1D array of length 4 containing class weights.
    """
    counts = np.bincount(self.y, minlength=4).astype(np.float64)
    total = len(self.y)
    num_classes = 4

    match method:
      case "inverse":
        weights = np.zeros(num_classes, dtype=np.float64)
        for i in range(num_classes):
          weights[i] = total / (num_classes * counts[i]) if counts[i] > 0 else 0.0
      case "sqrt_inverse":
        weights = np.sqrt(total / (num_classes * np.maximum(counts, 1.0)))
      case "effective":
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
        weights = weights / np.sum(weights) * num_classes
      case _:
        raise ValueError(f"Unknown method: {method}")

    return weights.astype(np.float32)


def get_data_loader(
  dataset: ECGDataset,
  batch_size: int = 64,
  shuffle: bool = True,
  balanced_sampling: bool = False,
  num_workers: int = 0,
) -> DataLoader | None:
  """
  Create a PyTorch DataLoader with optional class-balanced sampling.
  """
  if not HAS_TORCH:
    raise RuntimeError(
      "PyTorch is not installed. Install PyTorch or use ECGDataset with pure NumPy."
    )

  sampler = None
  if balanced_sampling:
    class_weights = dataset.compute_class_weights("inverse")
    sample_weights = class_weights[dataset.y]
    sampler = WeightedRandomSampler(
      weights=sample_weights,
      num_samples=len(sample_weights),
      replacement=True,
    )
    shuffle = False  # sampler is mutually exclusive with shuffle

  return DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    sampler=sampler,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
  )
