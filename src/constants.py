"""
constants.py
============
Centralized constants and configuration parameters for ECG Heartbeat
Classification on Edge-AI (Analog Devices MAX78002).

Decoupled from CRISP-DM phase-specific modules (e.g., data_understanding,
prepare_data) to provide a single, clean source of truth across the project.
"""

from pathlib import Path

from pyhere import here

# ==============================================================================
# Standard Dataset Partitioning Constants (AAMI / de Chazal et al., 2004)
# ==============================================================================

# DS1: Development set (22 records used for training, validation, and hyperparameter selection)
DS1_RECORDS: list[str] = [
  "101",
  "106",
  "108",
  "109",
  "112",
  "114",
  "115",
  "116",
  "118",
  "119",
  "122",
  "124",
  "201",
  "203",
  "205",
  "207",
  "208",
  "209",
  "215",
  "220",
  "223",
  "230",
]

# Validation set (5 records): chosen to ensure representation of all 4 classes
DS1_VAL_RECORDS: list[str] = ["101", "106", "108", "118", "205"]

# Training set (17 records): remainder of DS1
DS1_TRAIN_RECORDS: list[str] = [r for r in DS1_RECORDS if r not in DS1_VAL_RECORDS]

# DS2: Final test set (22 records held out strictly for final evaluation)
DS2_TEST_RECORDS: list[str] = [
  "100",
  "103",
  "105",
  "111",
  "113",
  "117",
  "121",
  "123",
  "200",
  "202",
  "210",
  "212",
  "213",
  "214",
  "219",
  "221",
  "222",
  "228",
  "231",
  "232",
  "233",
  "234",
]

# Paced records: excluded according to standard AAMI recommendations
PACED_RECORDS: list[str] = ["102", "104", "107", "217"]

# All 48 records in the MIT-BIH Arrhythmia Database
ALL_RECORDS: list[str] = sorted(DS1_RECORDS + DS2_TEST_RECORDS + PACED_RECORDS)


# ==============================================================================
# AAMI EC57 4-Class Mapping Dictionary
# ==============================================================================

# Mapping from original MIT-BIH annotation symbol to AAMI 4-class (N, S, V, F)
# According to ANSI/AAMI EC57:1998 standard recommendations:
# Class N: Normal / bundle branch block beats
# Class S: Supraventricular ectopic beats
# Class V: Ventricular ectopic beats
# Class F: Fusion beats
AAMI_MAPPING: dict[str, tuple[str, str]] = {
  # Class N: Normal / Bundle Branch Block Beats
  "N": ("N", "Normal beat"),
  "L": ("N", "Left bundle branch block beat"),
  "R": ("N", "Right bundle branch block beat"),
  "e": ("N", "Atrial escape beat"),
  "j": ("N", "Nodal (junctional) escape beat"),
  # Class S: Supraventricular Ectopic Beats (SVEB)
  "A": ("S", "Atrial premature beat"),
  "a": ("S", "Aberrated atrial premature beat"),
  "J": ("S", "Nodal (junctional) premature beat"),
  "S": ("S", "Supraventricular premature beat"),
  # Class V: Ventricular Ectopic Beats (VEB)
  "V": ("V", "Premature ventricular contraction"),
  "E": ("V", "Ventricular escape beat"),
  # Class F: Fusion Beats
  "F": ("F", "Fusion of ventricular and normal beat"),
}

# Excluded symbols: rare paced, unclassifiable beats, and rhythm/signal annotations
EXCLUDED_ANNOTATIONS: dict[str, str] = {
  "/": "Paced beat (excluded: artificial pacemaker morphology)",
  "f": "Fusion of paced and normal beat (excluded: pacemaker artifact)",
  "Q": "Unclassifiable beat (excluded: unknown morphology)",
  "x": "Non-conducted P-wave (excluded: blocked atrial conduction)",
  "+": "Rhythm change marker (non-beat annotation)",
  "~": "Signal quality / noise marker (non-beat annotation)",
  "!": "Ventricular flutter wave (non-beat rhythm marker)",
  "|": "Isolated QRS-like artifact (non-beat annotation)",
  "[": "Measurement / rhythm start mark (non-beat annotation)",
  "]": "Measurement / rhythm end mark (non-beat annotation)",
  '"': "Comment / measurement mark (non-beat annotation)",
}

CLASS_NAMES: list[str] = ["N", "S", "V", "F"]
NUM_CLASSES: int = len(CLASS_NAMES)
CLASS_TO_IDX: dict[str, int] = {
  cls_name: idx for idx, cls_name in enumerate(CLASS_NAMES)
}
IDX_TO_CLASS: dict[int, str] = {
  idx: cls_name for idx, cls_name in enumerate(CLASS_NAMES)
}

# Direct symbol-to-integer class index mapping for all valid AAMI beats
AAMI_SYMBOL_TO_IDX: dict[str, int] = {
  sym: CLASS_TO_IDX[target_class] for sym, (target_class, _) in AAMI_MAPPING.items()
}


# ==============================================================================
# Signal Processing & Segmentation Constants (MAX78002 Constraints)
# ==============================================================================

SAMPLING_RATE: int = 360  # Hz (native MIT-BIH rate)
DEFAULT_PRE_R: int = 90  # 250 ms before R-peak (P-wave and PR interval)
DEFAULT_POST_R: int = 166  # 461 ms after R-peak (QRS, ST, T-wave)
WINDOW_SIZE: int = (
  DEFAULT_PRE_R + DEFAULT_POST_R
)  # 256 = 2^8 samples for dyadic pooling

BANDPASS_LOWCUT: float = 0.5  # Hz
BANDPASS_HIGHCUT: float = 40.0  # Hz
BANDPASS_ORDER: int = 3

NORMALIZATION_CLIP_MIN: float = -3.0
NORMALIZATION_CLIP_MAX: float = 3.0
PREFERRED_LEAD: str = "MLII"


# ==============================================================================
# Dataset Directory Resolution (Anchored via pyhere)
# ==============================================================================

DEFAULT_DATA_DIR: str = str(here("dataset", "mit-bih-arrhythmia-database-1.0.0"))


def resolve_data_dir(data_dir: str | None = None, search_depth: int = 2) -> str:
  """
  Resolve the MIT-BIH dataset directory path.

  Searches for an existing directory containing MIT-BIH header files (*.hea)
  using pyhere to anchor paths to the project root regardless of the caller's
  working directory.

  Parameters
  ----------
  data_dir : str | None
      Explicit path hint. If provided and contains .hea files or exists, it is prioritized.
  search_depth : int
      Maximum recursion depth inside dataset folders (default: 2).

  Returns
  -------
  str
      Resolved path to directory containing MIT-BIH records.
  """
  if data_dir is not None:
    p = Path(data_dir)
    if p.exists() and p.is_dir() and any(p.glob("*.hea")):
      return str(p)
    p_here = here(data_dir)
    if p_here.exists() and p_here.is_dir() and any(p_here.glob("*.hea")):
      return str(p_here)

  candidates = [
    here("dataset", "mit-bih-arrhythmia-database-1.0.0"),
    here("dataset"),
    Path.cwd() / "dataset",
  ]

  for base in candidates:
    if not base.exists() or not base.is_dir():
      continue
    if any(base.glob("*.hea")):
      return str(base)
    level1_dirs = [p for p in base.iterdir() if p.is_dir()]
    for p1 in level1_dirs:
      if any(p1.glob("*.hea")):
        return str(p1)
    if search_depth >= 2:
      for p1 in level1_dirs:
        try:
          level2_dirs = [p for p in p1.iterdir() if p.is_dir()]
        except PermissionError, OSError:
          continue
        for p2 in level2_dirs:
          if any(p2.glob("*.hea")):
            return str(p2)

    deeper = next(base.rglob("*.hea"), None)
    if deeper is not None:
      return str(deeper.parent)

  return str(here("dataset", "mit-bih-arrhythmia-database-1.0.0"))


# ==============================================================================
# Pipeline Directory & File Path Defaults (Anchored via pyhere)
# ==============================================================================

DEFAULT_PROCESSED_DIR: str = str(here("dataset", "processed"))
DEFAULT_TRAIN_PATH: str = str(here("dataset", "processed", "train.npz"))
DEFAULT_VAL_PATH: str = str(here("dataset", "processed", "val.npz"))
DEFAULT_TEST_PATH: str = str(here("dataset", "processed", "test.npz"))

DEFAULT_RESULTS_DIR: str = str(here("results"))
DEFAULT_MODELS_DIR: str = str(here("results", "models"))
DEFAULT_FIGURES_DIR: str = str(here("results", "figures"))
DEFAULT_TABLES_DIR: str = str(here("results", "tables"))
DEFAULT_RUNS_DIR: str = str(here("results", "runs"))


# ==============================================================================
# Model & Training Defaults
# ==============================================================================

DEFAULT_EPOCHS: int = 15
DEFAULT_LR: float = 0.01
DEFAULT_OPTIMIZER: str = "sgd"
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_CLASS_WEIGHT: str = "balanced"
DEFAULT_RANDOM_STATE: int = 42
DEFAULT_SEED: int = DEFAULT_RANDOM_STATE

DEFAULT_RF_N_ESTIMATORS: int = 100
DEFAULT_RF_MAX_DEPTH: int = 20
DEFAULT_RF_MIN_SAMPLES_LEAF: int = 2
DEFAULT_RF_MAX_FEATURES: str = "sqrt"

DEFAULT_RESAMPLING_STRATEGY: str = "none"
RESAMPLING_STRATEGIES: tuple[str, ...] = (
  "none",
  "hybrid",
  "oversample",
  "undersample",
)


def set_seed(seed: int = DEFAULT_RANDOM_STATE) -> None:
  """
  Configure global seeds across Python, NumPy, and PyTorch for deterministic execution.

  Parameters
  ----------
  seed : int
      Random seed integer (default: 42).
  """
  import os
  import random

  import numpy as np

  os.environ["PYTHONHASHSEED"] = str(seed)
  random.seed(seed)
  np.random.seed(seed)

  try:
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False
  except ImportError:
    pass
