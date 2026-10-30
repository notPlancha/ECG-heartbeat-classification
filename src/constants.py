"""
constants.py
============
Centralized constants and configuration parameters for ECG Heartbeat
Classification on Edge-AI (Analog Devices MAX78002).

Decoupled from CRISP-DM phase-specific modules (e.g., data_understanding,
prepare_data) to provide a single, clean source of truth across the project.
"""

import os
from pathlib import Path

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
# Dataset Directory Resolution
# ==============================================================================

DEFAULT_DATA_DIR: str = os.path.join("dataset", "mit-bih-arrhythmia-database-1.0.0")


def resolve_data_dir(data_dir: str | None = None, search_depth: int = 2) -> str:
  """
  Resolve the MIT-BIH dataset directory path.

  Searches for an existing directory containing MIT-BIH header files (*.hea).
  Inspects candidate root directories and recursively checks subdirectories up
  to `search_depth` levels deep (default: 2). Works whether invoked from the
  workspace root, src/, notebooks/, or other subdirectories.

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
  # 1. If an explicit path was given and exists, check if valid
  if data_dir is not None:
    p = Path(data_dir)
    if p.exists() and p.is_dir() and any(p.glob("*.hea")):
      return str(p)
    # Also check relative to parent (e.g. from notebooks/)
    p_rel = Path("..") / data_dir
    if p_rel.exists() and p_rel.is_dir() and any(p_rel.glob("*.hea")):
      return str(p_rel)

  # 2. Candidate dataset root folders to search from
  cwd = Path.cwd()
  repo_root = Path(__file__).resolve().parent.parent

  base_candidates = [
    cwd / "dataset",
    cwd.parent / "dataset",
    repo_root / "dataset",
  ]

  # Deduplicate while preserving order
  seen: set[Path] = set()
  unique_candidates: list[Path] = []
  for b in base_candidates:
    resolved = b.resolve()
    if resolved not in seen:
      seen.add(resolved)
      unique_candidates.append(b)

  def _to_rel(target: Path) -> str:
    try:
      return os.path.relpath(str(target), str(cwd))
    except ValueError:
      return str(target)

  # 3. Recursively search candidates up to search_depth
  for base in unique_candidates:
    if not base.exists() or not base.is_dir():
      continue

    # Level 0: base itself
    if any(base.glob("*.hea")):
      return _to_rel(base)

    # Level 1: base/*
    level1_dirs = [p for p in base.iterdir() if p.is_dir()]
    for p1 in level1_dirs:
      if any(p1.glob("*.hea")):
        return _to_rel(p1)

    # Level 2: base/*/*
    if search_depth >= 2:
      for p1 in level1_dirs:
        try:
          level2_dirs = [p for p in p1.iterdir() if p.is_dir()]
        except PermissionError, OSError:
          continue
        for p2 in level2_dirs:
          if any(p2.glob("*.hea")):
            return _to_rel(p2)

    # Fallback: deeper search if nothing found up to search_depth
    deeper_match = next(base.rglob("*.hea"), None)
    if deeper_match is not None:
      return _to_rel(deeper_match.parent)

  # 4. Fallback to default path relative to cwd where possible
  default_path = repo_root / DEFAULT_DATA_DIR
  return _to_rel(default_path)
