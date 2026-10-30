"""
purge.py
========
Simple cleanup utility for model checkpoints and experiment runs in results/.
"""

import argparse
import shutil
from pathlib import Path

from pyhere import here

RESULTS_DIR = here("results")
RUNS_DIR = RESULTS_DIR / "runs"


def purge_models(results_dir: Path = RESULTS_DIR) -> None:
  """Remove all model checkpoint files (.pt, .joblib) from results and runs."""
  patterns = ("*.pt", "*.joblib")
  removed = 0
  for pattern in patterns:
    for file_path in results_dir.rglob(pattern):
      if file_path.is_file():
        file_path.unlink()
        print(f"Removed: {file_path}")
        removed += 1
  print(f"Purged {removed} model checkpoint(s).")


def purge_all(results_dir: Path = RESULTS_DIR, runs_dir: Path = RUNS_DIR) -> None:
  """Remove all run directories and generated outputs, keeping .gitkeep."""
  # Clean out results/runs/ directory contents except .gitkeep
  if runs_dir.exists():
    for item in runs_dir.iterdir():
      if item.name == ".gitkeep":
        continue
      if item.is_dir():
        shutil.rmtree(item)
      else:
        item.unlink()
      print(f"Removed run artifact: {item}")
  else:
    runs_dir.mkdir(parents=True, exist_ok=True)

  gitkeep = runs_dir / ".gitkeep"
  if not gitkeep.exists():
    gitkeep.touch()

  # Clean transient generated root result files
  for pattern in (
    "*.pt",
    "*.joblib",
    "*_training_history.csv",
    "*_per_class_metrics.csv",
    "*_confusion_matrix.*",
  ):
    for file_path in results_dir.glob(pattern):
      if file_path.is_file():
        file_path.unlink()
        print(f"Removed root artifact: {file_path}")

  print("Purged all experiment runs and transient outputs (preserved .gitkeep).")


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Purge model checkpoints or all experiment runs from results/."
  )
  parser.add_argument(
    "--all",
    "-a",
    dest="all",
    action="store_true",
    help="Purge all run directories and result outputs (keeping .gitkeep).",
  )
  parser.add_argument(
    "mode",
    nargs="?",
    choices=["models", "all"],
    default=None,
    help="Optional positional target: 'models' (default) or 'all'.",
  )
  args = parser.parse_args()

  if args.all or args.mode == "all":
    purge_all()
  else:
    purge_models()


if __name__ == "__main__":
  main()
