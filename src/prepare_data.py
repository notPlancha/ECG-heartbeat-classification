"""
prepare_data.py
===============
Reproducible data preparation pipeline for CRISP-DM Stage 3: Data Preparation.

Executes:
1. Patient-independent splitting (Train: 17 records, Val: 5 records, Test: 22 records).
2. MLII lead extraction with channel-swap handling (e.g. record 114).
3. 0.5 - 40.0 Hz zero-phase Butterworth bandpass filtering.
4. Fixed-size beat segmentation around R-peaks (256 samples: 90 pre, 166 post).
5. Per-beat Z-score normalization and [-3.0, 3.0] outlier clipping.
6. AAMI 4-class mapping (N, S, V, F) and exclusion tracking.
7. Export of compressed datasets (.npz) and statistical reports (.csv, .json).

Author: Course Instructor Solution
Course: ECG Heartbeat Classification on Edge-AI
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from pyhere import here
from tqdm import tqdm

# Import utilities from src
sys.path.insert(0, str(here("src")))
import data_understanding as du


def process_record_list(
  records: list[str],
  data_dir: str,
  pre_r: int = 90,
  post_r: int = 166,
  lowcut: float = 0.5,
  highcut: float = 40.0,
  fs: float = 360.0,
  filter_order: int = 3,
  normalize: bool = True,
  clip_range: tuple[float, float] = (-3.0, 3.0),
  preferred_lead: str = "MLII",
) -> tuple[
  np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], Counter, list[dict]
]:
  """
  Process a list of records: filter, segment beats, normalize, and collect arrays.
  """
  all_beats = []
  all_labels = []
  all_r_peaks = []
  all_symbols = []
  all_records = []
  class_counter: Counter = Counter()
  per_record_stats = []

  window_size = pre_r + post_r

  for r in tqdm(records, desc="Processing records"):
    rec = du.load_record(r, data_dir=data_dir)
    sig, lead, _ = du.extract_lead_signal(rec, preferred_lead=preferred_lead)
    filtered = du.bandpass_filter(
      sig, lowcut=lowcut, highcut=highcut, fs=fs, order=filter_order
    )
    ann = du.load_annotation(r, data_dir=data_dir)

    beats, labels, r_peaks, symbols = du.segment_beats(
      filtered,
      ann,
      pre_r=pre_r,
      post_r=post_r,
      normalize=normalize,
      clip_range=clip_range,
    )

    rec_counts = Counter(labels)
    per_record_stats.append(
      {
        "Record": r,
        "Lead": lead,
        "Total_Beats": len(labels),
        "N": rec_counts.get(0, 0),
        "S": rec_counts.get(1, 0),
        "V": rec_counts.get(2, 0),
        "F": rec_counts.get(3, 0),
      }
    )

    for b, lb, rp, sym in zip(beats, labels, r_peaks, symbols, strict=False):
      all_beats.append(b)
      all_labels.append(lb)
      all_r_peaks.append(rp)
      all_symbols.append(sym)
      all_records.append(r)
      class_counter[du.IDX_TO_CLASS[lb]] += 1

  if all_beats:
    X = np.stack(all_beats, axis=0).astype(np.float32)
    # Ensure channel dimension: (N, 1, window_size)
    if X.ndim == 2:
      X = np.expand_dims(X, axis=1)
    y = np.array(all_labels, dtype=np.int64)
    r_peaks_arr = np.array(all_r_peaks, dtype=np.int64)
    symbols_arr = np.array(all_symbols)
  else:
    X = np.empty((0, 1, window_size), dtype=np.float32)
    y = np.empty((0,), dtype=np.int64)
    r_peaks_arr = np.empty((0,), dtype=np.int64)
    symbols_arr = np.empty((0,), dtype=str)

  return (
    X,
    y,
    r_peaks_arr,
    symbols_arr,
    all_records,
    class_counter,
    per_record_stats,
  )


def run_pipeline(
  data_dir: str = du.DEFAULT_DATA_DIR,
  output_dir: str = du.DEFAULT_PROCESSED_DIR,
  results_dir: str = du.DEFAULT_RESULTS_DIR,
  pre_r: int = 90,
  post_r: int = 166,
  lowcut: float = 0.5,
  highcut: float = 40.0,
  fs: float = 360.0,
  filter_order: int = 3,
  clip_min: float = -3.0,
  clip_max: float = 3.0,
  seed: int = du.DEFAULT_RANDOM_STATE,
) -> dict[str, dict]:
  """
  Run the end-to-end dataset preparation pipeline.
  """
  du.set_seed(seed)
  data_dir = du.resolve_data_dir(data_dir)
  output_p = Path(output_dir) if Path(output_dir).is_absolute() else here(output_dir)
  results_p = (
    Path(results_dir) if Path(results_dir).is_absolute() else here(results_dir)
  )

  output_p.mkdir(parents=True, exist_ok=True)
  results_p.mkdir(parents=True, exist_ok=True)

  print("=" * 70)
  print("ECG HEARTBEAT CLASSIFICATION — DATA PREPARATION PIPELINE")
  print("=" * 70)
  print(f"Dataset directory : {data_dir}")
  print(f"Output directory  : {output_p}")
  print(f"Results directory : {results_p}")
  print(
    f"Window size       : {pre_r + post_r} samples (Pre: {pre_r}, Post: {post_r}) @ {fs} Hz"
  )
  print(f"Bandpass filter   : {lowcut} Hz to {highcut} Hz (order {filter_order})")
  print(f"Clipping range    : [{clip_min}, {clip_max}] std devs")
  print("=" * 70)

  splits = {
    "train": du.DS1_TRAIN_RECORDS,
    "val": du.DS1_VAL_RECORDS,
    "test": du.DS2_TEST_RECORDS,
  }

  split_results = {}
  all_per_record_stats = []

  for split_name, rec_list in splits.items():
    print(f"\nProcessing {split_name.upper()} split ({len(rec_list)} records)...")
    X, y, r_peaks, symbols, recs, class_cnt, per_rec = process_record_list(
      rec_list,
      data_dir=data_dir,
      pre_r=pre_r,
      post_r=post_r,
      lowcut=lowcut,
      highcut=highcut,
      fs=fs,
      filter_order=filter_order,
      normalize=True,
      clip_range=(clip_min, clip_max),
      preferred_lead=du.PREFERRED_LEAD,
    )

    all_per_record_stats.extend(per_rec)

    # Save to compressed .npz
    save_path = output_p / f"{split_name}.npz"
    np.savez_compressed(
      save_path,
      X=X,
      y=y,
      r_peaks=r_peaks,
      symbols=symbols,
      records=np.array(recs),
    )
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"Saved {save_path} ({file_size_mb:.2f} MB)")
    print(f"  Shape: X={X.shape}, y={y.shape}")
    print(f"  Counts: {dict(class_cnt)}")

    split_results[split_name] = {
      "records": rec_list,
      "num_records": len(rec_list),
      "total_beats": len(y),
      "class_distribution": dict(class_cnt),
      "shape_X": list(X.shape),
      "shape_y": list(y.shape),
      "file_path": str(save_path),
      "file_size_mb": round(file_size_mb, 2),
    }

  # Build official Section 3.8 Table
  rows = []
  for split_name in ["train", "val", "test"]:
    res = split_results[split_name]
    counts = res["class_distribution"]
    rows.append(
      {
        "Split": split_name.capitalize(),
        "Records": res["num_records"],
        "N (Normal)": counts.get("N", 0),
        "S (Supraventricular)": counts.get("S", 0),
        "V (Ventricular)": counts.get("V", 0),
        "F (Fusion)": counts.get("F", 0),
        "Total": res["total_beats"],
      }
    )

  summary_df = pd.DataFrame(rows)
  total_row = {
    "Split": "Total",
    "Records": summary_df["Records"].sum(),
    "N (Normal)": summary_df["N (Normal)"].sum(),
    "S (Supraventricular)": summary_df["S (Supraventricular)"].sum(),
    "V (Ventricular)": summary_df["V (Ventricular)"].sum(),
    "F (Fusion)": summary_df["F (Fusion)"].sum(),
    "Total": summary_df["Total"].sum(),
  }
  summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)

  print("\n" + "=" * 70)
  print("FINAL DATASET SPLIT SUMMARY (CRISP-DM Section 3.8)")
  print("=" * 70)
  print(summary_df.to_string(index=False))
  print("=" * 70)

  # Save summary tables to results
  summary_csv_path = results_p / "dataset_statistics.csv"
  summary_df.to_csv(summary_csv_path, index=False)
  print(f"\nSaved dataset summary to: {summary_csv_path}")

  stats_json_path = results_p / "dataset_statistics.json"
  with open(stats_json_path, "w", encoding="utf-8") as f:
    json.dump(split_results, f, indent=2)
  print(f"Saved detailed JSON statistics to: {stats_json_path}")

  # Per-record breakdown
  per_rec_df = pd.DataFrame(all_per_record_stats)
  per_rec_csv_path = results_p / "per_record_distribution.csv"
  per_rec_df.to_csv(per_rec_csv_path, index=False)
  print(f"Saved per-record distribution to: {per_rec_csv_path}")

  # Data Quality Assessment across all records
  print("\nRunning signal quality audit across all records...")
  quality_rows = []
  for r in du.ALL_RECORDS:
    rec = du.load_record(r, data_dir=data_dir)
    sig, lead, _idx = du.extract_lead_signal(rec, "MLII")
    q = du.assess_signal_quality(sig, fs=fs)
    q["Record"] = r
    q["Lead"] = lead
    quality_rows.append(q)

  quality_df = pd.DataFrame(quality_rows)
  cols = ["Record", "Lead"] + [
    c for c in quality_df.columns if c not in ["Record", "Lead"]
  ]
  quality_df = quality_df[cols]
  quality_csv_path = results_p / "data_quality_report.csv"
  quality_df.to_csv(quality_csv_path, index=False)
  print(f"Saved data quality report to: {quality_csv_path}")

  print("\nData preparation pipeline completed successfully!")
  return split_results


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="MIT-BIH ECG Data Preparation Pipeline for MAX78002"
  )
  parser.add_argument(
    "--data-dir",
    type=str,
    default=du.DEFAULT_DATA_DIR,
    help="Path to raw MIT-BIH dataset",
  )
  parser.add_argument(
    "--output-dir",
    type=str,
    default=du.DEFAULT_PROCESSED_DIR,
    help="Path to save processed .npz files",
  )
  parser.add_argument(
    "--results-dir",
    type=str,
    default=du.DEFAULT_RESULTS_DIR,
    help="Path to save output metrics and tables",
  )
  parser.add_argument(
    "--pre-r", type=int, default=90, help="Samples before R-peak (default: 90 = 250ms)"
  )
  parser.add_argument(
    "--post-r",
    type=int,
    default=166,
    help="Samples after R-peak (default: 166 = 461ms)",
  )
  parser.add_argument(
    "--lowcut", type=float, default=0.5, help="Bandpass lower cutoff (Hz)"
  )
  parser.add_argument(
    "--highcut", type=float, default=40.0, help="Bandpass upper cutoff (Hz)"
  )
  parser.add_argument(
    "--clip-min", type=float, default=-3.0, help="Min Z-score clipping limit"
  )
  parser.add_argument(
    "--clip-max", type=float, default=3.0, help="Max Z-score clipping limit"
  )
  parser.add_argument(
    "--seed",
    type=int,
    default=du.DEFAULT_RANDOM_STATE,
    help=f"Random seed for reproducibility (default: {du.DEFAULT_RANDOM_STATE})",
  )
  args = parser.parse_args()

  run_pipeline(
    data_dir=args.data_dir,
    output_dir=args.output_dir,
    results_dir=args.results_dir,
    pre_r=args.pre_r,
    post_r=args.post_r,
    lowcut=args.lowcut,
    highcut=args.highcut,
    clip_min=args.clip_min,
    clip_max=args.clip_max,
    seed=args.seed,
  )
