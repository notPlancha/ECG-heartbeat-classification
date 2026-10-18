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

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import utilities from src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
  all_windows = []
  all_labels = []
  all_record_ids = []
  all_r_peaks = []
  all_symbols = []
  total_exclusions: Counter = Counter()
  per_record_stats = []

  for r in tqdm(records, desc="Processing records", leave=False):
    # 1. Load record
    rec_data = du.load_record(r, data_dir=data_dir)

    # 2. Extract preferred lead (e.g. MLII)
    lead_signal, actual_lead, channel_idx = du.extract_lead_signal(
      rec_data, preferred_lead=preferred_lead
    )

    # 3. Apply zero-phase bandpass filter
    filtered_sig = du.butter_bandpass_filter(
      lead_signal,
      lowcut=lowcut,
      highcut=highcut,
      fs=fs,
      order=filter_order,
    )

    # 4. Segment heartbeats
    windows, labels, symbols, r_peaks, ex = du.segment_heartbeats(
      signal=filtered_sig,
      ann_indices=rec_data.ann_indices,
      ann_symbols=rec_data.ann_symbols,
      record_name=r,
      pre_r=pre_r,
      post_r=post_r,
      normalize=normalize,
      clip_range=clip_range,
    )

    total_exclusions += ex

    # Per-record class breakdown
    rec_counts = Counter(labels)
    per_record_stats.append(
      {
        "Record": r,
        "Lead": actual_lead,
        "Channel_Index": channel_idx,
        "N": rec_counts.get(du.CLASS_TO_IDX["N"], 0),
        "S": rec_counts.get(du.CLASS_TO_IDX["S"], 0),
        "V": rec_counts.get(du.CLASS_TO_IDX["V"], 0),
        "F": rec_counts.get(du.CLASS_TO_IDX["F"], 0),
        "Total_Beats": len(labels),
        "Excluded_Count": sum(ex.values()),
      }
    )

    if len(windows) > 0:
      all_windows.append(windows)
      all_labels.append(labels)
      all_record_ids.extend([r] * len(labels))
      all_r_peaks.append(r_peaks)
      all_symbols.extend(symbols)

  if len(all_windows) > 0:
    X = np.concatenate(all_windows, axis=0)
    # Reshape to (N, 1, L) for 1D CNN / MAX78002 accelerator convention
    X = np.expand_dims(X, axis=1)
    y = np.concatenate(all_labels, axis=0)
    r_peaks_arr = np.concatenate(all_r_peaks, axis=0)
    rec_ids_arr = np.array(all_record_ids, dtype=object)
  else:
    window_size = pre_r + post_r
    X = np.empty((0, 1, window_size), dtype=np.float32)
    y = np.empty((0,), dtype=np.int64)
    r_peaks_arr = np.empty((0,), dtype=np.int64)
    rec_ids_arr = np.empty((0,), dtype=object)

  return X, y, rec_ids_arr, r_peaks_arr, all_symbols, total_exclusions, per_record_stats


def run_pipeline(
  data_dir: str = du.DEFAULT_DATA_DIR,
  output_dir: str = "dataset/processed",
  results_dir: str = "results",
  pre_r: int = 90,
  post_r: int = 166,
  lowcut: float = 0.5,
  highcut: float = 40.0,
  fs: float = 360.0,
  filter_order: int = 3,
  clip_min: float = -3.0,
  clip_max: float = 3.0,
) -> dict[str, dict]:
  """
  Run the end-to-end dataset preparation pipeline.
  """
  os.makedirs(output_dir, exist_ok=True)
  os.makedirs(results_dir, exist_ok=True)

  print("=" * 70)
  print("ECG HEARTBEAT CLASSIFICATION — DATA PREPARATION PIPELINE")
  print("=" * 70)
  print(f"Dataset directory : {data_dir}")
  print(f"Output directory  : {output_dir}")
  print(f"Results directory : {results_dir}")
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
    print(f"\nProcessing '{split_name}' split ({len(rec_list)} records)...")
    X, y, rec_ids, r_peaks, symbols, exclusions, per_rec = process_record_list(
      records=rec_list,
      data_dir=data_dir,
      pre_r=pre_r,
      post_r=post_r,
      lowcut=lowcut,
      highcut=highcut,
      fs=fs,
      filter_order=filter_order,
      normalize=True,
      clip_range=(clip_min, clip_max),
      preferred_lead="MLII",
    )

    for p in per_rec:
      p["Split"] = split_name
      all_per_record_stats.append(p)

    # Save compressed NPZ file
    save_path = os.path.join(output_dir, f"{split_name}.npz")
    np.savez_compressed(
      save_path,
      X=X,
      y=y,
      records=rec_ids,
      r_peaks=r_peaks,
      symbols=np.array(symbols, dtype=object),
    )
    print(f"  -> Saved {save_path} (X: {X.shape}, y: {y.shape})")

    # Class counts
    cnts = Counter(y)
    split_results[split_name] = {
      "records": rec_list,
      "total_samples": len(y),
      "N": int(cnts.get(0, 0)),
      "S": int(cnts.get(1, 0)),
      "V": int(cnts.get(2, 0)),
      "F": int(cnts.get(3, 0)),
      "exclusions": dict(exclusions),
    }

  # Summary table across splits (CRISP-DM Section 3.8 Table)
  summary_rows = []
  for split_name in ["train", "val", "test"]:
    res = split_results[split_name]
    summary_rows.append(
      {
        "Split": split_name.capitalize(),
        "Record Count": len(res["records"]),
        "Records": ", ".join(res["records"]),
        "N": res["N"],
        "S": res["S"],
        "V": res["V"],
        "F": res["F"],
        "Total": res["total_samples"],
      }
    )

  summary_df = pd.DataFrame(summary_rows)

  # Add total row
  total_row = {
    "Split": "Total",
    "Record Count": summary_df["Record Count"].sum(),
    "Records": "All 44 DS1+DS2 records",
    "N": summary_df["N"].sum(),
    "S": summary_df["S"].sum(),
    "V": summary_df["V"].sum(),
    "F": summary_df["F"].sum(),
    "Total": summary_df["Total"].sum(),
  }
  summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)

  print("\n" + "=" * 70)
  print("FINAL DATASET SPLIT SUMMARY (CRISP-DM Section 3.8)")
  print("=" * 70)
  print(summary_df.to_string(index=False))
  print("=" * 70)

  # Save summary tables to results
  summary_csv_path = os.path.join(results_dir, "dataset_statistics.csv")
  summary_df.to_csv(summary_csv_path, index=False)
  print(f"\nSaved dataset summary to: {summary_csv_path}")

  stats_json_path = os.path.join(results_dir, "dataset_statistics.json")
  with open(stats_json_path, "w", encoding="utf-8") as f:
    json.dump(split_results, f, indent=2)
  print(f"Saved detailed JSON statistics to: {stats_json_path}")

  # Per-record breakdown
  per_rec_df = pd.DataFrame(all_per_record_stats)
  per_rec_csv_path = os.path.join(results_dir, "per_record_distribution.csv")
  per_rec_df.to_csv(per_rec_csv_path, index=False)
  print(f"Saved per-record distribution to: {per_rec_csv_path}")

  # Data Quality Assessment across all records
  print("\nRunning signal quality audit across all records...")
  quality_rows = []
  for r in du.ALL_RECORDS:
    rec = du.load_record(r, data_dir=data_dir)
    sig, lead, idx = du.extract_lead_signal(rec, "MLII")
    q = du.assess_signal_quality(sig, fs=fs)
    q["Record"] = r
    q["Lead"] = lead
    quality_rows.append(q)

  quality_df = pd.DataFrame(quality_rows)
  # Reorder columns
  cols = ["Record", "Lead"] + [
    c for c in quality_df.columns if c not in ["Record", "Lead"]
  ]
  quality_df = quality_df[cols]
  quality_csv_path = os.path.join(results_dir, "data_quality_report.csv")
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
    default="dataset/processed",
    help="Path to save processed .npz files",
  )
  parser.add_argument(
    "--results-dir",
    type=str,
    default="results",
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
  )
