"""
verify_consistency.py
=====================
Automated Sanity and Synchronization Linter for CRISP-DM Reporting.

Asserts:
1. All required models exist in results/metrics.csv.
2. CNN Macro F1 in metrics.csv matches mean of per-class F1 in per_class_metrics.csv.
3. Confusion matrix diagonal sum matches overall test accuracy.
4. Noise results (Clean condition) match clean benchmark metrics.
5. Quantization report (FP32 & INT8) matches metrics.csv.
6. Report (report.typ) contains exact matching figures.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from pyhere import here


def verify_pipeline_consistency() -> bool:
    print("=" * 70)
    print("PIPELINE CONSISTENCY & REPORT SYNCHRONIZATION AUDIT")
    print("=" * 70)

    # 1. Metrics CSV Check
    metrics_path = Path(here("results", "metrics.csv"))
    if not metrics_path.exists():
        print(f"[FAIL] Missing {metrics_path}", file=sys.stderr)
        return False
    metrics_df = pd.read_csv(metrics_path).set_index("Model")
    print("[PASS] results/metrics.csv loaded.")

    required_models = ["CNN", "RandomForest", "LogisticRegression", "Quantized_CNN_INT8"]
    for m in required_models:
        if m not in metrics_df.index:
            print(f"[FAIL] Model '{m}' missing from metrics.csv", file=sys.stderr)
            return False
    print(f"[PASS] All {len(required_models)} models documented in metrics.csv.")

    cnn_acc = metrics_df.loc["CNN", "Accuracy"]
    cnn_f1 = metrics_df.loc["CNN", "Macro_F1"]
    cnn_rec = metrics_df.loc["CNN", "Macro_Recall"]

    # 2. Per-Class Metrics Check
    per_class_path = Path(here("results", "per_class_metrics.csv"))
    if not per_class_path.exists():
        print(f"[FAIL] Missing {per_class_path}", file=sys.stderr)
        return False
    per_class_df = pd.read_csv(per_class_path)
    if not np.isclose(cnn_f1, per_class_df["F1"].mean(), atol=1e-4):
        print(
            f"[FAIL] CNN Macro F1 ({cnn_f1:.4f}) != mean per-class F1 ({per_class_df['F1'].mean():.4f})",
            file=sys.stderr,
        )
        return False
    if not np.isclose(cnn_rec, per_class_df["Recall_Sensitivity"].mean(), atol=1e-4):
        print(
            f"[FAIL] CNN Macro Recall ({cnn_rec:.4f}) != mean per-class Recall ({per_class_df['Recall_Sensitivity'].mean():.4f})",
            file=sys.stderr,
        )
        return False
    print("[PASS] results/per_class_metrics.csv matches metrics.csv.")

    # 3. Confusion Matrix Check
    cm_path = Path(here("results", "confusion_matrix.csv"))
    if not cm_path.exists():
        print(f"[FAIL] Missing {cm_path}", file=sys.stderr)
        return False
    cm_df = pd.read_csv(cm_path, index_col=0)
    cm_acc = np.trace(cm_df.values) / cm_df.values.sum()
    if not np.isclose(cnn_acc, cm_acc, atol=1e-4):
        print(
            f"[FAIL] CNN Accuracy ({cnn_acc:.4f}) != confusion matrix accuracy ({cm_acc:.4f})",
            file=sys.stderr,
        )
        return False
    print("[PASS] results/confusion_matrix.csv diagonal matches CNN accuracy.")

    # 4. Noise Results Check
    noise_path = Path(here("results", "noise_results.csv"))
    if not noise_path.exists():
        print(f"[FAIL] Missing {noise_path}", file=sys.stderr)
        return False
    noise_df = pd.read_csv(noise_path)
    clean_matches = noise_df[(noise_df["Model"] == "CNN") & (noise_df["Condition"] == "Clean")]
    if len(clean_matches) == 0:
        print("[FAIL] Missing Clean condition for CNN in noise_results.csv", file=sys.stderr)
        return False
    clean_row = clean_matches.iloc[0]
    if not np.isclose(cnn_acc, clean_row["Accuracy"], atol=1e-4) or not np.isclose(cnn_f1, clean_row["Macro_F1"], atol=1e-4):
        print("[FAIL] Noise Clean condition does not match metrics.csv", file=sys.stderr)
        return False
    print("[PASS] results/noise_results.csv Clean condition matches metrics.csv.")

    # 5. Quantization Report Check
    quant_path = Path(here("max78002", "quantization_output", "quantization_report.json"))
    if not quant_path.exists():
        print(f"[FAIL] Missing {quant_path}", file=sys.stderr)
        return False
    with open(quant_path, "r", encoding="utf-8") as f:
        quant_data = json.load(f)
    if not np.isclose(cnn_acc, quant_data["fp32_model"]["accuracy"], atol=1e-4):
        print("[FAIL] Quantization FP32 accuracy mismatch", file=sys.stderr)
        return False
    int8_acc = metrics_df.loc["Quantized_CNN_INT8", "Accuracy"]
    int8_f1 = metrics_df.loc["Quantized_CNN_INT8", "Macro_F1"]
    if not np.isclose(int8_acc, quant_data["int8_model"]["accuracy"], atol=1e-4) or not np.isclose(int8_f1, quant_data["int8_model"]["macro_f1"], atol=1e-4):
        print("[FAIL] Quantization INT8 metrics mismatch", file=sys.stderr)
        return False
    print("[PASS] max78002/quantization_output/quantization_report.json matches metrics.csv.")

    # 6. Report.typ Content Check
    report_path = Path(here("report.typ"))
    if report_path.exists():
        report_text = report_path.read_text(encoding="utf-8")
        acc_str = f"{cnn_acc*100:.2f}%"
        f1_str = f"{cnn_f1:.4f}"
        if acc_str not in report_text:
            print(f"[FAIL] Target accuracy '{acc_str}' not found in report.typ", file=sys.stderr)
            return False
        if f1_str not in report_text:
            print(f"[FAIL] Target Macro F1 '{f1_str}' not found in report.typ", file=sys.stderr)
            return False
        print(f"[PASS] report.typ contains synchronized metrics (Acc: {acc_str}, F1: {f1_str}).")
    else:
        print("[INFO] report.typ omitted from archive; skipping report synchronization check.")

    print("=" * 70)
    print("ALL INTEGRATION ASSERTIONS PASSED! Repository is fully synchronized.")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = verify_pipeline_consistency()
    sys.exit(0 if success else 1)
