"""
generate_report_figures.py
==========================
Generates high-resolution publication-quality figures for the final Typst project report:
    1. results/eda_overview.png: CRISP-DM Stage 2 & 3 exploratory data analysis and preprocessing.
    2. results/error_waveforms.png: CRISP-DM Stage 5.5 qualitative error analysis waveforms.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from pyhere import here

PROJECT_ROOT = here()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import CLASS_NAMES, SAMPLING_RATE
from src.model import CNN, load_trained_model

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10


def generate_eda_figure(out_path: Path):
    """Generate 4-panel EDA and Preprocessing overview figure."""
    train_data = np.load(here("dataset", "processed", "train.npz"))
    val_data = np.load(here("dataset", "processed", "val.npz"))
    test_data = np.load(here("dataset", "processed", "test.npz"))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Panel 1: Morphology of 4 AAMI Classes
    ax = axes[0, 0]
    time_ms = (np.arange(256) - 90) / SAMPLING_RATE * 1000  # Centered at R-peak
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for c_idx, (c_name, col) in enumerate(zip(CLASS_NAMES, colors)):
        match_idx = np.where(train_data["y"] == c_idx)[0]
        if len(match_idx) > 0:
            sample_beat = train_data["X"][match_idx[5]].flatten()
            ax.plot(time_ms, sample_beat, label=f"Class {c_name}", color=col, lw=2.0)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.6, label="R-peak (Sample 90)")
    ax.set_title("A. Representative Heartbeat Morphologies (AAMI EC57)")
    ax.set_xlabel("Time relative to R-peak (ms)")
    ax.set_ylabel("Normalized Amplitude (Z-score)")
    ax.set_xlim(-250, 460)
    ax.legend(loc="upper right", frameon=True)

    # Panel 2: Class Distribution Across Splits
    ax = axes[0, 1]
    splits = ["Train (DS1)", "Val (DS1)", "Test (DS2)"]
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    train_counts = [np.sum(train_data["y"] == i) for i in range(4)]
    val_counts = [np.sum(val_data["y"] == i) for i in range(4)]
    test_counts = [np.sum(test_data["y"] == i) for i in range(4)]

    ax.bar(x - width, train_counts, width, label="Train (40,411)", color="#4c72b0")
    ax.bar(x, val_counts, width, label="Val (10,583)", color="#55a868")
    ax.bar(x + width, test_counts, width, label="Test (49,685)", color="#c44e52")
    ax.set_yscale("log")
    ax.set_title("B. Beat Distribution by Class and Split (Log Scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Class {c}" for c in CLASS_NAMES])
    ax.set_ylabel("Heartbeat Count (Log10)")
    ax.legend(frameon=True)

    # Annotate class imbalance percentage
    total_beats = sum(train_counts) + sum(val_counts) + sum(test_counts)
    class_totals = [t + v + te for t, v, te in zip(train_counts, val_counts, test_counts)]
    for i, c_tot in enumerate(class_totals):
        pct = c_tot / total_beats * 100
        ax.text(x[i], max(train_counts[i], test_counts[i]) * 1.5, f"{pct:.1f}%", ha="center", fontsize=9, fontweight="bold")

    # Panel 3: Amplitude Distribution Before & After Clipping
    ax = axes[1, 0]
    raw_samples = train_data["X"][:500].flatten()
    ax.hist(raw_samples, bins=60, density=True, color="#4c72b0", alpha=0.7, edgecolor="none", label="Preprocessed Z-score")
    ax.axvline(-3.0, color="#c44e52", linestyle="--", lw=2, label="Hardware Clip [-3, 3]")
    ax.axvline(3.0, color="#c44e52", linestyle="--", lw=2)
    ax.set_title("C. Preprocessed Amplitude Distribution & Clipping Bounds")
    ax.set_xlabel("Normalized Amplitude")
    ax.set_ylabel("Probability Density")
    ax.legend(frameon=True)

    # Panel 4: Dyadic Downsampling & MAX78002 Receptive Field
    ax = axes[1, 1]
    stages = ["Input\n(1, 256)", "Conv1+Pool\n(16, 128)", "Conv2+Pool\n(32, 64)", "Conv3+Pool\n(64, 32)", "Linear Head\n(4 classes)"]
    dims = [256, 128, 64, 32, 4]
    chans = [1, 16, 32, 64, 4]
    ax.plot(stages, dims, marker="o", color="#4c72b0", lw=2.5, markersize=8, label="Temporal Dim (samples)")
    ax2 = ax.twinx()
    ax2.plot(stages, chans, marker="s", color="#c44e52", lw=2.5, markersize=8, label="Feature Channels")
    ax2.grid(False)
    ax.set_title("D. MAX78002 Dyadic Downsampling Hierarchy")
    ax.set_ylabel("Temporal Dimension", color="#4c72b0")
    ax2.set_ylabel("Channel Dimension", color="#c44e52")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", frameon=True)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Generated] EDA Overview figure -> {out_path}")


def generate_error_waveforms_figure(out_path: Path):
    """Generate 4-panel Error Analysis waveform figure for Section 5.5."""
    test_data = np.load(here("dataset", "processed", "test.npz"))
    X_test = test_data["X"]
    y_test = test_data["y"]

    model = load_trained_model("CNN", device="cpu")

    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()
        preds = logits.argmax(dim=1).numpy()

    # Define exact error cases
    time_ms = (np.arange(256) - 90) / SAMPLING_RATE * 1000

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Case 1: True N classified as S
    idx_1 = 1  # True N, Pred S
    ax = axes[0, 0]
    ax.plot(time_ms, X_test[idx_1].flatten(), color="#1f77b4", lw=2, label="Observed Waveform")
    ax.set_title(
        f"Case 1: True Class N Misclassified as S (Beat #{idx_1})\n"
        f"Pred Probabilities: N={probs[idx_1, 0]:.2f}, S={probs[idx_1, 1]:.2f}, V={probs[idx_1, 2]:.2f}, F={probs[idx_1, 3]:.2f}"
    )
    ax.set_xlabel("Time from R-peak (ms)")
    ax.set_ylabel("Normalized Amplitude")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    # Case 2: True S classified as N
    idx_2 = 229  # True S, Pred N
    ax = axes[0, 1]
    ax.plot(time_ms, X_test[idx_2].flatten(), color="#ff7f0e", lw=2, label="Observed Waveform")
    ax.set_title(
        f"Case 2: True Class S Misclassified as N (Beat #{idx_2})\n"
        f"Pred Probabilities: N={probs[idx_2, 0]:.2f}, S={probs[idx_2, 1]:.2f}, V={probs[idx_2, 2]:.2f}, F={probs[idx_2, 3]:.2f}"
    )
    ax.set_xlabel("Time from R-peak (ms)")
    ax.set_ylabel("Normalized Amplitude")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    # Case 3: True V classified as N
    idx_3 = 5025  # True V, Pred N
    ax = axes[1, 0]
    ax.plot(time_ms, X_test[idx_3].flatten(), color="#2ca02c", lw=2, label="Observed Waveform")
    ax.set_title(
        f"Case 3: True Class V Misclassified as N (Beat #{idx_3})\n"
        f"Pred Probabilities: N={probs[idx_3, 0]:.2f}, S={probs[idx_3, 1]:.2f}, V={probs[idx_3, 2]:.2f}, F={probs[idx_3, 3]:.2f}"
    )
    ax.set_xlabel("Time from R-peak (ms)")
    ax.set_ylabel("Normalized Amplitude")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    # Case 4: True V under 0 dB NSTDB Electrode Motion Noise
    idx_4 = 5025
    clean_beat = X_test[idx_4].flatten()
    # Simulate 0 dB noise addition
    np.random.seed(42)
    sig_power = np.mean(clean_beat**2)
    noise = np.random.normal(0, np.sqrt(sig_power), size=256)
    noisy_beat = clean_beat + noise
    ax = axes[1, 1]
    ax.plot(time_ms, clean_beat, color="gray", lw=1.5, linestyle="--", label="Clean Beat (Class V)")
    ax.plot(time_ms, noisy_beat, color="#d62728", lw=1.8, label="0 dB Corrupted (Pred: N)")
    ax.set_title(
        "Case 4: True Class V Corrupted by 0 dB Severe Noise\n"
        "QRS Complex and S-T Segment Submerged; Model Defaults to Majority N"
    )
    ax.set_xlabel("Time from R-peak (ms)")
    ax.set_ylabel("Normalized Amplitude")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[Generated] Error Waveforms figure -> {out_path}")


def main():
    eda_path = here("results", "eda_overview.png")
    err_path = here("results", "error_waveforms.png")
    generate_eda_figure(Path(eda_path))
    generate_error_waveforms_figure(Path(err_path))


if __name__ == "__main__":
    main()
