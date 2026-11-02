
from pathlib import Path

import numpy as np
import wfdb
import pywt
import matplotlib.pyplot as plt


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "dataset" / "mit-bih-arrhythmia-database-1.0.0"

num_classes = 4

BEFORE = 85
AFTER = 100
WINDOW_SIZE = BEFORE + 1 + AFTER


# --------------------------------------------------
# Label mapping
# --------------------------------------------------

LABEL_MAP = {
    "N": 0,
    "L": 0,
    "R": 0,
    "e": 0,
    "j": 0,

    "A": 1,
    "a": 1,
    "J": 1,
    "S": 1,

    "V": 2,
    "E": 2,

    "F": 3,
}


# --------------------------------------------------
# Training and testing records
# --------------------------------------------------

TRAINING_DATA = [
    "101", "106", "108", "109", "112", "114",
    "115", "116", "118", "119", "122", "124",
    "201", "203", "205", "207", "208", "209",
    "215", "220", "223", "230",
]

TESTING_DATA = [
    "100", "103", "105", "111", "113", "117",
    "121", "123", "200", "202", "210", "212",
    "213", "214", "219", "221", "222", "228",
    "231", "232", "233", "234",
]


# --------------------------------------------------
# Wavelet denoising
# --------------------------------------------------

def _denoise_signal(signal):
    """
    Decompose the ECG signal using Daubechies 6 (db6)
    and 9 decomposition levels.

    Retain:
        A9, D9, D8, D7, D6, D5, D4, D3

    Remove:
        D2, D1
    """

    wavelet = "db6"
    level = 9

    coefficients = pywt.wavedec(
        signal,
        wavelet,
        level=level
    )

    # Remove detail component D1
    coefficients[-1] = np.zeros_like(coefficients[-1])

    # Remove detail component D2
    coefficients[-2] = np.zeros_like(coefficients[-2])

    # Reconstruct the signal
    denoised_signal = pywt.waverec(
        coefficients,
        wavelet
    )

    return denoised_signal[:len(signal)]


# --------------------------------------------------
# Softmax transformation
# --------------------------------------------------

def _softmax_heartbeat(heartbeat):
    """
    Apply Softmax to an individual heartbeat.

    The resulting values are positive and sum to 1.
    """

    # Subtract maximum for numerical stability
    shifted_heartbeat = heartbeat - np.max(heartbeat)

    exponentials = np.exp(shifted_heartbeat)

    softmax = exponentials / np.sum(exponentials)

    return softmax


# --------------------------------------------------
# Load one record
# --------------------------------------------------

def _load_record(record_id):

    record_path = DATA_DIR / record_id

    record = wfdb.rdrecord(str(record_path))
    annotation = wfdb.rdann(str(record_path), "atr")

    # Use the first ECG channel
    signal = record.p_signal[:, 0]

    # Denoise the complete ECG signal
    signal = _denoise_signal(signal)

    extracted_beats = []
    extracted_annotations = []

    for sample, symbol in zip(
        annotation.sample,
        annotation.symbol
    ):

        # Ignore annotations outside our four target classes
        if symbol not in LABEL_MAP:
            continue

        # Determine heartbeat window
        start = sample - BEFORE
        end = sample + AFTER + 1

        # Skip incomplete heartbeat windows
        if start < 0 or end > len(signal):
            continue

        beat = signal[start:end]

        # Verify heartbeat window length
        if len(beat) != WINDOW_SIZE:
            continue

        # Apply Softmax to the heartbeat
        beat = _softmax_heartbeat(beat)

        extracted_beats.append(beat)
        extracted_annotations.append(LABEL_MAP[symbol])

    return extracted_beats, extracted_annotations


# --------------------------------------------------
# Load multiple records
# --------------------------------------------------

def _load_records(record_ids):

    all_beats = []
    all_annotations = []

    for record_id in record_ids:

        X, y = _load_record(record_id)

        all_beats.extend(X)
        all_annotations.extend(y)

        print(f"Loaded record {record_id}: {len(y)} beats")

    X = np.asarray(all_beats, dtype=np.float32)
    y = np.asarray(all_annotations, dtype=np.int64)

    print("\nDataset summary")
    print("--------------------------")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Total beats:", len(y))
    print("Class counts:", np.bincount(y, minlength=num_classes))

    print("\nSoftmax check")
    print("--------------------------")
    print("Minimum value:", X.min())
    print("Maximum value:", X.max())
    print("First heartbeat sum:", X[0].sum())

    return X, y


# --------------------------------------------------
# Public training and testing loaders
# --------------------------------------------------

def load_training_data():

    return _load_records(TRAINING_DATA)


def load_testing_data():

    return _load_records(TESTING_DATA)


# --------------------------------------------------
# Visualize original, denoised and Softmax heartbeat
# --------------------------------------------------

def show_denoising_example(record_id="101", beat_index=0):

    record_path = DATA_DIR / record_id

    # Load original ECG record and annotations
    record = wfdb.rdrecord(str(record_path))
    annotation = wfdb.rdann(str(record_path), "atr")

    original_signal = record.p_signal[:, 0]

    # Denoise the complete signal
    denoised_signal = _denoise_signal(original_signal)

    # Find valid heartbeat annotations
    valid_beats = []

    for sample, symbol in zip(
        annotation.sample,
        annotation.symbol
    ):

        if symbol not in LABEL_MAP:
            continue

        start = sample - BEFORE
        end = sample + AFTER + 1

        if start < 0 or end > len(original_signal):
            continue

        valid_beats.append((sample, symbol))

    if beat_index < 0 or beat_index >= len(valid_beats):
        raise ValueError(
            f"Beat index {beat_index} is unavailable. "
            f"Record {record_id} contains {len(valid_beats)} valid beats."
        )

    # Select heartbeat
    sample, symbol = valid_beats[beat_index]

    start = sample - BEFORE
    end = sample + AFTER + 1

    original_beat = original_signal[start:end]
    denoised_beat = denoised_signal[start:end]

    # Apply Softmax
    softmax_beat = _softmax_heartbeat(denoised_beat)

    # Convert sample positions to milliseconds
    time_axis = (
        np.arange(-BEFORE, AFTER + 1) / record.fs
    ) * 1000

    # Create three plots
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 10),
        sharex=True
    )

    # Original heartbeat
    axes[0].plot(time_axis, original_beat)
    axes[0].set_title("Original ECG Heartbeat")
    axes[0].set_ylabel("Amplitude (mV)")
    axes[0].grid(True)

    # Denoised heartbeat
    axes[1].plot(time_axis, denoised_beat)
    axes[1].set_title("Denoised ECG Heartbeat")
    axes[1].set_ylabel("Amplitude (mV)")
    axes[1].grid(True)

    # Softmax heartbeat
    axes[2].plot(time_axis, softmax_beat)
    axes[2].set_title("Softmax-transformed ECG Heartbeat")
    axes[2].set_ylabel("Softmax value")
    axes[2].set_xlabel("Time relative to annotation (ms)")
    axes[2].grid(True)

    # Mark annotation location
    for ax in axes:
        ax.axvline(
            x=0,
            color="red",
            linestyle="--",
            label="Annotated beat location"
        )
        ax.legend()

    fig.suptitle(
        f"ECG Processing - Record {record_id}, "
        f"Annotation {symbol}",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()


# --------------------------------------------------
# Run example when executing this file directly
# --------------------------------------------------

if __name__ == "__main__":

    show_denoising_example(
        record_id="101",
        beat_index=0
    )
