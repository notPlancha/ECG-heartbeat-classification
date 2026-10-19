from pathlib import Path

import numpy as np
import wfdb

# Set root path
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "mit-bih-arrhythmia-database-1.0.0"

num_classes = 4

# Class labeling
LABEL_MAP = {
    # Normal
    "N": 0,
    "L": 0,
    "R": 0,
    "e": 0,
    "j": 0,

    # Supraventricular ectopic
    "A": 1,
    "a": 1,
    "J": 1,
    "S": 1,

    # Ventricular ectopic
    "V": 2,
    "E": 2,

    # Fusion
    "F": 3,
}

# Training set
TRAINING_DATA = [
    "101", "106", "108", "109", "112", "114",
    "115", "116", "118", "119", "122", "124",
    "201", "203", "205", "207", "208", "209",
    "215", "220", "223", "230",
]

# Testing set
TESTING_DATA = [
    "100", "103", "105", "111", "113", "117",
    "121", "123", "200", "202", "210", "212",
    "213", "214", "219", "221", "222", "228",
    "231", "232", "233", "234",
]


# Load all heartbeats from the training records
def load_records():
    all_beats = []
    all_annotations = []

    for record_id in TRAINING_DATA:
        X, y = load_record(record_id)

        all_beats.extend(X)
        all_annotations.extend(y)

        print(f"Loaded record {record_id}: {len(y)} beats")

    # Convert lists into NumPy arrays
    X = np.asarray(all_beats, dtype=np.float32)
    y = np.asarray(all_annotations, dtype=np.int64)

    print("\nTraining dataset summary")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Total beats:", len(y))
    print("Class counts:", np.bincount(y, minlength=num_classes))

    return X, y

# Load one record and extract its annotated heartbeats
def load_record(record_id):
    BEFORE = 90
    AFTER = 166
    WINDOW_SIZE = BEFORE + AFTER

    record_path = DATA_DIR / record_id

    # Read ECG signal and annotations
    record = wfdb.rdrecord(str(record_path))
    annotation = wfdb.rdann(str(record_path), "atr")

    # Use the first ECG channel
    signal = record.p_signal[:, 0]

    extracted_beats = []
    extracted_annotations = []

    for sample, symbol in zip(
        annotation.sample, annotation.symbol
    ):
        # Skip annotations not in our four-class mapping
        if symbol not in LABEL_MAP:
            continue

        start = sample - BEFORE
        end = sample + AFTER

        # Skip beats too close to signal boundaries
        if start < 0 or end > len(signal):
            continue

        beat = signal[start:end]

        if len(beat) != WINDOW_SIZE:
            continue

        extracted_beats.append(beat)
        extracted_annotations.append(LABEL_MAP[symbol])

    # Return after processing ALL annotations in this record
    return extracted_beats, extracted_annotations


if __name__ == "__main__":
    X_train, y_train = load_records()
