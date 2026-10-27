import sys
from pathlib import Path

import numpy as np
import wfdb

# Add project root to sys.path if not present
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.constants import (
    AAMI_SYMBOL_TO_IDX,
    DS1_RECORDS,
    DS2_TEST_RECORDS,
    NUM_CLASSES,
    resolve_data_dir,
)

DATA_DIR = Path(resolve_data_dir())
num_classes = NUM_CLASSES
LABEL_MAP = AAMI_SYMBOL_TO_IDX
TRAINING_DATA = DS1_RECORDS
TESTING_DATA = DS2_TEST_RECORDS

# Load all heartbeats from the training records
def _load_records(record_ids):
    all_beats = []
    all_annotations = []

    for record_id in record_ids:
        X, y = _load_record(record_id)

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
def _load_record(record_id):
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

# public functions
def load_training_data():
    return _load_records(TRAINING_DATA)

def load_testing_data():
    return _load_records(TESTING_DATA)


if __name__ == "__main__":
    X_train, y_train = load_training_data()