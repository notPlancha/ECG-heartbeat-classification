
import warnings
from pathlib import Path

import numpy as np
import wfdb
import pywt


# ============================================================
# PATH CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT
    / "dataset"
    / "mit-bih-arrhythmia-database-1.0.0"
)

OUTPUT_DIR = ROOT / "ml-model" / "processed_data"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLING_RATE = 360

BEFORE_R = 85
AFTER_R = 100

BEAT_LENGTH = BEFORE_R + AFTER_R + 1


# ============================================================
# TRAINING AND TESTING RECORDS
# ============================================================

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


# ============================================================
# ECG CLASS CONFIGURATION
# Q CLASS REMOVED
# ============================================================

CLASSES = ["N", "S", "V", "F"]

CLASS_TO_INDEX = {
    name: i for i, name in enumerate(CLASSES)
}


ANNOTATION_MAPPING = {

    # N: Normal
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",

    # S: Supraventricular
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",

    # V: Ventricular
    "V": "V",
    "E": "V",

    # F: Fusion
    "F": "F",

    # Q annotations intentionally excluded
}


# ============================================================
# 1. LOAD ECG RECORD
# ============================================================

def load_ecg_record(record_path):

    record = wfdb.rdrecord(str(record_path))

    annotation = wfdb.rdann(
        str(record_path),
        "atr"
    )

    signal_names = [
        name.upper().strip()
        for name in record.sig_name
    ]

    if "MLII" in signal_names:

        lead_index = signal_names.index("MLII")

    else:

        warnings.warn(
            f"MLII not found in {record_path}. "
            "Using first available lead."
        )

        lead_index = 0

    ecg = record.p_signal[:, lead_index]

    return (
        ecg.astype(np.float32),
        annotation.sample,
        annotation.symbol
    )


# ============================================================
# 2. WAVELET DENOISING
# ============================================================

def wavelet_denoise(signal):

    wavelet = "db6"
    level = 9

    coeffs = pywt.wavedec(
        signal,
        wavelet,
        level=level
    )

    denoised_coeffs = []

    for i, coeff in enumerate(coeffs):

        if i <= 7:

            denoised_coeffs.append(coeff)

        else:

            denoised_coeffs.append(
                np.zeros_like(coeff)
            )

    reconstructed = pywt.waverec(
        denoised_coeffs,
        wavelet
    )

    return reconstructed[:len(signal)].astype(
        np.float32
    )


# ============================================================
# 3. HEARTBEAT EXTRACTION
# ============================================================

def extract_heartbeats(
    ecg,
    r_peaks,
    symbols
):

    beats = []
    labels = []

    for peak, symbol in zip(r_peaks, symbols):

        # Ignore annotations outside the four classes
        if symbol not in ANNOTATION_MAPPING:
            continue

        start = peak - BEFORE_R

        end = peak + AFTER_R + 1

        if start < 0 or end > len(ecg):
            continue

        beat = ecg[start:end]

        if len(beat) != BEAT_LENGTH:
            continue

        label = ANNOTATION_MAPPING[symbol]

        beats.append(beat)

        labels.append(
            CLASS_TO_INDEX[label]
        )

    return (
        np.asarray(beats, dtype=np.float32),
        np.asarray(labels, dtype=np.int64)
    )


# ============================================================
# 4. NORMALIZATION
# ============================================================

def normalize_heartbeat(
    beat,
    eps=1e-8
):

    minimum = np.min(beat)

    maximum = np.max(beat)

    normalized = (
        (beat - minimum) /
        (maximum - minimum + eps)
    )

    return normalized.astype(np.float32)


def normalize_dataset(beats):

    return np.stack([
        normalize_heartbeat(beat)
        for beat in beats
    ]).astype(np.float32)


# ============================================================
# 5. PROCESS RECORDINGS
# ============================================================

def process_records(
    record_paths,
    denoise=True
):

    X_list = []
    y_list = []
    record_ids = []

    for record_path in record_paths:

        record_name = Path(record_path).name

        print(
            f"Processing {record_name} | "
            f"Denoising: {denoise}"
        )

        try:

            ecg, peaks, symbols = load_ecg_record(
                record_path
            )

            if denoise:

                ecg = wavelet_denoise(ecg)

            beats, labels = extract_heartbeats(
                ecg,
                peaks,
                symbols
            )

            if len(beats) == 0:

                warnings.warn(
                    f"No valid heartbeats found in {record_name}"
                )

                continue

            beats = normalize_dataset(beats)

            X_list.append(beats)

            y_list.append(labels)

            record_ids.extend(
                [record_name] * len(labels)
            )

        except Exception as error:

            warnings.warn(
                f"Could not process {record_name}: {error}"
            )

    if not X_list:

        raise RuntimeError(
            "No ECG heartbeats were extracted."
        )

    X = np.concatenate(
        X_list,
        axis=0
    )

    y = np.concatenate(
        y_list,
        axis=0
    )

    record_ids = np.asarray(record_ids)

    return X, y, record_ids


# ============================================================
# 6. VALIDATE RECORDING SPLIT
# ============================================================

def validate_recordings():

    if not DATA_DIR.exists():

        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_DIR}"
        )

    overlap = set(TRAINING_DATA).intersection(
        set(TESTING_DATA)
    )

    if overlap:

        raise ValueError(
            f"Training/testing overlap detected: {overlap}"
        )

    all_records = TRAINING_DATA + TESTING_DATA

    for record in all_records:

        record_path = DATA_DIR / record

        for extension in [".hea", ".dat", ".atr"]:

            file_path = record_path.with_suffix(extension)

            if not file_path.exists():

                raise FileNotFoundError(
                    f"Missing file: {file_path}"
                )

    print("\nRecording split validated.")

    print("Training recordings:", len(TRAINING_DATA))

    print("Testing recordings:", len(TESTING_DATA))

    print("No overlapping recordings detected.")


# ============================================================
# 7. PREPARE TRAINING AND TESTING DATA
# ============================================================

def prepare_train_test():

    validate_recordings()

    train_records = [
        str(DATA_DIR / record)
        for record in TRAINING_DATA
    ]

    test_records = [
        str(DATA_DIR / record)
        for record in TESTING_DATA
    ]

    print("\n========== TRAINING DATA ==========")

    X_train, y_train, train_ids = process_records(
        train_records,
        denoise=True
    )

    print("\n========== TESTING DATA ==========")

    X_test, y_test, test_ids = process_records(
        test_records,
        denoise=True
    )

    print("\n========== DATASET SUMMARY ==========")

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    print("\nTraining class distribution:")

    for i, name in enumerate(CLASSES):

        print(
            f"{name}: {np.sum(y_train == i)}"
        )

    print("\nTesting class distribution:")

    for i, name in enumerate(CLASSES):

        print(
            f"{name}: {np.sum(y_test == i)}"
        )

    # Check that all four classes exist in training

    missing_classes = [
        CLASSES[i]
        for i in range(len(CLASSES))
        if np.sum(y_train == i) == 0
    ]

    if missing_classes:

        raise ValueError(
            f"Missing training classes: {missing_classes}"
        )

    # Save processed datasets

    output_file = OUTPUT_DIR / "ecg_train_test.npz"

    np.savez_compressed(
        output_file,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_ids=train_ids,
        test_ids=test_ids
    )

    print("\nSaved processed dataset to:")

    print(output_file)

    return (
        X_train,
        y_train,
        X_test,
        y_test
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    prepare_train_test()
