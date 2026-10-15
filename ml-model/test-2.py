from sympy import python
python
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# =====================================================
# 1. CONFIGURATION
# =====================================================

# X: extracted heartbeat signals, shape (num_beats, num_samples)
# y: corresponding annotation labels, e.g. ["N", "V", "N", "S", ...]

num_classes = 4

label_map = {
    "N": 0,  # Normal
    "S": 1,  # Supraventricular ectopic
    "V": 2,  # Ventricular ectopic
    "F": 3,  # Fusion
}

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================================
# 2. FILTER LABELS AND EXCLUDE Q / UNMAPPED LABELS
# =====================================================


# Path to your MIT-BIH database folder
DATA_DIR = path("../dataset/mit-bih-arrhythmia-database-1.0.0")

# Four-class mapping
label_map = {
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

# Window around each annotated beat
BEFORE = 90
AFTER = 166
WINDOW_SIZE = BEFORE + AFTER

X_list = []
y_list = []

# Find record names from the .hea files
record_files = sorted(DATA_DIR.glob("*.hea"))

if not record_files:
    raise FileNotFoundError(
        f"No .hea files found in {DATA_DIR.resolve()}"
    )

for record_file in record_files:
    record_name = record_file.stem

    try:
        # Load ECG signal and metadata
        record = wfdb.rdrecord(str(DATA_DIR / record_name))

        # Load beat annotations
        annotation = wfdb.rdann(
            str(DATA_DIR / record_name),
            "atr"
        )

        # Use the first ECG channel
        signal = record.p_signal[:, 0]

        for sample, symbol in zip(
            annotation.sample, annotation.symbol
        ):
            # Skip annotations not in our four-class mapping
            if symbol not in label_map:
                continue

            start = sample - BEFORE
            end = sample + AFTER

            # Skip beats too close to signal boundaries
            if start < 0 or end > len(signal):
                continue

            beat = signal[start:end]

            if len(beat) != WINDOW_SIZE:
                continue

            X_list.append(beat)
            y_list.append(label_map[symbol])

        print(f"Loaded record {record_name}")

    except Exception as exc:
        print(f"Skipping {record_name}: {exc}")

# Convert to NumPy arrays
X = np.asarray(X_list, dtype=np.float32)
y = np.asarray(y_list, dtype=np.int64)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Total beats:", len(y))
print("Class counts:", np.bincount(y, minlength=4))


if len(X) != len(y):
    raise ValueError("X and y must contain the same number of beats.")

# Keep only labels included in our four-class mapping.
keep = np.isin(y, list(label_map.keys()))

X = X[keep]
y = y[keep]

# Convert string labels to integer class IDs.
y_encoded = np.array(
    [label_map[label] for label in y],
    dtype=np.int64
)

print("Total retained beats:", len(y_encoded))
print("Classes: N=0, S=1, V=2, F=3")

# =====================================================
# 3. SPLIT INTO TRAINING AND TEST SETS
# =====================================================

# Demonstration split only.
# For reliable ECG evaluation, split by record/patient
# instead of randomly splitting individual beats.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining beats:", len(y_train))
print("Test beats:", len(y_test))

# =====================================================
# 4. CALCULATE CLASS WEIGHTS FROM TRAINING LABELS
# =====================================================

class_counts = torch.bincount(
    torch.tensor(y_train, dtype=torch.long),
    minlength=num_classes
).float()

if (class_counts == 0).any():
    raise ValueError(
        "At least one class is missing from the training set."
    )

# Inverse-frequency weighting.
# Each class receives approximately equal total weight.
class_weights = (
    class_counts.sum()
    / (num_classes * class_counts)
)

print("\nTraining class counts:")
for class_name, class_id in label_map.items():
    print(
        f"{class_name}: "
        f"{int(class_counts[class_id])} beats"
    )

print("\nClass weights:")
for class_name, class_id in label_map.items():
    print(
        f"{class_name}: "
        f"{class_weights[class_id].item():.4f}"
    )

# =====================================================
# 5. CREATE WEIGHTED CROSS-ENTROPY LOSS
# =====================================================

class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

print("\nDevice:", device)
print("Weighted CrossEntropyLoss is ready.")

# =====================================================
# 6. CONVERT DATA TO PYTORCH TENSORS
# =====================================================

# For a 1D CNN, input shape must be:
# (batch_size, channels, sequence_length)

X_train_tensor = torch.tensor(
    X_train, dtype=torch.float32
).unsqueeze(1)

X_test_tensor = torch.tensor(
    X_test, dtype=torch.float32
).unsqueeze(1)

y_train_tensor = torch.tensor(
    y_train, dtype=torch.long
)

y_test_tensor = torch.tensor(
    y_test, dtype=torch.long
)

print("\nTensor shapes:")
print("X_train:", X_train_tensor.shape)
print("y_train:", y_train_tensor.shape)
print("X_test:", X_test_tensor.shape)
print("y_test:", y_test_tensor.shape)

# Move the model and criterion to `device` when training.
# Move batches to `device` inside your training loop.
