
from pathlib import Path

import numpy as np
import torch

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)

from ecg_preprocessing import OUTPUT_DIR, CLASSES

from dataset import ECGDataset

from model import ECG_CNN_GRU


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_DIR = ROOT / "checkpoints"

MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"

DATA_FILE = OUTPUT_DIR / "ecg_train_test.npz"

RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BATCH_SIZE = 128

NUM_CLASSES = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# VERIFY CONFIGURATION
# ============================================================

print("Device:", DEVICE)

print("Number of classes:", NUM_CLASSES)

print("Classes:", CLASSES)


if len(CLASSES) != NUM_CLASSES:

    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"but found {len(CLASSES)}."
    )


# ============================================================
# CHECK FILES
# ============================================================

if not DATA_FILE.exists():

    raise FileNotFoundError(
        f"Processed dataset not found: {DATA_FILE}. "
        "Run ecg_preprocessing.py first."
    )


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}. "
        "Run train.py first."
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

data = np.load(DATA_FILE)

X_test = data["X_test"]

y_test = data["y_test"]


print("Test dataset:", X_test.shape)

print("Test labels:", np.unique(y_test))


# ============================================================
# VALIDATE TEST DATA
# ============================================================

if len(X_test) == 0:

    raise ValueError(
        "Test dataset is empty."
    )


if not np.all(np.isin(y_test, range(NUM_CLASSES))):

    raise ValueError(
        "Test dataset contains invalid class labels."
    )


# ============================================================
# CREATE TEST DATASET
# ============================================================

test_dataset = ECGDataset(
    X_test,
    y_test,
    training=False,
    augment=False
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained model...")

model = ECG_CNN_GRU(
    num_classes=NUM_CLASSES
).to(DEVICE)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# PREDICTIONS
# ============================================================

all_predictions = []

all_labels = []


print("\nGenerating predictions...")


with torch.no_grad():

    for signals, labels in test_loader:

        signals = signals.to(DEVICE)

        outputs = model(signals)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


all_predictions = np.asarray(
    all_predictions
)

all_labels = np.asarray(
    all_labels
)


# ============================================================
# PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)


precision = precision_score(
    all_labels,
    all_predictions,
    labels=[0, 1, 2, 3],
    average="macro",
    zero_division=0
)


sensitivity = recall_score(
    all_labels,
    all_predictions,
    labels=[0, 1, 2, 3],
    average="macro",
    zero_division=0
)


f1 = f1_score(
    all_labels,
    all_predictions,
    labels=[0, 1, 2, 3],
    average="macro",
    zero_division=0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n====================================")

print("ECG CLASSIFICATION RESULTS")

print("====================================")

print(f"Accuracy:    {accuracy:.4f}")

print(f"Precision:   {precision:.4f}")

print(f"Sensitivity: {sensitivity:.4f}")

print(f"F1-score:    {f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    labels=[0, 1, 2, 3],
    target_names=CLASSES,
    zero_division=0
)


print("\nClassification report:\n")

print(report)


with open(
    RESULTS_DIR / "classification_report.txt",
    "w"
) as file:

    file.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1, 2, 3]
)


print("\nConfusion matrix:\n")

print(cm)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASSES
)


fig, ax = plt.subplots(
    figsize=(8, 6)
)


display.plot(
    cmap="Blues",
    values_format="d",
    ax=ax
)


ax.set_title(
    "ECG CNN-GRU Confusion Matrix"
)


plt.tight_layout()


plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# ============================================================
# SAVE METRICS
# ============================================================

np.savez(
    RESULTS_DIR / "metrics.npz",
    accuracy=accuracy,
    precision=precision,
    sensitivity=sensitivity,
    f1=f1,
    confusion_matrix=cm,
    predictions=all_predictions,
    labels=all_labels
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n====================================")

print("EVALUATION COMPLETED")

print("====================================")

print(f"Accuracy:    {accuracy:.4f}")

print(f"Precision:   {precision:.4f}")

print(f"Sensitivity: {sensitivity:.4f}")

print(f"F1-score:    {f1:.4f}")


print("\nResults saved to:")

print(RESULTS_DIR)

print("\nGenerated files:")

print("- classification_report.txt")

print("- confusion_matrix.png")

print("- metrics.npz")
