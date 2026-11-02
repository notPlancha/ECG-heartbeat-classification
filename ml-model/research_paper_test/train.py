
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from ecg_preprocessing import (
    prepare_train_test,
    OUTPUT_DIR,
    CLASSES
)

from dataset import (
    ECGDataset,
    balance_training_data
)

from model import ECG_CNN_GRU


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_DIR = ROOT / "checkpoints"

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DATA_FILE = OUTPUT_DIR / "ecg_train_test.npz"

BATCH_SIZE = 128

EPOCHS = 50

LEARNING_RATE = 0.001

RANDOM_SEED = 42

NUM_CLASSES = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(RANDOM_SEED)


print("Device:", DEVICE)

print("Number of classes:", NUM_CLASSES)

print("Classes:", CLASSES)


# ============================================================
# LOAD DATA
# ============================================================

if DATA_FILE.exists():

    print("\nLoading existing processed dataset...")

    data = np.load(DATA_FILE)

    X_train = data["X_train"]

    y_train = data["y_train"]

    X_test = data["X_test"]

    y_test = data["y_test"]

else:

    print("\nPreparing ECG dataset...")

    (
        X_train,
        y_train,
        X_test,
        y_test
    ) = prepare_train_test()


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\nTraining samples:", len(X_train))

print("Testing samples:", len(X_test))

print("\nTraining shape:", X_train.shape)

print("Testing shape:", X_test.shape)


# ============================================================
# VALIDATE DATA
# ============================================================

if len(X_train) == 0 or len(X_test) == 0:

    raise ValueError(
        "Training or testing dataset is empty."
    )


if X_train.shape[1] != 186:

    raise ValueError(
        f"Expected heartbeat length 186, "
        f"got {X_train.shape[1]}"
    )


if X_test.shape[1] != 186:

    raise ValueError(
        f"Expected heartbeat length 186, "
        f"got {X_test.shape[1]}"
    )


if not np.all(np.isin(y_train, range(NUM_CLASSES))):

    raise ValueError(
        "Training dataset contains invalid class labels."
    )


if not np.all(np.isin(y_test, range(NUM_CLASSES))):

    raise ValueError(
        "Testing dataset contains invalid class labels."
    )


# ============================================================
# PRINT ORIGINAL CLASS DISTRIBUTION
# ============================================================

print("\nOriginal training class distribution:")

for i, name in enumerate(CLASSES):

    print(
        f"{name}: {np.sum(y_train == i)}"
    )


print("\nTesting class distribution:")

for i, name in enumerate(CLASSES):

    print(
        f"{name}: {np.sum(y_test == i)}"
    )


# ============================================================
# BALANCE TRAINING DATA ONLY
# ============================================================

print("\nBalancing training dataset...")

X_train, y_train = balance_training_data(
    X_train,
    y_train,
    random_state=RANDOM_SEED
)

print("\nBalanced training dataset:", X_train.shape)


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = ECGDataset(
    X_train,
    y_train,
    training=True,
    augment=True
)

test_dataset = ECGDataset(
    X_test,
    y_test,
    training=False,
    augment=False
)


# ============================================================
# CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
# ============================================================

model = ECG_CNN_GRU(
    num_classes=NUM_CLASSES
).to(DEVICE)


# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

best_loss = float("inf")

history = {
    "train_loss": [],
    "train_accuracy": []
}


print("\n========== STARTING TRAINING ==========\n")


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    for signals, labels in train_loader:

        signals = signals.to(DEVICE)

        labels = labels.to(DEVICE)

        # Reset gradients

        optimizer.zero_grad()

        # Forward pass

        outputs = model(signals)

        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation

        loss.backward()

        optimizer.step()

        # Track loss

        running_loss += loss.item()

        # Calculate accuracy

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    # Epoch metrics

    train_loss = (
        running_loss / len(train_loader)
    )

    train_accuracy = (
        100 * correct / total
    )

    history["train_loss"].append(
        train_loss
    )

    history["train_accuracy"].append(
        train_accuracy
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {train_loss:.4f} "
        f"Accuracy: {train_accuracy:.2f}%"
    )

    # Save best training-loss checkpoint

    if train_loss < best_loss:

        best_loss = train_loss

        torch.save(
            model.state_dict(),
            CHECKPOINT_DIR / "best_model.pth"
        )

        print("Best model checkpoint updated.")


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

np.savez(
    CHECKPOINT_DIR / "training_history.npz",
    train_loss=history["train_loss"],
    train_accuracy=history["train_accuracy"]
)


# ============================================================
# TRAINING SUMMARY
# ============================================================

print("\n====================================")

print("TRAINING COMPLETED")

print("====================================")

print("Best training loss:", best_loss)

print(
    "Best model saved to:",
    CHECKPOINT_DIR / "best_model.pth"
)

print(
    "Training history saved to:",
    CHECKPOINT_DIR / "training_history.npz"
)
