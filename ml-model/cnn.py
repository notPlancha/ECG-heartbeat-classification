from data_loader import load_training_data, load_testing_data

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# Reproducibility
# ============================================================

torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# CNN Model
# ============================================================

class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.classifier = nn.Linear(64 * 32, num_classes)

    def forward(self, x):

        if x.ndim == 2:
            x = x.unsqueeze(1)

        x = self.features(x)
        x = torch.flatten(x, start_dim=1)

        return self.classifier(x)


# ============================================================
# Load data
# ============================================================

heartbeats_training, annotations_training = load_training_data()
heartbeats_testing, annotations_testing = load_testing_data()


# ============================================================
# Convert training data to NumPy arrays
# ============================================================

heartbeats_training_np = np.asarray(heartbeats_training)
annotations_training_np = np.asarray(
    annotations_training
).reshape(-1)


# Check that all four classes exist
num_classes = 4
class_labels = np.arange(num_classes)

if not np.array_equal(
    np.unique(annotations_training_np),
    class_labels
):
    raise ValueError(
        "Training data must contain classes 0, 1, 2 and 3."
    )


# ============================================================
# Calculate class weights from ORIGINAL distribution
# ============================================================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=class_labels,
    y=annotations_training_np
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
)

print("\nClass weights:")

for cls, weight in zip(class_labels, class_weights):
    print(f"Class {cls}: {weight.item():.4f}")


# ============================================================
# Original class distribution
# ============================================================

classes, class_counts = np.unique(
    annotations_training_np,
    return_counts=True
)

majority_count = max(class_counts)

print("\nOriginal class distribution:")

for cls, count in zip(classes, class_counts):
    print(f"Class {cls}: {count} samples")


# ============================================================
# Define sampling strategy
# ============================================================

sampling_strategy = {
    1: int(majority_count * 0.75),  # S: 75%
    2: int(majority_count * 0.25),  # V: 25%
    3: int(majority_count * 0.75)   # F: 75%
}


# ============================================================
# Oversampling only
# ============================================================

oversampled_heartbeats = []
oversampled_annotations = []

for cls in classes:

    cls_indices = np.where(
        annotations_training_np == cls
    )[0]

    cls_heartbeats = heartbeats_training_np[cls_indices]
    cls_annotations = annotations_training_np[cls_indices]

    current_count = len(cls_indices)

    target_count = sampling_strategy.get(
        cls,
        current_count
    )

    # Oversample only if target exceeds current count
    if current_count < target_count:

        resampled_indices = resample(
            np.arange(current_count),
            replace=True,
            n_samples=target_count,
            random_state=42
        )

        cls_heartbeats = cls_heartbeats[resampled_indices]
        cls_annotations = cls_annotations[resampled_indices]

    oversampled_heartbeats.append(cls_heartbeats)
    oversampled_annotations.append(cls_annotations)


# ============================================================
# Combine oversampled classes
# ============================================================

heartbeats_training_resampled = np.vstack(
    oversampled_heartbeats
)

annotations_training_resampled = np.hstack(
    oversampled_annotations
)


# ============================================================
# Shuffle dataset
# ============================================================

shuffle_indices = np.random.RandomState(42).permutation(
    len(annotations_training_resampled)
)

heartbeats_training_resampled = (
    heartbeats_training_resampled[shuffle_indices]
)

annotations_training_resampled = (
    annotations_training_resampled[shuffle_indices]
)


# ============================================================
# Verify class distribution
# ============================================================

print("\nClass distribution after oversampling:")

unique, counts = np.unique(
    annotations_training_resampled,
    return_counts=True
)

for cls, count in zip(unique, counts):
    print(f"Class {cls}: {count} samples")


# ============================================================
# Convert training data to tensors
# ============================================================

heartbeats_training = torch.tensor(
    heartbeats_training_resampled,
    dtype=torch.float32
)

annotations_training = torch.tensor(
    annotations_training_resampled,
    dtype=torch.long
)


# ============================================================
# Convert testing data to tensors
# ============================================================

if not isinstance(heartbeats_testing, torch.Tensor):
    heartbeats_testing = torch.tensor(
        heartbeats_testing,
        dtype=torch.float32
    )
else:
    heartbeats_testing = heartbeats_testing.float()

if not isinstance(annotations_testing, torch.Tensor):
    annotations_testing = torch.tensor(
        annotations_testing,
        dtype=torch.long
    )
else:
    annotations_testing = annotations_testing.long()

annotations_testing = annotations_testing.reshape(-1)


# ============================================================
# Create DataLoaders
# ============================================================

train_dataset = TensorDataset(
    heartbeats_training,
    annotations_training
)

test_dataset = TensorDataset(
    heartbeats_testing,
    annotations_testing
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# ============================================================
# Define CNN model
# ============================================================

cnn = CNN(num_classes)


# ============================================================
# Optimizer and weighted loss function
# ============================================================

optimizer = torch.optim.Adam(
    cnn.parameters(),
    lr=0.001
)

# Class weighting combined with oversampling
criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3
)


# ============================================================
# Training loop
# ============================================================

epochs = 20

loss_history = []
accuracy_history = []

best_loss = float("inf")

for epoch in range(epochs):

    cnn.train()

    total_loss = 0.0

    for heartbeats, labels in train_loader:

        optimizer.zero_grad()

        logits = cnn(heartbeats)

        batch_loss = criterion(logits, labels)

        batch_loss.backward()

        optimizer.step()

        total_loss += batch_loss.item() * len(labels)

    average_loss = total_loss / len(train_dataset)

    loss_history.append(average_loss)

    # Update learning rate
    scheduler.step(average_loss)

    # Save best model based on training loss
    if average_loss < best_loss:

        best_loss = average_loss

        torch.save(
            cnn.state_dict(),
            "best_cnn_model.pth"
        )


    # ========================================================
    # Evaluate accuracy
    # ========================================================

    cnn.eval()

    correct = 0

    with torch.no_grad():

        for heartbeats, labels in test_loader:

            logits = cnn(heartbeats)

            predicted = logits.argmax(dim=1)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = 100 * correct / len(test_dataset)

    accuracy_history.append(accuracy)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch: {epoch + 1}. "
        f"Loss: {average_loss:.4f}. "
        f"Accuracy: {accuracy:.2f}%. "
        f"LR: {current_lr:.6f}"
    )


# ============================================================
# Load best model
# ============================================================

cnn.load_state_dict(
    torch.load(
        "best_cnn_model.pth",
        weights_only=True
    )
)

cnn.eval()


# ============================================================
# Final evaluation
# ============================================================

all_labels = []
all_predictions = []

with torch.no_grad():

    for heartbeats, labels in test_loader:

        logits = cnn(heartbeats)

        predicted = logits.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())

        all_predictions.extend(predicted.cpu().numpy())


# ============================================================
# Classification report
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    labels=[0, 1, 2, 3],
    target_names=["N", "S", "V", "F"],
    zero_division=0
)

print("\nClassification Report:")
print(report)


# ============================================================
# Plot training loss and testing accuracy
# ============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(accuracy_history)
plt.title("Testing Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

plt.tight_layout()
plt.show()