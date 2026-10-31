from data_loader import (
    load_training_data,
    load_validation_data,
    load_testing_data,
)

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_curve
)
from sklearn.utils import resample


# ============================================================
# Reproducibility and CUDA setup
# ============================================================

torch.manual_seed(42)
np.random.seed(42)

if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.cuda.manual_seed_all(42)

    print("CUDA is available!")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

print(f"Training device: {device}")


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


class FocalLoss(nn.Module):
    def __init__(self, class_weights, gamma=2.0):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.gamma = gamma

    def forward(self, logits, targets):
        log_probabilities = nn.functional.log_softmax(logits, dim=1)
        target_log_probabilities = log_probabilities.gather(
            dim=1,
            index=targets.unsqueeze(1)
        ).squeeze(1)
        target_probabilities = target_log_probabilities.exp()
        target_weights = self.class_weights[targets]

        loss = -target_weights * (
            1.0 - target_probabilities
        ).pow(self.gamma) * target_log_probabilities
        return loss.mean()


# ============================================================
# Load data
# ============================================================

heartbeats_training, annotations_training = load_training_data()
heartbeats_validation, annotations_validation = load_validation_data()
heartbeats_testing, annotations_testing = load_testing_data()


# ============================================================
# Convert training data to NumPy arrays
# ============================================================

heartbeats_training_np = np.asarray(heartbeats_training)

annotations_training_np = np.asarray(
    annotations_training
).reshape(-1)

heartbeats_validation_np = np.asarray(heartbeats_validation)
annotations_validation_np = np.asarray(
    annotations_validation
).reshape(-1)


# Check that all four classes exist
num_classes = 4
class_labels = np.arange(num_classes)

for split_name, labels in (
    ("training", annotations_training_np),
    ("validation", annotations_validation_np),
):
    if not np.array_equal(np.unique(labels), class_labels):
        raise ValueError(
            f"{split_name.title()} data must contain classes 0, 1, 2 and 3."
        )


# ============================================================
# Original class distribution
# ============================================================

classes, class_counts = np.unique(
    annotations_training_np,
    return_counts=True
)
majority_count = int(class_counts.max())

print("\nOriginal class distribution:")

for cls, count in zip(classes, class_counts):
    print(f"Class {cls}: {count} samples")


# ============================================================
# Moderate minority oversampling on training data only
# ============================================================

oversampling_ratio = 0.1
oversampled_heartbeats = []
oversampled_annotations = []

for cls in classes:
    cls_indices = np.where(annotations_training_np == cls)[0]
    cls_heartbeats = heartbeats_training_np[cls_indices]
    cls_annotations = annotations_training_np[cls_indices]
    target_count = max(
        len(cls_indices),
        int(majority_count * oversampling_ratio)
    )
    extra_count = target_count - len(cls_indices)

    if extra_count > 0:
        extra_indices = resample(
            np.arange(len(cls_indices)),
            replace=True,
            n_samples=extra_count,
            random_state=42
        )
        cls_heartbeats = np.concatenate(
            (cls_heartbeats, cls_heartbeats[extra_indices])
        )
        cls_annotations = np.concatenate(
            (cls_annotations, cls_annotations[extra_indices])
        )

    oversampled_heartbeats.append(cls_heartbeats)
    oversampled_annotations.append(cls_annotations)

heartbeats_training_resampled = np.vstack(oversampled_heartbeats)
annotations_training_resampled = np.hstack(oversampled_annotations)

resampled_classes, resampled_counts = np.unique(
    annotations_training_resampled,
    return_counts=True
)
print("\nTraining distribution after moderate oversampling:")
for cls, count in zip(resampled_classes, resampled_counts):
    print(f"Class {cls}: {count} samples")

class_counts_resampled = np.bincount(
    annotations_training_resampled,
    minlength=num_classes
)
class_weights = np.sqrt(
    class_counts_resampled.max() / class_counts_resampled
)
class_weights /= class_weights.mean()
print("Tempered class weights:", class_weights.tolist())


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

heartbeats_validation = torch.tensor(
    heartbeats_validation_np,
    dtype=torch.float32
)

annotations_validation = torch.tensor(
    annotations_validation_np,
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

validation_dataset = TensorDataset(
    heartbeats_validation,
    annotations_validation
)

test_dataset = TensorDataset(
    heartbeats_testing,
    annotations_testing
)

# Pin memory when using CUDA for faster transfers
use_cuda = device.type == "cuda"

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    pin_memory=use_cuda
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=64,
    shuffle=False,
    pin_memory=use_cuda
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    pin_memory=use_cuda
)


# ============================================================
# Define CNN model and move to GPU
# ============================================================

cnn = CNN(num_classes).to(device)

print(f"\nCNN model is running on: {next(cnn.parameters()).device}")


# ============================================================
# Optimizer and tempered class-weighted loss function
# ============================================================

optimizer = torch.optim.SGD(
    cnn.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)

criterion = FocalLoss(
    class_weights=torch.tensor(class_weights, dtype=torch.float32),
    gamma=2.0
).to(device)


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

epochs = 50

train_loss_history = []
validation_loss_history = []
validation_accuracy_history = []
validation_macro_f1_history = []

best_validation_macro_f1 = -1.0
best_validation_loss = float("inf")
checkpoint_path = "best_cnn_model_moderate_oversampling.pth"

for epoch in range(epochs):

    cnn.train()

    total_loss = 0.0

    for heartbeats, labels in train_loader:

        # Move batch to GPU
        heartbeats = heartbeats.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        logits = cnn(heartbeats)

        batch_loss = criterion(logits, labels)

        batch_loss.backward()

        optimizer.step()

        total_loss += batch_loss.item() * len(labels)

    average_train_loss = total_loss / len(train_dataset)
    train_loss_history.append(average_train_loss)

    cnn.eval()
    validation_total_loss = 0.0
    validation_correct = 0
    validation_labels_all = []
    validation_predictions = []

    with torch.no_grad():
        for heartbeats, labels in validation_loader:
            heartbeats = heartbeats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = cnn(heartbeats)
            batch_loss = criterion(logits, labels)
            predicted = logits.argmax(dim=1)

            validation_total_loss += batch_loss.item() * len(labels)
            validation_correct += (predicted == labels).sum().item()
            validation_labels_all.extend(labels.cpu().numpy())
            validation_predictions.extend(predicted.cpu().numpy())

    average_validation_loss = (
        validation_total_loss / len(validation_dataset)
    )
    validation_accuracy = (
        100 * validation_correct / len(validation_dataset)
    )
    validation_macro_f1 = f1_score(
        validation_labels_all,
        validation_predictions,
        labels=class_labels,
        average="macro",
        zero_division=0
    )

    validation_loss_history.append(average_validation_loss)
    validation_accuracy_history.append(validation_accuracy)
    validation_macro_f1_history.append(validation_macro_f1)
    scheduler.step(average_validation_loss)

    if (
        validation_macro_f1 > best_validation_macro_f1
        or (
            np.isclose(validation_macro_f1, best_validation_macro_f1)
            and average_validation_loss < best_validation_loss
        )
    ):
        best_validation_macro_f1 = validation_macro_f1
        best_validation_loss = average_validation_loss
        torch.save(cnn.state_dict(), checkpoint_path)

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"Epoch: {epoch + 1}. "
        f"Train Loss: {average_train_loss:.4f}. "
        f"Validation Loss: {average_validation_loss:.4f}. "
        f"Validation Accuracy: {validation_accuracy:.2f}%. "
        f"Validation Macro F1: {validation_macro_f1:.4f}. "
        f"LR: {current_lr:.6f}. "
        f"Device: {device}"
    )


# ============================================================
# Load best model
# ============================================================

cnn.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )
)

cnn.eval()


# ============================================================
# Tune the F threshold using validation data only
# ============================================================

validation_labels_all = []
validation_probabilities = []

with torch.no_grad():
    for heartbeats, labels in validation_loader:
        heartbeats = heartbeats.to(
            device,
            non_blocking=True
        )
        probabilities = torch.softmax(cnn(heartbeats), dim=1)
        validation_labels_all.extend(labels.numpy())
        validation_probabilities.extend(probabilities.cpu().numpy())

validation_probabilities = np.asarray(validation_probabilities)
validation_f_labels = (
    np.asarray(validation_labels_all) == 3
).astype(np.int64)
precision, recall, thresholds = precision_recall_curve(
    validation_f_labels,
    validation_probabilities[:, 3]
)
f1_scores = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-12)
)
best_threshold_index = int(np.argmax(f1_scores))
f_threshold = thresholds[best_threshold_index]

validation_predictions = validation_probabilities[:, :3].argmax(axis=1)
validation_predictions[
    validation_probabilities[:, 3] >= f_threshold
] = 3

print(f"\nValidation-selected F threshold: {f_threshold:.4f}")
print(
    "Validation F1 at selected threshold: "
    f"{f1_scores[best_threshold_index]:.4f}"
)
print("\nValidation classification report with tuned F threshold:")
print(
    classification_report(
        validation_labels_all,
        validation_predictions,
        labels=[0, 1, 2, 3],
        target_names=["N", "S", "V", "F"],
        zero_division=0
    )
)


# ============================================================
# Final evaluation
# ============================================================

all_labels = []
all_probabilities = []

with torch.no_grad():
    for heartbeats, labels in test_loader:
        heartbeats = heartbeats.to(
            device,
            non_blocking=True
        )
        probabilities = torch.softmax(cnn(heartbeats), dim=1)
        all_labels.extend(labels.numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

all_probabilities = np.asarray(all_probabilities)
all_predictions = all_probabilities[:, :3].argmax(axis=1)
all_predictions[all_probabilities[:, 3] >= f_threshold] = 3


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
plt.plot(train_loss_history, label="Training")
plt.plot(validation_loss_history, label="Validation")
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(validation_macro_f1_history)
plt.title("Validation Macro F1")
plt.xlabel("Epoch")
plt.ylabel("Macro F1")

plt.tight_layout()
plt.show()


# ============================================================
# GPU memory information
# ============================================================

if torch.cuda.is_available():

    print("\nGPU memory usage:")

    allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)

    print(f"Allocated GPU memory: {allocated:.2f} GB")
    print(f"Reserved GPU memory: {reserved:.2f} GB")
