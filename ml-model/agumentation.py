
from new_loader import load_training_data, load_testing_data

import torch
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    WeightedRandomSampler
)
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)


# --------------------------------------------------
# 1. Load training and testing data
# --------------------------------------------------

heartbeats_training, annotations_training = load_training_data()
heartbeats_testing, annotations_testing = load_testing_data()


# --------------------------------------------------
# 2. Convert data to tensors
# --------------------------------------------------

heartbeats_training = torch.tensor(
    heartbeats_training,
    dtype=torch.float32
)

annotations_training = torch.tensor(
    annotations_training,
    dtype=torch.long
)

heartbeats_testing = torch.tensor(
    heartbeats_testing,
    dtype=torch.float32
)

annotations_testing = torch.tensor(
    annotations_testing,
    dtype=torch.long
)


num_classes = 4

inputs = heartbeats_training.shape[1]

if heartbeats_training.shape[1] != heartbeats_testing.shape[1]:
    raise ValueError(
        "Training and testing heartbeat sizes do not match."
    )


# --------------------------------------------------
# 3. Display original dataset information
# --------------------------------------------------

class_counts = torch.bincount(
    annotations_training,
    minlength=num_classes
)

print("Original class counts:", class_counts.tolist())

print("Training shape:", heartbeats_training.shape)
print("Testing shape:", heartbeats_testing.shape)
print("Input features per heartbeat:", inputs)


# --------------------------------------------------
# 4. Scaling: Z-score standardization
# --------------------------------------------------

# Calculate mean and standard deviation using training data only.
# Each heartbeat position is scaled independently.

feature_mean = heartbeats_training.mean(dim=0)

feature_std = heartbeats_training.std(
    dim=0,
    unbiased=False
)

# Prevent division by zero for features with no variation
feature_std = torch.where(
    feature_std < 1e-8,
    torch.ones_like(feature_std),
    feature_std
)

# Apply the same training statistics to both datasets
heartbeats_training = (
    heartbeats_training - feature_mean
) / feature_std

heartbeats_testing = (
    heartbeats_testing - feature_mean
) / feature_std

print("\nScaling completed using Z-score standardization.")


# --------------------------------------------------
# 5. Clipping
# --------------------------------------------------

CLIP_MIN = -3.0
CLIP_MAX = 3.0

heartbeats_training = torch.clamp(
    heartbeats_training,
    min=CLIP_MIN,
    max=CLIP_MAX
)

heartbeats_testing = torch.clamp(
    heartbeats_testing,
    min=CLIP_MIN,
    max=CLIP_MAX
)

print("Clipping completed.")
print("Clip range:", CLIP_MIN, "to", CLIP_MAX)

print(
    "Training range:",
    heartbeats_training.min().item(),
    "to",
    heartbeats_training.max().item()
)


# --------------------------------------------------
# 6. Create datasets
# --------------------------------------------------

dataset = TensorDataset(
    heartbeats_training,
    annotations_training
)

testingset = TensorDataset(
    heartbeats_testing,
    annotations_testing
)


# --------------------------------------------------
# 7. Resampling: WeightedRandomSampler
# --------------------------------------------------

# Calculate inverse-frequency weights for each class
class_sample_weights = 1.0 / class_counts.float()

# Give each heartbeat the weight of its corresponding class
sample_weights = class_sample_weights[annotations_training]

# Oversample minority classes with replacement
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True
)


# --------------------------------------------------
# 8. Create DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    dataset,
    batch_size=64,
    sampler=sampler
)

# Keep the test dataset unchanged
test_loader = DataLoader(
    testingset,
    batch_size=64,
    shuffle=False
)


# --------------------------------------------------
# 9. Define Logistic Regression model
# --------------------------------------------------

regression = nn.Linear(
    inputs,
    num_classes
)

optimizer = torch.optim.SGD(
    regression.parameters(),
    lr=0.001
)

# Oversampling already addresses class imbalance
criterion = nn.CrossEntropyLoss()


# --------------------------------------------------
# 10. Training configuration
# --------------------------------------------------

epochs = 50

loss_history = []
accuracy_history = []


# --------------------------------------------------
# 11. Training loop
# --------------------------------------------------

for epoch in range(epochs):

    regression.train()

    total_loss = 0.0

    for heartbeats, labels in train_loader:

        optimizer.zero_grad()

        logits = regression(heartbeats)

        batch_loss = criterion(
            logits,
            labels
        )

        batch_loss.backward()

        optimizer.step()

        total_loss += batch_loss.item() * len(labels)

    average_loss = total_loss / len(dataset)

    loss_history.append(average_loss)


    # --------------------------------------------------
    # Evaluate accuracy
    # --------------------------------------------------

    regression.eval()

    correct = 0

    with torch.no_grad():

        for heartbeats, labels in test_loader:

            logits = regression(heartbeats)

            predicted = logits.argmax(dim=1)

            correct += (
                predicted == labels
            ).sum().item()

    accuracy = 100 * correct / len(testingset)

    accuracy_history.append(accuracy)

    print(
        f"Epoch: {epoch + 1}. "
        f"Loss: {average_loss:.4f}. "
        f"Accuracy: {accuracy:.2f}%"
    )


# --------------------------------------------------
# 12. Plot training loss
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(loss_history)

plt.xlabel("Number of epochs")
plt.ylabel("Average loss")

plt.title(
    "Training Loss - Resampling, Scaling and Clipping"
)

plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 13. Plot test accuracy
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(accuracy_history)

plt.xlabel("Number of epochs")
plt.ylabel("Accuracy (%)")

plt.title(
    "Test Accuracy - Resampling, Scaling and Clipping"
)

plt.grid(True)
plt.tight_layout()
plt.show()


# --------------------------------------------------
# 14. Final evaluation
# --------------------------------------------------

regression.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for heartbeats, labels in test_loader:

        logits = regression(heartbeats)

        predictions = logits.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_labels.extend(
            labels.cpu().tolist()
        )


# --------------------------------------------------
# 15. Classification report
# --------------------------------------------------

class_labels = [0, 1, 2, 3]

class_names = [
    "N - Normal",
    "S - Supraventricular",
    "V - Ventricular",
    "F - Fusion"
]

print("\nClassification Report:")

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=class_labels,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# --------------------------------------------------
# 16. Macro F1-score
# --------------------------------------------------

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    labels=class_labels,
    average="macro",
    zero_division=0
)

print(f"Macro F1-score: {macro_f1:.4f}")


# --------------------------------------------------
# 17. Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=class_labels
)

print("\nConfusion Matrix:")
print(cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
).plot(
    cmap="Blues",
    xticks_rotation=35,
    values_format="d"
)

plt.title(
    "Logistic Regression - Resampling, Scaling and Clipping"
)

plt.tight_layout()
plt.show()
