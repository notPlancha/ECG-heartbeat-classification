
from data_loader import load_training_data, load_testing_data
from class_weighting import calculate_class_weights

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

heartbeats_training, annotations_training = load_training_data()
heartbeats_testing, annotations_testing = load_testing_data()

# Convert data to tensors
heartbeats_training = torch.tensor(
    heartbeats_training, dtype=torch.float32
)
annotations_training = torch.tensor(
    annotations_training, dtype=torch.long
)

heartbeats_testing = torch.tensor(
    heartbeats_testing, dtype=torch.float32
)
annotations_testing = torch.tensor(
    annotations_testing, dtype=torch.long
)

num_classes = 4


class_counts = torch.bincount(
    annotations_training,
    minlength=num_classes
)

print("Original class counts:", class_counts.tolist())

print("Beat list shape:", heartbeats_training.shape)
print("Annotation list shape:", annotations_training.shape)
print("First label:", annotations_training[0].item())


dataset = TensorDataset(
    heartbeats_training,
    annotations_training
)

testingset = TensorDataset(
    heartbeats_testing,
    annotations_testing
)

# Calculate inverse-frequency weights for each class.
# Samples belonging to minority classes get larger weights.

class_sample_weights = 1.0 / class_counts.float()

# Assign each training heartbeat the weight of its class
sample_weights = class_sample_weights[annotations_training]

# Sample with replacement to oversample minority classes
sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(dataset),
    replacement=True
)

train_loader = DataLoader(
    dataset,
    batch_size=64,
    sampler=sampler
)

# Do not oversample the test set
test_loader = DataLoader(
    testingset,
    batch_size=64,
    shuffle=False
)

inputs = 256

regression = nn.Linear(inputs, num_classes)

optimizer = torch.optim.SGD(
    regression.parameters(),
    lr=0.001
)

# Use ordinary cross-entropy because oversampling
# already addresses class imbalance
criterion = nn.CrossEntropyLoss()

epochs = 50

loss_history = []
accuracy_history = []

for epoch in range(epochs):
    regression.train()
    total_loss = 0.0

    for heartbeats, labels in train_loader:
        optimizer.zero_grad()

        logits = regression(heartbeats)
        batch_loss = criterion(logits, labels)

        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item() * len(labels)

    average_loss = total_loss / len(dataset)
    loss_history.append(average_loss)

    regression.eval()
    correct = 0

    with torch.no_grad():
        for heartbeats, labels in test_loader:
            logits = regression(heartbeats)
            predicted = logits.argmax(dim=1)

            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / len(testingset)
    accuracy_history.append(accuracy)

    print(
        f"Epoch: {epoch + 1}. "
        f"Loss: {average_loss:.4f}. "
        f"Accuracy: {accuracy:.2f}%"
    )


plt.plot(loss_history)
plt.xlabel("Number of epochs")
plt.ylabel("Average loss")
plt.title("Training Loss with Oversampling")
plt.tight_layout()
plt.show()


plt.plot(accuracy_history)
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy with Oversampling")
plt.tight_layout()
plt.show()

regression.eval()

all_predictions = []
all_labels = []

with torch.no_grad():
    for heartbeats, labels in test_loader:
        logits = regression(heartbeats)
        predictions = logits.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

# Class names corresponding to labels 0, 1, 2 and 3
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

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    labels=class_labels,
    average="macro",
    zero_division=0
)

print(f"Macro F1-score: {macro_f1:.4f}")

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

plt.title("Logistic Regression - Confusion Matrix with Oversampling")
plt.tight_layout()
plt.show()