from data_loader import load_training_data, load_testing_data
from class_weighting import calculate_class_weights
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.utils import resample

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


# Load data
heartbeats_training, annotations_training = load_training_data()
heartbeats_testing, annotations_testing = load_testing_data()

# Convert training data to NumPy arrays for resampling
heartbeats_training_np = heartbeats_training
annotations_training_np = annotations_training

# Oversample the minority classes
oversampled_heartbeats = []
oversampled_annotations = []

# Get unique classes and their counts
classes, class_counts = np.unique(annotations_training_np, return_counts=True)
max_count = max(class_counts)  # Target count for oversampling

for cls in classes:
    # Extract all samples of the current class
    cls_indices = np.where(annotations_training_np == cls)[0]
    cls_heartbeats = heartbeats_training_np[cls_indices]
    cls_annotations = annotations_training_np[cls_indices]

    # Resample indices once to guarantee heartbeats and annotations stay perfectly aligned
    resampled_indices = resample(
        np.arange(len(cls_indices)),
        replace=True,  # Allow replacement
        n_samples=max_count,  # Match the maximum class count
        random_state=42
    )

    # Append the oversampled data using the same index mask
    oversampled_heartbeats.append(cls_heartbeats[resampled_indices])
    oversampled_annotations.append(cls_annotations[resampled_indices])

# Combine all oversampled classes
heartbeats_training_resampled = np.vstack(oversampled_heartbeats)
annotations_training_resampled = np.hstack(oversampled_annotations)

# Convert training data back to tensors
heartbeats_training = torch.tensor(heartbeats_training_resampled, dtype=torch.float32)
annotations_training = torch.tensor(annotations_training_resampled, dtype=torch.long)

# Ensure testing data are properly converted to tensors (prevents TypeError if returned as raw arrays/lists)
if not isinstance(heartbeats_testing, torch.Tensor):
    heartbeats_testing = torch.tensor(heartbeats_testing, dtype=torch.float32)
if not isinstance(annotations_testing, torch.Tensor):
    annotations_testing = torch.tensor(annotations_testing, dtype=torch.long)

# Create DataLoader objects
train_dataset = TensorDataset(heartbeats_training, annotations_training)
test_dataset = TensorDataset(heartbeats_testing, annotations_testing)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the CNN model
num_classes = 4
cnn = CNN(num_classes)

# Define optimizer and loss function
optimizer = torch.optim.SGD(cnn.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
epochs = 20
loss_history = []
accuracy_history = []

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

    # Evaluate without calculating gradients
    cnn.eval()
    correct = 0
    with torch.no_grad():
        for heartbeats, labels in test_loader:
            logits = cnn(heartbeats)
            predicted = logits.argmax(dim=1)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / len(test_dataset)
    accuracy_history.append(accuracy)
    print(
        f"Epoch: {epoch + 1}. "
        f"Loss: {average_loss:.4f}. "
        f"Accuracy: {accuracy:.2f}%"
    )

# Evaluate the model and calculate metrics
cnn.eval()
all_labels = []
all_predictions = []

# Collect all true labels and predictions
with torch.no_grad():
    for heartbeats, labels in test_loader:
        logits = cnn(heartbeats)
        predicted = logits.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

# Generate classification report
report = classification_report(
    all_labels,
    all_predictions,
    target_names=["N", "S", "V", "F"],  # Replace with your class names
    zero_division=0  # Avoid division by zero warnings
)
print("\nClassification Report:")
print(report)