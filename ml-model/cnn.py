from data_loader import load_training_data, load_testing_data
from class_weighting import calculate_class_weights
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

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
# Convert data to tensors
heartbeats_training = torch.tensor(heartbeats_training, dtype=torch.float32)
annotations_training = torch.tensor(annotations_training, dtype=torch.long)
heartbeats_testing = torch.tensor(heartbeats_testing, dtype=torch.float32)
annotations_testing = torch.tensor(annotations_testing, dtype=torch.long)
# Calculate weights using the training labels only
num_classes = 4
class_weights = calculate_class_weights(
    annotations_training,
    num_classes
)
print("Class counts:", torch.bincount(
    annotations_training, minlength=num_classes
))
print("Class weights:", class_weights)
print("Beat list shape:", heartbeats_training.shape)
print("Annotation list shape:", annotations_training.shape)
print("First label:", annotations_training[0].item())
# Pair heartbeats with their labels
dataset = TensorDataset(heartbeats_training, annotations_training)
testingset = TensorDataset(heartbeats_testing, annotations_testing)
train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
)
test_loader = DataLoader(
    testingset,
    batch_size=64,
    shuffle=False,
)

# Each heartbeat has 256 samples; CNN adds a single input-channel dimension.
cnn = CNN(num_classes)
optimizer = torch.optim.SGD(cnn.parameters(), lr=0.001)
# Apply the class weights to the training loss
criterion = nn.CrossEntropyLoss(weight=class_weights)
epochs = 50
loss_history = []
accuracy_history = []
# Training loop
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
    average_loss = total_loss / len(dataset)
    loss_history.append(average_loss)

    # Evaluate without calculating gradients
    cnn.eval()
    correct = 0
    with torch.no_grad():
        for heartbeats, labels in test_loader:
            logits = cnn(heartbeats)
            predicted = logits.argmax(dim=1)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / len(testingset)
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