from data_loader import load_training_data, load_testing_data
from class_weighting import calculate_class_weights

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

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
    batch_size=64,  # Number of heartbeats per batch
    shuffle=True,
)

test_loader = DataLoader(
    testingset,
    batch_size=64,
    shuffle=False,
)

# Model: 256 input values and one output logit per class
inputs = 256
regression = nn.Linear(inputs, num_classes)

optimizer = torch.optim.SGD(regression.parameters(), lr=0.001)

# Apply the class weights to the training loss
criterion = nn.CrossEntropyLoss(weight=class_weights)

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

    # Evaluate without calculating gradients
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
plt.title("Training Loss")
plt.show()

plt.plot(accuracy_history)
plt.xlabel("Number of epochs")
plt.ylabel("Accuracy (%)")
plt.title("Test Accuracy")
plt.show()