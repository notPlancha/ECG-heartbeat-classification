from data_loader import load_training_data, load_testing_data

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

heartbeats_training, annotations_training = load_training_data()
heartbeats_testing, annotations_testing = load_testing_data()

#transform to tensors
heartbeats_training = torch.tensor(heartbeats_training, dtype=torch.float32)
annotations_training = torch.tensor(annotations_training, dtype=torch.long)

heartbeats_testing = torch.tensor(heartbeats_testing, dtype=torch.float32)
annotations_testing = torch.tensor(annotations_testing, dtype=torch.long)

#check if data is transformed properly in pytorch tensors
print("beat list shape: ", heartbeats_training.shape)
print ("annotation list shape", annotations_training.shape)
print("first label: ", annotations_training[0].item())

#touple data
dataset = (TensorDataset(heartbeats_training, annotations_training))
testingset = (TensorDataset(heartbeats_testing, annotations_testing))

train_loader = DataLoader(
    dataset,
    batch_size=64, #number of heartbeats per epoch
    shuffle=True,
)

test_loader = DataLoader(
    testingset,
    batch_size=64,
    shuffle=False
)

#data validation set 20% of training data

inputs = 256 #number of signals per heartbeat
outputs = 4  #number of classifications

regression = nn.Linear(inputs, outputs)

optimizer = torch.optim.SGD(regression.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

epochs = 50
loss = []
acc = []

for epoch in range(epochs):
    total_loss = 0.0
    for heartbeats, labels in train_loader:
        
        optimizer.zero_grad()
        outputs = regression(heartbeats)
        batch_loss = criterion(outputs, labels)

        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item() * len(labels)

    average_loss = total_loss / len(heartbeats_training)
    loss.append(average_loss)
    correct = 0

    for heartbeats, labels in test_loader:
        outputs = regression(heartbeats)
        _, predicted = torch.max(outputs.data, 1)

        correct += (predicted == labels).sum()

    accuracy = 100 * (correct.item()) / len(heartbeats_testing)
    acc.append(accuracy)
    print('Epoch: {}. Loss: {:.4f}. Accuracy: {:.2f}%'.format(epoch + 1, average_loss, accuracy))

plt.plot(loss)
plt.xlabel("Number of epochs")
plt.ylabel("Average loss")
plt.title("Training Loss")
plt.show()

plt.plot(acc)
plt.xlabel("no. of epochs")
plt.ylabel("total accuracy")
plt.title("Accuracy")
plt.show()