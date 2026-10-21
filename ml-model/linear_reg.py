from data_loader import load_training_data

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader

heartbeats, annotations = load_training_data()

#transform to tensors
heartbeats = torch.tensor(heartbeats, dtype=torch.float32)
annotations = torch.tensor(annotations, dtype=torch.long)

#check if data is transformed properly in pytorch tensors
print("beat list shape: ", heartbeats.shape)
print ("annotation list shape", annotations.shape)
print("first label: ", annotations[0].item())

#join data
dataset = (TensorDataset(heartbeats, annotations))

train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
)
