
from data_loader import (
    load_training_data,
    load_validation_data,
    load_testing_data,
)
from int8_export import quantize_linear_parameters, predict_int8_linear

import torch
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    WeightedRandomSampler
)
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

heartbeats_training, annotations_training = load_training_data()
heartbeats_validation, annotations_validation = load_validation_data()
heartbeats_testing, annotations_testing = load_testing_data()

# Convert data to tensors
heartbeats_training = torch.tensor(
    heartbeats_training, dtype=torch.float32
)
annotations_training = torch.tensor(
    annotations_training, dtype=torch.long
)

heartbeats_validation = torch.tensor(
    heartbeats_validation, dtype=torch.float32
)
annotations_validation = torch.tensor(
    annotations_validation, dtype=torch.long
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

validationset = TensorDataset(
    heartbeats_validation,
    annotations_validation
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

# Do not oversample the validation or test sets
validation_loader = DataLoader(
    validationset,
    batch_size=64,
    shuffle=False
)

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
        for heartbeats, labels in validation_loader:
            logits = regression(heartbeats)
            predicted = logits.argmax(dim=1)

            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / len(validationset)
    accuracy_history.append(accuracy)

    print(
        f"Epoch: {epoch + 1}. "
        f"Loss: {average_loss:.4f}. "
        f"Validation Accuracy: {accuracy:.2f}%"
    )


plt.plot(loss_history)
plt.xlabel("Number of epochs")
plt.ylabel("Average loss")
plt.title("Training Loss with Oversampling")
plt.tight_layout()
plt.show()


plt.plot(accuracy_history)
plt.xlabel("Number of epochs")
plt.ylabel("Validation Accuracy (%)")
plt.title("Validation Accuracy with Oversampling")
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

class_names = [
    "N - Normal",
    "S - Supraventricular",
    "V - Ventricular",
    "F - Fusion"
]


def calculate_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.size == 0:
        raise ValueError("Cannot calculate test metrics for an empty test set.")
    if labels.shape != predictions.shape:
        raise ValueError("Labels and predictions must have the same shape.")
    if (
        np.any((labels < 0) | (labels >= num_classes))
        or np.any((predictions < 0) | (predictions >= num_classes))
    ):
        raise ValueError(
            f"Labels and predictions must be in [0, {num_classes - 1}]."
        )

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (labels, predictions), 1)

    true_positives = np.diag(cm)
    actual_counts = cm.sum(axis=1)
    predicted_counts = cm.sum(axis=0)
    precision = np.divide(
        true_positives,
        predicted_counts,
        out=np.zeros(num_classes, dtype=np.float64),
        where=predicted_counts != 0,
    )
    recall = np.divide(
        true_positives,
        actual_counts,
        out=np.zeros(num_classes, dtype=np.float64),
        where=actual_counts != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes, dtype=np.float64),
        where=(precision + recall) != 0,
    )

    accuracy = float(true_positives.sum() / labels.size)
    return cm, accuracy, precision, recall, f1


def print_metrics(model_name, labels, predictions):
    cm, accuracy, precision, recall, f1 = calculate_metrics(
        labels, predictions
    )
    print(f"\n{model_name} test metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    macro_f1 = float(f1.mean())
    print(f"Macro F1-score: {macro_f1:.4f}")
    print("\nPer-class metrics:")
    print(
        f"{'Class':<24} {'Precision':>10} "
        f"{'Recall/Sensitivity':>20} {'F1-score':>10}"
    )
    for name, class_precision, class_recall, class_f1 in zip(
        class_names, precision, recall, f1
    ):
        print(
            f"{name:<24} {class_precision:>10.4f} "
            f"{class_recall:>20.4f} {class_f1:>10.4f}"
        )
    print("Confusion matrix:")
    print(cm)
    return cm, accuracy, macro_f1


def predict_fp32_edge(inputs):
    """Run the FP32 linear layer through a NumPy edge-style inference path."""
    weights = regression.weight.detach().cpu().numpy()
    bias = regression.bias.detach().cpu().numpy()
    return np.asarray(inputs, dtype=np.float32) @ weights.T + bias


test_signals = heartbeats_testing.numpy()
quantized_model = quantize_linear_parameters(
    weights=regression.weight.detach().cpu().numpy(),
    bias=regression.bias.detach().cpu().numpy(),
    input_range=(-3.0, 3.0),
)


def report_model_metrics(condition, labels, predictions_by_model):
    metrics_by_model = {}
    for model_name, predictions in predictions_by_model.items():
        _, accuracy, _, _, f1 = calculate_metrics(labels, predictions)
        macro_f1 = float(f1.mean())
        metrics_by_model[model_name] = (accuracy, macro_f1)
    print(
        f"{condition:<12}"
        + "".join(
            f"{accuracy:>13.4f}{macro_f1:>13.4f}"
            for accuracy, macro_f1 in metrics_by_model.values()
        )
    )
    return metrics_by_model


baseline_predictions = np.asarray(all_predictions, dtype=np.int64)
fp32_edge_predictions = predict_fp32_edge(test_signals).argmax(axis=1)
int8_predictions = predict_int8_linear(
    test_signals, quantized_model
).argmax(axis=1)

print("\nClean test-set metrics:")
test_cm, _, baseline_macro_f1 = print_metrics(
    "Baseline (PyTorch FP32)", all_labels, baseline_predictions
)
_, _, fp32_edge_macro_f1 = print_metrics(
    "FP32 edge-style (NumPy)", all_labels, fp32_edge_predictions
)
_, _, int8_macro_f1 = print_metrics(
    "Quantized (INT8)", all_labels, int8_predictions
)

print("\nClean macro F1 comparison:")
print(f"Baseline (PyTorch FP32): {baseline_macro_f1:.4f}")
print(f"FP32 edge-style (NumPy): {fp32_edge_macro_f1:.4f}")
print(f"Quantized (INT8):        {int8_macro_f1:.4f}")

model_names = (
    "Baseline",
    "FP32 Edge",
    "INT8",
)
print("\nAccuracy and macro F1 by noise level:")
print(
    f"{'Condition':<12}"
    + "".join(f"{name + ' Acc':>13}{name + ' F1':>13}" for name in model_names)
)
report_model_metrics(
    "Clean",
    all_labels,
    {
        "Baseline": baseline_predictions,
        "FP32 Edge": fp32_edge_predictions,
        "INT8": int8_predictions,
    },
)

rng = np.random.default_rng(seed=42)
standard_noise = rng.normal(size=test_signals.shape).astype(np.float32)
signal_power = np.mean(np.square(test_signals), axis=1, keepdims=True)
noise_power = np.mean(np.square(standard_noise), axis=1, keepdims=True)

for snr_db in (12, 6, 0):
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noisy_signals = test_signals + standard_noise * np.sqrt(
        target_noise_power / noise_power
    )
    with torch.no_grad():
        baseline_noisy_predictions = regression(
            torch.from_numpy(noisy_signals)
        ).argmax(dim=1).cpu().numpy()
    fp32_edge_noisy_predictions = predict_fp32_edge(
        noisy_signals
    ).argmax(axis=1)
    int8_noisy_predictions = predict_int8_linear(
        noisy_signals, quantized_model
    ).argmax(axis=1)
    report_model_metrics(
        f"{snr_db} dB",
        all_labels,
        {
            "Baseline": baseline_noisy_predictions,
            "FP32 Edge": fp32_edge_noisy_predictions,
            "INT8": int8_noisy_predictions,
        },
    )

plt.imshow(test_cm, interpolation="nearest", cmap="Blues")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=35, ha="right")
plt.yticks(tick_marks, class_names)
threshold = test_cm.max() / 2
for row in range(test_cm.shape[0]):
    for column in range(test_cm.shape[1]):
        plt.text(
            column,
            row,
            f"{test_cm[row, column]:d}",
            ha="center",
            va="center",
            color="white" if test_cm[row, column] > threshold else "black",
        )
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.title("Baseline Logistic Regression - Test Confusion Matrix")
plt.tight_layout()
plt.show()