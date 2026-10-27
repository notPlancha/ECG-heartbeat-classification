import torch

def calculate_class_weights(labels, num_classes):
    """
    Calculate inverse-frequency weights for multiclass classification.

    labels: training labels encoded as integers from 0 to num_classes - 1
    num_classes: total number of classes
    """
    print("Calculating class weights...")
    labels = torch.as_tensor(labels, dtype=torch.long).reshape(-1)

    if labels.numel() == 0:
        raise ValueError("Training labels are empty.")

    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(
            f"Labels must be integers from 0 to {num_classes - 1}."
        )

    counts = torch.bincount(labels, minlength=num_classes)

    if (counts == 0).any():
        missing = torch.where(counts == 0)[0].tolist()
        raise ValueError(f"No training examples for classes: {missing}")

    weights = counts.sum() / (num_classes * counts.float())
    return weights