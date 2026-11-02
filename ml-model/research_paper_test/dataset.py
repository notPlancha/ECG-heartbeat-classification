
import numpy as np
import torch

from torch.utils.data import Dataset


# ============================================================
# CONFIGURATION
# ============================================================

NUM_CLASSES = 4


# ============================================================
# AUGMENTATION 1: RESAMPLING
# ============================================================

def resampling_augmentation(signal):

    length = len(signal)

    factor = np.random.uniform(0.75, 0.95)

    new_length = max(
        2,
        int(length * factor)
    )

    indices = np.linspace(
        0,
        length - 1,
        new_length
    ).astype(int)

    sampled = signal[indices]

    result = np.zeros(
        length,
        dtype=np.float32
    )

    result[:len(sampled)] = sampled

    return result


# ============================================================
# AUGMENTATION 2: SCALING
# ============================================================

def scaling_augmentation(signal):

    length = len(signal)

    interpolated = np.empty(
        2 * length - 1,
        dtype=np.float32
    )

    interpolated[::2] = signal

    interpolated[1::2] = (
        signal[:-1] + signal[1:]
    ) / 2

    start = np.random.randint(
        0,
        len(interpolated) - length + 1
    )

    return interpolated[
        start:start + length
    ].copy()


# ============================================================
# AUGMENTATION 3: CLIPPING
# ============================================================

def clipping_augmentation(signal):

    length = len(signal)

    left = np.random.randint(
        0,
        max(2, length // 10)
    )

    right = np.random.randint(
        0,
        max(2, length // 10)
    )

    result = np.zeros(
        length,
        dtype=np.float32
    )

    remaining = signal[left:length-right]

    result[:len(remaining)] = remaining

    return result


# ============================================================
# RANDOM AUGMENTATION
# ============================================================

def augment_signal(signal):

    operations = [
        resampling_augmentation,
        scaling_augmentation,
        clipping_augmentation
    ]

    operation = np.random.choice(operations)

    augmented = operation(signal)

    return np.clip(
        augmented,
        0,
        1
    ).astype(np.float32)


# ============================================================
# CLASS BALANCING
# ============================================================

def balance_training_data(
    X,
    y,
    random_state=42
):

    rng = np.random.default_rng(random_state)

    # Four classes: N=0, S=1, V=2, F=3

    class_indices = {
        c: np.where(y == c)[0]
        for c in range(NUM_CLASSES)
    }

    # Check missing classes

    missing_classes = [
        c
        for c, indices in class_indices.items()
        if len(indices) == 0
    ]

    if missing_classes:

        raise ValueError(
            f"Missing training classes: {missing_classes}. "
            "Check your training recordings and preprocessing."
        )

    print("\nOriginal training class distribution:")

    for c, indices in class_indices.items():

        print(
            f"Class {c}: {len(indices)}"
        )

    # Balance to the largest minority class

    target = max(
        len(indices)
        for c, indices in class_indices.items()
        if c != 0
    )

    print("\nBalancing target per class:", target)

    selected = []

    for c, indices in class_indices.items():

        if len(indices) >= target:

            chosen = rng.choice(
                indices,
                target,
                replace=False
            )

        else:

            chosen = rng.choice(
                indices,
                target,
                replace=True
            )

        selected.extend(chosen)

    selected = np.asarray(selected)

    rng.shuffle(selected)

    X_balanced = X[selected]

    y_balanced = y[selected]

    print("\nBalanced training class distribution:")

    for c in range(NUM_CLASSES):

        print(
            f"Class {c}: {np.sum(y_balanced == c)}"
        )

    return X_balanced, y_balanced


# ============================================================
# PYTORCH DATASET
# ============================================================

class ECGDataset(Dataset):

    def __init__(
        self,
        X,
        y,
        training=False,
        augment=True
    ):

        self.X = np.asarray(
            X,
            dtype=np.float32
        )

        self.y = np.asarray(
            y,
            dtype=np.int64
        )

        self.training = training

        self.augment = augment

        if len(self.X) != len(self.y):

            raise ValueError(
                "X and y must contain the same number of samples."
            )

        if len(self.y) > 0:

            if np.min(self.y) < 0 or np.max(self.y) >= NUM_CLASSES:

                raise ValueError(
                    "Labels must be between 0 and 3."
                )

    def __len__(self):

        return len(self.X)

    def __getitem__(self, index):

        signal = self.X[index].copy()

        label = self.y[index]

        # Augmentation only during training

        if self.training and self.augment:

            if np.random.random() < 0.5:

                signal = augment_signal(signal)

        signal = torch.tensor(
            signal,
            dtype=torch.float32
        ).unsqueeze(0)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return signal, label
