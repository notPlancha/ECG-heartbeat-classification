
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import (
    TensorDataset,
    DataLoader,
    WeightedRandomSampler,
)

from data_loader import (
    load_testing_data,
    load_training_data,
    load_validation_data,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

CLASS_NAMES = {
    0: "N",
    1: "S",
    2: "V",
    3: "F"
}

NUM_CLASSES = 4
BATCH_SIZE = 64
SNR_LEVELS_DB = (12, 6, 0)
NOISE_SEED = 42
MAX_NOISY_EXAMPLES = 6
TRAINING_EPOCHS = 50
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Path to the model saved by logistic_reg.py
ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT
    / "results"
    / "logistic_regression"
    / "logistic_regression.pth"
)


# --------------------------------------------------
# Load test data
# --------------------------------------------------

def load_test_dataloader():

    X_test, y_test = load_testing_data()

    X_test = torch.as_tensor(
        X_test,
        dtype=torch.float32
    )

    y_test = torch.as_tensor(
        y_test,
        dtype=torch.long
    ).reshape(-1)

    print("\nTest dataset information:")
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("Number of beats:", len(X_test))
    print("Number of labels:", len(y_test))

    if len(X_test) != len(y_test):
        raise ValueError(
            f"Mismatch: {len(X_test)} beats but "
            f"{len(y_test)} labels."
        )

    if X_test.ndim != 2:
        raise ValueError(
            "Expected test data with shape [beats, samples]."
        )

    dataset = TensorDataset(X_test, y_test)

    test_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return test_loader


# --------------------------------------------------
# Load trained logistic regression model
# --------------------------------------------------

def load_trained_model(input_size):
    """
    Load the saved logistic regression model or train it if missing.
    """

    if not MODEL_PATH.exists():
        print(
            f"No trained model found at {MODEL_PATH}. "
            "Training logistic regression before evaluation."
        )
        return train_logistic_regression(input_size)

    model = nn.Linear(
        input_size,
        NUM_CLASSES
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model = model.to(DEVICE)
    model.eval()

    print("\nTrained model loaded successfully.")
    print("Model input size:", input_size)
    print("Model output classes:", NUM_CLASSES)
    print("Model path:", MODEL_PATH)

    return model


def train_logistic_regression(input_size, epochs=TRAINING_EPOCHS):
    """Train the balanced linear classifier used by logistic_reg.py."""
    X_train, y_train = load_training_data()
    X_validation, y_validation = load_validation_data()

    X_train = torch.as_tensor(X_train, dtype=torch.float32)
    y_train = torch.as_tensor(y_train, dtype=torch.long).reshape(-1)
    X_validation = torch.as_tensor(X_validation, dtype=torch.float32)
    y_validation = torch.as_tensor(
        y_validation,
        dtype=torch.long,
    ).reshape(-1)

    if X_train.ndim != 2 or X_train.shape[1] != input_size:
        raise ValueError(
            f"Expected training data with shape [beats, {input_size}], "
            f"got {tuple(X_train.shape)}."
        )
    if len(X_train) != len(y_train) or len(X_validation) != len(y_validation):
        raise ValueError("Training and validation beats must match their labels.")
    if X_validation.ndim != 2 or X_validation.shape[1] != input_size:
        raise ValueError(
            f"Expected validation data with shape [beats, {input_size}], "
            f"got {tuple(X_validation.shape)}."
        )
    if len(X_train) == 0 or len(X_validation) == 0:
        raise ValueError("Training and validation datasets must not be empty.")
    if torch.any((y_train < 0) | (y_train >= NUM_CLASSES)):
        raise ValueError(f"Training labels must be in [0, {NUM_CLASSES - 1}].")
    if torch.any((y_validation < 0) | (y_validation >= NUM_CLASSES)):
        raise ValueError(
            f"Validation labels must be in [0, {NUM_CLASSES - 1}]."
        )

    class_counts = torch.bincount(y_train, minlength=NUM_CLASSES)
    if torch.any(class_counts == 0):
        raise ValueError(
            "Balanced training requires at least one training beat "
            "for every class."
        )

    train_dataset = TensorDataset(X_train, y_train)
    validation_dataset = TensorDataset(X_validation, y_validation)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
        generator=torch.Generator().manual_seed(NOISE_SEED),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    torch.manual_seed(NOISE_SEED)
    model = nn.Linear(input_size, NUM_CLASSES).to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(
        f"Training logistic regression for {epochs} epochs "
        f"on {len(train_dataset)} beats."
    )
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for beats, labels in train_loader:
            beats = beats.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(beats), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        model.eval()
        correct = 0
        with torch.no_grad():
            for beats, labels in validation_loader:
                predictions = model(beats.to(DEVICE)).argmax(dim=1)
                correct += (predictions.cpu() == labels).sum().item()

        validation_accuracy = correct / len(validation_dataset)
        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"loss={total_loss / len(train_dataset):.4f}, "
            f"validation accuracy={validation_accuracy:.2%}"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved trained model: {MODEL_PATH}")
    model.eval()
    return model


# --------------------------------------------------
# Get predictions
# --------------------------------------------------

def get_predictions(model, test_loader, device=DEVICE):
    """
    Run model inference using the test DataLoader.
    """

    model = model.to(device)
    model.eval()

    all_beats = []
    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for beats, labels in test_loader:

            beats = beats.to(device)

            outputs = model(beats)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_beats.append(beats.cpu())
            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())

    X = torch.cat(all_beats).numpy()
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_predictions).numpy()

    return X, y_true, y_pred


# --------------------------------------------------
# Plot incorrectly classified ECG beats
# --------------------------------------------------

def plot_error_beats(
    X,
    y_true,
    y_pred,
    condition_name,
    max_examples=5
):
    """
    Plot representative incorrectly classified ECG beats.
    """

    X = np.asarray(X)
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if not (
        len(X) == len(y_true) == len(y_pred)
    ):
        raise ValueError(
            "X, y_true and y_pred must contain the same "
            "number of samples."
        )

    error_pairs = [
        (0, 1),  # N -> S
        (1, 0),  # S -> N
        (2, 0),  # V -> N
        (2, 1),  # V -> S
        (2, 3),  # V -> F
        (3, 0),  # F -> N
        (3, 1),  # F -> S
        (3, 2),  # F -> V
    ]

    output_dir = (
        ROOT
        / "results"
        / "error_analysis"
        / condition_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"\nAnalysing errors: {condition_name}")

    for true_class, predicted_class in error_pairs:

        indices = np.where(
            (y_true == true_class) &
            (y_pred == predicted_class)
        )[0]

        if len(indices) == 0:
            print(
                f"{CLASS_NAMES[true_class]} -> "
                f"{CLASS_NAMES[predicted_class]}: "
                "No errors found."
            )
            continue

        selected_indices = indices[:max_examples]

        fig, axes = plt.subplots(
            len(selected_indices),
            1,
            figsize=(12, 2.5 * len(selected_indices)),
            squeeze=False
        )

        fig.suptitle(
            f"{condition_name}: "
            f"{CLASS_NAMES[true_class]} predicted as "
            f"{CLASS_NAMES[predicted_class]}\n"
            f"Total errors: {len(indices)}"
        )

        for row, sample_index in enumerate(selected_indices):

            axes[row, 0].plot(X[sample_index])

            axes[row, 0].set_title(
                f"Test beat index: {sample_index}"
            )

            axes[row, 0].set_xlabel("Sample")
            axes[row, 0].set_ylabel("Amplitude")
            axes[row, 0].grid(True, alpha=0.3)

        plt.tight_layout()

        filename = (
            f"{CLASS_NAMES[true_class]}_as_"
            f"{CLASS_NAMES[predicted_class]}.png"
        )

        plt.savefig(
            output_dir / filename,
            dpi=150
        )

        plt.close(fig)

        print(
            f"{CLASS_NAMES[true_class]} -> "
            f"{CLASS_NAMES[predicted_class]}: "
            f"{len(indices)} errors "
            f"(plotted {len(selected_indices)})"
        )

    print(f"Plots saved in: {output_dir}")


# --------------------------------------------------
# Analyse model
# --------------------------------------------------

def analyse_model(
    model,
    condition_name,
    test_loader=None,
    device=DEVICE
):
    """
    Run inference and plot incorrectly classified beats.
    """

    if test_loader is None:
        test_loader = load_test_dataloader()

    X, y_true, y_pred = get_predictions(
        model,
        test_loader,
        device
    )

    plot_error_beats(
        X,
        y_true,
        y_pred,
        condition_name
    )

    return y_true, y_pred


def add_gaussian_noise(X, snr_levels_db=SNR_LEVELS_DB, seed=NOISE_SEED):
    """Create reproducible per-beat Gaussian-noise conditions."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError("X must contain beats with shape [beats, samples].")

    rng = np.random.default_rng(seed)
    standard_noise = rng.normal(size=X.shape).astype(np.float32)
    signal_power = np.mean(np.square(X), axis=1, keepdims=True)
    noise_power = np.mean(np.square(standard_noise), axis=1, keepdims=True)
    if np.any(signal_power <= 0) or np.any(noise_power <= 0):
        raise ValueError("Cannot add SNR-scaled noise to a zero-power beat.")

    noisy_conditions = {}
    for snr_db in snr_levels_db:
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        scaled_noise = standard_noise * np.sqrt(
            target_noise_power / noise_power
        )
        noisy_conditions[f"{snr_db} dB"] = X + scaled_noise

    return noisy_conditions


def _select_error_indices(y_true, y_pred, max_examples):
    """Select representative mistakes across different class pairs."""
    error_pairs = [
        (true_class, (true_class + offset) % NUM_CLASSES)
        for offset in range(1, NUM_CLASSES)
        for true_class in range(NUM_CLASSES)
    ]
    selected = []

    for true_class, predicted_class in error_pairs:
        indices = np.flatnonzero(
            (y_true == true_class) & (y_pred == predicted_class)
        )
        if indices.size:
            selected.append(int(indices[0]))
        if len(selected) == max_examples:
            break

    if len(selected) < max_examples:
        selected_set = set(selected)
        for sample_index in np.flatnonzero(y_true != y_pred):
            if int(sample_index) not in selected_set:
                selected.append(int(sample_index))
                if len(selected) == max_examples:
                    break

    return selected


def plot_noise_error_overview(
    X,
    y_true,
    clean_predictions,
    predictions_by_condition,
):
    """Save confusion matrices and representative errors in one figure."""
    X = np.asarray(X)
    y_true = np.asarray(y_true).reshape(-1)
    if X.ndim != 2 or len(X) != len(y_true):
        raise ValueError("X and y_true must contain the same number of beats.")

    conditions = {
        "Clean": (X, clean_predictions),
        **predictions_by_condition,
    }
    for condition_name, (condition_X, predictions) in conditions.items():
        condition_X = np.asarray(condition_X)
        predictions = np.asarray(predictions).reshape(-1)
        if condition_X.shape != X.shape or len(predictions) != len(y_true):
            raise ValueError(
                f"{condition_name} signals and predictions must match "
                "the clean test data."
            )
        conditions[condition_name] = (condition_X, predictions)

    columns = MAX_NOISY_EXAMPLES + 1
    fig, axes = plt.subplots(
        len(conditions),
        columns,
        figsize=(19, 3.5 * len(conditions)),
        squeeze=False,
    )
    fig.suptitle(
        "Misclassified ECG beats under Gaussian noise\n"
        "Confusion matrices include every test beat; waveform panels show "
        "representative errors",
        fontsize=14,
    )
    error_counts = []

    for row, (condition_name, (condition_X, predictions)) in enumerate(
        conditions.items()
    ):
        confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        np.add.at(confusion, (y_true, predictions), 1)

        matrix_axis = axes[row, 0]
        matrix_axis.imshow(confusion, cmap="Blues")
        matrix_axis.set(
            title=f"{condition_name}\nConfusion matrix",
            xlabel="Predicted",
            ylabel="True",
        )
        matrix_axis.set_xticks(range(NUM_CLASSES), CLASS_NAMES.values())
        matrix_axis.set_yticks(range(NUM_CLASSES), CLASS_NAMES.values())
        threshold = confusion.max() / 2
        for true_class in range(NUM_CLASSES):
            for predicted_class in range(NUM_CLASSES):
                count = confusion[true_class, predicted_class]
                matrix_axis.text(
                    predicted_class,
                    true_class,
                    str(count),
                    ha="center",
                    va="center",
                    color="white" if count > threshold else "black",
                    fontsize=8,
                )

        error_count = int(np.count_nonzero(y_true != predictions))
        error_counts.append(error_count)
        error_rate = error_count / len(y_true)
        print(
            f"{condition_name}: {error_count} misclassified beats "
            f"out of {len(y_true)} ({error_rate:.1%})"
        )
        selected_indices = _select_error_indices(
            y_true, predictions, MAX_NOISY_EXAMPLES
        )

        for column, sample_index in enumerate(
            selected_indices, start=1
        ):
            axis = axes[row, column]
            true_class = int(y_true[sample_index])
            predicted_class = int(predictions[sample_index])
            axis.plot(condition_X[sample_index], linewidth=0.9)
            axis.axvline(90, color="gray", linestyle="--", linewidth=0.8)
            axis.set_title(
                f"{CLASS_NAMES[true_class]} → {CLASS_NAMES[predicted_class]}\n"
                f"Beat {sample_index}",
                fontsize=9,
            )
            axis.set_xlabel("Sample")
            axis.grid(True, alpha=0.25)

        for column in range(len(selected_indices) + 1, columns):
            axes[row, column].set_visible(False)

    output_dir = ROOT / "results" / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "noise_misclassified_beats.png"
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Noise error overview saved to: {output_path}")

    rate_path = output_dir / "noise_misclassification_rate.png"
    rates = np.asarray(error_counts, dtype=np.float64) / len(y_true) * 100
    rate_fig, rate_axis = plt.subplots(figsize=(9, 5.5))
    bars = rate_axis.bar(
        list(conditions),
        rates,
        color=["#4C78A8", "#72B7B2", "#F2CF5B", "#E45756"],
        width=0.65,
    )
    rate_axis.set(
        title="How often are beats misclassified as noise increases?",
        xlabel="Noise condition",
        ylabel="Misclassified test beats (%)",
        ylim=(0, 100),
    )
    rate_axis.grid(axis="y", alpha=0.3)
    rate_axis.set_axisbelow(True)
    for bar, count, rate in zip(bars, error_counts, rates):
        rate_axis.annotate(
            f"{count:,} beats\n{rate:.1f}%",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    rate_fig.tight_layout()
    rate_fig.savefig(rate_path, dpi=180)
    plt.close(rate_fig)
    print(f"Noise misclassification chart saved to: {rate_path}")

    class_rate_path = output_dir / "noise_misclassification_by_class.png"
    class_fig, class_axes = plt.subplots(2, 2, figsize=(14, 10))
    condition_names = list(conditions)
    colors = ["#4C78A8", "#72B7B2", "#F2CF5B", "#E45756"]
    x_positions = np.arange(len(condition_names))

    for class_index, axis in enumerate(class_axes.flat):
        class_mask = y_true == class_index
        class_total = int(np.count_nonzero(class_mask))
        class_error_counts = [
            int(np.count_nonzero(class_mask & (predictions != class_index)))
            for _, predictions in conditions.values()
        ]
        class_error_rates = (
            np.asarray(class_error_counts, dtype=np.float64) / class_total * 100
            if class_total
            else np.zeros(len(condition_names), dtype=np.float64)
        )

        bars = axis.bar(x_positions, class_error_rates, color=colors, width=0.65)
        axis.set(
            title=f"Class {CLASS_NAMES[class_index]} "
            f"({class_total:,} test beats)",
            xlabel="Noise condition",
            ylabel="Misclassified beats in class (%)",
            ylim=(0, 110),
            xticks=x_positions,
            xticklabels=condition_names,
        )
        axis.grid(axis="y", alpha=0.3)
        axis.set_axisbelow(True)

        for condition_name, bar, count, rate in zip(
            condition_names, bars, class_error_counts, class_error_rates
        ):
            axis.annotate(
                f"{count:,} beats\n{rate:.1f}%",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
            print(
                f"Class {CLASS_NAMES[class_index]}, {condition_name}: "
                f"{count} misclassified out of {class_total} ({rate:.1f}%)"
            )

    class_fig.suptitle(
        "Misclassification rate by heartbeat class and noise level",
        fontsize=15,
    )
    class_fig.tight_layout(rect=(0, 0, 1, 0.96))
    class_fig.savefig(class_rate_path, dpi=180)
    plt.close(class_fig)
    print(f"Per-class noise misclassification chart saved to: {class_rate_path}")

# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    print("Loading test data...")

    test_loader = load_test_dataloader()

    print(
        f"\nTest beats loaded: "
        f"{len(test_loader.dataset)}"
    )

    # Determine input size from the loaded ECG beats
    input_size = test_loader.dataset.tensors[0].shape[1]

    # Load the previously trained model
    model = load_trained_model(input_size)

    # Analyse classification errors
    y_true, clean_predictions = analyse_model(
        model=model,
        condition_name="Logistic_Regression_Clean",
        test_loader=test_loader,
        device=DEVICE
    )

    X_test = test_loader.dataset.tensors[0].numpy()
    noisy_conditions = add_gaussian_noise(X_test)
    noisy_predictions = {}
    for condition_name, noisy_X in noisy_conditions.items():
        noisy_loader = DataLoader(
            TensorDataset(
                torch.as_tensor(noisy_X, dtype=torch.float32),
                torch.as_tensor(y_true, dtype=torch.long),
            ),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
        _, _, noisy_predictions[condition_name] = get_predictions(
            model,
            noisy_loader,
            DEVICE,
        )

    plot_noise_error_overview(
        X_test,
        y_true,
        clean_predictions,
        {
            condition_name: (noisy_conditions[condition_name], predictions)
            for condition_name, predictions in noisy_predictions.items()
        },
    )

    print("\nError analysis completed.")
