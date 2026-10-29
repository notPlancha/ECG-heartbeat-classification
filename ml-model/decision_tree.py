from data_loader import load_training_data, load_testing_data

import numpy as np
import matplotlib.pyplot as plt

from imblearn.over_sampling import RandomOverSampler

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score
)

heartbeats_training, annotations_training = load_training_data()
heartbeats_testing, annotations_testing = load_testing_data()

# Convert to NumPy arrays for scikit-learn
X_train = np.asarray(heartbeats_training, dtype=np.float32)
y_train = np.asarray(annotations_training, dtype=np.int64)

X_test = np.asarray(heartbeats_testing, dtype=np.float32)
y_test = np.asarray(annotations_testing, dtype=np.int64)

# Four heartbeat classes
num_classes = 4

class_labels = [0, 1, 2, 3]

class_names = [
    "N - Normal",
    "S - Supraventricular",
    "V - Ventricular",
    "F - Fusion"
]

original_counts = np.bincount(
    y_train,
    minlength=num_classes
)

print("Original training class counts:")

for name, count in zip(class_names, original_counts):
    print(f"{name}: {count}")

# Check that all four classes are present
if np.any(original_counts == 0):
    raise ValueError(
        "At least one class is missing from the training data."
    )


# Target count = number of samples in the largest class
target_count = int(original_counts.max())

# Increase every smaller class to the target count
sampling_strategy = {
    cls: target_count
    for cls in range(num_classes)
    if original_counts[cls] < target_count
}

oversampler = RandomOverSampler(
    sampling_strategy=sampling_strategy,
    random_state=42
)

X_train_resampled, y_train_resampled = (
    oversampler.fit_resample(X_train, y_train)
)


resampled_counts = np.bincount(
    y_train_resampled,
    minlength=num_classes
)

print("\nBalanced training class counts:")

for name, count in zip(class_names, resampled_counts):
    print(f"{name}: {count}")

print(
    "\nTotal original training samples:",
    len(y_train)
)

print(
    "Total resampled training samples:",
    len(y_train_resampled)
)


random_forest = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight=None,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest with balanced data...")

random_forest.fit(
    X_train_resampled,
    y_train_resampled
)

print("Training complete!")


# IMPORTANT: The test data is not oversampled
y_pred = random_forest.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=class_labels,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    labels=class_labels,
    average="macro",
    zero_division=0
)

print(f"Macro F1-score: {macro_f1:.4f}")

cm = confusion_matrix(
    y_test,
    y_pred,
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

plt.title("Random Forest - Full Oversampling")
plt.tight_layout()
plt.show()

importances = random_forest.feature_importances_

top_n = min(20, len(importances))

top_indices = np.argsort(importances)[-top_n:][::-1]

plt.figure(figsize=(10, 6))

plt.bar(
    range(top_n),
    importances[top_indices]
)

plt.xticks(
    range(top_n),
    top_indices,
    rotation=45
)

plt.xlabel("Heartbeat input feature index")
plt.ylabel("Feature importance")
plt.title(f"Random Forest - Top {top_n} Features")
plt.tight_layout()
plt.show()
