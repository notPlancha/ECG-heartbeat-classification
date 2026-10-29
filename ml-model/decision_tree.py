from data_loader import load_training_data, load_testing_data

import numpy as np
import matplotlib.pyplot as plt

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    accuracy_score
)

# --------------------------------------------------
# 1. Load training and testing data
# --------------------------------------------------

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

# --------------------------------------------------
# 2. Inspect original class distribution
# --------------------------------------------------

original_counts = np.bincount(
    y_train,
    minlength=num_classes
)

print("Original training class counts:")

for name, count in zip(class_names, original_counts):
    print(f"{name}: {count}")

if np.any(original_counts == 0):
    raise ValueError(
        "At least one class is missing from the training data."
    )

# --------------------------------------------------
# 3. Undersample class N
# --------------------------------------------------

# Keep at most twice the original number of F samples
target_N = min(
    original_counts[0],
    int(original_counts[3] * 2)
)

print(f"\nTarget number of N samples: {target_N}")

under_sampler = RandomUnderSampler(
    sampling_strategy={0: target_N},
    random_state=42
)

X_train_under, y_train_under = under_sampler.fit_resample(
    X_train,
    y_train
)

under_counts = np.bincount(
    y_train_under,
    minlength=num_classes
)

print("\nClass counts after undersampling N:")

for name, count in zip(class_names, under_counts):
    print(f"{name}: {count}")

# --------------------------------------------------
# 4. Targeted oversampling
# --------------------------------------------------

# Targets are based on the original largest class
majority_count = int(original_counts.max())

sampling_strategy = {
    1: int(majority_count * 0.75),  # S: 75%
    2: int(majority_count * 0.25),  # V: 25%
    3: int(majority_count * 0.75)   # F: 75%
}

# Only oversample classes below their targets
sampling_strategy = {
    cls: target
    for cls, target in sampling_strategy.items()
    if under_counts[cls] < target
}

print("\nTargeted oversampling targets:")

for cls, target in sampling_strategy.items():
    print(f"{class_names[cls]}: {target}")

oversampler = RandomOverSampler(
    sampling_strategy=sampling_strategy,
    random_state=42
)

X_train_resampled, y_train_resampled = oversampler.fit_resample(
    X_train_under,
    y_train_under
)

resampled_counts = np.bincount(
    y_train_resampled,
    minlength=num_classes
)

print("\nFinal training class counts:")

for name, count in zip(class_names, resampled_counts):
    print(f"{name}: {count}")

print("\nTotal original training samples:", len(y_train))
print("Total final training samples:", len(y_train_resampled))

# --------------------------------------------------
# 5. Define the Random Forest
# --------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=500,
    max_features=0.2,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    max_samples=0.5,
    class_weight=None,
    random_state=42,
    n_jobs=-1
)

# --------------------------------------------------
# 6. Train the model
# --------------------------------------------------

print("\nTraining Random Forest with undersampling and targeted oversampling...")

random_forest.fit(
    X_train_resampled,
    y_train_resampled
)

print("Training complete!")

# --------------------------------------------------
# 7. Evaluate on the original, untouched test data
# --------------------------------------------------

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

# --------------------------------------------------
# 8. Confusion matrix
# --------------------------------------------------

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

plt.title("Random Forest - N Undersampling + Targeted Oversampling")
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 9. Feature importance
# --------------------------------------------------

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