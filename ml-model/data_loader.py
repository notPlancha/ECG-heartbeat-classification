from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(r"C:\Users\marcu\Desktop\mit-bih-arrhythmia-database-1.0.0")

sys.path.insert(0, str(ROOT / "src"))
import data_understanding as du
from prepare_data import process_record_list

num_classes = len(du.CLASS_NAMES)
TRAINING_DATA = du.DS1_TRAIN_RECORDS
VALIDATION_DATA = du.DS1_VAL_RECORDS
TESTING_DATA = du.DS2_TEST_RECORDS


def _load_records(record_ids, split_name):
    X, y, _, _, _, _, per_record_stats = process_record_list(
        records=record_ids,
        data_dir=str(DATA_DIR),
        pre_r=90,
        post_r=166,
        lowcut=0.5,
        highcut=40.0,
        fs=360.0,
        filter_order=3,
        normalize=True,
        clip_range=(-3.0, 3.0),
        preferred_lead="MLII"
    )
    X = np.asarray(X[:, 0, :], dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    for stats in per_record_stats:
        print(
            f"Loaded record {stats['Record']}: "
            f"{stats['Total_Beats']} beats"
        )

    print(f"\n{split_name} dataset summary")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Total beats:", len(y))
    print("Class counts:", np.bincount(y, minlength=num_classes))

    return X, y


def load_training_data():
    return _load_records(TRAINING_DATA, "Training")


def load_validation_data():
    return _load_records(VALIDATION_DATA, "Validation")


def load_testing_data():
    return _load_records(TESTING_DATA, "Testing")


if __name__ == "__main__":
    X_train, y_train = load_training_data()