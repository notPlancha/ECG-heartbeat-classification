# ECG Heartbeat Classification on Edge-AI

A production-grade, reproducible solution for **ECG Heartbeat Classification on Edge-AI**, targeting the **Analog Devices MAX78002** ultra-low-power convolutional microcontroller and adhering to the **CRISP-DM methodology**.

---

## 📋 Project Overview

The goal of this project is to classify single-lead ECG heartbeats into clinically actionable categories while adhering to the hardware constraints of the **Analog Devices MAX78002 Edge-AI microcontroller**:
- **Target Platform**: Analog Devices MAX78002 (Cortex-M4 + Hardware CNN Accelerator)
- **Constraint Budget**: Maximum model size $\le 2\text{ MB}$, fixed-point INT8 computation, and strictly supported operations (1D convolution, pooling, ReLU, linear layers).
- **Classification Task**: 4-class heartbeat classification following the **ANSI/AAMI EC57** standard:
  - **N**: Normal-type and bundle branch block beats
  - **S**: Supraventricular ectopic beats (SVEB)
  - **V**: Ventricular ectopic beats (VEB)
  - **F**: Fusion of ventricular and normal beats
- **Data Source**: MIT-BIH Arrhythmia Database ($360\text{ Hz}$, two-channel ambulatory ECG).

---

## 📁 Repository Structure

```text
ECG-heartbeat-classification/
│
├── README.md                      # Complete project documentation and workflow instructions
├── requirements.txt               # Fully resolved, locked Python package dependencies
├── pyproject.toml                 # Project metadata and dependencies (managed via uv)
├── justfile                       # Convenient task automation recipes (install, prepare-data, run-eda)
│
├── dataset/
│   ├── physionet.org/             # Raw MIT-BIH Arrhythmia Database files (.hea, .dat, .atr)
│   └── processed/                 # Preprocessed, segmented Edge-AI datasets (generated)
│       ├── train.npz              # Training set: 40,411 beats (17 DS1 records)
│       ├── val.npz                # Validation set: 10,583 beats (5 DS1 records)
│       └── test.npz               # Held-out test set: 49,685 beats (22 DS2 records)
│
├── notebooks/
│   ├── EDA.ipynb                  # Fully executed, comprehensive Data Understanding (CRISP-DM Stage 2)
│   └── play-around.ipynb          # Scratch notebook for quick tests
│
├── src/
│   ├── data_understanding.py      # Core modular utilities for record loading, filtering, and AAMI mapping
│   ├── prepare_data.py            # Reproducible preprocessing and dataset segmentation pipeline
│   ├── dataset.py                 # Flexible NumPy/PyTorch Dataset loader with training augmentations
│   ├── model.py                   # MAX78002-compatible CNN architecture (for Stage 4)
│   ├── train.py                   # Model training script (for Stage 4)
│   ├── evaluate.py                # Performance and confusion matrix evaluation (for Stage 5)
│   └── noise_test.py              # Noise robustness evaluation at 12dB, 6dB, 0dB (for Stage 5)
│
├── configs/
│   └── preprocessing_config.yaml  # Configurable filter, windowing, and split parameters
│
└── results/
    ├── dataset_statistics.csv     # Summary table of beat counts by split and class
    ├── dataset_statistics.json    # Machine-readable detailed split metrics and exclusions
    ├── per_record_distribution.csv# Per-patient class counts illustrating heterogeneity
    └── data_quality_report.csv    # Quality audit (missing data, amplitudes, baseline wander)
```

---

## ⚡ Quick Start

This project uses [`uv`](https://github.com/astral-sh/uv) and [`just`](https://github.com/casey/just) for reproducible environment and task management.

### 1. Installation
```powershell
# Install all locked dependencies into the virtual environment
just install
# Or via uv:
uv sync
```

### 2. Run Data Preprocessing (Stages 2 & 3)
```powershell
# Executes bandpass filtering, beat segmentation, normalization, and splits export
just prepare-data
# Or via python directly:
uv run python src/prepare_data.py
```

### 3. Run or Launch the Exploratory Data Analysis (EDA) Notebook
```powershell
# Re-execute the EDA notebook in place:
just run-eda

# Or launch Jupyter Lab / Notebook interactively:
just lab
# or:
just notebook
```

---

## 🔬 Data Understanding & Preparation Pipeline

### 1. Patient-Independent Partitioning (Zero Data Leakage)
A common error in ECG research is randomly shuffling heartbeats from the same subject between training and test sets. Because each subject possesses an **ECG biometric fingerprint** (unique ST-T morphology and conduction characteristics), random splitting causes models to memorize patient identities rather than learn pathology, achieving artificially high ($98\%+$) accuracy that collapses on new patients.

To guarantee zero data leakage, we enforce the **inter-patient MIT-BIH benchmark protocol (de Chazal et al., 2004)**:
- **DS1 — Development Set (22 records)**: Used exclusively for training, validation, and hyperparameter selection.
  - **Train (17 records)**: `109, 112, 114, 115, 116, 119, 122, 124, 201, 203, 207, 208, 209, 215, 220, 223, 230`
  - **Validation (5 records)**: `101, 106, 108, 118, 205` (selected to ensure all 4 classes are well represented).
- **DS2 — Final Test Set (22 records)**: `100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234`. Kept completely untouched until final model testing.
- **Paced Records Excluded (4 records)**: `102, 104, 107, 217` are excluded per AAMI guidelines due to artificial pacemaker artifacts.

### 2. Lead Selection
- **Primary Lead**: Modified Lead II (**MLII**) is chosen for all 44 records. MLII aligns with the primary ventricular electrical axis, providing clear QRS complexes, P-waves, and T-waves.
- **Channel Swap Handling**: In Record `114`, the lead order is reversed (Channel 0 is `V5`, Channel 1 is `MLII`). Our pipeline dynamically detects and extracts the correct channel.

### 3. Signal Filtering
- **Bandpass Filter**: 2nd/3rd-order zero-phase Butterworth filter ($0.5 - 40.0\text{ Hz}$).
- **Removal**: Eliminates low-frequency baseline wander ($< 0.5\text{ Hz}$) from respiration and electrode movement, and suppresses high-frequency EMG muscle tremors and $50/60\text{ Hz}$ powerline hum without phase distortion.

### 4. Beat Segmentation & Windowing for MAX78002
- **Sampling Frequency**: $360\text{ Hz}$ native.
- **Window Length**: **$256$ samples** ($711.1\text{ ms}$):
  - **Pre-R window**: $90$ samples ($250.0\text{ ms}$), capturing P-wave and PR interval.
  - **Post-R window**: $166$ samples ($461.1\text{ ms}$), capturing QRS complex, ST segment, and T-wave.
  - **Edge-AI Benefit**: $256 = 2^8$ samples allows clean dyadic pooling ($256 \to 128 \to 64 \to 32 \to 16 \to 8$) on the MAX78002 CNN hardware accelerator without fractional padding.
- **Normalization & Outlier Clamping**: Per-beat Z-score normalization clamped to $[-3.0, 3.0]$ standard deviations to suppress extreme motion spikes and prevent INT8 quantization overflow.

---

## 📊 Dataset Statistics & Class Distributions

The exact beat counts generated by `src/prepare_data.py` (matching Section 3.8 of the project rubric) are:

| Split | Records | N (Normal) | S (Supraventricular) | V (Ventricular) | F (Fusion) | Total Heartbeats |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 17 | 36,008 | 838 | 3,164 | 401 | **40,411** |
| **Validation** | 5 | 9,840 | 106 | 624 | 13 | **10,583** |
| **Test (DS2)** | 22 | 44,240 | 1,837 | 3,220 | 388 | **49,685** |
| **Total** | **44** | **90,088** | **2,781** | **7,008** | **802** | **100,679** |

### Key Observations
1. **Severe Class Imbalance**: Class N accounts for $\approx 89.5\%$ of all beats, while minority class F represents only $\approx 0.8\%$. Models must use class weighting or balanced batching (`src/dataset.py`).
2. **Patient Heterogeneity**: Class F beats are heavily concentrated in Record `208` (373 F beats). Our split guarantees both Train (401 F) and Validation (13 F) contain all 4 classes.

---

## 🛠️ Data Loading & Augmentation (`src/dataset.py`)

The module `src/dataset.py` provides an `ECGDataset` class compatible with pure NumPy and PyTorch:
- **Class Weights**: Computes inverse frequency weights (`compute_class_weights`) to counter the $110:1$ class imbalance.
- **Training Augmentations** (applied strictly during training):
  1. **Temporal Jitter**: Small shifts ($\pm 5$ samples) with edge padding.
  2. **Amplitude Scaling**: Random scaling factor $\alpha \sim \mathcal{U}(0.85, 1.15)$.
  3. **Gaussian Noise**: Additive sensor noise ($\sigma = 0.02$).
  4. **Baseline Drift Simulation**: Low-frequency synthetic drift.

```python
from dataset import ECGDataset

# Load training data with augmentation enabled
train_ds = ECGDataset.from_npz('dataset/processed/train.npz', augment=True)
print(f"Loaded {len(train_ds)} beats with class counts: {train_ds.get_class_counts()}")
```