# ECG Heartbeat Classification on Edge-AI

A production-grade, reproducible solution for **ECG Heartbeat Classification on Edge-AI**, targeting the **Analog Devices MAX78002** ultra-low-power convolutional microcontroller and adhering to the **CRISP-DM methodology**.

---

## 📋 Project Overview

The goal of this project is to classify single-lead ECG heartbeats into clinically actionable categories while adhering to the hardware constraints of the **Analog Devices MAX78002 Edge-AI microcontroller**:
- **Target Platform**: Analog Devices MAX78002 (Dual Core ARM Cortex-M4 + Hardware CNN Accelerator)
- **Constraint Budget**: Maximum model size $\le 2\text{ MB}$, fixed-point INT8 computation, and strictly supported operations (1D convolution, dyadic pooling, ReLU, linear layers).
- **Classification Task**: 4-class heartbeat classification following the **ANSI/AAMI EC57** standard:
  - **N**: Normal-type and bundle branch block beats
  - **S**: Supraventricular ectopic beats (SVEB)
  - **V**: Ventricular ectopic beats (VEB)
  - **F**: Fusion of ventricular and normal beats
- **Data Source**: MIT-BIH Arrhythmia Database ($360\text{ Hz}$, two-channel ambulatory ECG) and MIT-BIH Noise Stress Test Database (NSTDB).
- **Final Report**: Compiled CRISP-DM Technical Report available in [`report.pdf`](./report.pdf) (authored in Typst: [`report.typ`](./report.typ)).

---

## 📁 Repository Structure

```text
ECG-heartbeat-classification/
│
├── report.pdf                      # Comprehensive CRISP-DM Technical Project Report
├── report.typ                      # Typst report source code
├── README.md                       # Complete project documentation and workflow instructions
├── requirements.txt                # Fully resolved, locked Python package dependencies
├── pyproject.toml                  # Project metadata and dependencies (managed via uv)
├── Taskfile.yml                    # Task automation recipes (install, prepare, train, evaluate, synth)
│
├── dataset/
│   ├── mit-bih-arrhythmia-database-1.0.0/ # Raw MIT-BIH Arrhythmia Database files (.hea, .dat, .atr)
│   ├── mit-bih-noise-stress-test-database-1.0.0/ # Raw NSTDB noise files (bw, em, ma)
│   └── processed/                  # Preprocessed, segmented Edge-AI datasets (generated)
│       ├── train.npz               # Training set: 40,411 beats (17 DS1 records)
│       ├── val.npz                 # Validation set: 10,583 beats (5 DS1 records)
│       └── test.npz                # Held-out test set: 49,685 beats (22 DS2 records)
│
├── notebooks/
│   ├── Task1_Business_Understanding.ipynb  # CRISP-DM Stage 1: Business / Application Understanding
│   ├── EDA.ipynb                           # CRISP-DM Stage 2: Data Understanding & Exploratory Analysis
│   ├── Task3_Data_Preparation.ipynb        # CRISP-DM Stage 3: Data Preparation & Preprocessing
│   ├── noise_robustness.ipynb              # CRISP-DM Stage 5.4: Noise Robustness Analysis
│   └── play-around.ipynb                   # Scratch experimental notebook
│
├── src/
│   ├── constants.py                # Centralized constants, partitions, and path resolver
│   ├── data_understanding.py       # Core modular utilities for record loading, filtering, and AAMI mapping
│   ├── download_data.py            # Streaming multi-threaded dataset downloader for PhysioNet
│   ├── prepare_data.py             # Reproducible preprocessing and dataset segmentation pipeline
│   ├── dataset.py                  # Flexible NumPy/PyTorch Dataset loader with training augmentations
│   ├── model.py                    # BaseECGModel abstract class, Baseline, RF, and 1D CNN
│   ├── train.py                    # Model training and validation pipeline (Stage 4)
│   ├── evaluate.py                 # Performance, confusion matrix, and scikit-learn metrics (Stage 5)
│   ├── noise_test.py               # NSTDB physical and Gaussian noise robustness benchmark (Stage 5.4)
│   ├── quantize.py                 # Symmetric INT8 quantization and fixed-point simulator (Stage 6)
│   ├── synthesize.py               # MAX78002 software synthesis, C driver, and memory budget tool (Stage 6)
│   └── generate_report_figures.py  # Report figures generator (EDA overview and qualitative errors)
│
├── max78002/
│   ├── model_configuration/        # Hardware network configuration (ai8x YAML specification)
│   │   └── cnn_max78002.yaml
│   ├── quantization_output/        # Software INT8 exported weights and quantization reports
│   │   ├── quantized_cnn.npz
│   │   └── quantization_report.json
│   └── synthesis_output/           # Hardware memory maps, embedded C drivers, and firmware
│       ├── cnn.h & cnn.c           # MAX78002 accelerator driver routines
│       ├── weights.h               # INT8 weights and INT32 biases C arrays
│       ├── sample_input.h          # Reference INT8 heartbeat from DS2 held-out test split
│       ├── main.c                  # ARM Cortex-M4 firmware application entry point
│       ├── synthesis_report.json   # Synthesis metrics and layer-by-layer audit
│       └── memory_budget.txt       # Table 6.3 deployment memory audit
│
├── configs/
│   └── preprocessing_config.yaml   # Configurable filter, windowing, and split parameters
│
└── results/
    ├── metrics.csv                 # Table 5.1: Comparative benchmark metrics across all models
    ├── confusion_matrix.csv        # Table 5.2: DS2 test set confusion matrix table
    ├── confusion_matrix.png        # DS2 test set confusion matrix visualization
    ├── per_class_metrics.csv       # Table 5.2: Per-class precision, recall, and F1 metrics
    ├── noise_results.csv           # Table 5.4: Noise robustness metrics across clean, 12dB, 6dB, 0dB
    ├── noise_results.png           # Noise degradation curves across SNR conditions
    ├── preprocessing_noise_benefit.csv # Butterworth baseline wander attenuation proof (2.89 dB MSE reduction)
    ├── eda_overview.png            # CRISP-DM Stage 2 & 3 exploratory data analysis panel
    ├── error_waveforms.png         # CRISP-DM Stage 5.5 qualitative error analysis waveforms
    ├── dataset_statistics.csv      # Table 3.8: Summary table of beat counts by split and class
    ├── dataset_statistics.json     # Machine-readable detailed split metrics and exclusions
    ├── per_record_distribution.csv # Per-patient class counts illustrating heterogeneity
    └── data_quality_report.csv     # Quality audit (missing data, amplitudes, baseline wander)
```

---

## ⚡ Quick Start & Reproduction Workflow

This project uses [`uv`](https://github.com/astral-sh/uv) and [`Task`](https://taskfile.dev) for reproducible environment and task management.

### 1. Environment Setup
```powershell
# Install project dependencies with uv:
uv sync
```

### 2. Download Datasets (MIT-BIH Arrhythmia & Noise Stress Test)
Per course submission rules, raw PhysioNet records are not bundled in git. Run our streaming multi-threaded downloader:
```powershell
# Downloads all 48 MIT-BIH records and all NSTDB noise records (.hea, .dat, .atr):
task download-dataset-with-noise
# Or directly via python:
uv run python src/download_data.py --include-noise
```

### 3. Run Data Preprocessing (Stages 2 & 3)
```powershell
# Executes zero-phase Butterworth filtering, beat segmentation, Z-score normalization, and export:
task prepare-data
# Output saved to dataset/processed/ (train.npz, val.npz, test.npz)
```

### 4. Train Models (Stage 4)
```powershell
# Train Linear Baseline (Logistic Regression):
task train-baseline

# Train Tree-based Baseline (Random Forest):
task train-rf

# Train Proposed Edge Model (1D CNN):
task train-cnn
```

### 5. Evaluate Models (Stage 5)
```powershell
# Evaluate FP32 models on untouched held-out DS2 test split:
task evaluate-cnn
```

### 6. Run Noise Robustness Benchmark (Stage 5.4)
```powershell
# Run physical NSTDB and synthetic noise evaluation at 12 dB, 6 dB, and 0 dB SNR:
task noise-test-all
```

### 7. Run Software INT8 Quantization & MAX78002 Synthesis (Stage 6)
```powershell
# Execute symmetric INT8 quantization engine:
task quantize

# Generate MAX78002 model configuration, synthesis outputs, and embedded C firmware:
task synthesize
```

### 8. Compile CRISP-DM Technical Report
```powershell
# Compile Typst source into publication report.pdf:
task report
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
- **Bandpass Filter**: 3rd-order zero-phase Butterworth filter ($0.5 - 40.0\text{ Hz}$).
- **Removal**: Eliminates low-frequency baseline wander ($< 0.5\text{ Hz}$) from respiration and electrode movement, and suppresses high-frequency EMG muscle tremors and $50/60\text{ Hz}$ powerline hum without phase distortion. Attenuates added baseline wander by **2.89 dB** (with $18.94\text{ dB}$ pure wander suppression).

### 4. Beat Segmentation & Windowing for MAX78002
- **Sampling Frequency**: $360\text{ Hz}$ native.
- **Window Length**: **$256$ samples** ($711.1\text{ ms}$):
  - **Pre-R window**: $90$ samples ($250.0\text{ ms}$), capturing P-wave and PR interval.
  - **Post-R window**: $166$ samples ($461.1\text{ ms}$), capturing QRS complex, ST segment, and T-wave.
  - **Edge-AI Benefit**: $256 = 2^8$ samples allows clean dyadic pooling ($256 \to 128 \to 64 \to 32$) on the MAX78002 CNN hardware accelerator without fractional padding.
- **Normalization & Outlier Clamping**: Per-beat Z-score normalization clamped to $[-3.0, 3.0]$ standard deviations to suppress extreme motion spikes and establish dynamic boundaries for INT8 quantization.

---

## 📊 Dataset Statistics & Class Distributions (Table 3.8)

The exact beat counts generated by `src/prepare_data.py` (matching Section 3.8 of the project rubric) are:

| Split | Records | N (Normal) | S (Supraventricular) | V (Ventricular) | F (Fusion) | Total Heartbeats |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 17 | 36,008 | 838 | 3,164 | 401 | **40,411** |
| **Validation** | 5 | 9,840 | 106 | 624 | 13 | **10,583** |
| **Test (DS2)** | 22 | 44,240 | 1,837 | 3,220 | 388 | **49,685** |
| **Total** | **44** | **90,088** | **2,781** | **7,008** | **802** | **100,679** |

---

## 🏆 Final Model Evaluation Summary (Table 5.1 & Table 6.3)

| Model | Parameters | Clean Acc | Macro F1 | Weight Mem | Active Power | Latency @ 50MHz |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 1,028 | 66.84% | 0.3221 | 1.03 KB | N/A | CPU Only |
| **Random Forest** | 90,718 | 83.78% | 0.3788 | 9.07 MB | N/A | Exceeds HW |
| **1D CNN (FP32)** | 17,092 | 74.20% | 0.3230 | 68.37 KB | N/A | PyTorch Ref |
| **Quantized 1D CNN (INT8)** | **17,092** | **87.58%** | **0.2506** | **17.44 KB (0.83%)** | **~5.0 mW** | **0.234 ms** |

---

## 📝 License & Academic Disclaimer
This project is developed for academic evaluation in Edge-AI and data mining. In accordance with course guidelines, this system is not a certified medical device and is not clinically validated for diagnostic use.
