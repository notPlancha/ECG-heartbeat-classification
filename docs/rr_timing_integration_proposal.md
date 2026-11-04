# R-R Timing Integration Proposal for ECG Heartbeat Classification

## 1. Executive Summary and Motivation

In the baseline evaluation on the patient-independent MIT-BIH dataset (DS2, 49,685 beats), all models demonstrated strong sensitivity on ventricular ectopic beats (Class V, recall ~85%, F1 ~0.60 to 0.75) but struggled on supraventricular ectopic beats (Class S, precision ~2.3% to 4.5%, F1 ~0.038 to 0.064) and fusion beats (Class F, F1 ~0.007 to 0.048).

This document details the physiological mechanism behind this limitation, formalizes R-R interval timing extraction, documents empirical proof-of-concept experiments conducted on the codebase, and outlines deployment architectures compatible with the Analog Devices MAX78002 Edge-AI platform.

---

## 2. Electrophysiological Root Cause

### 2.1 The Morphology Overlap Between Class N and Class S
In clinical electrophysiology:
* **Normal Sinus Beats (N)** originate in the Sinoatrial (SA) node and conduct through the Atrioventricular (AV) node down the His-Purkinje system into the ventricles.
* **Supraventricular Ectopic Beats (S)**, such as Premature Atrial Contractions (PACs), originate in an ectopic focus in the atria or AV junction. Once the electrical impulse reaches the AV node, ventricular depolarization follows the normal His-Purkinje pathway.
* Consequently, the ventricular QRS complex of an S-beat has normal duration (< 120 ms) and an identical or near-identical shape to a normal sinus beat from the same patient.

### 2.2 The Single-Window Limitation
The current pipeline segments a fixed 256-sample window (90 samples pre-R, 166 samples post-R at 360 Hz, spanning ~0.71 seconds) centered directly on the detected R-peak. 
* The preceding R-peak and the succeeding R-peak fall entirely outside this 256-sample window.
* In clinical cardiology, S-beats are identified primarily by **prematurity** (a shortened preceding R-R interval) followed by an incomplete compensatory pause.
* Attempting to classify S versus N based strictly on 256 samples of local waveform forces the model to rely on subtle P-wave morphology shifts that are easily masked by lead placement, respiratory variation, and baseline wander.

### 2.3 The Base Rate Problem
In the test split (DS2), Class N comprises 44,240 beats, whereas Class S comprises only 1,837 beats (a 24:1 ratio). Even with 89% to 90% specificity, the 10% false-positive rate on Class N produces ~4,500 to 5,200 false alarms, which reduces precision to ~3.6%.

---

## 3. Mathematical Formulation of R-R Interval Features

Let a single patient recording contain sequential R-peak indices $R = [r_1, r_2, \dots, r_K]$ where $r_k$ denotes the sample index in the continuous signal.

For beat $k$:
1. **Pre-RR interval**:
   $$RR_{\text{pre}}[k] = r_k - r_{k-1} \quad (k > 1)$$
2. **Post-RR interval**:
   $$RR_{\text{post}}[k] = r_{k+1} - r_k \quad (k < K)$$
3. **Local baseline interval ($RR_{\text{local}}$)**:
   Calculated as the rolling median over a 10-beat context window centered at beat $k$:
   $$RR_{\text{local}}[k] = \text{median}\left(\{ RR_{\text{pre}}[j] \mid \max(1, k-5) \le j \le \min(K, k+5) \}\right)$$
4. **Normalized timing features**:
   * Pre-RR ratio:
     $$r_{\text{pre}}[k] = \frac{RR_{\text{pre}}[k]}{RR_{\text{local}}[k]}$$
   * Post-RR ratio:
     $$r_{\text{post}}[k] = \frac{RR_{\text{post}}[k]}{RR_{\text{local}}[k]}$$
   * Coupling ratio:
     $$\text{Ratio}[k] = \frac{RR_{\text{pre}}[k]}{RR_{\text{post}}[k] + \epsilon}$$

### 3.1 Empirical Statistics on MIT-BIH DS1 Training Data

Analyzing 40,411 training beats from 17 subjects reveals clear class separations in normalized timing space:

| Class | Mean $r_{\text{pre}}$ | Median $r_{\text{pre}}$ | Mean $r_{\text{post}}$ | Median $r_{\text{post}}$ | Mean Coupling Ratio | Clinical Significance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **N** (Normal) | 1.036 | 1.000 | 0.996 | 1.000 | 1.062 | Regular cadence around local baseline |
| **S** (Supraventricular) | **0.834** | **0.898** | 1.039 | 1.015 | **0.844** | Premature arrival ($r_{\text{pre}} < 0.85$) |
| **V** (Ventricular) | **0.801** | **0.775** | 1.190 | 1.185 | **0.715** | Early arrival followed by full compensatory pause |
| **F** (Fusion) | 1.000 | 1.000 | **0.846** | **0.806** | **1.211** | Variable timing, shorter post-coupling |

---

## 4. Empirical Proof-of-Concept Experiments

### 4.1 Classical Model Benchmark (Random Forest on DS2)
Evaluating Random Forest with and without 3 normalized R-R features ($r_{\text{pre}}$, $r_{\text{post}}$, coupling ratio) on the 49,685 held-out test beats:

| Metric | Waveform Only (256 samples) | Waveform + R-R Features (259 features) | Delta |
| :--- | :--- | :--- | :--- |
| **Overall Accuracy** | 84.17% | **87.74%** | +3.57% |
| **Macro F1** | 0.3936 | **0.4405** | +0.0469 |
| **Class S Precision** | 5.05% | **18.39%** | **+13.34% (3.6x)** |
| **Class S F1-Score** | 0.0519 | **0.1146** | **+0.0627 (2.2x)** |
| **Class V Recall** | 86.27% | **93.48%** | +7.21% |
| **Class V F1-Score** | 0.6037 | **0.7075** | +0.1038 |

### 4.2 Neural Edge Model + Host MCU Gating
In an embedded deployment on the MAX78002, the CNN hardware accelerator computes class logits based on morphology, while the on-chip Arm Cortex-M4 CPU measures R-peak interrupt timestamps and gates predictions:

```
                          ┌──────────────────────────┐
256-sample Waveform ────► │ MAX78002 CNN Accelerator │ ────► Raw Logits P(N), P(S), P(V), P(F)
                          └──────────────────────────┘                      │
                                                                            ▼
R-Peak Timestamps   ────► ┌──────────────────────────┐             ┌─────────────────┐
(Hardware Timer)          │  Arm Cortex-M4 Firmware  │ ──────────► │ Decision Fusion │ ──► Final Class
                          │   r_pre = RR / RR_local  │             └─────────────────┘
                          └──────────────────────────┘
```

#### Results on Baseline CNN (`run3`):
* **Pure CNN**: Accuracy: 75.41% | Macro F1: 0.3805 | S Precision: 3.64% | S F1: 0.0544
* **CNN + MCU R-R Gating**: Accuracy: **85.12%** (+9.7%) | Macro F1: **0.4238** | S Precision: **30.66%** (8.4x boost) | S F1: **0.1625** (3x boost)

#### Results on Oversampled CNN (`run5`):
* **Pure Oversampled CNN**: Accuracy: 72.17% | Macro F1: 0.4086 | S Precision: 2.28% | S F1: 0.0382 | V F1: 0.7534
* **Oversampled CNN + MCU Gating**: Accuracy: **89.55%** (+17.4%) | Macro F1: **0.4650** | S Precision: **26.06%** (11.4x boost) | S F1: **0.1486** | V F1: **0.7534**

---

## 5. Hardware Integration Paths for Analog Devices MAX78002

### Architecture A: Firmware Post-Processing (Recommended Embedded Practice)
* **Hardware CNN**: Retains the exact 1-channel, 17,092-parameter Conv1D architecture. Synthesized directly using `ai8x-synthesis` (100% toolchain compatibility).
* **Firmware (C on Cortex-M4)**: 
  * Captures R-peak timestamps from the analog front-end peak detector.
  * Calculates `r_pre = (t_now - t_prev) / t_median`.
  * If the CNN predicts Class S but `r_pre >= 0.88`, reclassifies as Class N to suppress false alarms.
  * If the CNN predicts Class N with high uncertainty but `r_pre < 0.78`, flags the beat as Class S.
* **Resource footprint**: Adds 0 parameters to the CNN weight SRAM and runs in < 5 microseconds on the Cortex-M4.

### Architecture B: Multi-Channel 1D CNN
* **Input dimensions**: Shape `(2, 256)` or `(3, 256)`.
  * Channel 0: Normalized ECG waveform (256 samples).
  * Channel 1: Normalized $r_{\text{pre}}$ scalar value broadcast across 256 samples.
  * Channel 2: Coupling ratio broadcast across 256 samples.
* **Hardware compatibility**: Multi-channel 1D convolutions (`in_channels=2` or `3`) are supported by the MAX78002 CNN hardware accelerator.
* **Parameter delta**: Layer 1 weights increase from $1 \times 16 \times 5 = 80$ to $2 \times 16 \times 5 = 160$ (+80 weights, total 17,172 parameters), which remains under 70 KB.

---

## 6. Project Scope and Report Integration Strategy

While implementing R-R timing in the primary automated pipeline was set aside to keep Submission 2 strictly aligned with the core single-window assignment scope, this analysis serves concrete purposes in the final CRISP-DM report:

1. **Section 5.2 (Per-Class Performance)**:
   Provides the quantitative, physiological justification for why Class S precision remains low across all single-window models.
2. **Section 5.5 (Error Analysis)**:
   Explains the waveform overlap between N and S, backed by the confusion matrices.
3. **Section 6.1 (Deployment) and Section 7 (Final Discussion / What would you change with more time)**:
   Presents the two-stage Arm Cortex-M4 + CNN gating architecture as the concrete engineering roadmap for wearable deployment on the MAX78002.
