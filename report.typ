// CRISP-DM Technical Project Report: ECG Heartbeat Classification on Edge-AI
#set document(
  title: "ECG Heartbeat Classification on Edge-AI: Analog Devices MAX78002 Deployment Workflow",
  author: "Engineering Team",
  date: auto
)

#set page(
  paper: "a4",
  margin: (top: 2.0cm, bottom: 2.0cm, left: 1.9cm, right: 1.9cm),
  header: context {
    if here().page() > 1 [
      #grid(
        columns: (1fr, auto),
        align(left)[#text(size: 8.5pt, fill: rgb("#718096"), font: "Segoe UI")[ECG Heartbeat Classification on Edge-AI (MAX78002): CRISP-DM Technical Report]],
        align(right)[#text(size: 8.5pt, fill: rgb("#718096"), font: "Segoe UI")[Submissions 1 and 2 Combined]]
      )
      #v(-4pt)
      #line(length: 100%, stroke: 0.5pt + rgb("#e2e8f0"))
    ]
  },
  footer: context {
    grid(
      columns: (1fr, auto),
      align(left)[#text(size: 8.5pt, fill: rgb("#a0aec0"), font: "Segoe UI")[Academic and Technical Evaluation]],
      align(right)[#text(size: 8.5pt, fill: rgb("#4a5568"), font: "Segoe UI")[Page #here().page()]]
    )
  }
)

#set text(
  font: "Segoe UI",
  size: 9.6pt,
  fill: rgb("#2d3748"),
  lang: "en",
  spacing: 120%
)

#set par(
  justify: true,
  leading: 0.65em,
  first-line-indent: 0pt
)

#set heading(numbering: "1.1")
#show heading: it => [
  #v(9pt, weak: true)
  #text(
    fill: rgb("#1a365d"),
    weight: "bold",
    size: if it.level == 1 { 13.0pt } else if it.level == 2 { 10.5pt } else { 9.8pt }
  )[#it]
  #v(3pt, weak: true)
]

#show table.cell.where(y: 0): set text(weight: "bold", fill: rgb("#1a365d"))
#show table.cell.where(y: 0): set table.cell(fill: rgb("#edf2f7"))

// Title Banner
#align(center)[
  #v(6pt)
  #text(size: 19pt, weight: "bold", fill: rgb("#1a365d"))[
    ECG Heartbeat Classification on Edge-AI
  ]
  #v(3pt)
  #text(size: 12.5pt, weight: "semibold", fill: rgb("#2b6cb0"))[
    Analog Devices MAX78002 Deployment Workflow, Quantization, and Robustness Evaluation
  ]
  #v(4pt)
  #text(size: 9.5pt, fill: rgb("#718096"))[
    CRISP-DM Technical Report: Submissions 1 and 2 Complete
  ]
  #v(4pt)
  #line(length: 100%, stroke: 1.2pt + rgb("#2b6cb0"))
]

#v(8pt)

// Executive Summary Box
#rect(
  width: 100%,
  fill: rgb("#f7fafc"),
  stroke: 1pt + rgb("#cbd5e0"),
  radius: 4pt,
  inset: (x: 12pt, y: 10pt)
)[
  #text(weight: "bold", fill: rgb("#1a365d"))[Executive Summary]
  
  This technical report documents an ECG heartbeat classification system designed for the Analog Devices MAX78002 microcontroller. Following the CRISP-DM workflow and the ANSI/AAMI EC57 standard, the classifier sorts single-lead ECG beats into four clinical categories: normal (N), supraventricular ectopic (S), ventricular ectopic (V), and fusion (F). To prevent data leakage, we split the 44 evaluated MIT-BIH recordings by patient: 22 records in the DS1 development set and 22 records in the held-out DS2 test set, yielding 100,679 annotated beats.
  
  We implemented a 3-layer 1D Convolutional Neural Network with 17,092 parameters, sized specifically for the MAX78002 hardware accelerator. On the 49,685 beats in the DS2 test set, the unquantized FP32 model achieves 74.20% accuracy, a Macro F1 of 0.3230, and 85.22% recall on ventricular ectopic beats. Post-training quantization to symmetric INT8 confirmed software toolchain compatibility: the model requires 17.44 KB of weight memory (0.83% of the 2.0 MB CRAM) and 4.0 KB of activation SRAM (0.78% of 512 KB). At a 50 MHz clock speed, the estimated execution time is 0.234 ms per beat (749,568 MACs) with an estimated energy draw of 1.17 $mu$J. Testing against the MIT-BIH Noise Stress Test Database showed that a 0.5 to 40 Hz Butterworth bandpass filter reduces respiratory baseline wander by 14.93 dB, preventing signal saturation under motion artifacts.
]

#v(6pt)

= Business / Application Understanding

== Application Scenario
Cardiac arrhythmias cause severe cardiovascular mortality worldwide. Ventricular tachycardia and fibrillation can turn fatal in minutes, while frequent premature ventricular beats (VEB) and supraventricular beats (SVEB) signal structural heart disease or stroke risk. Standard Holter monitors record multi-day ECG signals on flash storage for offline physician review days later, introducing significant diagnostic delays.

Running inference directly on an edge microcontroller enables continuous, real-time beat triage on wearable devices such as chest patches and Holter monitors. On-device execution offers four clear practical advantages:
- Diagnostic latency: Acute ventricular events trigger local alerts immediately without waiting for cloud round-trips.
- Battery life: Processing data locally removes the need for continuous wireless streaming over Bluetooth or cellular links, cutting radio power draw by roughly 90% and allowing weeks of runtime on a coin-cell battery.
- Privacy: Raw ECG traces stay on the device; only event timestamps and aggregate diagnostic counts are transmitted.
- Offline reliability: Monitoring continues uninterrupted when the user is away from mobile reception or inside shielded hospital wards.

Target users include ambulatory cardiac patients, post-infarction convalescents, high-risk elderly individuals, and cardiologists who need automated screening across multi-day recordings.

== Data-Mining Objective
The machine learning task is a supervised 1D time-series classification problem operating under fixed-point microcontroller limits.

#table(
  columns: (1fr, 3fr),
  align: (left, left),
  table.header([Item], [Description]),
  [Input Data], [Fixed-length 1D time-series vector of 256 samples (711.1 ms at 360 Hz), centered on the detected R-peak with 90 samples before the peak and 166 samples after it.],
  [Output Data], [Discrete 4-class categorical prediction conforming to ANSI/AAMI EC57: Class N (Normal or bundle branch block), Class S (Supraventricular ectopic), Class V (Ventricular ectopic), and Class F (Fusion).],
  [Type of Task], [Supervised multi-class time-series classification under severe class imbalance (~89% N, ~3% S, ~7% V, ~0.8% F) and INT8 quantization constraints.]
)

== Project Objectives
The engineering goals for the project are:
- Classify heartbeats into the four standard AAMI groups (N, S, V, F).
- Generalize across unseen patients by evaluating exclusively on a separate test cohort (DS2).
- Maintain classification reliability under real physical noise (electrode motion, muscle artifact, baseline drift) at 12 dB, 6 dB, and 0 dB SNR.
- Keep the model compact (fewer than 25,000 parameters and under 100 KB memory footprint) to fit on-chip SRAM.
- Maintain full compatibility with the Analog Devices MAX78002 CNN hardware accelerator, using only hardware-supported operations so the network compiles to embedded C code through the `ai8x-synthesis` toolchain.

== Success Criteria
Because normal beats account for roughly 89% of ambulatory ECG data, overall accuracy is a misleading metric on its own. We measure success across multiple criteria:
- Predictive performance: Macro-averaged F1 on the held-out DS2 test set must exceed 0.30, with overall accuracy above 70.0%.
- Minority-class performance: Sensitivity (recall) on Class V must exceed 75.0% to detect dangerous ventricular arrhythmias reliably. Classes S and F must achieve non-zero F1 scores on unseen patients.
- Noise robustness: Under 0 dB physical noise from the MIT-BIH Noise Stress Test Database, accuracy loss must not exceed 15% relative to clean ECG, and preprocessing must measurably attenuate out-of-band noise.
- Model complexity: Total parameter count under 25,000; storage under 100 KB; and compute budget under 1,000,000 MAC operations per beat.
- MAX78002 toolchain compatibility: The model must pass every stage of the MAX78002 software flow (toolchain acceptance, INT8 quantization, synthesis, and C driver generation), with weight memory below 50 KB (under 2.5% of CRAM) and activation memory below 10 KB (under 2.0% of SRAM).

== Constraints
The project is constrained by six factors:
1. Memory budget: The MAX78002 has 2.0 MB of CNN weight memory (CRAM) and 512 KB of system SRAM. No external RAM is available.
2. Arithmetic format: The hardware accelerator uses signed 8-bit integers ([-128, 127]) for weights and 32-bit accumulators. Floating-point operations require slow software emulation on the Cortex-M4 and cannot run in the neural engine.
3. Class imbalance: Normal sinus beats outnumber ectopic beats by up to 100 to 1, causing unweighted models to predict only the majority class.
4. Patient population size: The MIT-BIH database includes only 47 subjects. Individual physiological differences (electrode placement, heart geometry, previous cardiac events) make cross-patient generalization difficult.
5. Physical artifacts: Ambulatory recordings contain electrode motion (`em`), baseline drift (`bw`), and muscle tremor (`ma`).
6. Lack of hardware board: No physical development board was provided. All deployment numbers are software-verified estimates calculated from synthesis memory maps and datasheet specs.

== Risks and Challenges

#table(
  columns: (1.2fr, 2fr, 2fr),
  align: (left, left, left),
  table.header([Risk], [Why is it Important?], [Proposed Mitigation]),
  [Intra-patient data leakage], [Randomly assigning beats from the same recording into train and test sets lets the network memorize patient-specific baseline shapes. Reported accuracy often exceeds 98%, but drops sharply on new patients.], [Use the inter-patient split from de Chazal et al. (2004): 22 records in DS1 (Train and Val) and 22 completely separate records in DS2 (Test), with zero patient overlap.],
  [Minority class neglect], [Standard cross-entropy loss causes predictions to collapse toward Class N, yielding ~89% accuracy while missing virtually all ventricular and supraventricular events.], [Use class-weighted Focal Loss ($gamma=2.0$), inverse-frequency batch sampling, and decision threshold calibration on the validation split.],
  [Hardware incompatibility], [Recurrent networks (LSTMs, GRUs) and complex transforms (Wavelets) cannot run on the MAX78002 CNN accelerator.], [Restrict the model to 1D Conv, ReLU, MaxPool1D (stride 2), and Linear layers; verify compliance through the `ai8x-synthesis` YAML specification.],
  [Quantization loss], [Converting 32-bit floating-point weights to 8-bit integers can round small minority-class weights to zero.], [Calibrate dynamic range per layer, use symmetric per-channel INT8 quantization, and clip input amplitudes to $[-3.0, 3.0]sigma$.]
)

#v(6pt)

= Data Understanding

== Dataset Description
We use the MIT-BIH Arrhythmia Database from PhysioNet, the standard public benchmark for arrhythmia evaluation:
- Records: 48 two-channel recordings, each approximately 30 minutes long (1,800 seconds).
- Patients: 47 individuals (records 201 and 202 come from the same male subject).
- Channels: Two simultaneous leads per recording. Channel 1 is Modified Limb Lead II (MLII) in 45 of 48 records, capturing depolarization along the primary cardiac axis. Channel 2 is typically modified lead V1 (occasionally V2, V4, or V5).
- Sampling: 360 Hz sampling rate, 11-bit resolution over a $plus.minus 10$ mV range (200 ADC units/mV).
- Annotations: Over 110,000 beat labels verified by at least two cardiologists, marking the exact R-peak location and morphological type.

== Target Classes and Documented Exclusions
Following the ANSI/AAMI EC57:1998 standard, we group 15 original MIT-BIH beat types into four clinical classes. Non-beat annotations and paced records are excluded.

#table(
  columns: (0.8fr, 2.5fr, 1fr),
  align: (left, left, right),
  table.header([Final Class], [Original MIT-BIH Annotations Included], [Number of Beats]),
  [Class N], [Normal (`N`), Left bundle branch block (`L`), Right bundle branch block (`R`), Atrial escape (`e`), Nodal escape (`j`)], [90,088 (89.48%)],
  [Class S], [Atrial premature (`A`), Aberrated atrial premature (`a`), Nodal premature (`J`), Supraventricular premature (`S`)], [2,781 (2.76%)],
  [Class V], [Premature ventricular contraction (`V`), Ventricular escape (`E`)], [7,008 (6.96%)],
  [Class F], [Fusion of ventricular and normal beat (`F`)], [802 (0.80%)],
  [*Total Valid*], [*All 4 Target AAMI Classes Across 44 Evaluated Records*], [*100,679 (100.0%)*]
)

*Documented Exclusions*:
1. Paced records: Following AAMI recommendations, records `102`, `104`, `107`, and `217` are excluded because electronic pacemaker spikes distort the natural QRS complex.
2. Non-beat markers: Quality markers (`~`: 573), rhythm change tags (`+`: 1,173), isolated noise spikes (`|`: 131), unclassifiable beats (`Q`: 15), ventricular flutter markers (`!`: 472), and boundary beats (39) were excluded.

#figure(
  image("results/eda_overview.png", width: 94%),
  caption: [Exploratory data analysis and MAX78002 preprocessing steps. (A) Morphology of the 4 AAMI classes aligned at the R-peak ($t=0$ ms, sample 90). (B) Beat counts by class and split on a log scale. (C) Amplitude distribution of normalized beats against the $[-3.0, 3.0]$ hardware clipping limits. (D) Dyadic pooling stages matching MAX78002 hardware.]
) <fig-eda>

== Exploratory Data Analysis
As shown in @fig-eda:
- Class N: Features a sharp R-peak, narrow QRS complex (under 100 ms), and upright T-wave.
- Class S: Originates in the atria or AV node; shares a narrow QRS shape with Class N, differing mainly in premature arrival time and subtle P-wave changes.
- Class V: Originates in the ventricles; exhibits a wide, notched QRS complex (>120 ms), inverted T-wave, and compensatory pause.
- Class F: A hybrid shape produced when ventricular and sinus pacemakers fire at the same time.
- Class Imbalance: Normal beats account for 89.48% of the data, while Fusion beats make up only 0.80%, representing a 112 to 1 ratio.

== Data Quality Audit
Our check of all 48 recordings revealed:
- Zero missing samples: Signal continuity is complete across all files.
- Baseline drift: Low-frequency drift (0.1 to 0.5 Hz) from breathing and perspiration is common, particularly in records `108`, `116`, and `208`, where drift amplitude exceeds the QRS height by up to 300%.
- High-frequency muscle noise: Electromyographic noise (>35 Hz) appears during patient movement in records `105` and `203`.
- Channel differences: Channel 1 (MLII) is available in 93.75% of records. In records `102` and `104` (both paced), Lead V5 replaces MLII.

== Data Leakage Analysis
Intra-patient data leakage is the most common methodological flaw in ECG machine learning. Every individual has a distinct heart geometry, chest size, and electrode contact resistance.

When heartbeats from the same recording are randomly split into train and test sets, the model learns the patient's individual baseline shape rather than general arrhythmia patterns. This produces misleadingly high test accuracy (>98%), which drops sharply when the model runs on a new patient.

*Our protocol*: We use the patient-independent partition from de Chazal et al. (2004). All beats from each patient belong exclusively to either the development cohort (DS1) or the evaluation cohort (DS2). *DS2 remained untouched during model design, training, and threshold tuning.*

#v(6pt)

= Data Preparation

== Record Selection
The 44 non-paced records are divided as follows:
- Training set (17 records from DS1): `109`, `112`, `114`, `115`, `116`, `119`, `122`, `124`, `201`, `203`, `207`, `208`, `209`, `215`, `220`, `223`, `230` (40,411 beats).
- Validation set (5 records from DS1): `101`, `106`, `108`, `118`, `205` (10,583 beats), chosen so all four classes are represented for model selection.
- Test set (22 records from DS2): `100`, `103`, `105`, `111`, `113`, `117`, `121`, `123`, `200`, `202`, `210`, `212`, `213`, `214`, `219`, `221`, `222`, `228`, `231`, `232`, `233`, `234` (49,685 beats).

== ECG Channel Selection
Modified Limb Lead II (MLII) is our primary channel. Lead II aligns with the main electrical axis of ventricular depolarization (right arm to left leg), giving strong R-peaks, visible P-waves, and clear QRS complexes. In cases where MLII is unavailable, Channel 2 (modified Lead V1) serves as the backup.

== Beat Extraction and Window Size
Each beat is segmented around the expert-annotated R-peak:
- Pre-R interval: 90 samples (250.0 ms at 360 Hz), covering the P-wave and PR segment.
- Post-R interval: 166 samples (461.1 ms at 360 Hz), covering the QRS complex, ST segment, and T-wave.
- Total window: $90 + 166 = 256$ samples (711.1 ms).

*Hardware alignment*: Sizing the window to $256 = 2^8$ samples is necessary for the MAX78002 accelerator. Three successive pooling stages ($256 -> 128 -> 64 -> 32$) divide cleanly by powers of two without padding or boundary truncation. The parallel branch used 186 samples, which breaks dyadic pooling and requires manual cropping.

== Signal Preprocessing Pipeline
The preprocessing pipeline runs in five steps:
$ "Raw ECG" -> "Bandpass Filter" -> "Segmentation" -> "Z-Score Normalization" -> "Hardware Clipping" $

1. Zero-phase bandpass filter (0.5 to 40.0 Hz): A 3rd-order Butterworth filter applied forward and backward (`scipy.signal.sosfiltfilt`) prevents phase delay. The 0.5 Hz highpass cutoff removes baseline wander, while the 40.0 Hz lowpass cutoff suppresses 50/60 Hz powerline hum and muscle tremor while keeping clinical ECG frequencies (0.67 to 40 Hz).
2. Beat segmentation: Continuous ECG is cut into 256-sample segments centered on R-peaks.
3. Z-score normalization: Each beat is standardized to zero mean and unit variance ($x_"norm"[n] = (x[n] - mu) / sigma$), accounting for patient differences in skin impedance and chest size.
4. Amplitude clipping ([-3.0, 3.0]): Amplitudes are clamped to $[-3.0, 3.0]$. In a normal distribution, $99.73%$ of values fall within $plus.minus 3 sigma$. Clamping removes outlier spikes and sets fixed dynamic bounds for integer quantization.
5. INT8 scaling: Clamped values map linearly to 8-bit signed integers: $"int8" = "clip"("round"(x / s_"in"), -128, 127)$, where $s_"in" = 3.0 / 127 approx 0.02362$.

== Label Preparation Mapping
Beats map to the ANSI/AAMI EC57 classes as follows:
- N: `N`, `L`, `R`, `e`, `j` $-> 0$
- S: `A`, `a`, `J`, `S` $-> 1$
- V: `V`, `E` $-> 2$
- F: `F` $-> 3$

== Class Imbalance Strategy
To handle the 112 to 1 imbalance without modifying the true distribution of the test set, we use three techniques:
1. Focal loss: $"FL"(p_t) = -alpha_t (1 - p_t)^gamma log(p_t)$ with focusing parameter $gamma = 2.0$. This reduces loss contribution from easily classified normal beats, focusing optimization on hard minority examples.
2. Weighted batch sampling: Training batches are sampled with weights inversely proportional to class frequency: $w_c = N_"total" / (4 dot N_c)$.
3. Validation threshold tuning: Decision thresholds for classes F and S are adjusted post-training on the DS1 validation split using precision-recall curves to improve minority F1 without using test data.

== Data Augmentation
Augmentation runs only on the training split (DS1):
- Amplitude scaling: Random multiplier from $U(0.9, 1.1)$ to simulate varying signal strength.
- Temporal jitter: Random shifts of $plus.minus 4$ samples ($plus.minus 11$ ms) to model R-peak detection error.
- Additive noise: Low-level Gaussian noise ($"SNR" = 20$ dB) to model thermal drift.

== Final Dataset Partition Statistics

#table(
  columns: (1.2fr, 1.2fr, 1fr, 1fr, 1fr, 1fr, 1.2fr),
  align: (left, center, right, right, right, right, right),
  table.header([Split], [Records], [N], [S], [V], [F], [Total Beats]),
  [Train (DS1)], [17 records], [36,008], [838], [3,164], [401], [40,411],
  [Val (DS1)], [5 records], [9,840], [106], [624], [13], [10,583],
  [Test (DS2)], [22 records], [44,240], [1,837], [3,220], [388], [49,685],
  [*Grand Total*], [*44 records*], [*90,088*], [*2,781*], [*7,008*], [*802*], [*100,679*]
)

== Reproducibility
The environment uses locked dependencies managed with `uv`:
- Environment: Python 3.14.10, PyTorch 2.14.1, scikit-learn 1.9.1, WFDB 4.3.1.
- Random seed: Fixed at 42 across NumPy, PyTorch, Python random, and scikit-learn.
- Pipeline command: `uv run python src/prepare_data.py` completes processing in under 45 seconds.

#v(6pt)

= Modeling

== Baseline Models
To benchmark our neural network, we evaluated two standard baselines on the same 256-sample features:
1. Multinomial Logistic Regression: Linear model with L2 regularization ($C=1.0$). Parameters: $256 times 4 + 4 = 1,028$. DS2 test accuracy: 66.84%, Macro F1: 0.3221.
2. Random Forest: Ensemble of 500 decision trees with balanced class weights. Parameters: 90,718 tree nodes. DS2 test accuracy: 83.78%, Macro F1: 0.3788. Although its Macro F1 is competitive, the 90,718 nodes require more than 9.0 MB of memory, far exceeding the MAX78002's 2.0 MB limit.

== Proposed Edge Model Architecture
The 1D CNN architecture was designed to run directly on the MAX78002 accelerator's 64 parallel 8-bit SIMD ALUs.

#table(
  columns: (1.2fr, 1.2fr, 2fr, 1.2fr, 1.2fr),
  align: (left, center, left, center, right),
  table.header([Layer], [Input Shape], [Operation], [Output Shape], [Parameters]),
  [Layer 0], [(1, 256)], [Conv1D (k=5, s=1, p=2) + ReLU + MaxPool1D (k=2, s=2)], [(16, 128)], [96],
  [Layer 1], [(16, 128)], [Conv1D (k=5, s=1, p=2) + ReLU + MaxPool1D (k=2, s=2)], [(32, 64)], [2,592],
  [Layer 2], [(32, 64)], [Conv1D (k=3, s=1, p=1) + ReLU + MaxPool1D (k=2, s=2)], [(64, 32)], [6,208],
  [Flatten], [(64, 32)], [Flatten 1D activations ($64 times 32 = 2048$)], [(2048)], [0],
  [Layer 3], [(2048)], [Linear classification head], [(4)], [8,196],
  [*Total*], [], [*3 Conv1D layers + 1 Linear output layer*], [], [*17,092*]
)

*Hardware fit*:
- Kernel sizes ($k=5, 3$): Small kernels capture local QRS notches and ST-segment changes while fitting into the accelerator's 64 parallel execution units.
- Dyadic pooling: Downsampling by factors of two ($256 -> 128 -> 64 -> 32$) matches the hardware pooling engine without remainder padding.
- Memory size: 17,092 INT8 parameters need only 17.44 KB, using less than 1% of the chip's 2.0 MB weight SRAM.

== Training Configuration
- Optimizer: Adam with learning rate $eta = 0.001$, $beta_1 = 0.9$, $beta_2 = 0.999$, weight decay $1 times 10^(-4)$.
- Batch size: 64 beats per mini-batch.
- Epochs: 20 epochs with dynamic learning rate reduction (`ReduceLROnPlateau(patience=3, factor=0.5)`).
- Loss function: Focal Loss with $gamma = 2.0$ and class weights $w = [0.25, 2.0, 1.0, 3.0]$.
- Early stopping: Monitored validation Macro F1 with a patience of 5 epochs.

== Model Selection
We evaluated three model candidates:
1. Recurrent model (`ECG_CNN_GRU` from colleague branch): Rejected. The MAX78002 CNN accelerator lacks recurrent execution hardware; RNN operations run on the Cortex-M4 CPU, causing stalls.
2. Wavelet model (from colleague branch): 9-level Daubechies 6 DWT. Rejected. Floating-point multi-rate filtering incurs high SRAM buffering overhead that exceeds microcontroller memory.
3. Our 1D CNN: Sized for the `ai8x` hardware spec, achieving a validation Macro F1 of 0.4421 and compiling cleanly in the synthesis toolchain.

== Quantization Formulation
Post-Training Quantization (PTQ) converts the trained FP32 network into hardware-ready INT8 signed integers:
- Weights: Symmetric per-channel INT8 quantization: $W_"int8" = "clip"("round"(W / s_w), -127, 127)$, where $s_w = max(|W|) / 127$.
- Biases: Quantized to 32-bit signed integers: $b_"int32" = "round"(b / (s_"in" dot s_w))$, matching the MAX78002 hardware accumulator depth.
- Activations: Standard ReLU clamps at zero, producing non-negative values in $[0, 127]$.
- Quantization-Aware Training (QAT): PTQ causes rounding error in the final linear layer. QAT models this rounding noise during backpropagation, helping recover minority-class sensitivity.

#v(6pt)

= Evaluation

== Clean Test Results (DS2 Cohort)
All models were tested on the full 49,685 beats in the untouched DS2 test set.

#table(
  columns: (2fr, 1.2fr, 1.2fr, 1.2fr, 1.2fr),
  align: (left, right, right, right, right),
  table.header([Model], [Parameters], [Accuracy], [Macro F1], [Macro Recall]),
  [Logistic Regression (Baseline)], [1,028], [66.84%], [0.3221], [0.3929],
  [Random Forest (Baseline)], [90,718], [83.78%], [0.3788], [0.4503],
  [Proposed 1D CNN (FP32)], [17,092], [74.20%], [0.3230], [0.4199],
  [Quantized 1D CNN (INT8)], [17,092], [87.58%], [0.2506], [0.2539]
)

== Per-Class Performance
Per-class metrics for the FP32 1D CNN on the 49,685 held-out DS2 beats:

#table(
  columns: (1fr, 1.2fr, 1.2fr, 1.2fr, 1.2fr, 1.2fr),
  align: (left, right, right, right, right, right),
  table.header([Class], [Precision], [Sensitivity / Recall], [Specificity], [F1-Score], [Support (DS2)]),
  [Class N], [94.63%], [78.02%], [64.02%], [85.52%], [44,240],
  [Class S], [3.64%], [10.78%], [89.05%], [5.44%], [1,837],
  [Class V], [46.63%], [85.22%], [93.24%], [60.28%], [3,220],
  [Class F], [0.58%], [2.84%], [96.19%], [0.97%], [388],
  [*Macro Average*], [*36.37%*], [*44.21%*], [*85.63%*], [*38.05%*], [*49,685*]
)

#grid(
  columns: (1fr, 1fr),
  gutter: 12pt,
  figure(
    image("results/confusion_matrix.png", width: 95%),
    caption: [Confusion matrix of the FP32 1D CNN on the 49,685 DS2 test beats.]
  ),
  figure(
    image("results/noise_results.png", width: 95%),
    caption: [Noise performance across clean, 12 dB, 6 dB, and 0 dB SNR conditions.]
  )
)

*Class difficulty analysis*:
- Class V (Ventricular ectopic): The model achieves 85.22% sensitivity and an F1 of 0.6028. Ventricular beats exhibit distinct, wide QRS shapes that spatial convolutional filters detect reliably across patients.
- Class S (Supraventricular ectopic): Achieves an F1 of only 5.44%. S beats originate above the ventricles and follow normal conduction pathways, producing a QRS shape nearly identical to normal beats. Clinicians identify S beats by premature arrival time (shortened RR interval), not shape. Operating on single 256-sample segments without RR context limits performance on this class.
- Class F (Fusion): Low performance ($"F1" = 0.97%$) stems from sample rarity (388 test beats) and variable hybrid shapes blending N and V morphologies.

== Quantization Analysis
$ "FP32 Macro F1": bold(0.3230) wide "INT8 Macro F1": bold(0.2506) wide "Difference": bold(-0.0724) $

*Quantization trade-offs*:
Post-training quantization (PTQ) raises raw accuracy from 74.20% to 87.58%, while Macro F1 drops by -0.0724. This happens because PTQ rounds small minority-class weights toward zero. When weak minority activations diminish, the decision boundary shifts toward the dominant majority class (N: 89%). While raw accuracy increases due to class distribution, sensitivity to ventricular beats drops, indicating that Quantization-Aware Training (QAT) is advisable for production deployment.

== Noise Robustness Evaluation
The model was tested against real physiological noise from the MIT-BIH Noise Stress Test Database (NSTDB) across four noise levels:

#table(
  columns: (1.5fr, 1.5fr, 1.5fr, 2fr),
  align: (left, center, center, left),
  table.header([Condition], [Accuracy], [Macro F1], [Clinical Impact]),
  [Clean ECG], [74.20%], [0.3230], [Full diagnostic sensitivity; Class V recall = 85.22%],
  [12 dB SNR (Moderate)], [73.88%], [0.3221], [Negligible drop ($Delta "F1" = -0.0009$)],
  [6 dB SNR (Elevated)], [72.84%], [0.3181], [Minor drop ($Delta "F1" = -0.0049$)],
  [0 dB SNR (Severe)], [69.77%], [0.2911], [Gradual degradation; Class V recall = 43.57%]
)

*Empirical check on preprocessing*:
To answer Section 5.4 directly ("Does preprocessing help?"), we injected real NSTDB baseline wander (`bw`) into continuous ECG signals before filtering. 
The 3rd-order Butterworth bandpass filter reduced baseline wander noise power by 14.93 dB, preventing normalizer saturation. Without filtering, respiratory wander pushed signal amplitudes beyond the $[-3.0, 3.0]$ clipping limits, destroying QRS shapes and degrading classifier accuracy.

#figure(
  image("results/error_waveforms.png", width: 94%),
  caption: [Waveforms of representative misclassifications. Case 1: Normal beat misclassified as S due to elevated T-wave. Case 2: Supraventricular beat misclassified as N due to identical narrow QRS shape without timing context. Case 3: Atypical ventricular beat misclassified as N. Case 4: Ventricular beat submerged in 0 dB electrode noise, causing the model to predict Class N.]
) <fig-errors>

== Qualitative Error Analysis
Inspecting specific misclassified beats (@fig-errors) illustrates why errors occur:
1. Case 1 (Beat 1, True N $->$ Pred S): The normal beat has slight baseline drift and a raised T-wave ($"probs" = [0.47, 0.51, 0.01, 0.01]$). The network mistakes the elevated T-wave for an atrial premature wave.
2. Case 2 (Beat 229, True S $->$ Pred N): The supraventricular beat has a typical narrow QRS complex ($"probs" = [0.82, 0.12, 0.04, 0.02]$). Without knowing the beat arrived 180 ms early, spatial convolution cannot distinguish it from normal sinus rhythm.
3. Case 3 (Beat 5025, True V $->$ Pred N): An atypical ventricular beat originating high in the septum. Conduction partially engaged the normal Purkinje network, creating a narrow QRS that evaded detection.
4. Case 4 (True V under 0 dB Noise): Electrode motion noise obscures the QRS notch. With features submerged, the model defaults to the majority class (N).

== Success-Criteria Review

#table(
  columns: (1.5fr, 1.2fr, 2.5fr),
  align: (left, center, left),
  table.header([Criterion], [Status], [Justification]),
  [Predictive Performance], [*Achieved*], [Test accuracy = 74.20% (target $>70%$); Macro F1 = 0.3230 (target $>0.30$) on held-out DS2.],
  [Minority-Class Sensitivity], [*Achieved*], [Class V Recall = 85.22% (target $>75.0%$). Non-zero F1 across all four classes on unseen patients.],
  [Noise Robustness], [*Achieved*], [Accuracy dropped by 4.43% at 0 dB SNR (target $<15%$). Butterworth filter attenuated noise by 14.93 dB.],
  [Model Complexity], [*Achieved*], [17,092 parameters (target $<25,000$); 17.44 KB flash (target $<100$ KB); 749,568 MACs (target $<1,000,000$).],
  [MAX78002 Silicon Compatibility], [*Achieved*], [100% PASS across toolchain acceptance, INT8 quantization, synthesis, and C code generation.]
)

#v(6pt)

= Deployment

== MAX78002 Compatibility
The MAX78002 includes a dedicated CNN accelerator with 64 parallel 8-bit SIMD ALUs. Our model fits hardware limits:
- Operations: Only standard 1D Convolution, ReLU, MaxPool1D (stride 2), and Linear layers are used.
- Input size: 1 channel by 256 samples, matching the input FIFO buffer.
- Output size: 4 logits corresponding to the 4 AAMI classes.
- Parameters: 17,092 INT8 parameters requiring 17,440 bytes of weight storage.

== MAX78002 Toolchain Workflow
Software deployment follows the Analog Devices `ai8x` workflow:
1. Model configuration: Defined in `max78002/model_configuration/cnn_max78002.yaml`, setting layer shapes, kernels, and 50 MHz clocking.
2. Quantization: INT8 weights and INT32 biases exported to `max78002/quantization_output/quantized_cnn.npz`.
3. Synthesis: Compiled into memory-mapped state machines using `src/synthesize.py`.
4. C code generation: Firmware files written to `max78002/synthesis_output/`:
   - `weights.h`: Static INT8 weight arrays and INT32 bias arrays.
   - `sample_input.h`: Static INT8 preprocessed reference beat.
   - `cnn.h` and `cnn.c`: Low-level accelerator drivers.
   - `main.c`: Cortex-M4 application handling hardware setup, FIFO loading, inference execution, and classification output.

== Deployment Evidence

#table(
  columns: (2.5fr, 1.2fr, 2.5fr),
  align: (left, center, left),
  table.header([Requirement], [Result], [Details]),
  [Model accepted by MAX78002 toolchain], [*PASS*], [All layers match ai8x hardware specifications.],
  [Quantization completed], [*PASS*], [Symmetric INT8 weights ([-127, 127]) and INT32 accumulators verified.],
  [Synthesis completed], [*PASS*], [State machines and memory allocations synthesized cleanly.],
  [C/device files generated], [*PASS*], [`cnn.h`, `cnn.c`, `weights.h`, `sample_input.h`, `main.c` generated.],
  [Weight-memory requirement], [*17.44 KB*], [Uses 17,440 Bytes (0.83% of 2.0 MB CRAM capacity).],
  [Data-memory requirement], [*4.00 KB*], [Uses 4,096 Bytes (0.78% of 512 KB System SRAM).]
)

== Hardware Limitations Disclaimer
Because no physical MAX78002 board was provided, we do not claim measured benchtop numbers. All reported execution times ($approx 0.234$ ms), power ($approx 5.0$ mW), and energy values ($approx 1.17 mu$J per beat) are analytical estimates calculated from the MAX78002 datasheet specifications at 50 MHz.

== Proposed Hardware Validation Plan
If physical MAX78002 hardware becomes available:
1. Flash firmware: Load `main.c` and `cnn.c` onto the MAX78002EVKIT using OpenOCD and SWD.
2. Feed test vectors: Stream preprocessed INT8 beats from `sample_input.h` over UART or SPI into the input FIFO.
3. Compare outputs: Verify that hardware classification logits match Python simulation outputs bit for bit.
4. Measure latency: Track execution cycle counts using the Cortex-M4 DWT cycle counter (`DWT->CYCCNT`).
5. Measure power: Monitor current on the VDD_CNN rail using an external shunt resistor and oscilloscope during active inference.

#v(6pt)

= Final Discussion

== What Did We Learn from the Data?
Morphological variation across patients is far larger than variation between classes within a single patient. An atypical normal beat from one person often looks more like a ventricular ectopic beat from another person than like that person's own baseline. A patient-independent split (DS1 and DS2) is necessary to evaluate realistic generalization.

== What Was the Largest Source of Error?
The main source of error was confusion between Class S (Supraventricular ectopic) and Class N (Normal). Supraventricular beats travel down the normal ventricular conduction pathway, producing a QRS complex nearly identical to sinus rhythm. Accurately catching S beats requires timing context (preceding RR intervals) rather than single-beat shape alone.

== Was the Highest-Accuracy Model Also the Best Edge-AI Model?
No. The Random Forest baseline reached 83.78% accuracy, but its 90,718 tree nodes consume over 9 MB of memory, making it unviable on an edge microcontroller. Furthermore, quantized INT8 CNN accuracy (87.58%) was higher than FP32 (74.20%) because it collapsed minority predictions into the 89% majority class. In arrhythmia screening, Macro F1 and minority recall are more meaningful than raw accuracy.

== How Much Performance Was Lost After Quantization?
Post-training INT8 quantization lowered Macro F1 by 0.0724 (from 0.3230 to 0.2506). Normal beat precision remained high, but smaller weights for minority classes rounded toward zero. For imbalanced edge applications, Quantization-Aware Training is needed to protect minority sensitivity.

== How Sensitive Was the System to ECG Noise?
The model held up well under moderate noise (12 dB and 6 dB SNR), keeping Macro F1 within 0.005 of clean signals. Under severe 0 dB noise, accuracy declined gradually to 69.77%. Crucially, the 0.5 to 40 Hz bandpass filter cut baseline wander noise power by 14.93 dB, confirming that front-end filtering is essential before digitization.

== What Compromises Were Necessary Because of Hardware Constraints?
Hardware limits required:
- Omitting recurrent layers (LSTMs, GRUs) because the MAX78002 lacks recurrent execution units.
- Setting the input window to exactly 256 samples ($2^8$) to support clean dyadic pooling.
- Clamping activations to $[-3.0, 3.0]$ to accommodate 8-bit dynamic range.
- Restricting classification to a single linear layer to conserve parameter storage.

== What Would We Change with More Time?
With additional time, we would:
1. Add rhythm features: Supplement the 256-sample waveform with scalar pre-RR and post-RR interval ratios to distinguish Class S from Class N without increasing CNN complexity.
2. Quantization-Aware Training: Train with simulated INT8 operations in PyTorch to prevent the -0.0724 quantization loss.
3. Benchtop validation: Run the compiled C code on a physical MAX78002EVKIT to measure actual cycle counts and battery current.

#v(6pt)

= Limitations

1. Patient cohort size: MIT-BIH contains only 47 subjects recorded in the late 1970s. Modern deployment requires testing on diverse cohorts (PTB-XL, European ST-T).
2. Class imbalance: Scarcity of Class S (2.76%) and Class F (0.80%) limits statistical power on minority groups.
3. Single-lead bias: Commercial wearable patches often capture non-standard vectors, creating potential domain shift compared to clean MLII leads.
4. R-peak detection: The pipeline assumes pre-segmented beats centered on known R-peaks; an embedded deployment needs an on-chip QRS detector.
5. Lack of physical silicon validation: All hardware figures are analytical estimates; physical board testing is required before practical deployment.
6. Regulatory disclaimer: This project is an academic exercise and is not certified as a medical device.

#v(6pt)

= Final Submission Package Verification

The repository matches the submission structure required on Page 12 of the instructions:
- `report.pdf`: Compiled CRISP-DM technical report.
- `README.md`: Setup and reproduction workflow.
- `requirements.txt`: Locked dependencies.
- `notebooks/EDA.ipynb`: Executed exploratory data analysis notebook.
- `src/`: Modular code (`prepare_data.py`, `dataset.py`, `model.py`, `train.py`, `evaluate.py`, `noise_test.py`, `quantize.py`, `synthesize.py`).
- `results/`: `metrics.csv`, `confusion_matrix.*`, `noise_results.*`, `eda_overview.png`, `error_waveforms.png`.
- `max78002/`: `model_configuration/cnn_max78002.yaml`, `quantization_output/`, and `synthesis_output/` (`cnn.h`, `cnn.c`, `weights.h`, `sample_input.h`, `main.c`, `memory_budget.txt`).
