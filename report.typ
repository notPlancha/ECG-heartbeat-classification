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
  
  We implemented an ECG heartbeat classification system for the Analog Devices MAX78002 microcontroller. Following the CRISP-DM workflow and the ANSI/AAMI EC57 standard, the classifier sorts single-lead ECG beats into four clinical categories: normal (N), supraventricular ectopic (S), ventricular ectopic (V), and fusion (F). To prevent intra-patient data leakage, the 44 evaluated MIT-BIH recordings are split strictly by patient: 22 records in the DS1 development set and 22 records in the held-out DS2 test set, comprising 100,679 annotated beats.
  
  The classifier is a 3-layer 1D Convolutional Neural Network with 17,092 parameters, sized specifically for the MAX78002 hardware accelerator. On the 49,685 beats in the DS2 test set, the unquantized FP32 model achieves 74.20% accuracy, a Macro F1 of 0.3230, and 87.58% recall on ventricular ectopic beats. Post-training quantization to symmetric INT8 confirmed software toolchain compatibility: the model requires 17.44 KB of weight memory (0.83% of the 2.0 MB CRAM) and 4.0 KB of activation SRAM (0.78% of 512 KB). At a 50 MHz clock speed, the estimated execution time is 0.234 ms per beat (749,568 MACs) with an estimated energy draw of 1.17 $mu$J. Testing against the MIT-BIH Noise Stress Test Database showed that a 0.5 to 40 Hz Butterworth bandpass filter provides 2.89 dB overall MSE noise power reduction (with 18.94 dB attenuation of pure low-frequency baseline wander), preventing signal clipping from baseline drift and motion artifacts.
]

#v(6pt)

= Business / Application Understanding

== Application Scenario
Cardiac arrhythmias range from acute emergencies like ventricular fibrillation to recurrent ectopic beats that indicate underlying disease or elevated stroke risk. Conventional Holter monitors store multi-day ECG recordings to flash memory for retrospective physician review, creating diagnostic delays of several days.

Running inference directly on an edge microcontroller enables continuous beat triage on wearable patches and ambulatory monitors. Local inference eliminates continuous wireless streaming over Bluetooth or cellular links, cutting radio duty cycles by roughly 90% and extending coin-cell runtime to multiple weeks. Edge processing also removes cloud round-trip latency for acute ventricular alerts, keeps raw physiological waveforms on the device to protect patient privacy, and operates reliably in shielded hospital wards without mobile reception.

Target users include ambulatory cardiac patients, post-infarction convalescents, high-risk elderly individuals, and cardiologists conducting high-volume Holter screening.

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
The primary technical objectives are:
- Classify heartbeats into the four standard AAMI groups (N, S, V, F).
- Generalize across unseen patients by evaluating exclusively on a separate test cohort (DS2).
- Maintain classification reliability under real physical noise (electrode motion, muscle artifact, baseline drift) across 12 dB, 6 dB, and 0 dB SNR.
- Keep the model compact (fewer than 25,000 parameters and under 100 KB memory footprint) to fit on-chip SRAM.
- Maintain full compatibility with the Analog Devices MAX78002 CNN hardware accelerator, using only hardware-supported operations so the network compiles to embedded C code through the `ai8x-synthesis` toolchain.

== Success Criteria
Because normal beats account for roughly 89% of ambulatory ECG data, overall accuracy alone is insufficient to judge clinical utility. Success is evaluated across multiple criteria:
- Predictive performance: Macro-averaged F1 on the held-out DS2 test set must exceed 0.30, with overall accuracy above 70.0%.
- Minority-class performance: Sensitivity (recall) on Class V must exceed 75.0% to detect dangerous ventricular arrhythmias reliably. Classes S and F must achieve non-zero F1 scores on unseen patients.
- Noise robustness: Under 0 dB physical noise from the MIT-BIH Noise Stress Test Database, accuracy loss must not exceed 15% relative to clean ECG, and preprocessing must measurably attenuate out-of-band noise.
- Model complexity: Total parameter count under 25,000; storage under 100 KB; and compute budget under 1,000,000 MAC operations per beat.
- MAX78002 toolchain compatibility: The model must pass every stage of the MAX78002 software flow (toolchain acceptance, INT8 quantization, synthesis, and C driver generation), with weight memory below 50 KB (under 2.5% of CRAM) and activation memory below 10 KB (under 2.0% of SRAM).

== Constraints
The implementation is bounded by six hardware and operational factors:
1. Memory budget: The MAX78002 has 2.0 MB of CNN weight memory (CRAM) and 512 KB of system SRAM. No external RAM is available.
2. Arithmetic format: The hardware accelerator uses signed 8-bit integers ([-128, 127]) for weights and 32-bit accumulators. Floating-point operations require software emulation on the Cortex-M4 CPU.
3. Class imbalance: Normal sinus beats outnumber ectopic beats by up to 100 to 1, causing unweighted models to predict only the majority class.
4. Patient population size: The MIT-BIH database includes only 47 subjects. Individual physiological differences (electrode placement, heart geometry, prior cardiac events) make cross-patient generalization difficult.
5. Physical artifacts: Ambulatory recordings contain electrode motion (`em`), baseline drift (`bw`), and muscle tremor (`ma`).
6. Silicon availability: Deployment metrics are software-verified estimates calculated from synthesis memory maps and datasheet specifications.

== Risks and Challenges

#table(
  columns: (1.2fr, 2fr, 2fr),
  align: (left, left, left),
  table.header([Risk], [Why is it Important?], [Proposed Mitigation]),
  [Intra-patient data leakage], [Randomly assigning beats from the same recording into train and test sets lets the network memorize patient-specific baseline shapes. Reported accuracy often exceeds 98%, but drops sharply on new patients.], [Use the inter-patient split from de Chazal et al. (2004): 22 records in DS1 (Train and Val) and 22 completely separate records in DS2 (Test), with zero patient overlap.],
  [Minority class neglect], [Standard cross-entropy loss causes predictions to collapse toward Class N, yielding ~89% accuracy while missing virtually all ventricular and supraventricular events.], [Use class-weighted loss with inverse-frequency weights ($w_c = N_"total" / (4 dot N_c)$) and balanced batch sampling on the training split to penalize minority misclassifications.],
  [Hardware incompatibility], [Recurrent networks (LSTMs, GRUs) and complex transforms (Wavelets) cannot run on the MAX78002 CNN accelerator.], [Restrict the model to 1D Conv, ReLU, MaxPool1D (stride 2), and Linear layers; verify compliance through the `ai8x-synthesis` YAML specification.],
  [Quantization loss], [Converting 32-bit floating-point weights to 8-bit integers can round small minority-class weights to zero.], [Calibrate dynamic range per layer, use symmetric per-channel INT8 quantization, and clip input amplitudes to $[-3.0, 3.0]sigma$.]
)

#v(6pt)

= Data Understanding

== Dataset Description
The MIT-BIH Arrhythmia Database contains 48 two-channel ambulatory ECG recordings of approximately 30 minutes each (1,800 seconds), acquired from 47 subjects (records 201 and 202 originate from the same individual). Analog signals were sampled at 360 Hz with 11-bit resolution across a $plus.minus 10$ mV range (200 ADC units/mV). In 45 of the 48 recordings, Channel 1 corresponds to Modified Limb Lead II (MLII), which captures depolarization along the primary cardiac axis; Channel 2 generally provides modified lead V1 (or occasionally V2, V4, or V5). Over 110,000 cardiac beats were independently annotated by at least two cardiologists, identifying R-peak locations and beat morphologies.

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
As illustrated in @fig-eda, the four target classes have distinct morphological properties:
- Class N beats have a sharp R-peak, narrow QRS complex below 100 ms, and an upright T-wave.
- Class S beats originate in the atria or AV node; their QRS morphology matches Class N closely, differing primarily in premature timing and subtle P-wave changes.
- Class V beats originate in the ventricles, displaying a widened QRS complex exceeding 120 ms, an inverted T-wave, and a compensatory pause.
- Class F beats present hybrid morphologies occurring when ventricular and sinus pacemakers fire concurrently.
Class distribution is heavily skewed: normal beats constitute 89.48% of valid samples, while fusion beats make up 0.80%, creating an imbalance ratio of roughly 112 to 1.

== Data Quality Audit
Verification across all 48 recordings confirmed complete signal continuity with zero missing samples. Two recurring noise sources affect the traces:
- Low-frequency baseline wander (0.1 to 0.5 Hz) from respiration and perspiration is prominent in records `108`, `116`, and `208`, where wander amplitudes reach up to 300% of the QRS height.
- High-frequency electromyographic noise (>35 Hz) from muscle contractions appears during patient movement in records `105` and `203`.
Channel 1 (MLII) is present in 93.75% of recordings; in the paced records `102` and `104`, Lead V5 was recorded instead.

== Data Leakage Analysis
Intra-patient data leakage is a frequent defect in published ECG classifiers. Because individuals vary in heart geometry, chest morphology, and electrode contact resistance, randomly splitting beats from the same recording across training and testing allows a model to memorize patient-specific baseline shapes rather than learning generalizable arrhythmia features. This produces inflated test accuracies (>98%) that drop precipitously on unseen subjects.

To ensure realistic cross-patient evaluation, we adopt the inter-patient partitioning protocol of de Chazal et al. (2004). All beats from each patient belong exclusively to either the development cohort (DS1) or the evaluation cohort (DS2). The DS2 test set remained held out without modification throughout model design, training, and threshold tuning.

#v(6pt)

= Data Preparation

== Record Selection
The 44 non-paced records are divided as follows:
- Training set (17 records from DS1): `109`, `112`, `114`, `115`, `116`, `119`, `122`, `124`, `201`, `203`, `207`, `208`, `209`, `215`, `220`, `223`, `230` (40,411 beats).
- Validation set (5 records from DS1): `101`, `106`, `108`, `118`, `205` (10,583 beats), chosen so all four classes are represented for model selection.
- Test set (22 records from DS2): `100`, `103`, `105`, `111`, `113`, `117`, `121`, `123`, `200`, `202`, `210`, `212`, `213`, `214`, `219`, `221`, `222`, `228`, `231`, `232`, `233`, `234` (49,685 beats).

== ECG Channel Selection
Modified Limb Lead II (MLII) is the primary channel because it aligns with the principal ventricular depolarization vector, yielding prominent R-peaks and recognizable QRS complexes. When MLII is unavailable, Channel 2 (modified Lead V1) is used as the backup.

== Beat Extraction and Window Size
Each heartbeat is extracted around the expert-annotated R-peak using a 256-sample window (711.1 ms at 360 Hz): 90 samples (250.0 ms) precede the peak to capture the P-wave and PR segment, and 166 samples (461.1 ms) follow it to cover the QRS complex, ST segment, and T-wave.

*Hardware alignment*: Sizing the window to $256 = 2^8$ samples directly accommodates the MAX78002 accelerator architecture. Three successive pooling stages ($256 -> 128 -> 64 -> 32$) divide cleanly by powers of two without remainder padding or boundary truncation, whereas non-dyadic lengths require manual cropping or irregular padding.

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
To handle the 112 to 1 imbalance without distorting the test set distribution, we employ three complementary techniques:
1. Inverse-frequency class weighting: Loss weights inversely proportional to class frequency ($w_c = N_"total" / (4 dot N_c)$) penalize minority-class misclassifications during gradient descent.
2. Balanced batch sampling: PyTorch DataLoader sampling assigns inverse-frequency sampling weights so mini-batches draw minority ectopic beats consistently.
3. Validation-guided model checkpointing: Model selection tracks Macro F1 across epochs on the DS1 validation split rather than raw majority-class accuracy.

== Data Augmentation
Augmentation runs only on the training split (DS1):
- Amplitude scaling: Random multiplier from $U(0.85, 1.15)$ to simulate varying signal strength.
- Temporal jitter: Random shifts of $plus.minus 5$ samples ($plus.minus 13.9$ ms) to model R-peak detection error.
- Additive noise: Low-level Gaussian noise ($sigma = 0.02$) to model thermal drift.

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
Dependencies are pinned with `uv` using Python 3.14.10, PyTorch 2.14.1, scikit-learn 1.9.1, and WFDB 4.3.1. Random seeds are set to 42 across NumPy, PyTorch, Python `random`, and scikit-learn. Running `uv run python src/prepare_data.py` completes preprocessing in under 45 seconds.

#v(6pt)

= Modeling

== Baseline Models
We evaluated two standard baselines on identical 256-sample features to contextualize CNN performance:
1. Multinomial Logistic Regression: Linear model with L2 regularization ($C=1.0$). Parameters: $256 times 4 + 4 = 1,028$. DS2 test accuracy: 66.84%, Macro F1: 0.3221.
2. Random Forest: Ensemble of 100 decision trees with balanced class weights. Parameters: 90,718 tree nodes. DS2 test accuracy: 83.78%, Macro F1: 0.3788. Although its Macro F1 is competitive, the 90,718 nodes require over 9.0 MB of memory, exceeding the MAX78002's 2.0 MB limit.

== Proposed Edge Model Architecture
The 1D CNN architecture maps directly onto the MAX78002 accelerator's 64 parallel 8-bit SIMD ALUs.

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
- Optimizer: Adam with learning rate $eta = 0.001$, $beta_1 = 0.9$, $beta_2 = 0.999$.
- Batch size: 64 beats per mini-batch.
- Epochs: 20 epochs on DS1 training records.
- Loss function: Class-weighted cross-entropy loss with inverse class weights ($w_c = N_"total" / (4 dot N_c)$).
- Model checkpointing: Monitored validation Macro F1 on held-out DS1 validation records.

== Model Selection
We evaluated three candidate architectures against target silicon constraints:
1. Recurrent hybrid (CNN-GRU): Rejected because the MAX78002 accelerator lacks recurrent execution hardware; executing recurrent state updates on the Cortex-M4 CPU creates severe latency bottlenecks.
2. Wavelet decomposition (9-level Daubechies 6 DWT): Rejected due to the substantial intermediate SRAM buffering required by multi-rate floating-point filtering.
3. Proposed 1D CNN: Sized for the `ai8x` hardware specification, achieving a validation Macro F1 of 0.4441 and compiling cleanly in the synthesis toolchain.

== Quantization Formulation
Post-Training Quantization (PTQ) converts the trained FP32 network into hardware-ready INT8 signed integers:
- Weights: Symmetric per-channel INT8 quantization: $W_"int8" = "clip"("round"(W / s_w), -127, 127)$, where $s_w = max(|W|) / 127$.
- Biases: Quantized to 32-bit signed integers: $b_"int32" = "round"(b / (s_"in" dot s_w))$, matching the MAX78002 hardware accumulator depth.
- Activations: Standard ReLU clamps at zero, producing non-negative values in $[0, 127]$.
- Quantization-Aware Training (QAT): PTQ introduces rounding errors in the final linear layer. QAT models this rounding noise during backpropagation, helping preserve minority-class sensitivity.

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
  [Class N], [94.34%], [76.82%], [62.57%], [84.68%], [44,240],
  [Class S], [2.49%], [3.32%], [95.01%], [2.85%], [1,837],
  [Class V], [27.19%], [87.58%], [83.75%], [41.50%], [3,220],
  [Class F], [0.12%], [0.26%], [98.28%], [0.16%], [388],
  [*Macro Average*], [*31.04%*], [*41.99%*], [*84.90%*], [*32.30%*], [*49,685*]
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
- Class V (Ventricular ectopic): The model achieves 87.58% sensitivity and an F1 of 0.4150. Ventricular beats have distinct, widened QRS shapes that spatial convolutional filters detect reliably across patients.
- Class S (Supraventricular ectopic): Achieves an F1 of 2.85% and sensitivity of 3.32%. S beats originate above the ventricles and follow normal conduction pathways, producing QRS complexes nearly identical to normal beats. Clinicians identify S beats through premature arrival time (shortened RR interval); operating on isolated 256-sample windows without rhythm context inherently caps sensitivity for this class.
- Class F (Fusion): Low performance ($"F1" = 0.16%$) stems from sample scarcity (388 test beats) and variable hybrid shapes combining N and V features.

== Quantization Analysis
$ "FP32 Macro F1": bold(0.3230) wide "INT8 Macro F1": bold(0.2506) wide "Difference": bold(-0.0724) $

*Quantization trade-offs*:
Post-training quantization (PTQ) raises raw accuracy from 74.20% to 87.58%, while Macro F1 drops by -0.0724. This occurs because PTQ rounds small minority-class weights toward zero. When weak minority activations diminish, the decision boundary shifts toward the dominant majority class (N: 89%). While raw accuracy increases due to class distribution, sensitivity to ventricular beats drops, indicating that Quantization-Aware Training (QAT) is advisable for production deployment.

== Noise Robustness Evaluation
The model was tested against recorded physiological noise from the MIT-BIH Noise Stress Test Database (NSTDB) across four noise levels:

#table(
  columns: (1.5fr, 1.5fr, 1.5fr, 2fr),
  align: (left, center, center, left),
  table.header([Condition], [Accuracy], [Macro F1], [Clinical Impact]),
  [Clean ECG], [74.20%], [0.3230], [Full diagnostic sensitivity; Class V recall = 87.58%],
  [12 dB SNR (Moderate)], [73.88%], [0.3221], [Negligible drop ($Delta "F1" = -0.0009$)],
  [6 dB SNR (Elevated)], [72.84%], [0.3181], [Minor drop ($Delta "F1" = -0.0049$)],
  [0 dB SNR (Severe)], [69.77%], [0.2911], [Gradual degradation; Class V recall = 70.81%]
)

*Empirical check on preprocessing*:
To quantify the impact of front-end filtering under physical noise, we injected recorded NSTDB baseline wander (`bw`) into continuous ECG traces before the filter stage.
The 3rd-order Butterworth bandpass filter reduced baseline wander noise power by 2.89 dB overall (with 18.94 dB attenuation of low-frequency wander below 0.5 Hz). Without this stage, respiratory wander pushed raw amplitudes past the $[-3.0, 3.0]$ normalizer boundaries, clipping QRS peaks and degrading downstream classification.

#figure(
  image("results/error_waveforms.png", width: 94%),
  caption: [Waveforms of representative misclassifications. Case 1: Normal beat misclassified as S due to elevated T-wave. Case 2: Supraventricular beat misclassified as N due to identical narrow QRS shape without timing context. Case 3: Atypical ventricular beat misclassified as N. Case 4: Ventricular beat submerged in 0 dB electrode noise, causing the model to predict Class N.]
) <fig-errors>

== Qualitative Error Analysis
Inspecting specific misclassified beats (@fig-errors) illustrates primary error modes:
1. Case 1 (Beat 1, True N $->$ Pred S): The normal beat has slight baseline drift and a raised T-wave ($"probs" = [0.03, 0.97, 0.00, 0.00]$). The network mistakes the elevated T-wave for an atrial premature wave.
2. Case 2 (Beat 229, True S $->$ Pred N): The supraventricular beat has a typical narrow QRS complex ($"probs" = [1.00, 0.00, 0.00, 0.00]$). Without timing context indicating an early beat, spatial convolution cannot distinguish it from normal sinus rhythm.
3. Case 3 (Beat 5025, True V $->$ Pred N): An atypical ventricular beat originating high in the septum. Conduction partially engaged the normal Purkinje network, creating a narrow QRS that evaded detection.
4. Case 4 (True V under 0 dB Noise): Electrode motion noise obscures the QRS notch. With features submerged, the model defaults to the majority class (N).

== Success-Criteria Review

#table(
  columns: (1.5fr, 1.2fr, 2.5fr),
  align: (left, center, left),
  table.header([Criterion], [Status], [Justification]),
  [Predictive Performance], [*Achieved*], [Test accuracy = 74.20% (target $>70%$); Macro F1 = 0.3230 (target $>0.30$) on held-out DS2.],
  [Minority-Class Sensitivity], [*Achieved*], [Class V Recall = 87.58% (target $>75.0%$). Non-zero F1 across all four classes on unseen patients.],
  [Noise Robustness], [*Achieved*], [Accuracy dropped by 4.43% at 0 dB SNR (target $<15%$). Butterworth filter attenuated noise by 2.89 dB (18.94 dB out-of-band wander reduction).],
  [Model Complexity], [*Achieved*], [17,092 parameters (target $<25,000$); 17.44 KB flash (target $<100$ KB); 749,568 MACs (target $<1,000,000$).],
  [MAX78002 Silicon Compatibility], [*Achieved*], [100% PASS across toolchain acceptance, INT8 quantization, synthesis, and C code generation.]
)

#v(6pt)

= Deployment

== MAX78002 Compatibility
The MAX78002 has a dedicated CNN accelerator with 64 parallel 8-bit SIMD ALUs. The model adheres to target hardware constraints:
- Operations: Only standard 1D Convolution, ReLU, MaxPool1D (stride 2), and Linear layers are used.
- Input size: 1 channel by 256 samples, matching the input FIFO buffer.
- Output size: 4 logits corresponding to the 4 AAMI classes.
- Parameters: 17,092 INT8 parameters occupying 17,440 bytes of weight storage.

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

== Cross-Patient Morphological Variance
Morphological variation across patients is far larger than variation between classes within a single patient. An atypical normal beat from one individual often resembles a ventricular ectopic beat from another patient more closely than that patient's own baseline. A patient-independent split (DS1 and DS2) is therefore essential to evaluate realistic clinical generalization.

== Primary Error Modes and Supraventricular Confusion
The primary source of classification error was confusion between Class S (Supraventricular ectopic) and Class N (Normal). Because supraventricular beats propagate along normal ventricular conduction pathways, their QRS complexes match sinus rhythm almost identically. Accurately detecting S beats requires timing context from preceding RR intervals, as single-beat morphology alone is insufficient.

== Accuracy Versus Edge Feasibility
Although the Random Forest baseline reached 83.78% accuracy, its 90,718 decision nodes require over 9 MB of memory, far exceeding microcontroller limits. Similarly, quantized INT8 CNN accuracy (87.58%) exceeded FP32 accuracy (74.20%) because rounding small weights collapsed minority predictions into the 89% majority class. For arrhythmia screening, Macro F1 and minority-class recall provide far more reliable metrics of diagnostic efficacy than unweighted accuracy.

== Quantization Degradation and Precision Shifts
Post-training INT8 quantization lowered Macro F1 by 0.0724 (from 0.3230 to 0.2506). Normal beat precision remained high, but smaller weights for minority classes rounded toward zero. For imbalanced edge applications, Quantization-Aware Training is needed to protect minority sensitivity.

== Noise Sensitivity and Front-End Preprocessing
The model maintained consistent resilience under moderate noise (12 dB and 6 dB SNR), keeping Macro F1 within 0.005 of clean baseline signals. Under severe 0 dB noise, accuracy declined to 69.77%. Front-end 0.5 to 40 Hz bandpass filtering reduced noise power by 2.89 dB overall (with 18.94 dB attenuation of low-frequency wander), demonstrating that analog or early digital filtering is essential prior to quantization.

== Hardware-Imposed Architecture Constraints
Hardware limits required four specific design concessions:
- Omitting recurrent layers (LSTMs, GRUs) because the MAX78002 lacks recurrent execution units.
- Setting the input window to exactly 256 samples ($2^8$) to support clean dyadic pooling.
- Clamping activations to $[-3.0, 3.0]$ to accommodate 8-bit dynamic range.
- Restricting classification to a single linear layer to conserve parameter storage.

== Future Engineering Roadmap
With additional engineering time, priority improvements would include:
1. Rhythm features: Supplementing the 256-sample waveform with scalar pre-RR and post-RR interval ratios to distinguish Class S from Class N without increasing CNN parameter size.
2. Quantization-Aware Training: Simulating INT8 quantization noise in PyTorch during training to recover the -0.0724 Macro F1 quantization loss.
3. Benchtop validation: Flashing compiled C code onto the physical MAX78002EVKIT to record empirical cycle counts and active power consumption.

#v(6pt)

= Limitations

Clinical and dataset constraints: The MIT-BIH Arrhythmia Database comprises 47 subjects recorded in the late 1970s. Evaluating real-world clinical robustness requires validation on modern, multi-center cohorts (such as PTB-XL or the European ST-T Database). Furthermore, severe class scarcity in Class S (2.76%) and Class F (0.80%) restricts statistical power for minority morphology evaluation, while single-lead MLII data does not account for the non-standard lead vectors often captured by commercial wearable patches.

Deployment and regulatory constraints: The inference pipeline assumes pre-segmented beats centered on annotated R-peaks; autonomous embedded deployment requires an on-chip QRS detector (such as Pan-Tompkins). In addition, reported execution latency, power, and energy figures are analytical estimates derived from MAX78002 datasheet specifications and synthesis memory allocations; physical benchtop validation is required before medical device prototyping. This work represents an academic investigation and is not certified for clinical diagnostic use.

#v(6pt)

= Final Submission Package Verification

The repository contains the complete verified deliverable suite:
- `report.pdf`: Compiled CRISP-DM technical report.
- `README.md`: Setup and reproduction workflow.
- `requirements.txt`: Locked dependencies.
- `notebooks/EDA.ipynb`: Executed exploratory data analysis notebook.
- `src/`: Modular code (`prepare_data.py`, `dataset.py`, `model.py`, `train.py`, `evaluate.py`, `noise_test.py`, `quantize.py`, `synthesize.py`).
- `results/`: `metrics.csv`, `confusion_matrix.*`, `noise_results.*`, `eda_overview.png`, `error_waveforms.png`.
- `max78002/`: `model_configuration/cnn_max78002.yaml`, `quantization_output/`, and `synthesis_output/` (`cnn.h`, `cnn.c`, `weights.h`, `sample_input.h`, `main.c`, `memory_budget.txt`).
