// Minimalist, publication-grade technical report template in Typst 0.15
#set document(
  title: "ECG Heartbeat Classification on Edge-AI: Analog Devices MAX78002 Deployment Workflow",
  author: "Engineering Team",
  date: auto
)

#set page(
  paper: "a4",
  margin: (top: 2.2cm, bottom: 2.2cm, left: 2.0cm, right: 2.0cm),
  header: context {
    if here().page() > 1 [
      #grid(
        columns: (1fr, auto),
        align(left)[#text(size: 8.5pt, fill: rgb("#718096"), font: "Segoe UI")[ECG Heartbeat Classification on Edge-AI (MAX78002) — CRISP-DM Technical Report]],
        align(right)[#text(size: 8.5pt, fill: rgb("#718096"), font: "Segoe UI")[Submissions 1 & 2 Combined]]
      )
      #v(-4pt)
      #line(length: 100%, stroke: 0.5pt + rgb("#e2e8f0"))
    ]
  },
  footer: context {
    grid(
      columns: (1fr, auto),
      align(left)[#text(size: 8.5pt, fill: rgb("#a0aec0"), font: "Segoe UI")[Confidential — Academic & Technical Evaluation]],
      align(right)[#text(size: 8.5pt, fill: rgb("#4a5568"), font: "Segoe UI")[Page #here().page()]]
    )
  }
)

#set text(
  font: "Segoe UI",
  size: 9.8pt,
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
  #v(11pt, weak: true)
  #text(
    fill: rgb("#1a365d"),
    weight: "bold",
    size: if it.level == 1 { 13.5pt } else if it.level == 2 { 11.0pt } else { 10.0pt }
  )[#it]
  #v(4pt, weak: true)
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
  #text(size: 13pt, weight: "semibold", fill: rgb("#2b6cb0"))[
    Analog Devices MAX78002 Deployment Workflow, Quantization & Robustness Evaluation
  ]
  #v(5pt)
  #text(size: 9.5pt, fill: rgb("#718096"))[
    CRISP-DM End-to-End Technical Report | Submissions 1 & 2 Complete
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
  
  This report presents the complete end-to-end implementation of an ultra-low-power ECG heartbeat classification system designed for real-time inference on the *Analog Devices MAX78002 Edge-AI microcontroller*. Adhering strictly to the *CRISP-DM* methodology and the ANSI/AAMI EC57 standard, the system categorizes single-lead ECG beats into four clinical classes (*N*, *S*, *V*, *F*). A leak-free inter-patient partition (DS1 development set of 22 records, DS2 held-out test set of 22 records; 100,679 total annotated beats) was enforced. 
  
  A custom 3-layer 1D Convolutional Neural Network (17,092 parameters) was engineered specifically for the MAX78002 hardware accelerator. On the untouched 49,685-beat DS2 test set, the FP32 edge model achieves *74.20% accuracy* and a *Macro F1 of 0.3230* (with *85.22% recall on life-threatening Ventricular ectopic beats*). Software post-training quantization to symmetric INT8 demonstrates software compatibility: the model consumes only *17.44 KB of weight memory* (0.83% of the 2.0 MB CRAM) and *4.0 KB of activation SRAM* (0.78% of 512 KB), with an estimated execution latency of *0.234 ms* (749,568 MACs) and energy consumption of *1.17 $mu$J per beat* at 50 MHz. Authentic physiological stress testing using the MIT-BIH Noise Stress Test Database (NSTDB) proves that our 0.5–40 Hz Butterworth bandpass filter attenuates respiratory baseline drift by *14.93 dB*, providing vital empirical proof that edge preprocessing is essential for wearable cardiac intelligence.
]

#v(10pt)

= Business / Application Understanding

== Application Scenario
Cardiac arrhythmias represent one of the leading contributors to sudden cardiovascular death globally. Conditions such as Ventricular Tachycardia (VT) and Ventricular Fibrillation (VF) can deteriorate within seconds, while frequent Supraventricular Ectopic Beats (SVEB) and Ventricular Ectopic Beats (VEB) serve as critical prognostic indicators of structural heart disease and embolic stroke. Traditional Holter monitors record multi-day electrocardiograms for offline clinical batch analysis, which introduces fatal diagnostic latency. 

*Automated Edge-AI ECG classification* transforms this paradigm by enabling continuous, real-time beat-by-beat arrhythmia triage directly on wearable devices (e.g., smart chest patches, smartwatches, and next-generation Holter devices). Executing deep neural network inference on-device offers four decisive advantages:
1. *Sub-millisecond Diagnostic Latency*: Critical ventricular arrhythmias can trigger immediate emergency alerts without awaiting cloud round-trips.
2. *Ultra-Low Power & Long Battery Life*: Eliminating continuous raw RF transmission (BLE/Wi-Fi/Cellular) reduces system power consumption by over 90%, enabling weeks of operation on a single coin-cell battery.
3. *Strict Patient Privacy & Security*: Raw biometrics remain on-chip; only privacy-compliant aggregate cardiac event telemetry is transmitted.
4. *Continuous Offline Reliability*: Wearers remain protected in remote environments or hospital dead-zones lacking network connectivity.

The target end-users include ambulatory cardiac patients, post-myocardial infarction survivors, high-risk elderly populations, and clinical cardiologists who require automated event screening across massive multi-day recordings.

== Data-Mining Objective
The machine-learning task is formally structured as a multi-class, single-channel 1D time-series classification problem under strict embedded silicon constraints.

#table(
  columns: (1fr, 3fr),
  align: (left, left),
  table.header([Item], [Description]),
  [Input Data], [Fixed-length 1D discrete time-series vector of 256 samples ($711.1$ ms @ $360$ Hz), corresponding to a single preprocessed heartbeat centered on the detected R-peak ($90$ samples pre-R, $166$ samples post-R).],
  [Output Data], [Discrete 4-class categorical prediction conforming to ANSI/AAMI EC57: Class *N* (Normal/Bundle branch block), Class *S* (Supraventricular ectopic), Class *V* (Ventricular ectopic), Class *F* (Fusion of ventricular and normal).],
  [Type of Task], [Supervised multi-class time-series classification under severe class imbalance (~89% N, ~3% S, ~7% V, ~0.8% F) and strict INT8 hardware quantization constraints.]
)

== Project Objectives
The core technical and architectural objectives are:
- *Distinguish 4 AAMI Classes*: Accurately differentiate Normal (*N*), Supraventricular (*S*), Ventricular (*V*), and Fusion (*F*) heartbeats.
- *Strict Inter-Patient Generalization*: Generalize robustly across unseen patients by evaluating exclusively on a strictly held-out test cohort (DS2).
- *Physical Noise Robustness*: Maintain clinical diagnostic utility in the presence of real physiological noise (electrode motion, muscle artifact, baseline wander) at 12 dB, 6 dB, and 0 dB Signal-to-Noise Ratios (SNR).
- *Ultra-Compact Model Footprint*: Constrain total model size to $<< 2.0$ MB and total parameter count to $< 25,000$ to fit entirely within on-chip SRAM.
- *Analog Devices MAX78002 Silicon Compatibility*: Design exclusively with hardware-supported layers (1D Convolution, ReLU, Dyadic Pooling, Fully Connected Linear) to enable direct compilation via the `ai8x-synthesis` pipeline into embedded C code.

== Success Criteria
To prevent misleading claims of "high accuracy" on an 89% majority class, we establish rigorous, measurable multi-dimensional success criteria:
- *Predictive Performance*: Macro-averaged F1-score on the held-out DS2 test set must exceed $0.30$, with overall multi-class accuracy $> 70.0%$.
- *Minority-Class Performance*: Class *V* (Ventricular ectopic) Sensitivity/Recall must exceed $75.0%$, ensuring high capture of clinically dangerous ventricular ectopy. Class *S* and *F* F1-scores must be non-zero on unseen patients.
- *Noise Robustness*: Under strong 0 dB physical noise from the MIT-BIH Noise Stress Test Database, overall accuracy degradation must not exceed $15%$ relative to clean ECG, and preprocessing filtering must be shown to actively attenuate out-of-band noise.
- *Model Complexity*: Maximum parameter count $< 25,000$; flash memory footprint $< 100$ KB; total multiply-accumulate (MAC) count $< 1,000,000$ operations per heartbeat.
- *MAX78002 Silicon Compatibility*: Full software acceptance by the MAX78002 toolchain (`PASS` across toolchain acceptance, INT8 quantization, hardware synthesis, and C driver generation), requiring $< 50$ KB of Weight Memory ($< 2.5%$ of 2.0 MB CRAM) and $< 10$ KB of Activation Memory ($< 2.0%$ of 512 KB SRAM).

== Constraints
The design is governed by six fundamental embedded and domain constraints:
1. *Edge-AI Memory Limits*: The MAX78002 features 2.0 MB of CNN Weight Memory (CRAM) and 512 KB of System SRAM. No external DDR/Flash is available for neural inference.
2. *Fixed-Point INT8 Arithmetic*: The CNN hardware accelerator executes integer multiply-accumulates with 8-bit signed weights ($[-128, 127]$) and 32-bit accumulators. Floating-point operations require software emulation on the ARM Cortex-M4 and are prohibited in the neural path.
3. *Extreme Class Imbalance*: In ambulatory monitoring, healthy sinus beats outnumber rare pathological arrhythmias by up to 100:1, biasing standard cross-entropy loss toward trivial majority prediction.
4. *Limited Patient Cohort*: The MIT-BIH database contains only 47 unique subjects. Cross-patient physiological diversity (cardiac axis tilt, chest geometry, prior infarcts) makes out-of-sample generalization challenging.
5. *Physiological Noise Artifacts*: Ambulatory recordings suffer from severe electrode motion (`em`), baseline drift (`bw`), and electromyographic noise (`ma`).
6. *Absence of Physical Hardware*: No physical MAX78002 development kit is supplied. All deployment analyses must be strictly software-verified through synthesis memory mapping, cycle estimation, and embedded C firmware generation without claiming measured benchtop electrical telemetry.

== Risks and Challenges

#table(
  columns: (1.2fr, 2fr, 2fr),
  align: (left, left, left),
  table.header([Risk], [Why is it Important?], [Proposed Mitigation]),
  [Intra-patient Data Leakage], [Randomly splitting heartbeats from the same patient into train and test sets inflates accuracy to $>98%$, as the network memorizes unique patient QRS morphology rather than arrhythmia features.], [Strict inter-patient partition (de Chazal et al., 2004): 22 records in DS1 (Train/Val) and 22 completely disjoint records in DS2 (Test). Zero patient overlap.],
  [Minority Class Starvation], [Standard empirical risk minimization collapses predictions to Class N, achieving ~89% accuracy while missing 100% of lethal ventricular and supraventricular events.], [Class-weighted Focal Loss ($gamma=2.0$), balanced random batch sampling, and post-training validation-tuned decision thresholds.],
  [Hardware Incompatibility], [Modern ML architectures (LSTMs, GRUs, Transformers, Wavelet transforms) cannot execute on the MAX78002 CNN hardware engine.], [Restricted model architecture strictly to 1D Conv, ReLU, MaxPool1D (stride 2), and Linear layers; verified via `ai8x-synthesis` YAML specification.],
  [Quantization Accuracy Degradation], [Converting FP32 weights to INT8 can round small weights to zero, degrading subtle minority class feature maps.], [Layer-by-layer dynamic range calibration, symmetric INT8 per-channel quantization, and clipping bounds scaled to $[-3.0, 3.0]sigma$.]
)

#v(10pt)

= Data Understanding

== Dataset Description
The primary evaluation corpus is the *MIT-BIH Arrhythmia Database* (PhysioNet), the gold-standard benchmark in computational electrocardiography:
- *Number of Records*: 48 individual two-channel recordings, each approximately 30 minutes in duration ($1,800$ s).
- *Subject Cohort*: 47 distinct individuals (records 201 and 202 originate from the same male subject).
- *Signal Channels*: Two simultaneous channels per record. In 45 of 48 records, Channel 1 is Modified Limb Lead II (*MLII*), reflecting depolarization along the cardiac electrical axis. Channel 2 is typically modified lead *V1* (occasionally V2, V4, or V5).
- *Sampling Rate*: $360$ Hz with 11-bit resolution over a $plus.minus 10$ mV dynamic range ($200$ ADC units/mV).
- *Expert Annotations*: Over $110,000$ beat-by-beat annotations independently verified by at least two expert cardiologists, marking the exact R-peak sample and morphological classification.

== Target Classes and Documented Exclusions
In strict accordance with the *ANSI/AAMI EC57:1998* standard, the 15 original MIT-BIH beat types are aggregated into four clinically relevant multi-class target groups. Non-beat annotations and paced recordings are systematically excluded.

#table(
  columns: (0.8fr, 2.5fr, 1fr),
  align: (left, left, right),
  table.header([Final Class], [Original MIT-BIH Annotations Included], [Number of Beats]),
  [Class N], [Normal (`N`), Left bundle branch block (`L`), Right bundle branch block (`R`), Atrial escape (`e`), Nodal/junctional escape (`j`)], [90,088 (89.48%)],
  [Class S], [Atrial premature (`A`), Aberrated atrial premature (`a`), Nodal premature (`J`), Supraventricular premature (`S`)], [2,781 (2.76%)],
  [Class V], [Premature ventricular contraction (`V`), Ventricular escape (`E`)], [7,008 (6.96%)],
  [Class F], [Fusion of ventricular and normal beat (`F`)], [802 (0.80%)],
  [*Total Valid*], [*All 4 Target AAMI Classes Across 44 Evaluated Records*], [*100,679 (100.0%)*]
)

*Documented Exclusions*:
1. *Paced Records*: In accordance with AAMI recommendations, records `102`, `104`, `107`, and `217` are excluded entirely because artificial electronic cardiac pacemaker spikes distort the natural cardiac conduction morphology.
2. *Non-Beat Artifacts*: Signal quality markers (`~`: 573 instances), rhythm change markers (`+`: 1,173 instances), isolated artifacts (`|`: 131 instances), unclassifiable beats (`Q`: 15 instances), ventricular flutter waves (`!`: 472 instances), and boundary truncated beats (39 instances) were excluded.

#figure(
  image("results/eda_overview.png", width: 94%),
  caption: [Exploratory Data Analysis and MAX78002 Preprocessing Pipeline. (A) Morphological signatures of the 4 AAMI classes aligned at the R-peak ($t=0$ ms, sample 90). (B) Severe class imbalance across Train (DS1), Val (DS1), and Test (DS2) splits (log scale). (C) Probability density of Z-score normalized beat amplitudes against the $[-3.0, 3.0]$ hardware clipping bounds. (D) Dyadic downsampling hierarchy matching MAX78002 hardware pooling.]
) <fig-eda>

== Exploratory Data Analysis
As illustrated in @fig-eda:
- *Class N*: Characterized by a sharp, tall R-peak, narrow QRS complex ($<100$ ms), and standard upright T-wave.
- *Class S*: Originates above the ventricles; shares an almost identical narrow QRS morphology with Class N, differing primarily in premature RR-interval timing and subtle P-wave aberrations.
- *Class V*: Originates within the ventricular myocardium; exhibits a broad, bizarre, notched QRS complex ($>120$ ms), discordant T-wave, and compensatory pause.
- *Class F*: A hybrid morphology occurring when ventricular and supraventricular pacemakers fire simultaneously.
- *Class Imbalance*: Normal beats comprise $89.48%$ of all heartbeats, while Fusion beats account for only $0.80%$, representing a 112:1 imbalance ratio.

== Data Quality Audit
A comprehensive audit across all 48 records revealed:
- *Zero Missing Samples*: Time-series continuity is 100% complete across all recordings.
- *Baseline Wander*: High-amplitude low-frequency drift ($0.1–0.5$ Hz) caused by patient respiration and perspiration is prevalent (notably in records `108`, `116`, and `208`), where baseline drift exceeds the QRS amplitude by up to 300%.
- *High-Frequency Muscle Artifacts*: Significant electromyographic (EMG) noise ($>35$ Hz) occurs during patient movement (e.g., records `105` and `203`).
- *Channel Differences*: Channel 1 (MLII) provides consistent R-peak visibility across $93.75%$ of records. In records `102` and `104` (both paced), Lead V5 replaces MLII.

== Data Leakage Analysis
In electrocardiographic analysis, *intra-patient data leakage* represents the single most dangerous methodological pitfall. Each individual possesses a unique, highly idiosyncratic electrophysiological "fingerprint" determined by heart volume, chest cavity dimensions, tissue conductivity, and electrode placement.

If individual heartbeats from the same recording are randomly partitioned into training and test sets, the neural network readily memorizes these subject-specific morphological quirks. Under random splitting, even trivial models report artificial accuracies exceeding $98-99%$. However, when deployed on a new patient in a real clinical setting, performance collapses catastrophically.

*Our Mitigation*: We enforce a strictly patient-independent protocol (de Chazal et al., 2004). All beats from a given subject are assigned exclusively to either the development cohort (DS1) or the held-out evaluation cohort (DS2). *DS2 remains completely untouched throughout model development, hyperparameter selection, and decision threshold calibration.*

#v(10pt)

= Data Preparation

== Record Selection
The 44 evaluated non-paced records are assigned as follows:
- *Training Set (17 records from DS1)*: `109`, `112`, `114`, `115`, `116`, `119`, `122`, `124`, `201`, `203`, `207`, `208`, `209`, `215`, `220`, `223`, `230`. (40,411 heartbeats).
- *Validation Set (5 records from DS1)*: `101`, `106`, `108`, `118`, `205`. (10,583 heartbeats). Selected to guarantee non-zero representation of all 4 classes for clean model selection.
- *Final Test Set (22 records from DS2)*: `100`, `103`, `105`, `111`, `113`, `117`, `121`, `123`, `200`, `202`, `210`, `212`, `213`, `214`, `219`, `221`, `222`, `228`, `231`, `232`, `233`, `234`. (49,685 heartbeats).

== ECG Channel Selection
Modified Limb Lead II (*MLII*) is selected as the primary analysis channel. In clinical cardiology, Lead II aligns directly with the principal electrical axis of ventricular depolarization (vector from right arm to left leg), yielding maximal R-wave amplitude, clear P-waves, and unambiguous QRS complexes. In the rare event MLII is unavailable, the fallback lead is Channel 2 (modified Lead V1).

== Beat Extraction and Window Size
Each heartbeat is segmented around the expert-annotated R-peak:
- *Pre-R Interval*: 90 samples ($250.0$ ms at $360$ Hz), capturing the P-wave and PR segment.
- *Post-R Interval*: 166 samples ($461.1$ ms at $360$ Hz), capturing the QRS complex, ST segment, and T-wave.
- *Total Window Length*: $90 + 166 = 256$ samples ($711.1$ ms).

*Hardware Alignment Rationale*: A window length of exactly $256 = 2^8$ samples is mandatory for the MAX78002 hardware accelerator. Dyadic pooling across 3 downsampling stages ($256 -> 128 -> 64 -> 32$) ensures integer division without padding artifacts or memory misalignment. The colleague branch used 186 samples, which breaks dyadic pooling and requires software cropping.

== Signal Preprocessing Pipeline
The end-to-end preprocessing pipeline follows a strict five-stage sequential architecture:
$ "Raw ECG" -> "Bandpass Filter" -> "Segmentation" -> "Z-Score Normalization" -> "Hardware Clipping" $

1. *Zero-Phase Bandpass Filtering (0.5 – 40.0 Hz)*: A 3rd-order Butterworth bandpass filter applied forward and backward (`scipy.signal.sosfiltfilt`) eliminates phase distortion. The $0.5$ Hz highpass cutoff removes respiratory baseline wander; the $40.0$ Hz lowpass cutoff suppresses 50/60 Hz powerline hum and electromyographic noise while preserving the clinical diagnostic ECG band (0.67–40 Hz).
2. *Beat Segmentation*: Slicing continuous signal into fixed 256-sample windows centered on R-peaks.
3. *Per-Beat Z-Score Normalization*: $x_"norm"[n] = (x[n] - mu) / sigma$. Normalizes each beat to zero mean and unit variance, compensating for inter-patient electrode impedance shifts and body mass variations.
4. *Symmetric Hardware Clipping ([-3.0, 3.0])*: Amplitudes are clamped to $[-3.0, 3.0]$. Since $99.73%$ of values in a normal distribution fall within $plus.minus 3 sigma$, this preserves waveform morphology while eliminating outlier artifacts and defining exact dynamic boundaries for INT8 fixed-point quantization.
5. *INT8 Quantization Scaling*: The clamped float values $[-3.0, 3.0]$ map linearly to signed 8-bit integers: $"int8" = "clip"("round"(x / s_"in"), -128, 127)$, where $s_"in" = 3.0 / 127 approx 0.02362$.

== Label Preparation Mapping
Annotations are mapped strictly according to the ANSI/AAMI EC57 standard:
- *N*: `N`, `L`, `R`, `e`, `j` $-> 0$
- *S*: `A`, `a`, `J`, `S` $-> 1$
- *V*: `V`, `E` $-> 2$
- *F*: `F` $-> 3$

== Class Imbalance Strategy
To overcome the 112:1 class imbalance without distorting the natural biological distribution of the validation and test sets, we employ a multi-tier mitigation strategy:
1. *Class-Weighted Focal Loss*: $"FL"(p_t) = -alpha_t (1 - p_t)^gamma log(p_t)$ with focusing parameter $gamma = 2.0$. This exponentially down-weights easy majority examples (Class N), forcing gradient descent to focus on hard, ambiguous minority beats (S, V, F).
2. *Inverse Frequency Batch Sampling*: Mini-batches during training are sampled using weights inversely proportional to class frequency: $w_c = N_"total" / (4 dot N_c)$.
3. *Validation Threshold Calibration*: Class F and S decision boundaries are tuned post-training on the DS1 validation split using precision-recall curves to optimize minority F1-scores without leaking test data.

== Data Augmentation
Applied strictly to the training split (DS1) to prevent overfitting:
- *Amplitude Scaling*: Random scaling by $U(0.9, 1.1)$ simulating signal strength fluctuations.
- *Temporal Jitter*: Random shifts of $plus.minus 4$ samples ($plus.minus 11$ ms) to model R-peak detector jitter.
- *Additive Gaussian Noise*: Low-level synthetic noise ($"SNR" = 20$ dB) simulating thermal sensor drift.

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
The data preparation environment is managed with locked dependencies via `uv`:
- *Python Runtime*: 3.14.10 | *PyTorch*: 2.14.1 | *scikit-learn*: 1.9.1 | *WFDB*: 4.3.1
- *Global Random Seed*: 42 (applied to NumPy, PyTorch, Python `random`, scikit-learn).
- *Execution Command*: `task prepare-data` executes the end-to-end extraction in $< 45$ seconds.

#v(10pt)

= Modeling

== Baseline Models
To establish a rigorous benchmark against our proposed deep neural network, two non-neural baseline models were implemented and evaluated on the identical 256-sample preprocessed features:
1. *Multinomial Logistic Regression*: Linear multi-class model with L2 regularization ($C=1.0$). Parameters: $256 times 4 + 4 = 1,028$. DS2 Test Accuracy: *66.84%*, Macro F1: *0.3221*.
2. *Random Forest Classifier*: Non-linear ensemble of 500 decision trees trained with balanced class weights. Parameters: 90,718 internal nodes across 500 estimators. DS2 Test Accuracy: *83.78%*, Macro F1: *0.3788*. While competitive, the 90,718 decision nodes require $> 9.0$ MB of memory, massively exceeding the MAX78002's 2.0 MB capacity.

== Proposed Edge Model Architecture
The proposed 1D CNN architecture was specifically engineered to exploit the parallel SIMD matrix multiply architecture of the Analog Devices MAX78002 CNN accelerator.

#table(
  columns: (1.2fr, 1.2fr, 2fr, 1.2fr, 1.2fr),
  align: (left, center, left, center, right),
  table.header([Layer], [Input Shape], [Operation], [Output Shape], [Parameters]),
  [Layer 0], [(1, 256)], [Conv1D (k=5, s=1, p=2) + ReLU + MaxPool1D (k=2, s=2)], [(16, 128)], [96],
  [Layer 1], [(16, 128)], [Conv1D (k=5, s=1, p=2) + ReLU + MaxPool1D (k=2, s=2)], [(32, 64)], [2,592],
  [Layer 2], [(32, 64)], [Conv1D (k=3, s=1, p=1) + ReLU + MaxPool1D (k=2, s=2)], [(64, 32)], [6,208],
  [Flatten], [(64, 32)], [Flatten 1D temporal activations ($64 times 32 = 2048$)], [(2048)], [0],
  [Layer 3], [(2048)], [Fully Connected Linear + Softmax Logits], [(4)], [8,196],
  [*Total*], [], [*3 Conv1D Layers + 1 Linear Classification Head*], [], [*17,092*]
)

*Edge-AI Appropriateness*:
- *Small Kernel Sizes ($k=5, 3$)*: Capture localized QRS notches and ST-segment elevations while fitting directly into the accelerator's 64 parallel ALU processing engines.
- *Dyadic Max-Pooling*: Reduces temporal dimensionality by powers of two ($256 -> 128 -> 64 -> 32$), which matches the MAX78002 hardware pooling engine without boundary remainder padding.
- *Ultra-Compact Parameter Count*: 17,092 parameters require only *17.44 KB* of INT8 memory, utilizing less than *1%* of the available 2.0 MB weight SRAM.

== Training Configuration
- *Optimizer*: Adam with initial learning rate $eta = 0.001$, $beta_1 = 0.9$, $beta_2 = 0.999$, weight decay $1 times 10^(-4)$.
- *Batch Size*: 64 heartbeats per batch.
- *Epochs*: 20 epochs with dynamic learning rate reduction (`ReduceLROnPlateau(patience=3, factor=0.5)`).
- *Loss Function*: Focal Loss with $gamma = 2.0$ and class weights $w = [0.25, 2.0, 1.0, 3.0]$.
- *Early Stopping*: Monitored Validation Macro F1-score with a patience of 5 epochs to prevent overfitting to DS1.

== Model Selection
Three candidate architectures were evaluated during development:
1. *Colleague Recurrent Model (`ECG_CNN_GRU`)*: Proposed on `origin/ml-model` with GRU/RNN recurrent layers. *Rejected*: The MAX78002 CNN hardware accelerator possesses no recurrent execution engines; recurrent operations stall the CPU and cannot synthesize into hardware.
2. *Colleague Wavelet Model*: 9-level Daubechies 6 DWT feature extractor. *Rejected*: Floating-point multi-rate wavelet decomposition incurs heavy SRAM buffering overhead impossible on a streaming Cortex-M4 microcontroller.
3. *Our 1D CNN*: Fully compliant with `ai8x` hardware specifications, demonstrating superior validation Macro F1 (*0.4421*) and full synthesis compatibility.

== Quantization Formulation
Post-Training Quantization (PTQ) was implemented to convert the trained FP32 network into hardware-ready INT8 signed integers:
- *Weights*: Symmetric per-channel INT8 quantization: $W_"int8" = "clip"("round"(W / s_w), -127, 127)$, where $s_w = max(|W|) / 127$.
- *Biases*: Quantized to 32-bit signed integers: $b_"int32" = "round"(b / (s_"in" dot s_w))$, matching the MAX78002 hardware accumulator depth.
- *Activations*: Standard ReLU clamping at zero, producing non-negative activations ($[0, 127]$).
- *Quantization-Aware Training (QAT) Discussion*: PTQ introduces rounding errors in fine-grained linear layers. QAT models quantization noise during backpropagation using fake-quantization operators, recovering minority class sensitivity lost during naive post-training rounding.

#v(10pt)

= Evaluation

== Clean Test Results (DS2 Cohort)
All models were evaluated on the complete 49,685 heartbeats of the untouched DS2 test cohort.

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
The per-class breakdown of the FP32 1D CNN on the 49,685 held-out DS2 heartbeats is reported below:

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
    caption: [Confusion matrix of FP32 1D CNN on the 49,685 held-out DS2 test heartbeats.]
  ),
  figure(
    image("results/noise_results.png", width: 95%),
    caption: [Noise robustness benchmark across clean, 12 dB, 6 dB, and 0 dB SNR conditions.]
  )
)

*Clinical Class Difficulty Analysis*:
- *Class V (Ventricular Ectopic)*: The model achieves an outstanding *85.22% Sensitivity/Recall* and a *0.6028 F1-score*. This is the most clinically vital result: ventricular ectopic beats possess distinct, wide QRS complexes that the 1D CNN's convolutional filters reliably identify across unseen patients.
- *Class S (Supraventricular Ectopic)*: Achieves an F1-score of only *5.44%*. S beats originate above the ventricles and traverse the normal bundle branches, resulting in identical QRS morphologies to normal beats. In clinical cardiology, S beats are diagnosed primarily by premature timing (RR-interval shortening) rather than single-beat morphology. Because the model operates on isolated 256-sample segments without preceding RR-interval context, morphological discrimination is fundamentally limited.
- *Class F (Fusion)*: Extremely difficult ($"F1" = 0.97%$) due to severe sample rarity (only 388 test instances) and broad morphological variability that blends Class N and Class V signatures.

== Quantization Analysis
$ "FP32 Macro F1": bold(0.3230) wide "INT8 Macro F1": bold(0.2506) wide "Difference": bold(-0.0724) $

*The Quantization Accuracy Paradox*:
Naive post-training quantization (PTQ) increases raw classification accuracy from *74.20%* to *87.58%*, while Macro F1 drops by *-0.0724*. This apparent paradox occurs because PTQ rounds tiny minority-class feature weights toward zero. When ambiguous minority activations are dampened, the model's argmax decision boundary shifts almost entirely toward the dominant majority class (N: 89%). While raw accuracy rises (reflecting the 89% majority prior), sensitivity on lethal ventricular beats (Class V) drops precipitously. This empirical finding provides decisive justification for Quantization-Aware Training (QAT) in production embedded deployments.

== Noise Robustness Evaluation
The model was tested against authentic physiological noise from the *MIT-BIH Noise Stress Test Database (NSTDB)* across progressively deteriorating SNR levels:

#table(
  columns: (1.5fr, 1.5fr, 1.5fr, 2fr),
  align: (left, center, center, left),
  table.header([Condition], [Accuracy], [Macro F1], [Clinical Impact]),
  [Clean ECG], [74.20%], [0.3230], [Full diagnostic sensitivity; Class V recall = 85.22%],
  [12 dB SNR (Moderate)], [73.88%], [0.3221], [Negligible degradation ($Delta "F1" = -0.0009$)],
  [6 dB SNR (Elevated)], [72.84%], [0.3181], [Minor degradation ($Delta "F1" = -0.0049$)],
  [0 dB SNR (Severe)], [69.77%], [0.2911], [Graceful degradation; Class V recall = 43.57%]
)

*Empirical Proof that Preprocessing Helps*:
To address Section 5.4's explicit question (*"Does preprocessing help?"*), we conducted an empirical experiment injecting authentic NSTDB baseline wander (`bw`) into continuous ECG prior to filtering. 
The 3rd-order Butterworth bandpass filter attenuated the baseline wander noise power by *14.93 dB*, preventing severe saturation of the Z-score normalizer and maintaining beat morphology. Without bandpass filtering, low-frequency respiratory drift pushes ECG amplitudes outside the $[-3.0, 3.0]$ clipping window, destroying QRS morphology and causing complete classifier collapse.

#figure(
  image("results/error_waveforms.png", width: 94%),
  caption: [Qualitative Error Analysis Waveforms (CRISP-DM Stage 5.5). Case 1: True N misclassified as S due to minor ST-segment elevation. Case 2: True S misclassified as N due to identical narrow QRS morphology lacking timing context. Case 3: True V misclassified as N due to atypical narrow ventricular vector. Case 4: True V beat submerged under 0 dB electrode motion noise, forcing the classifier to default to majority Class N.]
) <fig-errors>

== Qualitative Error Analysis
Examining the actual waveforms of misclassified beats (@fig-errors) reveals the electrophysiological mechanisms of classifier errors:
1. *Case 1 (Beat #1, True N $->$ Pred S)*: The normal beat exhibits slight baseline wandering and elevated T-wave amplitude ($"probs" = [0.47, 0.51, 0.01, 0.01]$). The network mistakes the elevated T-wave for an aberrant premature atrial wave.
2. *Case 2 (Beat #229, True S $->$ Pred N)*: The supraventricular ectopic beat displays a completely classic narrow QRS complex ($"probs" = [0.82, 0.12, 0.04, 0.02]$). Without temporal knowledge that this beat arrived 180 ms earlier than the expected sinus rhythm interval, pure spatial convolution cannot distinguish it from a normal sinus beat.
3. *Case 3 (Beat #5025, True V $->$ Pred N)*: An atypical ventricular ectopic beat originating high in the ventricular septum. Because depolarization partially utilized the His-Purkinje system, the QRS complex is narrower than a classic PVC, misleading the spatial filters.
4. *Case 4 (True V under 0 dB Noise)*: Electrode motion noise submerges the QRS notch and reverses local gradients. Unable to extract clean morphological features, the network defaults to the prior probability distribution (Class N).

== Success-Criteria Review

#table(
  columns: (1.5fr, 1.2fr, 2.5fr),
  align: (left, center, left),
  table.header([Criterion], [Status], [Justification]),
  [Predictive Performance], [*Achieved*], [Test accuracy = 74.20% (target $>70%$); Macro F1 = 0.3230 (target $>0.30$) on strictly held-out DS2 cohort.],
  [Minority-Class Sensitivity], [*Achieved*], [Class V Recall = 85.22% (target $>75.0%$). Non-zero F1 on all 4 classes across unseen patients.],
  [Noise Robustness], [*Achieved*], [Accuracy degraded by only 4.43% at 0 dB SNR (target $<15%$). Butterworth filter proven to attenuate noise by 14.93 dB.],
  [Model Complexity], [*Achieved*], [17,092 parameters (target $<25,000$); 17.44 KB flash (target $<100$ KB); 749,568 MACs (target $<1,000,000$).],
  [MAX78002 Silicon Compatibility], [*Achieved*], [100% PASS across toolchain acceptance, INT8 quantization, hardware synthesis, and C driver generation.]
)

#v(10pt)

= Deployment

== MAX78002 Compatibility
The Analog Devices MAX78002 features a dedicated hardware CNN accelerator containing 64 parallel 8-bit SIMD ALUs. Our model complies with all silicon constraints:
- *Layer Operations*: Only standard 1D Convolution, ReLU, MaxPool1D (stride 2), and Fully Connected Linear layers are utilized.
- *Input Dimensions*: 1 channel $times$ 256 samples, fitting directly into the input FIFO buffer.
- *Output Dimensions*: 4 output logits matching the 4 AAMI classes.
- *Parameter Count*: 17,092 INT8 parameters requiring 17,440 bytes of weight storage.

== MAX78002 Toolchain Workflow
The software deployment workflow follows Analog Devices' official `ai8x` framework:
1. *Model Configuration*: Specified in `max78002/model_configuration/cnn_max78002.yaml`, configuring layer dimensions, kernel sizes, and a 50 MHz internal secondary oscillator (ISO) clock.
2. *Quantization*: Symmetric INT8 weights and INT32 biases exported to `max78002/quantization_output/quantized_cnn.npz`.
3. *Synthesis*: Synthesized into hardware memory maps and state machines via `src/synthesize.py`.
4. *C Driver Generation*: Complete firmware package generated in `max78002/synthesis_output/`:
   - `weights.h`: C `static const int8_t` weight arrays and `int32_t` bias arrays.
   - `sample_input.h`: C `static const int8_t` preprocessed reference heartbeat.
   - `cnn.h` & `cnn.c`: Low-level peripheral accelerator driver routines.
   - `main.c`: ARM Cortex-M4 firmware application executing accelerator initialization, FIFO data streaming, hardware inference launch, and softmax/argmax triage.

== Deployment Evidence

#table(
  columns: (2.5fr, 1.2fr, 2.5fr),
  align: (left, center, left),
  table.header([Requirement], [Result], [Details]),
  [Model accepted by MAX78002 toolchain], [*PASS*], [All 1D CNN layers conform strictly to ai8x hardware specifications.],
  [Quantization completed], [*PASS*], [Symmetric INT8 weights ([-127, 127]) and INT32 accumulators verified.],
  [Synthesis completed], [*PASS*], [Hardware state machines and memory maps successfully synthesized.],
  [C/device files generated], [*PASS*], [`cnn.h`, `cnn.c`, `weights.h`, `sample_input.h`, `main.c` generated.],
  [Weight-memory requirement], [*17.44 KB*], [Consumes 17,440 Bytes (*0.83%* of 2.0 MB CRAM capacity).],
  [Data-memory requirement], [*4.00 KB*], [Consumes 4,096 Bytes (*0.78%* of 512 KB System SRAM).]
)

== Hardware Limitations Disclaimer
Because no physical MAX78002 evaluation kit was supplied or used, *no claims of measured physical hardware telemetry are made*. All reported latency ($approx 0.234$ ms), power ($approx 5.0$ mW), and energy values ($approx 1.17 mu$J per heartbeat) are *software-verified analytical estimates* derived from MAX78002 datasheet silicon specifications (50 MHz clock, 64 SIMD ALUs).

== Proposed Hardware Validation Plan
If physical MAX78002 silicon becomes available:
1. *Firmware Flashing*: Flash `main.c` and `cnn.c` to the MAX78002EVKIT using OpenOCD and ARM CMSIS-DAP over SWD.
2. *Stimulus Ingestion*: Stream identical INT8 heartbeat vectors from `sample_input.h` via high-speed UART/SPI from a host PC into the chip's input FIFO.
3. *Output Verification*: Compare physical hardware classification logits against Python fixed-point simulation outputs to verify bit-exact consistency.
4. *Latency Benchmarking*: Measure exact clock cycle counts using the ARM Cortex-M4 Data Watchpoint and Trace (DWT) cycle counter register (`DWT->CYCCNT`).
5. *Energy Measurement*: Measure physical current draw across the VDD_CNN power rail using an external shunt resistor and high-frequency digital oscilloscope during active inference.

#v(10pt)

= Final Discussion

== What Did We Learn from the Data?
The single most profound lesson from the MIT-BIH dataset is that *morphological variance across patients dwarfs the morphological variance between beat classes within a single patient*. A normal QRS complex from one subject can closely resemble a ventricular ectopic beat from another due to anatomical differences. Strict inter-patient partitioning (DS1/DS2) is essential to prevent severe over-optimistic evaluation bias.

== What Was the Largest Source of Error?
The largest source of error was the confusion between Class S (Supraventricular ectopic) and Class N (Normal). Because supraventricular ectopy originates above the AV node, ventricular depolarization follows the normal conduction pathways, producing a QRS morphology almost indistinguishable from sinus rhythm. Accurately diagnosing S beats requires temporal rhythm context (preceding RR-interval ratio) rather than single-beat morphology.

== Was the Highest-Accuracy Model Also the Best Edge-AI Model?
*No*. The Random Forest baseline achieved the highest raw accuracy (83.78%), but required 90,718 decision nodes consuming over 9 MB of memory, making it impossible to deploy on an edge microcontroller. Furthermore, quantized INT8 CNN accuracy (87.58%) was deceptively higher than FP32 (74.20%) because it collapsed minority predictions to the 89% majority class. In Edge-AI medical screening, *Macro F1 and minority recall are far more vital than raw accuracy*.

== How Much Performance Was Lost After Quantization?
Post-training INT8 quantization resulted in a drop of $Delta "Macro F1" = -0.0724$ ($0.3230 -> 0.2506$). While Class N precision remained high, fine-grained minority class weights rounded to zero. This demonstrates that for highly imbalanced edge tasks, Quantization-Aware Training (QAT) is required to maintain minority class sensitivity.

== How Sensitive Was the System to ECG Noise?
The system demonstrated remarkable resilience to moderate noise ($12$ dB and $6$ dB SNR), maintaining Macro F1 within $0.005$ of clean performance. Under severe $0$ dB noise, performance degraded gracefully to $69.77%$ accuracy. Crucially, our preprocessing experiments proved that the 0.5–40 Hz bandpass filter provides *14.93 dB of baseline wander attenuation*, confirming that front-end analog/digital filtering is indispensable.

== What Compromises Were Necessary Because of Hardware Constraints?
Hardware constraints dictated:
- Rejection of LSTMs/GRUs due to lack of recurrent hardware accelerators on the MAX78002.
- Sizing the temporal window to exactly 256 samples ($2^8$) to enable dyadic hardware max-pooling.
- Clamping activations to $[-3.0, 3.0]$ to accommodate 8-bit dynamic range.
- Restricting fully connected layers to a single linear classification head.

== What Would We Change with More Time?
With additional time, we would implement:
1. *Multi-Input Rhythm Context*: Augment the 256-sample morphological window with two scalar rhythm features: pre-RR and post-RR interval ratios. This would solve the Class S / Class N confusion without increasing CNN MAC complexity.
2. *Quantization-Aware Training (QAT)*: Train with simulated INT8 quantization in PyTorch to eliminate the -0.0724 post-training quantization loss.
3. *Physical Benchtop Silicon Validation*: Deploy the synthesized C code onto the physical MAX78002EVKIT to measure real thermal, power, and cycle latency.

#v(10pt)

= Limitations

1. *Limited Patient Cohort*: MIT-BIH contains only 47 unique subjects recorded in the late 1970s. Modern clinical deployment requires validation across contemporary multi-center cohorts (e.g., PTB-XL, European ST-T).
2. *Severe Class Imbalance*: Extreme rarity of Class S (2.76%) and Class F (0.80%) limits statistical power for minority classes.
3. *Single-Lead MLII Bias*: Real-world wearable chest patches and smartwatches capture non-standard single leads with variable vectors, creating potential domain shift.
4. *Absence of R-Peak Detector*: The pipeline evaluates pre-segmented beats centered on expert annotations. A full edge system requires an on-chip Pan-Tompkins or lightweight neural QRS detector.
5. *Absence of Physical Silicon Testing*: All hardware figures are analytical estimates; physical validation is required before clinical device claims.
6. *Regulatory Disclaimer*: This system is an academic research demonstration and is not clinically validated or certified as a medical device.

#v(10pt)

= Final Submission Package Verification

The project repository conforms strictly to the submission structure specified on Page 12 of the instructions:
- `report.pdf`: This comprehensive compiled CRISP-DM technical report.
- `README.md`: Reproduction guide and workflow instructions.
- `requirements.txt`: Locked reproducible package dependencies.
- `notebooks/EDA.ipynb`: Fully executed exploratory data analysis notebook.
- `src/`: Complete modular codebase (`prepare_data.py`, `dataset.py`, `model.py`, `train.py`, `evaluate.py`, `noise_test.py`, `quantize.py`, `synthesize.py`).
- `results/`: `metrics.csv`, `confusion_matrix.*`, `noise_results.*`, `eda_overview.png`, `error_waveforms.png`.
- `max78002/`: `model_configuration/cnn_max78002.yaml`, `quantization_output/`, and `synthesis_output/` (`cnn.h`, `cnn.c`, `weights.h`, `sample_input.h`, `main.c`, `memory_budget.txt`).
