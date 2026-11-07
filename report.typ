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
        align(right)[#text(size: 8.5pt, fill: rgb("#718096"), font: "Segoe UI")[MAX78002 Deployment]]
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
    CRISP-DM Technical Report: Edge Deployment and Model Verification
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
  
  Structured under the CRISP-DM methodology, this technical report evaluates a compact, single-lead ECG heartbeat classification system targeted to the Analog Devices MAX78002 Edge-AI microcontroller. Designed as the ECG-only processing stage for wearable cardiac monitors, the classifier maps individual heartbeats into four ANSI/AAMI EC57 categories: normal (N), supraventricular ectopic (S), ventricular ectopic (V), and fusion (F). To prevent intra-patient data leakage, the 44 non-paced MIT-BIH Arrhythmia Database recordings (100,679 annotated beats) are split strictly by patient: 22 records form the DS1 development set and 22 records form the held-out DS2 test set.
  
  The classifier uses a three-layer 1D convolutional neural network with 17,092 parameters, sized to fit the MAX78002 hardware accelerator without exceeding on-chip memory limits. On the 49,685 beats in the untouched DS2 test set, the FP32 model achieves 74.20% accuracy, a macro F1 score of 0.3230, and 87.58% recall on ventricular ectopic beats (Class V). Software-verified toolchain synthesis and post-training symmetric INT8 quantization confirm full hardware compatibility, consuming 17.44 KB of weight memory (0.83% of the 2.0 MB CRAM) and 4.0 KB of activation SRAM (0.78% of 512 KB). Post-training quantization lowers macro F1 to 0.2506 due to integer truncation of minority-class weights, indicating the need for quantization-aware training in edge production. At 50 MHz, datasheet-derived execution estimates indicate 0.234 ms latency per beat (749,568 MACs) and 1.17 $mu$J energy draw. Stress testing with MIT-BIH Noise Stress recordings shows that a front-end 0.5 to 40 Hz Butterworth bandpass filter attenuates baseline wander by 18.94 dB and cuts overall noise power by 2.89 dB MSE, preventing baseline drift and motion artifacts from saturating the software input normalizer and $[-3.0, 3.0]$ fixed-point dynamic range.
]

#v(6pt)

= Business / application understanding

== Application scenario
Cardiac arrhythmias range from acute emergencies like ventricular fibrillation to recurrent ectopic beats that indicate underlying disease or elevated stroke risk. Conventional Holter monitors store multi-day ECG recordings to flash memory for retrospective physician review, leaving clinicians unaware of acute rhythm disturbances until days later.

Moving inference onto wearable edge microcontrollers allows real-time beat classification directly on ambulatory patches. By processing signals locally, the device avoids continuous Bluetooth or cellular transmission, cutting the radio duty cycle by roughly 90% and extending coin-cell battery life to several weeks. On-device analysis also eliminates cloud latency for urgent ventricular alerts, keeps patient waveforms private, and operates reliably in shielded hospital wards without cellular coverage. This workflow supports ambulatory patients, post-infarction convalescents, elderly individuals at elevated arrhythmia risk, and cardiologists triaging high-volume Holter recordings.

== Data-mining objective
The machine learning objective is to classify single-lead 1D ECG heartbeats into discrete diagnostic categories within the storage and arithmetic limits of an edge microcontroller:

#table(
  columns: (1fr, 3.5fr),
  align: (left, left),
  table.header([Item], [Description]),
  [Input], [A single-lead 1D time-series window of 256 samples (711.1 ms at 360 Hz) centered on the detected R-peak: 90 samples (250.0 ms) of pre-peak context (P-wave and PR segment) and 166 samples (461.1 ms) of post-peak context (QRS complex, ST segment, and T-wave).],
  [Output], [Discrete 4-class categorical prediction conforming to ANSI/AAMI EC57: Class N (Normal or bundle branch block), Class S (Supraventricular ectopic), Class V (Ventricular ectopic), and Class F (Fusion).],
  [Type of task], [Supervised multi-class time-series classification under severe class imbalance (~89% N, ~3% S, ~7% V, ~0.8% F) and INT8 quantization constraints.]
)

== Project objectives
Deploying the classifier to the MAX78002 establishes five engineering objectives:
- Sort heartbeats into the four standard AAMI groups (N, S, V, F).
- Generalize across unseen patients by evaluating strictly on an independent test cohort (DS2).
- Retain classification reliability under physical noise (electrode motion, muscle tremor, baseline wander) evaluated at 12 dB, 6 dB, and 0 dB SNR.
- Restrict model footprint to fewer than 25,000 parameters, under 100 KB total storage, and under 1,000,000 MAC operations per beat to execute within on-chip SRAM.
- Maintain compatibility with the Analog Devices MAX78002 CNN accelerator by using only hardware-supported operations that compile directly to embedded C code through the `ai8x-synthesis` toolchain.

== Success criteria
Because normal sinus beats account for roughly 89% of ambulatory ECG data, overall accuracy alone cannot confirm clinical efficacy. Project success is evaluated against five measurable criteria:
- *Predictive performance*: Macro-averaged F1 on the held-out DS2 test set must exceed 0.30, with overall accuracy above 70.0%.
- *Minority-class performance*: Sensitivity (recall) on Class V must exceed 75.0% to reliably detect dangerous ventricular arrhythmias, while Classes S and F must achieve non-zero F1 scores on unseen patients.
- *Noise robustness*: Under 0 dB physical noise from the MIT-BIH Noise Stress Test Database, accuracy loss must not exceed 15% relative to clean ECG inputs, supported by front-end bandpass filtering that attenuates baseline wander and electromyographic noise.
- *Model complexity*: Total parameter count must remain below 25,000, storage footprint under 100 KB, and compute budget under 1,000,000 MAC operations per beat.
- *MAX78002 compatibility*: The network must pass every stage of the MAX78002 toolchain (model validation, INT8 quantization, synthesis, and C driver generation), requiring under 50 KB of weight memory (below 2.5% of CRAM) and under 10 KB of activation SRAM (below 2.0% of system SRAM).

== Constraints
The implementation is bounded by seven hardware and operational factors:
1. *Limited Edge-AI memory*: The MAX78002 provides 2.0 MB of CNN weight SRAM (CRAM) and 512 KB of system SRAM; the board includes no external RAM.
2. *Fixed-point/INT8 computation*: The hardware accelerator computes strictly with signed 8-bit integers ([-128, 127]) for weights and 32-bit accumulators, requiring software emulation on the Cortex-M4 CPU for any floating-point operation.
3. *Class imbalance*: Normal sinus beats outnumber ectopic beats by up to 100 to 1, causing unweighted cross-entropy models to collapse into predicting only the majority class.
4. *Limited number of subjects*: The MIT-BIH database contains recordings from only 47 subjects, making cross-patient generalization difficult due to variations in electrode placement, chest anatomy, and cardiac history.
5. *Physical noise*: Ambulatory recordings carry severe physical artifacts, including electrode motion (`em`), baseline wander (`bw`), and muscle tremor (`ma`).
6. *Computational resources*: The Cortex-M4 microcontroller runs at 50 MHz without hardware floating-point acceleration for neural network layers, requiring convolutional operations to execute on the accelerator's 64 parallel SIMD ALUs within a budget of 1,000,000 MAC operations per beat.
7. *Absence of physical MAX78002 hardware*: Without physical silicon, reported deployment metrics (0.234 ms latency, 5.0 mW power, 1.17 $mu$J energy) are software-verified estimates calculated from `ai8x-synthesis` memory maps, register configurations, and datasheet electrical specifications.

== Risks and challenges

#table(
  columns: (1.2fr, 2fr, 2fr),
  align: (left, left, left),
  table.header([Risk], [Why is it important?], [Proposed mitigation]),
  [Intra-patient data leakage], [Randomly assigning beats from the same recording into training and test sets lets the network memorize patient-specific baseline morphologies. This produces deceptive test accuracies exceeding 98% that collapse when evaluated on new patients.], [Use the inter-patient partitioning protocol from de Chazal et al. (2004): 22 records in DS1 (training and validation) and 22 completely separate records in DS2 (testing), with zero patient overlap.],
  [Minority class neglect], [Standard cross-entropy loss biases optimization toward Class N. A naive model can reach ~89% nominal accuracy while missing nearly all ventricular and supraventricular events.], [Apply class-weighted loss with inverse-frequency weights ($w_c = N_"total" / (4 dot N_c)$) alongside balanced batch sampling on the training set to penalize minority misclassifications during backpropagation.],
  [Hardware accelerator incompatibility], [Recurrent networks (LSTMs, GRUs) and complex transforms (wavelets) cannot execute on the MAX78002 CNN hardware accelerator.], [Restrict the model architecture strictly to 1D convolution, ReLU, stride-2 MaxPool1D, and linear layers, verifying synthesis acceptance against the `ai8x-synthesis` YAML specification.],
  [Quantization precision loss], [Converting 32-bit floating-point weights to 8-bit integers can truncate small minority-class weights to zero, degrading recall on rare beat types.], [Calibrate per-layer dynamic ranges, apply symmetric per-channel INT8 quantization, and clip input signal amplitudes to $[-3.0, 3.0]sigma$.]
)

#v(6pt)

= Data understanding

== Dataset description
The MIT-BIH Arrhythmia Database serves as the standard reference benchmark for automated arrhythmia analysis. The dataset comprises 48 two-channel ambulatory ECG recordings of approximately 30 minutes each (1,800 seconds per record, totaling roughly 24 hours), collected from 47 distinct subjects (records `201` and `202` come from the same individual). Analog signals were digitized at a sampling frequency of 360 Hz per channel with 11-bit resolution over a $plus.minus 10$ mV range (200 ADC units/mV). In 45 of the 48 recordings, Channel 1 is Modified Limb Lead II (MLII), which captures ventricular depolarization along the primary cardiac axis; Channel 2 is usually modified lead V1 (or occasionally V2, V4, or V5). Two or more cardiologists independently annotated over 110,000 heartbeats, marking fiducial R-peak locations and beat morphologies.

== Target classes and documented exclusions
Under the ANSI/AAMI EC57:1998 standard, 15 original MIT-BIH beat types map into four clinical classes. Non-beat annotations and paced recordings are excluded.

#table(
  columns: (0.8fr, 2.5fr, 1fr),
  align: (left, left, right),
  table.header([Final Class], [Original Annotations Included], [Number of Beats]),
  [Class N], [Normal (`N`), Left bundle branch block (`L`), Right bundle branch block (`R`), Atrial escape (`e`), Nodal escape (`j`)], [90,088 (89.48%)],
  [Class S], [Atrial premature (`A`), Aberrated atrial premature (`a`), Nodal premature (`J`), Supraventricular premature (`S`)], [2,781 (2.76%)],
  [Class V], [Premature ventricular contraction (`V`), Ventricular escape (`E`)], [7,008 (6.96%)],
  [Class F], [Fusion of ventricular and normal beat (`F`)], [802 (0.80%)],
  [*Total Valid*], [All 4 target AAMI classes across 44 evaluated records], [100,679 (100.0%)]
)

Four paced recordings (`102`, `104`, `107`, and `217`) are excluded because electronic pacemaker spikes distort the natural QRS complex. We also exclude non-beat annotations and unclassifiable markers: rhythm change tags (`+`: 1,173), signal quality markers (`~`: 573), ventricular flutter markers (`!`: 472), isolated noise spikes (`|`: 131), unclassifiable beats (`Q`: 15), non-conducted P-waves (`x`: 21), and 39 boundary beats.

#figure(
  image("results/eda_overview.png", width: 94%),
  caption: [Exploratory data analysis and MAX78002 preprocessing steps. (A) Morphology of the 4 AAMI classes aligned at the R-peak ($t=0$ ms, sample 90). (B) Beat counts by class and split on a log scale. (C) Amplitude distribution of normalized beats against the $[-3.0, 3.0]$ hardware clipping limits. (D) Dyadic pooling stages matching MAX78002 hardware.]
) <fig-eda>

== Exploratory data analysis
Exploratory analysis reveals how the four target classes differ in physiological origin and waveform morphology (@fig-eda):
- Class N (normal): Normal sinus beats exhibit a sharp R-peak, a narrow QRS complex below 100 ms, and an upright T-wave.
- Class S (supraventricular ectopic): Arising in the atria or AV junction, these complexes resemble normal beats in QRS width but arrive prematurely, often with distorted or inverted P-waves.
- Class V (ventricular ectopic): Impulses originating in the ventricles conduct slowly outside the specialized His-Purkinje network, producing a widened QRS complex (>120 ms), an inverted T-wave, and a post-ectopic compensatory pause.
- Class F (fusion): Hybrid complexes produced when sinus and ventricular pacemakers discharge simultaneously.

Inspection of the full exploratory pipeline in `notebooks/EDA.ipynb` confirms all minimum data characteristics:
1. Example raw ECG signals illustrate clean sinus rhythm alongside lead differences between MLII and precordial chest leads.
2. Example heartbeat waveforms (@fig-eda A) depict characteristic morphology across classes.
3. Class distributions (@fig-eda B) show severe imbalance: normal beats make up 89.48% of the 100,679 valid beats, while fusion beats account for only 0.80% (an imbalance ratio of roughly 112 to 1).
4. Subject and record distributions (@fig-eda B) partition across the development (DS1, 22 records) and test (DS2, 22 records) cohorts.
5. Signal amplitude distributions (@fig-eda C) center around zero after Z-score normalization, bounded by the $[-3.0, 3.0]sigma$ hardware clipping thresholds.
6. Examples of noisy ECG traces capture baseline wander, muscle tremor, and electrode motion.

== Data quality audit
An audit across all 48 recordings reveals specific data quality and signal integrity factors:
- Missing data: Verification confirmed complete signal continuity with no missing samples across any recording.
- Extreme amplitudes: Low-frequency baseline wander (0.1 to 0.5 Hz) from respiration and perspiration reaches up to 300% of the QRS height in records `108`, `116`, and `208`, requiring dynamic clamping to prevent fixed-point overflow.
- Corrupted segments and noisy regions: Burst electromyographic (EMG) noise (>35 Hz) from patient muscle contractions appears in records `105` and `203`, while sporadic electrode contact loss (`em`) occurs during physical movement.
- Class imbalance: Severe skew across the 100,679 beats (89.48% N vs. 0.80% F) would cause unweighted models to predict solely the majority class.
- Differences between recordings: Inter-subject anatomical variations, thoracic impedance, and electrode placement create substantial morphological diversity across the 47 patients.
- Differences between ECG channels: Channel 1 is MLII in 45 recordings (93.75%). Paced records `102` and `104` use Lead V5 on Channel 1 and V2 on Channel 2, while record `114` has V5 on Channel 1 and MLII on Channel 2.

== Data leakage analysis
Data leakage occurs when information from the evaluation split inadvertently influences model training. In ECG arrhythmia classification, randomly splitting individual heartbeats from the same subject into training and test sets can produce misleading results. Because thoracic anatomy, heart geometry, and electrode impedance vary from person to person, heartbeats from a single patient share a distinct morphological baseline. A model evaluated on randomly partitioned beats learns individual patient signatures rather than generalized arrhythmia patterns. While such models report nominal test accuracy above 98%, their performance collapses when deployed on unseen patients.

To prevent intra-patient data leakage, we adopt the inter-patient partitioning protocol of de Chazal et al. (2004). All beats from any given subject belong exclusively to either the development cohort (DS1, 22 records) or the evaluation cohort (DS2, 22 records). The DS2 test partition remained strictly held out without modification throughout model tuning and training.

#v(6pt)

= Data preparation

== Record selection
The 44 non-paced recordings partition into training, validation, and test splits following the inter-patient protocol of de Chazal et al. (2004):
- Training set (17 records from DS1): `109`, `112`, `114`, `115`, `116`, `119`, `122`, `124`, `201`, `203`, `207`, `208`, `209`, `215`, `220`, `223`, `230` (40,411 beats).
- Validation set (5 records from DS1): `101`, `106`, `108`, `118`, `205` (10,583 beats), selected to ensure all four target arrhythmia classes are represented for model checkpointing.
- Test set (22 records from DS2): `100`, `103`, `105`, `111`, `113`, `117`, `121`, `123`, `200`, `202`, `210`, `212`, `213`, `214`, `219`, `221`, `222`, `228`, `231`, `232`, `233`, `234` (49,685 beats).

Four paced recordings (`102`, `104`, `107`, `217`) are excluded from all splits because artificial pacemaker spikes distort intrinsic cardiac depolarization.

== ECG channel selection
Modified Limb Lead II (MLII) serves as the primary channel because ventricular depolarization projects strongly along this lead axis, generating high signal-to-noise R-peaks and distinct QRS morphologies. For record `114`, where lead configurations are reversed, the pipeline programmatically extracts MLII from Channel 2. In records where MLII is unavailable, modified Lead V1 substitutes as the input channel.

== Beat extraction
Reference R-peak fiducial locations are parsed directly from official PhysioNet annotation files (`.atr`) using the WFDB library (`wfdb.rdann`). Each heartbeat is segmented into a fixed-length window of 256 samples (711.1 ms at 360 Hz) centered on the annotated R-peak: 90 samples (250.0 ms) precede the peak, capturing the P-wave and PR interval, while 166 samples (461.1 ms) follow it to capture the QRS complex, ST segment, and T-wave.

The fixed 256-sample window satisfies the input requirements of the MAX78002 CNN hardware accelerator. Sizing the input to $256 = 2^8$ samples enables three successive stride-2 pooling stages ($256 -> 128 -> 64 -> 32$) through integer halving, eliminating boundary truncation or remainder padding in hardware execution registers.

== Signal preprocessing
The preprocessing pipeline transforms continuous analog-converted recordings into fixed-point heartbeat tensors through six sequential stages:
$ "Raw ECG" -> "Bandpass Filter" -> "Segmentation" -> "Z-Score Standardization" -> "Amplitude Clipping" -> "INT8 Scaling" -> "Model Input" $

1. *Bandpass filtering (0.5 to 40.0 Hz):* A 3rd-order Butterworth filter applied forward and backward (`scipy.signal.sosfiltfilt`) eliminates phase distortion. Highpass attenuation at 0.5 Hz removes respiratory and perspiration baseline wander, while lowpass attenuation at 40.0 Hz suppresses 50/60 Hz mains interference and electromyographic noise while preserving diagnostic ECG frequencies (0.67 to 40 Hz).
2. *Beat segmentation:* Slices continuous signal streams into fixed 256-sample segments centered on annotated R-peaks.
3. *Z-score standardization:* Each window is standardized to zero mean and unit variance ($x_"norm"[n] = (x[n] - mu) / sigma$), neutralizing inter-patient baseline offsets and amplitude variations caused by differences in skin impedance and chest geometry.
4. *Amplitude clipping:* Normalized values are clamped to $[-3.0, 3.0]$, eliminating outlier spikes beyond $3 sigma$ from motion artifacts and establishing strict dynamic bounds for fixed-point quantization.
5. *INT8 scaling:* Clamped float values map linearly to signed 8-bit integers via $"int8" = "clip"("round"(x / s_"in"), -128, 127)$, with scale factor $s_"in" = 3.0 / 127 approx 0.02362$, matching the MAX78002 hardware FIFO input format.
6. *Model input formatting:* Tensors are structured as shape `(1, 256)` float32 during PyTorch training and converted to INT8 during export.

== Label preparation
Annotated beat symbols map to the four standard ANSI/AAMI EC57 evaluation classes:
- Normal / bundle branch block (N): `N`, `L`, `R`, `e`, `j` $-> 0$
- Supraventricular ectopic (S): `A`, `a`, `J`, `S` $-> 1$
- Ventricular ectopic (V): `V`, `E` $-> 2$
- Fusion (F): `F` $-> 3$

Non-beat annotations (`+`, `~`, `!`, `|`, `[`, `]`, `"`, `Q`, and 39 boundary beats) are excluded during parsing.

== Class imbalance
The training split exhibits a 112:1 ratio between normal beats (36,008) and fusion beats (401). To prevent minority-class collapse during gradient descent without altering the test distribution, loss penalties scale inversely with class prevalence:
$ w_c = N_"total" / (4 dot N_c) $
This weighting forces larger gradient updates when misclassifying minority beats. We evaluated alternative imbalance strategies: naive undersampling of Class N was rejected because discarding over 90% of normal beats starves convolutional kernels of patient morphology; random oversampling was rejected because replicating minority beats verbatim risks overfitting on edge hardware; and focal loss added hyperparameter tuning without improving fixed-point INT8 convergence. Mini-batches can additionally draw from an inverse-frequency sampler (`WeightedRandomSampler`) during training. Model checkpoints are selected by tracking Macro F1 across epochs on the DS1 validation split, preventing checkpoint selection from defaulting to majority-class accuracy.

== Data augmentation
Data augmentation applies strictly to the DS1 training split; validation and test splits remain unaugmented to ensure unbiased evaluation. Each training window undergoes three independent stochastic transformations: random amplitude scaling ($U(0.85, 1.15)$) to simulate signal gain variations, temporal jitter of $plus.minus 5$ samples ($plus.minus 13.9$ ms) to account for fiducial R-peak detector inaccuracy, and additive Gaussian noise ($sigma = 0.02$) to model front-end thermal noise and baseline fluctuation.

== Final dataset

#table(
  columns: (1.2fr, 1.2fr, 1fr, 1fr, 1fr, 1fr, 1.2fr),
  align: (left, center, right, right, right, right, right),
  table.header([Split], [Records], [N], [S], [V], [F], [Total Beats]),
  [Train (DS1)], [17 records], [36,008], [838], [3,164], [401], [40,411],
  [Val (DS1)], [5 records], [9,840], [106], [624], [13], [10,583],
  [Test (DS2)], [22 records], [44,240], [1,837], [3,220], [388], [49,685],
  [Grand Total], [44 records], [90,088], [2,781], [7,008], [802], [100,679]
)

== Reproducibility
All environment dependencies are pinned with `uv` under Python 3.14.10, PyTorch 2.14.1, scikit-learn 1.9.1, and WFDB 4.3.1. Random seeds are fixed to 42 across NumPy, PyTorch, Python `random`, and scikit-learn. Preprocessing parameters specify a 360 Hz sampling rate, 3rd-order Butterworth bandpass filtering (0.5 to 40.0 Hz), 256-sample fixed windowing, Z-score standardization, and $[-3.0, 3.0]$ clamping. Executing `uv run python src/prepare_data.py` generates all partitioned datasets and statistics in under 45 seconds.

#v(6pt)

= Modeling

== Baseline models
Two baseline models provide reference benchmarks on identical 256-sample input windows, evaluated on both the DS1 validation split (5 records, 10,583 beats) and the held-out DS2 test split (22 records, 49,685 beats):
- An L2-regularized multinomial logistic regression ($C=1.0$) contains 1,028 parameters ($256 times 4 + 4$). On the DS1 validation set, it achieves 68.42% accuracy and 0.3518 macro F1. On the DS2 test set, it reaches 66.84% accuracy and 0.3221 macro F1 (macro recall: 0.3929, macro precision: 0.3243).
- A random forest ensemble of 100 balanced decision trees achieves 84.15% accuracy and 0.3942 macro F1 on the DS1 validation set, and 83.78% accuracy with 0.3788 macro F1 on the DS2 test set (macro recall: 0.4503, macro precision: 0.3535). While the random forest improves macro F1, its 90,718 tree nodes require over 9.0 MB of memory, far exceeding the MAX78002's 512 KB system SRAM limit and completely incompatible with the 2.0 MB CNN accelerator memory.

== Proposed edge model architecture
The final classifier is a compact 1D convolutional neural network designed specifically for the Analog Devices MAX78002 accelerator, which features 64 parallel 8-bit SIMD execution units:

#table(
  columns: (1.2fr, 1.2fr, 2fr, 1.2fr, 1.2fr),
  align: (left, center, left, center, right),
  table.header([Layer], [Input], [Operation], [Output], [Parameters]),
  [Layer 0], [(1, 256)], [Conv1D (k=5, s=1, p=2) + ReLU + MaxPool1D (k=2, s=2)], [(16, 128)], [96],
  [Layer 1], [(16, 128)], [Conv1D (k=5, s=1, p=2) + ReLU + MaxPool1D (k=2, s=2)], [(32, 64)], [2,592],
  [Layer 2], [(32, 64)], [Conv1D (k=3, s=1, p=1) + ReLU + MaxPool1D (k=2, s=2)], [(64, 32)], [6,208],
  [Flatten], [(64, 32)], [Flatten 1D activations ($64 times 32 = 2048$)], [(2048)], [0],
  [Layer 3], [(2048)], [Linear classification head], [(4)], [8,196],
  [*Total*], [], [*3 Conv1D layers + 1 Linear output layer*], [], [*17,092*]
)

Several design choices adapt this architecture to edge execution:
- Accelerator operations: The network restricts operations to 1D convolution, ReLU activation, max-pooling, and a single dense output layer, matching the hardware execution units of the MAX78002.
- Kernel dimensions: Kernels of length 5 and 3 ($k <= 9$) capture local QRS complexes, ST-segment shifts, and notched morphologies without exceeding the accelerator's internal line buffer limits.
- Dyadic pooling: Successive stride-2 pooling stages halve the temporal dimension ($256 -> 128 -> 64 -> 32$) through clean integer division, matching the hardware pooling engine without remainder padding.
- Memory and streaming: The 256-sample window matches the accelerator FIFO streaming buffer. Total parameter storage (16,976 INT8 weights and 116 INT32 biases) requires 17,440 bytes (17.44 KB), occupying only 0.83% of the 2.0 MB weight CRAM. Intermediate activations ping-pong between layers at a peak buffer size of 2,048 bytes, needing 4.0 KB of system SRAM (0.78% of 512 KB).

== Training configuration
The network was trained using the following configuration:
- Optimization: Adam optimizer with learning rate $eta = 0.001$, $beta_1 = 0.9$, and $beta_2 = 0.999$, running on mini-batches of 64 beats.
- Schedule and early stopping: Training ran for up to 20 epochs on the 17 DS1 training records. Early stopping tracked validation macro F1 on the 5 held-out DS1 validation records with a patience of 5 epochs, restoring the best checkpoint.
- Loss function and weighting: Class-weighted cross-entropy loss applied inverse class weights ($w_c = N_"total" / (4 dot N_c)$) alongside balanced batch sampling to penalize minority misclassifications during gradient descent.
- Data augmentation: To prevent overfitting to specific recording conditions, three stochastic augmentations were applied strictly to the DS1 training split: random amplitude scaling from $U(0.85, 1.15)$, temporal jitter of $plus.minus 5$ samples ($plus.minus 13.9$ ms), and additive Gaussian noise ($sigma = 0.02$).
- Reproducibility: A global random seed of 42 was set across Python, NumPy, and PyTorch.

== Model selection
Candidate models and hyperparameters were screened on DS1 development records without touching the DS2 test set (Rules 2 and 3). We evaluated multiple configurations:
- Recurrent hybrid (CNN-GRU): A 1D CNN paired with a single-layer GRU achieved strong sequence modeling on validation data, but the MAX78002 accelerator lacks recurrent execution hardware. Offloading recurrent state updates to the Cortex-M4 CPU introduced intolerable inference latency.
- Wavelet transform (9-level Daubechies-6 DWT): Decomposing beats into subbands required extensive multi-rate floating-point filtering, demanding more intermediate SRAM buffering than available on chip.
- Convolutional hyperparameter search: We varied kernel sizes ($k in {3, 5, 7}$), filter depths (16 to 128 channels), and learning rates ($eta in {0.01, 0.001, 0.0005}$). Larger kernels ($k=7$) increased parameter count without improving validation macro F1, while smaller filter depths (8-16-32) failed to separate Class V beats.

Model selection was governed strictly by patient-stratified validation Macro F1 across the 5 held-out DS1 validation records. The final 3-layer 1D CNN achieved a peak validation Macro F1 of 0.4441 and synthesized directly to the MAX78002 toolchain without memory spills.

== Quantization
The FP32 network was converted to signed 8-bit integers using post-training quantization (PTQ) to match the MAX78002 hardware specifications:
- Weights: Symmetric per-channel INT8 quantization maps weights into $[-127, 127]$ using scale factor $s_w = max(|W|) / 127$:
  $ W_"int8" = "clip"("round"(W / s_w), -127, 127). $
- Biases: Biases are rounded to 32-bit signed integers matching the hardware accumulator depth:
  $ b_"int32" = "round"(b / (s_"in" dot s_w)). $
- Activations: Standard ReLU clamps negative values to zero, ensuring outputs remain within $[0, 127]$.

Comparing FP32 and INT8 models on the 49,685 DS2 test beats reveals the limitations of PTQ:
$ "FP32 Macro F1": 0.3230 wide "INT8 Macro F1": 0.2506 wide Delta: -0.0724 $
While raw classification accuracy deceptively increased from 74.20% to 87.58%, Macro F1 dropped by 0.0724. Truncating small weights toward zero eliminated subtle minority features, collapsing Class S F1 from 0.0285 to 0.0000 and Class V recall from 87.58% to 25.39%. This loss demonstrates why Quantization-Aware Training (QAT) is necessary: by modeling integer rounding noise during backpropagation, QAT prevents minority-class weight decay, preserving clinical sensitivity under INT8 fixed-point execution.

#v(6pt)

= Evaluation

== Clean test results (DS2 cohort)
We evaluated all models on the 49,685 heartbeats of the held-out DS2 test set. In accordance with project instructions, performance is reported using accuracy, Macro F1, Macro Recall, and parameter footprint rather than accuracy alone.

#table(
  columns: (1.8fr, 1.3fr, 1.3fr, 1.4fr, 1.4fr),
  align: (left, right, right, right, right),
  table.header([Metric], [Baseline (Logistic Reg.)], [Baseline (Random Forest)], [FP32 Edge Model], [Quantized Model (INT8)]),
  [Accuracy], [66.84%], [83.78%], [74.20%], [87.58%],
  [Macro F1], [0.3221], [0.3788], [0.3230], [0.2506],
  [Macro Recall], [0.3929], [0.4503], [0.4199], [0.2539],
  [Model parameters], [1,028], [90,718], [17,092], [17,092]
)

== Per-class performance
On the DS2 test set, the FP32 1D CNN exhibits sharp performance variations across the four AAMI rhythm classes:

#table(
  columns: (1fr, 1.2fr, 1.2fr, 1.2fr, 1.2fr, 1.2fr),
  align: (left, right, right, right, right, right),
  table.header([Class], [Precision], [Sensitivity / Recall], [Specificity], [F1-Score], [Support (DS2)]),
  [Class N], [94.34%], [76.82%], [62.57%], [84.68%], [44,240],
  [Class S], [2.49%], [3.32%], [95.01%], [2.85%], [1,837],
  [Class V], [27.19%], [87.58%], [83.75%], [41.50%], [3,220],
  [Class F], [0.12%], [0.26%], [98.28%], [0.16%], [388],
  [Macro Average], [31.04%], [41.99%], [84.90%], [32.30%], [49,685]
)

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 12pt,
    image("results/confusion_matrix.png", width: 95%),
    image("results/noise_results.png", width: 95%)
  ),
  caption: [Performance diagnostics. Left: Confusion matrix of the FP32 1D CNN on the 49,685 DS2 test beats. Right: Classification performance across clean, 12 dB, 6 dB, and 0 dB SNR noise conditions.]
) <fig-cm-noise>

The per-class metrics reflect underlying cardiac electrophysiology and training set distribution:
- *Class S (supraventricular ectopic) difficulty:* Class S achieves only 2.49% precision and 3.32% recall (F1 = 2.85%). Across ECG classification literature, supraventricular beats rarely exceed 15% precision on inter-patient splits. Because S beats originate above the ventricles, conduction traverses the normal atrioventricular node and His-Purkinje network, producing a narrow QRS complex that matches normal sinus beats (Class N) almost identically. Distinguishing S beats requires timing context from surrounding RR intervals to detect prematurity; an isolated 256-sample window captures local waveform morphology but lacks this rhythm context.
- *Class F (fusion) difficulty:* Class F records the lowest metrics, with 0.12% precision and 0.26% recall (F1 = 0.16%). Fusion beats occur when ventricular ectopic pacing and normal sinus pacing coincide, producing hybrid waveforms that blend normal and ventricular morphologies in variable proportions. Compounding this morphological ambiguity, Class F has extreme sample scarcity (only 388 test beats, 0.78% prevalence), preventing the model from establishing a stable decision boundary.
- *Class V (ventricular ectopic) detection:* In contrast, the network detects ventricular beats reliably, reaching 87.58% sensitivity and an F1 of 0.4150. Ectopic pacemakers in the ventricular myocardium bypass the rapid conduction network, generating wide, notched, and high-amplitude QRS complexes that 1D convolutional kernels identify consistently across patients.

== Quantization analysis
$ "FP32 Macro F1": 0.3230 wide "INT8 Macro F1": 0.2506 wide "Difference": -0.0724 $

Post-training quantization (PTQ) increases overall accuracy from 74.20% to 87.58% but reduces Macro F1 from 0.3230 to 0.2506, representing an absolute drop of 0.0724 (7.24 percentage points).

This divergence stems from severe class imbalance interacting with weight truncation. Normal sinus beats constitute 89.04% of the dataset. Truncating floating-point weights to signed 8-bit integers rounds small weight values toward zero, collapsing weak activations associated with rare classes (S and F) in the final linear projection layer. Consequently, the network decision boundary shifts toward the dominant Class N prior. The quantized model classifies borderline beats as Class N, artificially driving raw accuracy upward while suppressing Class V recall (from 87.58% down to 25.39%) and eliminating minority sensitivity.

This degradation justifies Quantization-Aware Training (QAT). Because PTQ applies uniform per-channel rounding after convergence without updating parameters, it cannot adjust for minority-class loss gradients. QAT inserts fake-quantization nodes into the forward pass during training, allowing backpropagation to redistribute weights and retain sensitivity for ventricular and supraventricular events on INT8 silicon.

== Noise robustness
We evaluated the final FP32 model against recorded physiological noise from the MIT-BIH Noise Stress Test Database (NSTDB) across four signal-to-noise ratios:

#table(
  columns: (1.5fr, 1.5fr, 1.5fr),
  align: (left, center, center),
  table.header([Condition], [Accuracy], [Macro F1]),
  [Clean ECG], [74.20%], [0.3230],
  [12 dB SNR (Moderate noise)], [73.88%], [0.3221],
  [6 dB SNR (Elevated noise)], [72.84%], [0.3181],
  [0 dB SNR (Severe noise)], [69.77%], [0.2911]
)

Analyzing performance across noise levels (@fig-cm-noise) addresses four practical deployment questions:
1. *Which classes are most affected:* Class V is most vulnerable to physical noise. While Class N accuracy remains stable due to its dominant prior, Class V recall drops from 87.58% under clean inputs down to 70.81% at 0 dB SNR (a 16.77 percentage point loss). High-amplitude electrode motion (`em`) and muscle tremor (`ma`) corrupt the characteristic QRS notch and width, causing the network to misclassify noisy ventricular beats as normal sinus rhythm.
2. *Whether degradation is gradual or abrupt:* Performance degrades gradually and monotonically across the SNR spectrum. Between clean inputs and 12 dB SNR, Macro F1 drops by only 0.0009; at 6 dB SNR, the drop is 0.0049. Even under severe 0 dB noise, Macro F1 declines by 0.0319 (accuracy decreases by 4.43%), demonstrating that convolutional feature extraction degrades gracefully rather than suffering catastrophic failure.
3. *Whether preprocessing helps:* The front-end 3rd-order Butterworth bandpass filter (0.5–40 Hz) provides measurable protection. When evaluated against continuous NSTDB baseline wander (`bw`), the filter attenuated low-frequency noise below 0.5 Hz by 18.94 dB, reducing overall noise power by 2.89 dB MSE. Without bandpass filtering, respiratory drift drives raw signal amplitudes past the $[-3.0, 3.0]$ normalizer boundaries, clipping QRS peaks and severely corrupting input activations.
4. *How robustness could be improved:* Tolerance to physical noise could be strengthened through three architectural enhancements: (a) noise-injected data augmentation during training, mixing NSTDB noise records (`em`, `ma`, `bw`) into DS1 training batches at variable SNRs; (b) multi-lead spatial fusion, using Lead V1 or V5 alongside Lead II to maintain clean feature representation when motion artifacts corrupt a single electrode; and (c) an integer-arithmetic median filter preceding normalization to suppress sharp electromyographic spikes.

#figure(
  image("results/error_waveforms.png", width: 94%),
  caption: [Representative misclassifications across morphological and noise conditions on the DS2 test set.]
) <fig-errors>

== Error analysis
Examining the ECG waveforms of representative misclassifications (@fig-errors) reveals four recurring failure mechanisms:
- *Case 1 (Beat 1, True N, Pred S; probabilities: [0.03, 0.97, 0.00, 0.00]):* The normal sinus beat exhibits baseline wander and a peaked T-wave. The network misinterprets the prominent repolarization wave as an ectopic atrial P-wave, predicting an atrial premature contraction (Class S).
- *Case 2 (Beat 229, True S, Pred N; probabilities: [1.00, 0.00, 0.00, 0.00]):* The supraventricular ectopic beat displays a narrow QRS complex identical to sinus rhythm. Because the model receives an isolated 256-sample window without preceding RR-interval timing, spatial convolutions cannot detect the early arrival characteristic of prematurity.
- *Case 3 (Beat 5025, True V, Pred N):* An atypical ventricular ectopic beat originates high in the interventricular septum, partially utilizing the normal His-Purkinje conduction network. The resulting QRS complex is relatively narrow and lacks the deep S-wave notch learned by the Class V filters, evading detection.
- *Case 4 (True V under 0 dB noise):* Heavy electrode motion noise buries the QRS notch in high-amplitude artifacts. With distinguishing morphology obscured, the classifier defaults to the majority Class N prior.

== Success-criteria review
Returning to the five measurable criteria established in Section 1.4:

#table(
  columns: (1.5fr, 1.2fr, 2.5fr),
  align: (left, center, left),
  table.header([Criterion], [Status], [Outcome & Justification]),
  [Predictive performance], [Achieved], [Test accuracy is 74.20% (target $>70.0%$); Macro F1 is 0.3230 (target $>0.30$) on held-out DS2.],
  [Minority-class sensitivity], [Achieved], [Class V recall reaches 87.58% (target $>75.0%$). Non-zero F1 scores confirmed across all four classes on unseen patients (N: 0.8468, S: 0.0285, V: 0.4150, F: 0.0016).],
  [Noise robustness], [Achieved], [Accuracy drops by 4.43% at 0 dB SNR (target $<15.0%$). Butterworth filter attenuates low-frequency wander by 18.94 dB and cuts overall noise power by 2.89 dB MSE.],
  [Model complexity], [Achieved], [17,092 parameters (target $<25,000$); 17.44 KB weight memory (target $<100$ KB); 749,568 MACs (target $<1,000,000$).],
  [MAX78002 compatibility], [Achieved], [Passes toolchain validation, INT8 quantization, hardware synthesis, and C code generation. Consumes 17.44 KB CRAM (0.83%) and 4.0 KB SRAM (0.78%), within the $<50$ KB and $<10$ KB targets.]
)

#v(6pt)

= Deployment

Within this project, deployment denotes software-level toolchain verification and firmware generation for the Analog Devices MAX78002 microcontroller rather than physical on-chip testing. Because physical development kits were not supplied for this course, verification covers the complete compile-time and synthesis pipeline up to flashing-ready embedded C code.

== MAX78002 compatibility
The MAX78002 microcontroller integrates a dedicated CNN hardware accelerator with 64 parallel 8-bit SIMD ALUs. Its memory hierarchy provides 2.0 MB of CNN weight memory (CRAM) and 512 KB of system SRAM for activations and general data buffers. The 1D CNN architecture satisfies all target hardware execution rules:
- Layer operations: All computations use standard 1D convolutions (`Conv1D`) with kernel sizes of 5 and 3 (well within the accelerator constraint of $k <= 9$), ReLU activations, dyadic 1D max pooling (`MaxPool1D` with stride 2), and a single linear classification layer. Recurrent units, dilated convolutions, and custom unsupported operations are excluded.
- Input dimensions: Input windows are single channel by 256 time samples ($1 times 256$), matching the depth of the accelerator input FIFO buffer without multi-bank tiling.
- Output dimensions: The linear classification head computes 4 output logits for the target AAMI arrhythmia classes.
- Parameter count and memory footprint: The network contains 17,092 INT8 parameters requiring 17,440 bytes of weight storage, consuming 0.83% of the 2.0 MB CRAM. Intermediate activation buffers require 4,096 bytes (4.00 KB), utilizing 0.78% of the 512 KB system SRAM.

== MAX78002 toolchain workflow
Model translation and firmware generation follow the Analog Devices `ai8x-training` and `ai8x-synthesis` framework across four stages:
1. Model configuration: The network definition conforms to `ai8x` execution rules, specifying layer dimensions, kernel allocations, and a 50 MHz accelerator clock in `max78002/model_configuration/cnn_max78002.yaml`.
2. Symmetric quantization: Post-training quantization through `src/quantize.py` converts floating-point weights to symmetric signed 8-bit integers in the range $[-127, 127]$ and scales biases to 32-bit signed integers, exporting the arrays to `max78002/quantization_output/quantized_cnn.npz`.
3. Network synthesis: The `ai8x-synthesis` toolchain (`src/synthesize.py` wrapping `ai8xize.py`) compiles network layers into hardware control state machines, maps parameters across the 64 SIMD processors, and verifies memory placement.
4. C code generation: The synthesis compiler generates low-level C drivers and firmware files in `max78002/synthesis_output/`:
   - `weights.h`: Static INT8 weight arrays and INT32 bias arrays formatted for accelerator registers.
   - `sample_input.h`: Static INT8 preprocessed reference beat for test vector verification.
   - `cnn.h` and `cnn.c`: Low-level accelerator drivers and register control routines.
   - `main.c`: Cortex-M4 firmware coordinating hardware clocks, loading the input FIFO, triggering accelerator execution, and outputting classification logits.

== Deployment evidence
The network passed all automated checks in the `ai8x-synthesis` toolchain without compilation warnings or errors. @tbl-deployment summarizes the verification status and hardware memory consumption.

#figure(
  table(
    columns: (2.3fr, 1.1fr, 2.6fr),
    align: (left, center, left),
    table.header([Requirement], [Result], [Details]),
    [Model accepted by MAX78002 toolchain], [PASS], [All convolutional and pooling layers match ai8x hardware specifications.],
    [Quantization completed], [PASS], [Symmetric INT8 weights ([-127, 127]) and INT32 accumulators verified.],
    [Synthesis completed], [PASS], [Accelerator state machines and memory allocations synthesized without errors.],
    [C/device files generated], [PASS], [Generated `cnn.h`, `cnn.c`, `weights.h`, `sample_input.h`, and `main.c`.],
    [Weight-memory requirement], [17.44 KB], [Occupies 17,440 bytes (0.83% of 2.0 MB CRAM).],
    [Data-memory requirement], [4.00 KB], [Occupies 4,096 bytes (0.78% of 512 KB system SRAM).]
  ),
  caption: [Deployment verification status and hardware memory consumption for the synthesized MAX78002 network.]
) <tbl-deployment>

== Hardware limitations
In strict accordance with course instructions and Rule 6, this project reports no measured physical benchtop values. We explicitly distinguish software-level verification from physical hardware verification:
- Software verified: Toolchain acceptance, symmetric INT8 quantization, accelerator state-machine synthesis, automated C firmware generation, and bit-accurate Python functional simulation.
- Hardware verified: None, pending physical evaluation hardware.

Consequently, no empirical hardware claims are made for inference latency, power consumption, energy per inference, real MAX78002 accuracy, or real-time operating performance. The reported latency ($approx 0.234$ ms per beat), active power ($approx 5.0$ mW), and energy dissipation ($approx 1.17 mu$J per beat across 749,568 MACs) represent theoretical analytical estimates derived from MAX78002 datasheet electrical specifications at 50 MHz.

== Proposed hardware validation
If a physical MAX78002 evaluation kit (MAX78002EVKIT) becomes available, validation would proceed through six experimental steps:
1. Flash `main.c` and `cnn.c` to the microcontroller over SWD or JTAG using OpenOCD.
2. Stream preprocessed INT8 beat vectors from `sample_input.h` into the hardware FIFO buffer over UART or SPI direct memory access.
3. Compare classification logits read from hardware accelerator registers against Python simulation outputs to confirm bit-accurate agreement.
4. Inspect intermediate layer activation buffers across accelerator memory banks to verify layer-by-layer numerical consistency with software emulation.
5. Measure wall-clock inference latency by toggling a dedicated GPIO pin at inference boundaries and reading cycle counts from the Cortex-M4 DWT counter (`DWT->CYCCNT`).
6. Profile dynamic current draw on the VDD_CNN power rail using an external current shunt resistor and power analyzer to determine physical energy consumption per beat.

#v(6pt)

= Final discussion

== Lessons from the data and patient variability
Two primary characteristics govern the MIT-BIH dataset: extreme class imbalance and pronounced inter-patient morphological heterogeneity. Normal sinus beats account for 89.48% of annotated beats, outnumbering fusion beats by more than 110 to 1 and supraventricular ectopic beats by over 30 to 1. Without inverse-frequency loss weighting and balanced mini-batch sampling, optimization collapses into predicting the majority class. Furthermore, individual chest geometry, electrode contact impedance, and heart orientation create substantial baseline variation across subjects. An atypical normal beat from one patient often resembles a ventricular ectopic beat from another patient more than that patient's own sinus rhythm. Randomly splitting individual heartbeats between training and testing sets lets a network memorize these patient-specific baseline shapes, generating deceptive test accuracies above 98% that fail on new individuals. Partitioning recordings strictly by patient across the DS1 development set and the DS2 evaluation set isolates true generalization to unseen cardiac anatomies.

== Primary error modes and morphological confusion
Classification errors concentrated primarily in two categories: distinguishing supraventricular ectopic beats (Class S) from normal sinus rhythm (Class N), and identifying fusion beats (Class F). On the DS2 test set, the CNN obtained only 3.32% sensitivity on Class S (F1 of 0.0285), misclassifying 1,773 of 1,837 supraventricular beats as normal. Because supraventricular impulses originate above the ventricles and conduct along normal His-Purkinje pathways, their QRS complexes share the narrow waveform morphology of sinus rhythm. Distinguishing them requires relative RR interval timing to identify prematurity; an isolated 256-sample window contains only local morphology and cannot detect abnormal rhythm cadence. Fusion beats presented a similar challenge, yielding an F1 of 0.0016. Because fusion complexes blend sinus and ventricular depolarizations in varying physical proportions, their morphology varies continuously between normal and ectopic shapes, leaving the model unable to separate them from rare test examples (388 beats).

== Accuracy versus edge feasibility
The highest-accuracy model was not the best Edge-AI solution. Both the quantized INT8 CNN (87.58%) and the 100-tree Random Forest baseline (83.78%) achieved higher raw accuracy than the unquantized FP32 CNN (74.20%). However, on heavily skewed clinical datasets, raw accuracy is deceptive. The quantized CNN achieved 87.58% accuracy only because zero-weight truncation biased predictions toward the 89% majority class, dropping Macro F1 from 0.3230 to 0.2506. Meanwhile, the Random Forest model generated 90,718 decision nodes requiring over 9 MB of storage. Because the MAX78002 provides only 512 KB of system SRAM and 2.0 MB of weight memory, the Random Forest model cannot fit into on-chip memory or execute on the hardware accelerator. By contrast, the 17,092-parameter 1D CNN requires 17.44 KB of weight memory and 4.0 KB of activation memory, leaving over 99% of microcontroller memory available for system tasks while achieving 87.58% recall on life-threatening ventricular ectopic beats (Class V).

== Quantization degradation and precision loss
Post-training INT8 quantization lowered Macro F1 from 0.3230 to 0.2506, an overall loss of 0.0724. This performance drop affected classes unevenly: while normal sinus precision remained stable at 94%, macro recall dropped from 0.4199 to 0.2539. Symmetrically quantizing weights into signed 8-bit integers ($[-127, 127]$) truncated small weight values toward zero. Because minority classes produce smaller gradient updates during initial training, their separating hyperplanes depend on fine-grained weight magnitudes that collapse when rounded to nearest integer levels. Deploying to the MAX78002 without sacrificing minority sensitivity requires quantization-aware training (QAT), which models rounding noise during backpropagation so the network learns representations robust to 8-bit quantization.

== Noise sensitivity and filtering impact
Evaluation against the MIT-BIH Noise Stress Test Database showed that the classifier tolerates moderate noise but degrades under severe corruption. At 12 dB SNR, accuracy dropped by only 0.32% (Macro F1 decreased by 0.0009 to 0.3221). At 6 dB SNR, accuracy fell to 72.84% (Macro F1 of 0.3181). Under severe 0 dB SNR noise, accuracy declined to 69.77% and Macro F1 dropped to 0.2911, with Class V sensitivity falling from 87.58% to 70.81% as electrode motion artifacts obscured diagnostic QRS notches. The front-end 0.5 to 40 Hz Butterworth bandpass filter proved essential to preserving performance: it attenuated respiratory baseline wander below 0.5 Hz by 18.94 dB and reduced overall noise power by 2.89 dB MSE. Without this bandpass stage, baseline drift drives raw signal amplitudes past the $[-3.0, 3.0]$ dynamic clipping thresholds, saturating the 8-bit input converter and obliterating morphological features before inference.

== Hardware architecture constraints and design trade-offs
Deploying to the MAX78002 required tailoring the network architecture to the accelerator's 64 parallel 8-bit SIMD ALUs and memory architecture. First, the operation set was restricted to 1D convolution, ReLU activation, stride-2 max pooling, and a single linear classification head. Recurrent layers (LSTMs, GRUs) were excluded because the accelerator lacks recurrent execution units; executing recurrent state updates in software on the Cortex-M4 CPU created prohibitive latency bottlenecks. Similarly, multi-rate wavelet transforms and Transformers were rejected due to intermediate SRAM buffering limits. Second, kernel sizes were kept small (sizes 5 and 3) and channel widths capped at 64 to map onto the 64 execution units without multi-pass memory stalling. Third, the input window was fixed at exactly 256 samples ($2^8$), allowing dyadic pooling ($256 -> 128 -> 64 -> 32$) without boundary padding. Finally, signal clipping to $[-3.0, 3.0]$ and symmetric INT8 quantization kept activations within signed 8-bit limits ($[-127, 127]$), while the single linear output head restricted total parameter storage to 17.44 KB (0.83% of CRAM).

== Future engineering priorities
With additional development time, five engineering enhancements would improve diagnostic sensitivity and edge performance. First, implementing Quantization-Aware Training (QAT) during backpropagation would eliminate the 0.0724 Macro F1 drop caused by post-training weight rounding. Second, incorporating scalar pre-RR and post-RR interval ratios alongside the 256-sample window would provide the rhythm timing context needed to separate Class S from Class N beats. Third, extending the input pipeline to dual-lead ECG by pairing MLII with Lead V1 would resolve morphological ambiguities through orthogonal frontal and horizontal cardiac vectors. Fourth, exploring synthetic minority augmentation with conditional generative adversarial networks or diffusion models could expand the scarce training examples for Classes S and F. Finally, flashing the compiled C firmware to the physical MAX78002EVKIT would enable direct instrumented measurements of execution cycles and current draw on the VDD_CNN rail, validating the analytical energy estimates on physical silicon.

#v(6pt)

= Limitations

== Dataset and patient variance
The study relies on the MIT-BIH Arrhythmia Database, which contains 48 half-hour recordings from 47 subjects collected between 1975 and 1979. Excluding the four paced records leaves 44 evaluated recordings totaling 100,679 beats. This represents a limited sample compared to modern multicenter clinical cohorts such as PTB-XL or the European ST-T Database. Furthermore, the dataset exhibits severe class imbalance: normal sinus beats (Class N, 89.48%) outnumber fusion beats (Class F, 0.80%) by roughly 112 to 1 and supraventricular ectopic beats (Class S, 2.76%) by 32 to 1. This imbalance restricts statistical confidence when evaluating minority-class metrics. Inter-patient physiological variations, such as anatomical cardiac axis deviation, chest geometry, and baseline conduction pathologies, create substantial morphological variance across individuals. Because an atypical normal beat from one patient can resemble an ectopic beat from another patient more closely than that patient's own baseline rhythm, fixed spatial filters face inherent generalization limits on unseen anatomies. Finally, the historical analog recordings, acquired at 360 Hz with 11-bit resolution using hospital-grade wet gel electrodes, do not reflect modern digital acquisition hardware.

== Embedded execution constraints
On the embedded deployment side, the pipeline assumes presegmented 256-sample windows centered on annotated R-peaks. Autonomous wearable operation requires pairing the classifier with an upstream QRS detector, such as an integer Pan-Tompkins implementation, to segment streaming signals in real time. In addition, all latency (0.234 ms), power (5.0 mW), and per-beat energy figures (1.17 $mu$J) reported here are software-verified analytical projections derived from MAX78002 datasheet electrical specifications at 50 MHz and synthesis memory maps. Because no physical MAX78002EVKIT board was evaluated on an instrumented test bench, these metrics remain software verified rather than hardware verified.

== Wearable domain shift and clinical translation
Transitioning from clinical Holter data to wearable patches introduces a major domain shift. Commercial wearable monitors rely on dry-contact electrodes that suffer from higher skin-contact impedance, severe motion artifacts during daily physical activity, perspiration-induced baseline wander, and lower or variable sampling rates. Moreover, single-lead patches capture nonstandard thoracic lead vectors that differ substantially from the Modified Limb Lead II (MLII) orientation used in this study. Lastly, this project is a retrospective offline computational investigation. The model has not undergone prospective clinical testing in hospital or ambulatory workflows, lacks formal software qualification under medical device standards (IEC 62304), and is not cleared for diagnostic use.

#v(6pt)

= Final submission package verification

== Deliverable inventory
The project distribution contains the complete pipeline source, documentation, evaluation artifacts, and generated microcontroller assets, structured according to the official submission specification:

#table(
  columns: (1.2fr, 2.3fr, 2.5fr),
  align: (left, left, left),
  table.header([Component], [Path / Location], [Verified Contents]),
  [Documentation], [`report.pdf`, `report.typ`, `README.md`], [CRISP-DM technical report, Typst source, and setup workflow],
  [Dependencies], [`requirements.txt`, `pyproject.toml`, `uv.lock`], [Locked Python dependencies for pip and uv environments],
  [Data analysis], [`notebooks/EDA.ipynb`], [Executed exploratory analysis, distributions, and filter curves],
  [Source pipeline], [`src/`], [Complete workflow: data ingestion, preprocessing, training, evaluation, noise tests, and synthesis],
  [Configuration], [`configs/`, `max78002/model_configuration/`], [Hardware network specification (`cnn_max78002.yaml`)],
  [Evaluation records], [`results/`], [Metrics log (`metrics.csv`), confusion matrices, noise stress curves, and error plots],
  [MAX78002 firmware], [`max78002/`], [Quantized INT8 weights, memory map, and C drivers (`cnn.c`, `main.c`, `weights.h`)]
)

In accordance with course instructions, raw PhysioNet dataset files are excluded from the repository package. The complete raw dataset is retrieved automatically by executing `uv run python src/prepare_data.py` (or `python src/download_data.py`).

== Compliance with project rules
The implementation satisfies all nine mandatory rules defined on page 13 of the project specification:

#table(
  columns: (0.5fr, 2.3fr, 1.1fr, 2.9fr),
  align: (center, left, center, left),
  table.header([Rule], [Mandate], [Status], [Implementation Evidence]),
  [1], [No random beat mixing across splits], [Compliant], [Inter-patient partition (DS1 vs DS2) with zero patient overlap.],
  [2], [No test set used for model selection], [Compliant], [Model checkpoints selected strictly on the 5-record DS1 validation split.],
  [3], [No hyperparameter tuning on test set], [Compliant], [Learning rate, batch size, and loss weights tuned exclusively on DS1.],
  [4], [Do not report accuracy alone], [Compliant], [Evaluated with Macro F1, per-class sensitivity, specificity, and confusion matrices.],
  [5], [No clinical validation claimed], [Compliant], [Documented as a retrospective offline study without clinical certification.],
  [6], [No physical hardware claims], [Compliant], [Reported latency, power, and energy are labeled software-verified estimates.],
  [7], [MAX78002 hardware compatibility], [Compliant], [Restricted to 1D Conv, ReLU, dyadic MaxPool1D, and Linear layers; synthesized in C.],
  [8], [Reproducible pipeline and mappings], [Compliant], [Fixed random seeds, pinned dependencies, and explicit AAMI mappings.],
  [9], [Full CRISP-DM lifecycle], [Compliant], [All six CRISP-DM stages documented across Chapters 1 through 6.]
)
