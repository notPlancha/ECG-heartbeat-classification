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
  
  This report evaluates an ECG heartbeat classification system targeted to the Analog Devices MAX78002 ultra-low-power microcontroller. Under the ANSI/AAMI EC57 standard, the classifier maps single-lead ECG beats into four clinical categories: normal (N), supraventricular ectopic (S), ventricular ectopic (V), and fusion (F). The 44 evaluated MIT-BIH recordings (100,679 annotated beats) partition by patient into equal 22-record development (DS1) and held-out test (DS2) cohorts, eliminating intra-patient data leakage.
  
  The classifier uses a three-layer 1D convolutional neural network with 17,092 parameters, dimensioned specifically for the MAX78002 hardware accelerator. On the 49,685 DS2 test beats, the unquantized FP32 model obtains 74.20% accuracy, a Macro F1 score of 0.3230, and 87.58% recall on ventricular ectopic beats. Post-training quantization to symmetric INT8 fits the microcontroller memory with substantial margin, requiring 17.44 KB of weight memory (0.83% of the 2.0 MB CRAM) and 4.0 KB of activation SRAM (0.78% of the 512 KB pool). At 50 MHz, estimated inference latency is 0.234 ms per beat (749,568 MACs) with an energy draw of 1.17 $mu$J. On the MIT-BIH Noise Stress Test Database, front-end 0.5 to 40 Hz Butterworth filtering attenuates baseline wander by 18.94 dB and cuts overall noise power by 2.89 dB, preventing baseline drift and motion artifacts from saturating the input normalizer and fixed-point dynamic range.
]

#v(6pt)

= Business / application understanding

== Application scenario
Cardiac arrhythmias range from acute emergencies like ventricular fibrillation to recurrent ectopic beats that indicate underlying disease or elevated stroke risk. Conventional Holter monitors store multi-day ECG recordings to flash memory for retrospective physician review, leaving clinicians unaware of acute rhythm disturbances until days later.

Moving inference onto wearable edge microcontrollers allows real-time beat classification directly on ambulatory patches. By processing signals locally, the device avoids continuous Bluetooth or cellular transmission, cutting the radio duty cycle by roughly 90% and extending coin-cell battery life to several weeks. On-device analysis also eliminates cloud latency for urgent ventricular alerts, keeps patient waveforms private, and operates reliably in shielded hospital wards without cellular coverage. This workflow supports ambulatory patients, post-infarction convalescents, elderly individuals at elevated arrhythmia risk, and cardiologists triaging high-volume Holter recordings.

== Data-mining objective
The machine learning objective is to classify single-lead 1D ECG heartbeats into discrete diagnostic categories within the arithmetic and storage limits of an edge microcontroller. Each input is a 256-sample window (711.1 ms at 360 Hz) centered on a detected R-peak, providing 90 samples (250.0 ms) of pre-peak context (P-wave and PR segment) and 166 samples (461.1 ms) of post-peak context (QRS complex, ST segment, and T-wave). The model outputs a discrete four-class prediction conforming to ANSI/AAMI EC57: normal or bundle branch block (Class N), supraventricular ectopic (Class S), ventricular ectopic (Class V), or fusion (Class F). Training must handle severe class imbalance (~89% N, ~3% S, ~7% V, ~0.8% F) and support symmetric INT8 quantization without unacceptable loss in minority-class recall.

== Technical objectives
Deploying the classifier to the MAX78002 establishes five engineering targets:
- Sort heartbeats into the four standard AAMI groups (N, S, V, F).
- Generalize across unseen patients by evaluating strictly on an independent test cohort (DS2).
- Retain classification reliability under physical noise (electrode motion, muscle tremor, baseline wander) evaluated at 12 dB, 6 dB, and 0 dB SNR.
- Restrict model footprint to fewer than 25,000 parameters, under 100 KB total storage, and under 1,000,000 MAC operations per beat to execute within on-chip SRAM.
- Maintain compatibility with the Analog Devices MAX78002 CNN accelerator by using only hardware-supported operations that compile directly to embedded C code through the `ai8x-synthesis` toolchain.

== Success criteria
Because normal sinus beats account for roughly 89% of ambulatory ECG data, overall accuracy alone cannot confirm clinical efficacy. Evaluation relies on five quantitative thresholds:
- Macro-averaged F1 on the held-out DS2 test set must exceed 0.30, with overall accuracy above 70.0%.
- Sensitivity (recall) on Class V must exceed 75.0% to detect life-threatening ventricular arrhythmias reliably, while Classes S and F must achieve non-zero F1 scores on unseen patients.
- Under 0 dB physical noise from the MIT-BIH Noise Stress Test Database, accuracy degradation must not exceed 15% relative to clean ECG inputs, supported by front-end bandpass filtering that attenuates baseline wander and electromyographic noise.
- Model footprint must stay within 25,000 parameters, 100 KB flash storage, and 1,000,000 MAC operations per beat.
- The network must pass every stage of the MAX78002 toolchain (model validation, INT8 quantization, synthesis, and C driver generation), requiring under 50 KB of weight memory (below 2.5% of CRAM) and under 10 KB of activation SRAM (below 2.0% of system SRAM).

== Hardware and operational constraints
The edge deployment environment imposes six operational constraints:
1. The MAX78002 provides 2.0 MB of CNN weight memory (CRAM) and 512 KB of system SRAM, with no external RAM available on the board.
2. The hardware accelerator computes strictly with signed 8-bit integers ([-127, 127] for symmetric weights) and 32-bit accumulators, requiring software emulation on the Cortex-M4 CPU for any floating-point operation.
3. Normal sinus beats outnumber ectopic beats by up to 100 to 1, causing unweighted cross-entropy models to collapse into predicting only the majority class.
4. The MIT-BIH database contains recordings from only 47 subjects, making cross-patient generalization difficult due to variations in electrode placement, chest anatomy, and cardiac history.
5. Ambulatory recordings carry severe physical artifacts, including electrode motion (`em`), baseline wander (`bw`), and muscle tremor (`ma`).
6. Reported deployment metrics are software-verified estimates derived from `ai8x-synthesis` memory maps and manufacturer datasheet specifications rather than physical benchtop measurements.

== Risks and mitigations

#table(
  columns: (1.2fr, 2fr, 2fr),
  align: (left, left, left),
  table.header([Identified Risk], [Technical and Clinical Impact], [Mitigation Strategy]),
  [Intra-patient data leakage], [Randomly assigning beats from the same recording into training and test sets lets the network memorize patient-specific baseline morphologies. This produces deceptive test accuracies exceeding 98% that collapse when evaluated on new patients.], [Use the inter-patient partitioning protocol from de Chazal et al. (2004): 22 records in DS1 (training and validation) and 22 completely separate records in DS2 (testing), with zero patient overlap.],
  [Minority class collapse], [Standard cross-entropy loss biases optimization toward Class N. A naive model can reach ~89% accuracy while missing nearly all ventricular and supraventricular events.], [Apply class-weighted loss with inverse-frequency weights ($w_c = N_"total" / (4 dot N_c)$) alongside balanced batch sampling on the training set to penalize minority misclassifications during backpropagation.],
  [Hardware accelerator incompatibility], [Recurrent networks (LSTMs, GRUs) and complex transforms (wavelets) cannot execute on the MAX78002 CNN hardware accelerator.], [Restrict the model architecture strictly to 1D convolution, ReLU, stride-2 MaxPool1D, and linear layers, verifying synthesis acceptance against the `ai8x-synthesis` YAML specification.],
  [Quantization precision loss], [Converting 32-bit floating-point weights to 8-bit integers can truncate small minority-class weights to zero, degrading recall on rare beat types.], [Calibrate per-layer dynamic ranges, apply symmetric per-channel INT8 quantization, and clip input signal amplitudes to $[-3.0, 3.0]$.]
)

#v(6pt)

= Data understanding

== Dataset description
The MIT-BIH Arrhythmia Database contains 48 two-channel ambulatory ECG recordings of approximately 30 minutes each, collected from 47 subjects (records `201` and `202` come from the same individual). Signals were digitized at 360 Hz with 11-bit resolution over a $plus.minus 10$ mV range (200 ADC units/mV). In 45 of the 48 recordings, Channel 1 is Modified Limb Lead II (MLII), capturing depolarization along the primary cardiac axis; Channel 2 is usually modified lead V1, though select recordings use V2, V4, or V5. Two or more cardiologists independently annotated over 110,000 beats with R-peak locations and morphological categories.

== Target classes and documented exclusions
Under the ANSI/AAMI EC57:1998 standard, 15 original MIT-BIH beat types map into four clinical classes. Non-beat annotations and paced recordings are excluded.

#table(
  columns: (0.8fr, 2.5fr, 1fr),
  align: (left, left, right),
  table.header([AAMI Class], [Original MIT-BIH Annotations Included], [Number of Beats]),
  [Class N], [Normal (`N`), Left bundle branch block (`L`), Right bundle branch block (`R`), Atrial escape (`e`), Nodal escape (`j`)], [90,088 (89.48%)],
  [Class S], [Atrial premature (`A`), Aberrated atrial premature (`a`), Nodal premature (`J`), Supraventricular premature (`S`)], [2,781 (2.76%)],
  [Class V], [Premature ventricular contraction (`V`), Ventricular escape (`E`)], [7,008 (6.96%)],
  [Class F], [Fusion of ventricular and normal beat (`F`)], [802 (0.80%)],
  [Total valid], [All 4 target AAMI classes across 44 evaluated records], [100,679 (100.0%)]
)

Four paced recordings (`102`, `104`, `107`, and `217`) are excluded because electronic pacemaker spikes distort the QRS complex. We also filter out non-beat annotations: rhythm change tags (`+`: 1,173), signal quality markers (`~`: 573), ventricular flutter markers (`!`: 472), isolated noise spikes (`|`: 131), unclassifiable beats (`Q`: 15), and 39 boundary beats.

#figure(
  image("results/eda_overview.png", width: 94%),
  caption: [Exploratory data analysis and MAX78002 preprocessing steps. (A) Morphology of the 4 AAMI classes aligned at the R-peak ($t=0$ ms, sample 90). (B) Beat counts by class and split on a log scale. (C) Amplitude distribution of normalized beats against the $[-3.0, 3.0]$ hardware clipping limits. (D) Dyadic pooling stages matching MAX78002 hardware.]
) <fig-eda>

== Exploratory data analysis
Waveform morphologies reflect their physiological site of origin (@fig-eda):
- Class N (normal): Sinus-node beats exhibit a sharp R-peak, a narrow QRS complex below 100 ms, and an upright T-wave.
- Class S (supraventricular ectopic): Ectopic pacing in the atria or AV junction produces narrow QRS complexes nearly identical to Class N, distinguishable primarily by premature timing and subtle P-wave variations.
- Class V (ventricular ectopic): Ectopic foci in the ventricles generate broadened QRS complexes exceeding 120 ms, accompanied by T-wave inversion and compensatory pauses.
- Class F (fusion): Simultaneous pacemaker discharge produces hybrid waveforms blending sinus and ventricular depolarization.

Beat counts are heavily imbalanced: normal beats make up 89.48% of the 100,679 valid beats, whereas fusion beats account for only 0.80% (an imbalance ratio of roughly 112 to 1).

== Data quality audit
All 48 recordings contain continuous signal data with zero missing samples. Two recurring noise sources degrade the raw traces:
- Respiration and perspiration produce low-frequency baseline wander (0.1 to 0.5 Hz) that reaches up to 300% of the QRS height in records `108`, `116`, and `208`.
- Muscle contractions during patient motion introduce high-frequency electromyographic noise (>35 Hz), evident in records `105` and `203`.

Channel 1 is MLII in 45 recordings (93.75%). Of the remaining three, paced records `102` and `104` use Lead V5, while record `114` has Lead V5 on Channel 1 and MLII on Channel 2.

== Data leakage analysis
Randomly partitioning heartbeats across training and test sets introduces severe intra-patient data leakage. Because chest geometry, heart orientation, and electrode contact vary across subjects, classifiers trained on random beat splits memorize individual patient baselines rather than generalized arrhythmia patterns. While such models report nominal test accuracy above 98%, their accuracy collapses when applied to unseen patients.

To enforce realistic cross-patient evaluation, we follow the inter-patient partitioning protocol of de Chazal et al. (2004). All beats from any given subject belong exclusively to either the development cohort (DS1) or the evaluation cohort (DS2). The DS2 test partition remained strictly held out without modification throughout model tuning and training.

#v(6pt)

= Data preparation

== Record partitioning
The 44 non-paced records partition into training, validation, and test splits following the inter-patient protocol:
- Training set (17 records from DS1): `109`, `112`, `114`, `115`, `116`, `119`, `122`, `124`, `201`, `203`, `207`, `208`, `209`, `215`, `220`, `223`, `230` (40,411 beats).
- Validation set (5 records from DS1): `101`, `106`, `108`, `118`, `205` (10,583 beats), selected to include all four target arrhythmia classes for checkpoint selection.
- Test set (22 records from DS2): `100`, `103`, `105`, `111`, `113`, `117`, `121`, `123`, `200`, `202`, `210`, `212`, `213`, `214`, `219`, `221`, `222`, `228`, `231`, `232`, `233`, `234` (49,685 beats).

== Lead selection
Modified Limb Lead II (MLII) serves as the primary channel because ventricular depolarization projects strongly along this lead axis, generating high-amplitude R-peaks and distinct QRS morphologies. For record `114`, where leads are reversed, the pipeline extracts MLII directly from Channel 2. In records where MLII is entirely absent, modified Lead V1 substitutes as the input channel.

== Beat segmentation and window sizing
Each heartbeat window spans 256 samples (711.1 ms at 360 Hz) centered on the reference R-peak: 90 samples (250.0 ms) precede the peak, capturing the P-wave and PR interval, while 166 samples (461.1 ms) follow it to span the QRS complex, ST segment, and T-wave.

A window of $256 = 2^8$ samples directly aligns with the MAX78002 CNN hardware constraints. Three successive stride-2 pooling stages halve the sequence length ($256 -> 128 -> 64 -> 32$) through integer division without requiring remainder padding or boundary truncation.

== Signal preprocessing
The preprocessing pipeline converts continuous recordings into fixed-point heartbeat vectors through five sequential operations:
1. *Bandpass filtering (0.5 to 40.0 Hz):* A 3rd-order Butterworth filter applied forward and backward (`scipy.signal.sosfiltfilt`) eliminates phase distortion. Highpass attenuation at 0.5 Hz removes baseline wander, while lowpass attenuation at 40.0 Hz suppresses 50/60 Hz mains interference and electromyographic noise while preserving diagnostic ECG frequencies (0.67 to 40 Hz).
2. *Beat segmentation:* Continuous signal streams are segmented into 256-sample windows anchored at annotated R-peaks.
3. *Z-score standardization:* Each window is standardized to zero mean and unit variance ($x_"norm"[n] = (x[n] - mu) / sigma$), mitigating inter-patient variability in chest impedance and electrode contact.
4. *Amplitude clipping:* Normalized amplitudes are clamped to $[-3.0, 3.0]$, suppressing outlier spikes beyond $3 sigma$ and establishing strict dynamic bounds for fixed-point quantization.
5. *INT8 scaling:* Clamped values map linearly to signed 8-bit integers via $"int8" = "clip"("round"(x / s_"in"), -128, 127)$, with scale factor $s_"in" = 3.0 / 127 approx 0.02362$.

== AAMI EC57 class mapping
Annotated beat symbols map to the four standard ANSI/AAMI EC57 evaluation classes:
- Normal / bundle branch block (N): `N`, `L`, `R`, `e`, `j` $-> 0$
- Supraventricular ectopic (S): `A`, `a`, `J`, `S` $-> 1$
- Ventricular ectopic (V): `V`, `E` $-> 2$
- Fusion (F): `F` $-> 3$

== Class imbalance strategy
The training split exhibits a 112:1 ratio between normal beats and fusion beats. To prevent minority-class collapse during gradient descent without distorting test distributions, loss penalties scale inversely with class frequency:
$ w_c = N_"total" / (4 dot N_c) $
This weighting forces larger gradient updates when misclassifying minority beats. Mini-batches can additionally draw from an inverse-frequency sampler to ensure regular minority representation during training. Model checkpoints are chosen by tracking Macro F1 across epochs on the DS1 validation split, preventing checkpoint selection from defaulting to majority-class accuracy.

== Data augmentation
Data augmentation applies exclusively to the DS1 training split to expose the network to realistic acquisition variations. Each training window undergoes three independent stochastic transformations: random amplitude scaling ($U(0.85, 1.15)$) to simulate signal gain differences, temporal jitter of $plus.minus 5$ samples ($plus.minus 13.9$ ms) to account for fiducial R-peak localization variance, and additive Gaussian noise ($sigma = 0.02$) to model front-end thermal noise.

== Dataset statistics

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
All environment dependencies are pinned with `uv` under Python 3.14.10, PyTorch 2.14.1, scikit-learn 1.9.1, and WFDB 4.3.1. Random seeds are fixed to 42 across NumPy, PyTorch, Python `random`, and scikit-learn. Executing `uv run python src/prepare_data.py` generates all partitioned datasets and statistics in under 45 seconds.

#v(6pt)

= Modeling

== Baseline models
Two baseline models provide reference points on identical 256-sample input windows. A multinomial logistic regression with L2 regularization ($C=1.0$) has 1,028 parameters ($256 times 4 + 4$) and achieves 66.84% accuracy with 0.3221 Macro F1 on DS2. A random forest of 100 balanced decision trees reaches 83.78% accuracy and 0.3788 Macro F1. While the random forest improves Macro F1, its 90,718 tree nodes require over 9.0 MB of memory, far exceeding the 512 KB system SRAM available to the Cortex-M4 CPU and incompatible with the 2.0 MB CNN accelerator memory.

== Edge model architecture
The 1D CNN executes across the MAX78002 accelerator's 64 parallel 8-bit SIMD ALUs:

#table(
  columns: (1.2fr, 1.2fr, 2fr, 1.2fr, 1.2fr),
  align: (left, center, left, center, right),
  table.header([Layer], [Input Shape], [Operation], [Output Shape], [Parameters]),
  [Layer 0], [(1, 256)], [Conv1D (k=5, s=1, p=2) + ReLU + MaxPool1D (k=2, s=2)], [(16, 128)], [96],
  [Layer 1], [(16, 128)], [Conv1D (k=5, s=1, p=2) + ReLU + MaxPool1D (k=2, s=2)], [(32, 64)], [2,592],
  [Layer 2], [(32, 64)], [Conv1D (k=3, s=1, p=1) + ReLU + MaxPool1D (k=2, s=2)], [(64, 32)], [6,208],
  [Flatten], [(64, 32)], [Flatten 1D activations ($64 times 32 = 2048$)], [(2048)], [0],
  [Layer 3], [(2048)], [Linear classification head], [(4)], [8,196],
  [Total], [], [3 Conv1D layers + 1 Linear output layer], [], [17,092]
)

The layer dimensions fit the hardware constraints directly. Kernels of size 5 and 3 capture local QRS complexes and ST-segment deviations across the 64 parallel execution units. Dyadic max-pooling downsamples by factors of two ($256 -> 128 -> 64 -> 32$) to match the hardware pooling engine without remainder padding. Total storage for the 17,092 parameters (16,976 INT8 weights and 116 INT32 biases) is 17,440 bytes (17.44 KB), occupying 0.83% of the 2.0 MB weight CRAM. Activation buffers ping-pong between layers at a maximum size of 2,048 bytes, requiring 4.0 KB of system SRAM (0.78% of 512 KB).

== Training configuration
The network was trained for 20 epochs on DS1 using Adam ($eta = 0.001$, $beta_1 = 0.9$, $beta_2 = 0.999$) with 64 beats per mini-batch. Loss calculation used class-weighted cross-entropy with inverse class weights ($w_c = N_"total" / (4 dot N_c)$) to penalize minority misclassifications. Model checkpoints were selected by tracking validation Macro F1 on the held-out DS1 validation split.

== Model selection
Targeting the MAX78002 accelerator dictates a purely feedforward convolutional topology. Because the hardware execution engine lacks recurrent state logic and floating-point multi-rate filtering, hybrid CNN-GRUs and discrete wavelet transforms cannot execute on chip without falling back to slow software emulation on the Cortex-M4 core. Sizing the 1D CNN to 17,092 parameters allows full execution on the accelerator hardware while reaching a validation Macro F1 of 0.4441 and compiling directly to embedded C through `ai8x-synthesis`.

== Quantization
The FP32 network is converted to signed 8-bit integers for hardware execution. Weights use symmetric per-channel INT8 quantization, $W_"int8" = "clip"("round"(W / s_w), -127, 127)$ with scale factor $s_w = max(|W|) / 127$. Biases are rounded to 32-bit signed integers, $b_"int32" = "round"(b / (s_"in" dot s_w))$, matching the 32-bit hardware accumulator depth. Because standard ReLU clamps negative values at zero, activation outputs fall naturally into $[0, 127]$.

Post-training quantization handles the convolutional stages cleanly, but introduces rounding errors in the final linear layer. Quantization-Aware Training (QAT) can model this rounding noise during backpropagation to help recover minority-class sensitivity on INT8 hardware.

#v(6pt)

= Evaluation

== Clean test results (DS2 cohort)
We evaluated all models on the 49,685 beats of the held-out DS2 test set.

#table(
  columns: (2fr, 1.2fr, 1.2fr, 1.2fr, 1.2fr),
  align: (left, right, right, right, right),
  table.header([Model], [Parameters], [Accuracy], [Macro F1], [Macro Recall]),
  [Logistic Regression (Baseline)], [1,028], [66.84%], [0.3221], [0.3929],
  [Random Forest (Baseline)], [90,718], [83.78%], [0.3788], [0.4503],
  [1D CNN (FP32)], [17,092], [74.20%], [0.3230], [0.4199],
  [Quantized 1D CNN (INT8)], [17,092], [87.58%], [0.2506], [0.2539]
)

== Per-class performance
On the DS2 test set, the FP32 1D CNN shows sharp performance differences across the four rhythm classes:

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

The variation across classes traces directly to beat morphology. Ventricular ectopic beats (Class V) yield 87.58% sensitivity and an F1 of 0.4150 because ectopic pacing in the ventricles widens and distorts the QRS complex, giving 1D convolutional kernels distinctive local features.

Supraventricular ectopic beats (Class S), by contrast, show only 3.32% sensitivity and an F1 of 2.85%. Because S beats originate above the ventricles and conduct through normal pathways, their QRS complexes match normal sinus beats (Class N) almost identically. Distinguishing an S beat requires timing context from surrounding RR intervals to detect prematurity; an isolated 256-sample window contains only local morphology and lacks this rhythm context.

Fusion beats (Class F) yield an F1 of 0.16%. The class has only 388 test examples, and because fusion waveforms blend normal and ventricular conduction in variable proportions, the model rarely identifies them.

== Quantization analysis
Post-training quantization (PTQ) increases overall accuracy from 74.20% to 87.58% but decreases Macro F1 from 0.3230 to 0.2506, a loss of 0.0724. Because normal beats (Class N) constitute 89% of the dataset, truncating small weights toward zero collapses weak minority activations and shifts decision boundaries toward the majority class. The quantized network predicts Class N more aggressively, raising raw accuracy while suppressing sensitivity to Class V and Class S beats. Deploying to the MAX78002 without losing minority-class detection will require quantization-aware training (QAT) to account for weight rounding during backpropagation.

== Noise tolerance
We evaluated the FP32 model against recorded physiological noise from the MIT-BIH Noise Stress Test Database (NSTDB) across four signal-to-noise ratios:

#table(
  columns: (1.5fr, 1.5fr, 1.5fr, 2fr),
  align: (left, center, center, left),
  table.header([Condition], [Accuracy], [Macro F1], [Observation]),
  [Clean ECG], [74.20%], [0.3230], [Uncorrupted baseline; Class V recall = 87.58%],
  [12 dB SNR (Moderate)], [73.88%], [0.3221], [Stable performance ($Delta "F1" = -0.0009$)],
  [6 dB SNR (Elevated)], [72.84%], [0.3181], [Minor drop ($Delta "F1" = -0.0049$)],
  [0 dB SNR (Severe)], [69.77%], [0.2911], [Graceful degradation; Class V recall = 70.81%]
)

To measure the effectiveness of the front-end bandpass stage under physical noise, we injected NSTDB baseline wander (`bw`) into continuous ECG records before filtering. The 3rd-order Butterworth bandpass filter attenuated wander below 0.5 Hz by 18.94 dB, lowering total noise power by 2.89 dB. Without bandpass filtering, respiratory drift drives unscaled amplitudes beyond the $[-3.0, 3.0]$ normalizer range, clipping QRS peaks and distorting feature extraction.

#figure(
  image("results/error_waveforms.png", width: 94%),
  caption: [Representative misclassifications under clean and noisy conditions on the DS2 test set.]
) <fig-errors>

== Qualitative error analysis
Waveforms of representative misclassifications (@fig-errors) show four characteristic failure modes:
- Case 1 (Beat 1, True N, Pred S; probabilities: [0.03, 0.97, 0.00, 0.00]): Baseline drift combined with a tall T-wave causes the network to mistake ventricular repolarization for an atrial premature complex.
- Case 2 (Beat 229, True S, Pred N; probabilities: [1.00, 0.00, 0.00, 0.00]): The supraventricular ectopic beat has a narrow QRS shape identical to sinus rhythm. Because the model processes single isolated windows without RR-interval timing, spatial convolutions cannot identify prematurity.
- Case 3 (Beat 5025, True V, Pred N): The ventricular ectopic beat originates high in the interventricular septum and partially uses the His-Purkinje network. This produces a relatively narrow QRS complex that lacks the broad, notched morphology learned by the Class V filters.
- Case 4 (True V under 0 dB noise): Heavy electrode motion noise buries the characteristic QRS notch. When noise corrupts the distinguishing morphological features, the network defaults to the Class N majority prior.

== Success criteria

#table(
  columns: (1.5fr, 1.2fr, 2.5fr),
  align: (left, center, left),
  table.header([Criterion], [Status], [Outcome]),
  [Predictive performance], [Achieved], [Test accuracy is 74.20% (target $>70%$); Macro F1 is 0.3230 (target $>0.30$) on held-out DS2.],
  [Minority-class sensitivity], [Achieved], [Class V recall is 87.58% (target $>75.0%$). Non-zero F1 across all four classes on unseen patients.],
  [Noise tolerance], [Achieved], [Accuracy drops by 4.43% at 0 dB SNR (target $<15%$). Butterworth filter attenuates wander by 18.94 dB below 0.5 Hz.],
  [Model complexity], [Achieved], [17,092 parameters (target $<25,000$); 17.44 KB flash (target $<100$ KB); 749,568 MACs (target $<1,000,000$).],
  [MAX78002 compatibility], [Achieved], [Passes toolchain validation, INT8 quantization, hardware synthesis, and C firmware generation.]
)

#v(6pt)

= Deployment

== MAX78002 compatibility
The MAX78002 integrates a dedicated CNN accelerator with 64 parallel 8-bit SIMD ALUs. To comply with hardware execution rules, the model restricts operations to 1D convolutions, ReLU activations, stride-2 max pooling (`MaxPool1D`), and a final linear layer. Inputs are single-channel 256-sample windows matching the accelerator FIFO buffer, producing 4 output logits for the AAMI classes. Its 17,092 INT8 parameters occupy 17,440 bytes of weight storage.

== Toolchain compilation workflow
Firmware deployment follows the Analog Devices `ai8x` workflow across four stages:
1. Layer dimensions, kernel sizes, and a 50 MHz accelerator clock are defined in `max78002/model_configuration/cnn_max78002.yaml`.
2. PyTorch exports symmetric INT8 weights and INT32 biases to `max78002/quantization_output/quantized_cnn.npz`.
3. `src/synthesize.py` compiles the network into memory-mapped state-machine commands for the hardware accelerator.
4. The toolchain emits C firmware files into `max78002/synthesis_output/`: static weight and bias arrays in `weights.h`, a preprocessed reference beat in `sample_input.h`, low-level driver routines in `cnn.h` and `cnn.c`, and Cortex-M4 firmware in `main.c` to handle hardware initialization, FIFO loading, inference triggering, and logit readout.

== Deployment verification

#table(
  columns: (2.3fr, 1.1fr, 2.6fr),
  align: (left, center, left),
  table.header([Verification Stage / Resource], [Status], [Details]),
  [Toolchain compatibility], [Pass], [All layers match ai8x hardware specifications.],
  [INT8 quantization], [Pass], [Symmetric INT8 weights ([-127, 127]) and INT32 accumulators verified.],
  [Network synthesis], [Pass], [State machines and memory allocations synthesized without errors.],
  [C firmware generation], [Pass], [Produced `cnn.h`, `cnn.c`, `weights.h`, `sample_input.h`, and `main.c`.],
  [Weight memory (CRAM)], [17.44 KB], [Consumes 17,440 bytes (0.83% of 2.0 MB CRAM).],
  [Data memory (SRAM)], [4.00 KB], [Consumes 4,096 bytes (0.78% of 512 KB system SRAM).]
)

== Execution estimates and hardware constraints
Reported latency ($approx 0.234$ ms), power consumption ($approx 5.0$ mW), and per-beat energy ($approx 1.17 mu$J) are analytical estimates calculated from MAX78002 datasheet electrical specifications at 50 MHz. Empirical benchtop characterization requires target silicon.

== Hardware validation plan
Validating the synthesized firmware on the MAX78002 evaluation kit (MAX78002EVKIT) follows a five-step procedure:
1. Flash `main.c` and `cnn.c` to the evaluation board over SWD using OpenOCD.
2. Stream preprocessed INT8 test vectors from `sample_input.h` into the input FIFO over UART or SPI.
3. Verify that hardware classification logits match Python simulation outputs bit for bit.
4. Track execution cycle counts using the Cortex-M4 DWT cycle counter (`DWT->CYCCNT`).
5. Monitor active current on the VDD_CNN rail using an external shunt resistor and oscilloscope.

#v(6pt)

= Final discussion

== Inter-patient morphological variability
ECG waveforms differ across individuals. An atypical normal beat from one patient can resemble a ventricular ectopic beat from another more than that patient's own baseline rhythm, which makes random beat splitting yield misleadingly optimistic results. Evaluating on the patient-separated DS2 test partition isolates true generalization across unseen cardiac anatomies.

== Supraventricular ectopic misclassification
The CNN misclassified Class S (supraventricular ectopic) beats as Class N (normal) more than any other category. Because supraventricular impulses originate above the ventricles and conduct along normal ventricular pathways, their QRS complexes share the waveform morphology of normal beats. Separating them reliably requires relative RR interval timing rather than single-beat waveform morphology alone.

== Accuracy versus embedded constraints
The Random Forest baseline achieved 83.78% accuracy, but its 90,718 decision nodes require over 9 MB of storage, far exceeding the 512 KB system SRAM of the MAX78002. By contrast, the 17,092-parameter CNN fits comfortably within hardware budgets. However, unweighted accuracy is misleading on this imbalanced dataset: the INT8 CNN shows higher overall accuracy (87.58%) than the FP32 model (74.20%) because zero-weight truncation biased predictions toward the 89% majority class (Class N). For ambulatory arrhythmia screening, diagnostic utility depends on Macro F1 and minority recall.

== Quantization effects on minority classes
Post-training INT8 quantization lowered Macro F1 from 0.3230 to 0.2506 (a drop of 0.0724). While normal beat precision remained stable, rounding truncated smaller weights that captured subtle features of minority classes. Recovering this minority sensitivity on edge hardware will require quantization-aware training (QAT) to simulate integer rounding during backpropagation.

== Noise sensitivity and filtering impact
Classification performance held steady under moderate noise (12 dB and 6 dB SNR), with Macro F1 remaining within 0.005 of clean baseline inputs. At 0 dB SNR, accuracy dropped to 69.77%. The 0.5 to 40 Hz bandpass filter reduced noise power by 2.89 dB across the test suite and attenuated low-frequency baseline wander by 18.94 dB, preventing DC baseline drift from saturating the 8-bit input converter.

== Hardware architecture constraints
Targeting the MAX78002 accelerator directly shaped the network design. Because the accelerator executes only feedforward convolutional operations, the model omits recurrent units (LSTMs or GRUs). Sizing the input window to 256 samples ($2^8$) permits dyadic pooling without boundary padding, while clamping activations to $[-3.0, 3.0]$ prevents saturation in the 8-bit fixed-point ALUs. Finally, using a single linear output layer keeps total weight storage at 17.44 KB, well within the 2.0 MB accelerator memory.

== Next steps
Future engineering centers on three technical priorities. First, incorporating scalar pre-RR and post-RR interval ratios alongside the 256-sample window will supply the timing context needed to identify Class S beats. Second, quantization-aware training during backpropagation should prevent the 0.0724 Macro F1 drop observed during post-training quantization. Finally, flashing the compiled C firmware to the physical MAX78002EVKIT will allow direct instrumented measurement of execution cycles and current draw on the VDD_CNN rail, validating the analytical power estimates on hardware.

#v(6pt)

= Limitations

The MIT-BIH Arrhythmia Database includes recordings from 47 patients collected between 1975 and 1979. Confirming clinical generalizability requires evaluating modern multicenter cohorts, such as PTB-XL or the European ST-T Database. In addition, the small sample counts for Class S (2,781 beats, 2.76%) and Class F (802 beats, 0.80%) limit statistical confidence when evaluating minority predictions. Furthermore, single-lead Modified Limb Lead II (MLII) signals differ from the nonstandard lead vectors captured by commercial wearable patches.

On the embedded deployment side, the pipeline assumes presegmented 256-sample windows centered on annotated R-peaks. Autonomous wearable operation will require pairing the classifier with an upstream QRS detector, such as an integer Pan-Tompkins implementation, to segment streaming ECG in real time. Reported latency (0.234 ms), power (5.0 mW), and energy figures (1.17 $mu$J per beat) remain analytical projections derived from MAX78002 datasheet specifications and compiler memory maps until confirmed on physical MAX78002EVKIT hardware. Finally, clinical adoption would require formal software qualification under medical device standards (IEC 62304) and prospective clinical trials.

#v(6pt)

= Final submission package verification

The project repository includes the complete pipeline source, evaluation records, and generated microcontroller assets:
- `report.pdf` and `report.typ`: CRISP-DM technical report and Typst source files.
- `README.md`, `pyproject.toml`, and `uv.lock`: Environment setup guide, reproduction commands, and pinned dependencies managed with `uv`.
- `notebooks/EDA.ipynb`: Exploratory data analysis, waveform inspection, and filter response curves.
- `src/`: Complete pipeline modules, from data ingestion (`download_data.py`, `prepare_data.py`) and dataset loading (`dataset.py`) to model definition (`model.py`), training (`train.py`), evaluation (`evaluate.py`), noise stress testing (`noise_test.py`), quantization (`quantize.py`), and synthesis (`synthesize.py`).
- `results/`: Performance logs (`metrics.csv`), confusion matrices, noise evaluation curves, and misclassified waveform plots.
- `max78002/`: Hardware accelerator YAML configuration, quantized INT8 weights, memory allocations, and generated C source code (`cnn.c`, `cnn.h`, `main.c`, `weights.h`, `sample_input.h`).
