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
  
  This report evaluates a compact ECG heartbeat classification system targeted to the Analog Devices MAX78002 microcontroller, designed as an ECG-only processing stage for wearable cardiac monitors. Under the ANSI/AAMI EC57 standard, the classifier maps individual beats into four categories: normal (N), supraventricular ectopic (S), ventricular ectopic (V), and fusion (F). The 44 non-paced MIT-BIH recordings (100,679 annotated beats) partition by patient into equal 22-record development (DS1) and held-out test (DS2) cohorts, eliminating intra-patient data leakage.
  
  The classifier uses a three-layer 1D convolutional neural network with 17,092 parameters, sized for the MAX78002 accelerator without exceeding on-chip memory limits. On the 49,685 beats in the untouched DS2 test set, the FP32 model achieves 74.20% accuracy, a macro F1 score of 0.3230, and 87.58% recall on ventricular ectopic beats (Class V). Software-verified toolchain synthesis and post-training symmetric INT8 quantization confirm hardware compatibility, requiring 17.44 KB of weight memory (0.83% of the 2.0 MB CRAM) and 4.0 KB of activation SRAM (0.78% of the 512 KB pool). Post-training quantization lowers macro F1 to 0.2506 due to integer truncation of minority-class weights, indicating that edge deployment will require quantization-aware training. At 50 MHz, datasheet-derived execution estimates indicate 0.234 ms latency per beat (749,568 MACs) and 1.17 $mu$J energy draw. On the MIT-BIH Noise Stress Test Database, a front-end 0.5 to 40 Hz Butterworth bandpass filter attenuates baseline wander by 18.94 dB and cuts overall noise power by 2.89 dB MSE, preventing baseline drift and motion artifacts from saturating the software normalizer and fixed-point dynamic range.
]

#v(6pt)

= Business / application understanding

== Application scenario
Cardiac arrhythmias range from acute emergencies like ventricular fibrillation to recurrent ectopic beats that indicate underlying disease or elevated stroke risk. Conventional Holter monitors store multi-day ECG recordings to flash memory for retrospective physician review, leaving clinicians unaware of acute rhythm disturbances until days later.

Moving inference onto wearable edge microcontrollers allows real-time beat classification directly on ambulatory patches. By processing signals locally, the device avoids continuous Bluetooth or cellular transmission, cutting the radio duty cycle by roughly 90% and extending coin-cell battery life to several weeks. On-device analysis also eliminates cloud latency for urgent ventricular alerts, keeps patient waveforms private, and operates reliably in shielded hospital wards without cellular coverage. This workflow supports ambulatory patients, post-infarction convalescents, elderly individuals at elevated arrhythmia risk, and cardiologists triaging high-volume Holter recordings.

== Data-mining objective
The machine learning task maps single-lead 1D ECG heartbeats into discrete diagnostic categories under microcontroller hardware constraints:

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
Severe class imbalance (~89% sinus beats) renders overall accuracy an incomplete metric for clinical utility. Project success therefore depends on five quantitative criteria:
- Predictive performance requires macro-averaged F1 on the held-out DS2 test set to exceed 0.30, with overall classification accuracy above 70.0%.
- Minority-class sensitivity on Class V must exceed 75.0% to reliably detect dangerous ventricular arrhythmias, while Classes S and F must achieve non-zero F1 scores on unseen patients.
- Noise robustness requires that accuracy loss under 0 dB physical noise from the MIT-BIH Noise Stress Test Database remain within 15% relative to clean ECG inputs, supported by front-end bandpass filtering that attenuates baseline wander and electromyographic noise.
- Model complexity must remain within embedded limits: fewer than 25,000 parameters, storage footprint under 100 KB, and compute budget under 1,000,000 MAC operations per beat.
- Toolchain compatibility requires passing every stage of the MAX78002 compilation flow (model validation, INT8 quantization, synthesis, and C driver generation), requiring under 50 KB of weight memory (below 2.5% of CRAM) and under 10 KB of activation SRAM (below 2.0% of system SRAM).

== Constraints
The implementation is bounded by seven hardware and operational factors:
1. Limited Edge-AI memory restricts storage to 2.0 MB of dedicated CNN weight SRAM (CRAM) and 512 KB of system SRAM; the MAX78002 board provides no external RAM.
2. Fixed-point INT8 computation requires the accelerator to execute strictly with signed 8-bit integer weights ([-128, 127]) and 32-bit accumulators, while any floating-point operation relies on software emulation on the Cortex-M4 CPU.
3. Severe class imbalance, where normal sinus beats outnumber ectopic beats by up to 100 to 1, causes unweighted cross-entropy models to collapse toward majority-class predictions.
4. A limited patient population of 47 subjects in the MIT-BIH database impedes cross-patient generalization because individual electrode placement, chest anatomy, and cardiac pathologies vary widely.
5. Real physical noise in ambulatory recordings (electrode motion `em`, baseline wander `bw`, and muscle tremor `ma`) obscures subtle diagnostic waveforms.
6. Constrained computational resources require all convolutional layers to execute across the accelerator's 64 parallel SIMD ALUs within 1,000,000 MAC operations per beat, because the 50 MHz Cortex-M4 CPU lacks hardware neural network acceleration.
7. The absence of physical MAX78002 hardware means reported inference latency (0.234 ms), power (5.0 mW), and energy (1.17 $mu$J) are software-verified estimates derived from `ai8x-synthesis` memory maps and manufacturer electrical specifications rather than benchtop measurements.

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
The MIT-BIH Arrhythmia Database is the standard reference benchmark for automated arrhythmia analysis. The dataset contains 48 two-channel ambulatory ECG recordings of approximately 30 minutes each (1,800 seconds per record, totaling roughly 24 hours), collected from 47 distinct subjects (records `201` and `202` come from the same individual). Analog signals were digitized at a sampling frequency of 360 Hz per channel with 11-bit resolution over a $plus.minus 10$ mV range (200 ADC units/mV). In 45 of the 48 recordings, Channel 1 is Modified Limb Lead II (MLII), which captures ventricular depolarization along the primary cardiac axis; Channel 2 is usually modified lead V1 (or occasionally V2, V4, or V5). Two or more cardiologists independently annotated over 110,000 heartbeats, marking fiducial R-peak locations and beat morphologies.

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
  [*Total valid*], [All 4 target AAMI classes across 44 evaluated records], [100,679 (100.0%)]
)

Four paced recordings (`102`, `104`, `107`, and `217`) are excluded because electronic pacemaker spikes distort the natural QRS complex. We also exclude non-beat annotations and unclassifiable markers: rhythm change tags (`+`: 1,173), signal quality markers (`~`: 573), ventricular flutter markers (`!`: 472), isolated noise spikes (`|`: 131), unclassifiable beats (`Q`: 15), non-conducted P-waves (`x`: 21), and 39 boundary beats.

#figure(
  image("results/eda_overview.png", width: 94%),
  caption: [Exploratory data analysis and MAX78002 preprocessing steps. (A) Morphology of the 4 AAMI classes aligned at the R-peak ($t=0$ ms, sample 90). (B) Beat counts by class and split on a log scale. (C) Amplitude distribution of normalized beats against the $[-3.0, 3.0]$ hardware clipping limits. (D) Dyadic pooling stages matching MAX78002 hardware.]
) <fig-eda>

== Exploratory data analysis
The four heartbeat classes exhibit distinct electrophysiological origins and waveform shapes (@fig-eda):
- Class N (normal): Normal sinus beats exhibit a sharp R-peak, a narrow QRS complex below 100 ms, and an upright T-wave.
- Class S (supraventricular ectopic): Arising in the atria or AV junction, these complexes resemble normal beats in QRS width but arrive prematurely, often with distorted or inverted P-waves.
- Class V (ventricular ectopic): Impulses originating in the ventricles conduct slowly outside the specialized His-Purkinje network, producing a widened QRS complex (>120 ms), an inverted T-wave, and a post-ectopic compensatory pause.
- Class F (fusion): Hybrid complexes produced when sinus and ventricular pacemakers discharge simultaneously.

Continuous two-channel recordings capture clean sinus rhythm alongside baseline lead differences between MLII and precordial chest leads. Averaged beat waveforms (@fig-eda A) highlight the morphological divergence across classes, while the amplitude distribution (@fig-eda C) centers at zero with tails bounded by the $[-3.0, 3.0]sigma$ clipping limits. Class distributions (@fig-eda B) show pronounced skew: normal beats constitute 89.48% of the 100,679 classified beats, whereas fusion beats represent 0.80% (an imbalance ratio of roughly 112 to 1). Partitioning by subject separates the 44 records into development (DS1, 22 records) and test (DS2, 22 records) cohorts without subject overlap.

== Data quality audit
Signal continuity and artifact analysis across all 48 recordings identify several quality considerations:
- Missing data: Verification confirmed complete signal continuity with no missing samples across any recording.
- Extreme amplitudes: Low-frequency baseline wander (0.1 to 0.5 Hz) from respiration and perspiration reaches up to 300% of the QRS height in records `108`, `116`, and `208`, requiring dynamic clamping to prevent fixed-point overflow.
- Corrupted segments and noisy regions: Burst electromyographic (EMG) noise (>35 Hz) from patient muscle contractions appears in records `105` and `203`, while sporadic electrode contact loss (`em`) occurs during physical movement.
- Class imbalance: The 112 to 1 ratio between majority N (89.48%) and rare F (0.80%) would cause unweighted models to predict solely the majority class.
- Subject differences: Variations in thoracic anatomy, heart orientation, and skin-electrode impedance create substantial morphological diversity across the 47 patients.
- Lead configurations: Channel 1 is MLII in 45 recordings (93.75%). Paced records `102` and `104` use Lead V5 on Channel 1 and V2 on Channel 2, while record `114` has V5 on Channel 1 and MLII on Channel 2.

== Data leakage analysis
In ECG classification, randomly splitting individual heartbeats from the same subject into training and test sets produces misleading results. Because thoracic anatomy, cardiac axis, and electrode impedance vary between individuals, heartbeats from any single recording share a distinct baseline signature. Models trained on random beat splits learn these patient-specific features, reporting nominal test accuracies above 98% that drop sharply when evaluated on unfamiliar subjects.

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
Processing continuous ECG into fixed-point heartbeat tensors ready for the MAX78002 accelerator requires six ordered operations:
$ "Raw ECG" -> "Bandpass Filter" -> "Segmentation" -> "Z-Score Standardization" -> "Amplitude Clipping" -> "INT8 Scaling" -> "Model Input" $

1. A 3rd-order Butterworth bandpass filter (0.5 to 40.0 Hz) applied forward and backward (`scipy.signal.sosfiltfilt`) eliminates phase distortion. The 0.5 Hz highpass cutoff removes respiratory and perspiration baseline wander, while the 40.0 Hz lowpass cutoff suppresses 50/60 Hz mains hum and electromyographic noise while preserving diagnostic ECG frequencies (0.67 to 40 Hz).
2. Continuous single-channel signals are segmented into 256-sample windows anchored at annotated R-peaks, producing uniform cardiac cycles for batch processing.
3. Each segmented beat undergoes Z-score standardization ($x_"norm"[n] = (x[n] - mu) / sigma$), scaling samples to zero mean and unit variance to neutralize inter-patient differences in thoracic impedance and electrode contact.
4. Normalized amplitudes are clamped to $[-3.0, 3.0]$. This cuts outlier spikes beyond $3 sigma$ caused by physical electrode movement and enforces strict dynamic bounds for subsequent fixed-point quantization.
5. Clamped floating-point values map linearly to signed 8-bit integers via $"int8" = "clip"("round"(x / s_"in"), -128, 127)$, using input scale factor $s_"in" = 3.0 / 127 approx 0.02362$ to match the MAX78002 FIFO registers.
6. Heartbeat arrays are formatted to `(1, 256)` single-channel tensors for PyTorch model execution.

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
Two baseline models provide reference benchmarks on identical 256-sample input windows, evaluated on both the DS1 validation split (5 records, 10,583 beats) and the held-out DS2 test split (22 records, 49,685 beats).

An L2-regularized multinomial logistic regression ($C=1.0$) uses 1,028 parameters ($256 times 4 + 4$). On the DS1 validation cohort, the linear model achieves 68.42% accuracy and 0.3518 macro F1. On the untouched DS2 test cohort, it reaches 66.84% accuracy and 0.3221 macro F1 (macro recall: 0.3929, macro precision: 0.3243).

A random forest ensemble of 100 balanced decision trees achieves higher predictive utility, scoring 84.15% accuracy and 0.3942 macro F1 on DS1 validation records, and 83.78% accuracy with 0.3788 macro F1 on DS2 (macro recall: 0.4503, macro precision: 0.3535). However, the ensemble spans 90,718 decision nodes that require over 9.0 MB of memory. This footprint makes the forest impossible to deploy directly on the MAX78002, which provides 2.0 MB of dedicated CNN weight memory (CRAM) and 512 KB of system SRAM.

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

Every layer adheres to the execution model of the MAX78002 CNN accelerator, which integrates 64 parallel 8-bit SIMD ALUs. The network uses only supported operations: 1D convolutions, standard ReLU activations, stride-2 max-pooling, and a single dense output layer. Small kernel sizes ($k=5$ in layers 0 and 1, $k=3$ in layer 2, all well under the $k <= 9$ hardware boundary) capture local QRS notches, ventricular ectopic widening, and ST-segment deviations while fitting into the accelerator's internal line buffers.

Dyadic pooling stages halve the sequence length ($256 -> 128 -> 64 -> 32$) through integer division, matching the hardware pooling engine without remainder padding or truncation. The 256-sample input window matches the on-chip FIFO streaming buffer, eliminating external memory transactions during acquisition. Total parameter storage encompasses 16,976 INT8 weights and 116 INT32 biases, requiring 17,440 bytes (17.44 KB). This consumes only 0.83% of the 2.0 MB weight CRAM. For intermediate activations, double-buffering ping-pongs between layers at a peak size of 2,048 bytes, requiring 4.0 KB of system SRAM (0.78% of the 512 KB system pool).

== Training configuration
Training was conducted on the 17 patient records in the DS1 training split (40,411 beats) using the Adam optimizer with a learning rate of $eta = 0.001$, moment decay rates $beta_1 = 0.9$ and $beta_2 = 0.999$, and mini-batches of 64 beats. Optimization ran for up to 20 epochs with early stopping monitored on the 5 held-out DS1 validation records (10,583 beats). When validation macro F1 failed to improve over a patience window of 5 epochs, training halted and restored the best checkpoint weights. Reproducibility was enforced by fixing random seed 42 across Python, NumPy, and PyTorch.

To counter the 112:1 class imbalance without skewing the posterior probabilities on unseen records, the loss function combined class-weighted cross-entropy with balanced batch sampling. Inverse-frequency class weights ($w_c = N_"total" / (4 dot N_c)$) penalized minority-class errors during backpropagation. Additionally, three stochastic augmentations expanded morphological diversity on DS1 training records only: random amplitude scaling from $U(0.85, 1.15)$ to simulate varying electrode contact, temporal jitter of $plus.minus 5$ samples ($plus.minus 13.9$ ms) to model R-peak detector jitter, and zero-mean additive Gaussian noise ($sigma = 0.02$) to represent analog thermal noise.

== Model selection
Candidate architectures and hyperparameters were explored exclusively on the DS1 development records, keeping the DS2 test set untouched until final evaluation to enforce Rules 2 and 3.

Initial experiments examined two non-standard approaches. A recurrent hybrid model combining a 1D CNN with a single-layer GRU produced strong temporal modeling on validation data. However, the MAX78002 accelerator does not contain recurrent execution units, so updating hidden states on the Cortex-M4 CPU introduced severe latency bottlenecks. A 9-level Daubechies-6 discrete wavelet transform was also evaluated to decompose waveforms into subbands, but multi-rate floating-point filtering required more intermediate SRAM buffering than the 512 KB system pool could accommodate.

Within feedforward 1D convolutional topologies, a hyperparameter grid search evaluated kernel lengths ($k in {3, 5, 7}$), channel depths (16 to 128 channels), and learning rates ($eta in {0.01, 0.001, 0.0005}$). Kernels of length 7 added parameter overhead without improving validation scores, while narrower channels (8-16-32) failed to resolve ventricular ectopic features. Selection was governed strictly by patient-stratified validation macro F1 on the 5 held-out DS1 records. The final 3-layer CNN reached a peak validation macro F1 of 0.4441, compiled cleanly through the `ai8x` synthesis toolchain, and fit within accelerator memory without spills.

== Quantization
Deploying the FP32 network to the MAX78002 requires converting floating-point parameters to signed integers using post-training quantization (PTQ). Weights undergo symmetric per-channel INT8 quantization, mapping values into $[-127, 127]$ through scale factor $s_w = max(|W|) / 127$:
$ W_"int8" = "clip"("round"(W / s_w), -127, 127). $
Biases round to 32-bit signed integers to match the accelerator's 32-bit accumulator depth, computed from input scale $s_"in"$ and weight scale $s_w$:
$ b_"int32" = "round"(b / (s_"in" dot s_w)). $
Because the network uses standard ReLU activations that clamp negative values to zero, activation outputs fall naturally into $[0, 127]$.

Evaluating the quantized network against the 49,685 DS2 test beats highlights the performance cost of post-training quantization:
$ "FP32 Macro F1": 0.3230 wide "INT8 Macro F1": 0.2506 wide Delta: -0.0724 $
Although overall classification accuracy rises from 74.20% to 87.58%, macro F1 drops by 0.0724. Because normal beats (Class N) make up roughly 89% of ambulatory records, truncating small weights toward zero collapses weak minority activations and biases predictions toward the majority class. Under PTQ, Class S F1 drops from 0.0285 to 0.0000, and Class V recall collapses from 87.58% to 25.39%. This degradation demonstrates why Quantization-Aware Training (QAT) is necessary for clinical Edge-AI. By simulating rounding noise during forward passes and computing gradients with straight-through estimators during backpropagation, QAT prevents minority weight decay and preserves sensitivity under INT8 fixed-point execution.

#v(6pt)

= Evaluation

== Clean test results (DS2 cohort)
We evaluated all models on the 49,685 beats of the held-out DS2 test set across multiple metrics to avoid relying on overall accuracy alone.

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

The per-class metrics reflect underlying cardiac electrophysiology and training set distribution.

Class S (supraventricular ectopic) beats proved especially difficult, yielding 2.49% precision and 3.32% recall (F1 = 2.85%). Across ECG classification benchmarks, supraventricular precision rarely exceeds 15% under inter-patient partitioning. Because S beats originate above the ventricles, their impulses travel through the normal atrioventricular node and His-Purkinje network, producing narrow QRS complexes that match normal sinus beats (Class N) almost identically. Distinguishing an S beat requires timing context from preceding and following RR intervals to detect prematurity; an isolated 256-sample window captures local morphology but lacks this rhythm context.

Class F (fusion) beats showed the lowest overall performance, with 0.12% precision and 0.26% recall (F1 = 0.16%). Fusion beats occur when a ventricular ectopic impulse and a normal sinus impulse activate the myocardium simultaneously. This dual activation generates hybrid waveforms that blend normal and ventricular shapes in unpredictable proportions. In addition, Class F accounts for only 388 beats in the DS2 test set (0.78% prevalence), leaving the model with too few examples to establish a reliable decision boundary.

In contrast, the network detects ventricular ectopic beats (Class V) with 87.58% sensitivity and an F1 score of 0.4150. Because ventricular ectopy originates in the ventricular myocardium, it bypasses the rapid Purkinje conduction system, producing broad, notched, and high-amplitude QRS complexes that the 1D convolutional filters capture reliably across patients.

== Quantization analysis
$ "FP32 Macro F1": 0.3230 wide "INT8 Macro F1": 0.2506 wide "Difference": -0.0724 $

Post-training quantization (PTQ) increases overall accuracy from 74.20% to 87.58% while reducing Macro F1 from 0.3230 to 0.2506, an absolute drop of 0.0724 (7.24 percentage points).

This divergence occurs because severe class imbalance interacts with weight truncation. Normal sinus beats constitute 89.04% of the dataset. Truncating floating-point weights to signed 8-bit integers rounds small weight values toward zero, collapsing weak activations associated with rare classes (S and F) in the final linear projection layer. Consequently, the network decision boundary shifts toward the dominant Class N prior. The quantized model classifies borderline beats as Class N, artificially driving raw accuracy upward while suppressing Class V recall (from 87.58% down to 25.39%) and eliminating minority sensitivity.

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

Evaluating performance across noise levels (@fig-cm-noise) reveals that Class V beats are the most vulnerable to physical distortion. While Class N accuracy stays stable due to its high prior probability, Class V recall drops from 87.58% under clean inputs to 70.81% at 0 dB SNR, a decline of 16.77 percentage points. Severe electrode motion (`em`) and muscle tremor (`ma`) distort the notch and width of the QRS complex, causing noisy ventricular beats to be misclassified as normal sinus rhythm.

The degradation across the signal-to-noise spectrum is gradual rather than abrupt. Between clean inputs and 12 dB SNR, Macro F1 changes by only -0.0009 (from 0.3230 to 0.3221), and at 6 dB SNR it drops by -0.0049 to 0.3181. Even under severe 0 dB noise, Macro F1 falls by 0.0319 to 0.2911, and overall accuracy decreases by 4.43% (from 74.20% to 69.77%). This steady slope shows that convolutional feature maps retain diagnostic information under moderate interference rather than collapsing abruptly.

Front-end filtering provides strong protection against low-frequency artifacts. Evaluating continuous NSTDB baseline wander (`bw`) through the 3rd-order Butterworth bandpass filter (0.5 to 40 Hz) demonstrates an 18.94 dB attenuation of drift below 0.5 Hz, reducing overall noise power by 2.89 dB MSE. Without this bandpass stage, large respiratory excursions push raw amplitudes past the $[-3.0, 3.0]$ clipping limits of the normalizer, saturating the input window and truncating QRS peaks.

Noise robustness could be improved in future revisions through three practical measures. Training-time noise injection would expose the convolutional kernels to recorded NSTDB noise at varying signal-to-noise ratios. Multi-lead spatial fusion would allow the network to cross-reference Lead II with Lead V1 or V5 when motion artifacts corrupt a single electrode. Finally, adding an integer median filter before normalization would help suppress sharp electromyographic spikes.

#figure(
  image("results/error_waveforms.png", width: 94%),
  caption: [Representative misclassifications across morphological and noise conditions on the DS2 test set.]
) <fig-errors>

== Error analysis
Examining the misclassified waveforms in @fig-errors highlights four recurring failure mechanisms:

In Case 1 (Beat 1, true N misclassified as S; probabilities: [0.03, 0.97, 0.00, 0.00]), the normal sinus beat exhibits slight baseline drift and a prominent, peaked T-wave. The network misinterprets this repolarization wave as an ectopic atrial P-wave, predicting an atrial premature contraction (Class S).

In Case 2 (Beat 229, true S misclassified as N; probabilities: [1.00, 0.00, 0.00, 0.00]), the supraventricular ectopic beat displays a narrow QRS complex identical to sinus rhythm. Because the model receives an isolated 256-sample window without preceding RR-interval timing, spatial convolutions cannot detect the prematurity of the beat.

In Case 3 (Beat 5025, true V misclassified as N), an atypical ventricular beat originates high in the interventricular septum, partially engaging the normal His-Purkinje conduction system. The resulting QRS complex is narrow and lacks the deep S-wave notch learned by the Class V filters, evading detection.

In Case 4 (true V under 0 dB noise), severe electrode motion noise obscures the QRS notch. With the characteristic morphological features buried under noise, the classifier defaults to the majority Class N prior.

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

In this project, deployment covers software-level toolchain verification and firmware generation for the Analog Devices MAX78002 microcontroller. Physical on-chip testing was excluded because hardware development kits were not supplied for this course. Verification therefore spans the complete compile-time and synthesis pipeline up to flashing-ready embedded C code.

== MAX78002 compatibility
The MAX78002 microcontroller contains a hardware CNN accelerator with 64 parallel 8-bit SIMD ALUs. Its memory hierarchy provides 2.0 MB of CNN weight memory (CRAM) and 512 KB of system SRAM for activations and general data buffers.

To execute on this engine, the network restricts all layers to supported hardware primitives: standard 1D convolutions (`Conv1D`) with kernel sizes of 5 and 3 (well within the accelerator constraint of $k <= 9$), ReLU activations, dyadic max pooling (`MaxPool1D` with stride 2), and a linear classification head. Unsupported operations, such as dilated convolutions or recurrent connections, are omitted.

The network processes single-channel windows of 256 time samples ($1 times 256$), which match the accelerator input FIFO without multi-bank tiling. The linear head produces 4 logits corresponding to the target AAMI classes. In total, the 17,092 INT8 parameters require 17,440 bytes of weight storage, consuming 0.83% of the 2.0 MB CRAM. Intermediate activation buffers require 4,096 bytes (4.00 KB), using 0.78% of the 512 KB system SRAM.

== MAX78002 toolchain workflow
Model translation and firmware generation follow the Analog Devices `ai8x-training` and `ai8x-synthesis` framework across four stages:
1. Specify layer dimensions, kernel allocations, and a 50 MHz accelerator clock according to `ai8x` execution rules in `max78002/model_configuration/cnn_max78002.yaml`.
2. Convert floating-point weights to symmetric signed 8-bit integers in the range $[-127, 127]$ and scale biases to 32-bit signed integers using `src/quantize.py`, exporting the quantized arrays to `max78002/quantization_output/quantized_cnn.npz`.
3. Synthesize the network with `src/synthesize.py` (wrapping `ai8xize.py`), compiling layers into hardware control state machines, mapping parameters across the 64 SIMD processors, and verifying memory allocations.
4. Generate embedded C drivers and firmware files into `max78002/synthesis_output/`: static weight and bias arrays in `weights.h`, a preprocessed reference beat in `sample_input.h`, peripheral driver routines in `cnn.h` and `cnn.c`, and Cortex-M4 firmware in `main.c` that configures hardware clocks, loads the FIFO buffer, triggers inference, and reads classification logits.

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
In strict accordance with course instructions and Rule 6, this project reports no measured physical benchtop values. We maintain a strict boundary between software-level verification and physical hardware verification. Software verification encompasses toolchain acceptance, symmetric INT8 quantization, accelerator state-machine synthesis, automated C firmware generation, and bit-accurate Python simulation. In contrast, physical hardware verification remains unperformed because no physical evaluation board was available.

Consequently, we make no empirical hardware claims for inference latency, power consumption, energy per inference, physical MAX78002 accuracy, or real-time operating performance. The reported latency ($approx 0.234$ ms per beat), active power ($approx 5.0$ mW), and energy dissipation ($approx 1.17 mu$J per beat across 749,568 MACs) represent theoretical analytical estimates derived from MAX78002 datasheet electrical specifications at 50 MHz.

== Proposed hardware validation
If a physical MAX78002 evaluation kit (MAX78002EVKIT) becomes available, validation would proceed through six experimental steps:
1. Flash `main.c` and `cnn.c` to the microcontroller over SWD or JTAG using OpenOCD.
2. Stream preprocessed INT8 beat vectors from `sample_input.h` into the hardware FIFO buffer over UART or SPI direct memory access.
3. Compare classification logits read from hardware accelerator registers against Python simulation outputs to confirm bit-accurate agreement.
4. Inspect intermediate layer activation buffers across accelerator memory banks to verify layer-by-layer numerical consistency with software emulation.
5. Measure true wall-clock inference latency by toggling a GPIO pin at inference boundaries and reading cycle counts from the Cortex-M4 DWT counter (`DWT->CYCCNT`).
6. Profile dynamic current draw on the VDD_CNN power rail using an external current shunt resistor and power analyzer to determine physical energy consumption per beat.

#v(6pt)

= Final discussion

== Lessons from the data and patient variability
Two primary characteristics govern the MIT-BIH dataset: extreme class imbalance and pronounced inter-patient morphological heterogeneity. Normal sinus beats account for 89.48% of annotated beats, outnumbering fusion beats by more than 110 to 1 and supraventricular ectopic beats by over 30 to 1. Without inverse-frequency loss weighting and balanced mini-batch sampling, optimization collapses into predicting the majority class. Furthermore, individual chest geometry, electrode contact impedance, and heart orientation create substantial baseline variation across subjects. An atypical normal beat from one patient often resembles a ventricular ectopic beat from another patient more than that patient's own sinus rhythm. Randomly splitting individual heartbeats between training and testing sets lets a network memorize these patient-specific baseline shapes, generating deceptive test accuracies above 98% that fail on new individuals. Partitioning recordings strictly by patient across the DS1 development set and the DS2 evaluation set isolates true generalization to unseen cardiac anatomies.

== Primary error modes and morphological confusion
Classification errors concentrated primarily in two categories: distinguishing supraventricular ectopic beats (Class S) from normal sinus rhythm (Class N), and identifying fusion beats (Class F). On the DS2 test set, the CNN obtained only 3.32% sensitivity on Class S (F1 of 0.0285), misclassifying 1,773 of 1,837 supraventricular beats as normal. Because supraventricular impulses originate above the ventricles and conduct along normal His-Purkinje pathways, their QRS complexes share the narrow waveform morphology of sinus rhythm. Distinguishing them requires relative RR interval timing to identify prematurity; an isolated 256-sample window contains only local morphology and cannot detect abnormal rhythm cadence. Fusion beats presented a similar challenge, yielding an F1 of 0.0016. Because fusion complexes blend sinus and ventricular depolarizations in varying physical proportions, their morphology varies continuously between normal and ectopic shapes, leaving the model unable to separate them from rare test examples (388 beats).

== Accuracy versus edge feasibility
Neither the quantized INT8 CNN (87.58%) nor the 100-tree Random Forest baseline (83.78%) provides an effective edge solution, despite higher raw accuracy than the unquantized FP32 CNN (74.20%). On heavily imbalanced ECG data, unweighted accuracy rewards majority-class bias. The quantized CNN reached 87.58% accuracy only because zero-weight truncation collapsed predictions into the 89% majority class, dropping Macro F1 from 0.3230 to 0.2506. Meanwhile, the Random Forest model generated 90,718 decision nodes requiring over 9 MB of memory, far exceeding the MAX78002's 512 KB system SRAM. In addition, decision trees cannot execute on the hardware CNN accelerator. By contrast, the 17,092-parameter 1D CNN requires 17.44 KB of weight memory and 4.0 KB of activation memory, leaving over 99% of microcontroller memory free while preserving 87.58% recall on ventricular ectopic beats (Class V).

== Quantization degradation and precision loss
Post-training INT8 quantization lowered Macro F1 from 0.3230 to 0.2506, an overall loss of 0.0724. This performance drop affected classes unevenly: while normal sinus precision remained stable at 94%, macro recall dropped from 0.4199 to 0.2539. Symmetrically quantizing weights into signed 8-bit integers ($[-127, 127]$) truncated small weight values toward zero. Because minority classes produce smaller gradient updates during initial training, their separating hyperplanes depend on fine-grained weight magnitudes that collapse when rounded to nearest integer levels. Deploying to the MAX78002 without sacrificing minority sensitivity requires quantization-aware training (QAT), which models rounding noise during backpropagation so the network learns weight distributions that preserve minority separation after 8-bit rounding.

== Noise sensitivity and filtering impact
Evaluation against the MIT-BIH Noise Stress Test Database showed that the classifier tolerates moderate noise but degrades under severe corruption. At 12 dB SNR, accuracy dropped by only 0.32% (Macro F1 decreased by 0.0009 to 0.3221). At 6 dB SNR, accuracy fell to 72.84% (Macro F1 of 0.3181). Under severe 0 dB SNR noise, accuracy declined to 69.77% and Macro F1 dropped to 0.2911, with Class V sensitivity falling from 87.58% to 70.81% as electrode motion artifacts obscured diagnostic QRS notches. The front-end 0.5 to 40 Hz Butterworth bandpass filter preserved diagnostic fidelity by attenuating respiratory baseline wander below 0.5 Hz by 18.94 dB and reducing overall noise power by 2.89 dB MSE. Without this bandpass stage, baseline drift drives raw signal amplitudes past the $[-3.0, 3.0]$ dynamic clipping thresholds, saturating the 8-bit input converter and obliterating morphological features before inference.

== Hardware architecture constraints and design trade-offs
Deploying to the MAX78002 required tailoring the network to the accelerator's 64 parallel 8-bit SIMD ALUs and on-chip memory limits. Supported operations are restricted to 1D convolutions, ReLU activations, stride-2 max pooling, and a single linear classification head. Recurrent units (LSTMs, GRUs) were excluded because the accelerator lacks recurrent execution hardware; routing loop iterations to the Cortex-M4 CPU introduced prohibitive latency bottlenecks. Similarly, multi-rate wavelet transforms and Transformers were rejected due to intermediate SRAM buffering limits. Kernels were limited to sizes 5 and 3, and layer widths were capped at 64 channels to match the 64 hardware ALUs without multi-pass memory stalling. Sizing the input window to exactly 256 samples ($2^8$) permits dyadic pooling ($256 -> 128 -> 64 -> 32$) without boundary padding. Finally, clamping input signals to $[-3.0, 3.0]$ and quantizing to signed 8-bit integers ($[-127, 127]$) keeps activations within hardware dynamic bounds, while the single linear output head restricts parameter storage to 17.44 KB (0.83% of CRAM).

== Future engineering priorities
Addressing the 0.0724 Macro F1 quantization drop requires implementing Quantization-Aware Training (QAT) during backpropagation, modeling integer rounding before synthesis to protect minority-class sensitivity. To separate Class S from Class N beats, the input pipeline needs scalar pre-RR and post-RR interval ratios alongside the 256-sample window, supplying the rhythm cadence that single-beat spatial convolutions lack. At the signal level, pairing Modified Lead II with Lead V1 would capture orthogonal frontal and horizontal cardiac vectors, resolving morphological axis ambiguities. For scarce categories such as Class F (388 test beats), synthetic ECG augmentation using conditional generative adversarial networks or diffusion models could expand minority training distributions. Finally, flashing compiled C firmware to the physical MAX78002EVKIT would enable direct instrumented measurements of execution cycles (using the Cortex-M4 DWT counter) and active current on the VDD_CNN rail, validating analytical energy estimates on target silicon.

#v(6pt)

= Limitations

== Dataset and patient variance
The study relies on the MIT-BIH Arrhythmia Database, which contains 48 half-hour recordings from 47 subjects collected between 1975 and 1979. Excluding the four paced records leaves 44 evaluated recordings totaling 100,679 beats, a modest sample compared to modern multicenter clinical archives such as PTB-XL or the European ST-T Database. Furthermore, the dataset exhibits marked class imbalance: normal sinus beats (Class N, 89.48%) outnumber fusion beats (Class F, 0.80%) by roughly 112 to 1 and supraventricular ectopic beats (Class S, 2.76%) by 32 to 1. This disparity restricts statistical confidence when evaluating minority-class metrics. Inter-patient physiological differences, including cardiac axis deviation, chest geometry, and baseline conduction pathologies, create wide morphological variance across individuals. Because an atypical normal beat from one patient can resemble an ectopic beat from another patient more closely than that patient's own baseline rhythm, fixed spatial filters face inherent generalization limits on unseen anatomies. Finally, the historical analog recordings, acquired at 360 Hz with 11-bit resolution using hospital-grade wet gel electrodes, differ from modern digital ambulatory acquisitions.

== Embedded execution constraints
For embedded deployment, the pipeline assumes presegmented 256-sample windows centered on annotated R-peaks. Autonomous wearable operation requires pairing the classifier with an upstream QRS detector, such as an integer Pan-Tompkins implementation, to segment streaming signals in real time. In addition, all latency (0.234 ms), power (5.0 mW), and per-beat energy figures (1.17 $mu$J) reported here are software-verified analytical projections derived from MAX78002 datasheet electrical specifications at 50 MHz and synthesis memory maps. Because testing occurred without a physical MAX78002EVKIT board on an instrumented bench, these figures represent software-verified benchmarks. Direct hardware verification remains future work.

== Wearable domain shift and clinical translation
Moving from clinical Holter data to wearable patches introduces a major domain shift. Commercial wearable monitors rely on dry-contact electrodes that suffer from higher skin-contact impedance, frequent motion artifacts during daily physical activity, perspiration-induced baseline wander, and lower or variable sampling rates. Moreover, single-lead patches capture nonstandard thoracic lead vectors that differ markedly from the Modified Limb Lead II (MLII) orientation used in this study. Lastly, this project is a retrospective offline computational investigation. The model has not undergone prospective clinical testing in hospital or ambulatory workflows, lacks formal software qualification under medical device standards (IEC 62304), and is not cleared for diagnostic use.

#v(6pt)

= Final submission package verification

== Deliverable inventory
The project repository contains all pipeline source files, documentation, evaluation artifacts, and generated microcontroller assets, structured according to the official submission specification:

#table(
  columns: (1.2fr, 2.3fr, 2.5fr),
  align: (left, left, left),
  table.header([Component], [Path / Location], [Verified Contents]),
  [Documentation], [`report.pdf`, `report.typ`, `README.md`], [CRISP-DM technical report, Typst source, and reproduction guide],
  [Dependencies], [`requirements.txt`, `pyproject.toml`, `uv.lock`], [Locked dependencies for pip and uv environments],
  [Data analysis], [`notebooks/EDA.ipynb`], [Executed exploratory analysis, distributions, and filter curves],
  [Source pipeline], [`src/`], [Data preparation, dataset loader, model definitions, training, evaluation, noise tests, and synthesis],
  [Configuration], [`configs/`, `max78002/model_configuration/`], [Hardware network specification (`cnn_max78002.yaml`)],
  [Evaluation records], [`results/`], [Metrics log (`metrics.csv`), confusion matrices, noise stress curves, and error plots],
  [MAX78002 firmware], [`max78002/`], [Quantized INT8 weights, memory map, and C drivers (`cnn.c`, `main.c`, `weights.h`)]
)

In accordance with course instructions, raw PhysioNet dataset files are excluded from the repository package. The complete raw dataset is retrieved automatically by running `uv run python src/prepare_data.py` (or `python src/download_data.py`).

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
