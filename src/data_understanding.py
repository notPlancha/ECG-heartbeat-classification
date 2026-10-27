"""
data_understanding.py
=====================
Core utility module for CRISP-DM Stage 2: Data Understanding.

Provides clean, user-friendly, and reproducible functions for:
1. MIT-BIH Arrhythmia Database record loading and header inspection.
2. AAMI EC57 standard 4-class heartbeat mapping (N, S, V, F) and exclusion handling.
3. Lead/channel extraction with automatic fallback and channel-swap detection (e.g. record 114).
4. Zero-phase Butterworth bandpass filtering for baseline wander and noise suppression.
5. Heartbeat segmentation with fixed-size Edge-AI compatible windows (256 samples).
6. Signal quality, amplitude, and data leakage analysis across subjects.

Author: Course Instructor Solution
Course: ECG Heartbeat Classification on Edge-AI
"""

import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import wfdb


# ==============================================================================
# Standard Dataset Partitioning Constants (AAMI / de Chazal et al., 2004)
# ==============================================================================

# DS1: Development set (22 records used for training, validation, and hyperparameter selection)
DS1_RECORDS = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
]

# Recommended patient-stratified split of DS1 into Train and Validation:
# Validation set (5 records): chosen to ensure representation of all 4 classes
DS1_VAL_RECORDS = ["101", "106", "108", "118", "205"]

# Training set (17 records): remainder of DS1
DS1_TRAIN_RECORDS = [r for r in DS1_RECORDS if r not in DS1_VAL_RECORDS]

# DS2: Final test set (22 records held out strictly for final evaluation)
DS2_TEST_RECORDS = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
]

# Paced records: excluded according to standard AAMI recommendations
PACED_RECORDS = ["102", "104", "107", "217"]

# All 48 records in the MIT-BIH Arrhythmia Database
ALL_RECORDS = sorted(DS1_RECORDS + DS2_TEST_RECORDS + PACED_RECORDS)

# Default absolut path to dataset in the workspace
DEFAULT_DATA_DIR = os.path.join("dataset/mit-bih-arrhythmia-database-1.0.0")


def resolve_data_dir(data_dir: str = DEFAULT_DATA_DIR) -> str:
    """
    Resolve data directory path using a relative path where possible.
    Works whether invoked from the project root or subdirectories (e.g. notebooks/).
    """
    if os.path.exists(data_dir):
        return data_dir
    # Check relative path from a subdirectory like notebooks/
    rel_parent = os.path.join("..", data_dir)
    if os.path.exists(rel_parent):
        return rel_parent
    # Fallback to module directory relative path
    module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_candidate = os.path.join(module_dir, data_dir)
    if os.path.exists(abs_candidate):
        try:
            return os.path.relpath(abs_candidate)
        except ValueError:
            return abs_candidate
    return data_dir


# ==============================================================================
# AAMI EC57 4-Class Mapping Dictionary
# ==============================================================================

# Mapping from original MIT-BIH annotation symbol to AAMI 4-class (N, S, V, F)
# According to ANSI/AAMI EC57:1998 standard recommendations:
# Class N: Normal / bundle branch block beats
# Class S: Supraventricular ectopic beats
# Class V: Ventricular ectopic beats
# Class F: Fusion beats
# Non-beat markers and unclassifiable beats are excluded and logged.

AAMI_MAPPING: Dict[str, Tuple[str, str]] = {
    # Class N: Normal / Bundle Branch Block Beats
    "N": ("N", "Normal beat"),
    "L": ("N", "Left bundle branch block beat"),
    "R": ("N", "Right bundle branch block beat"),
    "e": ("N", "Atrial escape beat"),
    "j": ("N", "Nodal (junctional) escape beat"),

    # Class S: Supraventricular Ectopic Beats (SVEB)
    "A": ("S", "Atrial premature beat"),
    "a": ("S", "Aberrated atrial premature beat"),
    "J": ("S", "Nodal (junctional) premature beat"),
    "S": ("S", "Supraventricular premature beat"),

    # Class V: Ventricular Ectopic Beats (VEB)
    "V": ("V", "Premature ventricular contraction"),
    "E": ("V", "Ventricular escape beat"),

    # Class F: Fusion Beats
    "F": ("F", "Fusion of ventricular and normal beat"),
}

# Excluded symbols: rare paced, unclassifiable beats, and rhythm/signal annotations
EXCLUDED_ANNOTATIONS: Dict[str, str] = {
    "/": "Paced beat (excluded: artificial pacemaker morphology)",
    "f": "Fusion of paced and normal beat (excluded: pacemaker artifact)",
    "Q": "Unclassifiable beat (excluded: unknown morphology)",
    "x": "Non-conducted P-wave (excluded: blocked atrial conduction)",
    "+": "Rhythm change marker (non-beat annotation)",
    "~": "Signal quality / noise marker (non-beat annotation)",
    "!": "Ventricular flutter wave (non-beat rhythm marker)",
    "|": "Isolated QRS-like artifact (non-beat annotation)",
    "[": "Measurement / rhythm start mark (non-beat annotation)",
    "]": "Measurement / rhythm end mark (non-beat annotation)",
    '"': "Comment / measurement mark (non-beat annotation)",
}

CLASS_NAMES = ["N", "S", "V", "F"]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASS_NAMES)}


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class RecordData:
    """Structured container holding a single MIT-BIH record's data."""
    record_name: str
    signal: np.ndarray             # shape: (samples, channels), in mV
    fs: int                        # sampling frequency (360 Hz)
    sig_names: List[str]          # lead names, e.g. ['MLII', 'V1']
    units: List[str]              # e.g. ['mV', 'mV']
    comments: List[str]           # clinical patient info, medications
    ann_indices: np.ndarray       # sample index of each annotation
    ann_symbols: List[str]        # original MIT-BIH annotation symbols


# ==============================================================================
# Record Loading & Channel Management
# ==============================================================================

def load_record(
    record_name: str,
    data_dir: str = DEFAULT_DATA_DIR,
) -> RecordData:
    """
    Load a raw MIT-BIH record (header, signal, and annotations) using WFDB.

    Parameters
    ----------
    record_name : str
        The 3-digit record name, e.g. '100' or '208'.
    data_dir : str
        Directory containing the WFDB database files.

    Returns
    -------
    RecordData
        Structured dataclass containing signal, annotations, and metadata.
    """
    data_dir = resolve_data_dir(data_dir)
    record_path = os.path.join(data_dir, str(record_name))
    
    # Read record signal and header
    record = wfdb.rdrecord(record_path)
    # Read annotations (using standard 'atr' annotator)
    annotation = wfdb.rdann(record_path, "atr")

    return RecordData(
        record_name=str(record_name),
        signal=record.p_signal,
        fs=record.fs,
        sig_names=record.sig_name,
        units=record.units,
        comments=record.comments,
        ann_indices=np.array(annotation.sample),
        ann_symbols=list(annotation.symbol),
    )


def extract_lead_signal(
    record_data: RecordData,
    preferred_lead: str = "MLII",
) -> Tuple[np.ndarray, str, int]:
    """
    Extract a single lead signal from the record.

    In the MIT-BIH Arrhythmia Database:
    - 40 records have lead pair ('MLII', 'V1') with MLII at index 0.
    - 3 records have ('MLII', 'V5'), ('MLII', 'V2'), ('MLII', 'V4') with MLII at index 0.
    - Record 114 has ('V5', 'MLII'), so MLII is at channel index 1!
    - Records 102 and 104 have ('V5', 'V2') with no MLII (both are paced and excluded).

    This function automatically resolves the correct channel index.

    Parameters
    ----------
    record_data : RecordData
        The loaded record.
    preferred_lead : str
        The preferred lead name, default is 'MLII'.

    Returns
    -------
    lead_signal : np.ndarray
        1D numpy array of signal values in mV.
    lead_name : str
        The actual lead extracted.
    lead_index : int
        The channel index used (0 or 1).
    """
    sig_names = [name.upper() for name in record_data.sig_names]
    target = preferred_lead.upper()

    if target in sig_names:
        channel_idx = sig_names.index(target)
        actual_lead = record_data.sig_names[channel_idx]
    else:
        # Fallback to the first available channel (e.g. for paced records lacking MLII)
        channel_idx = 0
        actual_lead = record_data.sig_names[channel_idx]

    return record_data.signal[:, channel_idx], actual_lead, channel_idx


# ==============================================================================
# Signal Preprocessing: Filtering
# ==============================================================================

def butter_bandpass_filter(
    signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 40.0,
    fs: float = 360.0,
    order: int = 3,
) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter to remove baseline wander
    and high-frequency noise (EMG, 50/60 Hz powerline).

    Parameters
    ----------
    signal : np.ndarray
        Raw 1D ECG signal.
    lowcut : float
        Lower cutoff frequency in Hz (typically 0.5 Hz for baseline removal).
    highcut : float
        Upper cutoff frequency in Hz (typically 40.0 Hz to suppress noise).
    fs : float
        Sampling frequency in Hz (360 Hz for MIT-BIH).
    order : int
        Filter order (default 3, giving 18 dB/octave rolloff).

    Returns
    -------
    np.ndarray
        Zero-phase filtered signal.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Clip to valid digital frequency range [0, 1)
    low = max(0.001, min(low, 0.99))
    high = max(low + 0.001, min(high, 0.99))

    b, a = butter(order, [low, high], btype="bandpass")
    # Use zero-phase digital filtering (filtfilt) to prevent phase distortion of QRS
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal.astype(np.float32)


# ==============================================================================
# Beat Extraction & Window Segmentation
# ==============================================================================

def segment_heartbeats(
    signal: np.ndarray,
    ann_indices: np.ndarray,
    ann_symbols: List[str],
    record_name: str,
    pre_r: int = 90,
    post_r: int = 166,
    normalize: bool = True,
    clip_range: Optional[Tuple[float, float]] = (-3.0, 3.0),
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, Counter]:
    """
    Extract fixed-size heartbeat windows centered around annotated R-peaks.

    At fs = 360 Hz:
    - pre_r = 90 samples = 250 ms (captures P-wave and PR interval)
    - post_r = 166 samples = 461 ms (captures QRS complex, ST segment, T-wave)
    - Total window length = 90 + 166 = 256 samples (711 ms, exactly 2^8 for MAX78002)

    Parameters
    ----------
    signal : np.ndarray
        Filtered 1D ECG signal.
    ann_indices : np.ndarray
        Sample indices of annotated beats.
    ann_symbols : List[str]
        MIT-BIH annotation symbols corresponding to ann_indices.
    record_name : str
        Record identifier for tracking.
    pre_r : int
        Number of samples before the R-peak (default 90).
    post_r : int
        Number of samples after the R-peak (default 166).
    normalize : bool
        Whether to apply per-beat Z-score normalization.
    clip_range : tuple of (float, float), optional
        Range to clip normalized values (e.g. (-3.0, 3.0)) to suppress artifacts.

    Returns
    -------
    windows : np.ndarray
        Array of shape (N_beats, window_size), float32.
    labels : np.ndarray
        Array of integer class indices (0: N, 1: S, 2: V, 3: F).
    symbols : List[str]
        Original MIT-BIH symbols for each extracted beat.
    r_peaks : np.ndarray
        Original sample indices of the R-peaks in the recording.
    excluded_counts : Counter
        Counts of annotations excluded during extraction.
    """
    window_size = pre_r + post_r
    sig_len = len(signal)

    extracted_windows = []
    extracted_labels = []
    extracted_symbols = []
    extracted_r_peaks = []
    excluded_counts: Counter = Counter()

    for r_idx, sym in zip(ann_indices, ann_symbols):
        # 1. Check if annotation maps to one of the 4 AAMI classes
        if sym not in AAMI_MAPPING:
            excluded_counts[sym] += 1
            continue

        target_class, _ = AAMI_MAPPING[sym]
        class_idx = CLASS_TO_IDX[target_class]

        # 2. Check signal boundaries
        start = r_idx - pre_r
        end = r_idx + post_r

        if start < 0 or end > sig_len:
            excluded_counts["BOUNDARY_SKIP"] += 1
            continue

        beat_window = signal[start:end].copy()

        # 3. Per-beat Z-score normalization
        if normalize:
            mean = np.mean(beat_window)
            std = np.std(beat_window)
            if std > 1e-6:
                beat_window = (beat_window - mean) / std
            else:
                beat_window = beat_window - mean

        # 4. Outlier clipping
        if clip_range is not None:
            beat_window = np.clip(beat_window, clip_range[0], clip_range[1])

        extracted_windows.append(beat_window)
        extracted_labels.append(class_idx)
        extracted_symbols.append(sym)
        extracted_r_peaks.append(r_idx)

    if len(extracted_windows) > 0:
        windows_arr = np.array(extracted_windows, dtype=np.float32)
        labels_arr = np.array(extracted_labels, dtype=np.int64)
        r_peaks_arr = np.array(extracted_r_peaks, dtype=np.int64)
    else:
        windows_arr = np.empty((0, window_size), dtype=np.float32)
        labels_arr = np.empty((0,), dtype=np.int64)
        r_peaks_arr = np.empty((0,), dtype=np.int64)

    return windows_arr, labels_arr, extracted_symbols, r_peaks_arr, excluded_counts


# ==============================================================================
# Dataset Summary & Inspection Utilities
# ==============================================================================

def get_dataset_metadata_table(
    data_dir: str = DEFAULT_DATA_DIR,
    records: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compile a complete metadata summary table across all records in the database.

    Reports:
    - Record ID
    - Patient / subject comments
    - Channels available (e.g. ['MLII', 'V1'])
    - Preferred lead selected
    - Channel index of preferred lead
    - Duration in minutes
    - Sampling frequency
    - Total annotations
    - Benchmark split (DS1 Train, DS1 Val, DS2 Test, or Paced Excluded)

    Parameters
    ----------
    data_dir : str
        Path to database directory.
    records : List[str], optional
        List of records to inspect. Defaults to all 48 records.

    Returns
    -------
    pd.DataFrame
        Formatted pandas DataFrame with metadata.
    """
    if records is None:
        records = ALL_RECORDS

    rows = []
    for r in records:
        rec = load_record(r, data_dir=data_dir)
        _, lead_name, lead_idx = extract_lead_signal(rec, preferred_lead="MLII")

        # Determine partition tag
        if r in PACED_RECORDS:
            split_tag = "Paced (Excluded)"
        elif r in DS1_VAL_RECORDS:
            split_tag = "DS1 - Validation"
        elif r in DS1_TRAIN_RECORDS:
            split_tag = "DS1 - Train"
        elif r in DS2_TEST_RECORDS:
            split_tag = "DS2 - Final Test"
        else:
            split_tag = "Other"

        duration_min = round(rec.signal.shape[0] / rec.fs / 60.0, 2)
        meds = "; ".join(rec.comments) if rec.comments else "None"

        rows.append({
            "Record": r,
            "Split": split_tag,
            "Leads Available": ", ".join(rec.sig_names),
            "Selected Lead": lead_name,
            "Lead Channel Index": lead_idx,
            "Sampling Rate (Hz)": rec.fs,
            "Duration (min)": duration_min,
            "Total Annotations": len(rec.ann_symbols),
            "Clinical Notes": meds,
        })

    return pd.DataFrame(rows)


def get_class_breakdown_table(
    data_dir: str = DEFAULT_DATA_DIR,
    records: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Calculate the exact counts of each target class (N, S, V, F) and excluded
    annotations for each record and aggregate across splits.

    Parameters
    ----------
    data_dir : str
        Path to dataset directory.
    records : List[str], optional
        Records to process.

    Returns
    -------
    pd.DataFrame
        Table with per-record counts for N, S, V, F, Total Classified, and Excluded.
    """
    if records is None:
        records = ALL_RECORDS

    rows = []
    for r in records:
        rec = load_record(r, data_dir=data_dir)
        c: Counter = Counter()
        ex_count = 0

        for sym in rec.ann_symbols:
            if sym in AAMI_MAPPING:
                c[AAMI_MAPPING[sym][0]] += 1
            else:
                ex_count += 1

        total_classified = sum(c.values())
        total_all = total_classified + ex_count

        rows.append({
            "Record": r,
            "N": c["N"],
            "S": c["S"],
            "V": c["V"],
            "F": c["F"],
            "Total Classified": total_classified,
            "Excluded": ex_count,
            "Total Annotations": total_all,
        })

    df = pd.DataFrame(rows)
    return df


def get_target_classes_table() -> pd.DataFrame:
    """
    Generate the formal CRISP-DM Target Classes table (Section 2.2).
    """
    rows = [
        {
            "Final Class": "N",
            "Meaning": "Normal-type beats",
            "Original Annotations Included": "N (Normal), L (LBBB), R (RBBB), e (Atrial escape), j (Nodal escape)",
            "Clinical Significance": "Dominant healthy or conduction-delayed ventricular activations without premature ectopic origin.",
        },
        {
            "Final Class": "S",
            "Meaning": "Supraventricular ectopic beats (SVEB)",
            "Original Annotations Included": "A (Atrial premature), a (Aberrated atrial premature), J (Nodal premature), S (Supraventricular premature)",
            "Clinical Significance": "Beats originating from atria or AV junction above the ventricles; markers of atrial fibrillation risk.",
        },
        {
            "Final Class": "V",
            "Meaning": "Ventricular ectopic beats (VEB)",
            "Original Annotations Included": "V (Premature ventricular contraction), E (Ventricular escape)",
            "Clinical Significance": "Beats originating within ventricular myocardium; frequent PVCs indicate risk of dangerous arrhythmias.",
        },
        {
            "Final Class": "F",
            "Meaning": "Fusion beats",
            "Original Annotations Included": "F (Fusion of ventricular and normal beat)",
            "Clinical Significance": "Hybrid morphology caused by simultaneous ventricular depolarization from ectopic and sinus foci.",
        },
    ]
    return pd.DataFrame(rows)


def assess_signal_quality(
    signal: np.ndarray,
    fs: float = 360.0,
) -> Dict[str, Union[float, int, bool]]:
    """
    Investigate data quality indicators for an ECG signal.

    Checks:
    - Missing samples (NaN or Inf values).
    - Signal dynamic range (min, max, peak-to-peak amplitude).
    - Extreme amplitudes (> 5.0 mV or < -5.0 mV, indicating motion artifacts/leads off).
    - Baseline drift estimate (amplitude of low-frequency wander).
    - Flatline / saturation detection (regions where derivative is zero).

    Parameters
    ----------
    signal : np.ndarray
        1D ECG signal array.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    dict
        Quality assessment metrics.
    """
    has_nan = bool(np.isnan(signal).any())
    has_inf = bool(np.isinf(signal).any())
    min_val = float(np.min(signal))
    max_val = float(np.max(signal))
    ptp_val = max_val - min_val

    # Detect extreme spikes outside physiological bounds (-5 mV to +5 mV)
    extreme_outliers = int(np.sum((signal > 5.0) | (signal < -5.0)))
    outlier_pct = float((extreme_outliers / len(signal)) * 100)

    # Estimate baseline wander via low-pass median or low-order filter
    wander_b, wander_a = butter(2, 0.5 / (0.5 * fs), btype="lowpass")
    baseline = filtfilt(wander_b, wander_a, signal)
    wander_range = float(np.max(baseline) - np.min(baseline))

    # Flatline detection: consecutive constant values for > 50 ms (18 samples at 360 Hz)
    diffs = np.diff(signal)
    flat_segments = int(np.sum(diffs == 0))

    return {
        "has_nan": has_nan,
        "has_inf": has_inf,
        "min_amplitude_mv": round(min_val, 3),
        "max_amplitude_mv": round(max_val, 3),
        "peak_to_peak_mv": round(ptp_val, 3),
        "extreme_outlier_count": extreme_outliers,
        "extreme_outlier_pct": round(outlier_pct, 4),
        "baseline_wander_range_mv": round(wander_range, 3),
        "flatline_samples": flat_segments,
    }