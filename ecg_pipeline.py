"""
Signal processing for the ECG Streamlit app (numpy / scipy only).

Mirrors the training notebook exactly: same band-pass filter, per-record
normalisation, 216-sample window around the R-peak and the same 4 RR features.
The only difference is that R-peaks are DETECTED here (the notebook used the
expert annotations).
"""
from fractions import Fraction

import numpy as np
from scipy.signal import butter, find_peaks, resample_poly, sosfiltfilt

FS = 360
BEFORE, AFTER = 72, 144
SEG_LEN = BEFORE + AFTER          # 216
MIN_SECONDS = 10
MAX_SECONDS = 120


def to_360hz(sig, fs):
    """Resample to 360 Hz (the rate the model was trained on)."""
    if abs(fs - FS) < 1e-6:
        return np.asarray(sig, dtype=np.float64)
    frac = Fraction(FS / fs).limit_denominator(1000)
    return resample_poly(sig, frac.numerator, frac.denominator)


def bandpass(sig, fs=FS, low=0.5, high=40.0):
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)


def normalize_record(sig):
    sig = (sig - np.median(sig)) / (np.std(sig) + 1e-8)
    return np.clip(sig, -10, 10)


def refine_peaks(sig, peaks, radius=int(0.1 * FS), min_dist=int(0.2 * FS)):
    """Move each detected peak to the strongest deflection nearby, drop near-duplicates."""
    moved = []
    for p in np.asarray(peaks, dtype=int):
        a, b = max(0, p - radius), min(len(sig), p + radius + 1)
        seg = sig[a:b]
        moved.append(a + int(np.argmax(np.abs(seg - np.median(seg)))))
    moved = np.unique(moved)
    keep = []
    for p in moved:
        if keep and p - keep[-1] < min_dist:
            if abs(sig[p]) > abs(sig[keep[-1]]):
                keep[-1] = p
        else:
            keep.append(p)
    return np.asarray(keep, dtype=int)


def detect_rpeaks_builtin(sig, fs=FS):
    """Simple Pan-Tompkins-style detector (band-pass, derivative, square, integrate)."""
    sos = butter(2, [5, 15], btype="bandpass", fs=fs, output="sos")
    f = sosfiltfilt(sos, sig)
    energy = np.gradient(f) ** 2
    win = max(1, int(0.12 * fs))
    integ = np.convolve(energy, np.ones(win) / win, mode="same")
    cand, _ = find_peaks(integ, distance=int(0.2 * fs))
    if len(cand) == 0:
        return np.array([], dtype=int)
    amps = integ[cand]
    typical = np.median(amps[amps >= np.median(amps)])
    peaks = cand[amps >= 0.25 * typical]
    return refine_peaks(sig, peaks, radius=int(0.1 * fs), min_dist=int(0.2 * fs))


def detect_rpeaks(sig_norm, method="auto"):
    """Returns (peaks, method_used). 'auto' tries WFDB's XQRS first, then the built-in detector."""
    if method == "auto":
        try:
            from wfdb.processing import xqrs_detect
            peaks = np.asarray(xqrs_detect(sig=sig_norm, fs=FS, verbose=False), dtype=int)
            if len(peaks) >= 3:
                return refine_peaks(sig_norm, peaks), "WFDB XQRS"
        except Exception:
            pass
    return detect_rpeaks_builtin(sig_norm), "built-in"


def beats_from_peaks(sig_norm, peaks, fs=FS):
    """
    Same logic as the notebook's extract_beats, but starting from R-peak positions.
    Returns X (n, 216), RR (n, 4) raw features, beat_peaks (n,) sample positions.
    The first and last beat are skipped (they lack a previous / next beat).
    """
    peaks = np.asarray(peaks, dtype=int)
    n = len(peaks)
    empty = (np.empty((0, SEG_LEN), np.float32), np.empty((0, 4), np.float32), np.empty(0, int))
    if n < 3:
        return empty
    iv = np.diff(peaks) / fs
    X, RR, kept = [], [], []
    for i in range(1, n - 1):
        s = peaks[i]
        a, b = s - BEFORE, s + AFTER
        if a < 0 or b > len(sig_norm):
            continue
        pre, post = iv[i - 1], iv[i]
        local = iv[max(0, i - 10):i].mean()
        X.append(sig_norm[a:b])
        RR.append([np.clip(pre, 0.2, 3.0), np.clip(post, 0.2, 3.0),
                   np.clip(pre / local, 0.0, 5.0), np.clip(post / local, 0.0, 5.0)])
        kept.append(s)
    if not X:
        return empty
    return (np.asarray(X, np.float32), np.asarray(RR, np.float32), np.asarray(kept, dtype=int))


def process_recording(sig, fs, method="auto"):
    """Raw single-lead signal -> normalised signal, R-peaks and model-ready beats."""
    sig = np.nan_to_num(np.asarray(sig, dtype=np.float64))
    if fs < 100:
        raise ValueError("Sampling rate must be at least 100 Hz.")
    sig = sig[: int(MAX_SECONDS * fs)]
    sig = to_360hz(sig, fs)
    if len(sig) < MIN_SECONDS * FS:
        raise ValueError(f"The recording is too short: at least {MIN_SECONDS} seconds are needed "
                         f"(got {len(sig) / FS:.1f} s).")
    if np.std(sig) < 1e-8:
        raise ValueError("The signal is flat (no variation).")
    norm = normalize_record(bandpass(sig))
    peaks, used = detect_rpeaks(norm, method)
    X, RR, beat_peaks = beats_from_peaks(norm, peaks)
    return {"signal": norm, "peaks": peaks, "X": X, "RR": RR,
            "beat_peaks": beat_peaks, "method": used, "duration": len(norm) / FS}
