# scripts/preprocessing.py
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

# ---------- Core building blocks ----------

def remove_dc(X):
    """Subtract per-channel mean (DC)"""
    X = np.asarray(X)
    return X - np.mean(X, axis=0, keepdims=True)

def butter_bandpass(sig, fs, low=20, high=450, order=4):
    b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, sig)

def butter_lowpass(sig, fs, cutoff, order=4):
    b, a = butter(order, cutoff/(fs/2), btype='low')
    return filtfilt(b, a, sig)

def notch_filter(sig, fs, f0=60.0, Q=30.0):
    """Single-frequency notch (e.g., 60 Hz)."""
    b, a = iirnotch(f0/(fs/2), Q)
    return filtfilt(b, a, sig)

def notch_harmonics(sig, fs, base=60.0, n_harmonics=2, Q=30.0):
    """Apply notch at base, 2*base, 3*base..."""
    y = sig
    for k in range(1, n_harmonics+1):
        y = notch_filter(y, fs, f0=base*k, Q=Q)
    return y

def rectify(X):
    """Full-wave rectification"""
    return np.abs(X)

def moving_rms(x, win_samples):
    """Channel-wise moving RMS using cumulative trick (valid-mode window)."""
    # pad to keep length same
    if win_samples <= 1:
        return np.sqrt(x**2)
    # square
    x2 = x**2
    # cumulative sum per channel
    cs = np.cumsum(x2, axis=0)
    cs_pad = np.vstack([np.zeros((1, x.shape[1])), cs])
    # windowed sum
    wsum = cs_pad[win_samples:] - cs_pad[:-win_samples]
    rms = np.sqrt(wsum / win_samples)
    # pad front to keep same length
    pad = np.zeros((win_samples-1, x.shape[1]))
    return np.vstack([pad, rms])

# ---------- High-level pipelines ----------

def apply_filters(X, fs, band_low=20, band_high=450, notch_base=60, notch_Q=30, notch_harmonics_count=0):
    """
    Steps 1–3: DC removal -> bandpass -> (optional) notch & harmonics.
    Returns the cleaned (but not rectified/smoothed) signal.
    """
    X = np.asarray(X)
    Y = np.zeros_like(X)

    # 1) DC removal
    X_dc = remove_dc(X)

    # Per-channel bandpass + notch
    for ch in range(X.shape[1]):
        y = butter_bandpass(X_dc[:, ch], fs, band_low, band_high)
        if notch_base:
            if notch_harmonics_count and notch_harmonics_count > 0:
                y = notch_harmonics(y, fs, base=notch_base, n_harmonics=notch_harmonics_count, Q=notch_Q)
            else:
                y = notch_filter(y, fs, f0=notch_base, Q=notch_Q)
        Y[:, ch] = y
    return Y

def apply_full_pipeline(
    X, fs,
    band_low=20, band_high=450,
    notch_base=60, notch_Q=30, notch_harmonics_count=1,
    rectify_signal=True,
    smoothing="lowpass",   # "lowpass" or "rms"
    smooth_cutoff_hz=5,    # for lowpass envelope
    rms_window_ms=200      # for moving RMS envelope
):
    """
    Complete 5-step pipeline:
    1) Remove DC
    2) Band-pass (low/high)
    3) Notch (base + optional harmonics)
    4) Rectify
    5) Smooth (low-pass envelope or moving RMS)
    """
    # Steps 1–3
    Y = apply_filters(
        X, fs,
        band_low=band_low, band_high=band_high,
        notch_base=notch_base, notch_Q=notch_Q,
        notch_harmonics_count=notch_harmonics_count
    )

    # Step 4: Rectify
    if rectify_signal:
        Y = rectify(Y)

    # Step 5: Smooth
    if smoothing == "lowpass":
        for ch in range(Y.shape[1]):
            Y[:, ch] = butter_lowpass(Y[:, ch], fs, smooth_cutoff_hz)
    elif smoothing == "rms":
        win = max(1, int((rms_window_ms/1000.0) * fs))
        Y = moving_rms(Y, win)
    else:
        # no smoothing
        pass

    return Y
