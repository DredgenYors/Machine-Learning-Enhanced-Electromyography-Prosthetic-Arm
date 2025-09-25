import os, sys
import numpy as np
import scipy.io
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# import your reusable filters
sys.path.append(os.getcwd())
from scripts.preprocessing import apply_full_pipeline  # DC -> bandpass -> notch -> rectify -> smooth

# ---------- Config ----------
FS = 1000                 # <-- set to your dataset's sampling rate
SMOOTH_MODE = "rms"       # "rms" or "lowpass"
RMS_WIN_MS = 200          # if smoothing="rms"
LOWPASS_CUTOFF = 5        # if smoothing="lowpass"
BAND_LOW, BAND_HIGH = 20, 450
NOTCH_BASE, NOTCH_Q = 60, 30
NOTCH_HARMONICS = 1       # also notch 120 Hz

WIN_MS = 250              # feature window length (ms)
STEP_MS = 125             # hop length (ms)

# ---------- Helpers ----------
def window_indices(n, fs, win_ms=250, step_ms=125):
    win = int(win_ms * fs / 1000)
    step = int(step_ms * fs / 1000)
    idx = []
    for i in range(0, n - win + 1, step):
        idx.append((i, i + win))
    return idx, win, step

def extract_features(X_win):
    """
    X_win: (win_samples, n_channels)
    Returns features concatenated across channels:
      [RMS, MAV, WL, ZC, SSC] per channel
    """
    feats = []
    for ch in range(X_win.shape[1]):
        w = X_win[:, ch]
        rms = np.sqrt(np.mean(w**2))
        mav = np.mean(np.abs(w))
        wl  = np.sum(np.abs(np.diff(w)))
        zc  = int(np.sum((w[:-1] * w[1:]) < 0))
        d   = np.diff(w)
        ssc = int(np.sum((d[:-1] * d[1:]) < 0))
        feats.extend([rms, mav, wl, zc, ssc])
    return np.array(feats, dtype=float)

def window_features(X, y, fs, win_ms=250, step_ms=125):
    """
    X: (n_samples, n_channels) cleaned/envelope signal
    y: (n_samples,) per-sample labels
    Returns: X_feat (n_windows, n_feat), y_win (n_windows,)
    """
    idx_list, _, _ = window_indices(len(X), fs, win_ms, step_ms)
    X_feat = []
    y_win = []
    for (i0, i1) in idx_list:
        X_feat.append(extract_features(X[i0:i1, :]))
        # majority label in the window
        if y is not None:
            vals, counts = np.unique(y[i0:i1], return_counts=True)
            y_win.append(vals[np.argmax(counts)])
    X_feat = np.vstack(X_feat)
    y_win = np.array(y_win) if y is not None else None
    return X_feat, y_win

# ---------- Load data ----------
Tk().withdraw()
mat_file = askopenfilename(title="Select a MATLAB .mat file", filetypes=[("MAT files", "*.mat")])
if not mat_file:
    raise SystemExit("No file selected.")

mat = scipy.io.loadmat(mat_file)
if 'emg' not in mat:
    raise ValueError("Expected variable 'emg' in the .mat file.")
emg = np.squeeze(mat['emg'])
if emg.ndim == 1:
    emg = emg[:, None]

labels = None
if 'stimulus' in mat:
    labels = np.squeeze(mat['stimulus']).astype(int)

# ---------- Filtering + envelope (full 5-step pipeline) ----------
envelope = apply_full_pipeline(
    emg, FS,
    band_low=BAND_LOW, band_high=BAND_HIGH,
    notch_base=NOTCH_BASE, notch_Q=NOTCH_Q, notch_harmonics_count=NOTCH_HARMONICS,
    rectify_signal=True,
    smoothing=SMOOTH_MODE,
    smooth_cutoff_hz=LOWPASS_CUTOFF,
    rms_window_ms=RMS_WIN_MS
)

# ---------- Features ----------
X_feat, y_win = window_features(envelope, labels, FS, win_ms=WIN_MS, step_ms=STEP_MS)
print("Feature matrix:", X_feat.shape)
if y_win is None:
    raise ValueError("No 'stimulus' labels found in .mat; cannot train.")

# ---------- Train/test ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_feat, y_win, test_size=0.2, random_state=42, stratify=y_win
)

clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
