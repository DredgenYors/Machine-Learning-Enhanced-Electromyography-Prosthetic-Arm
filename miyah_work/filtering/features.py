# filtering/features.py
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from typing import Dict, Any
from .pipeline_state import PipelineState

def rms(x): return np.sqrt(np.mean(x**2))
def mav(x): return np.mean(np.abs(x))
def wl(x):  return np.sum(np.abs(np.diff(x)))

def zc(x, thr=0.01):
    cnt = 0
    for i in range(len(x)-1):
        if abs(x[i]) > thr and abs(x[i+1]) > thr and x[i]*x[i+1] < 0:
            cnt += 1
    return cnt

def ssc(x, thr=0.01):
    cnt = 0
    for i in range(1, len(x)-1):
        d1 = x[i] - x[i-1]
        d2 = x[i+1] - x[i]
        if (d1*d2 < 0) and (abs(d1) > thr or abs(d2) > thr):
            cnt += 1
    return cnt

def extract_windowed_features_optimal(
    state: PipelineState, window_size: int=None, overlap: float=None,
    zc_threshold: float=0.01, ssc_threshold: float=0.01
) -> Dict[str, Any]:

    if state.original_signal is None:
        raise ValueError("No signal loaded.")

    if window_size is None: window_size = state.window_size
    if overlap is None: overlap = state.overlap
    step = int(window_size * (1 - overlap))

    chans = state.selected_channels if state.selected_channels is not None else list(range(state.num_channels))
    orig = state.original_signal[:, chans]
    n, C = orig.shape

    rectified_detected = (np.min(orig) >= 0) and (np.mean(orig < 0) < 0.01)

    if rectified_detected:
        pre_rect = np.zeros_like(orig)
        b, a = iirnotch(50, 30, state.fs)
        post = filtfilt(b, a, orig, axis=0)
        post = post - np.mean(post, axis=0)
        nyq = 0.5 * state.fs
        wc = min(6.0/nyq, 0.99)
        b, a = butter(4, wc, btype='low')
        post = filtfilt(b, a, post, axis=0)
    else:
        nyq = 0.5*state.fs
        low = 20.0/nyq; high = min(500.0/nyq, 0.99)
        b, a = butter(4, [low, high], btype='band')
        pre_rect = filtfilt(b, a, orig, axis=0)
        b, a = iirnotch(50, 30, state.fs)
        pre_rect = filtfilt(b, a, pre_rect, axis=0)
        pre_rect = pre_rect - np.mean(pre_rect, axis=0)
        post = np.abs(pre_rect)
        wc = min(6.0/nyq, 0.99)
        b, a = butter(4, wc, btype='low')
        post = filtfilt(b, a, post, axis=0)

    RMS, MAV, ZC, SSC, WL, centers = [], [], [], [], [], []
    for s in range(0, n - window_size + 1, step):
        e = s + window_size
        pre_w, post_w = pre_rect[s:e, :], post[s:e, :]

        zc_feat = [0]*C if rectified_detected else [zc(pre_w[:, ch], zc_threshold) for ch in range(C)]
        ssc_feat = [0]*C if rectified_detected else [ssc(pre_w[:, ch], ssc_threshold) for ch in range(C)]
        rms_feat = [rms(post_w[:, ch]) for ch in range(C)]
        mav_feat = [mav(post_w[:, ch]) for ch in range(C)]
        wl_feat  = [wl(post_w[:, ch])  for ch in range(C)]

        RMS.append(rms_feat); MAV.append(mav_feat)
        ZC.append(zc_feat);   SSC.append(ssc_feat); WL.append(wl_feat)
        centers.append(s + window_size//2)

    state.extracted_features = {
        'RMS': np.array(RMS), 'MAV': np.array(MAV), 'ZC': np.array(ZC),
        'SSC': np.array(SSC), 'WL': np.array(WL)
    }
    state.window_centers = np.array(centers)
    mats = [state.extracted_features[k] for k in ['RMS','MAV','ZC','SSC','WL']]
    state.feature_matrix = np.concatenate(mats, axis=1)

    return {
        "features": state.extracted_features,
        "feature_matrix": state.feature_matrix,
        "window_centers": state.window_centers,
        "window_size": window_size,
        "overlap": overlap,
        "extraction_method": "optimal_dual_stage"
    }

def extract_windowed_features_legacy(
    state: PipelineState, window_size: int=None, overlap: float=None,
    zc_threshold: float=0.01, ssc_threshold: float=0.01
):
    if state.filtered_signal is None:
        raise ValueError("No filtered signal.")
    if window_size is None: window_size = state.window_size
    if overlap is None: overlap = state.overlap
    step = int(window_size * (1 - overlap))

    X = state.filtered_signal
    if X.ndim == 1: X = X[:, None]
    chans = state.selected_channels if state.selected_channels is not None else list(range(X.shape[1]))
    X = X[:, chans]
    n, C = X.shape

    RMS, MAV, ZC, SSC, WL, centers = [], [], [], [], [], []
    for s in range(0, n - window_size + 1, step):
        e = s + window_size
        W = X[s:e, :]
        RMS.append([rms(W[:, ch]) for ch in range(C)])
        MAV.append([mav(W[:, ch]) for ch in range(C)])
        ZC.append([zc(W[:, ch], zc_threshold) for ch in range(C)])
        SSC.append([ssc(W[:, ch], ssc_threshold) for ch in range(C)])
        WL.append([wl(W[:, ch]) for ch in range(C)])
        centers.append(s + window_size//2)

    state.extracted_features = {
        'RMS': np.array(RMS), 'MAV': np.array(MAV), 'ZC': np.array(ZC),
        'SSC': np.array(SSC), 'WL': np.array(WL)
    }
    state.window_centers = np.array(centers)
    mats = [state.extracted_features[k] for k in ['RMS','MAV','ZC','SSC','WL']]
    state.feature_matrix = np.concatenate(mats, axis=1)

    return {
        "features": state.extracted_features,
        "feature_matrix": state.feature_matrix,
        "window_centers": state.window_centers,
        "window_size": window_size,
        "overlap": overlap,
        "extraction_method": "legacy_single_stage"
    }

def save_features_to_csv(state: PipelineState, filename: str="emg_features.csv"):
    import pandas as pd, os
    if state.extracted_features is None or state.window_centers is None:
        print("No features to save.")
        return
    feature_names = ['RMS','MAV','ZC','SSC','WL']
    rows = []
    for i, t in enumerate(state.window_centers):
        row = {"Window": i+1, "Time_s": t/state.fs}
        for name in feature_names:
            mat = state.extracted_features[name]
            for ch in range(mat.shape[1]):
                row[f"{name}_Ch{ch+1}"] = mat[i, ch]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Saved: {os.path.abspath(filename)}")
