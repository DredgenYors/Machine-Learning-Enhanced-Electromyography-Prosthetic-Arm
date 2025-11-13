# filtering/filters.py
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from .pipeline_state import PipelineState

def _subset(state: PipelineState, X: np.ndarray) -> np.ndarray:
    return X[:, state.selected_channels] if state.selected_channels is not None else X

def bandpass_filter(state: PipelineState, lowcut=20, highcut=500, order=4):
    nyq = 0.5 * state.fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    b, a = butter(order, [low, high], btype='band')
    X = state.filtered_signal.copy()
    if state.selected_channels:
        X[:, state.selected_channels] = filtfilt(b, a, _subset(state, X), axis=0)
    else:
        X = filtfilt(b, a, X, axis=0)
    state.filtered_signal = X
    state.filter_history.append(f"Bandpass {lowcut}-{highcut} Hz")

def notch_filter(state: PipelineState, notch_freq=50, Q=30):
    b, a = iirnotch(notch_freq, Q, state.fs)
    X = state.filtered_signal.copy()
    if state.selected_channels:
        X[:, state.selected_channels] = filtfilt(b, a, _subset(state, X), axis=0)
    else:
        X = filtfilt(b, a, X, axis=0)
    state.filtered_signal = X
    state.filter_history.append(f"Notch {notch_freq} Hz")

def dc_removal(state: PipelineState):
    X = state.filtered_signal.copy()
    if state.selected_channels:
        mu = np.mean(_subset(state, X), axis=0)
        X[:, state.selected_channels] = _subset(state, X) - mu
    else:
        X = X - np.mean(X, axis=0)
    state.filtered_signal = X
    state.filter_history.append("DC removal")

def rectify(state: PipelineState):
    X = state.filtered_signal.copy()
    if state.selected_channels:
        X[:, state.selected_channels] = np.abs(_subset(state, X))
    else:
        X = np.abs(X)
    state.filtered_signal = X
    state.filter_history.append("Rectification")

def lowpass_envelope(state: PipelineState, cutoff=6, order=4):
    nyq = 0.5 * state.fs
    wc = min(cutoff / nyq, 0.99)
    b, a = butter(order, wc, btype='low')
    X = state.filtered_signal.copy()
    if state.selected_channels:
        X[:, state.selected_channels] = filtfilt(b, a, _subset(state, X), axis=0)
    else:
        X = filtfilt(b, a, X, axis=0)
    state.filtered_signal = X
    state.filter_history.append(f"Low-pass {cutoff} Hz")

def apply_complete_pipeline(state: PipelineState):
    bandpass_filter(state, 20, 500, order=4)
    notch_filter(state, 50, Q=30)
    dc_removal(state)
    rectify(state)
    lowpass_envelope(state, 6, order=4)

def reset_signal(state: PipelineState):
    if state.original_signal is not None:
        state.filtered_signal = state.original_signal.copy()
        state.filter_history.clear()
