# filtering/plots.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch
from .pipeline_state import PipelineState
from .features import save_features_to_csv
from typing import Optional

def plot_raw_signal(state: PipelineState, plot_duration: float=10.0):
    if state.original_signal is None:
        print("No signal loaded.")
        return
    n = state.original_signal.shape[0]
    samples = int(min(plot_duration*state.fs, n)) if plot_duration>0 else n
    t = np.arange(samples)/state.fs
    chans = state.selected_channels if state.selected_channels is not None else list(range(state.num_channels))
    k = min(4, len(chans))
    fig, axes = plt.subplots(k, 1, figsize=(12, 3*k))
    if k == 1: axes = [axes]
    colors = ['blue','red','green','orange']
    for i in range(k):
        ch = chans[i]
        axes[i].plot(t, state.original_signal[:samples, ch], color=colors[i%len(colors)], lw=0.8)
        axes[i].set_title(f"Raw EMG - Channel {ch}")
        axes[i].grid(alpha=0.3)
        axes[i].set_xlabel("Time (s)"); axes[i].set_ylabel("Amp")
    plt.suptitle("Raw EMG Signal (No filtering)")
    plt.tight_layout(); plt.show()

def plot_results_comparison(state: PipelineState, plot_duration: float=10.0):
    if state.original_signal is None or state.filtered_signal is None:
        print("No signal loaded.")
        return
    n = state.original_signal.shape[0]
    samples = int(min(plot_duration*state.fs, n)) if plot_duration>0 else n
    t = np.arange(samples)/state.fs
    chans = state.selected_channels if state.selected_channels is not None else list(range(state.num_channels))
    k = min(4, len(chans))
    fig, axes = plt.subplots(k, 2, figsize=(15, 3*k))
    if k == 1: axes = np.array([axes])
    colors = ['blue','red','green','orange']
    for i in range(k):
        ch = chans[i]
        axes[i,0].plot(t, state.original_signal[:samples, ch], color=colors[i%len(colors)], lw=0.8)
        axes[i,0].set_title(f"Original - Ch {ch}"); axes[i,0].grid(alpha=0.3)
        axes[i,1].plot(t, state.filtered_signal[:samples, ch], color=colors[i%len(colors)], lw=0.8)
        axes[i,1].set_title(f"Filtered - Ch {ch}"); axes[i,1].grid(alpha=0.3)
        for j in range(2):
            axes[i,j].set_xlabel("Time (s)"); axes[i,j].set_ylabel("Amp")
    filt_text = " → ".join(state.filter_history) if state.filter_history else "No filters"
    plt.suptitle(f"EMG Filtering Results\nApplied: {filt_text}")
    plt.tight_layout(); plt.show()

def plot_signal_at_each_stage(state: PipelineState, plot_duration: float=10.0):
    if state.original_signal is None:
        print("No signal loaded.")
        return

    signals, labels = [], []
    signals.append(state.original_signal.copy()); labels.append("Raw/original")
    nyq = 0.5*state.fs
    from scipy.signal import butter, filtfilt, iirnotch
    low = 20.0/nyq; high = min(500.0/nyq, 0.99)
    b, a = butter(4, [low, high], btype='band')
    band = filtfilt(b, a, state.original_signal, axis=0)
    signals.append(band); labels.append("After bandpass (20-500 Hz)")
    b, a = iirnotch(50, 30, state.fs)
    notched = filtfilt(b, a, band, axis=0)
    signals.append(notched); labels.append("After notch (50 Hz)")
    dc = notched - np.mean(notched, axis=0)
    signals.append(dc); labels.append("After DC removal")
    rect = np.abs(dc)
    signals.append(rect); labels.append("After rectification")
    wc = min(6.0/nyq, 0.99)
    b, a = butter(4, wc, btype='low')
    env = filtfilt(b, a, rect, axis=0)
    signals.append(env); labels.append("After low-pass envelope (6 Hz)")

    n = state.original_signal.shape[0]
    samples = int(min(plot_duration*state.fs, n)) if plot_duration>0 else n
    t = np.arange(samples)/state.fs
    chans = state.selected_channels if state.selected_channels is not None else list(range(state.num_channels))
    k = min(4, len(chans))
    colors = ['blue','red','green','orange']

    for i in range(k):
        ch = chans[i]
        fig, axes = plt.subplots(len(signals), 1, figsize=(14, 2.5*len(signals)))
        if len(signals) == 1: axes = [axes]
        for s, (sig, lab) in enumerate(zip(signals, labels)):
            axes[s].plot(t, sig[:samples, ch], color=colors[i%len(colors)], lw=0.8)
            axes[s].set_title(f"{lab} - Channel {ch}")
            axes[s].grid(alpha=0.3); axes[s].set_xlabel("Time (s)"); axes[s].set_ylabel("Amp")
        plt.tight_layout(); plt.suptitle(f"Signal at Each Stage - Ch {ch}"); plt.show()

def plot_rectified_filter_stages(state: PipelineState, plot_duration: float=10.0):
    if state.original_signal is None:
        print("No signal loaded.")
        return
    chans = state.selected_channels if state.selected_channels is not None else list(range(state.num_channels))
    orig = state.original_signal[:, chans]
    rectified = (np.min(orig) >= 0) and (np.mean(orig < 0) < 0.01)
    if not rectified:
        print("Original EMG does not appear rectified. Use stage plot for standard pipeline.")
        return

    signals, labels = [], []
    signals.append(orig.copy()); labels.append("Raw/original (rectified)")
    b, a = iirnotch(50, 30, state.fs)
    notched = filtfilt(b, a, orig, axis=0)
    signals.append(notched); labels.append("After notch (50 Hz)")
    dc = notched - np.mean(notched, axis=0)
    signals.append(dc); labels.append("After DC removal")
    nyq = 0.5*state.fs
    wc = min(6.0/nyq, 0.99)
    b, a = butter(4, wc, btype='low')
    env = filtfilt(b, a, dc, axis=0)
    signals.append(env); labels.append("After low-pass envelope (6 Hz)")

    n = orig.shape[0]
    samples = int(min(plot_duration*state.fs, n)) if plot_duration>0 else n
    t = np.arange(samples)/state.fs
    k = min(4, len(chans))
    colors = ['blue','red','green','orange']
    for i in range(k):
        ch = i
        fig, axes = plt.subplots(len(signals), 1, figsize=(14, 2.5*len(signals)))
        if len(signals) == 1: axes = [axes]
        for s, (sig, lab) in enumerate(zip(signals, labels)):
            axes[s].plot(t, sig[:samples, ch], color=colors[i%len(colors)], lw=0.8)
            axes[s].set_title(f"{lab} - Channel {chans[ch]}")
            axes[s].grid(alpha=0.3); axes[s].set_xlabel("Time (s)"); axes[s].set_ylabel("Amp")
        plt.tight_layout(); plt.suptitle(f"Rectified EMG: Stages - Ch {chans[ch]}"); plt.show()

def plot_features_per_channel(state: PipelineState, plot_duration: float=10.0, channel: Optional[int]=None):
    if state.extracted_features is None or state.filtered_signal is None:
        print("Nothing to plot.")
        return
    time_feat = state.window_centers / state.fs
    time_sig  = np.arange(state.filtered_signal.shape[0]) / state.fs
    chans = state.selected_channels if state.selected_channels is not None else list(range(state.num_channels))
    if channel is None:
        channels_to_plot = chans[:4]
    else:
        if channel not in chans:
            print("Invalid channel.")
            return
        channels_to_plot = [channel]

    if plot_duration > 0:
        feat_mask = time_feat <= plot_duration
        sig_mask  = time_sig <= plot_duration
    else:
        feat_mask = slice(None); sig_mask = slice(None)

    features = ['RMS','MAV','ZC','SSC','WL']
    colors = ['red','green','orange','purple','brown']

    for idx, ch in enumerate(channels_to_plot):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10)); axes = axes.flatten()
        axes[0].plot(time_sig[sig_mask], state.filtered_signal[sig_mask, ch], color='black', lw=0.8)
        axes[0].set_title(f"Filtered EMG - Channel {ch}"); axes[0].grid(alpha=0.3)
        if state.filter_history:
            axes[0].text(0.02, 0.98, f"Filters: {' → '.join(state.filter_history)}",
                         transform=axes[0].transAxes, fontsize=8, va='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        for i, name in enumerate(features):
            mat = state.extracted_features[name]
            if mat.shape[1] > idx:
                vals = mat[feat_mask, idx]
                axes[i+1].plot(time_feat[feat_mask], vals, color=colors[i], marker='o', ms=3, lw=1.5)
            axes[i+1].set_title(f"{name} - Channel {ch}"); axes[i+1].grid(alpha=0.3)
        plt.tight_layout()
        plt.suptitle(f"EMG Analysis - Channel {ch} | Window={state.window_size/state.fs:.3f}s Overlap={state.overlap*100:.0f}%")
        plt.show()

def plot_features_overview(state: PipelineState, plot_duration: float=10.0):
    if state.extracted_features is None:
        print("No features extracted.")
        return
    t = state.window_centers / state.fs
    mask = t <= plot_duration if plot_duration>0 else slice(None)
    t = t[mask]
    chans = state.selected_channels if state.selected_channels is not None else list(range(state.num_channels))
    k = min(4, len(chans))
    colors = ['blue','red','green','orange']
    feats = ['RMS','MAV','ZC','SSC','WL']

    fig, axes = plt.subplots(2, 3, figsize=(18, 12)); axes = axes.flatten()
    for i, name in enumerate(feats):
        ax = axes[i]
        M = state.extracted_features[name][mask, :k]
        for c in range(M.shape[1]):
            ax.plot(t, M[:, c], color=colors[c%len(colors)], label=f"Ch {chans[c]}", lw=1.5, marker='o', ms=2)
        ax.set_title(f"{name} Feature - Overlay"); ax.grid(alpha=0.3)
        if k <= 4: ax.legend(fontsize=8)
    fig.delaxes(axes[5])
    plt.tight_layout(); plt.suptitle("EMG Features Overview - Selected Channels"); plt.show()
