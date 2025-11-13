# filtering/verify.py
import numpy as np
from scipy.fft import fft, fftfreq
from .pipeline_state import PipelineState
from datetime import datetime
import io
from contextlib import redirect_stdout
import os

def verify_filtering_pipeline(state: PipelineState):
    if state.original_signal is None or state.filtered_signal is None:
        print("No signal loaded.")
        return
    n0 = state.original_signal.shape[0]
    print("\n=== EMG FILTERING PIPELINE VERIFICATION ===")
    print(f"Original shape: {state.original_signal.shape} | Filtered shape: {state.filtered_signal.shape}")
    print(f"Sampling: {state.fs} Hz | Duration: {n0/state.fs:.2f}s")
    print(f"NaN original/filtered: {np.isnan(state.original_signal).sum()} / {np.isnan(state.filtered_signal).sum()}")
    print(f"Inf original/filtered: {np.isinf(state.original_signal).sum()} / {np.isinf(state.filtered_signal).sum()}")

    n = min(8192, n0)
    freqs = fftfreq(n, 1/state.fs)[:n//2]
    ch = 0
    o_fft = np.abs(fft(state.original_signal[:n, ch]))[:n//2]
    f_fft = np.abs(fft(state.filtered_signal[:n, ch]))[:n//2]
    print(f"Dominant freq original/filtered: {freqs[np.argmax(o_fft[1:])+1]:.1f} / {freqs[np.argmax(f_fft[1:])+1]:.1f} Hz")

    print("\nApplied filters:")
    if state.filter_history:
        for i, f in enumerate(state.filter_history, 1):
            print(f"  {i}. {f}")
    else:
        print("  (none)")

def verify_feature_extraction(state: PipelineState):
    if state.extracted_features is None:
        print("No features extracted yet.")
        return
    print("\n=== FEATURE EXTRACTION VERIFICATION ===")
    for name, mat in state.extracted_features.items():
        print(f"{name}: shape={mat.shape}, mean={np.mean(mat):.4f}, std={np.std(mat):.4f}, "
              f"range=[{np.min(mat):.4f},{np.max(mat):.4f}]")

def create_verification_report(state: PipelineState, filename: str=None):
    if filename is None:
        filename = f"emg_verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    buf = io.StringIO()
    with redirect_stdout(buf):
        verify_filtering_pipeline(state)
        verify_feature_extraction(state)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"Report saved to: {os.path.abspath(filename)}")

