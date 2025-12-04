# filtering/io_utils.py
import os, glob, zipfile, shutil
import numpy as np
import scipy.io
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from .pipeline_state import PipelineState
from .filters import reset_signal, apply_complete_pipeline
from .features import extract_windowed_features_optimal
import pandas as pd

def load_matlab_file(state: PipelineState) -> bool:
    print("EMG Filtering Pipeline")
    print("="*50)
    Tk().withdraw()
    mat_file = askopenfilename(title="Select a MATLAB .mat file", filetypes=[("MAT files","*.mat")])
    if not mat_file:
        print("No file selected.")
        return False

    mat = scipy.io.loadmat(mat_file)
    if 'emg' not in mat:
        raise ValueError("Expected 'emg' in the .mat file.")
    emg = np.squeeze(mat['emg'])
    if emg.ndim == 1: emg = emg[:, None]
    state.original_signal = emg
    state.filtered_signal = emg.copy()
    state.exercise = np.squeeze(mat['exercise']) if 'exercise' in mat else None

    if state.num_channels == 1:
        state.selected_channels = [0]
    else:
        print(f"Available channels: 0..{state.num_channels-1}")
        raw = input("Enter channels (e.g., 0,2,4) or 'all': ").strip().lower()
        if raw == 'all':
            state.selected_channels = list(range(state.num_channels))
        else:
            try:
                chs = [int(x) for x in raw.split(',') if x.strip().isdigit()]
            except Exception:
                chs = []
            if not chs or any(ch<0 or ch>=state.num_channels for ch in chs):
                print("Invalid selection, defaulting to all.")
                chs = list(range(state.num_channels))
            state.selected_channels = chs
    return True

def batch_process_zip_of_mat_files(zip_path: str, state: PipelineState, out_csv: str="combined_emg_features.csv"):
    if not zip_path or not os.path.exists(zip_path):
        print("ZIP not found.")
        return
    extract_dir = "unzipped_matlab_files"
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)

    mats = glob.glob(os.path.join(extract_dir, "*.mat"))
    all_dfs = []
    for path in mats:
        try:
            mat = scipy.io.loadmat(path)
            emg = np.squeeze(mat['emg'])
            if emg.ndim == 1: emg = emg[:, None]
            state.original_signal = emg
            state.filtered_signal = emg.copy()
            state.selected_channels = list(range(emg.shape[1]))
            reset_signal(state)
            apply_complete_pipeline(state)
            extract_windowed_features_optimal(state)
            F = state.feature_matrix
            if isinstance(F, np.ndarray):
                df = pd.DataFrame(F, columns=[f"feat_{i+1}" for i in range(F.shape[1])])
                if 'exercise' in mat:
                    df['Exercise'] = int(np.squeeze(mat['exercise']))
                all_dfs.append(df)
        except Exception as e:
            print(f"Error processing {path}: {e}")

    if all_dfs:
        combo = pd.concat(all_dfs, ignore_index=True)
        combo.to_csv(out_csv, index=False)
        print(f"Saved combined: {os.path.abspath(out_csv)}")
    else:
        print("No features extracted from any files.")
