# filtering/io_utils.py
import os, glob, zipfile, shutil
import numpy as np
import scipy.io
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from miyah_work.filtering.filters import reset_signal, apply_complete_pipeline
from miyah_work.filtering.features import extract_windowed_features_optimal
from miyah_work.filtering.pipeline_state import PipelineState
import pandas as pd

def load_csv_file(state: PipelineState, csv_file: str = None) -> bool:
    """
    Load EMG data from a CSV file into the pipeline state.
    """
    if csv_file is None:
        Tk().withdraw()
        csv_file = askopenfilename(title="Select a CSV file", filetypes=[("CSV files", "*.csv")])
    if not csv_file:
        print("No file selected.")
        return False

    try:
        data = pd.read_csv(csv_file)
        # Ensure required columns are present
        if 'voltage' not in data.columns:
            print("CSV file must contain a 'voltage' column.")
            return False

        # Load the voltage column as the EMG signal
        emg = data['voltage'].values
        if emg.ndim == 1:
            emg = emg[:, None]  # Ensure 2D array for consistency

        # Load the gesture column if present
        state.original_signal = emg
        state.filtered_signal = emg.copy()
        state.selected_channels = [0]  # Default to the first channel
        state.exercise = data['gesture'].values if 'gesture' in data.columns else None

        print("CSV file loaded successfully.")
        return True
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return False
    
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

def save_filtered_data_to_csv(state: PipelineState, filename: str = "filtered_emg_data.csv"):
    """
    Save the filtered EMG data to a CSV file.
    """
    try:
        if state.filtered_signal is None:
            raise ValueError("No filtered signal available to save.")
        
        data = {
            'voltage': state.filtered_signal.flatten()
        }
        if state.exercise is not None:
            data['label'] = state.exercise
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Filtered data saved to: {filename}")
    except Exception as e:
        print(f"Error saving filtered data: {e}")
