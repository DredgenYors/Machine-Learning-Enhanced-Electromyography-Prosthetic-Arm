import numpy as np
from train_cnn import make_windows

# Load your concatenated envelopes + labels
from train_cnn import load_all_csv

envelope, labels = load_all_csv(r"C:\Users\miyah\OneDrive\Desktop\Machine-Learning-Enhanced-Electromyography-Prosthetic-Arm-1\emg_datasets")

print("Raw labels in dataset:", np.unique(labels, return_counts=True))

# Windowing test
Xw, yw = make_windows(envelope, labels, fs=1000, win_ms=200, step_ms=50)

print("Labels after windowing:", np.unique(yw, return_counts=True))
