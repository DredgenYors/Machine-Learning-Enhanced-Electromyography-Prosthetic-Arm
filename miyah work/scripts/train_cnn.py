
# scripts/train_cnn.py
import os, sys, json, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Make local packages importable
sys.path.append(os.getcwd())
from filtering.pipeline_state import PipelineState
from filtering.filters import apply_complete_pipeline  # fixed 5-step pipeline


# -----------------------
# CLI
# -----------------------
def get_args():
    p = argparse.ArgumentParser(description="Train 1D CNN for EMG gesture classification")
    p.add_argument("--fs", type=int, default=1000, help="Sampling rate (Hz)")
    p.add_argument("--win_ms", type=int, default=250, help="Window length (ms)")
    p.add_argument("--step_ms", type=int, default=125, help="Hop length (ms)")
    p.add_argument("--smooth", choices=["rms","lowpass"], default="rms", help="(reserved) envelope smoothing mode")
    p.add_argument("--rms_ms", type=int, default=200, help="(reserved) RMS window (ms)")
    p.add_argument("--lpf", type=float, default=5.0, help="(reserved) Lowpass cutoff (Hz)")
    p.add_argument("--band_low", type=float, default=20.0, help="(ignored by fixed pipeline)")
    p.add_argument("--band_high", type=float, default=450.0, help="(ignored by fixed pipeline)")
    p.add_argument("--notch_base", type=float, default=60.0, help="(ignored by fixed pipeline)")
    p.add_argument("--notch_q", type=float, default=30.0, help="(ignored by fixed pipeline)")
    p.add_argument("--notch_harmonics", type=int, default=1, help="(ignored by fixed pipeline)")
    p.add_argument("--data_dir", type=str, default="", help="Folder of .mat files to use. If empty, a file dialog will pop up.")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--test_split", type=float, default=0.2)
    p.add_argument("--outdir", type=str, default="models", help="Where to save runs/models")
    p.add_argument("--run_tag", type=str, default="", help="Suffix for run folder name")
    p.add_argument("--silent", action="store_true", help="No progress prints during fit")

    # Resume options
    p.add_argument("--resume", action="store_true",
                   help="Resume from the latest models/cnn_run_*/checkpoints/best.keras if present")
    p.add_argument("--resume_from", type=str, default="",
                   help="Resume from a specific checkpoint or run folder")
    return p.parse_args()


# -----------------------
# Helpers
# -----------------------
def window_indices(n, fs, win_ms=250, step_ms=125):
    win = int(win_ms * fs / 1000)
    step = int(step_ms * fs / 1000)
    idx = [(i, i + win) for i in range(0, n - win + 1, step)]
    return idx, win, step

def make_windows(X, y, fs, win_ms=250, step_ms=125):
    if X.ndim == 1:
        X = X[:, None]
    idx_list, win, _ = window_indices(len(X), fs, win_ms, step_ms)
    Xw = np.stack([X[i0:i1, :] for (i0, i1) in idx_list], axis=0).astype(np.float32)  # (n_win, win, C)

    yw = None
    if y is not None:
        tmp = []
        for (i0, i1) in idx_list:
            vals, counts = np.unique(y[i0:i1], return_counts=True)
            tmp.append(vals[np.argmax(counts)])
        yw = np.array(tmp, dtype=int)
    return Xw, yw

def load_single_mat_via_dialog():
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename
    Tk().withdraw()
    mat_file = askopenfilename(title="Select a MATLAB .mat file", filetypes=[("MAT files", "*.mat")])
    if not mat_file:
        raise SystemExit("No file selected.")
    return [mat_file]

def load_emg_label_from_mat(path):
    mat = scipy.io.loadmat(path)
    if 'emg' not in mat:
        raise ValueError(f"{path} missing variable 'emg'")
    emg = np.squeeze(mat['emg'])
    if emg.ndim == 1:
        emg = emg[:, None]
    labels = None
    if 'stimulus' in mat:
        labels = np.squeeze(mat['stimulus']).astype(int)
    elif 'exercise' in mat:
        ex = int(np.squeeze(mat['exercise']))
        labels = np.full(emg.shape[0], ex, dtype=int)
    return emg, labels

def find_latest_checkpoint(root="models"):
    root = Path(root)
    if not root.exists():
        return None
    runs = sorted(root.glob("cnn_run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for r in runs:
        ck = r / "checkpoints" / "best.keras"
        if ck.exists():
            return ck
    return None

def resolve_resume_path(resume_from):
    p = Path(resume_from)
    if p.is_dir():
        ck = p / "checkpoints" / "best.keras"
        return ck if ck.exists() else None
    elif p.is_file() and p.suffix == ".keras":
        return p
    return None


# -----------------------
# Model
# -----------------------
def build_cnn(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv1D(64, 7, padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),

        layers.Conv1D(128, 5, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),

        layers.Conv1D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# -----------------------
# Training pipeline
# -----------------------
def main():
    args = get_args()
    fs = args.fs

    # Output dirs
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.run_tag}" if args.run_tag else ""
    out_root = Path(args.outdir) / f"cnn_run_{run_id}{tag}"
    (out_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_root / "figs").mkdir(parents=True, exist_ok=True)
    (out_root / "reports").mkdir(parents=True, exist_ok=True)

    # Collect files
    if args.data_dir:
        mat_files = [str(p) for p in Path(args.data_dir).glob("*.mat")]
        if not mat_files:
            raise SystemExit(f"No .mat files in {args.data_dir}")
    else:
        mat_files = load_single_mat_via_dialog()

    # Load + filter each file to "envelope"
    envelopes = []
    labels_all = []
    for mf in mat_files:
        emg, labels = load_emg_label_from_mat(mf)

        # Build a PipelineState and run the fixed pipeline (bandpass->notch->dc->rectify->lowpass)
        try:
            state = PipelineState(fs=fs, original_signal=emg, filtered_signal=emg.copy(), selected_channels=None)
        except TypeError:
            state = PipelineState()
            state.fs = fs
            state.original_signal = emg
            state.filtered_signal = emg.copy()
            state.selected_channels = None
            state.filter_history = []

        apply_complete_pipeline(state)
        env = state.filtered_signal  # envelope/cleaned signal

        envelopes.append(env.astype(np.float32))
        labels_all.append(labels if labels is not None else np.zeros(env.shape[0], dtype=int))

    # Concatenate across files
    envelope = np.concatenate(envelopes, axis=0)
    labels = np.concatenate(labels_all, axis=0)
    print("Envelope:", envelope.shape, "Labels:", labels.shape)

    # Windowing
    Xw, yw = make_windows(envelope, labels, fs, win_ms=args.win_ms, step_ms=args.step_ms)
    print("Windows:", Xw.shape, "Unique classes:", np.unique(yw))

    # Splits
    X_train, X_test, y_train, y_test = train_test_split(
        Xw, yw, test_size=args.test_split, random_state=42, stratify=yw
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=args.val_split, random_state=42, stratify=y_train
    )

    # Standardize (time, channels)
    mu = X_train.mean(axis=(0, 1), keepdims=True)
    sd = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    X_train = (X_train - mu) / sd
    X_val   = (X_val   - mu) / sd
    X_test  = (X_test  - mu) / sd

    # Save scaler
    np.savez(out_root / "scaler_mu_sd.npz", mu=mu, sd=sd)

    # Resume support
    resume_ckpt = None
    if args.resume_from:
        resume_ckpt = resolve_resume_path(args.resume_from)
        if resume_ckpt is None:
            raise SystemExit(f"--resume_from path not valid: {args.resume_from}")
    elif args.resume:
        resume_ckpt = find_latest_checkpoint("models")
        if resume_ckpt is None:
            print("[WARN] --resume set but no prior checkpoint found. Starting fresh.")

    # Build or load model
    num_classes = len(np.unique(y_train))
    if resume_ckpt and resume_ckpt.exists():
        print(f"[RESUME] Loading checkpoint: {resume_ckpt}")
        model = tf.keras.models.load_model(resume_ckpt)
        # If class count changed, rebuild head
        # check output units safely
        try:
             out_units = model.output_shape[-1]
        except AttributeError:
            out_units = model.layers[-1].units if hasattr(model.layers[-1], "units") else None

        if out_units is not None and out_units != num_classes:
            print("[INFO] Number of classes changed; rebuilding final Dense to match new data.")
            base = model.layers[-2].output
            new_out = layers.Dense(num_classes, activation='softmax')(base)
            model = models.Model(inputs=model.input, outputs=new_out)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    else:
        model = build_cnn(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=num_classes)

    model.summary()

    # Class weights
    classes = np.unique(y_train)
    cw_values = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = {int(c): float(w) for c, w in zip(classes, cw_values)}
    json.dump({"class_weights": class_weights}, open(out_root / "class_weights.json", "w"), indent=2)

    # Callbacks
    ckpt_path = out_root / "checkpoints" / "best.keras"
    cbs = [
        EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5),
        ModelCheckpoint(str(ckpt_path), monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    # Train
    verbose = 0 if args.silent else 1
    hist = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs, batch_size=args.batch,
        class_weight=class_weights,
        callbacks=cbs,
        verbose=verbose
    )

    # Save history
    json.dump(hist.history, open(out_root / "history.json", "w"), indent=2)

    # Training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.plot(hist.history['accuracy']); plt.plot(hist.history['val_accuracy'])
    plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend(["train", "val"]); plt.grid(True)
    plt.subplot(1, 2, 2); plt.plot(hist.history['loss']); plt.plot(hist.history['val_loss'])
    plt.title("Loss"); plt.xlabel("Epoch"); plt.legend(["train", "val"]); plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_root / "figs" / "training_curves.png", dpi=150)
    plt.show()

    # Evaluate
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = (y_pred == y_test).mean()
    print("Test accuracy:", round(float(acc), 4))
    rep = classification_report(y_test, y_pred, digits=4)
    print(rep)
    with open(out_root / "reports" / "classification_report.txt", "w") as f:
        f.write(rep)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    classes_all = np.unique(np.concatenate([y_train, y_val, y_test]))
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title("Confusion Matrix"); plt.colorbar()
    ticks = np.arange(len(classes_all))
    plt.xticks(ticks, classes_all); plt.yticks(ticks, classes_all)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color="white" if cm[i, j] > thresh else "black")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    plt.savefig(out_root / "figs" / "confusion_matrix.png", dpi=150)
    plt.show()

    # Save final model
    final_path = out_root / "cnn_final.keras"
    model.save(final_path)
    print("Saved final model to:", final_path)

    # Manifest
    manifest = {
        "run_id": str(run_id),
        "fs": fs,
        "win_ms": args.win_ms,
        "step_ms": args.step_ms,
        "smooth": args.smooth,
        "rms_ms": args.rms_ms,
        "lpf": args.lpf,
        "band_low": args.band_low,
        "band_high": args.band_high,
        "notch_base": args.notch_base,
        "notch_q": args.notch_q,
        "notch_harmonics": args.notch_harmonics,
        "epochs": args.epochs,
        "batch": args.batch,
        "test_split": args.test_split,
        "val_split": args.val_split,
        "classes": [int(c) for c in classes_all]
    }
    json.dump(manifest, open(out_root / "run_manifest.json", "w"), indent=2)

    print("\nAll artifacts saved under:", out_root.resolve())


if __name__ == "__main__":
    main()

