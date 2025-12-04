import os, sys, json, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
layers= tf.keras.layers
models= tf.keras.models
EarlyStopping= tf.keras.callbacks.EarlyStopping
ReduceLROnPlateau= tf.keras.callbacks.ReduceLROnPlateau
ModelCheckpoint= tf.keras.callbacks.ModelCheckpoint


# CLI

def get_args():
    p = argparse.ArgumentParser(description="Train 1D CNN for EMG gesture classification")
    p.add_argument("--fs", type=int, default=1000)
    p.add_argument("--win_ms", type=int, default=50)
    p.add_argument("--step_ms", type=int, default=25)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--test_split", type=float, default=0.2)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--outdir", type=str, default="models")
    p.add_argument("--silent", action="store_true")
    return p.parse_args()


# Load your CSV files

def load_all_csv(data_dir):
    data_dir = Path(data_dir)
    all_emg = []
    all_labels = []

    for csv_file in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(csv_file)

        if "voltage" not in df.columns:
            print(f"[WARN] Skipping {csv_file}: no 'voltage' column")
            continue

        if "gesture" not in df.columns:
            print(f"[WARN] Skipping {csv_file}: no 'gesture' column")
            continue

        emg = df["voltage"].to_numpy(dtype=float)[:, None]  # shape (N,1)
        labels = df["gesture"].to_numpy(dtype=int)

        if len(emg) != len(labels):
            print(f"[WARN] Skipping {csv_file}: EMG/label length mismatch")
            continue

        print(csv_file.name, "unique labels:", np.unique(labels))

        all_emg.append(emg)
        all_labels.append(labels)

    if not all_emg:
        raise SystemExit("No valid CSV files found!")

    envelope = np.concatenate(all_emg, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    print("\nOverall label counts:", np.unique(labels, return_counts=True))
    print()

    return envelope, labels

# Windowing functions
def window_indices(n, fs, win_ms, step_ms):
    win = int(win_ms * fs / 1000)
    step = int(step_ms * fs / 1000)
    idx = [(i, i + win) for i in range(0, n - win + 1, step)]
    return idx, win


def make_windows(X, y, fs, win_ms, step_ms):
    idx_list, win = window_indices(len(X), fs, win_ms, step_ms)
    Xw = np.stack([X[i0:i1] for (i0, i1) in idx_list])

    yw = []
    for (i0, i1) in idx_list:
        vals, counts = np.unique(y[i0:i1], return_counts=True)
        yw.append(vals[np.argmax(counts)])

    return Xw.astype(np.float32), np.array(yw)


# Build CNN
def build_cnn(input_shape, num_classes):
    """
    Smaller 1D CNN to reduce overfitting and force the model
    to learn only the most important EMG patterns.
    """
    model = models.Sequential([
        # Conv block 1
        layers.Conv1D(32, 5, padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),

        # Conv block 2
        layers.Conv1D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.3),

        # Global pooling instead of big dense layers
        layers.GlobalAveragePooling1D(),

        # Small dense head
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Main Training Loop
def main():
    args = get_args()
    fs = args.fs

    # Output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.outdir) / f"cnn_run_{run_id}"
    (out_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_root / "figs").mkdir(exist_ok=True)
    (out_root / "reports").mkdir(exist_ok=True)

    # Load CSV dataset
    envelope, labels = load_all_csv(args.data_dir)

    # Windowing
    Xw, yw = make_windows(envelope, labels, fs, args.win_ms, args.step_ms)
    print("Windows:", Xw.shape, "Classes:", np.unique(yw))

    # All classes (for confusion matrix ticks & manifest)
    classes_all = np.unique(yw)

    # Train/val/test split
    X_train, X_test, y_train, y_test = train_test_split(
        Xw, yw,
        test_size=args.test_split,
        random_state=42,
        stratify=yw
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=args.val_split,
        random_state=42,
        stratify=y_train
    )

    # Standardize
    mu = X_train.mean()
    sd = X_train.std() + 1e-8
    X_train = (X_train - mu) / sd
    X_val = (X_val - mu) / sd
    X_test = (X_test - mu) / sd
    np.savez(out_root / "scaler_mu_sd.npz", mu=mu, sd=sd)

    # Build model
    num_classes = len(np.unique(y_train))
    model = build_cnn((X_train.shape[1], X_train.shape[2]), num_classes)
    model.summary()

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = {i: float(class_weights[i]) for i in range(len(class_weights))}

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        ModelCheckpoint(str(out_root / "checkpoints" / "best.keras"),
                        save_best_only=True,
                        monitor="val_accuracy"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
    ]

    verbose = 0 if args.silent else 1
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=verbose
    )

    # Save history
    with open(out_root / "history.json", "w") as f:
        json.dump(history.history, f, indent=2)

    # Accuracy plot
    plt.figure()
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_root / "figs" / "accuracy.png", dpi=150)
    if not args.silent:
        plt.show()
    plt.clf()


    # Evaluate
   
    if X_test is not None and y_test is not None:
        # Predictions
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

        # Accuracy
        acc = (y_pred == y_test).mean()
        print("Test accuracy:", round(float(acc), 4))

        # Classification report (print + save)
        rep = classification_report(y_test, y_pred, digits=4)
        print(rep)
        with open(out_root / "reports" / "classification_report.txt", "w") as f:
            f.write(rep)

        # Confusion matrix (with labels/annotations)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, cmap="Blues", interpolation="nearest")
        plt.title("Confusion Matrix")
        plt.colorbar()

        ticks = np.arange(len(classes_all))
        plt.xticks(ticks, classes_all)
        plt.yticks(ticks, classes_all)

        thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black"
                )

        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(out_root / "figs" / "confusion_matrix.png", dpi=150)
        if not args.silent:
            plt.show()
        plt.clf()
    else:
        print("[INFO] Skipping evaluation due to lack of test data.")

    # Save final model
    final_path = out_root / "cnn_final.keras"
    model.save(final_path)
    print("Saved final model to:", final_path)

   
    # Manifest
   
    manifest = {
        "run_id": str(run_id),
        "data_dir": str(args.data_dir),
        "outdir": str(out_root),
        "fs": fs,
        "win_ms": args.win_ms,
        "step_ms": args.step_ms,
        "epochs": args.epochs,
        "batch": args.batch,
        "test_split": args.test_split,
        "val_split": args.val_split,
        "num_classes": int(len(classes_all)),
        "classes": [int(c) for c in classes_all],
        "scaler_file": "scaler_mu_sd.npz",
        "best_checkpoint": "checkpoints/best.keras",
        "final_model": "cnn_final.keras"
    }

    with open(out_root / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\nAll artifacts saved under:", out_root.resolve())


if __name__ == "__main__":
    main()
