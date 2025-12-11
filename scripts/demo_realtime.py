import argparse
import time
from pathlib import Path
import numpy as np
import serial
import tensorflow as tf


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True,
                   help="Path to trained run folder (e.g. models/cnn_run_20251114_123456)")
    p.add_argument("--port", required=True, help="Serial port, e.g. COM4")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--fs", type=int, default=1000,
                   help="Sampling rate (Hz). Will be overridden by manifest if present.")
    p.add_argument("--win_ms", type=int, default=250,
                   help="Window length (ms). Will be overridden by manifest if present.")
    p.add_argument("--predict_every", type=int, default=20,
                   help="How many new samples between predictions")
    p.add_argument("--cooldown", type=float, default=1.0,
                   help="Seconds to wait before sending a new command")
    return p.parse_args()


def parse_emg_line(line: str) -> float:
    """
    Extract numeric EMG value from a line.
    Falls back to first numeric token if that fails.
    """
    # Try CSV-style last field first
    if "," in line:
        parts = line.split(",")
        last = parts[-1].strip()
        try:
            return float(last)
        except ValueError:
            pass  
        
    # Generic fallback: first numeric token
    try:
        return float(line)
    except ValueError:
        pass

    cleaned = "".join(ch if (ch.isdigit() or ch in ".-") else " " for ch in line)
    parts = cleaned.split()
    if not parts:
        raise ValueError(f"No numeric token in line: {line!r}")
    return float(parts[0])


def main():
    args = get_args()
    run_dir = Path(args.run_dir)

    # ---- Load model & scaler ----
    model_path = run_dir / "cnn_final.keras"
    scaler_path = run_dir / "scaler_mu_sd.npz"
    manifest_path = run_dir / "run_manifest.json"

    print(f"[INFO] Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(f"[INFO] Loading scaler from {scaler_path}")
    scaler = np.load(scaler_path)
    mu, sd = scaler["mu"], scaler["sd"]

    # ---- Try to load manifest to get classes, fs, win_ms ----
    classes = None
    if manifest_path.exists():
        import json
        manifest = json.load(open(manifest_path, "r"))
        classes = manifest.get("classes", None)
        fs_train = manifest.get("fs", args.fs)
        win_ms_train = manifest.get("win_ms", args.win_ms)
        print(f"[INFO] Classes from manifest: {classes}")
        print(f"[INFO] Using fs={fs_train} Hz, win_ms={win_ms_train} ms from manifest")
        fs = fs_train
        win_ms = win_ms_train
    else:
        print("[WARN] No manifest found, using CLI fs/win_ms")
        fs = args.fs
        win_ms = args.win_ms

    # ---- Window length in samples ----
    win_len = int(fs * win_ms / 1000)
    print(f"[INFO] Window length: {win_len} samples")

    # ---- Open serial port ----
    print(f"[INFO] Opening serial port {args.port} at {args.baud}...")
    ser = serial.Serial(args.port, args.baud, timeout=1.0)
    time.sleep(2.0)  # Let Arduino reset

    buffer = []
    samples_since_pred = 0
    last_sent = None
    last_send_time = 0.0
    buffer_ready = False

    print("[INFO] Starting real-time loop. Press Ctrl+C to stop.")
    try:
        while True:
            # Read one line from Arduino
            line_bytes = ser.readline()
            if not line_bytes:
                continue

            line = line_bytes.decode(errors="ignore").strip()
            if not line:
                continue

            # Log what we are receiving
            print(f"[RX] {line}")

            # Try to parse numeric EMG value
            try:
                raw = parse_emg_line(line)
            except ValueError:
                # Non-numeric debug line, ignore
                print(f"[PARSE] Ignoring non-numeric line")
                continue

            # Add to buffer
            buffer.append(raw)
            if len(buffer) > win_len:
                buffer = buffer[-win_len:]  # keep last window

            samples_since_pred += 1

            # Debugging
            if len(buffer) % 50 == 0:
                print(f"[BUF] len={len(buffer)} / {win_len}")

            # Only predict if we have a full window
            if len(buffer) < win_len:
                continue

            if not buffer_ready:
                buffer_ready = True
                print("[READY] Buffer filled, starting predictions")

            # Predict every N new samples
            if samples_since_pred < args.predict_every:
                print(f"[SKIP] samples_since_pred={samples_since_pred} < predict_every={args.predict_every}")
                continue
            samples_since_pred = 0

            # Prepare input
            x = np.array(buffer, dtype=np.float32)[:, None]
            x = x[None, :, :]  

            # Standardize using training stats
            x = (x - mu) / sd

            # Run model
            probs = model.predict(x, verbose=0)[0]
            pred_idx = int(np.argmax(probs))
            conf = float(np.max(probs))

            # Map index to class id if manifest has them
            if classes is not None and pred_idx < len(classes):
                gesture_id = int(classes[pred_idx])
            else:
                gesture_id = pred_idx  # assume 0,1,2,...

            print(f"[MODEL] pred_idx={pred_idx}, gesture_id={gesture_id}, conf={conf:.2f}")

            # Cooldown
            now = time.time()
            if now - last_send_time > args.cooldown:
                if gesture_id in (0, 1, 2):   
                    msg = f"{gesture_id}\n"   
                    ser.write(msg.encode("ascii"))
                    last_sent = gesture_id
                    last_send_time = now
                    print(f"[TX] Sent to Arduino: {gesture_id}")
                else:
                    print(f"[WARN] Gesture id {gesture_id} out of expected range, not sending.")
            else:
                print(f"[COOLDOWN] Waiting: {now - last_send_time:.2f}s < {args.cooldown}s")


    except KeyboardInterrupt:
        print("\n[INFO] Stopping demo.")
    finally:
        ser.close()
        print("[INFO] Serial port closed.")


if __name__ == "__main__":
    main()

