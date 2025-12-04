import serial
import csv
import time
import os
from datetime import datetime
import pandas as pd

class EMGDataCollector:
    def __init__(self, port='COM3', baudrate=115200):
        self.ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)
        self.data = []

        # ------------------------------
        # SAVE DIRECTORY (MANUAL PATH)
        # ------------------------------
        self.save_dir = r"C:\Users\miyah\OneDrive\Desktop\Machine-Learning-Enhanced-Electromyography-Prosthetic-Arm-1\emg_datasets"
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"📁 Saving CSV files to: {self.save_dir}")

    def collect_gesture(self, gesture_name, samples_needed=100):
        print(f"\n🎯 Collecting: {gesture_name.upper()} until {samples_needed} samples are received")

        # Tell Arduino which gesture this is
        self.ser.write(f"gesture={gesture_name}\n".encode())
        time.sleep(1)

        input("Press ENTER when ready and maintaining the gesture...")
        self.ser.write(b"start\n")

        samples_collected = 0

        while samples_collected < samples_needed:
            if self.ser.in_waiting:
                line = self.ser.readline().decode().strip()

                if line.startswith("DATA,"):
                    parts = line.split(",")

                    if len(parts) == 4:
                        sample = {
                            'gesture': int(parts[1]),
                            'raw_adc': int(parts[2]),
                            'voltage': float(parts[3]),
                            'timestamp': time.time()
                        }
                        self.data.append(sample)
                        samples_collected += 1

                        if samples_collected % 20 == 0:
                            print(f"  Samples: {samples_collected}")

                elif line == "COLLECTION_COMPLETE":
                    break

        print("\n⏰ Data collection complete.")
        print(f"✓ {gesture_name}: {samples_collected} samples")
        return samples_collected

    def save_data(self, gesture_name, sample_count):
        if not self.data:
            print("No data collected!")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gesture_{gesture_name}_samples_{sample_count}_{timestamp}.csv"
        full_path = os.path.join(self.save_dir, filename)

        df = pd.DataFrame(self.data)
        df.to_csv(full_path, index=False)

        print(f"\n📁 DATA SAVED: {full_path}")
        return full_path

    def close(self):
        self.ser.close()


if __name__ == "__main__":
    gesture = input("Enter gesture (rock, paper, scissors): ").strip().lower()
    samples_needed = 100

    collector = EMGDataCollector('COM3')

    try:
        start_collection = input("Start data collection? (yes/no): ").strip().lower()
        if start_collection == "yes":
            collected = collector.collect_gesture(gesture, samples_needed)
            collector.save_data(gesture, collected)
            print("\n✅ Data collection complete.")
        else:
            print("Data collection aborted.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        collector.close()
        