# Machine-Learning-Enhanced-Electromyography-Prosthetic-Arm

A machine learning system for classifying EMG (electromyography) signals to control prosthetic hand movements. This project implements signal processing pipelines, feature extraction, and multiple machine learning models (CNN and LDA) for gesture recognition from EMG data.

## Overview

This repository contains a complete pipeline for EMG signal processing and gesture classification:
- **Signal Processing**: Complete filtering pipeline including bandpass, notch, DC removal, rectification, and envelope filtering
- **Feature Extraction**: Multiple EMG features (RMS, MAV, ZC, SSC, WL)
- **Machine Learning Models**: CNN (TensorFlow) and LDA classifiers for gesture recognition
- **Hardware Integration**: Arduino code for controlling servos based on EMG signals
- **NinaPro Dataset**: Integration with standard EMG datasets for training

## Repository Structure

```
├── miyah work/              # Main development codebase
│   ├── filtering/           # Signal processing pipeline
│   │   ├── filters.py       # Filtering functions (bandpass, notch, rectification, etc.)
│   │   ├── features.py      # Feature extraction (RMS, MAV, ZC, SSC, WL)
│   │   ├── menu_interface.py # Interactive CLI for pipeline operations
│   │   ├── plots.py         # Visualization utilities
│   │   └── io_utils.py      # Data loading and batch processing
│   └── scripts/
│       └── train_cnn.py     # CNN training script
├── MachineLearning/
│   └── emg lda classifier (testing)  # LDA classifier implementation
├── ArduinoCode/
│   ├── EMG_Arduino          # EMG sensor + servo control code
│   └── RPS_Arduino_Code     # Rock-Paper-Scissors demo code
├── ArduinoandPython/        # Arduino-Python serial communication
│   ├── ArduinoCode          # Arduino sketch for serial communication
│   └── PythonScript         # Python script for processing and sending commands
├── ProcessingPipeline/      # Documentation of processing steps
├── filtering/
│   └── ProcessingPipelineV2 # Alternative pipeline implementation
├── initialdatatesting/      # Initial EMG data exploration scripts
├── MatlabvsPython/          # Comparison scripts for MATLAB vs Python plotting
├── NinaPro Datasets/        # NinaPro EMG dataset (subject 1)
└── requirements.txt         # Python dependencies
```

## Signal Processing Pipeline

The EMG signal processing follows these steps:

1. **Amplification** - Hardware or software amplification of raw EMG
2. **Bandpass Filter** (20-500 Hz) - Removes unwanted high and low frequencies
3. **Notch Filter** (50-60 Hz) - Removes power line interference
4. **DC Removal** - Removes DC offset/bias from signal
5. **Rectification** - Converts bipolar signal to unipolar (absolute value)
6. **Low-Pass Envelope Filter** (2-6 Hz) - Smooths signal for muscle activation detection

## Feature Extraction

The system extracts five key EMG features for classification:

- **RMS (Root Mean Square)** - Measure of signal power/strength
- **MAV (Mean Absolute Value)** - Average muscle activity indicator
- **ZC (Zero Crossings)** - Signal complexity and oscillation measure
- **SSC (Slope Sign Changes)** - Muscle coordination indicator
- **WL (Waveform Length)** - Total signal variation and activity measure

## Installation

1. Clone the repository:
```bash
git clone https://github.com/DredgenYors/Machine-Learning-Enhanced-Electromyography-Prosthetic-Arm.git
cd Machine-Learning-Enhanced-Electromyography-Prosthetic-Arm
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Interactive Signal Processing Pipeline

Run the interactive menu interface for signal processing and feature extraction:

```bash
cd "miyah work"
python -m filtering.menu_interface
```

This provides options for:
- Applying individual or complete filtering pipelines
- Extracting features using optimal or legacy methods
- Visualizing signals at each processing stage
- Batch processing multiple .mat files
- Saving features to CSV for ML training

### Training Machine Learning Models

**CNN Model:**
```bash
cd "miyah work"
python scripts/train_cnn.py --data_dir "path/to/mat/files" --epochs 40 --batch 64
```

**LDA Classifier:**
```bash
python "MachineLearning/emg lda classifier (testing)"
```
(Ensure you have a features CSV file generated from the filtering pipeline)

### Arduino Integration

1. Upload the Arduino sketch from `ArduinoCode/EMG_Arduino` to your Arduino board
2. Connect EMG sensor to analog pin A0
3. Connect servos via PCA9685 PWM driver
4. Optionally run the Python serial communication script from `ArduinoandPython/PythonScript`

## Dataset

The repository includes the NinaPro Database subject 1 dataset, which contains EMG recordings for various hand gestures and movements. The dataset supports training models to recognize up to 50+ different hand movements.

## Dependencies

- numpy
- scipy
- pandas
- matplotlib
- scikit-learn
- jupyterlab
- tensorflow
- pyserial

## Hardware Requirements

- Arduino board (for real-time control)
- EMG sensor (analog output)
- PCA9685 PWM servo driver
- Servos (up to 5 for finger control)
- Power supply for servos

## License

This project is open-source. Please check individual files for specific license information.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

