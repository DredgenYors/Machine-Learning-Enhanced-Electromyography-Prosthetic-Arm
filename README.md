# Machine-Learning-Enhanced-Electromyography-Prosthetic-Arm
Road Map / TODO:
1. Set up the Github Repo and Coding Environment
   - virtualenv on Python
2. Collect datasets
   - Collect EMG data from other similar repos and NinaPro
   - organize data into raw, processed, and notebooks
3. Set up preprocessing pipeline
   - Processing includes:
     - Filtering
     - Windowing
     - Feature Extraction (RMS, MAV, ZC, SSC, WL)
     - Amplifying
4. Begin creating baseline ML model
   - Start small with classical machine learning (SVM, RandomForest, k-NN)
   - Test using confusion matrix, per-class recall, and latency of prediction
5. Start creating an advanced model
   - Build a small CNN in TensorFlow
   - Train the CNN
6. Compress and Export Model
   - Convert TensorFlow model to TensorFlow Lite
   - Use post-training quantization to normalize
   - Ensure the new model fits and runs on the microcontroller
7. Integration Testing
   - Implement a small Python program to simulate reading inputs and mapping them
   - Send Python commands to Arduino over serial
  
