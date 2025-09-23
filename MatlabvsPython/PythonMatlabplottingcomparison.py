import scipy.io
import matplotlib.pyplot as plt

# Replace 'yourfile.mat' and 'your_variable' with your actual file and variable names
mat = scipy.io.loadmat(r'c:\Users\Brandon\Desktop\FALL 2025\ECE416 - ECE Engr Project II\NinaPro Datasets\s1\S1_A1_E1.mat')
data = mat['emg'].squeeze()  # .squeeze() flattens to 1D if needed

plt.plot(data[:100, 0], marker='o')
plt.title('First 100 values (Python)')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()

# Print first 10 values for direct comparison
print("First 10 values:", data[:10])