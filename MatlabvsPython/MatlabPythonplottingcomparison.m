% Replace 'yourfile.mat' and 'your_variable' with your actual file and variable names
load('c:\Users\Brandon\Desktop\FALL 2025\ECE416 - ECE Engr Project II\NinaPro Datasets\s1\S1_A1_E1.mat');
data = emg(:); % Ensure column vector

plot(data(1:100), 'o-');
title('First 100 values (MATLAB)');
xlabel('Index');
ylabel('Value');

% Display first 10 values for direct comparison
disp('First 10 values:');
disp(data(1:10)');