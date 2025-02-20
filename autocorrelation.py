import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy import stats

# Load Data
filename = 'C:\\Users\\Part 2 Users\\Documents\\DLS\\E2b\\DLS_files'  # Change to your file's path if necessary
data = np.loadtxt(filename, delimiter='\t', skiprows=3)  # Skips first 3 rows
t = data[:, 0]  # Time
voltage = data[:, 1]  # Voltage

fs = 1000 / (t[1] - t[0])  # Sampling frequency
f0 = 100  # Frequency to remove (50 Hz)
Q = 10  # Quality factor: Controls the width of the notch

    # Create the bandstop filter
b, a = butter(2, [f0 - 1, f0 + 1], btype='bandstop', fs=fs)

    # Apply the bandstop filter to the voltage signal
voltage_filtered = filtfilt(b, a, voltage)

# Get the number of samples (N)
N = len(voltage_filtered)

# Calculate Autocorrelation functionpip (ACF)
ac = np.zeros(N-1)  # Prepare the array for autocorrelation
m = np.mean(voltage_filtered)  # Mean of the data

# Compute the autocorrelation
for j in range(1, N):
    ac[j-1] = np.mean((voltage_filtered[0:N-j] - m) * (voltage_filtered[j:N] - m))

# Plot the autocorrelation function in linear and semilogy scale
#plt.figure(figsize=(10, 8))

# Plot ACF in linear scale
#plt.subplot(2, 1, 1)
#plt.plot(t[1:], ac, '.')
#plt.title('Autocorrelation function')
#plt.xlabel('Time [s]')
#plt.ylabel('C(t) [V^2]')

# Plot ACF in semilogy scale
#plt.subplot(2, 1, 2)
#plt.semilogy(t[1:], ac, '.')
#plt.title('Autocorrelation function LOGY scale')
#plt.xlabel('Time [s]')
#plt.ylabel('C(t) [V^2]')

#plt.tight_layout()
#plt.show()

# Exponential decay fitting using a linear fit on log(ACF)
rangemax = 13  # Limit of data to use for fitting

# Perform the linear fit on the log of the autocorrelation for the first 'rangemax' points
x = np.arange(rangemax)
y = np.log(ac[:rangemax])

# Linear fit: y = m * x + b
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Characteristic time of exponential decay
tau = -1 / slope
print(f"Tau (characteristic time): {tau}")

# Plot the autocorrelation function with the exponential fit
plt.figure(figsize=(10, 6))

# Plot the first 'rangemax' data points
plt.semilogy(x, ac[:rangemax], 'ro', label='Exp data')
plt.plot(x, np.exp(slope * x + intercept), 'b-', linewidth=2, label='Fit - exp(-t/tau)')

plt.xlabel('Time [s]')
plt.ylabel('C(t) [V^2]')
plt.legend(loc='best', fontsize=14)
plt.title('Correlation function with Exponential Fit (Semi-Log scale)', fontsize=14)
plt.tight_layout()
plt.show()

# Plot the ACF with the exponential fit in linear scale for a larger range
rangemax = rangemax + 500  # Increase range for linear plot

plt.figure(figsize=(10, 6))
plt.plot(np.arange(rangemax), ac[:rangemax], 'ro', label='Exp data')
plt.plot(np.arange(rangemax), np.exp(slope * np.arange(rangemax) + intercept), 'b-', linewidth=2, label='Fit - exp(-t/tau)')

plt.xlabel('Time [s]')
plt.ylabel('C(t) [V^2]')
plt.legend(loc='best', fontsize=14)
plt.title('Correlation function with Exponential Fit (Linear scale)', fontsize=14)
plt.tight_layout()
plt.show()
