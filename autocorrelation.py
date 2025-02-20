import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy import stats

# Load Data
filename = 'DLS_files/0.75micro_100dil/30deg.prn'  # Change to your file's path if necessary
data = np.loadtxt(filename, delimiter='\t', skiprows=3)  # Skips first 3 rows
t = data[:, 0]  # Time
voltage = data[:, 1]  # Voltage

fs = 1000 / (t[1] - t[0])  # Sampling frequency
f0 = 100  # Frequency to remove (50 Hz)

# Create the bandstop filter
b, a = butter(2, [f0 - 1, f0 + 1], btype='bandstop', fs=fs)

# Apply the bandstop filter to the voltage signal
voltage_filtered = filtfilt(b, a, voltage)

# Get the number of samples (N)
N = len(voltage_filtered)

# Calculate Autocorrelation function (ACF)
ac = np.zeros(N-1)  # Prepare the array for autocorrelation
m = np.mean(voltage_filtered)  # Mean of the data

# Compute the autocorrelation
for j in range(1, N):
    ac[j-1] = np.mean((voltage_filtered[0:N-j] - m) * (voltage_filtered[j:N] - m))

# Exponential decay fitting using a linear fit on log(ACF)
rangemax = 30  # Limit of data to use for fitting

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
plt.plot(x, np.exp(slope * x + intercept), linewidth=2, label='Fit - exp(-t/τ)')

# Display the value of tau on the graph
plt.text(2, 0.5, f"$\\tau = {tau:.3f}$", fontsize=12, color='blue')

plt.xlabel(r'Time [ms]', fontsize=14)
plt.ylabel(r'$C(t)\ [V^2]$', fontsize=14)
plt.legend(loc='best', fontsize=14)
plt.title(r'Correlation function with Exponential Fit (Semi-Log scale)', fontsize=14)
plt.show()

# Plot the ACF with the exponential fit in linear scale for a larger range
rangemax = 500  # Increase range for linear plot

plt.figure(figsize=(10, 6))
# Use 't' for the time values in the plot instead of np.arange(rangemax)
plt.plot(t[:rangemax], ac[:rangemax], 'ro', markersize=3, label='Exp data')
plt.plot(t[:rangemax], np.exp(slope * t[:rangemax] + intercept), linewidth=2, label='Fit - exp(-t/τ)')

# Display the value of tau on the graph
plt.text(0.1, 0.2, f"$\\tau = {tau:.3f}$", fontsize=12, color='blue')

plt.xlabel(r'Time [ms]', fontsize=14)
plt.ylabel(r'$C(t)\ [V^2]$', fontsize=14)
plt.legend(loc='best', fontsize=14)
plt.title(r'Correlation function with Exponential Fit (Linear scale)', fontsize=14)
plt.show()
