import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import butter, filtfilt
import os

# Function to compute tau and its standard deviation from a .prn file
def compute_tau_from_file(filename):
    # Load Data
    data = np.loadtxt(filename, delimiter='\t', skiprows=3)  # Skips first 3 rows
    t = data[:, 0]  # Time
    voltage = data[:, 1]  # Voltage

    # Bandstop filter to remove 50 Hz noise
    fs = 1000 / (t[1] - t[0])  # Sampling frequency
    f0 = 100  # Frequency to remove (50 Hz)
    Q = 10  # Quality factor: Controls the width of the notch

    # Create the bandstop filter
    b, a = butter(2, [f0 - 1, f0 + 1], btype='bandstop', fs=fs)

    # Apply the bandstop filter to the voltage signal
    voltage_filtered = filtfilt(b, a, voltage)

    # Get the number of samples (N)
    N = len(voltage_filtered)

    # Calculate Autocorrelation function (ACF)
    ac = np.zeros(N-1)  # Prepare the array for autocorrelation
    m = np.mean(voltage_filtered)  # Mean of the filtered data

    # Compute the autocorrelation
    for j in range(1, N):
        ac[j-1] = np.mean((voltage_filtered[0:N-j] - m) * (voltage_filtered[j:N] - m))

    # Exponential decay fitting using a linear fit on log(ACF)
    rangemax = 20  # Limit of data to use for fitting

    # Prevent log(0) issues by adding a small value if needed
    ac_log = np.log(ac[:rangemax] + 1e-10)

    # Perform the linear fit on the log of the autocorrelation for the first 'rangemax' points
    x = np.arange(rangemax)
    y = ac_log

    # Linear fit: y = m * x + b
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Characteristic time of exponential decay
    tau = -1 / slope

    # Standard deviation of tau (calculated from the standard error of the slope)
    sigma_tau = abs(1 / slope**2) * std_err

    return tau, sigma_tau


# Function to process all .prn files in a folder and compute the weighted mean of tau
def process_folder_for_tau(folder_path):
    taus = []
    sigma_taus = []

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".prn"):
            file_path = os.path.join(folder_path, filename)

            # Compute tau and standard deviation of tau for the current file
            tau, sigma_tau = compute_tau_from_file(file_path)
            # Append tau and its standard deviation
            taus.append(tau)
            sigma_taus.append(sigma_tau)

    # Calculate weighted mean of tau
    weights = 1 / np.array(sigma_taus) ** 2
    weighted_mean_tau = np.sum(weights * np.array(taus)) / np.sum(weights)

    # Calculate the error in the weighted mean of tau
    squared_differences = (taus - weighted_mean_tau) ** 2 / (len(taus) - 1)
    error_in_weighted_mean = np.sqrt(np.sum(squared_differences))

    return weighted_mean_tau, error_in_weighted_mean


# Function to process all folders within a parent folder
def process_parent_folder(parent_folder_path):
    taus =[]
    rel_error_taus = []
    # Iterate over all subdirectories in the parent folder
    for folder_name in os.listdir(parent_folder_path):
        folder_path = os.path.join(parent_folder_path, folder_name)

        # Ensure that it's a directory (not a file)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_path}")

            # Process the folder and calculate the weighted mean and error for tau
            weighted_mean_tau, error_in_weighted_mean = process_folder_for_tau(folder_path)
            taus.append(weighted_mean_tau)
            rel_error_taus.append(error_in_weighted_mean/weighted_mean_tau)
        
    return taus, rel_error_taus

#calculating q errors

def compute_sin_error(a, del_a, b, del_b):
    # Calculate f(a, b)
    theta = np.arctan(a/b)
    del_theta = np.sqrt((a*del_b)**2+(b*del_a)**2)/(a**2 +b**2)
    sin = np.sin(0.5*theta)
    del_sin = 0.5*np.cos(0.5*theta)*del_theta
    return sin, del_sin

def q_and_error(parent_folder_path):
    q=[]
    error_q=[]

    for folder_name in os.listdir(parent_folder_path):
        a = float(folder_name)
        print(a)
        del_a = 0.2
        del_b = 0.3
        b = 5.5
        sin ,del_sin = compute_sin_error(a, del_a, b, del_b)
        const =4*np.pi*1.3/(635*(10**(-9)))
        q.append(const*sin)
        error_q.append(const*del_sin)
        rel_error_q = np.array(error_q) / np.array(q)

    return q, rel_error_q


def a_and_error(parent_folder_path):
    
    del_d = []
    q, rel_error_q = q_and_error(parent_folder_path)
    taus, rel_error_taus = process_parent_folder(parent_folder_path)
    c = 4.82*10**(-22)
    d = c*((np.array(q))**2)*np.array(taus)
    del_d = d*( (np.array(rel_error_q))**2 + (2*np.array(rel_error_taus))**2)**0.5
    return d, del_d

parent_folder_path = 'data-0.75mu-10^3'  

# Print the results for each folder
#for folder_name, result in results.items():
#    print(f"Folder: {folder_name}")
#   print(f"  Weighted mean of tau: {result['weighted_mean_tau']}")
#   print(f"  Error in the weighted mean of tau: {result['error_in_weighted_mean']}")

print( a_and_error(parent_folder_path))