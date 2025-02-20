import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy import stats
import os
import re
import csv

class DLS:
    def __init__(self, dir: str, noise_freq: float=100.):
        self.dir = dir
        self.noise_freq = noise_freq
        self.tau_vals = []

        # set sampling frequency
        for entry in os.scandir(dir):
            if entry.is_file():
                filename = entry.path
                data = np.loadtxt(filename, delimiter='\t', skiprows=3)
                self.sample_freq = 1000 / (data[1, 0] - data[0, 0])
                break
    
    def extract_degree_from_filename(self, filename: str):
        """
        Extract degree value from filename. Assumes format '26.6deg.prn'
        """
        match = re.search(r'(\d+\.\d+)deg', filename)
        if match:
            return match.group(1)
        else:
            return None
    
    def AutoCorrelation(self, lin_range: int=30, exp_range: int=500):
        """
        Args:
            lin_range: number of points to use for linear plot
            exp_range: number of points to use for exponential plot
        """
        for entry in os.scandir(self.dir):
            if entry.is_file():
                filename = entry.path
                
                data = np.loadtxt(filename, delimiter='\t', skiprows=3)
                t = data[:, 0]
                voltage = data[:, 1]

                # create bandstop filter
                b, a = butter(2, [self.noise_freq - 1, self.noise_freq + 1], btype='bandstop', fs=self.sample_freq)
                voltage_filtered = filtfilt(b, a, voltage)

                # perform autocorrelation
                N = len(voltage_filtered)
                ac = np.zeros(N-1)
                m = np.mean(voltage_filtered)

                for j in range(1, N):
                    ac[j-1] = np.mean((voltage_filtered[0:N-j] - m) * (voltage_filtered[j:N] - m))

                # fit logarithm of autocorrelation for tau
                x = np.arange(lin_range)
                y = np.log(ac[:lin_range])

                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                tau = -1 / slope

                # calculate error in tau from std
                error_tau = abs(1 / slope**2) * std_err

                # extract degree from the filename
                degree = self.extract_degree_from_filename(filename)
                if degree is not None:
                    self.tau_vals.append((degree, tau, error_tau))

                # plot results on log scale
                plt.figure(figsize=(10, 6))
                plt.semilogy(x, ac[:lin_range], 'ro', label='Exp data')
                plt.plot(x, np.exp(slope * x + intercept), linewidth=2, label='Fit - exp(-t/τ)')
                plt.text(2, 0.5, f"$\\tau = {tau:.3f}$", fontsize=12, color='blue')
                plt.xlabel(r'Time [ms]', fontsize=14)
                plt.ylabel(r'$C(t)\ [V^2]$', fontsize=14)
                plt.legend(loc='best', fontsize=14)
                plt.title(r'Correlation function with Exponential Fit (Semi-Log scale)', fontsize=14)
                plt.show()

                # plot results on linear scale
                plt.figure(figsize=(10, 6))
                plt.plot(t[:exp_range], ac[:exp_range], 'ro', markersize=3, label='Exp data')
                plt.plot(t[:exp_range], np.exp(slope * t[:exp_range] + intercept), linewidth=2, label='Fit - exp(-t/τ)')
                plt.text(0.1, 0.2, f"$\\tau = {tau:.3f}$", fontsize=12, color='blue')
                plt.xlabel(r'Time [ms]', fontsize=14)
                plt.ylabel(r'$C(t)\ [V^2]$', fontsize=14)
                plt.legend(loc='best', fontsize=14)
                plt.title(r'Correlation function with Exponential Fit (Linear scale)', fontsize=14)
                plt.show()

        dir_name = os.path.basename(self.dir)
        csv_filename = f"{dir_name}_taus.csv"

        # save tau values and associated errors to csv
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Degree', 'Tau', 'Error'])
            writer.writerows(self.tau_vals)
        print(f"Tau values and errors have been saved to '{csv_filename}'")

    def q_and_error(self):
        """
        Function to calculate q values and errors.
        """
        q_vals = []
        error_q_vals = []

        for entry in os.scandir(self.dir):
            if entry.is_file():
                filename = entry.path
                # extract numerical angle value from filename
                match = re.search(r'(\d+\.\d+)deg', filename)
                if match:
                    a = float(match.group(1))
                    q_vals.append(a)
                    
                    
                    # FILL WITH ERROR CALCULATION FOR Q


                    # example error calc
                    error_q_vals.append(a * 0.1)
                else:
                    print(f"Filename {filename} does not match the expected pattern.")

        return q_vals, error_q_vals

    def compute_sin_error(self, a, del_a, b, del_b):
        """
        Calculate sin of angle and associated error.
        """
        theta = np.arctan(a / b)
        del_theta = np.sqrt((a * del_b)**2 + (b * del_a)**2) / (a**2 + b**2)
        sin = np.sin(0.5 * theta)
        del_sin = 0.5 * np.cos(0.5 * theta) * del_theta
        return sin, del_sin

# Example usage of the class
dls = DLS('DLS_files/0.75micro_100dil')
dls.AutoCorrelation()  # Computes autocorrelation and saves tau and errors to CSV

# If you also want to calculate q and errors, use:
q_vals, error_q_vals = dls.q_and_error()
print(f"Q values: {q_vals}")
print(f"Error in Q values: {error_q_vals}")
