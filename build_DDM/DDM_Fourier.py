import numpy as np
from ImageStack import ImageStack
from typing import List
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import leastsq
from matplotlib.widgets import SpanSelector
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt

kB = 1.38e-23  # Boltzmann constant (J/K)
T = 298         # Temperature (K)
mu = 8.9e-4       # Viscosity (Pa.s)

class RadialAverager(object):
    """Radial average of a 2D array centred on (0,0), like the result of fft2d."""
    def __init__(self, shape):
        """A RadialAverager instance can process only arrays of a given shape, fixed at instanciation."""
        assert len(shape)==2
        #matrix of distances
        self.dists = np.sqrt(np.fft.fftfreq(shape[0])[:,None]**2 +  np.fft.fftfreq(shape[1])[None,:]**2)
        #dump the cross
        self.dists[0] = 0
        self.dists[:,0] = 0
        #discretize distances into bins
        self.bins = np.arange(max(shape)/2+1)/float(max(shape))
        #number of pixels at each distance
        self.hd = np.histogram(self.dists, self.bins)[0]
    
    def __call__(self, im):
        """Perform and return the radial average of the specrum 'im'"""
        assert im.shape == self.dists.shape
        hw = np.histogram(self.dists, self.bins, weights=im)[0]
        return hw/self.hd
    
class DDM_Fourier:
    def __init__(self, filepath: str, pixel_size: float, particle_size: float, renormalise=False):
        # create the stack attribute
        self.stack = ImageStack(filepath)
        # create the numpy array preloaded stack attribute
        self.frames = self.stack.pre_load_stack(renormalise=renormalise)

        self.pixel_size = pixel_size
        self.particle_size = particle_size
        self.fps = self.stack.fps
        self.frame_count = self.stack.frame_count

    def spectrumDiff(self, im0, im1) -> np.ndarray:
        """
        Computes squared modulus of 2D Fourier Transform of difference between im0 and im1
        Args:
            im0: matrix type object for frame at time t
            im1: matrix type object for frame at time t + tau
        """
        return np.abs(np.fft.fft2(im1-im0.astype(float)))**2

    def timeAveraged(self, dframes: int, maxNCouples: int=20):
        """
        Does at most maxNCouples spectreDiff on regularly spaced couples of images. 
        Args:
            dframes: interval between frames (integer)
            maxNCouples: maximum number of couples to average over
        """
        # create array of initial times (the 'im0' in spectrumDiff) of length maxNCouples AT MOST
        # evenly spaced in increments of 'increment'
        increment = max([(self.frames.shape[0] - dframes) / maxNCouples, 1])
        initialTimes = np.arange(0, self.frames.shape[0] - dframes, increment)

        avgFFT = np.zeros(self.frames.shape[1:])
        failed = 0
        for t in initialTimes:
            if t + dframes > self.frame_count - 1:
                failed += 1
                continue

            t = np.floor(t).astype(int)

            im0 = self.frames[t]
            im1 = self.frames[t+dframes]
            if im0 is None or im1 is None:
                failed +=1
                continue
            avgFFT += self.spectrumDiff(im0, im1)
        return avgFFT / (initialTimes.size - failed)
    
    def logSpaced(self, pointsPerDecade: int=15) -> List[int]:
        """Generate an array of log spaced integers smaller than frame_count"""
        nbdecades = np.log10(self.frame_count)

        return np.unique(np.logspace(
            start=0, stop=nbdecades, 
            num=int(nbdecades * pointsPerDecade), 
            base=10, endpoint=False
            ).astype(int))
    
    def calculate_isf(self, idts: List[float], maxNCouples: int = 1000, plot_heat_map: bool=False) -> np.ndarray:
        """
        Perform time-averaged and radial-averaged DDM for given time intervals.
        Returns ISF (Intermediate Scattering Function).

        Args:
            idts: List of integer rounded indices (within range) to specify
                which frames to time-average between
            maxNCouples: Maximum number of pairs to perform time averaging over
            n_jobs: Number of parallel jobs to run (set to -1 for all cores)
        """
        # create instance of radial averager callable
        ra = RadialAverager(self.stack.shape)

        print("\nStarting the parallelised ISF calculation...")
        
        # parallelise the time averaging
        with Parallel(n_jobs=-1, backend='threading') as parallel:
            time_avg_results = parallel(delayed(self.timeAveraged)(idt, maxNCouples) for idt in idts)

        print("\nTime Averaged Spectral Differences completed...")
        print("\nCalculating Radial Average for each tau time average...")

        # parallelize the radial averaging
        with Parallel(n_jobs=-1, backend='threading') as parallel:
            isf = np.array(parallel(delayed(ra)(ta) for ta in time_avg_results))

        self.isf = isf

        qs = 2*np.pi/(2*isf.shape[-1]*self.pixel_size) * np.arange(isf.shape[-1])
        self.qs = qs

        dts = idts / self.fps
        self.dts = dts

        # if plotting feature is enabled, a heatmap will be produced
        if plot_heat_map:

            plt.figure(figsize=(5,5))
            ISF_transposed = np.transpose(isf)
            plt.imshow(ISF_transposed, cmap='viridis', aspect='auto', extent=[dts[0], dts[-1], qs[-1], qs[0]], norm=LogNorm())
            plt.colorbar(label='I(q,$\\tau$),[a.u.]')
            plt.title('Image Structure Function I(q,$\\tau$)')
            plt.xlabel('Lag time ($\\tau$) [s]')
            plt.ylabel('Spatial Frequency (q) [$\\mu m ^{-1}$]')
            plt.savefig(f'{self.particle_size}μm_{self.fps}fps_ISFHeatmap.png')
            plt.show()
    
    def BrownianCorrelation(self, ISF, tmax=-1, beta_guess:float=1.):
        # Logarithmic form of the ISF function
        # take max value between evaluated log and 1e-10 to avoid 0 error in log
        # p_0 = A(q), p_1 = B(q), p_2 = tau(q)
        LogISF = lambda p, dts: np.log(np.maximum(p[0] * (1 - np.exp(-dts**beta_guess / p[2])) + p[1], 1e-10))

        # Initialize parameter array
        # intialise A(q) = peak-to-peak range of ISF
        #           B(q) = min value of ISF
        #           tau  = 1
        params = np.zeros((ISF.shape[-1], 3))
        for iq, ddm in enumerate(ISF[:tmax].T):
            params[iq] = leastsq(
                lambda p, dts, logd: LogISF(p, dts) - logd,
                [np.ptp(ISF), ddm.min(), 1],
                args=(self.dts[:tmax], np.log(ddm))
            )[0]

        # initialize selection range
        iqmin, iqmax = 0, self.qs.size - 1

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.qs, params[:, 2], 'o', label="Data")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$q\ [\mu m^{-1}]$')
        ax.set_ylabel(r'Characteristic time $\tau_c\ [s]$')
        ax.set_title('Click and drag to select a valid range of q values')

        alpha_text = ax.text(0.05, 0.95, "", transform=ax.transAxes, fontsize=12, verticalalignment='top')
        diameter_text = ax.text(0.05, 0.90, "", transform=ax.transAxes, fontsize=12, verticalalignment='top')

        def onselect(xmin, xmax):
            nonlocal iqmin, iqmax

            # convert selected range to indices
            iqmin = np.searchsorted(self.qs, xmin)
            iqmax = min(np.searchsorted(self.qs, xmax), len(self.qs) - 1)

            print(f"Selected range: {self.qs[iqmin]:.2f} to {self.qs[iqmax]:.2f}")

            # perform least squares fits for α and D
            fit_params = leastsq(
                lambda p, q, td: p[0] - p[1] * np.log(np.abs(q)) - np.log(np.abs(td)),
                [30, 2], # initial parameter guesses for log(D) and alpha
                args=(self.qs[iqmin:iqmax], params[iqmin:iqmax, 2])
            )[0]
            alpha = fit_params[1]

            D = np.exp(-leastsq(
                lambda p, q, td: p[0] - 2 * np.log(q) - np.log(td),
                [13],
                args=(self.qs[iqmin:iqmax], params[iqmin:iqmax, 2])
            )[0][0])

            # calculate diameter in µm
            predicted_a = kB * T / (3 * np.pi * mu * D) * 1e12 * 1e6

            alpha_text.set_text(rf"$\alpha = {alpha:.2f}$")
            diameter_text.set_text(rf"Diameter = {predicted_a:.2f} µm")

            plt.draw()

        # enable interactive selection
        span = SpanSelector(ax, onselect, 'horizontal', useblit=True, interactive=True, props=dict(alpha=0.5, facecolor='red'))

        plt.legend()
        plt.show()

    def BrownianCorrelationSubDiffusive(self, ISF, q_fixed:float=1.):
        """
        Extracts the autocorrelation (γ(q, t) function by reducing the ISF.
        For sub-diffusive motion, <x(t)^2> ~ t^β for 0 < β < 1.
            γ(q, t) ~ exp[-<x(t)^2>q^2]
        We expect that plotting γ(t) on logarithmic y-axis will NOT be linear.
        """
        B_q = ISF[0, :]
        A_q = ISF[-1, :] - B_q
        ISF_normalized = (ISF - B_q) / A_q
        autocorrelation = 1 - ISF_normalized

        q_idx = np.argmin(np.abs(self.qs - q_fixed))
        autocorr_q_idx = autocorrelation[:, q_idx]
                
        plt.figure(figsize=(8, 6))
        plt.plot(self.dts, autocorr_q_idx[:self.dts.size], label=f"q = {self.qs[q_idx]:.2f} µm⁻¹")
        plt.yscale('log')
        plt.xlabel('Time [s]')
        plt.ylabel('γ(t)')
        plt.title(f'γ(q, t) for q = {self.qs[q_idx]:.2f} µm⁻¹')
        plt.legend()
        plt.show()

    def FFT_temporal(self, ISF, q_selected):
        """
        Compute the Fourier transform of ISF along the time axis to extract velocity information.
        Identifies peaks above a threshold value and calculates <v_terminal> for each peak, 
        then averages the velocities and prints the result.
        """
        B_q = ISF[0, :]
        A_q = ISF[-1, :] - B_q
        ISF_normalized = (ISF - B_q) / A_q

        # γ(q,t) ~ exp(-t q^2 D) cos(q dot v t)
        exp_cos_component = 1 - ISF_normalized

        q_idx = np.argmin(np.abs(self.qs - q_selected))

        plt.figure(figsize=(8, 6))
        plt.plot(self.dts, exp_cos_component[:, q_idx])
        plt.xlabel('Time [s]')
        plt.ylabel('γ(q,t)')
        plt.title(f'γ(q,t) q = {self.qs[q_idx]} μm$^{{-1}}$')
        plt.grid(True)
        plt.show()
        
        # band-stop filter design (40-50 Hz for mains noise)
        def band_stop_filter(dts, low_cutoff, high_cutoff, fs):
            # design a band-stop filter using butterworth
            nyquist = 0.5 * fs
            low = low_cutoff / nyquist
            high = high_cutoff / nyquist
            b, a = butter(4, [low, high], btype='bandstop')
            return b, a

        # Filter out the noise in the frequency domain (40-50 Hz)
        fs = 1 / (self.dts[1] - self.dts[0])  # Sampling frequency
        low_cutoff = 47.5  # Lower bound of the mains noise frequency band (40 Hz)
        high_cutoff = 48.5  # Upper bound of the mains noise frequency band (50 Hz)

        # Apply the filter to the time domain signal (exp_cos_component)
        b, a = band_stop_filter(self.dts, low_cutoff, high_cutoff, fs)

        # Filter the signal using filtfilt (applies the filter forwards and backwards to avoid phase distortion)
        exp_cos_component_filtered = np.copy(exp_cos_component)
        for i in range(exp_cos_component.shape[1]):
            exp_cos_component_filtered[:, i] = filtfilt(b, a, exp_cos_component[:, i])

        # Plot the filtered signal
        plt.figure(figsize=(8, 6))
        plt.plot(self.dts, exp_cos_component_filtered[:, q_idx])
        plt.xlabel('Time [s]')
        plt.ylabel('γ(q,t)')
        plt.title(f'γ(q,t) - Filtered q = {self.qs[q_idx]} μm$^{{-1}}$')
        plt.grid(True)
        plt.show()

        # Step 5: Apply FFT to the filtered time-domain signal: exp(-t q^2 D) cos(q dot v t)
        ft_exp_cos_component_filtered = np.fft.fft(exp_cos_component_filtered, axis=0)
        
        # Shift the Fourier transform for correct ordering
        ft_exp_cos_component_shifted = np.fft.fftshift(ft_exp_cos_component_filtered, axes=0)
        
        # Get temporal frequencies
        freqs = np.fft.fftfreq(len(self.dts), d=self.dts[1] - self.dts[0])
        freqs_shifted = np.fft.fftshift(freqs)

        # Step 6: Plot the Fourier transform after filtering
        ft_exp_cos_component_q_filtered = ft_exp_cos_component_shifted[q_idx, :]
        abs_spectrum_q_filtered = np.abs(ft_exp_cos_component_q_filtered)

        # Ensure the lengths match
        if len(freqs_shifted) != len(abs_spectrum_q_filtered):
            min_len = min(len(freqs_shifted), len(abs_spectrum_q_filtered))
            freqs_shifted = freqs_shifted[:min_len]
            abs_spectrum_q_filtered = abs_spectrum_q_filtered[:min_len]
        
        plt.figure(figsize=(8, 6))
        plt.plot(freqs_shifted, abs_spectrum_q_filtered)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Magnitude of Fourier Transform (Filtered)')
        plt.title(f'Fourier Transform After Band-Stop Filtering q = {self.qs[q_idx]} μm$^{{-1}}$')
        plt.grid(True)
        plt.show()
    
    def LogISF_2particles(self, p, dts):
        """
        Logarithmic form of the ISF function for two particles with different sizes.
        p[0] = A1 (Amplitude of particle 1)
        p[1] = A2 (Amplitude of particle 2)
        p[2] = tau1 (Time constant for particle 1)
        p[3] = tau2 (Time constant for particle 2)
        p[4] = B (Baseline term)
        """
        A1, A2, tau1, tau2, B = p
        return np.log(np.maximum(A1 * (1 - np.exp(-dts / tau1)) + A2 * (1 - np.exp(-dts / tau2)) + B, 1e-10))
            
    def TwoParticleCorrelation(self, ISF, bottom:float=0., top:float=50., tmax=-1):
        """
        Fit the ISF data to the theoretical two-particle ISF function.
        
        Args:
            ISF: 2D ISF data array shape len(times) x len(qs)
            bottom: bottom limit to set for filtering taus
            top: top limit to set for filtering taus
            tmax: maximum number of time points to use for the fitting
        """
        # initialise fit parameter array
        # dims: len(qs) x 5 (each column stores a parameter through all q)
        params = np.zeros((ISF.shape[-1], 5))  # [A1, A2, tau1, tau2, B]

        # for each q we extract ISF(t) here labelled "ddm"
        for iq, ddm in enumerate(ISF[:tmax].T):
            # initial guess for parameters [A1, A2, tau1, tau2, B]
            initial_params = [np.ptp(ddm), np.ptp(ddm) * 0.5, 1.0, 2.0, np.mean(ddm)]

            # perform least squares fit at current q value
            # least squares will optimise [A1, A2, tau1, tau2, B]
            # to minimise difference between log(ISF)_observed & log(ISF)_fit
            params[iq] = leastsq(
                lambda p, dts, logd: self.LogISF_2particles(p, dts) - logd,
                initial_params,
                args=(self.dts[:tmax], np.log(ddm))
            )[0]

        # retrieve the tau parameters for all q
        tau1_vals = params[:, 2]
        tau2_vals = params[:, 3]
        

        # apply mask to remove erroneously large or negative q values
        valid_indices = ((tau1_vals >= bottom) & (tau1_vals <= top)) & ((tau2_vals >= bottom) & (tau2_vals <= top))
        tau1_filtered = tau1_vals[valid_indices]
        tau2_filtered = tau2_vals[valid_indices]
        qs_filtered   = self.qs[valid_indices]

        # graph selection range
        iqmin, iqmax = 0, qs_filtered.size - 1

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(qs_filtered, tau1_filtered, 'o', label=r"$\tau_1(q)$", color='red')
        ax.plot(qs_filtered, tau2_filtered, 'o', label=r"$\tau_2(q)$", color='blue')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$q\ [\mu m^{-1}]$')
        ax.set_ylabel(r'Characteristic times $\tau_1, \tau_2\ [s]$')
        ax.set_title('Click and drag to select a valid range of q values')

        def onselect(xmin, xmax):
            nonlocal iqmin, iqmax

            # convert selected range to indices
            iqmin = np.searchsorted(qs_filtered, xmin)
            iqmax = min(np.searchsorted(qs_filtered, xmax), qs_filtered.size - 1)

            print(f"Selected range: {qs_filtered[iqmin]:.2f} to {qs_filtered[iqmax]:.2f}")

            # perform least squares fits for alpha and diameter
            # expect tau = 1 / Dq^2 -> log(tau) = -log(D) - 2log(q)
            # p[0] is fitting for -log(D)
            # p[1] is fitting for 2 (alpha)
            fit_params_tau1 = leastsq(
                lambda p, q, td: p[0] - p[1] * np.log(np.abs(q)) - np.log(np.abs(td)),
                [30, 2],
                args=(qs_filtered[iqmin:iqmax], tau1_filtered[iqmin:iqmax])
            )[0]
            
            fit_params_tau2 = leastsq(
                lambda p, q, td: p[0] - p[1] * np.log(np.abs(q)) - np.log(np.abs(td)),
                [30, 2],
                args=(qs_filtered[iqmin:iqmax], tau2_filtered[iqmin:iqmax])
            )[0]

            # fit the Diffusivities by enforcing alpha = 2
            D1 = np.exp(-leastsq(
                lambda p, q, td: p[0] - 2 * np.log(q) - np.log(td),
                [13.],
                args=(qs_filtered[iqmin:iqmax], tau1_filtered[iqmin:iqmax])
            )[0][0])

            D2 = np.exp(-leastsq(
                lambda p, q, td: p[0] - 2 * np.log(q) - np.log(td),
                [13.],
                args=(qs_filtered[iqmin:iqmax], tau2_filtered[iqmin:iqmax])
            )[0][0])

            alpha1 = fit_params_tau1[1]
            alpha2 = fit_params_tau2[1]

            # diameters calculated in μm
            diameter1 = kB * T / (3 * np.pi * mu * D1) * 1e12 * 1e6
            diameter2 = kB * T / (3 * np.pi * mu * D2) * 1e12 * 1e6

            # plot data with scrollable range selection
            ax.clear()
            ax.plot(qs_filtered, tau1_filtered, 'o', label=r"$\tau_1(q)$", color='red')
            ax.plot(qs_filtered, tau2_filtered, 'o', label=r"$\tau_2(q)$", color='blue')
            ax.axvspan(qs_filtered[iqmin], qs_filtered[iqmax], color=(0.9, 0.9, 0.9))

            # reapply axis settings
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$q\ [\mu m^{-1}]$')
            ax.set_ylabel(r'Characteristic times $\tau_1, \tau_2\ [s]$')
            ax.set_title('Click and drag to select a valid range of q values')

            # recreate annotation text objects with updated values
            ax.text(0.05, 0.95, rf"$\alpha_1 = {alpha1:.2f}, \alpha_2 = {alpha2:.2f}$",
                    transform=ax.transAxes, fontsize=12, verticalalignment='top')
            ax.text(0.05, 0.90, rf"Diameter 1 = {diameter1:.2f} µm, Diameter 2 = {diameter2:.2f} µm",
                    transform=ax.transAxes, fontsize=12, verticalalignment='top')

            ax.legend()
            plt.draw()

        # enable interactive selection of q range
        span = SpanSelector(ax, onselect, 'horizontal', useblit=True, interactive=True, props=dict(alpha=0.5, facecolor='red'))

        plt.legend()
        plt.show()

        # we theorise that the ratio of A1(q) : A2(q) corresponds to the concentrations
        conc_ratio = np.mean(params[:, 0] / params[:, 1])
        print(f"\nRecovered concentration ratio from A1(q)/A2(q) = {conc_ratio}")

        # plot for a selected value the reduced ISF

        B_q = ISF[0, :]
        ISF_reduced = ISF - B_q

        q_value = float(input("\nPlease enter a q value within the previously selected range: "))

        q_idx = np.argmin(np.abs(self.qs - q_value))
        
        dts_plot = self.dts[:tmax]
        ISF_measured = ISF_reduced[:tmax, q_idx]
        
        # retrieve fit parameters from before
        A1, A2, tau1, tau2, _ = params[q_idx, :]
        
        # compute reduced ISF using fit parameters
        # ISF_model(t) = A1 * (1 - exp(-t/tau1)) + A2 * (1 - exp(-t/tau2))
        fitted_curve = A1 * (1 - np.exp(-dts_plot / tau1)) + A2 * (1 - np.exp(-dts_plot / tau2))
        
        plt.figure(figsize=(8, 6))
        plt.plot(dts_plot, ISF_measured, 'o', label='Reduced ISF data', color='blue')
        plt.plot(dts_plot, fitted_curve, '-', label='Fitted curve', color='red')
        plt.xlabel('Time [s]')
        plt.ylabel('Reduced ISF')
        plt.title(rf'Reduced ISF at $q = {self.qs[q_idx]:.2f}\ \mu m^{{-1}}$')
        plt.legend(loc='center right')
        plt.text(0.95, 0.05, 
         rf"$\tau_1 = {round(tau1,2)}\,\mathrm{{s}} \mid \tau_2 = {round(tau2,2)}\,\mathrm{{s}}$", 
         ha='right', va='bottom', transform=plt.gca().transAxes)
        plt.show()
