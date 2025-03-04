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
    def __init__(self, filepath: str, pixel_size: float, particle_size: float):
        # create the stack attribute
        self.stack = ImageStack(filepath)
        # create the numpy array preloaded stack attribute
        self.frames = self.stack.pre_load_stack()

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

        # if plotting feature is enabled, a heatmap will be graphed
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
                lambda p, dts, logd: LogISF(p, dts) - logd,  # Function to minimize
                [np.ptp(ISF), ddm.min(), 1],  # initial parameter values (p to sub into LogISF)
                args=(self.dts[:tmax], np.log(ddm))  # Data on which to perform minimization
            )[0]

        # Initialize selection range
        iqmin, iqmax = 0, len(self.qs) - 1

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.qs, params[:, 2], 'o', label="Data")
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$q\ [\mu m^{-1}]$')
        ax.set_ylabel(r'Characteristic time $\tau_c\ [s]$')
        ax.set_title('Click and drag to select a valid range of q values')

        # Annotation for alpha and diameter
        alpha_text = ax.text(0.05, 0.95, "", transform=ax.transAxes, fontsize=12, verticalalignment='top')
        diameter_text = ax.text(0.05, 0.90, "", transform=ax.transAxes, fontsize=12, verticalalignment='top')

        def onselect(xmin, xmax):
            nonlocal iqmin, iqmax  # Avoid using global variables

            # Convert selected range to indices
            iqmin = np.searchsorted(self.qs, xmin)
            iqmax = min(np.searchsorted(self.qs, xmax), len(self.qs) - 1)  # Prevent out-of-bounds

            print(f"Selected range: {self.qs[iqmin]:.2f} to {self.qs[iqmax]:.2f}")

            # Perform least squares fits for α and D
            fit_params = leastsq(
                lambda p, q, td: p[0] - p[1] * np.log(np.abs(q)) - np.log(np.abs(td)),
                [1, 2], # initial parameter guesses for log(D) and alpha
                args=(self.qs[iqmin:iqmax], params[iqmin:iqmax, 2]) # params[:, 2] is Dq^2 fit
            )[0]
            alpha = fit_params[1]

            D = np.exp(-leastsq(
                lambda p, q, td: p[0] - 2 * np.log(q) - np.log(td),
                [-1.],
                args=(self.qs[iqmin:iqmax], params[iqmin:iqmax, 2])
            )[0][0])

            # Calculate diameter
            predicted_a = kB * T / (3 * np.pi * mu * D) * 1e12 * 1e6  # Convert to µm

            # Update annotation text
            alpha_text.set_text(rf"$\alpha = {alpha:.2f}$")
            diameter_text.set_text(rf"Diameter = {predicted_a:.2f} µm")

            # Redraw the plot to update the text
            plt.draw()

        # Enable interactive selection
        span = SpanSelector(ax, onselect, 'horizontal', useblit=True, interactive=True, props=dict(alpha=0.5, facecolor='red'))

        plt.legend()
        plt.show()

    def BrownianCorrelationSubDiffusive(self, ISF, tmax=-1, q_fixed:float=1.):
        B_q = ISF[0, :]
        A_q = ISF[-1, :] - B_q
        ISF_normalized = (ISF - B_q) / A_q
        autocorrelation = 1 - ISF_normalized

        log_log_auto = np.log(-np.log(autocorrelation))
        
        # Find the closest index corresponding to the selected q_fixed
        q_index = np.argmin(np.abs(self.qs - q_fixed))
        log_log_auto_q = log_log_auto[:, q_index]
                
        # Plot log_auto vs time on a logarithmic axis
        plt.figure(figsize=(8, 6))
        plt.plot(self.dts[:tmax], log_log_auto_q[:tmax], label=f"q = {self.qs[q_index]:.2f} µm⁻¹", color='blue')
                
        # Set the plot labels and title
        plt.xlabel('log(t)')
        plt.ylabel('log(log(Autocorrelation))')
        plt.title(f'log(log(gamma)) vs log(t) for q = {self.qs[q_index]:.2f} µm⁻¹')
        
        # Display the plot
        plt.legend()
        plt.show()

    def FFT_temporal(self, ISF, q_selected):
        """
        Compute the Fourier transform of ISF along the time axis to extract velocity information.
        Identifies peaks above a threshold value and calculates <v_terminal> for each peak, 
        then averages the velocities and prints the result.
        """
        # Step 1: Extract B(q) from the t = 0 intercept for each q
        B_q = ISF[0, :]  # ISF at t=0 gives B(q) for each q
        
        # Step 2: Extract A(q) using the long-time behavior
        A_q = ISF[-1, :] - B_q  # Assuming ISF(t -> infinity) is approximated by ISF at large t
        
        # Step 3: Recover 1 - (ISF - B(q)) / A(q)
        ISF_normalized = (ISF - B_q) / A_q  # Normalized ISF without the static component
        
        # Step 4: Calculate exp(-t q^2 D) cos(q dot v t) by 1 - the normalized ISF
        exp_cos_component = 1 - ISF_normalized  # This is exp(-t q^2 D) cos(q dot v t)

        # Get the index of the smallest non-zero q
        q_idx = np.argmin(np.abs(self.qs - q_selected))

        # Plot the original renormalized autocorrelation
        plt.figure(figsize=(8, 6))
        plt.plot(self.dts, exp_cos_component[:, q_idx])
        plt.xlabel('Time [s]')
        plt.ylabel('Gamma(t, q_selected)')
        plt.title(f'Renormalized Autocorrelation q = {self.qs[q_idx]} μm$^{{-1}}$')
        plt.grid(True)
        plt.show()
        
        # Band-stop filter design (40-50 Hz for mains noise)
        def band_stop_filter(dts, low_cutoff, high_cutoff, fs):
            # Design a band-stop filter using butterworth
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
        plt.ylabel('Gamma(t, q_selected) - Filtered')
        plt.title(f'Renormalized Autocorrelation (Filtered) q = {self.qs[q_idx]} μm$^{{-1}}$')
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
        
        # Plot the filtered Fourier transform spectrum
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
    
    def compute_viscoelastic_moduli(self, ISF, q_selected: float=1.):

            if not hasattr(self, 'isf') or not hasattr(self, 'dts'):
                raise ValueError("ISF and dts must be calculated first. Run calculate_isf() method.")
            
            q_index = np.argmin(np.abs(self.qs - q_selected))
            q_val = self.qs[q_index]
            
            B_q = ISF[0, :]
            A_q = ISF[-1, :] - B_q

            ISF_normalised = (ISF - B_q) / A_q

            f_qs = 1 - ISF_normalised
            f = f_qs[:, q_index]

            # Compute MSD from f(q,τ) = exp(-1/q^2 MSD)
            msd = - 1/(q_val**2) * np.log(f)

            msd = msd[np.isfinite(msd)]

            print(f"MSD = {msd}")

            # Perform FFT on MSD to get frequency domain representation
            msd_fft = np.fft.fft(msd)

            # Generate frequency array corresponding to the FFT output
            dt = self.dts[1] - self.dts[0]  # Assuming uniform time spacing
            freqs = np.fft.fftfreq(len(msd), dt)  # Frequencies in Hz

            # Convert frequency (Hz) to omega (rad/s)
            omega = 2 * np.pi * freqs

            # We want to ignore negative frequencies in the FFT result
            positive_freqs = freqs > 0
            msd_fft = msd_fft[positive_freqs]
            omega = omega[positive_freqs]

            # G*(omega) = 2 * kB * T / (pi * a * omega * F_s) (for viscoelastic modulus)
            # Here we compute the real and imaginary components
            G_star = 2 * kB * T / (np.pi * self.particle_size * omega * msd_fft)

            G_prime = np.real(G_star)  # Storage modulus (real part)
            G_doubleprime = np.imag(G_star)  # Loss modulus (imaginary part)

            print(f"Frequency (omega): {omega[:10]}")
            print(f"G' (Storage modulus) [first 10 values]: {G_prime[:10]}")
            print(f"G'' (Loss modulus) [first 10 values]: {G_doubleprime[:10]}")

            # Plotting the results
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(omega, G_prime, label=r"$G'(\omega)$ (Storage Modulus)")
            ax.plot(omega, G_doubleprime, label=r"$G''(\omega)$ (Loss Modulus)")
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$\omega$ (rad/s)')
            ax.set_ylabel(r'$G(\omega)$')
            ax.legend()
            plt.grid(True)
            plt.show()
            
    def TwoParticleCorrelation(self, ISF, tmax=-1):
        """
        Fit the ISF data to the theoretical two-particle ISF function.
        
        Args:
        - ISF: 2D ISF data array
        - tmax: maximum number of time points to use for the fitting
        """
        # Initialize parameter array to store the fitted parameters
        params = np.zeros((ISF.shape[-1], 5))  # [A1, A2, tau1, tau2, B]
        for iq, ddm in enumerate(ISF[:tmax].T):
            # Initial guess for the parameters [A1, A2, tau1, tau2, B]
            initial_params = [np.ptp(ddm), np.ptp(ddm) * 0.5, 1.0, 2.0, np.mean(ddm)]  # Initial guess

            # Perform least squares fit
            params[iq] = leastsq(
                lambda p, dts, logd: self.LogISF_2particles(p, dts) - logd,
                initial_params,  # Initial guess for parameters
                args=(self.dts[:tmax], np.log(ddm))  # Data for minimization
            )[0]

        # Extract tau1 and tau2 (characteristic times for the two particles)
        tau1_vals = params[:, 2]
        tau2_vals = params[:, 3]

        # Initialize selection range
        iqmin, iqmax = 0, len(self.qs) - 1

        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.qs, tau1_vals, 'o', label=r"$\tau_1(q)$", color='red')
        ax.plot(self.qs, tau2_vals, 'o', label=r"$\tau_2(q)$", color='blue')
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$q\ [\mu m^{-1}]$')
        ax.set_ylabel(r'Characteristic times $\tau_1, \tau_2\ [s]$')
        ax.set_title('Click and drag to select a valid range of q values')

        # Annotation for alpha and particle size
        alpha_text = ax.text(0.05, 0.95, "", transform=ax.transAxes, fontsize=12, verticalalignment='top')
        size_text = ax.text(0.05, 0.90, "", transform=ax.transAxes, fontsize=12, verticalalignment='top')

        def onselect(xmin, xmax):
            nonlocal iqmin, iqmax  # Avoid using global variables

            # Convert selected range to indices
            iqmin = np.searchsorted(self.qs, xmin)
            iqmax = min(np.searchsorted(self.qs, xmax), len(self.qs) - 1)  # Prevent out-of-bounds

            print(f"Selected range: {self.qs[iqmin]:.2f} to {self.qs[iqmax]:.2f}")

            # Perform least squares fits for alpha and diameter (size)
            fit_params_tau1 = leastsq(
                lambda p, q, td: p[0] - p[1] * np.log(np.abs(q)) - np.log(np.abs(td)),
                [1, 2],
                args=(self.qs[iqmin:iqmax], tau1_vals[iqmin:iqmax])
            )[0]
            
            fit_params_tau2 = leastsq(
                lambda p, q, td: p[0] - p[1] * np.log(np.abs(q)) - np.log(np.abs(td)),
                [1, 2],
                args=(self.qs[iqmin:iqmax], tau2_vals[iqmin:iqmax])
            )[0]

            alpha1 = fit_params_tau1[1]
            alpha2 = fit_params_tau2[1]

            # Calculate the particle sizes (diameters) for tau1 and tau2
            D1 = np.exp(-fit_params_tau1[0])  # Diffusion coefficient
            D2 = np.exp(-fit_params_tau2[0])

            diameter1 = kB * T / (3 * np.pi * mu * D1) * 1e12 * 1e6  # Convert to µm
            diameter2 = kB * T / (3 * np.pi * mu * D2) * 1e12 * 1e6  # Convert to µm

            # Plot and extract information for the selected range
            ax.clear()
            ax.plot(self.qs, tau1_vals, 'o', label=r"$\tau_1(q)$", color='red')
            ax.plot(self.qs, tau2_vals, 'o', label=r"$\tau_2(q)$", color='blue')
            ax.axvspan(self.qs[iqmin], self.qs[iqmax], color=(0.9, 0.9, 0.9))  # Highlight selected range

            # Update plot with new selected range
            ax.legend()
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(r'$q\ [\mu m^{-1}]$')
            ax.set_ylabel(r'Characteristic times $\tau_1, \tau_2\ [s]$')

            # Update annotation text with alpha and diameter
            alpha_text.set_text(rf"$\alpha_1 = {alpha1:.2f}, \alpha_2 = {alpha2:.2f}$")
            size_text.set_text(rf"Diameter 1 = {diameter1:.2f} µm, Diameter 2 = {diameter2:.2f} µm")

            # Redraw the plot to update the text
            plt.draw()

        # Enable interactive selection of q range
        span = SpanSelector(ax, onselect, 'horizontal', useblit=True, interactive=True, props=dict(alpha=0.5, facecolor='red'))

        plt.legend()
        plt.show()



