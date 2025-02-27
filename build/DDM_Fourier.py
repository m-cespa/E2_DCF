import numpy as np
from ImageStack import ImageStack
from typing import List
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import leastsq
from matplotlib.widgets import SpanSelector

kB = 1.38e-23  # Boltzmann constant (J/K)
T = 298         # Temperature (K)
mu = 1e-3       # Viscosity (Pa.s)

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
    
    def BrownianCorrelation(self, ISF, tmax=-1):
        # Logarithmic form of the ISF function
        LogISF = lambda p, dts: np.log(np.maximum(p[0] * (1 - np.exp(-dts / p[2])) + p[1], 1e-10))

        # Initialize parameter array
        params = np.zeros((ISF.shape[-1], 3))
        for iq, ddm in enumerate(ISF[:tmax].T):
            params[iq] = leastsq(
                lambda p, dts, logd: LogISF(p, dts) - logd,  # Function to minimize
                [np.ptp(ISF), ddm.min(), 1],  # Initial parameters
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
                [1, 2],
                args=(self.qs[iqmin:iqmax], params[iqmin:iqmax, 2])
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
