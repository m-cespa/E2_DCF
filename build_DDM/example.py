from DDM_Fourier import DDM_Fourier
import os

# Set environment variables for OpenCV
os.environ['OPENCV_LOG_LEVEL'] = 'FATAL'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = "-8"

filepath = r"C:\Users\Demonstrators\Desktop\DDM_E2b\github_clone\python\Data\ballistic_10s\2micron_x10.avi"

# video = cv2.VideoCapture(filepath)

# if not video.isOpened():
#     print("Failed to open video")
# else:
#     print("Video opened successfully")

pixel_size = 0.229
example = DDM_Fourier(filepath=filepath, pixel_size=pixel_size, particle_size=0.75)

# number of points to sample between each power of 10 in time intervals
pointsPerDecade = 30
# recommended values: 10 for speed, 300 for accuracy
maxNCouples = 10

# generate list of indices log spaced
idts = example.logSpaced(pointsPerDecade)

example.calculate_isf(idts, maxNCouples, plot_heat_map=False)

ISF = example.isf

example.FFT_temporal(ISF, q_selected=0.7)

# example.BrownianCorrelation(ISF, beta_guess=1.)

# example.BrownianCorrelationSubDiffusive(ISF, q_fixed=1)

# example.TwoParticleCorrelation(ISF)
