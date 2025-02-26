from ImageStack import ImageStack
from DDM_Fourier import DDM_Fourier

filepath = 'example_path'
pixel_size = 0.229
example = DDM_Fourier(filepath=filepath, pixel_size=pixel_size, particle_size=0.75)

# number of points to sample between each power of 10 in time intervals
pointsPerDecade = 30
# recommended values: 10 for speed, 300 for accuracy
maxNCouples = 10

# generate list of indices log spaced
idts = example.logSpaced(pointsPerDecade)

example.calculate_isf(idts, maxNCouples, plot_heat_map=True)

ISF = example.isf

example.BrownianCorrelation(ISF)
