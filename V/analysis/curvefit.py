import numpy, scipy, scipy.ndimage, scipy.interpolate, numpy.fft, math

# create simple square
img = numpy.zeros( (10, 10) )
img[1:9, 1:9] = 1
img[2:8, 2:8] = 0

# find contour
x, y = numpy.nonzero(img)

# find center point and conver to polar coords
x0, y0 = numpy.mean(x), numpy.mean(y)
C = (x - x0) + 1j * (y - y0)
angles = numpy.angle(C)
distances = numpy.absolute(C)
sortidx = numpy.argsort( angles )
angles = angles[ sortidx ]
distances = distances[ sortidx ]

# copy first and last elements with angles wrapped around
# this is needed so can interpolate over full range -pi to pi
angles = numpy.hstack(([ angles[-1] - 2*math.pi ], angles, [ angles[0] + 2*math.pi ]))
distances = numpy.hstack(([distances[-1]], distances, [distances[0]]))

# interpolate to evenly spaced angles
f = scipy.interpolate.interp1d(angles, distances)
angles_uniform = scipy.linspace(-math.pi, math.pi, num=100, endpoint=False)
distances_uniform = f(angles_uniform)

# fft and inverse fft
fft_coeffs = numpy.fft.rfft(distances_uniform)
# zero out all but lowest 10 coefficients
fft_coeffs[11:] = 0
distances_fit = numpy.fft.irfft(fft_coeffs)

# plot results
import matplotlib.pyplot as plt
plt.polar(angles, distances)
plt.polar(angles_uniform, distances_uniform)
plt.polar(angles_uniform, distances_fit)
plt.show()