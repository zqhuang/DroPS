import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

# Define parameters
NSIDE = 16 # Resolution parameter
vec = hp.ang2vec(np.pi / 2, np.pi * 3 / 4) # Direction vector (latitude, longitude)
radius = np.radians(10) # Radius in radians
print(radius, 10*np.pi/180.)
# Find pixels within the disk
ipix_disc = hp.query_disc(nside=NSIDE, vec=vec, radius=radius)

# Create a map and highlight the queried pixels
NPIX = hp.nside2npix(NSIDE)
m = np.arange(NPIX)
m[ipix_disc] = m.max()

# Visualize the result
hp.mollview(m, title="Mollview image with queried disk")
plt.show()

