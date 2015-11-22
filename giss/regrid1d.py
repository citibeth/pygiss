import bisect
import numpy as np

# Simple conservative regridding for constant-valued regions.

def integrate_weights(dz, z0, z1)
	"""dz: Depth of each layer"

	top = np.cumsum(dz)		# Top of each layer
	ix = []
	weights = []

	i0 = biset.bisect_left(top, z0)
	i1 = biset.bisect_left(top, z1)

	print(i0, i1)