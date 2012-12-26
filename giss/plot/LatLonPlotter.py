import numpy as np

class LatLonPlotter :
	"""A plotter for lat/lon GCM grid cell data.

	Plotters provide a pcolormesh() subroutine that abstracts away
	the specifics of the grid used.

	Attributes:
		nlons (int): Number of grid cells in longitude direction
		nlats (int): Number of grid cells in latitude direction
	"""
		
	def __init__(self, lons, lats, boundaries = False) :
		"""Constructs a lat/lon plotter.

		Args:
			lons[]:
				Longitude of center of each cell
				(as is read from Scaled ACC files).
			lats[]:
				Latitude of center of each cell
				(as is read from Scaled ACC files).
			boundaries (boolean):
				If True, then lons[] and lats[] represent boundaries
				of grid cells, not cell centers.
				(as is read from Overlap matrix files).
		"""
		if boundaries :
			self.lonb = lons
			self.latb = np.array([-89.999] + list(lats) + [89.999])
		else :
			# --------- Reprocess lat/lon format for a quadrilateral mesh
			# (Assume latlon grid)
			# Shift lats to represent edges of grid boxes
			latb = np.zeros(len(lats)+1)
			latb[0] = lats[0]		# -90
			latb[1] = lats[1] - (lats[2] - lats[1])*.5
			latb[-1] = lats[-1]		# 90
			latb[-2] = lats[-2] + (lats[-1] - lats[-2])*.5
			for i in range(2,len(lats)-1) :
				latb[i] = (lats[i-1] + lats[i]) * .5

			# Polar projections get upset with pcolormesh()
			# if we go all the way to the pole
			if latb[0] < -89.999 : latb[0] = -89.999
			if latb[-1] > 89.999 : latb[-1] = 89.999
			if latb[0] < -89.9 : latb[0] = -89.9
			if latb[-1] > 89.9 : latb[-1] = 89.9

			# Shift lons to represent edges of grid boxes
			lonb = np.zeros(len(lons)+1)
			lonb[0] = (lons[0] + (lons[-1]-360.)) * .5	# Assume no overlap
			for i in range(1,len(lons)) :
				lonb[i] = (lons[i] + lons[i-1]) * .5
			lonb[-1] = lonb[0]+360		# SST demo repeated the longitude, don't know if it's neede

			self.lonb = lonb
			self.latb = latb

		self.nlons = len(self.lonb)-1
		self.nlats = len(self.latb)-1

	def pcolormesh(self, mymap, val1, **plotargs) :
		# compute map projection coordinates of grid.
		xx, yy = mymap(*np.meshgrid(self.lonb, self.latb))
		val = val1.reshape((self.nlats, self.nlons))
		return mymap.pcolormesh(xx, yy, val, **plotargs)
