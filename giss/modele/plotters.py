import numpy as np
import giss.util
import numpy.ma as ma

class Grid1Plotter_LL :
	# @param boundaries Do lons/lats represent grid cell boundaries?  (Or centers)?
	def __init__(self, lons, lats, boundaries = False) :
		if boundaries :
			self.lonb = lons
			self.latb = lats
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


# ====================================================================


# Definitions of specific grids used in ModelE
#
_lats_4x5 = np.array([-90., -86., -82., -78., -74., -70., -66., -62., -58., -54., -50.,
       -46., -42., -38., -34., -30., -26., -22., -18., -14., -10.,  -6.,
        -2.,   2.,   6.,  10.,  14.,  18.,  22.,  26.,  30.,  34.,  38.,
        42.,  46.,  50.,  54.,  58.,  62.,  66.,  70.,  74.,  78.,  82.,
        86.,  90.])

_lons_4x5 = np.array([-177.5, -172.5, -167.5, -162.5, -157.5, -152.5, -147.5, -142.5,
       -137.5, -132.5, -127.5, -122.5, -117.5, -112.5, -107.5, -102.5,
        -97.5,  -92.5,  -87.5,  -82.5,  -77.5,  -72.5,  -67.5,  -62.5,
        -57.5,  -52.5,  -47.5,  -42.5,  -37.5,  -32.5,  -27.5,  -22.5,
        -17.5,  -12.5,   -7.5,   -2.5,    2.5,    7.5,   12.5,   17.5,
         22.5,   27.5,   32.5,   37.5,   42.5,   47.5,   52.5,   57.5,
         62.5,   67.5,   72.5,   77.5,   82.5,   87.5,   92.5,   97.5,
        102.5,  107.5,  112.5,  117.5,  122.5,  127.5,  132.5,  137.5,
        142.5,  147.5,  152.5,  157.5,  162.5,  167.5,  172.5,  177.5])

_plotter_lookup = {
	(len(_lats_4x5), len(_lons_4x5)) : Grid1Plotter_LL(_lons_4x5, _lats_4x5)
}

# Guesses on a plotter to use, based on the dimension of a field from
# ModelE.
def guess_plotter(field) :
	# Infer the grid from the size of the input array
	return _plotter_lookup[field.shape]
