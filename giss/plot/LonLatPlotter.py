# pyGISS: GISS Python Library
# Copyright (c) 2013 by Robert Fischer
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import numpy.ma as ma

class LonLatPlotter :
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
			# TODO: Handle pole stuff properly here
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

	def _plot_data2d(self, mymap, val1, plot_fn, **plotargs) :
		# compute map projection coordinates of grid.
		xx, yy = mymap(*np.meshgrid(self.lonb, self.latb))
		val = val1.reshape((self.nlats, self.nlons))

		if not issubclass(type(val), ma.MaskedArray) :
			# Not a masked array, mask out invalid values (eg NaN)
			val = ma.masked_invalid(val)

		return plot_fn(xx, yy, val, **plotargs)

	def pcolormesh(self, mymap, val1, **plotargs) :
		return self._plot_data2d(mymap, val1, mymap.pcolormesh, **plotargs)

	def contour(self, mymap, val1, **plotargs) :
		return self._plot_data2d(mymap, val1, mymap.contour, **plotargs)

	def contourf(self, mymap, val1, **plotargs) :
		return self._plot_data2d(mymap, val1, mymap.contourf, **plotargs)


	# Returns the polygon describing a grid cell (spherical coordinates)
	def cell_poly(self, i,j) :
		lon0 = self.lonb[i]
		lon1 = self.lonb[i+1]
		lat0 = self.latb[j]
		lat1 = self.latb[j+1]

		lons = [lon0, lon1, lon1, lon0, lon0]
		lats = [lat0, lat0, lat1, lat1, lat0]
		return lons, lats

