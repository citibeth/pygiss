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

import bisect
import functools
import giss.proj
import numpy as np
import pyproj
import operator
import numpy.ma as ma

class Plotter(object):

	def context(self, vals=None):
		"""Generate data specific to a particular set of values we're
		trying to plot, which will be used by further plot()
		functions.  This way, we avoid re-regridding the same data
		over and over again, for example if we want a contour plot.
		The contents of the context are plotter-dependent."""
		return None


	def plot(self, context, basemap_plot_fn, **plotargs):
		"""basemap_plot_fn:
			A bound method on a basemap instance.
			For example: mymap.pcolormesh, mymap.contour, mymap.contourf"""
		raise NotImplementedError('Subclass should implement this.')

	# Returns the polygon describing a grid cell (spherical coordinates)
	def cell_poly(self, i,j) :
		raise NotImplementedError('Subclass should implement this.')


#	def pcolormesh(self, mymap, val1, **plotargs) :
#		return self.plot(mymap.pcolormesh, val1, **plotargs)
#
#	def contour(self, mymap, val1, **plotargs) :
#		return self.plot(mymap.contour, val1, **plotargs)
#
#	def contourf(self, mymap, val1, **plotargs) :
#		return self.plot(mymap.contourf, val1, **plotargs)




# --------------------------------------------------------------
class LonLatPlotter(Plotter):
	"""A plotter for lat/lon GCM grid cell data.

	Plotters provide a pcolormesh() subroutine that abstracts away
	the specifics of the grid used.

	Attributes:
		nlons (int): Number of grid cells in longitude direction
		nlats (int): Number of grid cells in latitude direction
	"""
		
	def __init__(self, lons, lats, boundaries=False, transpose=False) :
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
			transpose (bool):
				Data being plotted is indexed via:
					(lat, lon) if False  (ModelE's netCDF)
					(lon, lat) if True
		"""
		self.transpose = transpose

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

	def context(self, basemap, vals):
		context = giss.util.LazyDict()

		context['basemap'] = basemap
		context.lazy['mesh_xy'] = lambda: basemap(*np.meshgrid(
			self.lonb, self.latb, indexing='ij' if self.transpose else 'xy'))

		# ------ context['vals']
		vals_ll = vals.reshape(
			(self.nlons, self.nlats) if self.transpose else (self.nlats, self.nlons))

		if not issubclass(type(vals_ll), ma.MaskedArray) :
			# Not a masked array, mask out invalid values (eg NaN)
			vals_ll = ma.masked_invalid(vals_ll)

		context['vals_ll'] = vals_ll

		return context


	def plot(self, context, basemap_plot_fn, **plotargs) :
		# compute map projection coordinates of grid.
		xx, yy = context['mesh_xy']

		return basemap_plot_fn(xx, yy, context['vals_ll'], **plotargs)

	# Returns the polygon describing a grid cell (spherical coordinates)
	def cell_poly(self, i,j) :
		lon0 = self.lonb[i]
		lon1 = self.lonb[i+1]
		lat0 = self.latb[j]
		lat1 = self.latb[j+1]

		lons = [lon0, lon1, lon1, lon0, lon0]
		lats = [lat0, lat0, lat1, lat1, lat0]
		return lons, lats

	def coords(self, lon_d, lat_d):
		i = bisect.bisect_left(self.lonb, lon_d)  # "rounds down"
		j = bisect.bisect_left(self.latb, lat_d)
		return (i,j) if self.transpose else (j,i)


	def lookup(self, context, lon_d, lat_d):
		coords = self.coords(lon_d, lat_d)
		try:
			val = context['vals_ll'][coords]
		except:
			# Out of bounds
			val = None
		return coords, val

class ProjXYPlotter(Plotter):
	"""A plotter for Cartesian-gridded data projected onto the sphere.
	(Typically used to plot data on the ice grid)

	Plotters provide a pcolormesh() subroutine that abstracts away
	the specifics of the grid used.

	"""
	
	def __init__(self, xb2, yb2, sproj, transpose=False):
		"""Construct the plotter.

		Args:
			xb2[nx]:
				Boundaries of grid cells in the X direction
			yb2[nx]:
				Boundaries of grid cells in the Y direction
			sproj (string):
				Proj.4 string for the projection used to map between Cartesian and the globe.
			transpose (bool):
				False if the data being plotted will be in (y, x) order (as in ModelE's netCDF),
				rather than (x,y) order (as in PISM's netCDF).

		Attributes:
			nx2, ny2:
				Number of grid cells in X and Y directions
			n2:
				Total number of grid cells
			mesh_lons[n2], mesh_lats[n2]:
				Lat/lon of every grid cell boundary intersection.
		"""


		self.xb2 = xb2
		self.yb2 = yb2
		self.sproj = sproj
		self.transpose = transpose

		self.nx2 = len(xb2)-1
		self.ny2 = len(yb2)-1

		# Check dims
		self.n2 = self.nx2 * self.ny2

		# Get the ice grid projection as well, so we can convert our local
		# mesh to lat/lon
		(self.llproj, self.xyproj) = giss.proj.make_projs(sproj)

		# ------------- Mesh on cell boundaries (for pcolormesh())
		# Create a quadrilateral mesh in X/Y space
		# Quad mesh needs cell BOUNDARIES
		xs, ys = np.meshgrid(xb2, yb2, indexing='ij' if self.transpose else 'xy')

		# Transform it to lat/lon space
		self.bmesh_lons, self.bmesh_lats = pyproj.transform(self.xyproj, self.llproj, xs, ys)

		# ------------- Mesh on cell centers (for contour)
		# Grid cell centers
		xc2 = .5 * (xb2[0:-1] + xb2[1:])
		yc2 = .5 * (yb2[0:-1] + yb2[1:])
		xs, ys = np.meshgrid(xc2, yc2)
		self.cmesh_lons, self.cmesh_lats = pyproj.transform(self.xyproj, self.llproj, xs, ys)


	def __getstate__(self):
		return ((self.xb2, self.yb2, self.sproj), dict(transpose=self.transpose))
	def __setstate__(self, state):
		self.__init__(*state[0], **state[1])


	def context(self, basemap, vals):

		# Check dimensions
		n2 = functools.reduce(operator.mul, vals.shape)
		if self.n2 != n2 :
			raise Exception("Plotter's n2 (%d) != data's n2 (%d)" % (self.n2, n2))


		context = giss.util.LazyDict()

		context['basemap'] = basemap
		context.lazy['mesh_xy'] = \
			lambda: basemap(self.bmesh_lons, self.bmesh_lats)

		# --------- Reshape and mask values
		vals_xy = vals.reshape(
			(self.nx2, self.ny2) if self.transpose else (self.ny2, self.nx2))

		if not issubclass(type(vals_xy), ma.MaskedArray) :
			# Not a masked array, mask out invalid values (eg NaN)
			vals_xy = ma.masked_invalid(vals_xy)

		context['vals_xy'] = vals_xy
		return context

	def plot(self, context, basemap_plot_fn, **plotargs) :
		print('XY.plot: context=', context)
		mesh_mapx, mesh_mapy = context['mesh_xy']

		# Plot our result using the quadrilateral mesh
		return basemap_plot_fn(mesh_mapx, mesh_mapy, context['vals_xy'], **plotargs)


	# Returns the polygon describing a grid cell (spherical coordinates)
	def cell_poly(self, i,j) :
		x0 = self.xb2[i]
		x1 = self.xb2[i+1]
		y0 = self.yb2[j]
		y1 = self.yb2[j+1]

		xs = [x0, x1, x1, x0, x0]
		ys = [y0, y0, y1, y1, y0]

		lons, lats = pyproj.transform(self.xyproj, self.llproj, xs, ys)
		return lons, lats

	def coords(self, lon_d, lat_d):
		# Convert lon/lat to x/y for the mesh's local projection
		# (which is not the same as the map projection)
		x,y = pyproj.transform(self.llproj, self.xyproj, lon_d, lat_d)		

		i = bisect.bisect_right(self.xb2, x) - 1
		j = bisect.bisect_right(self.yb2, y) - 1

		coords = (i,j) if self.transpose else (j,i)
		return coords

	def lookup(self, context, lon_d, lat_d):
		coords = self.coords(lon_d, lat_d)

		vals_xy = context['vals_xy']
		try:
			if vals_xy.mask[coords]:
				val = None
			else:
				val = vals_xy.data[coords]
		except IndexError:
			# Out of bounds
			val = None

		return coords,val

