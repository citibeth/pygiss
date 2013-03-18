import giss.plot
import pyproj
import numpy as np
import giss.proj
import netCDF4

# Re-name variables for grid data files coming from ISSM
def standardize_names(ivars) :
	ovars = {}
	for vname,var in ivars.items() :
		print vname
		if vname != 'grid.info' and 'name' in var.__dict__ :
			ovars[var.name] = var
		else :
			print '********* ' + vname
			ovars[vname] = var
	return ovars

# Local imports
import sys
from overlap import *
from cext import *
glint2 = sys.modules[__name__]

class pyGrid :
	# Read Grid from a netCDF file
	def __init__(self, nc, vname) :
#		variables = standardize_names(nc.variables)
		variables = nc.variables

		info = variables[vname + '.info']

		# Read all attributes under .info:
		# name, type, cells.num_full, vertices.num_full
		self.__dict__.update(info.__dict__)
		self.cells_num_full = self.__dict__['cells.num_full']
		self.vertices_num_full = self.__dict__['vertices.num_full']

		self.vertices_index = variables[vname + '.vertices.index'][:]
		self.vertices_pos = dict()		# Position (by index) of each vertex in our arrays

		self.vertices_xy = variables[vname + '.vertices.xy'][:]
		self.cells_index = variables[vname + '.cells.index'][:]
		if vname + '.cells.ijk' in variables :
			self.cells_ijk = variables[vname + '.cells.ijk'][:]
		self.cells_area = variables[vname + '.cells.area'][:]
#		self.cells_proj_area = variables[vname + '.cells.proj_area'][:]
		self.cells_vertex_refs = variables[vname + '.cells.vertex_refs'][:]
		self.cells_vertex_refs_start = variables[vname + '.cells.vertex_refs_start'][:]

		# Compute vertices_pos
		for i in range(0, len(self.vertices_index)) :
			self.vertices_pos[self.vertices_index[i]] = i

		# Correct ISSM file
#		self.cells_vertex_refs = self.cells_vertex_refs - 1

		
		if self.coordinates == 'XY' :
			print 'Projection = "' + self.projection + '"'
			print type(self.projection)
			(self.llproj, self.xyproj) = giss.proj.make_projs(self.projection)
		else :
			self.xyproj = None

		if self.type == 'XY' :
			self.x_boundaries = variables[vname + '.x_boundaries'][:]
			self.y_boundaries = variables[vname + '.y_boundaries'][:]


	def plot(self, basemap, **kwargs) :
		"""Plots the grid cell outlines
	
		Args:
			basemap: Map on which to plot
			**kwargs: Any options passed through to Matplotlib plotting
		"""

		npoly = len(self.cells_vertex_refs_start)-1		# -1 for sentinel
		npoints = len(self.cells_vertex_refs)
#		xdata = np.zeros(npoints + npoly * 2)
#		ydata = np.zeros(npoints + npoly * 2)
		xdata = []
		ydata = []

		ipoint_dst = 0
		for ipoly in range(0,npoly) :
			iistart = self.cells_vertex_refs_start[ipoly]
			iinext = self.cells_vertex_refs_start[ipoly+1]
			npoints_this = iinext - iistart

#			print ipoint_dst, npoints_this, len(xdata)
#			print iistart, iinext, len(self.cells_vertex_refs)
#			print len(self.vertices_xy), self.cells_vertex_refs[iistart:iinext]

#			xdata[ipoint_dst:ipoint_dst + npoints_this] = \
#				self.vertices_xy[self.cells_vertex_refs[iistart:iinext], 0]
#			ydata[ipoint_dst:ipoint_dst + npoints_this] = \
#				self.vertices_xy[self.cells_vertex_refs[iistart:iinext], 1]

			refs = self.cells_vertex_refs[iistart:iinext]
			for i in range(0,len(refs)) :
				refs[i] = self.vertices_pos[refs[i]]
#			print refs
			xdata += list(self.vertices_xy[refs, 0])
			ydata += list(self.vertices_xy[refs, 1])

			ipoint_dst += npoints_this

			# Repeat the first point in the polygon
#			xdata[ipoint_dst] = \
#				self.vertices_xy[self.cells_vertex_refs[iistart], 0]
#			ydata[ipoint_dst] = \
#				self.vertices_xy[self.cells_vertex_refs[iistart], 1]

			xdata.append(self.vertices_xy[self.cells_vertex_refs[iistart], 0])
			ydata.append(self.vertices_xy[self.cells_vertex_refs[iistart], 1])


			ipoint_dst += 1

			# Add a NaN
#			xdata[ipoint_dst] = np.nan
#			ydata[ipoint_dst] = np.nan
			xdata.append(np.nan)
			ydata.append(np.nan)
			ipoint_dst += 1

		xdata = np.array(xdata)
		ydata = np.array(ydata)

		
		if self.xyproj is not None :	# translate xy->ll
			londata, latdata = pyproj.transform(self.xyproj, self.llproj, xdata, ydata)
			londata[np.isnan(xdata)] = np.nan
			latdata[np.isnan(ydata)] = np.nan

		else :		# Already in lon/lat coordinates
			londata = xdata
			latdata = ydata

		giss.basemap.plot_lines(basemap, londata, latdata, **kwargs)


	def plotter(self) :
		if self.type == 'XY' :
			return ProjXYPlotter(self.x_boundaries, self.y_boundaries, self.sproj)
		return None


def _Grid_XY_read_plotter(nc, vname) :
	"""Reads an plotter out of a netCDF file for a simple Cartesian grid"""

	# ======= Read our own grid2 info from the overlap file
	# Assumes an XY grid for grid2
	xb2 = nc.variables[vname + '.x_boundaries'][:]
	yb2 = nc.variables[vname + '.y_boundaries'][:]
	info_var = nc.variables[vname + '.info']
	sproj = info_var.projection
	return giss.plot.ProjXYPlotter(xb2, yb2, sproj)

def _Grid_LonLat_read_plotter(nc, vname) :
	lonb2 = nc.variables[vname + '.lon_boundaries'][:]
	latb2 = nc.variables[vname + '.lat_boundaries'][:]
	return giss.plot.LonLatPlotter(lonb2, latb2, True)

# -------------------------------
read_plotter_fn = {'XY' : _Grid_XY_read_plotter,
	'LONLAT' : _Grid_LonLat_read_plotter}

def Grid_read_plotter(grid_fname, vname) :
	nc = netCDF4.Dataset(grid_fname)
	stype = nc.variables[vname + '.info'].__dict__['type']
	read_fn = read_plotter_fn[stype]
	ret = read_fn(nc, vname)
	nc.close()
	return ret

# ---------------------------------------------------
# Plots Height-Classified Data
#
# (elevation2, mask2) can come from giss.searise.read_elevation2_mask2
# @param overlap The overlap matrix between grid1 (GCM) and grid2 (ice).
#        OR: Name of exchange grid file.
# @param height_max1h (nhc x n1) array of height class definitions
class Plotter_HC :
	def __init__(self, grid2_plotter, overlap, elev2, hcmax) :
		self.grid2_plotter = grid2_plotter

		# Read overlap matrix if we were just given name of
		# Exchange Grid file.
		if issubclass(type(overlap), str) :
			nc = netCDF4.Dataset(overlap)
			overlap = glint2.read_overlap(nc, 'grid')
			nc.close()

		# Height-classify the overlap matrix
		# (Do not correct for native/proj area, since this is for plotting)
		self.overlaph = glint2.height_classify(overlap, elev2, hcmax)

	def pcolormesh(self, mymap, val1h, **plotargs) :
		val1h = val1h.reshape(-1)

		# Mask overlap matrix based on the data we're given.
		mask1h = (np.isnan(val1h)).astype(np.int32)
		overlaph_m = glint2.mask_out(self.overlaph, mask1h, mask2=None)
		mat_1h_to_2 = glint2.grid1_to_grid2(overlaph_m)

		val2 = glint2.coo_multiply(mat_1h_to_2, val1h, fill=np.nan)

		return self.grid2_plotter.pcolormesh(mymap, val2, **plotargs)
