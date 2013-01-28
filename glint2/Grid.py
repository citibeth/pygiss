import giss.plot
import pyproj
import numpy as np
import giss.proj
import netCDF4

class Grid :
	# Read Grid from a netCDF file
	def __init__(self, nc, vname) :
		info = nc.variables[vname + '.info']

		# Read all attributes under .info:
		# name, type, cells.num_full, vertices.num_full
		self.__dict__.update(info.__dict__)
		self.cells_num_full = self.__dict__['cells.num_full']
		self.vertices_num_full = self.__dict__['vertices.num_full']

		self.vertices_index = nc.variables[vname + '.vertices.index'][:]

		self.vertices_xy = nc.variables[vname + '.vertices.xy'][:]
		self.cells_index = nc.variables[vname + '.cells.index'][:]
		self.cells_ijk = nc.variables[vname + '.cells.ijk'][:]
		self.cells_area = nc.variables[vname + '.cells.area'][:]
#		self.cells_proj_area = nc.variables[vname + '.cells.proj_area'][:]
		self.cells_vertex_refs = nc.variables[vname + '.cells.vertex_refs'][:]
		self.cells_vertex_refs_start = nc.variables[vname + '.cells.vertex_refs_start'][:]

		
		if self.coordinates == 'xy' :
			print 'Projection = "' + self.projection + '"'
			print type(self.projection)
			(self.llproj, self.xyproj) = giss.proj.make_projs(str(self.projection))
		else :
			self.xyproj = None

		if self.type == 'xy' :
			self.x_boundaries = nc.variables[vname + '.x_boundaries'][:]
			self.y_boundaries = nc.variables[vname + '.y_boundaries'][:]


	def plot(self, basemap, **kwargs) :
		"""Plots the grid cell outlines
	
		Args:
			basemap: Map on which to plot
			**kwargs: Any options passed through to Matplotlib plotting
		"""

		npoly = len(self.cells_vertex_refs_start)-1		# -1 for sentinel
		npoints = len(self.cells_vertex_refs)
		xdata = np.zeros(npoints + npoly * 2)
		ydata = np.zeros(npoints + npoly * 2)

		ipoint_dst = 0
		for ipoly in range(0,npoly) :
			iistart = self.cells_vertex_refs_start[ipoly]
			iinext = self.cells_vertex_refs_start[ipoly+1]
			npoints_this = iinext - iistart

			xdata[ipoint_dst:ipoint_dst + npoints_this] = \
				self.vertices_xy[self.cells_vertex_refs[iistart:iinext], 0]
			ydata[ipoint_dst:ipoint_dst + npoints_this] = \
				self.vertices_xy[self.cells_vertex_refs[iistart:iinext], 1]

			ipoint_dst += npoints_this

			# Repeat the first point in the polygon
			xdata[ipoint_dst] = \
				self.vertices_xy[self.cells_vertex_refs[iistart], 0]
			ydata[ipoint_dst] = \
				self.vertices_xy[self.cells_vertex_refs[iistart], 1]
			ipoint_dst += 1

			# Add a NaN
			xdata[ipoint_dst] = np.nan
			ydata[ipoint_dst] = np.nan
			ipoint_dst += 1

		

		if self.xyproj is not None :	# translate xy->ll
			londata, latdata = pyproj.transform(self.xyproj, self.llproj, xdata, ydata)
			londata[np.isnan(xdata)] = np.nan
			latdata[np.isnan(ydata)] = np.nan

		else :		# Alreayd in lon/lat coordinates
			londata = xdata
			latdata = ydata

		giss.basemap.plot_lines(basemap, londata, latdata, **kwargs)


	def plotter(self) :
		if self.type == 'xy' :
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
read_plotter_fn = {'xy' : _Grid_XY_read_plotter,
	'lonlat' : _Grid_LonLat_read_plotter}

def Grid_read_plotter(grid_fname, vname) :
	nc = netCDF4.Dataset(grid_fname)
	stype = nc.variables[vname + '.info'].__dict__['type']
	read_fn = read_plotter_fn[stype]
	ret = read_fn(nc, vname)
	nc.close()
	return ret

