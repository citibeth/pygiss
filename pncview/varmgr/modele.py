import netCDF4
import memoize
from pncview.types import *
import giss.modele
import numpy as np
import glint2
import giss.modele.plotters


VT_TIME = 1		# This var has a time dimension
VT_HP = 2		# This var is on the elevation grid
def var_type(var):
	"""Determine properties of this variable."""
	type = 0
	dims = var.dimensions
	if dims[0] == 'time':
		type = type | VT_TIME
		if dims[1] == 'nhp':
			type = type | VT_HP
	else:
		if dims[0] == 'nhp':
			type = type | VT_HP
	return type


class ModelEVarManager(object):

	def __init__(self, fname, config):
		"""config: from pncview.rc files"""
		self.fname = fname
		self.config = config
		self.nc = netCDF4.Dataset(self.fname)

		# The gridfile spec is optional.  It won't be there on plain
		# aij ModelE files.
		if 'grid' in self.nc.variables:
			self.glint2_fname = self.nc.variables['grid'].file
		else:
			try:
				self.glint2_fname = config['glint2_fname']
			except:
				pass

		# ---------- Get machine-readable set of times
		try:
			time_nc = self.nc.variables['time']
			try:
				units = time_nc.units
				calendar = time_nc.calendar
				self.times = netCDF4.num2date(time_nc[:], units, calendar=calendar)
			except Exception as e:
				# Don't know what to do, don't try to convert.
				self.times = time_nc[:]
		except:
			self.times = []

		# ---------- Get list of entries
		self.entries = list()
		self.entries_by_name = dict()
		for vname, var in self.nc.variables.items():
			if len(var.shape) < 2:
				continue
			vtype = var_type(var)
			entry = VarEntry(vtype, vname, var.shape)

			self.entries.append(entry)
			self.entries_by_name[vname] = entry

	def get_info(self):
		"""Pass back stuff we created in __init__()"""
		return self.fname, self.entries, self.times

	# -----------------------------------

	def plot_params_aij(self, vname, time_ix):
		pass

	def plot_params(self, vname, time_ix):
		# Retrieve netcdf/etc. variable
		var = self.nc.variables[vname]
		vtype = var_type(var)

		# Check for valid time_ix, and set title
		title = '{} ({})'.format(vname, var.units)
		if vtype & VT_TIME:
			if time_ix is None:
				return None		# User must select a time!
			title = '{}\n{}: {}'.format(title, time_ix, self.times[time_ix])

		val_t = self.val_t(vname, time_ix)

		# TODO: Inline this
		pp = giss.modele.plot_params(title, val=val_t, plotter='DUMMY')

		plot_args = pp['plot_args']
		plot_args['vmin'] = np.nanmin(pp['val'])
		plot_args['vmax'] = np.nanmax(pp['val'])

		if vtype & VT_HP:
			pp['basemap_spec'] = ('giss.basemap', 'greenland_laea')
			pp['figsize'] = (4.25,5.5)

		else:
			pp['basemap_spec'] = ('giss.basemap', 'global_map')
			pp['figsize'] = (11,8.5)

		if vtype & VT_HP:
			pp['plotter_spec'] = 'plotter1h'
		else:
			pp['plotter_spec'] = 'plotter1'

		del pp['val']	# This is big, don't send it initially
		del pp['plotter']
		return pp
	# -------------------------------------------------------
	@memoize.mproperty
	def plotter1h(self):
		return glint2.Plotter1h(self.glint2_fname, ice_sheet='greenland')

	@property
	def plotter2(self):
		return self.plotter1h.plotter2

	# This will do for now, just doing 2x2.5...
	@property
	def plotter1(self):
		return giss.modele.plotters.get_byname('2x2.5')

	def plotter(self, spec):
		"""Must always return the same plotter object for the same spec."""
		return getattr(self, spec)
	# -------------------------------------------------------

	def val_t(self, vname, time_ix):
		# Retrieve netcdf/etc. variable
		var = self.nc.variables[vname]
		vtype = var_type(var)

		if (vtype == 0):
			val_t = var[:]
		elif (vtype == VT_TIME):
			val_t = var[time_ix, :]
		elif (vtype == VT_HP):
			val_t = var[1:, :]
		else:
			val_t = var[time_ix, 1:, :]

		# Turn the NetCDF fill value into NaN
		if hasattr(var, '_FillValue'):
			fill_value = var._FillValue
			val_t[val_t == fill_value] = np.nan
		else:
			val_t[val_t == 1e30] = np.nan
			val_t[val_t == -1e30] = np.nan

		# Convert to doubles, for the sake of Glint2
		if val_t.dtype != np.float64:
			val_t1 = np.zeros(val_t.shape)
			val_t1[:] = val_t[:]
			val_t = val_t1

		return val_t
