import netCDF4
import memoize
from pncview.types import *
import giss.modele
import numpy as np
import glint2

class ModelEVarManager(object):

	def __init__(self, fname):
		self.fname = fname
		self.nc = netCDF4.Dataset(self.fname)
		self.grid_fname = self.nc.variables['grid'].file

		# ---------- Get machine-readable set of times
		time_nc = self.nc.variables['time']
		try:
			units = time_nc.units
			calendar = time_nc.calendar
			self.times = netCDF4.num2date(time_nc[:], units, calendar=calendar)
		except Exception as e:
			# Don't know what to do, don't try to convert.
			self.times = time_nc[:]

		# ---------- Get list of entries
		self.entries = list()
		self.entries_by_name = dict()
		for vname, var in self.nc.variables.items():
			if (len(var.shape) < 2): continue
			if (len(var.shape) == 4):
				entry = VarEntry('modele.ijhc', vname, var.shape)
				self.entries.append(entry)
				self.entries_by_name[vname] = entry

	def get_info(self):
		"""Pass back stuff we created in __init__()"""
		return self.fname, self.entries, self.times

	# -----------------------------------
	def plot_params_ijhc(self, vname, time_ix):
		# Retrieve netcdf/etc. variable
		var = self.nc.variables[vname]
		val_t = var[time_ix, 1:, :]

		title = '{} ({})'.format(vname, var.units)
		title = '{}\n{}: {}'.format(title, time_ix, self.times[time_ix])
		# TODO: Inline this
		pp = giss.modele.plot_params(title, val=val_t, plotter='DUMMY')

		plot_args = pp['plot_args']
		plot_args['vmin'] = np.nanmin(pp['val'])
		plot_args['vmax'] = np.nanmax(pp['val'])

		pp['basemap_spec'] = ('giss.basemap', 'greenland_laea')
		pp['plotter_spec'] = 'plotter1h'
		del pp['val']	# This is big, don't send it initially
		return pp

	def plot_params(self, vname, time_ix):
		entry = self.entries_by_name[vname]
		plot_params_fn = ModelEVarManager.plot_params_fns[entry.type]
		return plot_params_fn(self, vname, time_ix)
	# -------------------------------------------------------
	@memoize.mproperty
	def plotter1h(self):
		return glint2.Plotter1h(self.grid_fname, ice_sheet='greenland')

	@property
	def plotter2(self):
		return self.plotter1h.plotter2

	def plotter(self, spec):
		"""Must always return the same plotter object for the same spec."""
		return getattr(self, spec)
	# -------------------------------------------------------
	def val_t(self, vname, time_ix):
		# Retrieve netcdf/etc. variable
		var = self.nc.variables[vname]
		val_t = var[time_ix, 1:, :]
		return val_t

ModelEVarManager.plot_params_fns = { 'modele.ijhc' : ModelEVarManager.plot_params_ijhc}
