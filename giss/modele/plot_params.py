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

import giss.plot
import giss.util
import numpy as np
import numpy.ma as ma
from giss.modele import plotters

_zero_centered = {'impm', 'evap_lndice', 'evap', 'imph_lndice', 'impm_lndice', 'netht_lndice', 'trht_lndice'}#, 'sensht_lndice'} #, 'trht_lndice'}

_reverse_scale = {'impm', 'impm_lndice'}

# TODO: Make this _change_units instead, use a lambda function
# This way, we can do C <--> K conversion as well
#_rescale_factors = {'impm_lndice' : (86400., 'kg/day*m^2')}

#_change_units = {
#	'impm_lndice' : ('kg/day*m^2', lambda x : x * 86400.)
#}


def _default_plot_boundaries(basemap) :
	"""Default function for 'plot_boundaries' output of plot_params()"""

	# ---------- Draw other stuff on the map
	# draw line around map projection limb.
	# color background of map projection region.
	# missing values over land will show up this color.
	basemap.drawmapboundary(fill_color='0.5')
	basemap.drawcoastlines()

	# draw parallels and meridians, but don't bother labelling them.
	basemap.drawparallels(np.arange(-90.,120.,30.))
	basemap.drawmeridians(np.arange(0.,420.,60.))


def plot_params(var_name='', nc=None, val=None, plotter=None) :
	"""Suggests a number of plot parameters for a ModelE ScaledACC output variable.
	Output to be used directly as kwargs for giss.plot.plot_var()

	Args:
		var_name (string):
			Name of the variable to plot (from the Scaled ACC file)
		nc (netCDF4.Dataset, OPTIONAL):
			Open netCDF Scaled ACC file.
			If set, then data and meta-data will be read from this file.
		val (np.array, OPTIONAL):
			Field to plot.  If not set, then will be read from the netCDF file.
		plotter (OPTIONAL):
			Plotter to use when plotting data of this shape.

	Returns: Dictionary with the following elements
		plotter (giss.plot.*Plotter):
			Abstracts away grid geometry from pcolormesh() call.
			Guess at the right plotter (since this IS ModelE data).
		var_name (string):
			Name of the variable to plot (same as var_name arg)
		val (np.ma.MaskedArray):
			The value to plot
		units (string, OPTIONAL):
			Units in which val is expressed
		title (string):
			Suggested title for the plot
		plot_args (dict):
			Suggested keyword arguments for pcolormesh() command.
			Override if you like: norm, cmap, vmin, vmax
		cb_args (dict):
			Suggested keyword arguments for colorbar command.
			Override if you like: ticks, format
		plot_boundaries (function(basemap)):
			Plot map boundaries, coastlines, paralells, meridians, etc.
	"""

	info = {'var_name' : var_name}

	# Read meta-data out of the netCDF file, if we can
	if nc is not None and var_name in nc.variables :
		info.update(nc.variables[var_name].__dict__)

	# Init kwargs for Plotter.pcolormesh() command
	plot_args = {}
	info['plot_args'] = plot_args

	# Init kwargs for colorbar command
	cb_args = {}
	info['cb_args'] = cb_args

	info['var_name'] = var_name

	# Get the data
	if val is None and nc is not None:
		info['val'] = giss.modele.read_ncvar(nc, var_name)
	else :
		info['val'] = ma.copy(val)

	# Guess a plotter
	if plotter is None :
		info['plotter'] = plotters.guess_plotter(info['val'])
	else :
		info['plotter'] = plotter

#	# Rescale if needed
#	if var_name in _change_units :
#		rs = _change_units[var_name]
#		info['units'] = rs[0]
#		info['val'] = rs[1](info['val'])	# Run the scaling function

	if var_name in _zero_centered :
		plot_args['norm'] = giss.plot.AsymmetricNormalize()
		reverse = (var_name in _reverse_scale)
		plot_args['cmap'] = giss.plot.cpt('giss-cpt/BlRe.cpt', reverse=reverse).cmap
		plot_args['vmin'] = np.nanmin(info['val'])
		plot_args['vmax'] = np.nanmax(info['val'])
		cb_args['ticks'] = [plot_args['vmin'], 0, plot_args['vmax']]
		cb_args['format'] = '%0.2f'

	# These could be decent defaults for anything with a colorbar
	cb_args['location'] = 'bottom'
	cb_args['size'] = '5%'
	cb_args['pad'] = '2%'

	# Suggest a title
	if 'units' in info :
		info['title'] = '%s\n(%s)' % (info['var_name'], info['units'])
	else :
		info['title'] = info['var_name']

	# Default coastlines
	info['plot_boundaries'] = _default_plot_boundaries

	return info
