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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.	If not, see <http://www.gnu.org/licenses/>.

import giss.basemap
import matplotlib
import matplotlib.pyplot
from giss.plot import config
import giss.util
import numpy as np
import copy


def formatter(plotter, basemap, plotter_context, lon_format='{:1.0f}', lat_format='{:1.0f}', val_format='{:g}'):

	format_string = '[%s{} %s{}] {} {}' % (lon_format, lat_format)

	def _format_coord(x, y):
		lon_d, lat_d = basemap(x,y,inverse=True)
		lon_ew = 'E' if lon_d >=0 else 'W'
		lat_ns = 'N' if lat_d >=0 else 'S'
		try:
			coords, val = plotter.lookup(plotter_context, lon_d, lat_d)
		except:
			coords = ()
			val = None

		if val is None:
			sval = ''
		else:
			sval = val_format.format(val)

		return format_string.format(lon_d, lon_ew, lat_d, lat_ns, coords, sval)
	return _format_coord
# ---------------------------------------------------------------

def plot_var(ax=None, basemap=None,
show=None, fname=None, savefig_args={},
plotter=None, var_name=None, val=None, title=None, plot_args={}, cb_args=None, plot_boundaries=None,
**extra_kwargs) :
	"""Convenient all-in-one plot routine.
	Meant to be used in conjunction with a plot_params() function,
		eg: giss.modele.plot_params()

	There is a separation between application-specific plotting logic
	(plot_params) and general plotting/graphic management (this
	method).  Many/most of the keyword args are meant to be passed
	along from plot_params().  The output of plot_params() may be
	modified along the way in order to fully customize anything.

	Args (supplied by user):
		ax (matplotlib.axes.Axes) OPTIONAL:
			The axes instance on which to plot.
			If not supplied, then basemap.ax is used.
			If that doesn't exist, then it creates a new figure (US Letter Landscape)
		basemap (matplotlib.basemap) OPTIONAL:
			The map projection to be used to plot.
			If not supplied, then default basemap is used.
			See: giss.basemap.global_map()
		show (boolean) OPTIONAL:
			If True, display the plot to the user.
			If not supplied, show the plot if both of:
				(a) ax not supplied (this is a "quick plot")
				(b) fname not supplied (we don't want to save it to a file)
		fname (string) OPTIONAL:
			If supplied, save the figure to this file via figure.savefig()
			Only works if ax is NOT supplied
			(i.e. plot_var() generated its own figure).
		savefig_args (dict) OPTIONAL:
			Keyword arguments to pass along to the figure.savefig() call.
			Eg: dpi, transparent, format
		**extra_kwargs OPTIONAL:
			Unrecognized keyword args will be displayed in the console
			but otherwise silently ignored.

	Args (supplied by plot_params()):
		plotter (giss.plot.*Plotter):
			Abstracts away grid geometry from pcolormesh() call.
		val (np.ma.MaskedArray):
			The value to plot
		title (string) OPTIONAL:
			Title for the plot.	 If not supplied, no title will be drawn.
		plot_args (dict) OPTIONAL:
			Suggested keyword arguments for pcolormesh() command.
			Eg: norm, cmap, vmin, vmax, etc.
		cb_args (dict) OPTIONAL:
			Keyword arguments for colorbar command: Eg: ticks, format, etc.
		plot_boundaries (function(basemap)) OPTIONAL:
			Call this function to plot map boundaries, graticules,
			coastlines, paralells, meridians, etc.	It can be a
			pas-sthrough to the standard basemap boundary plotting, or
			your own function.	See: giss.basemap.drawcoastline_file()

	Returns for further customization (dict):
		figure:
			The figure created (IF we created one)
		axes:
			The Axes object used.
		image:
			The result of pcolormesh()
		colorbar (basemap.colorbar) OPTIONAL:
			The colorbar object in the plot.

	See:
		giss.modele.plot_params()
		matplotlib.figure.Figure.savefig()
		giss.basemap.drawcoastline_file()
		giss.basemap.global_map()
		giss.plot.config
	"""
	ret = {}

	if extra_kwargs is not None and len(extra_kwargs) > 0 :
		print('WARNING: Unrecognized arguments to giss.plot.plot_var() (IGNORED):')
		print(extra_kwargs)

	# No basemap?  Use our default
	if basemap is None :
		basemap = giss.basemap.global_map()

	old_basemap_ax = basemap.ax

	try :
		# Make sure we know what ax we're using
		figure = None
		if ax is not None :
			# Use an ax if we've specified it!
			basemap.ax = ax
		else :
			if basemap.ax is not None :
				# ax specified implicitly in basemap
				ax = basemap.ax
			else :
				# No ax specified: create one.
				figure = matplotlib.pyplot.figure(figsize=config.default_figsize)
				ret['figure'] = figure
				ax = figure.add_subplot(111)
				basemap.ax = ax

		ret['axes'] = ax

		plotter_context = plotter.context(basemap, val)
		image = plotter.plot(plotter_context, basemap.pcolormesh, **plot_args)
		ret['image'] = image
		ax.format_coord = formatter(plotter, basemap, plotter_context)

		# Plot a colorbar if the user has called for it
		if cb_args is not None:
			print('cb_args', cb_args)
			colorbar = basemap.colorbar(image, **cb_args)
			ret['colorbar'] = colorbar

		# Add a title if the user has called for it
		if title is not None:
			ax.set_title(title)

		if plot_boundaries is not None:
			plot_boundaries(basemap)

		# Decide on whether to show and save the figure
		if figure is not None :
			# Logic for show/save if we created our own figure

			# Save the plot
			if fname is not None :
				figure.savefig(fname, **savefig_args)

			# Show the plot
			if show is None :
				show = (fname is None)
			if show :
				# Somehow, figure.show() doesn't work
				matplotlib.pyplot.show()
		else :
			# We didn't create our own figure, only show if requested.
			if (show is not None and show) :
				matplotlib.pyplot.show()

			if fname is not None :
				print("WARNING: giss.plot.plot_var() can only save figure to a file if it created the figure.  Please save the figure yourself, using figure.savefig().	 Or... call plot_var() without the optional argument ax")

	finally :
		basemap.ax = old_basemap_ax
	return ret

