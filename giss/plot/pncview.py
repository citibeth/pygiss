import pickle
from gi.repository import Gtk
import netCDF4
import giss.basemap
import giss.modele
import sys
import datetime
import numpy as np
import giss.basemap
import giss.modele
import matplotlib.pyplot
import mpl_toolkits.basemap
import importlib

def plot_pp(pp):
	print('pp is', type(pp))
	figure = matplotlib.pyplot.figure(figsize=(11,8.5))

	ax = figure.add_subplot(111)
	basemap_mod = importlib.import_module(pp['basemap_fn'][0])
	basemap_fn = getattr(basemap_mod, pp['basemap_fn'][1])

	giss.plot.plot_var(ax=ax, basemap=basemap_fn(ax), **pp)

	# Also show on screen
	matplotlib.pyplot.show()
