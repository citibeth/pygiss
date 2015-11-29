import matplotlib
import pickle
import gi
gi.require_version('Gtk', '3.0')
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
matplotlib.pyplot.ion()		# Turn interactive mode on
import mpl_toolkits.basemap
import glint2
import gzip
from giss import util as giutil
import importlib
import types
import memoize
from giss import thunkserver
import collections
import copy


# http://python-gtk-3-tutorial.readthedocs.org/en/latest/treeview.html#the-view

# ----------------------------------------------------
		
# ----------------------------------------------------
class EventThrower(object):
	"""Simple class used to manage and call events."""
	def __init__(self):
		self.events = dict()
	def connect(self, eventid, fn):
		if eventid in self.events:
			self.events[eventid].append(fn)
		else:
			self.events[eventid] = [fn]

	def run_event(self, eventid, *args):
		if eventid in self.events:
			for fn in self.events[eventid]: fn(eventid, *args)

	def unconnect(self, eventid, fn):
		self.events.remove(fn)

# ----------------------------------------------------
class VarEntry(giutil.SlotStruct):
	__slots__ = ('type', 'vname', 'shape')


class ModelEVarManager(object):

	def __init__(self, fname):
		self.fname = fname
		self.nc = netCDF4.Dataset(self.fname)

#		# Load up Glint2 plotters
#		if hasattr(self.nc.variables['grid'], 'file'):
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


# ====================================================
# Server-side stuff

def ss_init(con, argv):
	fname = '/Users/rpfische/exp/151014-integrate/build/e4f40-twoway/run05-firstyear/modele_out.nc'
	# fname = argv[1]

	con['vmgr'] = ModelEVarManager(fname)
	con['fname'] = fname

# ====================================================

# ----------------------------------------------------
class ClientVarMgr(object):
	def __init__(self, ts, context_vname):
		"""Initializes by connecting to a VarMgr on the ThunkServer.
		ts:
			A thunkserver.Client instance
		context_vname:
			The name of the VarMgr the Thunkserver's context.

		Also creates the following properties:
		fname:
			An overall name for the VarMgr
		entries:
			List of 
		"""

		self.thunkserver = ts
		self.context_vname = context_vname
		self.fname, self.entries, self.times = self.thunkserver.exec(thunkserver.ObjThunk(self.context_vname, 'get_info'))

		# Used to memoize...
		self._plotters = dict()
		self._plot_params = dict()

	def plot_params(self, *args):
		"""vname:
			Name of variable in VarMgr to plot
		time_ix:
			Timestep to plot

		Returns an almost-complete plot_params structure.  The
		following two items must be fixed before plotting:
			* basemap_spec --> basemap
			* plotter_spec --> plotter
		"""
		if args not in self._plot_params:
			self._plot_params[args] = self.thunkserver.exec(
				thunkserver.ObjThunk(self.context_vname, 'plot_params',  *args))
		return self._plot_params[args]

	def plotter(self, spec):
		if spec not in self._plotters:
			self._plotters[spec] = self.thunkserver.exec(thunkserver.ObjThunk(self.context_vname, 'plotter',  spec))
		return self._plotters[spec]

	def val_t(self, *args):
		"""Returns the value to be plotted.
		vname:
			Name of variable in VarMgr to plot
		time_ix:
			Timestep to plot"""
		return self.thunkserver.exec(thunkserver.ObjThunk(self.context_vname, 'val_t',  *args))

# ----------------------------------------------------

class PCViewModel(object):
	"""The model class used by the GUI"""
	def __init__(self, vmgr):
		self.events = EventThrower()

		# Create list of variables to plot
		self.vmgr = vmgr
		self.title = self.vmgr.fname
		self.time_ix = None
		self.vname = None
		self.plot_params = None


	def load_plot_params(self):
		# Ignore if Variable / time have not yet been selected
		if self.vname is None: return
		if self.time_ix is None: return
		self.plot_params = self.vmgr.plot_params(self.vname, self.time_ix)
		self.events.run_event('plot_params_changed', self, self.plot_params)

	def set_time_ix(self, time_ix):
		self.time_ix = time_ix
		self.load_plot_params()

	def set_var(self, vname):
		self.vname = vname
		self.load_plot_params()

# ----------------------------------------------------
# http://stackoverflow.com/questions/2726839/creating-a-pygtk-text-field-that-only-accepts-number
class NumberEntry(Gtk.Entry):
	def __init__(self):
		Gtk.Entry.__init__(self)
		self.connect('changed', self.on_changed)

	def on_changed(self, *args):
		text = self.get_text().strip()
		self.set_text(''.join([i for i in text if i in '0123456789.e-']))

# ----------------------------------------------------
class ArgsPanel(object):
	def __init__(self, gmodel):
		self.gmodel = gmodel

		listbox = Gtk.ListBox()
		listbox.set_selection_mode(Gtk.SelectionMode.NONE)

		widgets = list()
		self.vmin_w = NumberEntry()
		widgets.append(('vmin', self.vmin_w))
		self.vmax_w = NumberEntry()
		widgets.append(('vmax', self.vmax_w))

		button = Gtk.Button.new_with_label("Plot");
		button.connect("clicked", self.go_plot)
		widgets.append(('Plot', button))

		for label, widget in widgets:
			row = Gtk.ListBoxRow()
			hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=50)
			row.add(hbox)
			label = Gtk.Label(label, xalign=0)
#			check = Gtk.CheckButton()
			hbox.pack_start(label, True, True, 0)
			hbox.pack_start(widget, False, True, 0)

			listbox.add(row)

		self.the_widget = listbox	 # Caller looks for this

		# Listen to the gmodel
		self.gmodel.events.connect('plot_params_changed', self.on_plot_params_changed)

	def go_plot(self, *args):
		gmodel = self.gmodel

		# Start the figure
		figure = matplotlib.pyplot.figure(figsize=(4.25,5.5))
		ax = figure.add_subplot(111)

		# Obtain the plot_params, and complete it now.
		pp = copy.copy(gmodel.plot_params)

		# Put the value back in
		pp['val'] = gmodel.vmgr.val_t(gmodel.vname, gmodel.time_ix)

		# Get the basemap
		smod, sfn = pp['basemap_spec']
		basemap_fn = getattr(importlib.import_module(smod), sfn)
		pp['basemap'] = basemap_fn(ax)

		# Get the plotter
		pp['plotter'] = self.gmodel.vmgr.plotter(pp['plotter_spec'])

		# Load vmin and vmax from GUI
		plot_args = self.gmodel.plot_params['plot_args']
		plot_args['vmin'] = float(self.vmin_w.get_text())
		plot_args['vmax'] = float(self.vmax_w.get_text())

#		with gzip.open('plot.piz', 'wb') as f:
#			pickler = pickle.Pickler(f)
#			# pickler_add_trace(pickler, lambda x : print(type(x)))
#			pickler.dump(pp)
#			print('Done dumping plot.piz')

		giss.plot.plot_var(ax=ax, **pp)

		# Save to a file as png
		figure.savefig('fig.png', dpi=300, transparent=True)

		# Also show on screen
		matplotlib.pyplot.show()


	def on_plot_params_changed(self, *args):
		# Set args
		plot_args = self.gmodel.plot_params['plot_args']
		print(plot_args)
		self.vmin_w.set_text(str(plot_args['vmin']))
		self.vmax_w.set_text(str(plot_args['vmax']))

# ----------------------------------------------------
def on_time_selection_changed(gmodel, selection):
	model, treeiter = selection.get_selected()
	if treeiter != None:
		row = model[treeiter]
		gmodel.set_time_ix(row[0])
		print("You selected", row[0], row[1])

def time_panel(gmodel):

	#Creating the ListStore model
	liststore = Gtk.ListStore(int, str)
	for ix,times in enumerate(gmodel.vmgr.times):
		liststore.append((ix, str(times)))

	treeview = Gtk.TreeView.new_with_model(liststore)

	#creating the treeview, making it use the filter as a model, and adding the columns
	treeview = Gtk.TreeView.new_with_model(liststore)
	for i, column_title in enumerate(["index", "Time"]):
		renderer = Gtk.CellRendererText()
		column = Gtk.TreeViewColumn(column_title, renderer, text=i)
		treeview.append_column(column)

	#setting up the layout, putting the treeview in a scrollwindow, and the buttons in a row
	scrollable_treelist = Gtk.ScrolledWindow()
	scrollable_treelist.set_vexpand(True)
	scrollable_treelist.add(treeview)

	select = treeview.get_selection()
	select.set_mode(Gtk.SelectionMode.SINGLE)
	select.connect("changed", lambda selection : on_time_selection_changed(gmodel, selection))

	return scrollable_treelist

# ----------------------------------------------------
def on_var_selection_changed(gmodel, selection):
	model, treeiter = selection.get_selected()
	if treeiter != None:
		ix = model[treeiter][0]
		entry = gmodel.vmgr.entries[ix]
		gmodel.set_var(entry.vname)

def vars_panel(gmodel):

	#Creating the ListStore model
	var_liststore = Gtk.ListStore(int, str, str)
	for ix,entry in enumerate(gmodel.vmgr.entries):
		var_liststore.append((ix, entry.vname, str(entry.shape)))

	treeview = Gtk.TreeView.new_with_model(var_liststore)

	#creating the treeview, making it use the filter as a model, and adding the columns
	treeview = Gtk.TreeView.new_with_model(var_liststore)
	treeview.append_column(Gtk.TreeViewColumn('Variable', Gtk.CellRendererText(), text=1))
	treeview.append_column(Gtk.TreeViewColumn('Shape', Gtk.CellRendererText(), text=2))

#	for i, column_title in enumerate(["Variable", "Shape"]):
#		renderer = Gtk.CellRendererText()
#		column = Gtk.TreeViewColumn(column_title, renderer, text=i)
#		treeview.append_column(column)


	#setting up the layout, putting the treeview in a scrollwindow, and the buttons in a row
	scrollable_treelist = Gtk.ScrolledWindow()
	scrollable_treelist.set_vexpand(True)
	scrollable_treelist.add(treeview)

	select = treeview.get_selection()
	select.set_mode(Gtk.SelectionMode.SINGLE)
	select.connect("changed", lambda selection : on_var_selection_changed(gmodel, selection))

	return scrollable_treelist

# ----------------------------------------------------

class PCViewWindow(Gtk.Window):
	def __init__(self, gmodel):
		"""gmodel: PCViewModel"""
		self.gmodel = gmodel

		Gtk.Window.__init__(self, title=gmodel.title)
		self.set_border_width(10)

		#Setting up the self.grid in which the elements are to be positionned
		self.grid = Gtk.Grid()
		self.grid.set_column_homogeneous(True)
		self.grid.set_row_homogeneous(True)
		self.add(self.grid)


		self.grid.attach(vars_panel(self.gmodel), 0, 0, 8, 10)
		self.grid.attach(time_panel(self.gmodel), 0, 10, 8, 10)
		args_panel = ArgsPanel(self.gmodel)
		self.grid.attach(args_panel.the_widget, 0,20,8,10)
		self.show_all()



#	def on_tree_selection_changed(self, selection, model, path, is_set):
#path, is_selected):
#	def on_tree_selection_changed(self, selection, model, path, is_set):
#		print('yyy', model[path][0], is_set)
#		model, treeiter = selection.get_selected()
#		if treeiter != None:
#			print("You selected", model[treeiter][0])
#		return True
		

