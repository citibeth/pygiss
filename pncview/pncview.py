import gi
import netCDF4
import sys
import giss.modele
import importlib
from giss import thunkserver
import os
import types
import pncview.varmgr.multi

# http://python-gtk-3-tutorial.readthedocs.org/en/latest/treeview.html#the-view

# ----------------------------------------------------
# ----------------------------------------------------

# ====================================================
# BEGIN Server-side stuff


def default_config():
    """Produces the built-in hardcoded default configuration"""
    config = dict()
    config['varmgrs'] = [ \
        'pncview.varmgr.modele.ModelEVarManager',
        'pncview.varmgr.pism.PISMVarManager']

    return config

def read_configs(start_dir):
    """Reads a hierarchy of pncview.rc files, starting with default_config()."""
    dir = os.path.abspath(start_dir)

    HOME = os.environ['HOME']
    config = default_config()

    # Config files to read (in order)
    fnames = list()
    while True:
        fname = os.path.join(dir, 'pncview.rc')
        if os.path.isfile(fname):
            fnames.append(fname)
        if dir == HOME:
            break

        # Go to <dir>/..
        split = os.path.split(dir)
        if (split[1] == ''):
            # We're at /, nowhere more to go
            break
        dir = split[0]

    for fname in reversed(fnames):
        try:
            with open(fname, 'rb') as fin:
                scode = fin.read()

            exec(compile(scode, fname, 'exec'), config)
            print('Found config file {}'.format(fname))
        except Exception as e:
            sys.stderr.write('WARNING: Error reading config file {}: {}'.format(fname, e))

    # -------------- Remove stuff caller doesn't want to see
    try:
        del config['__builtins__']
    except:
        pass
    for k,v in list(config.items()):
        if isinstance(v, types.ModuleType):
            del config[k]

    return config

def ss_init(con, fname):
    """
    fname: str
        Name of file to open
    varmgrs: [str]
        List of default VarManagers class names to try (in order) if we
        cannot determine the type."""

    config = read_configs(os.path.split(fname)[0])
    print('Processed config:', config)

    # ------- Get list of varmgrs to try
    varmgrs = list()
    try:
        # Start with one specified in the data file!
        with netCDF4.Dataset(fname) as nc:
            varmgrs.append(nc.pncview_varmgr)
    except:
        pass

    # Add on varmgrs to try, from the config
    try:
        varmgrs += config['varmgrs']
    except:
        pass


    # -------- Try the varmgrs
    vmgr = None
    print('varmgrs to try:', varmgrs)
    for svarmgr in varmgrs:
        try:
            dot = svarmgr.rindex('.')
            smod = svarmgr[:dot]
            sklass = svarmgr[dot+1:]
            klass = getattr(importlib.import_module(smod), sklass)
            vmgr = klass(fname, config)
            break
        except Exception as e:
            sys.stderr.write("VarMgr '{}' doesn't work:\n    {}\n".format(svarmgr, e))

    # See that we got one
    if vmgr is None:
        raise ValueError('Could not find any varmgr for {}\n'.format(fname))


    vmgr = pncview.varmgr.multi.MultiVarManager([vmgr], config)

    con['vmgr'] = vmgr
    con['fname'] = fname

# END Server-Side Stuff
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
        self.fname, self.entries, self.times = self.thunkserver.exec( \
            thunkserver.ObjThunk(self.context_vname, 'get_info'))

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
        ret = self._plotters[spec]
        print('plotter({}) --> {}'.format(spec, ret))
        return ret

    def val_t(self, *args):
        """Returns the value to be plotted.
        vname:
            Name of variable in VarMgr to plot
        time_ix:
            Timestep to plot"""
        return self.thunkserver.exec(thunkserver.ObjThunk(self.context_vname, 'val_t',  *args))

# ----------------------------------------------------

