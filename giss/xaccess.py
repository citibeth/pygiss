import numpy as np
import importlib

__all__ = ('_ix',)

# -------------------------------------------------------------
# Allows us to do indexing without an array to index ON.
#   We can say:   index = _ix[4:5,:]
class IndexClass(object):
    """Extract slice objects"""
    def __getitem__(self, t):
        if isinstance(t, tuple):
            return t
        return (t,)

# Singleton allows us to say, eg: ix[2,3,4:5]
_ix = IndexClass()

def reslice_subdim(_slice, subdim):
    """A sub-dimension is a portion of the range of a full dimension.  For
    example... if dim i is in range 0...17, then a sub-dimension of i
    would be 2..5.  This subroutine takes a slice meant to apply to a
    sub-dimension, and converts it to a slice that works on the
    overall dimension.

    subdim: (begin, end)
        Begin and end(+1) indices of the sub-dimension
    """

    if isinstance(_slice, int):
        return _slice

    x = _slice.start
    if x is None:
        start = subdim[0]
    elif x < 0:
        start = subdim[1] + x
    else:
        start = subdim[0] + x

    x = _slice.stop
    if x is None:
        stop = subdim[1]
    elif x < 0:
        stop = subdim[1] + x
    else:
        stop = subdim[0] + x

    return slice(start, stop, _slice.step)

def reindex_subdim(ix, subdim):
    """Takes an index (result of _ix[]) meant to apply to a sub-dimension,
    and converts it to an index that works on the overall dimension."""
    return tuple(reslice_subdim(x, subdim) for x in ix)

# -----------------------------------------------------------
def get_plotter(attrs, **kwargs):
    """Create a plotter previously set up in ncfetch() or similar.
    attrs:
        (Unwrapped) attributes dict from ncfetch() or similar call."""

    fnname = attrs[('plotter', 'function')]
    mod = importlib.import_module(fnname[0])
    plotter_fn = getattr(mod, fnname[1])


    gkwargs = dict(attrs[('plotter', 'kwargs')], **kwargs)
    return plotter_fn(attrs, **gkwargs)
# -----------------------------------------------------------
def xfer(idict, ikey, odict, okey):
    if ikey in idict:
        odict[okey] = idict[ikey]

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


def plot_params(fetch):
    """Sets up default plot parameters, based on the result of a fetch.
    Can be modified later by the caller...
    Output to be used directly as kwargs for giss.plot.plot_var()

    Args:
        fetch:
            Result of a fetch command (eg ncfetch()

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

    attrs = fetch.attrs()

    # Transfer attributes
    pp = dict()
    xfer(attrs, ('var', 'units'), pp, 'units')
    xfer(attrs, ('var', 'name'), pp, 'var_name')

    pp['plotter'] = get_plotter(attrs)

    # Init kwargs for Plotter.pcolormesh() command
    plot_args = {}
    pp['plot_args'] = plot_args

    # Init kwargs for colorbar command
    cb_args = {}
    pp['cb_args'] = cb_args

    # Get the data
    # We need this now; plot parameters depend on VALUE of data.
    pp['val'] = fetch.data()

    # These could be decent defaults for anything with a colorbar
    cb_args['location'] = 'bottom'
    cb_args['size'] = '5%'
    cb_args['pad'] = '2%'

    # Suggest a title
    cond = ('var_name' in pp, 'units' in pp)
    if cond == (True, True):
        pp['title'] = '%s\n[%s]' % (pp['var_name'], pp['units'])
    elif cond == (True, False):
        pp['title'] = pp['var_name']
    elif cond == (False, True):
        pp['title'] = '[{}]'.format(pp['units'])
    else:
        pp['title'] = ''

    # Default coastlines
    pp['plot_boundaries'] = _default_plot_boundaries

    return pp
