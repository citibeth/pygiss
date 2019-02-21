import functools,operator
from giss.functional import *
from giss import functional
from giss import checksum,memoize
import collections
import cf_units

# -----------------------------------------------
# Stuff for xaccess

# -------------------------------------------------------------
@memoize.local()
def ncopen(name):
    nc = netCDF4.Dataset(name, 'r')
    return nc

# ---------------------------------------------------------------
def ncdata(fname, var_name, *index, nan=np.nan, missing_value=None, missing_threshold=None):
    """Simple accessor function for data in NetCDF files.
    Ops on this aren't very interesting because it is a
    fully-bound thunk."""

    nc = ncopen(fname)
    var = nc.variables[var_name]

    data = var[index]

    if not np.issubdtype(var.dtype, np.integer):
        # Missing value stuff only makes sense for floating point

        if missing_value is not None:
            # User override of NetCDF standard
            data[data == missing_value] = nan
        elif hasattr(var, 'missing_value'):
            # NetCDF standard
            data[data == var.missing_value] = nan
        elif missing_threshold is not None:
            # Last attempt to fix a broken file
            data[np.abs(data) > missing_threshold] = nan

    return data

# --------------------------------------------
def _fetch_shape(var, index):
    """Given a variable and indexing, determines the shape of the
    resulting fetch (without actually doing it)."""
    shape = []
    dims = []
    for i in range(0,len(var.shape)):
        if i >= len(index):    # Implied ':' for this dim
            dims.append(var.dimensions[i])
            shape.append(var.shape[i])
        else:
            ix = index[i]
            if isinstance(ix, slice):
                dims.append(var.dimensions[i])
                start = 0 if ix.start is None else ix.start
                stop = var.shape[i] if ix.stop is None else ix.stop
                step = 1 if ix.step is None else ix.step
                shape.append((stop - start) // step)
    return tuple(shape), tuple(dims)

FetchTuple = xnamedtuple('FetchTuple', ('attrs', 'data'))



@function()
def ncattrs(file_name, var_name):
    """Produces extended attributes on a variable fetch operation"""
    nc = ncopen(file_name)

    attrs = {}
    var = nc.variables[var_name]

    # User can retrieve nc.ncattrs(), etc.
    for key in nc.ncattrs():
        attrs[('file', key)] = getattr(nc, key)

    # User can retrieve var.dimensions, var.shape, var.name, var.xxx, var.ncattrs(), etc.

    attrs[('var', 'dimensions')] = var.dimensions
    attrs[('var', 'dtype')] = var.dtype
    attrs[('var', 'datatype')] = var.datatype
    attrs[('var', 'ndim')] = var.ndim
    attrs[('var', 'shape')] = var.shape
    attrs[('var', 'scale')] = var.scale
    # Don't know why this doesn't work.  See:
    # http://unidata.github.io/netcdf4-python/#netCDF4.Variable
    # attrs[('var', 'least_significant_digit')] = var.least_significant_digit
    attrs[('var', 'name')] = var.name
    attrs[('var', 'size')] = var.size
    for key in var.ncattrs():
        attrs[('var', key)] = getattr(var, key)


    return wrap_combine(attrs, intersect_dicts)

def add_fetch_attrs(attrs, file_name, var_name, *index, nan=np.nan, missing_value=None, missing_threshold=None):

    nc = ncopen(file_name)
    var = nc.variables[var_name]

    attrs[('fetch', 'file_name')] = file_name
    attrs[('fetch', 'var_name')] = var_name
    attrs[('fetch', 'missing_value')] = missing_value
    attrs[('fetch', 'missing_threshold')] = missing_threshold
    attrs[('fetch', 'nan')] = np.nan
    attrs[('fetch', 'index')] = index
    fetch_shape,fetch_dims = _fetch_shape(var, index)
    attrs[('fetch', 'shape')] = fetch_shape
    attrs[('fetch', 'dimensions')] = fetch_dims
    attrs[('fetch', 'size')] = functools.reduce(operator.mul, fetch_shape, 1)
    attrs[('fetch', 'dtype')] = attrs[('var', 'dtype')]


@function()
def ncfetch(file_name, var_name, *index, nan=np.nan, missing_value=None, missing_threshold=None, **kwargs):

    attrsW = ncattrs(file_name, var_name)
    add_fetch_attrs(attrsW(), file_name, var_name, **kwargs)

    return FetchTuple(
        attrsW,
            bind(ncdata, file_name, var_name, *index, **kwargs))
# ---------------------------------------------------------
@function()
def sum_fetch1(fetch, axis=None, dtype=None, out=None, keepdims=False):
    """Works like np.sum, but on a fetch record..."""
    attrs0 = fetch.attrs()
    data1 = fetch.data

    ashape = attrs0[('fetch', 'shape')]
    adims = attrs0[('fetch', 'dimensions')]

    # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.sum.html
    # axis : None or int or tuple of ints, optional
    # 
    # Axis or axes along which a sum is performed. The default (axis =
    # None) is perform a sum over all the dimensions of the input
    # array. axis may be negative, in which case it counts from the
    # last to the first axis.
    #
    # If this is a tuple of ints, a sum is performed on multiple axes,
    # instead of a single axis or all the axes as before.
    if axis is None:
        axes = set(range(len(ashape)))
    if isinstance(axis, tuple):
        axes = set(axis)
    else:
        axes = set((axis,))

    shape = []
    dims = []
    for i in range(0,len(ashape)):
        if i in axes:
            if keepdims:
                shape.append(1)
                dims.append(adims[i])
        else:
            shape.append(ashape[i])
            dims.append(adims[i])

    attrs0[('fetch', 'shape')] = tuple(shape)
    attrs0[('fetch', 'dimensions')] = tuple(dims)

    data = functional.lift_once(np.sum, fetch.data, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    return FetchTuple(fetch.attrs, data)

sum_fetch2 = functional.lift()(sum_fetch1)
# ---------------------------------------------------------
_zero_one = np.array([0.,1.])

def convert_unitsV(fetch, ounits):
    """fetch:
        Result of the fetch() function

    V = works on values (not functions)
    """
    attrs = fetch.attrs()
    cf_iunits = cf_units.Unit(attrs[('var', 'units')])
    cf_ounits = cf_units.Unit(ounits)
    zo2 = cf_iunits.convert(_zero_one, cf_ounits)

    # y = mx + b slope & intercept
    b = zo2[0]    # b = y-intercept
    m = zo2[1] - zo2[0]    # m = slope

    attrs[('var', 'units')] = ounits
#    return attrdictx(
#        attrs = fetch.attrs,
#        data = fetch.data*m + b)
    return FetchTuple(fetch.attrs, fetch.data*m + b)

convert_unitsF = functional.lift()(convert_unitsV)


