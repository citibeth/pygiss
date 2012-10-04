import numpy as np
import numpy.ma as ma
import netCDF4
import giss.util
import giss.io.bashfile
import giss.ncutil
import os

# Utilities to read, post-process, analyze and plot output from ModelE
#
# --------------------------------------------------------
# Same as giss.util.read_ncvar, but eliminates bad data
# @return numpy masked array
def read_ncvar(nc, var_name, mask_var = True) :
	var = nc.variables[var_name]
	val = giss.ncutil.read_ncvar(nc, var_name)

	# Mask the variable
	if mask_var :
		if 'missing_value' in var.__dict__ :
			val = ma.masked_where(val == var.missing_value, val)
		else :
			# Use generic cutoff
			val = ma.masked_where(np.abs(val) > 1.e10, val)

	# Eliminate masked values.  This way, things that don't understand
	# masking (eg. snowdrift) won't get confused.
	val.data[ma.getmask(val)] = np.nan

	return val

# --------------------------------------------------------
# Reads a bunch of variables from a lat/lon scaled file
# @return (variables, grid-info)
def read_scaledacc(fname, vars, mask_var = True) :
	nc = netCDF4.Dataset(fname, 'r')
	ret = {}

	# Assume lat-lon file for now, this will be relaxed with Cubed Sphere
	grid = {}
	grid['type'] = 'll'
	grid['lon'] = giss.ncutil.read_ncvar(nc, 'lon')
	grid['lat'] = giss.ncutil.read_ncvar(nc, 'lat')

	for v in vars :
		var = dict(nc.variables[v].__dict__)
		val = giss.ncutil.read_ncvar(nc, v)
		# Mask the variable
		if mask_var :
			if 'missing_value' in var :
				val = ma.masked_where(val == var['missing_value'], val)
			else :
				# Use generic cutoff
				val = ma.masked_where(np.abs(val) > 1.e10, val)

		var['sname'] = v
		var['val'] = val
		ret[v] = giss.util.Struct(var)

	return (ret, giss.util.Struct(grid))
# --------------------------------------------------------
## Remove missing_value items from a variable
#def mask_acc(var) :
#	if 'missing_value' in var.__dict__ :
#		return ma.masked_where(var.val == missing_value, var.val)
#	else :
#		# Use generic cutoff
#		return ma.masked_where(np.abs(var.val) > 1.e10, var.val)
# --------------------------------------------------------
# Constructs an acc variable Struct
def acc_var(sname2, val2, **kwargs) :
	var = dict(kwargs)
	var['sname' ] = sname2
	var['val'] = val2
	return giss.util.Struct(var)

# --------------------------------------------------------
# @return a dict of name/value pairs
def read_modelerc(fname = None) :
	if fname is None :
		fname = os.path.join(os.environ['HOME'], '.modelErc')

	return giss.io.bashfile.read_env(fname)
# --------------------------------------------------------
