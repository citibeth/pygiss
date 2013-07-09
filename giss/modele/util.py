import numpy as np
import numpy.ma as ma
import netCDF4
import giss.util
import giss.bashfile
import os


def read_ncvar(nc, var_name, dtype='d') :
	"""Reads a variable out of a scaled ACC file (netCDF format).

	Understands the missing_value attribute (or without it, just
	treates anything >1e10 as missing).

	Returns:	(Numpy Array)
		The variable.  Invalid values are set to np.nan

	"""

	var = nc.variables[var_name]
	val = np.zeros(var.shape, dtype=dtype)
	val[:] = var[:]
#	val = var[:]


	if 'missing_value' in var.__dict__ :
		val[val == var.missing_value] = np.nan
	else :
		# Use generic cutoff
		val[np.abs(val) > 1.e10] = np.nan

	return val


# def read_ncvar(nc, var_name, mask_var = True) :
# 	"""Reads a variable out of a scaled ACC file (netCDF format).
# 
# 	Understands the missing_value attribute (or without it, just
# 	treates anything >1e10 as missing).
# 
# 	Returns:	(np.ma.MaskedArray)
# 		The variable.  Invalid values are masked
# 		out.  Inside the data array, they are also set to np.nan
# 
# 	"""
# 
# 	var = nc.variables[var_name]
# 	val = var[:]
# 
# 	# Mask the variable
# 	if mask_var :
# 		if 'missing_value' in var.__dict__ :
# 			val = ma.masked_where(val == var.missing_value, val)
# 		else :
# 			# Use generic cutoff
# 			val = ma.masked_where(np.abs(val) > 1.e10, val)
# 
# 	# Eliminate masked values.  This way, things that don't understand
# 	# masking (eg. snowdrift) won't get confused.
# 	if ma.getmask(val) is not ma.nomask :
# 		val.data[ma.getmask(val)] = np.nan
# 
# 
# 	return val
# 
# --------------------------------------------------------
def read_modelerc(fname = None) :
	"""Reads the settings in the user's .modelErc file.

	Returns:	{string : string}
		Dictionary of the name/value pairs found in the file.

	See:
		giss.bashfile.read_env()"""
		
	if fname is None :
		fname = os.path.join(os.environ['HOME'], '.modelErc')

	return giss.bashfile.read_env(fname)
# --------------------------------------------------------
