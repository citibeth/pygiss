import giss.plotutil
import giss.util
import numpy as np
import numpy.ma as ma

_zero_centered = {'impm', 'evap_lndice', 'imph_lndice', 'impm_lndice', 'netht_lndice', 'trht_lndice'}#, 'sensht_lndice'} #, 'trht_lndice'}

_reverse_scale = {'impm', 'impm_lndice'}

_rescale_factors = {'impm_lndice' : (86400., 'kg/day*m^2')}

# Reads a variable out of a netCDF file and plots it.
# @return Info structure, good for title, units and other stuff
def plot_var(plotter, mymap, scaled_nc, var_name, val=None, info=None, **_plotargs) :

	plotargs = dict(_plotargs)

	# Look up units and long_name
	if info is None :
		if var_name in scaled_nc.variables :
			info = dict(scaled_nc.variables[var_name].__dict__)
		else :
			info = {'units' : '<units>', 'long_name' : '<long-name>'}

	# Get the value (if we weren't already passed it)
	if val is None :
		val = giss.modele.read_ncvar(scaled_nc, var_name)
	else :
		val = ma.copy(val)

	# Rescale the variable if called for
	if var_name in _rescale_factors :
		rs = _rescale_factors[var_name]
		val *= rs[0]
		info['units'] = rs[1]

	# See if we should use a positive/negative colorbar
	cbargs = {}
	info['cbargs'] = cbargs

	if var_name in _zero_centered :
		if 'norm' not in plotargs :
			plotargs['norm'] = giss.plotutil.AsymmetricNormalize()
		if 'cmap' not in plotargs :
			reverse = (var_name in _reverse_scale)
			plotargs['cmap'] = giss.plotutil.cpt('BlRe', reverse=reverse).cmap
		if 'vmin' not in plotargs :
			plotargs['vmin'] = np.min(val)
		if 'vmax' not in plotargs :
			plotargs['vmax'] = np.max(val)
		cbargs['ticks'] = [plotargs['vmin'], 0, plotargs['vmax']]
		cbargs['format'] = '%0.2f'

	info['im'] = plotter.pcolormesh(mymap, val, **plotargs)
	info['sname'] = var_name

	return giss.util.Struct(info)
