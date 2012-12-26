# Simplest possible ModelE data plotting demo

import netCDF4
import giss.basemap
import giss.modele
import sys

if len(sys.argv) < 2 :
	var_name = 'tsurf'
else :
	var_name = sys.argv[1]


nc = netCDF4.Dataset('data/ANN1950.aijhctest45_lr05.nc')
pp = giss.modele.plot_params(var_name, nc=nc)
giss.plot.plot_var(**pp)		# Plot, and show on screen

# Slightly more complex alternatives:
# Save figure:
# 	giss.plot.plot_var(fname='plottest1.png', **pp)
# Save figure and snow on screen
# 	giss.plot.plot_var(fname='plottest1.png', show=True, **pp)
