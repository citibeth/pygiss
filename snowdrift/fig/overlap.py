# Set up a sample function on the ice grid, upgrids to the GCM grid,
# and then plot it (rasterized) on the GCM grid

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import snowdrift
import netCDF4
import sys
import pyproj
import time
import os
import giss.io.noaa
import giss.maputil
import giss.plotutil
import figure

#matplotlib.use('Agg')

# ------------------------------------------------------------------

#overlap_fname = sys.argv[1]
overlap_fname = 'searise_ll_overlap-50.nc'

fig = plt.figure(figsize=(figure.X,figure.Y))	# Size of figure (inches)

figure.init_figure(overlap_fname)
from figure import *		# Bring in our global variables

lonb = np.array(overlap_nc.variables['grid1.lon_boundaries'])
latb = np.array(overlap_nc.variables['grid1.lat_boundaries'])
#lon0 = lonb[0]
#lon1 = lonb[-1]
#lat0 = latb[0]
#lat1 = latb[-1]
nlon = len(lonb)-1
nlat = len(latb)-1

print 'nlon=',nlon

#raster_x = ice_nx
#raster_y = ice_ny

# Open our searise data
#searise_nc = netCDF4.Dataset(os.path.join(sys.path[0], 'data/searise/Greenland_5km_v1.1.nc'))

# -----------------------------------------------------------------------
ax = fig.add_subplot(1,1,curplot)
curplot += 1
#init_plot(ax, '', x0,x1,y0,y1)

ax.set_title('Exchange Grid')

giss.maputil.plot_graticules(ax, range(-75,1,10), range(60, 81, 4), x0,x1,y0,y1, projs, False)
ax.set_xlim((x0, x1))
ax.set_ylim((y0, y1))



# ========= Plot the ice grid lines
(xdata, ydata) = giss.plotutil.points_to_plotlines(
	np.array(overlap_nc.variables['grid2.polygons']),
	np.array(overlap_nc.variables['grid2.points']));
ax.plot(xdata, ydata, 'green', alpha=.7, linewidth=.5)

# ======== Plot the GCM grid lines
(xdata, ydata) = giss.plotutil.points_to_plotlines(
	np.array(overlap_nc.variables['grid1.polygons']),
	np.array(overlap_nc.variables['grid1.points']));
ax.plot(xdata, ydata, 'blue', alpha=.7)


ax.plot(greenland_xy[0], greenland_xy[1], 'grey', alpha=.9)


# ============= Store output
#fig.savefig('gcm_grid.eps', format='eps')
#fig.savefig('gcm_grid.pdf', format='pdf')
basename = os.path.splitext(sys.argv[0])[0]
fig.savefig(basename + '.png', format='png',
#	transparent=True,
	dpi=figure.DPI, bbox_inches='tight')


#plt.show()

