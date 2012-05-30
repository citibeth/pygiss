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

matplotlib.use('Agg')

# ------------------------------------------------------------------
# See: http://stackoverflow.com/questions/2216273/irregular-matplotlib-date-x-axis-labels

# Image plot
#Acceptable values are None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
def plot_image(ax, Z, extent) :
	Zt = Z.transpose()
	Zmasked = np.ma.array(Zt, mask=np.isnan(Zt))

	ret = ax.imshow(Zmasked, origin='lower',
		interpolation='nearest',
		extent=np.array([x0,x1,y0,y1]),
		cmap='bwr', alpha=.8)

# Contour plot
#	ret = ax.contour(Z.transpose(), interpolation='nearest', origin='lower')
	return ret

# ------------------------------------------------------------------

#overlap_fname = sys.argv[1]
overlap_fname = 'searise_ll_overlap-5.nc'


fig = plt.figure(figsize=(figure.X, figure.Y))	# Size of figure (inches)

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

raster_x = ice_nx
raster_y = ice_ny

# Open our searise data
searise_nc = netCDF4.Dataset(os.path.join(sys.path[0], 'data/searise/Greenland_5km_v1.1.nc'))

# ========= Get a simple low-res field (checkerboard)
tops = np.array([200,400,700,1000,1300,1600,2000,2500,3000,10000], dtype='d')
num_hclass = tops.shape[0]
height_max1 = np.tile(tops, (n1,1))		# Produces an n1xnum_hclass array

nheight=10
ZG = np.zeros((n1, num_hclass))
ZG[:] = np.nan
val = -1.0
for heighti in range(0,num_hclass) :
	for lat in range(0,nlat) :
		for lon in range(0,nlon,2) :
			ZG[lat*nlon + lon, heighti] = val
			ZG[lat*nlon + lon + 1, heighti] = -val
		val = -val
	val = -val

# ========= Get the land mask
mask2 = get_landmask(searise_nc)

# ---- Make a simple field based on it
#ZH = np.ones(n2)
#ZH[mask2==0] = np.nan

# ======== Get orography (elevation)
topg = np.array(searise_nc.variables['topg'], dtype='d').flatten('C')
thk = np.array(searise_nc.variables['thk'], dtype='d').flatten('C')
elevation2 = topg + thk

# ============= Rasterize and Plot original field

ZG_r = np.zeros((raster_x, raster_y))
grid1 = snowdrift.Grid(overlap_fname, 'grid1')
rast1 = snowdrift.Rasterizer(grid1, x0,x1,raster_x, y0,y1,raster_y);
grid2 = snowdrift.Grid(overlap_fname, 'grid2')
rast2 = snowdrift.Rasterizer(grid2, x0,x1,raster_x, y0,y1,raster_y);
#snowdrift.rasterize(rast1, ZG, ZG_r)
snowdrift.rasterize_hc(rast1, rast2, ZG, elevation2, mask2, height_max1, ZG_r)

ax = fig.add_subplot(1,1,curplot)
curplot += 1
init_plot(ax, '', x0,x1,y0,y1)

cax2 = plot_image(ax, ZG_r, np.array([x0,x1,y0,y1]))
#fig.colorbar(cax2)
ax.plot(greenland_xy[0], greenland_xy[1], 'grey', alpha=.9)

basename = os.path.splitext(sys.argv[0])[0]
fig.savefig(basename + '.png', format='png', transparent=True, dpi=figure.DPI, bbox_inches='tight')


#plt.show()
