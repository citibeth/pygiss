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
import string

km = 1000.0




#matplotlib.use('Agg')

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
		#cmap='bwr', 
		alpha=1)

# Contour plot
#	ret = ax.contour(Z.transpose(), interpolation='nearest', origin='lower')


	return ret


def plot_T(ax, Z, extent, cpt_fname) :
	Zt = Z.transpose()
	Zmasked = np.ma.array(Zt, mask=np.isnan(Zt))

	cmap,vmin,vmax = giss.plotutil.read_cpt(cpt_fname)
	ret = ax.imshow(Zmasked, origin='lower',
		interpolation='nearest',
		extent=np.array([x0,x1,y0,y1]),
		vmin=vmin,vmax=vmax,
		cmap=cmap,
		alpha=1)

# Contour plot
#	ret = ax.contour(Z.transpose(), interpolation='nearest', origin='lower')


	return ret



# ------------------------------------------------------------------

#overlap_fname = sys.argv[1]
overlap_fname = 'searise_ll_overlap-5.nc'


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


# ========= Get the land mask
mask2 = get_landmask(searise_nc)

# ---- Make a simple field based on it
#ZH = np.ones(n2)
#ZH[mask2==0] = np.nan


# ============= Initialize Rasterizers
grid1 = snowdrift.Grid(overlap_fname, 'grid1')
rast1 = snowdrift.Rasterizer(grid1, x0,x1,raster_x, y0,y1,raster_y);
grid2 = snowdrift.Grid(overlap_fname, 'grid2')
rast2 = snowdrift.Rasterizer(grid2, x0,x1,raster_x, y0,y1,raster_y);


# ==============================================================
# Part 2: Downscaled Temperature
# ==============================================================

# ========= Setup height classes
tops = np.array([200,400,700,1000,1300,1600,2000,2500,3000,10000], dtype='d')
num_hclass = tops.shape[0]
height_max1 = np.tile(tops, (n1,1))		# Produces an n1xnum_hclass array

# ======== Get orography (elevation)
topg = np.array(searise_nc.variables['topg'], dtype='d').flatten('C')
thk = np.array(searise_nc.variables['thk'], dtype='d').flatten('C')
elevation2 = topg + thk

# ============= Set up Snowdrift data structures
sd = snowdrift.Snowdrift(overlap_fname)
sd.init(elevation2, mask2, height_max1)

for month in ['JUN', 'DEC'] :
	fig = plt.figure(figsize=(figure.X*2+figure.colorbarX*1,figure.Y))	# Size of figure (inches)
	curplot=1

	modele_nc = netCDF4.Dataset(os.path.join(sys.path[0],
	'data/gavin/cmip5a/GISS-E2R/NINT/E135f9aF40oQ32/' + month + '1980-2004.aijE135f9aF40oQ32.nc'))

	# ========= Read T and elevation from ModelE
	tsurf1 = np.array(modele_nc.variables['tsurf_lndice'], dtype='d').flatten('C')
	tsurf1[abs(tsurf1) > 1e10] = np.nan
	elevation1 = np.array(modele_nc.variables['topog'], dtype='d').flatten('C')


	# ============== Get mean elevation per height-classified GCM cell
	elevation1h = np.zeros((n1,num_hclass))
	elevation1h[:,:] = np.nan
	sd.upgrid(elevation2, elevation1h)

	# ========= Downscale temperature using simple lapse rate
	lapse_rate = -6.5 / km;
	tsurf1h = np.zeros((n1,num_hclass))
	tsurf1h_down = np.zeros((n1,num_hclass))
	tsurf1h_down[:] = np.nan
	tsurf1h_up = np.zeros((n1,num_hclass))
	tsurf1h_up[:] = np.nan
	for i1 in range(0,n1) :
		for hc in range(0,num_hclass) :
			ele = elevation1h[i1,hc]
			#if not np.isnan(ele) :
			tsurf1h[i1,hc] = tsurf1[i1] + (ele - elevation1[i1]) * lapse_rate
			if ele <= elevation1[i1] :
				tsurf1h_down[i1,hc] = tsurf1h[i1,hc]
			if ele > elevation1[i1] :
				tsurf1h_up[i1,hc] = tsurf1h[i1,hc]

	# ============ Get colormap
	cpt_fname = os.path.join(sys.path[0], 'temp-greenland-' + string.lower(month) + '.cpt')

	# ============= Plot original field
	tsurf1_r = np.zeros((raster_x, raster_y))
	#snowdrift.rasterize(rast1, tsurf1, tsurf1_r)
	snowdrift.rasterize_mask(rast1, rast2, tsurf1, mask2, tsurf1_r)

	ax = fig.add_subplot(1,2,curplot)
	curplot += 1
	init_plot(ax, 'GCM T', x0,x1,y0,y1)

	cax2 = plot_T(ax, tsurf1_r, np.array([x0,x1,y0,y1]), cpt_fname)
	#fig.colorbar(cax2)
	ax.plot(greenland_xy[0], greenland_xy[1], 'grey', alpha=.9)

	# ========== Plot downscaled temperature
	tsurf1h_r = np.zeros((raster_x, raster_y))
	snowdrift.rasterize_hc(rast1, rast2, tsurf1h, elevation2, mask2, height_max1, tsurf1h_r)

	ax = fig.add_subplot(1,2,curplot)
	curplot += 1
	init_plot(ax, 'Downscaled T', x0,x1,y0,y1)

	cax2 = plot_T(ax, tsurf1h_r, np.array([x0,x1,y0,y1]), cpt_fname)
	fig.colorbar(cax2)
	ax.plot(greenland_xy[0], greenland_xy[1], 'grey', alpha=.9)


	# ============= Store output
	#fig.savefig('gcm_grid.eps', format='eps')
	#fig.savefig('gcm_grid.pdf', format='pdf')
	basename = os.path.splitext(sys.argv[0])[0]
	fig.savefig(basename + '-' + string.lower(month) + '.png', format='png',
		transparent=True,
		dpi=figure.DPI, bbox_inches='tight')

# // end for month


# ===========================================================
# Plot elevation fields to a different file
# ===========================================================

fig = plt.figure(figsize=(figure.X*2+figure.colorbarX*2,figure.Y))	# Size of figure (inches)
curplot=1

# ======== Plot original (ModelE) elevation field
elevation1_r = np.zeros((raster_x, raster_y))
snowdrift.rasterize(rast1, elevation1, elevation1_r)
#snowdrift.rasterize_hc(rast1, rast2, elevation1, elevation2, mask2, height_max1, elevation1_r)

ax = fig.add_subplot(1,2,curplot)
curplot += 1
init_plot(ax, 'elevation1', x0,x1,y0,y1)

cax2 = plot_image(ax, elevation1_r, np.array([x0,x1,y0,y1]))
fig.colorbar(cax2)
ax.plot(greenland_xy[0], greenland_xy[1], 'grey', alpha=.9)

# ========= Plot height-classified elevation
elevation1h_r = np.zeros((raster_x, raster_y))
#snowdrift.rasterize(rast1, elevation1h, elevation1h_r)
snowdrift.rasterize_hc(rast1, rast2, elevation1h, elevation2, mask2, height_max1, elevation1h_r)

ax = fig.add_subplot(1,2,curplot)
curplot += 1
init_plot(ax, 'elevation1h', x0,x1,y0,y1)

cax2 = plot_image(ax, elevation1h_r, np.array([x0,x1,y0,y1]))
fig.colorbar(cax2)
ax.plot(greenland_xy[0], greenland_xy[1], 'grey', alpha=.9)

# ============= Store output
#fig.savefig('gcm_grid.eps', format='eps')
#fig.savefig('gcm_grid.pdf', format='pdf')
basename = os.path.splitext(sys.argv[0])[0]
fig.savefig(basename + '-elevation.png', format='png',
	transparent=True,
	dpi=figure.DPI, bbox_inches='tight')


#plt.show()
