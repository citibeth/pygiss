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


# ------------------------------------------------------------------
# See: http://stackoverflow.com/questions/2216273/irregular-matplotlib-date-x-axis-labels

# Image plot
#Acceptable values are None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
def plot_image(ax, Z, extent) :
	Zt = Z.transpose()
	Zmasked = np.ma.array(Zt, mask=np.isnan(Zt))

	zmin = np.min(Zmasked)
	zmax = np.max(Zmasked)

	absmax = max(-zmin,zmax)

	cpt_fname = os.path.join(sys.path[0], 'smb.cpt')

	cmap,vmin,vmax = giss.plotutil.read_cpt(cpt_fname)
	ret = ax.imshow(Zmasked, origin='lower',
		interpolation='nearest',
		extent=np.array([x0,x1,y0,y1]),
		#cmap=cmap,vmin=-absmax,vmax=absmax,
		cmap=cmap,vmin=vmin, vmax=vmax,
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
overlap_fname = 'searise_cesm_overlap.nc'


figure.init_figure(overlap_fname)
from figure import *		# Bring in our global variables

raster_x = ice_nx
raster_y = ice_ny

# Open our searise data
searise_nc = netCDF4.Dataset(os.path.join(figure.data_root, 'searise/Greenland_5km_v1.1.nc'))

# ========= Get the land mask
mask2 = get_landmask(searise_nc)

# ============= Initialize Rasterizers
grid1 = snowdrift.Grid(overlap_fname, 'grid1')
rast1 = snowdrift.Rasterizer(grid1, x0,x1,raster_x, y0,y1,raster_y);
grid2 = snowdrift.Grid(overlap_fname, 'grid2')
rast2 = snowdrift.Rasterizer(grid2, x0,x1,raster_x, y0,y1,raster_y);


fig = plt.figure(figsize=(figure.X*2+figure.colorbarX*1,figure.Y))	# Size of figure (inches)
curplot=1


# ==============================================================
# Part 2: Downscaled SMB
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

# ============== Read the SMB per height class
cesm_nc = netCDF4.Dataset(os.path.join(figure.data_root, 'cesm/smb_CESM.nc'))
smb1h = np.zeros((n1,num_hclass))
for hc in range(0,num_hclass) :
	varname = 's2x_Fgss_qice%02d' % (hc+1)
	smb_hc = np.array(cesm_nc.variables[varname], dtype='d').flatten('C')
	smb_hc[abs(smb_hc) > 1e10] = np.nan
	smb_hc *= 31556000.0	# Convert from kg m-2 s-2 to mm/yr 
	smb1h[:,hc] = smb_hc

# ============== Plot height-classified SMB
smb1h_r = np.zeros((raster_x, raster_y))
snowdrift.rasterize_hc(rast1, rast2, smb1h, elevation2, mask2, height_max1, smb1h_r)

ax = fig.add_subplot(1,2,curplot)
curplot += 1
init_plot(ax, 'SMB (mm/yr)', x0,x1,y0,y1)

color_ticks=[-1600 ,-1400 ,-1200 ,-1000 ,-800,-600,-400,-200,000,200   ,400,600]
cax2 = plot_image(ax, smb1h_r, np.array([x0,x1,y0,y1]))
cbar = fig.colorbar(cax2, ticks=color_ticks)
#cbar.ax.yaxis.set_ticklabels([''])
ax.plot(greenland_xy[0], greenland_xy[1], 'grey', alpha=.9)

# =======================================================
# Smoothed SMB

smb2 = np.zeros(ice_nx*ice_ny)
smb2[:] = np.nan
sd.downgrid(smb1h, smb2, use_snowdrift=1, merge_or_replace=1)

# ======== Plot it
smb2_r = np.zeros((raster_x, raster_y))
snowdrift.rasterize(rast2, smb2, smb2_r)

ax = fig.add_subplot(1,2,curplot)
curplot += 1
init_plot(ax, 'Smoothed SMB (mm/yr)', x0,x1,y0,y1)

cax2 = plot_image(ax, smb2_r, np.array([x0,x1,y0,y1]))
cbar = fig.colorbar(cax2, ticks=color_ticks)
#cbar.ax.yaxis.set_ticklabels([''])
ax.plot(greenland_xy[0], greenland_xy[1], 'grey', alpha=.9)

# ============= Store output
basename = os.path.splitext(sys.argv[0])[0]
fig.savefig(basename + '.png', format='png',
#	transparent=True,
	dpi=figure.DPI, bbox_inches='tight')
