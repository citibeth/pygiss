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


# ------------------------------------------------------------------
# See: http://stackoverflow.com/questions/2216273/irregular-matplotlib-date-x-axis-labels


# Image plot
#Acceptable values are None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
def plot_image(ax, Z, extent) :
	Zt = Z.transpose()
	Zmasked = np.ma.array(Zt, mask=np.isnan(Zt))

	# For SMB
	cmap = 'bwr'
	norm = giss.plotutil.AsymmetricNormalize(vmin=-2.45,vmax=3.3)
	norm = giss.plotutil.AsymmetricNormalize(vmin=-2.45*.5,vmax=3.3*.5)

#	# For precipitation
	cmap = 'spectral'
	norm = matplotlib.colors.Normalize(vmin=0,vmax=4.5)	# For precipitation

	ret = ax.imshow(Zmasked, origin='lower',
		interpolation='nearest',
		extent=np.array([x0,x1,y0,y1]),
		cmap=cmap, norm=norm)
#		cmap='bwr', vmin=-3.4,vmax=3.4)		# For SMB
#		cmap='spectral', vmin=0,vmax=4.5)	# For precipitation

# Contour plot
#	cax = ax.contour(ZG0_r.transpose(), interpolation='nearest', origin='lower')


	return ret

# ------------------------------------------------------------------


overlap_fname = sys.argv[1]
field_name = sys.argv[2]

fig = plt.figure(figsize=(1000/80,400/80))	# Size of figure (inches)


figure.init_figure(overlap_fname)
from figure import *		# Bring in our global variables

# =============== Read a hi-res function on the ice grid
searise_nc = netCDF4.Dataset(os.path.join(figure.data_root, 'searise/Greenland_5km_v1.1.nc'))
#ZH0 = np.ndarray.flatten(np.array(searise_nc.variables[field_name], dtype='d'))
ZH0 = np.array(searise_nc.variables[field_name], dtype='d').flatten('C')
print 'ZH0 has %d elements, vs (%d x %d) = %d' % (ZH0.shape[0], ice_nx, ice_ny, ice_nx * ice_ny)

# ============ Mask to only ice areas
mask2 = get_landmask(searise_nc)
ZH0[mask2==0] = np.nan

# ============= Rasterize and Plot original field

ZH0_r = np.zeros((raster_x, raster_y))
ZH0_r[:] = np.nan
grid2 = snowdrift.Grid(overlap_fname, 'grid2')
rast2 = snowdrift.Rasterizer(grid2, x0,x1,raster_x, y0,y1,raster_y);
snowdrift.rasterize(rast2, ZH0, ZH0_r)

ax = fig.add_subplot(1,3,curplot)
curplot += 1
init_plot(ax, field_name, x0,x1,y0,y1)

cax2 = plot_image(ax, ZH0_r, np.array([x0,x1,y0,y1]))
fig.colorbar(cax2)
ax.plot(greenland_xy[0], greenland_xy[1], 'black', alpha=.5)



# ============= Set up Snowdrift data structures
grid1_var = overlap_nc.variables['grid1.info']
n1 = grid1_var.__dict__['max_index'] - grid1_var.__dict__['index_base'] + 1
grid2_var = overlap_nc.variables['grid2.info']
n2 = grid2_var.__dict__['max_index'] - grid2_var.__dict__['index_base'] + 1
#info = sd.info()
#	n1 = info['n1']
#	n2 = info['n2']

sd = snowdrift.Snowdrift(overlap_fname)
topg = np.array(searise_nc.variables['topg'], dtype='d').flatten('C')
thk = np.array(searise_nc.variables['thk'], dtype='d').flatten('C')
elevation2 = topg + thk
print elevation2


# See p. 14 of "The CESM Land Ice Model: Documentation and User's Guide" by William Lipscomb (June 2010)
tops = np.array([10000], dtype='d')
#tops = np.array([200,400,700,1000,1300,1600,2000,2500,3000,10000], dtype='d')
#tops = np.array([400, 700, 1000, 1300, 1600, 2000, 2500, 3000, 10000], dtype='d')
#tops = np.array([100000], dtype='d')
num_hclass = tops.shape[0]
height_max1 = np.tile(tops, (n1,1))		# Produces an n1xnum_hclass array
#height_max1 = np.ones((n1,num_hclass)) * 1e20
print 'Shape of height_max1 = ' + str(height_max1.shape)

sd.init(elevation2, mask2, height_max1, problem_file='snowdrift.nc')

# ================ Upgrid it to the GCM Grid
ZG0 = np.zeros((n1,num_hclass))
ZG0[:] = np.nan
sd.upgrid(ZH0, ZG0)		# 1 = Replace, 0 = Merge


# =============== Rasterize on the GCM Grid
# Rasterize it over same region as ice grid
print '==== Rasterize on the GCM Grid'
grid1 = snowdrift.Grid(overlap_fname, 'grid1')

ZG0_r = np.zeros((raster_x, raster_y))
ZG0_r[:] = np.nan
print 'BEGIN Rasterize'
rast1 = snowdrift.Rasterizer(grid1, x0,x1,raster_x, y0,y1,raster_y)
#snowdrift.rasterize(rast1, ZG0, ZG0_r)
snowdrift.rasterize_hc(rast1, rast2, ZG0, elevation2, mask2, height_max1, ZG0_r)
print 'END Rasterize'
print 'Rasterized to array of shape ', ZG0_r.shape

# ================ Plot it!
print '==== Plot it!'

ax = fig.add_subplot(1,3,curplot)
curplot += 1
init_plot(ax, 'Upscaled', x0,x1,y0,y1)

cax1 = plot_image(ax, ZG0_r, np.array([x0,x1,y0,y1]))
fig.colorbar(cax1)

ax.plot(greenland_xy[0], greenland_xy[1], 'black', alpha=.5)

# ================ Downgrid to the ice grid
print '==== Downgrid to the ice grid, ZG0 -> ZH1'
ZH1 = np.zeros(ice_nx*ice_ny)
ZH1[:] = np.nan
time0 = time.time()
sd.downgrid(ZG0, ZH1, use_snowdrift=1, merge_or_replace=1)
time1 = time.time()
print ZH1[1:200]
print 'Finished with Downgrid, took %f seconds' % (time1-time0,)
ZH1_r = np.zeros((raster_x, raster_y))
ZH1_r[:] = np.nan
print 'BEGIN Rasterize'
snowdrift.rasterize(rast2, ZH1, ZH1_r)
#ZH1_r[ZH1_r < 0] = 4.0			# Debugging
print 'END Rasterize'

ax = fig.add_subplot(1,3,curplot)
curplot += 1
init_plot(ax, 'Downscaled', x0,x1,y0,y1)

cax2 = plot_image(ax, ZH1_r, np.array([x0,x1,y0,y1]))
fig.colorbar(cax2)

ax.plot(greenland_xy[0], greenland_xy[1], 'black', alpha=.5)


# # =======================================================
# # =========== Read and plot Greenland
# # Plot Greenland
# sproj = str(overlap_nc.variables['grid1.info'].getncattr('projection'))
# sllproj = str(overlap_nc.variables['grid1.info'].getncattr('latlon_projection'))
# 
# print 'proj=' + sproj
# print 'llproj=' + sllproj
# 
# proj = pyproj.Proj(sproj)
# llproj = pyproj.Proj(sllproj);
# 
# # Read and plot Greenland Coastline
# lons, lats = read_coastline('data/18969-greenland-coastline.dat')
# xs, ys = pyproj.transform(llproj, proj, lons, lats)


# ================================================================
# Test conservation on ice grid
print 'np.nansum(ZH0) = %f' % np.nansum(ZH0)
print 'np.nansum(ZH1) = %f' % np.nansum(ZH1)


plt.show()
