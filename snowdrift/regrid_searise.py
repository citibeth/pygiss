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

import array
import re

# Image plot
#Acceptable values are None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
def plot_image(ax, Z, extent) :
	Zt = Z.transpose()
	Zmasked = np.ma.array(Zt, mask=np.isnan(Zt))

	return ax.imshow(Zmasked, origin='lower',
		interpolation='bilinear',
		#cmap='afmhot',
		cmap='spectral',
		extent=np.array([x0,x1,y0,y1])/km, vmin=0,vmax=4.5)

# Contour plot
#	cax = ax.contour(ZG0_r.transpose(), interpolation='nearest', origin='lower')


lineRE=re.compile('(.*?)\s+(.*)')
def read_coastline(fname) :
	nlines = 0
	xdata = array.array('d')
	ydata = array.array('d')
	for line in file(fname) :
#		if (nlines % 10000 == 0) :
#			print 'nlines = %d' % (nlines,)
		if (nlines % 10 == 0 or line[0:3] == 'nan') :
			match = lineRE.match(line)
			lon = float(match.group(1))
			lat = float(match.group(2))

			xdata.append(lon)
			ydata.append(lat)
		nlines = nlines + 1


	return (np.array(xdata),np.array(ydata))


km=1000.0

overlap_fname = sys.argv[1]
field_name = 'presprcp'

fig = plt.figure(figsize=(1000/80,400/80))	# Size of figure (inches)
curplot = 1

# ============= Read info from netCDF file
nc = netCDF4.Dataset(overlap_fname, 'r')
xb = np.array(nc.variables['grid2.x_boundaries'])
yb = np.array(nc.variables['grid2.y_boundaries'])
x0 = xb[0]
x1 = xb[-1]
y0 = yb[0]
y1 = yb[-1]
ice_nx = xb.shape[0]-1
ice_ny = yb.shape[0]-1


raster_x = 400
raster_y = 800


# =========== Read Greenland
# Plot Greenland
sproj = str(nc.variables['grid1.info'].getncattr('projection'))
sllproj = str(nc.variables['grid1.info'].getncattr('latlon_projection'))

print 'proj=' + sproj
print 'llproj=' + sllproj

proj = pyproj.Proj(sproj)
llproj = pyproj.Proj(sllproj);

# Read and plot Greenland Coastline
lons, lats = read_coastline('data/18969-greenland-coastline.dat')
greenland_xy = pyproj.transform(llproj, proj, lons, lats)

# Decide what to remove
#greenland_x = np.array(greenland_xy[0])
clip_out = (greenland_xy[0] < -800*km) | (greenland_xy[0] > 681*km)
greenland_xy[0][clip_out] = np.nan
#greenland_y = np.array(greenland_xy[1])
greenland_xy[1][clip_out] = np.nan

# =============== Read a hi-res function on the ice grid
searise_nc = netCDF4.Dataset('Greenland_5km_v1.1.nc')
#ZH0 = np.ndarray.flatten(np.array(searise_nc.variables[field_name], dtype='d'))
ZH0 = np.array(searise_nc.variables[field_name], dtype='d').flatten('C')
print 'ZH0 has %d elements, vs (%d x %d) = %d' % (ZH0.shape[0], ice_nx, ice_ny, ice_nx * ice_ny)

ZH0_r = np.zeros((raster_x, raster_y))
ZH0_r[:] = np.nan
grid2 = snowdrift.Grid(overlap_fname, 'grid2')
grid2.rasterize(x0,x1,raster_x, y0,y1,raster_y, ZH0, ZH0_r)

ax = fig.add_subplot(1,3,curplot)
ax.set_xlim((x0/km, x1/km))
ax.set_ylim((y0/km, y1/km))
ax.set_xlabel('km')
ax.set_ylabel('km')
curplot += 1
ax.set_title(field_name)

cax2 = plot_image(ax, ZH0_r, np.array([x0,x1,y0,y1])/km)
fig.colorbar(cax2)
ax.plot(greenland_xy[0]/km, greenland_xy[1]/km, 'black', alpha=.5)

# ================ Upgrid it to the GCM Grid
sd = snowdrift.Snowdrift(overlap_fname)
grid1_var = nc.variables['grid1.info']
ZG0_len = grid1_var.__dict__['max_index'] - grid1_var.__dict__['index_base'] + 1
ZG0 = np.zeros(ZG0_len)
sd.upgrid(0, ZH0, ZG0)		# 0 = Replace, 1 = Merge


# =============== Rasterize on the GCM Grid
# Rasterize it over same region as ice grid
grid1 = snowdrift.Grid(overlap_fname, 'grid1')

ZG0_r = np.zeros((raster_x, raster_y))
ZG0_r[:] = np.nan
print 'BEGIN Rasterize'
grid1.rasterize(x0,x1,raster_x, y0,y1,raster_y, ZG0, ZG0_r)
print 'END Rasterize'
print 'Rasterized to array of shape ', ZG0_r.shape

# ================ Plot it!

ax = fig.add_subplot(1,3,curplot)
ax.set_xlim((x0/km, x1/km))
ax.set_ylim((y0/km, y1/km))
curplot += 1
ax.set_title('Upscaled')
ax.set_xlabel('km')
ax.set_ylabel('km')

cax1 = plot_image(ax, ZG0_r, np.array([x0,x1,y0,y1])/km)
fig.colorbar(cax1)

ax.plot(greenland_xy[0]/km, greenland_xy[1]/km, 'black', alpha=.5)

# ================ Downgrid to the ice grid
ZH1 = np.zeros(ice_nx*ice_ny)
#sd.downgrid(0, ZG0, ZH1)		# Simple downgridding
time0 = time.time()
sd.downgrid_snowdrift(ZG0, ZH1)
time1 = time.time()
print ZH1[1:200]
print 'Finished with Downgrid, took %f seconds' % (time1-time0,)
ZH1_r = np.zeros((raster_x, raster_y))
ZH1_r[:] = np.nan
print 'BEGIN Rasterize'
grid2.rasterize(x0,x1,raster_x, y0,y1,raster_y, ZH1, ZH1_r)
print 'END Rasterize'

ax = fig.add_subplot(1,3,curplot)
ax.set_xlim((x0/km, x1/km))
ax.set_ylim((y0/km, y1/km))
curplot += 1
ax.set_title('Downscaled')
ax.set_xlabel('km')
ax.set_ylabel('km')

cax2 = plot_image(ax, ZH1_r, np.array([x0,x1,y0,y1])/km)
fig.colorbar(cax2)

ax.plot(greenland_xy[0]/km, greenland_xy[1]/km, 'black', alpha=.5)


# # =======================================================
# # =========== Read and plot Greenland
# # Plot Greenland
# sproj = str(nc.variables['grid1.info'].getncattr('projection'))
# sllproj = str(nc.variables['grid1.info'].getncattr('latlon_projection'))
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





plt.show()
