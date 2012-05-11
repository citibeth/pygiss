# Set up a sample function on the ice grid, upgrids to the GCM grid,
# and then plot it (rasterized) on the GCM grid
# Eg: python snowdrift_test3.py xy_overlap.nc

import matplotlib.pyplot as plt
import numpy as np
import snowdrift
import netCDF4
import sys
import pyproj
import time

import array
import re
import collections

# Image plot
#Acceptable values are None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
def plot_image(ax, Z, extent) :
	Zt = Z.transpose()
	Zmasked = np.ma.array(Zt, mask=np.isnan(Zt))

	return ax.imshow(Zmasked, origin='lower',
		interpolation='bilinear',
		extent=np.array([x0,x1,y0,y1])/km, vmin=0,vmax=105)

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

def sum_nonan(vec) :
	v = vec
	v[np.isnan(vec)] = 0
	return sum(v)

# -------------------------------------------------------------------
# Reads info we need about each grid to test conservation
def read_grid_info(nc, gname) :
	ret = collections.namedtuple('Grid', 'index_base max_index total_coverage proj_area native_area')
	gvar = nc.variables[gname + '.info']
	ret.index_base = gvar.__dict__['index_base']
	ret.max_index = gvar.__dict__['max_index']
	ret.total_coverage = np.zeros(ret.max_index - ret.index_base + 1)
	ret.proj_area = np.zeros(ret.max_index - ret.index_base + 1)
	ret.native_area = np.zeros(ret.max_index - ret.index_base + 1)

	realized = np.array(nc.variables[gname + '.realized_cells'])
	proj_area = np.array(nc.variables[gname + '.proj_area'])
	native_area = np.array(nc.variables[gname + '.native_area'])
	for i in range(0, realized.shape[0]) :
		ret.proj_area[realized[i] - ret.index_base] = proj_area[i]
		ret.native_area[realized[i] - ret.index_base] = native_area[i]

	return ret

# Reads info we need about the overlap matrix, to test conservation
def read_overlap_info(nc) :
	print 'Reading Grid1'
	grid1 = read_grid_info(nc, 'grid1')
	print 'Reading Grid2'
	grid2 = read_grid_info(nc, 'grid2')

	print 'Reading Overlap Info'
	cells = np.array(nc.variables['overlap.index'])
	area = np.array(nc.variables['overlap.val'])

	print 'Processing Overlap Info'
	for i in range(0,area.shape[0]) :
		grid1.total_coverage[cells[i,0] - grid1.index_base] += area[i]
		grid2.total_coverage[cells[i,1] - grid2.index_base] += area[i]

	return (grid1, grid2)
# -------------------------------------------------------------------

km=1000.0

overlap_fname = sys.argv[1]


# ============= Read info from netCDF file
nc = netCDF4.Dataset(overlap_fname, 'r')
xb = np.array(nc.variables['grid2.x_boundaries'])
yb = np.array(nc.variables['grid2.y_boundaries'])
x0 = xb[0]
x1 = xb[-1]
y0 = yb[0]
y1 = yb[-1]



# # =========== Read Greenland
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
# greenland_xy = pyproj.transform(llproj, proj, lons, lats)

# =============== Make a hi-res function on the ice grid
ice_nx = xb.shape[0]-1
ice_ny = yb.shape[0]-1
ZH0 = np.zeros(ice_nx * ice_ny)
for ix in range(0,ice_nx) :
	for iy in range(0,ice_ny) :
#		ZH0[iy * ice_nx + ix] = (ix+1) * 1.8
		ZH0[iy * ice_nx + ix] = (ix+1) * 1.8 + (iy+1)
#		ZH0[iy * ice_nx + ix] = 1

print 'ZH0[0] = %f' % ZH0[0]

# ============= Set up Snowdrift data structures
grid1_var = nc.variables['grid1.info']
n1 = grid1_var.__dict__['max_index'] - grid1_var.__dict__['index_base'] + 1
grid2_var = nc.variables['grid2.info']
n2 = grid2_var.__dict__['max_index'] - grid2_var.__dict__['index_base'] + 1

sd = snowdrift.Snowdrift(overlap_fname)
nheight_class = 1

elevation2 = np.ones((n2,))
mask2 = np.ones((n2,),'i')
height_max1 = np.ones((n1,nheight_class)) * 1e20
print 'Shape of height_max1 = ' + str(height_max1.shape)

sd.init(elevation2, mask2, height_max1)

# ================ Upgrid it to the GCM Grid
ZG0 = np.zeros((n1, nheight_class))
sd.upgrid(ZH0, ZG0)		# 1 = Replace, 0 = Merge

print 'ZG0[0] = %f' % ZG0[0]


# =============== Rasterize on the GCM Grid
# Rasterize it over same region as ice grid
raster_x = 100
raster_y = 200

grid1 = snowdrift.Grid(overlap_fname, 'grid1')
rast1 = snowdrift.Rasterizer(grid1, x0,x1,raster_x, y0,y1,raster_y)

ZG0_r = np.zeros((raster_x, raster_y))
ZG0_r[:] = np.nan
print 'BEGIN Rasterize'
snowdrift.rasterize(rast1, ZG0, ZG0_r)
print 'END Rasterize'
print 'Rasterized to array of shape ', ZG0_r.shape

# ================ Plot it!
fig = plt.figure()

ax = fig.add_subplot(1,2,1)
ax.set_title('GCM')
ax.set_xlabel('km')
ax.set_ylabel('km')

cax1 = plot_image(ax, ZG0_r, np.array([x0,x1,y0,y1])/km)
fig.colorbar(cax1)

# ax.plot(greenland_xy[0]/km, greenland_xy[1]/km, 'b', alpha=.7)

# ================ Downgrid to the ice grid
ZH1 = np.zeros(ice_nx*ice_ny)
time0 = time.time()
sd.downgrid(ZG0, ZH1, use_snowdrift=1)
time1 = time.time()
print ZH1[1:200]
print 'Finished with Downgrid, took %f seconds' % (time1-time0,)
grid2 = snowdrift.Grid(overlap_fname, 'grid2')
rast2 = snowdrift.Rasterizer(grid2, x0,x1,raster_x, y0,y1,raster_y)
ZH1_r = np.zeros((raster_x, raster_y))
ZH1_r[:] = np.nan
print 'BEGIN Rasterize'
snowdrift.rasterize(rast2, ZH1, ZH1_r)
print 'END Rasterize'

ax = fig.add_subplot(1,2,2)
ax.set_title('Ice')
ax.set_xlabel('km')
ax.set_ylabel('km')

cax2 = plot_image(ax, ZH1_r, np.array([x0,x1,y0,y1])/km)
fig.colorbar(cax2)

#ax.plot(greenland_xy[0]/km, greenland_xy[1]/km, 'b', alpha=.7)


# ========================================================
# Test conservation (slow in Python)
grid1, grid2 = read_overlap_info(nc)
ZH0_sum = sum(ZH0 * grid2.native_area * (grid2.total_coverage / grid2.proj_area))
ZG0_sum = sum_nonan(ZG0 * grid1.native_area * (grid1.total_coverage / grid1.proj_area))
ZH1_sum = sum(ZH1 * grid2.native_area * (grid2.total_coverage / grid2.proj_area))

print 'Conservation: ', ZH0_sum, (ZG0_sum - ZH0_sum)/ZH0_sum, (ZH1_sum - ZH0_sum)/ZH0_sum

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
