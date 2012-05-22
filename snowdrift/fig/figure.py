# Common stuff for Snowdrift-related figures
#

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

X=3.6	# inches
colorbarX=1
Y=6.1
DPI=300

# ---------------------------------------------------------------------
def read_greenland_coastline(data_root, projs) :

	# Read and plot Greenland Coastline
	lons, lats = giss.io.noaa.read_coastline(os.path.join(data_root,
		'ngdc.noaa.gov/coastline/18969-greenland-coastline.dat'), 10)
	greenland_xy = pyproj.transform(projs[0], projs[1], lons, lats)

	# Decide what to remove, to leave (basically) just greenland
	#greenland_x = np.array(greenland_xy[0])
	clip_out = (greenland_xy[0] < -800*km) | (greenland_xy[0] > 681*km)
	greenland_xy[0][clip_out] = np.nan
	greenland_xy[1][clip_out] = np.nan

	return greenland_xy
# ------------------------------------------------------------------

def init_figure(overlap_fname) :
	global overlap_nc
	global x0,x1,y0,y1,ice_nx,ice_ny,raster_x,raster_y,fig,curplot,km
	global projs, greenland_xy
	global n1, n2

	km=1000.0

	# ============= Read info from netCDF file
	overlap_nc = netCDF4.Dataset(overlap_fname, 'r')

	grid1_var = overlap_nc.variables['grid1.info']
	n1 = grid1_var.__dict__['max_index'] - grid1_var.__dict__['index_base'] + 1
	grid2_var = overlap_nc.variables['grid2.info']
	n2 = grid2_var.__dict__['max_index'] - grid2_var.__dict__['index_base'] + 1


	xb = np.array(overlap_nc.variables['grid2.x_boundaries'])
	yb = np.array(overlap_nc.variables['grid2.y_boundaries'])
	x0 = xb[0]
	x1 = xb[-1]
	y0 = yb[0]
	y1 = yb[-1]
	ice_nx = xb.shape[0]-1
	ice_ny = yb.shape[0]-1

	raster_x = ice_nx/3
	raster_y = ice_ny/3

	curplot = 1


	# =========== Read Greenland
	# Plot Greenland
	sproj = str(overlap_nc.variables['grid1.info'].getncattr('projection'))
	sllproj = str(overlap_nc.variables['grid1.info'].getncattr('latlon_projection'))

	print 'proj=' + sproj
	print 'llproj=' + sllproj

	projs = (pyproj.Proj(sllproj), pyproj.Proj(sproj))	# src & destination projection
	greenland_xy = read_greenland_coastline(os.path.join(sys.path[0], 'data'), projs)
		


def init_plot(ax, title, x0,x1,y0,y1) :
	ax.set_title(title)

	giss.maputil.plot_graticules(ax, range(-75,1,10), range(60, 81, 5), x0,x1,y0,y1, projs)
	ax.set_xlim((x0, x1))
	ax.set_ylim((y0, y1))



def get_landmask(searise_nc) :
# landcover:ice_sheet = 4 ;
# landcover:land = 2 ;
# landcover:local_ice_caps_not_connected_to_the_ice_sheet = 3 ;
# landcover:long_name = "Land Cover" ;
# landcover:no_data = 0 ;
# landcover:ocean = 1 ;
# landcover:standard_name = "land_cover" ;
	mask2 = np.array(searise_nc.variables['landcover'], dtype=np.int32).flatten('C')
	mask2 = np.where(mask2==4,np.int32(1),np.int32(0))
	return mask2
