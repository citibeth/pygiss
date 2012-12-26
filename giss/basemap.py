import mpl_toolkits.basemap
import numpy as np
import re
import array

"""Some pre-defined map projections, commonly used at GISS"""

# The entire globe --- good for modelE output
def global_map(ax=None) :
	"""Args:
		ax (axes):
			Set default axes instance (see mpltoolkits.basemap.Basemap)
				
	Returns: Basemap instance
		With low-resolution coastlines
	See:
		mpl_toolkits.basemap.Basemap"""

	return mpl_toolkits.basemap.Basemap(ax=ax, projection='kav7',lon_0=0,resolution='l')
#	return mpl_toolkits.basemap.Basemap(ax=ax, projection='robin',lon_0=0,resolution='l')



def north_laea(ax=None) :
	"""Args:
		ax (axes):
			Set default axes instance (see mpltoolkits.basemap.Basemap)
				
	Returns:	North polar, Lambert Equal Area Projection.
		With low-resolution coastlines.
		Distortion minimized at 72 north.
		Map is 7000x7000 km

	See:
		mpl_toolkits.basemap.Basemap
	"""
	km = 1000.0
	return mpl_toolkits.basemap.Basemap(ax=ax,
		width=7000 * km,height=7000 * km,
		resolution='l',projection='laea',\
		lat_ts=72,lat_0=90,lon_0=-90.)

# South polar projection
def south_laea(ax=None) :
	"""Args:
		ax (axes):
			Set default axes instance (see mpltoolkits.basemap.Basemap)
				
	Returns:	South polar, Lambert Equal Area Projection.
		With low-resolution coastlines.
		Distortion minimized at 72 north.
		Map is 7000x7000 km

	See:
		mpl_toolkits.basemap.Basemap"""

	km = 1000.0
	return mpl_toolkits.basemap.Basemap(ax=ax,
		width=7000 * km,height=7000 * km,
		resolution='l',projection='laea',\
		lat_ts=-72,lat_0=-90,lon_0=0.)

def greenland_laea(ax=None) :
	"""Args:
		ax (axes):
			Set default axes instance (see mpltoolkits.basemap.Basemap)
				
	Returns:	Map for plotting Greenland.
		With low-resolution coastlines.

	See:
		mpl_toolkits.basemap.Basemap"""
	km = 1000.0
	return mpl_toolkits.basemap.Basemap(ax=ax,
		resolution='l',projection='laea',\
		lat_ts=72,lat_0=90,lon_0=-40.,
		llcrnrlon=-54., llcrnrlat=58.,
#		width=1634940, height=2622574)
#		width=1634940, height=2622574)
		urcrnrlon=5., urcrnrlat=80.)


#llcrnrlon 	longitude of lower left hand corner of the desired map domain (degrees).
#llcrnrlat 	latitude of lower left hand corner of the desired map domain (degrees).
#urcrnrlon 	longitude of upper right hand corner of the desired map domain (degrees).
#urcrnrlat 	latitude of upper right hand corner of the desired map domain (degrees).
#
#or these
#width 	width of desired map domain in projection coordinates (meters).
#height 	height of desired map domain in projection coordinates (meters).
#lon_0 	center of desired map domain (in degrees).
#lat_0 	center of desired map domain (in degrees).

# ---------------------------------------------------------------
def drawcoastline(mymap, lons, lats, **kwargs) :
	"""Plots a custom coastline.  This plots simple lines, not
	ArcInfo-style SHAPE files.

	Args:
		lons: Longitude coordinates for line segments (degrees E)
		lats: Latitude coordinates for line segments (degrees N)

	Type Info:
		len(lons) == len(lats)
		A NaN in lons and lats signifies a new line segment.

	See:
		giss.noaa.drawcoastline_file()
	"""

	# Project onto the map
	x, y = mymap(lons, lats)

	# BUG workaround: Basemap projects our NaN's to 1e30.
	x[x==1e30] = np.nan
	y[y==1e30] = np.nan

	# Plot projected line segments.
	mymap.plot(x, y, **kwargs)


# Read "Matlab" format files from NOAA Coastline Extractor.
# See: http://www.ngdc.noaa.gov/mgg/coast/

lineRE=re.compile('(.*?)\s+(.*)')
def read_coastline(fname, take_every=1) :
	nlines = 0
	xdata = array.array('d')
	ydata = array.array('d')
	for line in file(fname) :
#		if (nlines % 10000 == 0) :
#			print 'nlines = %d' % (nlines,)
		if (nlines % take_every == 0 or line[0:3] == 'nan') :
			match = lineRE.match(line)
			lon = float(match.group(1))
			lat = float(match.group(2))

			xdata.append(lon)
			ydata.append(lat)
		nlines = nlines + 1


	return (np.array(xdata),np.array(ydata))

def drawcoastline_file(mymap, fname, **kwargs) :
	"""Reads and plots a coastline file.
	See:
		giss.basemap.drawcoastline()
		giss.basemap.read_coastline()
	"""

	lons, lats = read_coastline(fname, take_every=1)
	return drawcoastline(mymap, lons, lats, **kwargs)
