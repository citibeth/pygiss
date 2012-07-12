import numpy as np
import pyproj
import matplotlib

# Produces (x,y) arrays to plot
# projs = (src-proj, dest-proj)
def ll2xy_latitude(n, lon0, lon1, lat, projs = None) :
	lons = np.zeros(n+1)
	lats = np.zeros(n+1)
	for i in range(0,n) :
		lon = lon0 + (lon1 - lon0) * (float(i) / float(n))
		lons[i] = lon
		lats[i] = lat
	lons[n] = lon1
	lats[n] = lat

	if projs is None :
		print 'projs is None: ' + str(lons)
		return (lons, lats)
	else :
		return pyproj.transform(projs[0], projs[1], lons, lats)


# Produces (x,y) arrays to plot
# projs = (src-proj, dest-proj)
def ll2xy_longitude(n, lon, lat0, lat1, projs = None) :
	lons = np.zeros(n+1)
	lats = np.zeros(n+1)
	for i in range(0,n) :
		lat = lat0 + (lat1 - lat0) * (float(i) / float(n))
		lons[i] = lon
		lats[i] = lat
	lons[n] = lon
	lats[n] = lat1

	if projs is None :
		print 'projs is None: ' + str(lons)
		return (lons, lats)
	else :
		return pyproj.transform(projs[0], projs[1], lons, lats)


# Finds intersection between a vertical line (Cartesian) and a latitude line
# lon0 = (lat, lon0) is to the left of the vertical line at x
# lon1 = (lat, lon1) is to the right of the vertical line at x
# Returns ((x,y), (lon,lat))
def vertical_latitude_intersection(x, lon0, lon1, lat, projs) :

	(x0, y0) = pyproj.transform(projs[0], projs[1], lon0, lat)
	(x1, y1) = pyproj.transform(projs[0], projs[1], lon1, lat)

	while True :
		lon2 = (lon0 + lon1) * .5
		(x2,y2) = pyproj.transform(projs[0], projs[1], lon2, lat)
		# print lon2,x2,y2

		if abs(x2-x) < 1e-5 :
			return ((x,y2), (lon2, lat))

		if x2 < x :
			lon0 = lon2
		else :
			lon1 = lon2



# Finds intersection between a vertical line (Cartesian) and a latitude line
# lon0 = (lat, lon0) is to the left of the vertical line at x
# lon1 = (lat, lon1) is to the right of the vertical line at x
# Returns ((x,y), (lon,lat))
def horizontal_longitude_intersection(y, lon, lat0, lat1, projs) :

	(x0, y0) = pyproj.transform(projs[0], projs[1], lon, lat0)
	(x1, y1) = pyproj.transform(projs[0], projs[1], lon, lat1)

	while True :
		lat2 = (lat0 + lat1) * .5
		(x2,y2) = pyproj.transform(projs[0], projs[1], lon, lat2)
		# print lat2,x2,y2,y

		if abs(y2-y) < 1e-5 :
			return ((x2,y), (lon, lat2))

		if y2 < y :
			lat0 = lat2
		else :
			lat1 = lat2

# Puts graticules on a plot, also sets the axis labels
def plot_graticules(ax, lons, lats, x0,x1,y0,y1, projs, draw_graticules = True) :
	fmt = u'%#d\u00b0'

	plt = ([], [])

	# -------------------------------------
	# Get minimum and maximum lon and lat ranges.
	# This works as long as the map area doesn't do something funny (like contain a pole)

	# Take care of crossing the date line
	lon0 = pyproj.transform(projs[1], projs[0], x0, y0)[0]
	lon1 = pyproj.transform(projs[1], projs[0], x1, y0)[0]
	if lon0 < lon1 :
		lon_offset = 0
	else :
		lon_offset = 360.

	# Look for min and max lat & lon along boundary line.
	# Approximate this by looking at the four corners.
	points = []
	points.append(pyproj.transform(projs[1], projs[0], x0, y0))
	points.append(pyproj.transform(projs[1], projs[0], x1, y0))
	ll = pyproj.transform(projs[1], projs[0], x1, y1)
	points.append((ll[0] + lon_offset, ll[1]))
	ll = pyproj.transform(projs[1], projs[0], x0, y1)
	points.append((ll[0] + lon_offset, ll[1]))
	xlons = map(lambda p: p[0], points)
	xlats = map(lambda p: p[1], points)
	lon_min = min(xlons) - 1
	lon_max = max(xlons) + 1
	lat_min = -90
	lat_max = 90

	print lon_min,lon_max
	print lat_min,lat_max

	# --------------------------------------
	# Plot longitude lines, and also set x-axis labels
	# lons = range(-75,1,10)
	x0s = [] #np.zeros(len(lats))
	x0labels = []
	for i in range(0,len(lons)) :
		# Find where latitude line intersets the upper and lower bounds
		(xy0,ll0) = horizontal_longitude_intersection(y0, lons[i], lat_min, lat_max, projs)
		(xy1,ll1) = horizontal_longitude_intersection(y1, lons[i], lat_min, lat_max, projs)
		if xy0[0] >= x0 and xy0[0] <= x1 :
			x0s.append(xy0[0])
			slon = fmt % lons[i]
			x0labels.append(fmt % lons[i])

		# Plot latitude line --- shortcut, assume straight line on map
		xys = ll2xy_longitude(50, lons[i], ll0[1], ll1[1], projs)
		plt[0].extend(xys[0])
		plt[1].extend(xys[1])
#		plt[0].append(xy0[0])
#		plt[1].append(xy0[1])
#		plt[0].append(xy1[0])
#		plt[1].append(xy1[1])
		plt[0].append(np.nan)
		plt[1].append(np.nan)

	ax.xaxis.set_ticks(x0s)
	ax.xaxis.set_ticklabels(x0labels)
 	# ax.xaxis.set_label('Longitude')

	# --------------------------------------
	# Plot latitude lines, and also set y-axis labels
	# lats = range(60,81,5)
	y0s = [] #np.zeros(len(lats))
	y0labels = []
	for i in range(0,len(lats)) :
		# Find where latitude line intersets the left bound
		(xy0,ll0) = vertical_latitude_intersection(x0, lon_min, lon_max, lats[i], projs)
		(xy1,ll1) = vertical_latitude_intersection(x1, lon_min, lon_max, lats[i], projs)
		if xy0[1] >= y0 and xy0[1] <= y1 :
			y0s.append(xy0[1])
			y0labels.append(str(lats[i]) + u'\u00b0')

		# Plot latitude line
		xys = ll2xy_latitude(50, ll0[0], ll1[0], lats[i], projs)
		plt[0].extend(xys[0])
		plt[1].extend(xys[1])
		plt[0].append(np.nan)
		plt[1].append(np.nan)

	
#		xy = ll2xy_latitude(50, -74, -9, lats[i], projs)

	ax.yaxis.set_ticks(y0s)
	ax.yaxis.set_ticklabels(y0labels)
 	# ax.yaxis.set_label('Latitude')

	if draw_graticules :
		ax.plot(plt[0], plt[1], 'gray', alpha=.9)

	return plt

# -------------------------------------------------------------------------
# def set_cartesian_axes() :
# 	ax.set_xlabel('km')
# 	ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_km))
# 	for tl in ax.xaxis.get_ticklabels() :
# 		# tl.set_fontsize(10)
# 		tl.set_rotation(60)
# 
# 	ax.set_ylabel('x1000 km')
# 	ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_Mm))
# #	for tl in ax.yaxis.get_ticklabels() :
# #		tl.set_rotation(90)
# 
# def format_km(x, pos=None) :
# 	return str(int(x*.001))
# 
# def format_Mm(x, pos=None) :
# 	return '%#0.1f' % (x * .000001)
