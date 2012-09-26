import mpl_toolkits.basemap

# Some pre-defined maps

# The entire globe --- good for modelE output
def global_robin() :
	# create Basemap instance for Robinson projection.
	# coastlines not used, so resolution set to None to skip
	# continent processing (this speeds things up a bit)
	return mpl_toolkits.basemap.Basemap(projection='robin',lon_0=0,resolution='l')

# North polar projection
def north_laea() :
	# create Basemap instance for Robinson projection.
	# coastlines not used, so resolution set to None to skip
	# continent processing (this speeds things up a bit)
	km = 1000.0
	return mpl_toolkits.basemap.Basemap(
		width=7000 * km,height=7000 * km,
		resolution='l',projection='laea',\
		lat_ts=72,lat_0=90,lon_0=-90.)

# South polar projection
def south_laea() :
	# create Basemap instance for Robinson projection.
	# coastlines not used, so resolution set to None to skip
	# continent processing (this speeds things up a bit)
	km = 1000.0
	return mpl_toolkits.basemap.Basemap(
		width=7000 * km,height=7000 * km,
		resolution='l',projection='laea',\
		lat_ts=72,lat_0=-90,lon_0=0.)

# Just greenland
def greenland_laea() :
	# create Basemap instance for Robinson projection.
	# coastlines not used, so resolution set to None to skip
	# continent processing (this speeds things up a bit)
	km = 1000.0
	return mpl_toolkits.basemap.Basemap(
		#width=7000 * km,height=7000 * km,
		resolution='l',projection='laea',\
		lat_ts=72,lat_0=90,lon_0=-40.,
		llcrnrlon=-54., llcrnrlat=58.,
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
