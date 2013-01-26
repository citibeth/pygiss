import pyproj
import unicodedata

def make_projs(sproj) :
	""" Construts a tuple of Proj.4 instances from a Proj.4 string.

	Args:
		sproj (string):
			Proj.4 string for the projection used to map between Cartesian and the globe.
	Returns:	proj[2]
		proj[0] = Lon/Lat
		proj[1] = X/Y
	"""

#	sproj = u'+proj=stere +lon_0=-39 +lat_0=90 +lat_ts=71.0 +ellps=WGS84'
	sproj = unicodedata.normalize('NFKD', sproj).encode('ascii','ignore')
	print 'sproj = "%s"' % sproj

	xyproj = pyproj.Proj(sproj)
	llproj = pyproj.Proj(sproj)
	llproj.to_latlong()

	return (llproj, xyproj)

	return projs
