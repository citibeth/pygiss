import pyproj

def make_projs(sproj) :
	""" Construts a tuple of Proj.4 instances from a Proj.4 string.

	Args:
		sproj (string):
			Proj.4 string for the projection used to map between Cartesian and the globe.
	Returns:	proj[2]
		proj[0] = Lon/Lat
		proj[1] = X/Y
	"""

	print 'sproj = %s' % sproj
	xyproj = pyproj.Proj(sproj)
	llproj = pyproj.Proj(sproj)
	llproj.to_latlong()

	return (llproj, xyproj)

	return projs
