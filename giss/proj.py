import pyproj

def make_projs(sproj) :
	""" Construts a tuple of Proj.4 instances from a Proj.4 string.

	Args:
		sproj (string):
			Proj.4 string for the projection used to map between Cartesian and the globe.
	Returns:	proj[2]
		proj[0] = Forward projection (Spherical --> Cartesian)
		proj[1] = Reverse projection (Cartesian --> Spherical)
	"""

	print 'sproj = %s' % sproj
	sllproj = str(nc.variables[info_name].latlon_projection)
	print 'sllproj = %s' % sllproj
	projs = (pyproj.Proj(sllproj), pyproj.Proj(sproj))	# src & destination projection

	return projs
