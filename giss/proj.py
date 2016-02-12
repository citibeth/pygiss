# pyGISS: GISS Python Library
# Copyright (c) 2013 by Robert Fischer
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
#	print('sproj = ', type(sproj), sproj)
#	if type(sproj) == unicode :
#		sproj = unicodedata.normalize('NFKD', sproj).encode('ascii','ignore')
#	print 'sproj = "%s"' % sproj

	xyproj = pyproj.Proj(sproj)
	llproj = xyproj.to_latlong()

	return (llproj, xyproj)
