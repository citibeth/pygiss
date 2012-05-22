import re
import numpy as np
import array

# Read "Matlab" format files from NOAA Coastline Extractor.
# See: http://www.ngdc.noaa.gov/mgg/coast/

lineRE=re.compile('(.*?)\s+(.*)')
def read_coastline(fname, take_every=10) :
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
