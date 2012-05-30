import matplotlib
import numpy as np
import re
import math



# Used to make colormaps with zero in the center but asymmetric range on either side.
# See: http://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
class AsymmetricNormalize(matplotlib.colors.Normalize):	   
	def __init__(self,vmin=None,vmax=None,clip=False):
		matplotlib.colors.Normalize.__init__(self,vmin,vmax,clip)
		#self.linthresh=linthresh

	def __call__(self, value, clip=None):
		if clip is None:
			clip = self.clip

		result, is_scalar = self.process_value(value)

		self.autoscale_None(result)
		vmin, vmax = self.vmin, self.vmax
		if vmin > 0:
			raise ValueError("minvalue must be less than 0")
		if vmax < 0:
			raise ValueError("maxvalue must be more than 0")			
		elif vmin == vmax:
			result.fill(0) # Or should it be all masked? Or 0.5?
		else:
			vmin = float(vmin)
			vmax = float(vmax)
			if clip:
				mask = np.ma.getmask(result)
				result = np.ma.array(np.clip(result.filled(vmax), vmin, vmax),
								  mask=mask)
			# ma division is very slow; we can take a shortcut
			resdat = result.data

			resdat[resdat>0] /= vmax
			resdat[resdat<0] /= -vmin
			resdat=resdat/2.+0.5
			result = np.ma.array(resdat, mask=result.mask, copy=False)

		if is_scalar:
			result = result[0]

		return result

	def inverse(self, value):
		if not self.scaled():
			raise ValueError("Not invertible until scaled")
		vmin, vmax = self.vmin, self.vmax

		if matplotlib.cbook.iterable(value):
			val = np.ma.asarray(value)
			val=2*(val-0.5) 
			val[val>0]*=vmax
			val[val<0]*=-vmin
			return val
		else:
			if val<0.5: 
				return	2*val*(-vmin)
			else:
				return val*vmax

# =================================================================
# Convert HSV <--> RGB
# R, G, B values are [0, 255]. H value is [0, 360]. S, V values are [0, 1].
#Implementation based on pseudo-code from Wikipedia. http://en.wikipedia.org/wiki/HSL_and_HSV
# http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/

def hsv2rgb(h, s, v):
	h = float(h)
	s = float(s)
	v = float(v)
	h60 = h / 60.0
	h60f = math.floor(h60)
	hi = int(h60f) % 6
	f = h60 - h60f
	p = v * (1 - s)
	q = v * (1 - f * s)
	t = v * (1 - (1 - f) * s)
	r, g, b = 0, 0, 0
	if hi == 0: r, g, b = v, t, p
	elif hi == 1: r, g, b = q, v, p
	elif hi == 2: r, g, b = p, v, t
	elif hi == 3: r, g, b = p, q, v
	elif hi == 4: r, g, b = t, p, v
	elif hi == 5: r, g, b = v, p, q
	r, g, b = int(r * 255), int(g * 255), int(b * 255)
	return r, g, b
	
def rgb2hsv(r, g, b):
	r, g, b = r/255.0, g/255.0, b/255.0
	mx = max(r, g, b)
	mn = min(r, g, b)
	df = mx-mn
	if mx == mn:
		h = 0
	elif mx == r:
		h = (60 * ((g-b)/df) + 360) % 360
	elif mx == g:
		h = (60 * ((b-r)/df) + 120) % 360
	elif mx == b:
		h = (60 * ((r-g)/df) + 240) % 360
	if mx == 0:
		s = 0
	else:
		s = df/mx
	v = mx
	return h, s, v
# =============================================================
# Read a .cpt file and construct a colormap from it.
# See:
# http://soliton.vm.bytemark.co.uk/pub/cpt-city/
# http://osdir.com/ml/python.matplotlib.general/2005-01/msg00023.html
# http://assorted-experience.blogspot.com/2007/07/custom-colormaps.html
color_modelRE = re.compile('#\s*COLOR_MODEL\s*=\s*(.*)')
lineRE = re.compile('([-+0123456789\.]\S*)\s+(\S*)\s+(\S*)\s+(\S*)\s+(\S*)\s+(\S*)\s+(\S*)\s+(\S*)')
def read_cpt(fname) :

	# --------- Read the file
	cmapx = []
	use_hsv = False
	for line in open(fname, 'r') :
		match = color_modelRE.match(line)
		if match is not None :
			smodel = match.group(1)
			if smodel == 'HSV' :
				use_hsv = True
		else :
			match = lineRE.match(line)
			if match is not None :
#				print 'matched: ' + line
				for base in [1, 5] :
#					print 'base=' + str(base)
					val = match.group(base)
					c1 = match.group(base+1)
					c2 = match.group(base+2)
					c3 = match.group(base+3)
					if use_hsv :
						rgb = hsv2rgb(int(c1), float(c2), float(c3))
						cmapx.append((float(val), rgb))
					else :
#						print 'val="' + val + '"'
						cmapx.append((float(val), (int(c1), int(c2), int(c3))))

	# ------------ Get the colormap's range
	vmin = cmapx[0][0]
	vmax = cmapx[-1][0]

	# ------------- Create the colormap, converting the form it's in
	vrange = vmax - vmin
	rgbs = ([],[],[])
	for k in range(0,3) :
		rgbs[k].append(( (cmapx[0][0]-vmin) / vrange, cmapx[0][1][k]/255.0, cmapx[0][1][0]/255.0))
	for i in range(1,len(cmapx)-1,2) :
		c0 = cmapx[i]
		c1 = cmapx[i+1]
#		print c0
		for k in range(0,3) :
			rgbs[k].append(( (c0[0]-vmin)/vrange, c0[1][k]/255.0, c1[1][k]/255.0))
	n = len(cmapx)-1
	for k in range(0,3) :
		rgbs[k].append(( (cmapx[n][0]-vmin)/vrange, cmapx[n][1][k]/255.0, cmapx[n][1][0]/255.0))

	cdict = {'red' : rgbs[0], 'green' : rgbs[1], 'blue' : rgbs[2]}
	cmap =  matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
	return (cmap, vmin, vmax)
# -------------------------------------------------------
# Converts .points and .polygons array in netCDF file into
# arrays that can be directly plotted
def points_to_plotlines(polygons, points) :

	npoly = len(polygons)-1		# -1 for sentinel
	npoints = len(points)
	xdata = np.zeros(npoints + npoly * 2)
	ydata = np.zeros(npoints + npoly * 2)

	ipoint_dst = 0
	for ipoly in range(0,npoly) :
		ipoint_src = polygons[ipoly]
		npoints_this = polygons[ipoly+1] - ipoint_src
		xdata[ipoint_dst:ipoint_dst + npoints_this] = points[ipoint_src:ipoint_src + npoints_this,0]
		ydata[ipoint_dst:ipoint_dst + npoints_this] = points[ipoint_src:ipoint_src + npoints_this,1]
		ipoint_dst += npoints_this

		# Repeat the first point in the polygon
		xdata[ipoint_dst] = points[ipoint_src,0]
		ydata[ipoint_dst] = points[ipoint_src,1]
		ipoint_dst += 1

		# Add a NaN
		xdata[ipoint_dst] = np.nan
		ydata[ipoint_dst] = np.nan
		ipoint_dst += 1

	return (xdata, ydata)
# ----------------------------------------------------

