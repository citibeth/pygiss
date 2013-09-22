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

import matplotlib
import numpy as np
import re
import math
import giss.util
import StringIO
import os

# General utilities for makng plots and maps

# --------------------------------------------------------------
class AsymmetricNormalize(matplotlib.colors.Normalize):
	"""Used to make colormaps with zero in the center but asymmetric range on either side.
	Pass as the 'norm' parameter to plotting functions, (eg: pcolormesh())

	This is meant for use in conjunction with symmetric colormaps (eg,
	ones that have a "neutral" color in the middle.  Eg:
		cmap = giss.plot.cpt('giss-cpt/BlRe.cpt')

	Usage (Adapted from the matplotlib documentation):
		class AsymmetricNormalize(vmin=None, vmax=None, clip=False)

	    Normalize a given value to the 0-1 range

	    If vmin or vmax is not given, they are taken from the input's
	    minimum and maximum value respectively. If clip is True and
	    the given value falls outside the range, the returned value
	    will be 0 or 1, whichever is closer. Returns 0 if:
		    vmin==vmax

	    Works with scalars or arrays, including masked arrays. If clip
	    is True, masked values are set to 1; otherwise they remain
	    masked. Clipping silently defeats the purpose of setting the
	    over, under, and masked colors in the colormap, so it is
	    likely to lead to surprises; therefore the default is clip =
	    False.

	See:
		For more information on constructor parameters:
			matplotlib.colors.Normalize (superclass)

		http://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
	"""

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
def hsv2rgb(h, s, v):
	"""Convert HSV colors to RGB.

	Args:
		h (scalar int in range [0,255])
		s,v (scalar double in range [0,1])

	Returns:
		r, g b (scalar int in range [0,255]

	See:
		http://en.wikipedia.org/wiki/HSL_and_HSV
		http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/
	"""

	# Implementation based on pseudo-code from Wikipedia.

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
	
# ---------------------------------------------------------------
def rgb2hsv(r, g, b):
	"""Convert RGB colors to HSV.

	Args:
		r, g b (scalar int in range [0,255]

	Returns:
		h (scalar int in range [0,255])
		s,v (scalar double in range [0,1])

	See:
		http://en.wikipedia.org/wiki/HSL_and_HSV
		http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/
	"""
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
def _cpt_path() :
	search_path = []

	# Internal library serach path.  This is first to give our library
	# predictable behavior for things in the giss-cpt/ directory.  If
	# you want to use a different cmap, pass it in directly.
	search_path.append(os.path.split(os.path.dirname(__file__))[0])

	# Look for cpt-city in standard locations
	if 'CPT_PATH' in os.environ :
		search_path.extend(os.environ['CPT_PATH'].split(os.pathsep))
	elif 'DATA_PATH' in os.environ :
		# No CPT_PATH: append '/cpt-city' to everything in DATA_PATH
		search_path.extend([
			os.path.join(xx, 'cpt-city')
			for xx in os.environ['DATA_PATH'].split(os.pathsep)
		])
	else :
		# Use default location for DATA_PATH
		search_path.append(os.path.join(os.environ['HOME'], 'data', 'cpt-city'))

	# Search current directory, for user.  This is last because the
	# user shouldn't be ovverriding any of the standard locations.
	search_path.append('.')

	return search_path

# Pre-compute the search path for cpt files
_CPT_PATH = _cpt_path()
# -------------------------------------------------------
def cpt(cpt_name, cpt_path = _CPT_PATH, **kwargs) :
	"""Reads a color map directly from cpt-city directory

	cpt files are found in the following search paths:
		1. Internal library directory (inside this library)
			cpt files generally start with giss-cpt/
		2. cpt-city directory:
			If CPT_PATH variable exists:
				$(CPT_PATH)
			else:
				$(DATA_PATH)/cpt-city
			else:
				$(HOME)/data/cpt-city
		3. User directory
			.
	If you wish any other locations, set the CPT_PATH env variable.

	Args:
		cpt_name (string):
			Name of cpt file (including .cpt on end)
		cpt_path[] (string):
			List of directories to search for cpt file
		kwargs: Passed along to parse_cpt()
			reverse (bool): If true, reverse the colors on this palette

	Returns struct with:
		.cmap (matplotlib.colors.LinearSegmentedColormap):
			The color map
		.vmin:
			Minimum value specified in the colormap
		.vmax:
			Maximum value specified in the colormap
		Some colormaps come with built-in minimum and maximum values:
		for example, an elevation colormap may map specific elevations
		to specific colors.  vmin and vmax are provided in case you
		wish to use these min and max values in your plot.  They may
		also be ignored if not needed.

	See:
		read_cpt()
		parse_cpt()
		http://soliton.vm.bytemark.co.uk/pub/cpt-city/pkg/
	"""

	fname = giss.util.search_file(cpt_name, cpt_path)
	if fname is None :
		raise Exception('Cannot find color palette %s in search path %s' %
			(cpt_name, os.pathsep.join(cpt_path)))
	return read_cpt(fname, **kwargs)
# ---------------------------------------------------------------
def read_cpt(fname, **kwargs) :
	"""Reads a color map from a .cpt file

	NOTE: This returns a struct, not just the colrmap itself.

	Args:
		fname (string):
			Name of cpt file (includin .cpt at the end)

	Returns:
		(See cpt())
	"""

	fin = open(fname, 'r')
	ret = parse_cpt(fin.read(), **kwargs)
	fin.close()
	return ret
# -----------------------------------------------------------------
_color_modelRE = re.compile('#\s*COLOR_MODEL\s*=\s*(.*)')
_lineRE = re.compile('\s*([-+0123456789\.]\S*)\s+(\S*)\s+(\S*)\s+(\S*)\s+(\S*)\s+(\S*)\s+(\S*)\s+(\S*)')
def parse_cpt(cpt_str, reverse=False) :
	"""Parses an already-read cpt file.

	Args:
		cpt_str (string):
			Contents of a cpt file
		reverse (bool):
			If true, reverse the colors on this palette

	Returns:
		(see cpt())

	See:
		http://soliton.vm.bytemark.co.uk/pub/cpt-city/
		http://osdir.com/ml/python.matplotlib.general/2005-01/msg00023.html
		http://assorted-experience.blogspot.com/2007/07/custom-colormaps.html

	"""

	# --------- Read the file
	cmap_vals = []
	cmap_rgbs = []
	use_hsv = False
	for line in StringIO.StringIO(cpt_str) :
		match = _color_modelRE.match(line)
		if match is not None :
			smodel = match.group(1)
			if smodel == 'HSV' :
				use_hsv = True
		else :
			match = _lineRE.match(line)
			if match is not None :
				for base in [1, 5] :
					val = match.group(base)
					c1 = match.group(base+1)
					c2 = match.group(base+2)
					c3 = match.group(base+3)
#					print 'base=' + str(base) + ', tuple=',val,c1,c2,c3, use_hsv
					if use_hsv :
						rgb = hsv2rgb(int(c1), float(c2), float(c3))
						cmap_vals.append(float(val))
						cmap_rgbs.append(rgb)
					else :
						cmap_vals.append(float(val))
						cmap_rgbs.append((int(c1), int(c2), int(c3)))

	# Assemble into cmapx
	if reverse : cmap_rgbs.reverse()
	cmapx = zip(cmap_vals, cmap_rgbs)

	# ------------ Get the colormap's range
	vmin = cmapx[0][0]
	vmax = cmapx[-1][0]

	# ------------- Create the colormap, converting the form it's in
	vrange = vmax - vmin
	rgbs = ([],[],[])

	c0 = cmapx[0]
	cur_val = c0[0]
	cur_rgb = c0[1]
	for k in range(0,3) :
		rgbs[k].append(( (cur_val-vmin) / vrange, cur_rgb[k]/255.0, cur_rgb[k]/255.0))

	for i in range(1,len(cmapx)-1,2) :
		cur_rgb = cmapx[i][1]
		next_rgb = cmapx[i+1][1]
		cur_val = cmapx[i][0]	# also equals to next_val
		for k in range(0,3) :
			rgbs[k].append(( (cur_val-vmin)/vrange, cur_rgb[k]/255.0, next_rgb[k]/255.0))

	c0 = cmapx[-1]
	cur_val = c0[0]
	cur_rgb = c0[1]
	for k in range(0,3) :
		rgbs[k].append(( (cur_val-vmin)/vrange, cur_rgb[k]/255.0, cur_rgb[k]/255.0))

	cdict = {'red' : rgbs[0], 'green' : rgbs[1], 'blue' : rgbs[2]}
	cmap =  matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

	return giss.util.Struct({'cmap' : cmap, 'vmin' : vmin, 'vmax' : vmax})
# -------------------------------------------------------
def points_to_plotlines(polygons, points) :

	"""Converts a set of polygon (grid cell) definitions to a set of
	line segments that can be plotted.

	Args:
		points[nvertices, 2] (np.array, dtype=double):
			A set of (x, y) or (lon, lat) points used to define the
			polygons vertices.  Points are "in order"

		polygons[npolygons+1] (np.array, dtype=int):
			Index of the start of each polygon in the points[] array.
			(Index base = 0)
			Polygon i has vertices points[polygons[i]:polygons[i+1], :]

		This is the format as found in grid and overlap matrix netCDF files.

	Returns:
		xdata, ydata
			Arrays of (x,y) points suitable for use with plot() to
			plot the polygon boundaries.
	"""

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
