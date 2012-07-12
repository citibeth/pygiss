import sys
import argparse
import io
import struct
import numpy
import re
from giss.util import *


# =========================================================================
# Reader for the GISS data file format (giss.io) used in ModelE

# Parse the type specification
datatypes = {'real' : numpy.dtype('>f')}

# These shapes are in (jm,im).  I decided to use C-style
# row major indexing because numPy uses zero-based indexing
# In this way, numPy will index exactly the same as C netCDF
# utilities, and we avoid introducing a THIRD indexing convention.
shapes = {'8x10' : (24,36)}

def dict2re_frag(dict) :
	ret = []
	for sdim in dict.iterkeys() :
		ret.append(sdim)
		ret.append('|')
	return ''.join(ret[0:-1])

# Regular expression to parse the title strings
#     match.group(1) = Name of variable
#     match.group(2) = Data type
#     match.group(3) = Dimension string
titleRE = re.compile(''.join([ \
	'(.*?)[ :].*?(', \
	dict2re_frag(datatypes), \
	') (', \
	dict2re_frag(shapes), \
	')']))
title2RE = re.compile('(.*?)[ :].*')

class Record :
	def __init__(self, var, data, comment) :
		self.var = var
		self.data = data
		self.comment = comment

	def __str__(self) :
		return ''.join(['[var=', self.var, ', data=', str(self.data.dtype), ' ', str(self.data.shape), ']' ])
		

def reader(ifname) :
	fin = open(ifname,'rb')

	try :
		while True :
			# Read the record
			slen = fin.read(4)
			if len(slen) == 0 :
				break
			if len(slen) < 4 :
				print 'Found %d extra bytes at end of file' % (len(slen))
				break

			len0 = struct.unpack(">I",slen)[0]

			stitle = fin.read(80)
			sdata = fin.read(len0 - 80)

			len1 = struct.unpack(">I",fin.read(4))[0]

			if len0 != len1 :
				print 'Error reading record, %d (len0) != %d (len1)' % (len0,len1)
				break

			# ======================================
			# Parse the Record
			match = titleRE.match(stitle)
			if match is None :
				match = title2RE.match(stitle)
				var = match.group(1)
				dtype = numpy.dtype('>f')   # Big-endian single precision
				shape = (46, 72)
			else :
				var = match.group(1)
				dtype = datatypes[match.group(2)]
				shape = shapes[match.group(3)]


			# Read and parse the data now
			data1d = numpy.frombuffer(sdata, dtype=dtype)
			data = numpy.reshape(data1d, shape, order='C')

	#		print len(sdata)/4,len(data1d)
	#		print data1d[1:10]


			yield Record(var, data, stitle)
	finally :
		fin.close()

# Reads a list of variables out of a GISS format file
# Returns a dictionary of their values
def read_vars(fname, vars) :
	svars = set(vars)   # Allow user to pass in any collection
	ret = {}
	rd = reader(fname)
	for rec in rd :
		if rec.var in svars :
			ret[rec.var] = rec.data
			svars.remove(rec.var)
			if len(svars) == 0 :
				return ret
	print 'Could not find all variables: %s' % (svars,)

# Reads just one variable
def read_var(fname, var) :
	vars = read_vars(fname, (var,))
	return vars[var]

#for rec in giss_reader("TOPO") :
#	print rec.data.dtype,rec.var,rec.data.shape


