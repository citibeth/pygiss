#!/usr/bin/env python
#
# Subroutines for adding height classes to existing ModelE input files.
# INPUT:
#     Global Initial Conditions file (from GIC in ModelE rundeck)
#     TOPO file (from TOPO in ModelE rundeck, GISS format)
#     Overlap matrix file
# OUTPUT:
#     Height-classified GIC file
#     Height-classified TOPO file (netCDF format)


#import os.path
#import re
#import sys
#import argparse
#import io
#import numpy as np
#import netCDF4
#import struct
#
#import giss.io.giss
#from giss.util import *
#from odict import odict
#import snowdrift

from giss.hcmodele.io import *
import snowdrift


# ---------------------------------------------------------------
# The core subroutine that height-classifies variables in GIC and TOPO files
# @param height_max1h[n1 x nhc] Height class definitions
# @param overlap_fnames[nis = # ice sheets] Overlap matrix for each ice sheet
# @param elevations2[nis] DEM for each ice sheet
# @param masks2[nis] Ice landmask for each ice sheet
# @param ivars Variables needed to do height classification:
#        fgrnd1, flake1, focean1, fgice1, zatmo1, tlandi1, snowli1
def hc_vars_with_snowdrift(
# Parameters to be curried
height_max1h,
ice_sheet_descrs,	# [].{overlap_fname, elevation2, mask2}
# "Generic" parameters to remain
ivars) :

	# Fetch input variables from ivars
	fgrnd1 = ivars['fgrnd1']
	flake1 = ivars['flake1']
	focean1 = ivars['focean1']
	fgice1 = ivars['fgice1']
	zatmo1 = ivars['zatmo1']
	tlandi1 = ivars['tlandi1']
	snowli1 = ivars['snowli1']

	# Get dimensions
	n1 = height_max1h.shape[0]
	nhc = height_max1h.shape[1]

	# Check dimensions
	check_shape(fgice1, (n1,), 'fgice1')
	check_shape(fgrnd1, (n1,), 'fgrnd1')
	check_shape(zatmo1, (n1,), 'zatmo1')
	check_shape(tlandi1, (n1,2), 'tlandi1')
	check_shape(snowli1, (n1,), 'snowli1')
	check_shape(height_max1h, (n1,nhc), 'height_max1h')
	for descr in ice_sheet_descrs :
#		print descr.__dict__
		check_shape(descr.elevation2, (descr.n2,), '%s:elevation2' % descr.overlap_fname)
		check_shape(descr.mask2, (descr.n2,), '%s:mask2' % descr.overlap_fname)

	# Make height-classified versions of vars by copying
	o_tlandi1h = np.zeros((nhc,n1,2))
	o_snowli1h = np.zeros((nhc,n1))
	o_elev1h = np.zeros((nhc,n1))
	for ihc in range(0,nhc) :
		o_tlandi1h[ihc,:] = tlandi1[:]
		o_snowli1h[ihc,:] = snowli1[:]
		o_elev1h[ihc,:] = zatmo1[:]

	# Initialize fhc1h to assign full weight to first height class
	o_fhc1h = np.zeros((nhc,n1))
	o_fhc1h[:] = 0
	o_fhc1h[0,:] = 1

	# Loop over each ice sheet
	for descr in ice_sheet_descrs :
		# Load the overlap matrix
		sd = snowdrift.Snowdrift(descr.overlap_fname)
		sd.init(descr.elevation2, descr.mask2, height_max1h)
		if sd.grid1().n != n1 :
			raise Exception('%s:grid1[%d] should have dimension %d', (fname, sd.grid1().n, n1))
		if sd.grid2().n != descr.n2 :
			raise Exception('%s:grid2[%d] should have dimension %d', (fname, sd.grid2().n, descr.n2))

		# Use it to compute useful stuff
		sd.compute_fhc(o_fhc1h, o_elev1h, fgice1)

	# ======= Adjust fgrnd accordingly, to keep (FGICE + FGRND) constant
	o_fgrnd1 = np.ones(n1) - focean1 - flake1 - fgice1

	# Compute zatmo, overlaying file from disk
	x_zatmo1 = np.nansum(o_fhc1h * o_elev1h, 0)
	mask = np.logical_not(np.isnan(x_zatmo1))
	o_zatmo1 = np.zeros(n1)
	o_zatmo1[:] = zatmo1[:]
	o_zatmo1[mask] = x_zatmo1[mask]

	# Return result
	ovars = odict.odict({
		'tlandi1h' : o_tlandi1h,
		'snowli1h' : o_snowli1h,
		'elev1h' : o_elev1h,
		'fhc1h' : o_fhc1h,
		'fgrnd1' : o_fgrnd1,
		'zatmo1' : o_zatmo1})
	return ovars
# ----------------------------------------------------------
# Height-classify a TOPO and GIC file
# @param nhc Number of height classes to use
def hc_files(TOPO_iname, GIC_iname, TOPO_oname, GIC_oname, hc_vars, nhc) :

	ituples = {}
	ivars = {}

	# ================ Read the TOPO file and get dimensions from it
	topo = read_gissfile_struct(TOPO_iname)
	jm = topo['zatmo'].val.shape[0]		# int
	im = topo['zatmo'].val.shape[1]		# int
	n1 = jm * im
	s_jm_im = topo['zatmo'].sdims		# symbolic
	s_nhc_jm_im = ('nhc',) + s_jm_im	# symbolic


	# ================ Prepare the input variables
	# ...from the TOPO file
	for var in ('fgrnd', 'flake', 'focean', 'fgice', 'zatmo') :
		tuple = topo[var]
		ituples[var] = tuple
		ivars[var + '1'] = tuple.val.reshape((n1,))

	# ...from the GIC file
	ncgic = netCDF4.Dataset(GIC_iname, 'r')
	for var in ('tlandi', 'snowli') :
		tuple = read_ncvar_struct(ncgic, var)
		ituples[var] = tuple
		print tuple.name,n1,str(tuple.val.shape)
		new_dims = (n1,) + tuple.val.shape[2:]
		ivars[var + '1'] = tuple.val.reshape(new_dims)
	# Leave this file open, we'll need it later whne we copy out all vars

	# ================= Height-classify the variables
	ovars = hc_vars(ivars)

	# ============= Set up new variables to output as tuples
	otlist = [
		# (name, val, sdims, dtype)
		('tlandi',
			ovars['tlandi1h'].reshape((nhc,jm,im,2)),
			('nhc',) + ituples['tlandi'].sdims,
			ituples['tlandi'].dtype),
		('snowli',
			ovars['snowli1h'].reshape((nhc,jm,im)),
			s_nhc_jm_im, ituples['snowli'].dtype),
		('elevhc',
			ovars['elev1h'].reshape((nhc,jm,im)),
			s_nhc_jm_im, ituples['zatmo'].dtype),
		('fhc',
			ovars['fhc1h'].reshape((nhc,jm,im)),
			s_nhc_jm_im, 'f8'),
		('fgrnd',
			ovars['fgrnd1'].reshape((jm,im)),
			s_jm_im, ituples['fgrnd'].dtype),
		('zatmo',
			ovars['zatmo1'].reshape((jm,im)),
			s_jm_im, ituples['zatmo'].dtype)]

	# Convert to a dict of tuple Structs
	otuples = odict.odict()
	for ot in otlist :
		otuples[ot[0]] = giss.util.Struct({
			'name' : ot[0], 'val' : ot[1],
			'sdims' : ot[2], 'dtype' : ot[3]})

	# ============= Write the TOPO file (based on old variables)
	# Collect variables to write
	otuples_remain = odict.odict(otuples)
	wvars = []	# Holds pairs (source, name)
	for name in topo.iterkeys() :
		# Fetch variable from correct place
		if name in otuples :
			wvars.append((otuples, name))
			del otuples_remain[name]
		else :
			wvars.append((topo, name))

	# Define and write the variables
	write_netcdf(TOPO_oname, wvars)


	# ============= Write out new GIC
	# Collect variables to write (from original GIC file)
	wvars = []	# Holds pairs (source, name)
	for name in ncgic.variables :
		# Fetch variable from correct place
		if name in otuples :
			wvars.append((otuples, name))
			del otuples_remain[name]
		else :
			wvars.append((ncgic, name))

	# Add on new variable(s) we've created
	for name in otuples_remain.iterkeys() :
		wvars.append((otuples, name))

	# Define and write the variables
	write_netcdf(GIC_oname, wvars)

# -------------------------------------------------------------

