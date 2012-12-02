import giss.snowdrift
import netCDF4
import numpy as np
import giss.ncutil
import pyproj
import scipy.sparse
import operator

# Given an overlap matrix, for each column, finds the row with the biggest value.
# @param overlap_shifted scipy.sparse.coo_matrix, overlap between grid2
#        and a shifted version of grid1
# @return lower1[n2] Equals lower_ilat*nlon + lower_ilon, giving the lower "nearest"
#         grid1 centers in the latitude and longitude dimensions, for cell i2
def max_by_column(overlap_shifted) :
	n1 = overlap_shifted.shape[0]
	n2 = overlap_shifted.shape[1]
	max_overlap = np.zeros(n2)
	lower1 = np.zeros(n2, dtype=np.int32) - 1

	# Choose the cell in shifted-grid1 that has the MOST overlap with each cell in grid2
#	print len(overlap_shifted.row)
#	print overlap_shifted.row[0:20]
#	print overlap_shifted.col[0:20]
#	print overlap_shifted.data[0:20]
	for (i1, i2, val) in zip(overlap_shifted.row, overlap_shifted.col, overlap_shifted.data) :
		if val > max_overlap[i2] :
			max_overlap[i2] = val
			lower1[i2] = i1
#			print 'lower1[%d] = %d (%f)' % (i2, i1, val)

	return lower1


# @return Sparse matrix A.  Regrid with v2 = A.transpose()*v1
def BilinInterp2(overlap_nc, overlap_shifted_fname, elevation2, mask2, mask1h, tops) :
	projs = giss.snowdrift.read_projs(overlap_nc, 'grid1')
	latb1 = giss.ncutil.read_ncvar(overlap_nc, 'grid1.lat_boundaries')
#	print ' $$$$$$$$$$ ' + str(len(giss.snowdrift.cell_centers(latb1)))
#	print len(latb1)
	lats1 = np.zeros(len(latb1) + 1, 'd')
	lats1[0] = -90.0
	lats1[1:-1] = giss.snowdrift.cell_centers(latb1)
	lats1[-1] = 90.0

#	print 'lats1 = ' + str(lats1)
	lons1 = giss.snowdrift.cell_centers(giss.ncutil.read_ncvar(overlap_nc, 'grid1.lon_boundaries'))

	# Set up for CESM-style bilinear interpolation
	overlap_shifted_nc = netCDF4.Dataset(overlap_shifted_fname)
	overlap_shifted = giss.snowdrift.read_sparse_matrix(overlap_shifted_nc, 'overlap')
	lower1 = max_by_column(overlap_shifted)
	overlap_shifted_nc.close()

	# ======== Get (lat, lon) of every cell in grid2
	x_centers2 = giss.snowdrift.cell_centers(giss.ncutil.read_ncvar(overlap_nc, 'grid2.x_boundaries'))
	y_centers2 = giss.snowdrift.cell_centers(giss.ncutil.read_ncvar(overlap_nc, 'grid2.y_boundaries'))

	# Cartesian product magic to get list of centers of cells in grid2
	xys2 = np.transpose([np.tile(x_centers2,len(y_centers2)), np.repeat(y_centers2, len(x_centers2))])
	xs2 = xys2[:,0]
	ys2 = xys2[:,1]

	# Convert to lats and lons
	(lons2, lats2) = pyproj.transform(projs[1], projs[0], xs2, ys2)
	# ==========

	return BilinInterp(lower1, lons1, lats1, lons2, lats2, tops, elevation2, mask2, mask1h)


# Interpolate in Z direction as well.
#
# Bi-linear interpolation from a lat-lon grid to an arbitrary set of points.
# Grid1 is a set of gridded points with rectilinear lat/lon coordinates, NOT regions here.
# @param int lower1[n2] The center of gridcell2 has four nearest neighbor centers in grid1.
#         This gives the one with the smallest i1 value (lat * nlon + lon).
# @param lats1[nlat1] Latitude of centers of cells in grid1 (must be lat/lon grid)
# @param lons1[nlon1] Longitude of center of cells in grid1 (must be lat/lon grid)
# @param lats2[n2] Latitude of centers of cells in grid2
# @param lons2[n2] Longitude of centers of cells in grid2
# @param mask2[n2] True for gridcells in ice grid that have ice
# @return Sparse matrix A.  Regrid with v2 = A.transpose()*v1
class BilinInterp :
	def interpolate(self, _val1h) :
		val1h = _val1h.reshape(reduce(operator.mul, _val1h.shape))	# Convert to 1-D
		v1h = np.copy(val1h)
		v1h[np.isnan(val1h)] = 0	# Prepare for multiply
		val2 = self.bilinz.transpose() * v1h
		val2[np.logical_not(self.mask2)] = np.nan
		return val2

	def __init__(self, lower1, lons1, lats1, lons2, lats2, tops, elevation2, mask2, _mask1h) :
		mask1h = _mask1h.reshape(reduce(operator.mul, _mask1h.shape))	# 1-dimensionalize
		self.mask2 = mask2
	#	(lons2, lats2) = pyproj.transform(projs[1], proj2[0], xs2, ys2)

		height_classifier = giss.snowdrift.HeightClassifier(tops)

		# This must match elev1h definition in hc_vars_ncar.py
		hclev = [tops[0] * .5]
		hclev.extend(.5 * (tops[1:] + tops[0:-1]))
		hclev = np.array(hclev)
		print 'tops = ' + str(tops)
		print 'hclev = ' + str(hclev)
		height_classifier_shifted = giss.snowdrift.HeightClassifier(hclev)


		rows = []
		cols = []
		vals = []

		n2 = lower1.shape[0]
		n1 = len(lons1)*len(lats1)
		nhc = len(tops)
	#	print 'nhc = %d %d' % (nhc, len(hclev))
		nlon = len(lons1)
		print 'nlon=%d' % nlon
		for i2 in range(0,n2) :

			if not mask2[i2] : continue

			# Figure out the two height classes to interpolate between
			# Assume height classes are the same for all gridcells, or else interpolating
			# in the xy plane within the same height class wouldn't make sense.
			# Therefore, we just use the height class for GCM gridcell i1=0
			ihc1 = height_classifier_shifted.get_hclass(0, elevation2[i2])
			if ihc1 == 0:
				hcfrac = [(0,1.0)]	# Linear combination of height classes we need for this gridcell
			else :
				ihc0 = ihc1-1
				elev_frac = 1.0-(elevation2[i2] - hclev[ihc0]) / (hclev[ihc1] - hclev[ihc0])
				hcfrac = [(ihc0, elev_frac), (ihc1, 1.0-elev_frac)]

			for ihc, hcweight in hcfrac :
				i2hc = ihc*n2 + i2

				i0 = lower1[i2]
				ilat0 = i0 / nlon
				ilon0 = i0 - ilat0 * nlon

				ilon1 = ilon0+1
				lon_frac = 1.0 - (lons2[i2] - lons1[ilon0]) / (lons1[ilon1] - lons1[ilon0])
				if np.abs(lon_frac) > 1.0 :
					print '|lon_frac| = %f must be < 1', lon_frac
					raise Exception

				ilat1 = ilat0+1
				lat_frac = 1.0 - (lats2[i2] - lats1[ilat0]) / (lats1[ilat1] - lats1[ilat0])
				if np.abs(lat_frac) > 1.0 :
					print '|lat_frac| = %f must be < 1', lat_frac
					raise Exception

				# Assume nhc dimension first, as in netCDF files (opposite of Snowdrift i1h convention)
				xrows = [ihc*n1 + x for x in [ilat0*nlon + ilon0, ilat0*nlon + ilon1, ilat1*nlon + ilon0, ilat1*nlon + ilon1]]
				xcols = [i2, i2, i2, i2]
				xvals = [lat_frac*lon_frac, lat_frac*(1-lon_frac), (1-lat_frac)*lon_frac, (1-lat_frac)*(1-lon_frac)]

				# Avoid gridcells that won't have data in them
				xxs = filter(lambda xx : mask1h[xx[0]], zip(xrows, xcols, xvals))
				xrows = [xx[0] for xx in xxs]
				xcols = [xx[1] for xx in xxs]
				xvals = [xx[2] for xx in xxs]

				factor = hcweight / sum(xvals)
				xvals = [factor * x for x in xvals]

#				print 'xvals = ',xvals,sum(xvals) / hcweight

				rows.extend(xrows)
				cols.extend(xcols)
				vals.extend(xvals)



		print 'making bilinz nhc=%d, n1=%d, n2=%d nele=%d' % (nhc, n1, n2, len(vals))
		self.bilinz = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(n1*nhc,n2))

