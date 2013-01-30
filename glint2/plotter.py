# (elevation2, mask2) can come from giss.searise.read_elevation2_mask2
# @param mask2 a Boolean array, True where we want landice, False where none
#        (This is the opposite convention from numpy masked arrays)
# @param height_max1h (nhc x n1) array of height class definitions
class Plotter_hc :
	def __init__(self, grid2_plotter, overlaph) :
		self.grid2_plotter = grid2_plotter
		self.sd = snowdrift

		# Check dims
		if grid2_plotter.n2 != self.sd.grid2().n :
			raise Exception('n2 (%d) != sd.grid2().n (%d)' % (grid2_plotter.n2, sd.grid2().n))

	def pcolormesh(self, mymap, val1h_varshape, **plotargs) :
		# Consolidate dimensions so this is (nhc, n1)
		nhc = val1h_varshape.shape[0]
		n1 = reduce(operator.mul, val1h_varshape.shape[1:])
		val1h = val1h_varshape.reshape((nhc, n1))

		if self.sd.grid1().n != n1 :
			raise Exception('sd.grid1().n (%d) != n1 (%d)' % (sd.grid1().n, n1))

		# Do a simple regrid to grid 2 (the local / ice grid)
		val2 = np.zeros((self.grid2_plotter.n2,))
		val2[:] = np.nan
		self.sd.downgrid(val1h, val2, merge_or_replace = 'replace', correct_proj_area = 0)	# Area-weighted remapping

		# Masked areas will remain nan in val2
		# Create a numpy plotting mask with this in mind.
		#val2_masked = ma.masked_array(val2, np.isnan(val2))
		val2_masked = ma.masked_invalid(val2)
		print 'Grid1hPlotter: (min,max) = (%f, %f)' % (np.min(val2_masked), np.max(val2_masked))
		# Plot using our local plotter
		return self.grid2_plotter.pcolormesh(mymap, val2_masked, **plotargs)
