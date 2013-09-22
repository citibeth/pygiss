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

def test() :
	data_root = os.environ['SNOWDRIFT_FIG_DATA']
	cmrun = os.path.join(os.environ['HOME'], 'cmrun')

	hcsuffix = '_hc'
	TOPO_in  = os.path.join(cmrun, 'Z72X46N.cor4_nocasp')
	TOPO_out = os.path.join(cmrun, 'Z72X46N.cor4_nocasp' + hcsuffix + '.nc')
	GIC_in  = os.path.join(cmrun, 'GIC.E046D3M20A.1DEC1955.ext.nc')
	GIC_out = os.path.join(cmrun, 'GIC.E046D3M20A.1DEC1955.ext' + hcsuffic + '.nc')


	grid1_name = 'll_4x5'

	# A set of pairs, one per ice sheet: (overlap_fname, searise_fname)
	fnamess = (
		(os.path.join(data_root,
			'overlap/' + grid1_name + '-searise_Greenland_5km.nc'),
		os.path.join(data_root,
			'searise/Greenland_5km_v1.1.nc')),
	)

	n1 = get_grid1_size(fnamess[0][0])

	# ice_sheet_descrs.{overlap_fname, n2, elevation2, mask2}
	ice_sheet_descrs = []
	for fnames in fnamess :  # Greenland, Antarctica, etc
		overlap_fname = fnames[0]
		searise_fname = fnames[1]
		elevation2, mask2 = read_searise(searise_fname)
		ice_sheet_descrs.append(giss.util.Struct({
			'overlap_fname' : overlap_fname,
			'n2' : elevation2.shape[0],
			'elevation2' : elevation2,
			'mask2' : mask2}))

	tops = np.array([200,400,700,1000,1300,1600,2000,2500,3000,10000], dtype='d')
	height_max1h = const_height_max1h(tops, n1)

	hc_vars = curry(hc_vars_with_snowdrift, height_max1h, ice_sheet_descrs)
	#overlap_fnames, elevation2, masks2)
	hc_files(TOPO_in, GIC_in TOPO_out, GIC_out, hc_vars)
