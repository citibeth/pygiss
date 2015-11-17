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

# Demonstrate more than one figure on a page.

import netCDF4
import giss.basemap
import giss.modele
import matplotlib.pyplot
import mpl_toolkits.basemap

#fname = '/Users/rpfische/exp/120915-hccontrol/DEC1949.aijhctest45.nc'
#nc = netCDF4.Dataset(fname)
nc = netCDF4.Dataset('data/ANN1950.aijhctest45_lr05.nc')

# Use a custom basemap
basemap = giss.basemap.global_map()

basemap = mpl_toolkits.basemap.Basemap(projection='cea',llcrnrlat=-90,urcrnrlat=90,\
            llcrnrlon=-180,urcrnrlon=180,resolution='c')



# Plot multiple plots on one page
figure = matplotlib.pyplot.figure(figsize=(11,8.5))

ax = figure.add_subplot(121)
pp = giss.modele.plot_params('pr_lndice', nc=nc)
print(pp)
print(ax)
giss.plot.plot_var(ax=ax, basemap=basemap, **pp)

ax = figure.add_subplot(122)
pp = giss.modele.plot_params('evap', nc=nc)
giss.plot.plot_var(ax=ax, basemap=basemap, **pp)

# Save to a file as png
figure.savefig('fig.png', dpi=300, transparent=True)

# Also show on screen
matplotlib.pyplot.show()
