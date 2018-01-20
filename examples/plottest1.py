# PyGISS: Misc. Python library
# Copyright (c) 2013-2016 by Elizabeth Fischer
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Simplest possible ModelE data plotting demo

import netCDF4
import giss.basemap
import modele.deprecated_plot_params as plot_params
import sys

if len(sys.argv) < 2 :
    var_name = 'tsurf'
else :
    var_name = sys.argv[1]


nc = netCDF4.Dataset('data/JAN1950.aijE4F40.R.nc')
pp = plot_params.plot_params(var_name, nc=nc)
giss.plot.plot_var(**pp)        # Plot, and show on screen

# Slightly more complex alternatives:
# Save figure:
#   giss.plot.plot_var(fname='plottest1.png', **pp)
# Save figure and snow on screen
#   giss.plot.plot_var(fname='plottest1.png', show=True, **pp)
