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

from giss.plot.plotutil import *
from giss.plot.plot_var import *
from giss.plot.plotters import *


def plot_one(pp, fname='fig.png'):
    """Creates a single plot from a plot_params object"""

    # ---------- Plot it!
    figure = matplotlib.pyplot.figure(figsize=(8.5,11))
    ax = figure.add_subplot(111)
    basemap = giss.basemap.greenland_laea(ax=ax)
    giss.plot.plot_var(ax=ax, basemap=basemap, **pp)
#    matplotlib.pyplot.show()
    print('Writing {}'.format(fname))
    figure.savefig(fname, dpi=100, transparent=False)
