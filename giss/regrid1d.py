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

import bisect
import numpy as np

# Simple conservative regridding for constant-valued regions.

def integrate_weights(dz, z0, z1):
    """dz: Depth of each layer"""

    top = np.cumsum(dz)     # Top of each layer
    ix = []
    weights = []

    i0 = biset.bisect_left(top, z0)
    i1 = biset.bisect_left(top, z1)

    print(i0, i1)
