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

import pprint as xpprint

def to_dict(obj):
    """Reads all properties out of an f90wrap object."""
    ret = {}
    for attr,prop in type(obj).__dict__.items():
        if type(prop) == property:
            ret[attr] = prop.fget(obj)
    return ret

def pprint(obj):
    return xpprint.pprint(to_dict(obj))
