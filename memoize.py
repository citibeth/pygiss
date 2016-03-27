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

# https://pypi.python.org/pypi/memoize/

from functools import wraps


def mproperty(fn):
    attribute = "_memo_%s" % fn.__name__

    @property
    @wraps(fn)
    def _property(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, fn(self))
        return getattr(self, attribute)

    return _property

