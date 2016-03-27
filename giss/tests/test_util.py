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

import unittest
from giss import giutil

class TestLazyDict(unittest.TestCase):

    def test_lazy_dict(self):
        ld = giutil.LazyDict()
        ld['xworld'] = 'world'
        ld.lazy['xbar'] = giutil.CallCounter(lambda: 'bar')

        self.assertEqual('world', ld['xworld'])
        self.assertEqual(0, ld.lazy['xbar'].count)
        self.assertEqual('bar', ld['xbar'])
        self.assertEqual(1, ld.lazy['xbar'].count)
        self.assertEqual('bar', ld['xbar'])
        self.assertEqual(1, ld.lazy['xbar'].count)

        self.assertEqual('world', ld.lazy['xworld']())
        self.assertEqual('bar', ld.lazy['xbar']())


if __name__ == '__main__':
    unittest.main()
