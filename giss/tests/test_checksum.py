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
from giss import checksum

def fib(ms, n):
    if n < 2:
        return n
    return fib(ms, n-1) + fib(ms, n-2)

class TestChecksum(unittest.TestCase):

    hash_version=1
    def hashup(self, hash):
        pass

    def fn2(self, x):
        pass

    def test_checksum(self):
        cs = checksum.checksum
        self.assertEqual(cs(17), cs(17))
        self.assertNotEqual(cs(17), cs(18))

        self.assertEqual(cs([17, 18]), cs([17, 18]))
        self.assertNotEqual(cs([17, 18]), cs([18, 17]))

        cs(fib)
        cs(self.fn2)

if __name__ == '__main__':
    unittest.main()
