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
from giss import memoize,checksum
#import giss.tests.test_memoize_support as support


nfib = 0

@memoize.memoized()
def fib(ms, n):
    global nfib
    nfib += 1
    if n < 2:
        return n
    return fib(ms, n-1) + fib(ms, n-2)


class TestMemoize(unittest.TestCase):

    hash_version=1
    def hashup(self, hash):
        pass

    def setUp(self):
        self.nfib = 0

    def fib_plain(self, ms, n):
        self.nfib += 1
        if n < 2:
            return n
        return self.fib_plain(ms, n-1) + self.fib_plain(ms, n-2)

    @memoize.memoized()
    def fib(self, ms, n):
        self.nfib += 1
        if n < 2:
            return n
        return self.fib(ms, n-1) + self.fib(ms, n-2)

    def test_fib(self):
        global nfib

        # Memoization state...
        ms = dict()

        # Plain fib -- no memoization
        self.nfib = 0
        self.assertEqual(21, self.fib_plain(ms, 8))
        self.assertEqual(67, self.nfib)

        # Memoized fib fn
        nfib = 0
        self.assertEqual(21, fib(ms, 8))
        self.assertEqual(9, nfib)

        # Memoized fib fn
        self.nfib = 0
        self.assertEqual(21, self.fib(ms, 8))
        self.assertEqual(9, self.nfib)

if __name__ == '__main__':
    unittest.main()
