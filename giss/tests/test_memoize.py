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
import tempfile,shutil
import os
from giss import memoize,checksum
#import giss.tests.test_memoize_support as support


nfib = 0

@memoize.local()
def fib(n):
    global nfib
    nfib += 1
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)


class TestMemoizeLocal(unittest.TestCase):

    hash_version=1
    def hashup(self, hash):
        pass

    def setUp(self):
        self.nfib = 0

    def fib_plain(self, n):
        self.nfib += 1
        if n < 2:
            return n
        return self.fib_plain(n-1) + self.fib_plain(n-2)

    @memoize.local()
    def fib(self, n):
        self.nfib += 1
        if n < 2:
            return n
        return self.fib(n-1) + self.fib(n-2)

    def test_fib(self):
        global nfib

        # Memoization state...
        ms = dict()

        # Plain fib -- no memoization
        self.nfib = 0
        self.assertEqual(21, self.fib_plain(8))
        self.assertEqual(67, self.nfib)

        # Memoized fib fn
        nfib = 0
        self.assertEqual(21, fib(8))
        self.assertEqual(9, nfib)

        # Memoized fib fn
        self.nfib = 0
        self.assertEqual(21, self.fib(8))
        self.assertEqual(9, self.nfib)

# ---------------------------------------------------


def touch(fname, times=None):
    """For testing, touch() will change the size, not the timestamp.
    Tests run faster than the timestamp resolution."""
    with open(fname, 'ab') as fout:
        fout.write(b'\n')
#        os.utime(fname, times)

file_fn1_ncall = 0
@memoize.files()
class filefn1(object):
    hash_version = 0
    def __init__(self, ival, ofname):
        self.ival = ival
        ofname = os.path.realpath(ofname)
        self.inputs = []
        self.outputs = [(ofname, (self.ival,))]
        self.value = self.outputs[0][0]

    def __call__(self):
        global file_fn1_ncall
        file_fn1_ncall += 1

        with open(self.outputs[0][0], 'w') as out:
            out.write('{}\n'.format(self.ival))
        return self.value

copy_files_ncall = 0
@memoize.files()
class copy_file(object):
    hash_version = 0
    def __init__(self, ifname, odir):
        self.odir = odir
        ileaf = os.path.split(ifname)[1]
        self.inputs = [os.path.realpath(ifname)]
        self.outputs = [(
            os.path.realpath(os.path.join(odir, ileaf)),
            (memoize.File(self.inputs[0]),)
        )]
        self.value = self.outputs[0][0]

    def __call__(self):
        global copy_file_ncall
        copy_file_ncall += 1

        shutil.copy(self.inputs[0], self.outputs[0][0])
        return self.value


class TestMemoizeFiles(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_memoize_files(self):
        global file_fn1_ncall

        # Basic memoization test
        ofname1 = os.path.join(self.dir, 'file1')
        val1a = filefn1(17, ofname1)
        file_fn1_ncall = 0
        val1b = filefn1(17, ofname1)
        self.assertEqual(val1b, val1a)
        self.assertEqual(0, file_fn1_ncall)

        # Try different inputs
        file_fn1_ncall = 0
        val1a = filefn1(23, ofname1)    # In same file
        self.assertEqual(1, file_fn1_ncall)

        file_fn1_ncall = 0
        val1a = filefn1(23, ofname1+'-23')    # In different file
        self.assertEqual(1, file_fn1_ncall)

        # Try touching output file
        file_fn1_ncall = 0
        val1a = filefn1(23, ofname1)    # In same file
        self.assertEqual(0, file_fn1_ncall)
        touch(ofname1)
        file_fn1_ncall = 0
        val1a = filefn1(23, ofname1)    # In same file
        self.assertEqual(1, file_fn1_ncall)

        # Try removing output's origin file
        os.remove(os.path.join(self.dir, '.file1.origin'))
        file_fn1_ncall = 0
        val1a = filefn1(23, ofname1)    # In same file
        self.assertEqual(1, file_fn1_ncall)

    def test_chained_memoize(self):
        global file_fn1_ncall
        global copy_file_ncall

        odir = os.path.join(self.dir, 'copy_fn')
        os.makedirs(odir)

        ofname1 = os.path.join(self.dir, 'file1')
        fname1 = filefn1(17, ofname1)

        # Make sure it has the right value inside
        with open(fname1, 'r') as fin:
            val = int(next(fin))
            self.assertEqual(17, val)

        # Chain to another function
        file_fn1_ncall = 0
        copy_file_ncall = 0
        fname2 = copy_file(filefn1(17, ofname1), odir)
        self.assertTrue(os.path.exists(fname2))
        self.assertEqual(0, file_fn1_ncall)
        self.assertEqual(1, copy_file_ncall)

        # Make sure it has the right value inside
        with open(fname2, 'r') as fin:
            val = int(next(fin))
            self.assertEqual(17, val)

        # Test chained memoization
        file_fn1_ncall = 0
        copy_file_ncall = 0
        copy_file(fname1, odir)
        # Careful to change the inputs to filefn1() here to ensure
        # a different SIZE file; we cannot rely on timestamps in unit test.
        copy_file(filefn1(289, ofname1), odir)
        self.assertEqual(1, file_fn1_ncall)
        self.assertEqual(1, copy_file_ncall)

        # Test chained memoization
        file_fn1_ncall = 0
        copy_file_ncall = 0
        touch(ofname1)
        copy_file(filefn1(23, ofname1), odir)
        self.assertEqual(1, file_fn1_ncall)
        self.assertEqual(1, copy_file_ncall)



if __name__ == '__main__':
    unittest.main()
