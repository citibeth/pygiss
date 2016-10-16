import unittest
import tempfile
import os
import shutil

from giss import xaccess
from giss.xaccess import _ix,ncdata,ncattrs
import types
from giss import functional
from giss.functional import _arg
import sys
import numpy as np
import netCDF4
import sys

class TestXAccess(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.ncname = os.path.join(self.dir, 'test.nc')

        # Create a sample NetCDF file to read
        with netCDF4.Dataset(self.ncname, 'w') as nc:
            ni=3
            nj=4
            nc.createDimension('ni', ni)
            nc.createDimension('nj', nj)
            var = nc.createVariable('sample1', 'd', ('ni', 'nj'))
            var.sameattr = 'sameattr'
            var.differentattr = 17
            val = np.zeros((ni,nj))
            for i in range(0,ni):
                for j in range(0,nj):
                    val[i,j] = i+j
            var[:] = val

            var = nc.createVariable('sample2', 'd', ('ni', 'nj'))
            var.sameattr = 'sameattr'
            var.differentattr = 23
            val = np.zeros((ni,nj))
            for i in range(0,ni):
                for j in range(0,nj):
                    val[i,j] = i-j
            var[:] = val
                    
    def tearDown(self):
        shutil.rmtree(self.dir)

    def xtest_ncdata(self):
        """Test fetching and combining of data"""

        sample1 = functional.bind(ncdata, _arg(0), 'sample1')
        sample2 = functional.bind(ncdata, _arg(0), 'sample2')
        sumsample = sample1 + sample2

        ix = _ix[:]
        val1 = sample1(self.ncname, *ix)()
        val2 = sample2(self.ncname, *ix)()
        sum1 = val1 + val2
        sum2 = sumsample(self.ncname, *ix)()
        self.assertTrue(np.all(sum1 == sum2))

        # Test ability to add recurisvly
        sampleB = sumsample + sample1
        sumB = sampleB(self.ncname, *ix)()
        self.assertTrue(np.all(sumB == sum2 + val1))

    def xtest_ncattrs(self):
        """Test fetching and combining of meta-data"""

        sample1 = functional.bind(ncattrs, self.ncname, 'sample1')
        sample2 = functional.bind(ncattrs, self.ncname, 'sample2')
        print('sample1', sample1)
        sumsample = sample1 + sample2

        ix = _ix[:]
        attrs1 = sample1(*ix)()
        attrs2 = sample2(*ix)()
        attrs3 = sumsample(self.ncname, *ix)()
        self.assertTrue(('var', 'sameattr') in attrs3)
        self.assertFalse(('var', 'differentattr') in attrs3)
        self.assertFalse(('fetch', 'var') in attrs3)
        self.assertTrue(('fetch', 'var') in attrs1)

# ---------------------------------------------
    def try_tuple(self, sample1, *args):
        # Test indexing through the Tuple
        d1 = sample1[0](*args)()
        d2 = sample1(*args)[0]()
        d3 = sample1(*args)()[0]
        self.assertEqual(d1,d2)
        self.assertEqual(d1,d3)
        self.assertIsInstance(d1, dict)


    def test_ncfetch_tuple(self):
        ix = _ix[:]

        ncfetch = functional.Tuple((ncattrs, ncdata))
        args = (self.ncname, 'sample1', *ix)
        self.try_tuple(ncfetch, *args)
        self.try_tuple(ncfetch+ncfetch, *args)

        sample1 = functional.bind(ncfetch,
            self.ncname, 'sample1')
        self.try_tuple(sample1, *ix)
        self.try_tuple(sample1+sample1, *ix)

# ---------------------------------------------
    def try_tuple(self, sample1, *args):
        # Test indexing through the Tuple
        d1 = sample1[0](*args)()
        d2 = sample1(*args)[0]()
        d3 = sample1(*args)()[0]
        self.assertEqual(d1,d2)
        self.assertEqual(d1,d3)
        self.assertIsInstance(d1, dict)


    def test_ncfetch_tuple(self):
        ix = _ix[:]

        ncfetch = functional.Tuple((ncattrs, ncdata))
        args = (self.ncname, 'sample1', *ix)
        self.try_tuple(ncfetch, *args)
        self.try_tuple(ncfetch+ncfetch, *args)

        sample1 = functional.bind(ncfetch,
            self.ncname, 'sample1')
        self.try_tuple(sample1, *ix)
        self.try_tuple(sample1+sample1, *ix)




    def xtest_ncfetch_sum(self):
        sample1 = functional.bind(ncattrs, _arg(0), 'sample1')
        sample2 = functional.bind(ncattrs, _arg(0), 'sample2')
        sumsample = sample1 + sample2
        sum2 = sumsample + sample1

        attrs1 = sample1(self.ncname)
        print('attrs1', attrs1)
        attrs = sumsample(self.ncname)
        print('attrs', attrs)

    def xtest_ncfetch(self):
        sample1 = functional.bind(ncfetch, _arg(0), 'sample1')
        sample2 = functional.bind(ncfetch, _arg(0), 'sample2')
        print('---- sample1 bases', type(sample1).__bases__)
        sumsample = sample1 + sample2

        ix = _ix[:]
        attrs1,thunk1 = sample1(self.ncname, *ix)
        attrs2,thunk2 = sample2(self.ncname, *ix)

        xx = sumsample(self.ncname, *ix)
        print('******', xx)

#        print(thunk1, attrs1)
#        print(thunk2, attrs2)

        val1 = ncdata(self.ncname, 'sample1', *ix)
        val2 = ncdata(self.ncname, 'sample2', *ix)
        self.assertTrue(np.all(val2 == thunk2()))
        self.assertTrue(np.all(val1 == thunk1()))

#        # Now try the sums...
#        sumsample = sample1 + sample2
#        print('sumsample', sumsample)
#        print(sumsample(self.ncname, *ix))

#        sum1 = val1 + val2
#        sum2 = sumsample(self.ncname, *ix)
#
#        self.assertTrue(np.all(sum1 == sum2))



if __name__ == '__main__':
    unittest.main()

