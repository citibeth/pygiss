import unittest
import tempfile
import os
import shutil

from giss import xaccess
from giss.xaccess import _ix,ncdata,ncfetch,ncdata
import types
from giss import functional
from giss.functional import _arg
import sys
import numpy as np
import netCDF4
import sys
from giss import giutil

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


# ---------------------------------------------
    def test_ncfetch(self):
        ix = _ix[:]

        sample1 = functional.bind(ncfetch, self.ncname, 'sample1')
        sample2 = functional.bind(ncfetch, self.ncname, 'sample2')
        samplesum = sample1 + sample2

        # Make sure the meta-data combined properly
        self.assertEqual('sample1',
            sample1(*ix).attrs()[('var', 'name')])
        self.assertFalse(
            'name' in samplesum(*ix).attrs())
        self.assertEqual(('ni', 'nj'),
            samplesum(*ix).attrs()[('var', 'dimensions')])

        # Now retrieve data
        data1 = sample1(*ix).data()
        data2 = sample2(*ix).data()
        datasum = samplesum(*ix).data()

        self.assertTrue(np.all(datasum == data1+data2))


if __name__ == '__main__':
    unittest.main()

