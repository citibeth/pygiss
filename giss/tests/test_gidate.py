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
from giss import gidate
import datetime

def date_iter(low,high):
    """Iterates through a bunch of giutil.Date values"""
    plow = datetime.datetime(*(tuple(low) + (1,1)))
    phigh = datetime.datetime(*(tuple(high) + (1,1)))

    print('Testing dates in range', plow, phigh)

    pd = plow
    while pd != phigh:
        yield gidate.Date(pd.year, pd.month, pd.day)
        pd = pd + datetime.timedelta(days=1)

class TestGIDate(unittest.TestCase):

    def _tst_date_range(self, low, high):
        last = gidate.date_to_jday(low)
        for date in date_iter(low+1, high):
            jday = gidate.date_to_jday(date)
            # print(date, jday)
            self.assertEqual(last+1, jday)
            self.assertEqual(date, date+0)
            last = jday

    def test_date(self):
        self._tst_date_range(gidate.Date(1000,1,2), gidate.Date(1200,12,31))
        self._tst_date_range(gidate.Date(2000,1,2), gidate.Date(2100,12,31))
        self._tst_date_range(gidate.Date(4000,1,2), gidate.Date(4100,12,31))
#        self._tst_date_range(gidate.Date(0,1,2), gidate.Date(100,12,31))

if __name__ == '__main__':
    unittest.main()
