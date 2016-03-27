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

import datetime
import re

monthRE_pat = r'JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC\d\d\d\d'

monthRE = re.compile(r'(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d\d\d\d)')
str_to_month = {
    'JAN' : 1,'FEB' : 2,'MAR' : 3,'APR' : 4,'MAY' : 5,'JUN' : 6,
    'JUL' : 7,'AUG' : 8,'SEP' : 9,'OCT' : 10,'NOV' : 11,'DEC' : 12}

def monthstr_to_date(ms):
    month = _monthnumbs[ms[0:3]]
    year = int(ms[3:])
    return datetime.date(month,year)

def monthnum(year, month):
    return year * 12 + (month - 1)

def date_to_monthnum(date):
    return date.year * 12 + (date.month - 1)

def monthnum_to_date(mn):
    year = int(mn) / 12
    month = (mn - year * 12) + 1
    return datetime.date(year,month,1)

