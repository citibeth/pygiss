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

