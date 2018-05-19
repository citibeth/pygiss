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

import sys
import argparse
import struct
import numpy
import re
import numpy as np
import netCDF4

# =========================================================================
"""Reader for the GISS data file format (giss.io) used in ModelE"""

# Parse the type specification
datatypes = {'real' : numpy.dtype('>f')}

# These shapes are in (jm,im).  I decided to use C-style
# row major indexing because numPy uses zero-based indexing
# In this way, numPy will index exactly the same as C netCDF
# utilities, and we avoid introducing a THIRD indexing convention.
_shapes = {'8x10' : (24,36)}

def _dict2re_frag(dict) :
    ret = []
    for sdim in dict.keys() :
        ret.append(sdim)
        ret.append('|')
    return ''.join(ret[0:-1])

# Regular expression to parse the title strings
#     match.group(1) = Name of variable
#     match.group(2) = Data type
#     match.group(3) = Dimension string
_titleRE = re.compile(''.join([ \
    '(.*?)[ :].*?(', \
    _dict2re_frag(datatypes), \
    ') (', \
    _dict2re_frag(_shapes), \
    ')']))
_title2RE = re.compile('(.*?)[ :].*')
# -------------------------------------------------------------

class Record(object):
    def __init__(self, grid_name, var, data, comment) :
        self.grid_name = grid_name
        self.var = var
        self.data = data
        self.comment = comment

    def __str__(self) :
        return '[grid={}, var={}, data={}, shape={}]'.format(
            self.grid_name, self.var, self.data.dtype, self.data.shape)       

# Guess shape of 2D array based on its 1D length

_len_shapes = {
    xx[1]*xx[2] : (xx[0], (xx[1], xx[2])) for xx in (
        ('1mx1m_gridreg', 10801, 21600),    # g1mx1m JM1m: 1-minute
        ('1mx1m', 10800, 21600),    # g1mx1m JM1m: 1-minute
        ('2mx2m', 5400, 10800),     # g2mx2m JM2: 2-minute
        ('10mx10m', 1080, 2160),      # g10mx10m JMS: 10-minute
        ('hxh', 360, 720),        # ghxh JMH: 1/2-degree
        ('1x1', 180, 360),        # g1x1 JM1: 1-degree
        ('q1x1', 180, 288),        # g1qx1
        ('2hx2', 90, 144),         # g2hx2
        ('5x4', 46, 72),          # g5x4??? (should be 45x72???)
    )
}






# -----------------------------------------------------------------
def reader(ifname):
    """Read records from a GISS-format file one at a time.

    Yields: Each successive record
        .var (string):
            Name of variable
        .data[nlon, nlat] (np.array, dtype=f4):
            Data that was read
        .stitle (string):
            Given name of the variable in the file"""

    fin = open(ifname,'rb')

    try :
        while True :
            # Read the record
            slen = fin.read(4)
            if len(slen) == 0:
                break
            if len(slen) < 4:
                print('Found %d extra bytes at end of file' % (len(slen)))
                break

            len0 = struct.unpack(">I",slen)[0]

            stitle = fin.read(80).decode()
            sdata = fin.read(len0 - 80)
            print('Read {} bytes'.format(len(sdata)))

            slen = fin.read(4)
            if (len(slen) == 0):
                break
            if len(slen) < 4:
                print('Found %d extra bytes at end of file' % (len(slen)))
                break
            len1 = struct.unpack(">I",slen)[0]

            if len0 != len1 :
                print('Error reading record, %d (len0) != %d (len1)' % (len0,len1))
                break

            # ======================================
            # Parse the Record
            match = _titleRE.match(stitle)
            if match is None :
                match = _title2RE.match(stitle)
                var = match.group(1)
                dtype = numpy.dtype('>f')   # Big-endian single precision
                shape = None # Guess the 2D shape down below, based on 1D length
            else :
                var = match.group(1)
                dtype = datatypes[match.group(2)]
                shape = _shapes[match.group(3)]


            # Read and parse the data now
            data1d = numpy.frombuffer(sdata, dtype=dtype)

            # Guess the shape based on length read
            if shape is None :
                grid_name, shape = _len_shapes[len(data1d)]
            data = numpy.reshape(data1d, shape, order='C')

            yield Record(grid_name, var, data, stitle)
    finally :
        fin.close()

# ------------------------------------------------------------
# Reads a list of variables out of a GISS format file
# Returns a dictionary of their values
def read_vars(fname, vars):
    """Reads a list of variables out of a GISS format file.

    Args:
        fname:
            Name of GISS-format file to read
        vars (any collection type):
            List/set of variables to read.  Ignore all other variabls.

    Returns:    {string : np.array(dtype=f4)}
        A dictionary of their values

    Raises:
        Exception:
            If not all given variables could be found in the
            GISS-format file."""

    svars = set(vars)   # Allow user to pass in any collection
    ret = {}
    rd = reader(fname)
    for rec in rd :
        if rec.var in svars :
            ret[rec.var] = rec.data
            svars.remove(rec.var)
            if len(svars) == 0 :
                return ret
    raise Exception('Could not find all variables: %s' % svars)
# ----------------------------------------------------------
def read_var(fname, var) :
    """Reads just one variable from a GISS-format file.

    Args:
        fname:
            Name of GISS-format file to read
        var (string):
            Name of variable to read out of file.

    Returns:    np.array(dtype=f4)
        Value of that variable.

    Raises:
        Exception:
            If the variable could not be found."""


    vars = read_vars(fname, (var,))
    return vars[var]
