# pyGISS: GISS Python Library
# Copyright (c) 2013 by Robert Fischer
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import mpl_toolkits.basemap
import numpy as np
import re
import array

"""Some pre-defined map projections, commonly used at GISS"""

# The entire globe --- good for modelE output
def global_map(ax=None) :
    """Args:
        ax (axes):
            Set default axes instance (see mpltoolkits.basemap.Basemap)
                
    Returns: Basemap instance
        With low-resolution coastlines
    See:
        mpl_toolkits.basemap.Basemap"""

    return mpl_toolkits.basemap.Basemap(ax=ax, projection='kav7',lon_0=0,resolution='l')
#   return mpl_toolkits.basemap.Basemap(ax=ax, projection='robin',lon_0=0,resolution='l')



def north_laea(ax=None) :
    """Args:
        ax (axes):
            Set default axes instance (see mpltoolkits.basemap.Basemap)
                
    Returns:    North polar, Lambert Equal Area Projection.
        With low-resolution coastlines.
        Distortion minimized at 72 north.
        Map is 7000x7000 km

    See:
        mpl_toolkits.basemap.Basemap
    """
    km = 1000.0
    return mpl_toolkits.basemap.Basemap(ax=ax,
        width=7000 * km,height=7000 * km,
        resolution='l',projection='laea',\
        lat_ts=72,lat_0=90,lon_0=-90.)

# South polar projection
def south_laea(ax=None) :
    """Args:
        ax (axes):
            Set default axes instance (see mpltoolkits.basemap.Basemap)
                
    Returns:    South polar, Lambert Equal Area Projection.
        With low-resolution coastlines.
        Distortion minimized at 72 north.
        Map is 7000x7000 km

    See:
        mpl_toolkits.basemap.Basemap"""

    km = 1000.0
    return mpl_toolkits.basemap.Basemap(ax=ax,
        width=7000 * km,height=7000 * km,
        resolution='l',projection='laea',\
        lat_ts=-72,lat_0=-90,lon_0=0.)

def greenland_laea(ax=None) :
    """Args:
        ax (axes):
            Set default axes instance (see mpltoolkits.basemap.Basemap)
                
    Returns:    Map for plotting Greenland.
        With low-resolution coastlines.

    See:
        mpl_toolkits.basemap.Basemap"""
    km = 1000.0
    return mpl_toolkits.basemap.Basemap(ax=ax,
        resolution='l',projection='laea',\
        lat_ts=72,lat_0=90,lon_0=-40.,
        llcrnrlon=-54., llcrnrlat=58.,
#       width=1634940, height=2622574)
#       width=1634940, height=2622574)
        urcrnrlon=5., urcrnrlat=80.)


#llcrnrlon  longitude of lower left hand corner of the desired map domain (degrees).
#llcrnrlat  latitude of lower left hand corner of the desired map domain (degrees).
#urcrnrlon  longitude of upper right hand corner of the desired map domain (degrees).
#urcrnrlat  latitude of upper right hand corner of the desired map domain (degrees).
#
#or these
#width  width of desired map domain in projection coordinates (meters).
#height     height of desired map domain in projection coordinates (meters).
#lon_0  center of desired map domain (in degrees).
#lat_0  center of desired map domain (in degrees).

# ---------------------------------------------------------------
def plot(mymap, *args, **kwargs) :
    """Plots a custom coastline.  This plots simple lines, not
    ArcInfo-style SHAPE files.

    Args:
        lons: Longitude coordinates for line segments (degrees E)
        lats: Latitude coordinates for line segments (degrees N)

    Type Info:
        len(lons) == len(lats)
        A NaN in lons and lats signifies a new line segment.

    See:
                http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot
        giss.basemap.drawcoastline_file()
    """

    # Convert all arguments from lon/lat to x/y
    ilons = -1
    nargs = []
    i = 0
    while i < len(args) :
        if isinstance(args[i], str) :
            nargs.append(args[i])
            i += 1
            continue
        if ilons < 0 :
            ilons = i
        else :
            lons = args[ilons]
            lats = args[i]

            # Project onto the map
            x, y = mymap(lons, lats)
            x = np.array(x)
            y = np.array(y)

            # BUG workaround: Basemap projects our NaN's to 1e30.
            x[x==1e30] = np.nan
            y[y==1e30] = np.nan

            nargs.append(x)
            nargs.append(y)
            ilons = -1

        i += 1

#   print nargs

    mymap.plot(*tuple(nargs), **kwargs)

# Backward compatibility
plot_lines = plot


# Read "Matlab" format files from NOAA Coastline Extractor.
# See: http://www.ngdc.noaa.gov/mgg/coast/

lineRE=re.compile('(.*?)\s+(.*)')
def read_coastline(fname, take_every=1) :
    nlines = 0
    xdata = array.array('d')
    ydata = array.array('d')
    for line in file(fname) :
#       if (nlines % 10000 == 0) :
#           print 'nlines = %d' % (nlines,)
        if (nlines % take_every == 0 or line[0:3] == 'nan') :
            match = lineRE.match(line)
            lon = float(match.group(1))
            lat = float(match.group(2))

            xdata.append(lon)
            ydata.append(lat)
        nlines = nlines + 1


    return (np.array(xdata),np.array(ydata))

def drawcoastline_file(mymap, fname, **kwargs) :
    """Reads and plots a coastline file.
    See:
        giss.basemap.drawcoastline()
        giss.basemap.read_coastline()
    """

    lons, lats = read_coastline(fname, take_every=1)
    return plot(mymap, lons, lats, **kwargs)
