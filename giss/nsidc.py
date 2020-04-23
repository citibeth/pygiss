from giss import ioutil,giutil,iopfile
import re,io,os
import datetime
import gdal
from cdo import Cdo
cdo = Cdo()

"""Parsers and formatters for NSIDC file sets"""


class PFile_0481(iopfile.PFile):

    key_fn = lambda x: (x['source'], x['grid'], x['startdate'], x['enddate'],
        x['parameter'], x['nominal_time'], x['version'], x['ext'])


    def format(self, **overrides):
        # Override self with overrides
        pfile = giutil.merge_dicts(self, overrides)

        pfile['sstartdate'] = datetime.datetime.strftime(pfile['startdate'], '%d%b%y')
        pfile['senddate'] = datetime.datetime.strftime(pfile['enddate'], '%d%b%y')
        pfile['snominal_time'] = '{:02d}-{:02d}-{:02d}'.format(*pfile['nominal_time'])
        # Override ext with user-given value
        if pfile['parameter'] == '':
            fmt = '{source}_{grid}_{sstartdate}_{senddate}_{snominal_time}_v{version}{ext}'
        else:
            fmt = '{source}_{grid}_{sstartdate}_{senddate}_{snominal_time}_{parameter}_v{version}{ext}'

        return fmt.format(**pfile)

imonth = { 'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7,
    'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}


reNSIDC_0481 = re.compile(r'(TSX|TDX)_([EWS][0-9.]+[NS])_(\d\d(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\d\d)_(\d\d(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\d\d)_(\d\d)-(\d\d)-(\d\d)(_(vv|vx|vy|ex|ey)?)_v([0-9.]+)(\..+)')

def parse_0481(path):
    """
    See: https://nsidc.org/data/nsidc-0481"""

    dir,leaf = os.path.split(path)

    match = reNSIDC_0481.match(leaf)
    if match is None:
        return None

    sstartdate = match.group(3)
    senddate = match.group(5)
    ret = PFile_0481(
        dir=dir,
        leaf=leaf,
        source=match.group(1),
        grid=match.group(2),
        startdate=datetime.datetime.strptime(sstartdate, "%d%b%y"),
        enddate=datetime.datetime.strptime(senddate, "%d%b%y"),
        nominal_time=(int(match.group(7)), int(match.group(8)), int(match.group(9))),
        parameter=match.group(11),   # Could be None
        version=match.group(12),
        ext=match.group(13))

    if ret['parameter'] is None:
        ret['parameter'] = ''    # Don't like None for sorting

    return ret

# -------------------------------------------------------------
def tiff_to_netcdf(pfile, odir, all_files=None, oext='.nc', reftime='2008-01-01'):
    """Converts single GeoTIFF to NetCDF
    ifname:
        The input GeoTIFF file
    oext:
        Extension to use on the output file
    Returns:
        Name of the output NetCDF file (as a pfile parsed file)"""

    os.makedirs(odir, exist_ok=True)

    # Generate ofname
    opfile = type(pfile)(pfile.items())
    opfile['dir'] = odir
    opfile['ext'] = oext
    tmp0 = os.path.join(odir, opfile.format(ext='.tiff_to_netcdf_0.nc'))

    # Don't regenerate files already built
    if not ioutil.needs_regen((opfile.path,), (pfile.path,)):
        return opfile

    try:
        print("Converting {} to {}".format(pfile.path, opfile.path))
        # use gdal's python binging to convert GeoTiff to netCDF
        # advantage of GDAL: it gets the projection information right
        # disadvantage: the variable is named "Band1", lacks metadata
        ds = gdal.Open(pfile.path)
        ds = gdal.Translate(tmp0, ds)
        ds = None

        # This deduces the mid-point (nominal) date from the filename
        nominal_date = pfile['startdate'] + (pfile['enddate'] - pfile['startdate']) / 2

        # Set the time axis
        var = pfile['parameter']
        inputs = [
            f'-setreftime,{reftime}',
            f'-setattribute,{var}@units="m year-1"',
            f'-chname,Band1,{var}',
            f'{tmp0}']
        cdo.settaxis(
            nominal_date.isoformat(),
            input=' '.join(inputs),
            output=opfile.path,
            options="-f nc4 -z zip_2")

        if all_files is not None:
            all_files.append(opfile.path)
        return opfile
    finally:
        try:
            os.remove(tmp0)
        except FileNotFoundError:
            pass
