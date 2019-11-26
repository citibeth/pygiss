#import requests
import subprocess
import re
import os
import contextlib
import sys
import tempfile
import shutil

# https://www.nccs.nasa.gov/nccs-users/instructional/ftp-deprecation

hrefRE = re.compile(r'<a\s+href="([^"]*)">', flags=re.MULTILINE)

@contextlib.contextmanager
def tempdir(prefix='tmp'):
    """A context manager for creating and then deleting a temporary directory."""
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir)

def dir_to_url(dir, trailing=False):
    url = PORTAL_ROOT + dir
    if trailing and url[-1] != '/':
        url = url + '/'
    return url

headerRE = re.compile(r'HTTP/[^\s]+\s*(\d+)\s*(.*)')
def curl(curl_args):
    """Run a curl command
    curl_args:
        Arguments on the curl command line
    header_file:
        Name of (temporary) header file to create
    Returns:
        stdout
    """
    with tempdir() as tmp:
        header_file = os.path.join(tmp, '_header')

        cmd = ['curl', '-D', header_file] + list(curl_args)
        # print(' '.join(cmd))
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        stdout, stderr = proc.communicate()
        try:
            proc.wait()

            # Parse the header
            with open(header_file) as fin:
                line = next(fin)
                match = headerRE.match(line)
                http_status = int(match.group(1))
                http_msg = match.group(2)

            if (http_status != 200):
                sys.stderr.write(' '.join(cmd)+'\n')
                sys.stderr.write(stderr.decode())
                raise FileNotFoundError('[curl errno {}]: ({})'.format(http_status, http_msg))

            return stdout
        except:
            sys.stderr.write(' '.join(cmd)+'\n')
            sys.stderr.write(stderr.decode())
            raise




#def list_hrefs(dir):
#    """dir:
#        Directory on portal to list; must start with /"""
#    with tempdir() as tmp:
#        cmd = ['--location', dir_to_url(dir), '-D', os.path.join(tmp, 'header')]
#        with open(os.devnull, 'w') as FNULL:
#            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#    html = proc.stdout.read().decode()
#    err = proc.stdout.read().decode()
#    proc.wait()    # Ensure we get a returncode
#    with open(os.path.join(tmp,'header')) as fin:
#        header = fin.read()
#
#    # FileNotFoundError: [Errno 2] No such file or directory: 'xxx'
#
#    for matches in hrefRE.finditer(html):
##        requests.get(url, allow_redirects=True).content):
#        yield matches.group(1)

PORTAL_ROOT = 'https://portal.nccs.nasa.gov/datashare'
def list_hrefs(dir):
    """dir:
        Directory on portal to list; must start with /"""
    html = curl(['--location', dir_to_url(dir, trailing=True)]).decode()

    for matches in hrefRE.finditer(html):
        yield matches.group(1)

def list_files(dir):
    state = 0
    for href in list_hrefs(dir):
        if state==0:
            if href.startswith('/datashare'):
                state=1
                continue
        if state==1:
            if href.startswith('http://'):
                return
            yield href



class NasaPortalEntry(object):
    """Compatibility with DirEntry returned by os.scandir() used in os.walk()"""
    def __init__(self, dir, fname):
        if fname[-1] == '/':
            self.name = fname[:-1]
            self._is_dir = True
        else:
            self.name = fname
            self._is_dir = False
        self.path = '{}/{}'.format(dir, self.name)

    def is_dir(self):
        return self._is_dir
    def is_symlink(self):
        return False

    def __repr__(self):
        return 'nasa:{}'.format(self.path)

def list_entries(dir):
    for fname in list_files(dir):
        yield NasaPortalEntry(dir, fname)

# =====================================================================

# NOTE: This subroutine is copied from CPython's os.walk()
# See licensing at: https://github.com/python/cpython/blob/master/LICENSE
def walk(top, topdown=True, onerror=None, followlinks=False):
    """Directory tree generator.
    For each directory in the directory tree rooted at top (including top
    itself, but excluding '.' and '..'), yields a 3-tuple
        dirpath, dirnames, filenames
    dirpath is a string, the path to the directory.  dirnames is a list of
    the names of the subdirectories in dirpath (excluding '.' and '..').
    filenames is a list of the names of the non-directory files in dirpath.
    Note that the names in the lists are just names, with no path components.
    To get a full path (which begins with top) to a file or directory in
    dirpath, do os.path.join(dirpath, name).
    If optional arg 'topdown' is true or not specified, the triple for a
    directory is generated before the triples for any of its subdirectories
    (directories are generated top down).  If topdown is false, the triple
    for a directory is generated after the triples for all of its
    subdirectories (directories are generated bottom up).
    When topdown is true, the caller can modify the dirnames list in-place
    (e.g., via del or slice assignment), and walk will only recurse into the
    subdirectories whose names remain in dirnames; this can be used to prune the
    search, or to impose a specific order of visiting.  Modifying dirnames when
    topdown is false has no effect on the behavior of os.walk(), since the
    directories in dirnames have already been generated by the time dirnames
    itself is generated. No matter the value of topdown, the list of
    subdirectories is retrieved before the tuples for the directory and its
    subdirectories are generated.
    By default errors from the os.scandir() call are ignored.  If
    optional arg 'onerror' is specified, it should be a function; it
    will be called with one argument, an OSError instance.  It can
    report the error to continue with the walk, or raise the exception
    to abort the walk.  Note that the filename is available as the
    filename attribute of the exception object.
    By default, os.walk does not follow symbolic links to subdirectories on
    systems that support them.  In order to get this functionality, set the
    optional argument 'followlinks' to true.
    Caution:  if you pass a relative pathname for top, don't change the
    current working directory between resumptions of walk.  walk never
    changes the current directory, and assumes that the client doesn't
    either.
    Example:
    import os
    from os.path import join, getsize
    for root, dirs, files in os.walk('python/Lib/email'):
        print(root, "consumes", end="")
        print(sum(getsize(join(root, name)) for name in files), end="")
        print("bytes in", len(files), "non-directory files")
        if 'CVS' in dirs:
            dirs.remove('CVS')  # don't visit CVS directories
    """
#    top = fspath(top)
    dirs = []
    nondirs = []
    walk_dirs = []

    # We may not have read permission for top, in which case we can't
    # get a list of the files the directory contains.  os.walk
    # always suppressed the exception then, rather than blow up for a
    # minor reason when (say) a thousand readable directories are still
    # left to visit.  That logic is copied here.
    try:
        # Note that scandir is global in this module due
        # to earlier import-*.
#        scandir_it = scandir(top)
        scandir_it = list_entries(top)
    except OSError as error:
        if onerror is not None:
            onerror(error)
        return

    for entry in scandir_it:
        if entry.is_dir():
            dirs.append(entry.name)
        else:
            nondirs.append(entry.name)

        walk_dirs.append(entry.path)

    # Yield before recursion if going top down
    if topdown:
        yield top, dirs, nondirs

        # Recurse into sub-directories
#        islink, join = path.islink, path.join
        islink = lambda x : False
        join = '/'.join
        for dirname in dirs:
            new_path = join((top, dirname))
            # Issue #23605: os.path.islink() is used instead of caching
            # entry.is_symlink() result during the loop on os.scandir() because
            # the caller can replace the directory entry during the "yield"
            # above.
            if followlinks or not islink(new_path):
                yield from walk(new_path, topdown, onerror, followlinks)
    else:
        # Recurse into sub-directories
        for new_path in walk_dirs:
            yield from walk(new_path, topdown, onerror, followlinks)
        # Yield after recursion if going bottom up
        yield top, dirs, nondirs

def download(src, odir=None, ofile=None):
    """Copies file from NASA portal to destination directory on local machine.
    This is atomic; partial files will not be left around if it fails in the middle.
    src:
        Name of source file on server.  Must start with /.
    odir:
        Directory on local machine where file will be stored (under the same name)."""
    url = dir_to_url(src)
    leaf = src.split('/')[-1]
    if ofile is None:
        ofile = os.path.join(odir, leaf)
    # sys.stderr.write('Downloading {} to {}\n'.format(url, ofile))
    tmpout = ofile + '.download'
    try:
        _ = curl(['--create-dirs', '--output', tmpout, url])
        os.rename(tmpout, ofile)
    finally:
        try:
            os.remove(tmpout)
        except:
            pass



# =============================================================
# Poor man's testing code
#download('/GISS/Ent_TBM/EntGVSD/data/lc_crops/Monfreda2008/5min/Monfreda_crops_5min.bin.gz', 'x/y')

#for root, dirs, files in walk("/GISS/Ent_TBM/EntGVSD/data"):
#   for name in files:
#      print(os.path.join(root, name))
#
#
#with tempdir() as tmp:
#    print(tmp)
#    x = curl([PORTAL_ROOT+'/','--location', '--output', 'x'], os.path.join(tmp, '_head'))

#   for name in dirs:
#      print(os.path.join(root, name))


#print(list(list_files('/GISS/Ent_TBM/EntGVSD/inputs/lai/BNU/doy/2004')))
#print(list(repr(x) for x in list_entries('/GISS/Ent_TBM/EntGVSD/inputs/lai/BNU/doy/2004')))
