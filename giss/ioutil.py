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

import contextlib
import sys
import os

# http://stackoverflow.com/questions/13250050/redirecting-the-output-of-a-python-function-from-stdout-to-variable-in-python
@contextlib.contextmanager
def redirect(out=sys.stdout, err=sys.stderr):
    """A context manager that redirects stdout and stderr"""
    saved = (sys.stdout, sys.stderr)
    sys.stdout = out
    sys.stderr = err
    try:
        yield
    finally:
        sys.stdout, sys.stderr = saved

@contextlib.contextmanager
def pushd(path):
    """A context manager which changes the working directory to the given
    path, and then changes it back to its previous value on exit.

    """
    prev_cwd = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(prev_cwd)


