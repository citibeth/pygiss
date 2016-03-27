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

import giss.sparse

# Finite difference helper routines

def center_difference_2(grid, i, c):
    """Produces coefficients for center difference second derivative.
    Formula from SeaRISE website.

    grid: Array of points
    i: Index of point at which to compute FD
    c: Multiply the derivative by this amount
    add_result:
        Function to add to a row of a matrix or vector, sparse or dense"""

    x = grid
    dx1 = x[i] - x[i-1]
    dx2 = x[i+1] - x[i]

    # All our constants together
    K = c * 2.0 / (dx1 * dx2 * (dx1 + dx2))
    return (
        (i+1, K * dx1),
        (i, -K * (dx1 + dx2)),
        (i-1, K * dx2))

def one_sided_difference_2(grid, i, side, c):
    """Computes a 1-sided approximation to d^2 Z/dx^2
    side:
        1 if points i+1, i+2, etc. are to be used
        -1 if points i-1, i-2, etc. are to be used"""

    x = grid
    dx1 = side * (x[i+2*side] - x[i+side])
    dx2 = side * (x[i+side] - x[i])

    K = c * 2.0 / (dx1 * dx2 * (dx1 + dx2))
    return (
        (i,        K * dx1),
        (i+side,  -K * (dx1 + dx2)),
        (i+2*side, K * dx2))

def center_difference_1(grid, i, c):
    x = grid
    dx1 = x[i] - x[i-1]
    dx2 = x[i+1] - x[i]

    dx1_2 = dx1*dx1
    dx2_2 = dx2*dx2

    # All our constants together
    K = c / (dx1 * dx2 * (dx1 + dx2))
    return (
        (i+1,  K * dx1_2),
        (i,    K * (dx2_2 - dx1_2)),
        (i-1, -K * dx2_2))
    
def one_sided_difference_1(grid, i, side, c):
    x = grid
    dx1 = side * (x[i+2*side] - x[i+side])
    dx2 = side * (x[i+side] - x[i])

    dx1_2 = dx1*dx1
    dx2_2 = dx2*dx2

    K = c / (dx1 * dx2 * (dx1 + dx2))
    return (
        (i,       -side*K * (dx1_2 + 2.*dx1*dx2)),
        (i+side,  side*K * (dx1 + dx2) * (dx1 + dx2)),
        (i+2*side, -side*K * dx2_2))

# ----------------------------------------------
def deriv2_matrix(sparse_maker, xx, K=1.):
    nx = len(xx)
    sparse_maker.add_to_row(0, one_sided_difference_2(xx, 0, 1, K))
    for i in xrange(1,nx-1):
        sparse_maker.add_to_row(i, center_difference_2(xx, i, K))
    sparse_maker.add_to_row(nx-1, one_sided_difference_2(xx, nx-1, -1, K))

def deriv1_matrix(sparse_maker, xx, K=1.):
    nx = len(xx)
    sparse_maker.add_to_row(0, one_sided_difference_1(xx, 0, 1, K))
    for i in xrange(1,nx-1):
        sparse_maker.add_to_row(i, center_difference_1(xx, i, K))
    sparse_maker.add_to_row(nx-1, one_sided_difference_1(xx, nx-1, -1, K))
