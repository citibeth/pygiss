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

import scipy.sparse

class SparseMaker(object):
    def __init__(self, shape, dtype='d'):
        self.shape = shape
        self.dtype = dtype
        self.rows = []
        self.cols = []
        self.data = []

    def add(self, row, col, x):
        self.rows.append(row)
        self.cols.append(col)
        self.data.append(x)

    def clear(self, indices_set):
        """Clears all entries in the set of indices"""
        for i in xrange(len(self.data)):
            if (self.rows[i], self.cols[i]) in indices_set:
                self.data[i] = 0

    def clear_rows(self, row_set):
        """Clears all entries in the set of indices"""
        for i in xrange(len(self.data)):
            if self.rows[i] in row_set:
                self.data[i] = 0

    def clear_cols(self, col_set):
        """Clears all entries in the set of indices"""
        for i in xrange(len(self.data)):
            if self.cols[i] in col_set:
                self.data[i] = 0



    def construct_coo(self):
        return scipy.sparse.coo_matrix(
            (self.data, (self.rows, self.cols)),
            shape=self.shape, dtype=self.dtype)

    def construct_csr(self):
        return scipy.sparse.csr_matrix(
            (self.data, (self.rows, self.cols)),
            shape=shape, dtype=self.dtype)

    def add_to_row(mat, row, ivpairs):
        """For use with giss.fd"""
        for i,x in ivpairs:
            mat.add(row,i,x)

    def add_to_col(mat, col, ivpairs):
        """For use with giss.fd"""
        for i,x in ivpairs:
            mat.add(i,col,x)




def identity_matrix(sparse_maker, K=1.):
    nx = min(sparse_maker.shape)
    for i in xrange(0, nx):
        sparse_maker.add(i,i,K)
