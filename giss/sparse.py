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
