import unittest
import tempfile
import os
import shutil
import numpy as np
from giss.functional import _arg,bind
import inspect

def fn(a,b,c):
    return a+b+c

class Bind(unittest.TestCase):
    def test_bind(self):
        print(inspect.signature(fn))
        f = bind(fn, 3,4)
        self.assertEqual(12, f(5))
        self.assertEqual(f.attrs['a'],3)
        self.assertEqual(f.attrs['b'],4)
        params = [p.name for p in f.signature.parameters.values()]
        self.assertEqual(['c'], params)

        f = bind(fn, _arg(1), 4, _arg(0))
        self.assertEqual(f.attrs['b'],4)
        params = [p.name for p in f.signature.parameters.values()]
        self.assertEqual(['c', 'a'], params)

        f = bind(fn)
        params = [p.name for p in f.signature.parameters.values()]
        self.assertEqual(0, len(f.attrs))
        self.assertEqual(['a', 'b', 'c'], params)

        f = bind(fn, _arg(1), 7, _arg(0))
        attrs,thunk=f.defer(8,6)
        self.assertEqual({'a': 6, 'b': 7, 'c': 8}, attrs)
        print(attrs)

if __name__ == '__main__':
    unittest.main()

