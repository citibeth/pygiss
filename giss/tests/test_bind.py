import unittest
import tempfile
import os
import shutil
import numpy as np
from giss.functional import _arg,bind
import inspect

def fn(a,b,c):
    return a+b+c

class TestBind(unittest.TestCase):
    def test_bind(self):
        print(inspect.signature(fn))
        f = bind(fn, 3,4)
        self.assertEqual(12, f(5))

        f = bind(fn, _arg(1), 4, _arg(0))
        self.assertEqual(12, f(5,3))

if __name__ == '__main__':
    unittest.main()

