import unittest
import tempfile
import os
import shutil
import numpy as np
from giss.bind import _arg,bind


def fn(a,b,c):
    return a+b+c

class Bind(unittest.TestCase):
    def test_bind(self):
        f = bind(fn, 3,4,5)
        self.assertEqual(12, f())


if __name__ == '__main__':
    unittest.main()

