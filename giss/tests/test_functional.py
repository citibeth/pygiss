import unittest
import tempfile
#import inspect
from giss.functional import *

def fn(a,b,c):
    return a+b+c

class TestBind(unittest.TestCase):
    def test_bind(self):
        #print(inspect.signature(fn))
        f = bind(fn, 3,4)
        self.assertEqual(12, f(5))

        f = bind(fn, _arg(1), 4, _arg(0))
        self.assertEqual(12, f(5,3))

@function()
def times(x,n):
    return wrap_value(x*n)

@function()
def plus(x,n):
    return wrap_value(x+n)


class TestTuple(unittest.TestCase):
    def test_tuple(self):
        times2 = bind(times, _arg(0), 2)
        plus2 = bind(plus, _arg(0), 2)
        times3 = bind(times, _arg(0), 3)
        plus3 = bind(plus, _arg(0), 3)

        both = tuplex((times, plus))
        both2 = bind(both, _arg(0), 2)
        both3 = bind(both, _arg(0), 3)

        self.assertEqual(both2(17)(),
            tuplex((times2,plus2))(17)())

        times5 = times2 + times3
        plus5 = plus2 + plus3

        both5 = both2 + both3
        both5b = tuplex((times5, plus5))

        self.assertEqual(both5(17)[0](), both5(17)()[0])
        self.assertEqual(both5(17)[1](), both5(17)()[1])
        self.assertEqual(both5b(17)[0](), both5b(17)()[0])
        self.assertEqual(both5b(17)[1](), both5b(17)()[1])

        self.assertEqual(both5(17)(),
            tuplex((times5,plus5))(17)())

        self.assertEqual(both5b(17)(), both5(17)())

        x = both5b(17)()
        self.assertEqual((85,39), x)

    def test_namedtuple(self):
        MyTuple = namedtuplex('MyTuple', ('times', 'plus'))

        times2 = bind(times, _arg(0), 2)
        plus2 = bind(plus, _arg(0), 2)
        times3 = bind(times, _arg(0), 3)
        plus3 = bind(plus, _arg(0), 3)

        both = MyTuple(times, plus)
        both2 = bind(both, _arg(0), 2)
        both3 = bind(both, _arg(0), 3)

        self.assertEqual(both2(17)(),
            MyTuple(times2,plus2)(17)())

        times5 = times2 + times3
        plus5 = plus2 + plus3

        both5 = both2 + both3
        both5b = MyTuple(times5, plus5)

        self.assertEqual(both5(17).times(), both5(17)().times)
        self.assertEqual(both5(17).plus(), both5(17)().plus)
        self.assertEqual(both5b(17).times(), both5b(17)().times)
        self.assertEqual(both5b(17).plus(), both5b(17)().plus)

        self.assertEqual(both5(17)(),
            MyTuple(times5,plus5)(17)())

        self.assertEqual(both5b(17)(), both5(17)())

        x = both5b(17)()
        self.assertEqual(x.times, 85)
        self.assertEqual(x.plus, 39)

if __name__ == '__main__':
    unittest.main()

