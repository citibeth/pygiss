import unittest
import giss.util as giutil





class TestLazyDict(unittest.TestCase):

	def test_lazy_dict(self):
		ld = giutil.LazyDict()
		ld['xworld'] = 'world'
		ld.lazy['xbar'] = giutil.CallCounter(lambda: 'bar')

		self.assertEqual('world', ld['xworld'])
		self.assertEqual(0, ld.lazy['xbar'].count)
		self.assertEqual('bar', ld['xbar'])
		self.assertEqual(1, ld.lazy['xbar'].count)
		self.assertEqual('bar', ld['xbar'])
		self.assertEqual(1, ld.lazy['xbar'].count)

		self.assertEqual('world', ld.lazy['xworld']())
		self.assertEqual('bar', ld.lazy['xbar']())


if __name__ == '__main__':
	unittest.main()
