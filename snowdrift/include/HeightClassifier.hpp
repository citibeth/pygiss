#pragma once

#include <vector>
#include <blitz/array.h>

namespace giss {

class HeightClassifier {
public :
	typedef std::vector<blitz::Array<double,1>> HeightMaxArray;

private:
	HeightMaxArray *height_max;

	class iterator {
		HeightMaxArray::iterator ii;
		int idx;

	public:
		typedef std::random_access_iterator_tag iterator_category;
		typedef double value_type;
		typedef int difference_type;
		typedef double *pointer;
		typedef double &reference;

		double &operator*() { return (*ii)(idx); }
		void operator++() { ++ii; }
		void operator+=(int n) { ii += n; }
		void operator-=(int n) { ii -= n; }
		int operator-(iterator const &rhs)
			{ return &*ii - &*rhs.ii; }
//		bool operator<(iterator const &rhs) { return ii < rhs.ii; }

		iterator() : idx(-1) {}
		bool operator==(iterator const &rhs) { return ii == rhs.ii; }
		bool operator!=(iterator const &rhs) { return !(*this == rhs); }
		iterator &operator=(iterator const &rhs) { ii=rhs.ii; idx=rhs.idx; return *this; }
		iterator(HeightMaxArray::iterator _ii, int _idx) : 
			ii(_ii), idx(_idx) {}
		iterator(iterator const &rhs) : ii(rhs.ii), idx(rhs.idx) {}
	};
public :
	iterator begin(int idx) { return iterator(height_max->begin(), idx); }
	iterator end(int idx) { return iterator(height_max->end(), idx); }

	size_t num_hclass() { return (int)height_max->size(); }

	/** Number of GCM grid cells */
	size_t size() { return (*height_max)[0].extent(0); }

	/** Our lifetime must be shorter than that of _height_max.  That is OK,
	typically HeightClassifier is just a local-var decorator used to
	access height_max. */
	HeightClassifier(std::vector<blitz::Array<double,1>> *_height_max);
	int get_hclass(int idx, double elevation);
};

}
