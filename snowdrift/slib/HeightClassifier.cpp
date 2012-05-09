#include "HeightClassifier.hpp"
#include <algorithm>

namespace giss {

HeightClassifier::HeightClassifier(
	std::vector<blitz::Array<double,1>> &&_height_max)
{
	height_max = std::move(_height_max);
}

int HeightClassifier::get_hclass(int idx, double elevation)
{
	iterator begin(this->begin(idx));
	iterator end(this->end(idx));
	end -= 1;		// Last height class always ends at infinity
	iterator top(std::upper_bound(begin, end, elevation));

	int hc = top - begin;
//	if (hc < 0) hc = 0;		// Redundant
//	if (hc >= num_hclass) hc = num_hclass - 1;
	return hc;
}

}
