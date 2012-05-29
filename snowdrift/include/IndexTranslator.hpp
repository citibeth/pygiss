#pragma once

#include <vector>
#include <set>
#include <cstdio>

namespace giss {

class IndexTranslator {
	std::vector<int> _a2b;
	std::vector<int> _b2a;
public:
	/** @param n Size of space a (runs [0...n-1])
	@param used Set of indices that are used in space a */
	void init(int size_a, std::set<int> const &used);

	int na() const { return _a2b.size(); }
	int nb() const { return _b2a.size(); }

	int a2b(int a, bool check_result = true) const;
	int b2a(int b, bool check_result = true) const;
};



inline int IndexTranslator::a2b(int a, bool check_result) const {
	if (a < 0 || a >= _a2b.size()) {
		fprintf(stderr, "a=%d is out of range (%d, %d)\n", a, 0, _a2b.size());
		throw std::exception();
	}
	int b = _a2b[a];
	if (check_result && b < 0) {
		fprintf(stderr, "a=%d produces invalid b=%d\n", a, b);
		throw std::exception();
	}
	return b;
}

inline int IndexTranslator::b2a(int b, bool check_result) const {
	if (b < 0 || b >= _b2a.size()) {
		fprintf(stderr, "b=%d is out of range (%d, %d)\n", b, 0, _b2a.size());
		throw std::exception();
	}
	int a = _b2a[b];
	if (check_result && a < 0) {
		fprintf(stderr, "b=%d produces invalid a=%d\n", b, a);
		throw std::exception();
	}
	return a;
}










}

#if 0
class RowColTranslator {
public :
	IndexTranslator row;
	IndexTranslator col;

	void init(
		int nrow, std::set<int> const &used_row,
		int ncol, std::set<int> const &used_col)
	{
		row(nrow, used_row);
		col(ncol, used_col) {}
}
// ------------------------------------------------------------
#endif