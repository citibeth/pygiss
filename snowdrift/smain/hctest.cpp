#include "HeightClassifier.hpp"

using namespace giss;

int main(int argc, char **argv)
{
	int n1 = 10;
	int nhclass = 2;

	// Set up a dense hmax array
	double *hmax0 = new double[n1];	// hc = 0
	double hmax1[n1];					// hc = 1
	for (int i1=0; i1<n1; ++i1) {
		hmax0[i1] = (i1 < 4 ? 1 : 2);
		hmax1[i1] = (i1 < 4 ? 2 : 3);
	}
	printf("hmax0=%p, hmax1=%p\n", hmax0, hmax1);

	// Slice it up and reassemble
	std::vector<blitz::Array<double,1>> height_max;
    blitz::TinyVector<int,1> shape(0); shape[0] = n1;
    blitz::TinyVector<int,1> strides(0); strides[0] = 1; //&hmax(1,0) - &hmax(0,0);
printf("strides[0] = %d\n", strides[0]);
	height_max.push_back(blitz::Array<double,1>(hmax0, shape, strides, blitz::neverDeleteData));
	height_max.push_back(blitz::Array<double,1>(hmax1, shape, strides, blitz::neverDeleteData));

//	for (int hc=0; hc<nhclass; ++hc) {
//		height_max.push_back(blitz::Array<double,1>(&hmax(0, hc), shape, strides, blitz::neverDeleteData));
//	}

	// Try it out!
	HeightClassifier classifier(std::move(height_max));

	int idx = 2;
	for (auto ii=classifier.begin(idx); ii != classifier.end(idx); ++ii) {
		printf("val(idx=%d)=%f\n", -1, *ii);
	}


	for (int i1=0; i1<n1; ++i1) {
		double elevation = 1;
		printf("i1=%d, ele=%f --> %d\n", i1, elevation,
			classifier.get_hclass(i1, elevation));
	}
}

