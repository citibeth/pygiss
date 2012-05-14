#pragma once

#include <boost/function.hpp>
#include "hsl_zd11_x.hpp"

namespace giss {

class QPT_problem_f;		// Fortran type, opaque
class QPT_problem;			// C++ class

}

class NcFile;

// =============== Fortran Subroutines
extern "C" giss::QPT_problem_f *qpt_problem_new_c_();
extern "C" void qpt_problem_delete_c_(giss::QPT_problem_f *ptr);
extern "C" void qpt_problem_c_init_(
		giss::QPT_problem *self, giss::QPT_problem_f *main,
		int m, int n,
		int A_ne, int H_ne, int eqp_bool);
// =============== C++ Peer Classes

namespace giss {

class QPT_problem {
public :
	QPT_problem_f &main;
	int &m;					// Number of constraints
	int &n;					// Number of variables
	double &f;				// constant term in objective function
	double * const G;		// double[n] Linear term of objective function
	double * const X_l;		// double[n] Lower bound on variables
	double * const X_u;		// double[n] Upper bound on variables
	double * const C_l;		// double[m] Lower bound, RHS of inequality constraints
	double * const C_u;		// double[m] Upper bound, RHS of inequality constraints
	double * const C;		// double[m] RHS of equality constraints
	double * const X;		// double[n] Value of variables (input & output)
	double * const Y;		// double[m]
	double * const Z;		// double[n]

	ZD11 A;				// m*n Constraints matrix
	ZD11 H;				// n*n Hessian (quadratic) matrix

	/** @param A_ne Number of elements in the constraint matrix */
	QPT_problem(
		int m, int n,
		int A_ne, int H_ne, bool eqp);

	~QPT_problem();

	boost::function<void()> netcdf_define(NcFile &nc, std::string const &vname);
};

}	// namespace giss
