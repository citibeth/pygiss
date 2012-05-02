#pragma once

#include "hsl_zd11_x.hpp"

namespace giss {

class QPT_problem_f;		// Fortran type, opaque
class QPT_problem;			// C++ class

}

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
	double * const C;		// double[m] RHS of equality constraints
	double * const C_l;		// double[m] Lower bound, RHS of inequality constraints
	double * const C_u;		// double[m] Upper bound, RHS of inequality constraints
	double * const X;		// double[n] Value of variables (input & output)
	double * const Y;		// double[m]
	double * const Z;		// double[n]

	ZD11 A;				// m*n Constraints matrix
	ZD11 H;				// n*n Hessian (quadratic) matrix

	/** @param A_ne Number of elements in the constraint matrix */
	QPT_problem(
		int m, int n,
		int A_ne, int H_ne, bool eqp) :
	main(*(QPT_problem_f *)0),
	m(*(int *)0),
	n(*(int *)0),
	f(*(double *)0),
	G(0), X_l(0), X_u(0), C(0), C_l(0), C_u(0), X(0), Y(0), Z(0)
	{
		QPT_problem_f *main = qpt_problem_new_c_();
		qpt_problem_c_init_(this, main, m, n, A_ne, H_ne, eqp);
	}

	~QPT_problem()
	{
		qpt_problem_delete_c_(&main);
	}
};

}	// namespace giss
