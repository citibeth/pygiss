#include <boost/bind.hpp>
#include <netcdfcpp.h>
#include "ncutil.hpp"
#include "qpt_x.hpp"

namespace giss {


/** @param A_ne Number of elements in the constraint matrix */
QPT_problem::QPT_problem(
	int m, int n,
	int A_ne, int H_ne, bool eqp) :
main(*(QPT_problem_f *)0),
m(*(int *)0),
n(*(int *)0),
f(*(double *)0),
G(0), X_l(0), X_u(0), C(0), C_l(0), C_u(0), X(0), Y(0), Z(0)
{
	QPT_problem_f *main = qpt_problem_new_c_();
printf("qpt_x: A_ne=%d\n", A_ne);
	int eqp_bool = eqp;
	qpt_problem_c_init_(this, main, m, n, A_ne, H_ne, eqp_bool);
}

QPT_problem::~QPT_problem()
{
	qpt_problem_delete_c_(&main);
}

static void netcdf_write(QPT_problem *qp,
	NcFile *nc, std::string const &vname,
	boost::function<void()> const &A_write,
	boost::function<void()> const &H_write)
{
	
	auto fvar = nc->get_var((vname + ".f").c_str());
	fvar->put(&qp->f, 1);
	auto Gvar = nc->get_var((vname + ".G").c_str());
	Gvar->put(qp->G, qp->n);
#if 0	// Not in EQP
	nc->get_var((vname + ".X_l").c_str());
	nc->get_var((vname + ".X_u").c_str());
	nc->get_var((vname + ".C_l").c_str());
	nc->get_var((vname + ".C_u").c_str());
#endif
	auto Cvar = nc->get_var((vname + ".C").c_str());
	Cvar->put(qp->C, qp->m);
	auto Xvar = nc->get_var((vname + ".X").c_str());
	Xvar->put(qp->X, qp->n);
	auto Yvar = nc->get_var((vname + ".Y").c_str());
	Yvar->put(qp->Y, qp->m);
	auto Zvar = nc->get_var((vname + ".Z").c_str());
	Zvar->put(qp->Z, qp->n);

	A_write();
	H_write();
}

boost::function<void()> QPT_problem::netcdf_define(NcFile &nc, std::string const &vname)
{
	auto mDim = nc.add_dim((vname + ".m").c_str(), this->m);
	auto nDim = nc.add_dim((vname + ".n").c_str(), this->n);
	auto oneDim = get_or_add_dim(nc, "one", 1);

	nc.add_var((vname + ".f").c_str(), ncDouble, oneDim);
	nc.add_var((vname + ".G").c_str(), ncDouble, nDim);
#if 0	// Not in EQP
	nc.add_var((vname + ".X_l").c_str(), ncDouble, nDim);
	nc.add_var((vname + ".X_u").c_str(), ncDouble, nDim);
	nc.add_var((vname + ".C_l").c_str(), ncDouble, mDim);
	nc.add_var((vname + ".C_u").c_str(), ncDouble, mDim);
#endif
	nc.add_var((vname + ".C").c_str(), ncDouble, mDim);
	nc.add_var((vname + ".X").c_str(), ncDouble, nDim);
	nc.add_var((vname + ".Y").c_str(), ncDouble, mDim);
	nc.add_var((vname + ".Z").c_str(), ncDouble, nDim);

	auto A_write = this->A.netcdf_define(nc, vname + ".A");
	auto H_write = this->H.netcdf_define(nc, vname + ".H");

	return boost::bind(&netcdf_write, this, &nc, vname, A_write, H_write);
}

}
