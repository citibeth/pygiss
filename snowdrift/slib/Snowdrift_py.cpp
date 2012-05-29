#include <Python.h>
#include <arrayobject.h>
#include <math.h>
#include "pyutil.hpp"
#include "Snowdrift.hpp"
#include <blitz/array.h>

using namespace giss;

// ========================================================================
/// Classmembers of the Python class
typedef struct {
	PyObject_HEAD
	giss::Snowdrift *snowdrift;
} SnowdriftDict;

// ========= class snowdrift.Snowdrift :

static PyObject *Snowdrift_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	SnowdriftDict *self;

	self = (SnowdriftDict *)type->tp_alloc(type, 0);

    if (self != NULL) {
		self->snowdrift = NULL;
    }

    return (PyObject *)self;
}

static int Snowdrift__init(SnowdriftDict *self, PyObject *args, PyObject *kwds)
{
	// ========= Parse the arguments
	const char *fname;
	if (!PyArg_ParseTuple(args, "s", &fname)) {
		// Throw an exception...
		PyErr_SetString(PyExc_ValueError, "Snowdrift__init(): invalid arguments.");
		return -1;
	}

	// =========== Instantiate pointer: load main matrix from netCDF file
	if (self->snowdrift) delete self->snowdrift;
	self->snowdrift = new giss::Snowdrift(std::string(fname));

//printf("snowdrift = %p\n", self->snowdrift);

	return 0;
}

static PyObject *Snowdrift_init(SnowdriftDict *self, PyObject *args, PyObject *keywords)
{
	Snowdrift * const sd(self->snowdrift);

//printf("Snowdrift_init1 sd=%p, n1=%d, n2=%d\n", sd, sd->n1, sd->n2);

	// ========= Parse the arguments
	PyArrayObject *elevation_py;
	PyArrayObject *mask_py;
	PyArrayObject *height_max_py;
	char const *problem_file = "";
	char const *sconstraints = "default";

	static char const *keyword_list[] = {"elevation", "mask", "height_max", "problem_file", "constraints", NULL};
	if (!PyArg_ParseTupleAndKeywords(
		args, keywords, "OOO|ss",
		const_cast<char **>(keyword_list),
		&elevation_py, &mask_py, &height_max_py, &problem_file, &sconstraints))
	{
		// Throw an exception...
		PyErr_SetString(PyExc_ValueError, "Snowdrift_init(): invalid arguments."); return NULL;
	}

	// ========== Get a boost::function from the constraints
	boost::function<std::unique_ptr<VectorSparseMatrix> (VectorSparseMatrix &)> constraints;
	if (strcmp(sconstraints, "cesm") == 0) {
		constraints = boost::bind(&Snowdrift::get_constraints_cesm, sd, _1);
	} else {
		constraints = boost::bind(&Snowdrift::get_constraints_default, _1);
	}

	// =========== Typecheck bounds on the arrays
	if (!check_dimensions(elevation_py, "elevation_py", NPY_DOUBLE, {sd->n2})) return NULL;
		auto elevation(py_to_blitz<double,1>(elevation_py));
	if (!check_dimensions(mask_py, "mask_py", NPY_INT, {sd->n2})) return NULL;
		auto mask(py_to_blitz<int,1>(mask_py));
	if (!check_dimensions(height_max_py, "height_max_py", NPY_DOUBLE, {sd->n1, -1})) return NULL;
		auto height_max(py_to_blitz_Z1(height_max_py));

	// ========== Finish initialization
	self->snowdrift->init(elevation, mask, height_max, constraints);
	self->snowdrift->problem_file = std::string(problem_file);

//printf("Snowdrift::init(%s) called, snowdrift=%p\n", fname, self->snowdrift);
printf("snowdrift = %p\n", self->snowdrift);

	return Py_BuildValue("");
}


static void Snowdrift_dealloc(SnowdriftDict *self)
{
	if (self->snowdrift) delete self->snowdrift;
	self->snowdrift = NULL;
	self->ob_type->tp_free((PyObject *)self);
}



static PyObject * Snowdrift_downgrid(SnowdriftDict *self, PyObject *args, PyObject *keywords)
{
	Snowdrift * const sd(self->snowdrift);

	// ======== Parse and typecheck the arguments
	PyArrayObject *Z1_py;
	PyArrayObject *Z2_py;
	Snowdrift::MergeOrReplace merge_or_replace = Snowdrift::MergeOrReplace::MERGE;
	int use_snowdrift = 0;
// For some reason, this didn't work.  Maybe a reference count problem.
//	if (!PyArg_ParseTuple(args, "O!O!",
//		&PyArray_Type, &Z1_py,
//		&PyArray_Type, &Z2_py))
	static char const *keyword_list[] = {"Z1", "Z2", "merge_or_replace", "use_snowdrift", NULL};
	if (!PyArg_ParseTupleAndKeywords(
		args, keywords, "OO|ii",
		const_cast<char **>(keyword_list),
		&Z1_py, &Z2_py, &merge_or_replace, &use_snowdrift)) return NULL;

printf("use_snowdrift = %d\n", use_snowdrift);

	if (!check_dimensions(Z1_py, "Z1_py", NPY_DOUBLE, {sd->n1, sd->num_hclass})) return NULL;
	if (!check_dimensions(Z2_py, "Z2_py", NPY_DOUBLE, {sd->n2})) return NULL;

	std::vector<blitz::Array<double,1>> Z1(py_to_blitz_Z1(Z1_py));
	auto Z2(py_to_blitz<double,1>(Z2_py));

	// ====== Downgrid!
	bool ret = sd->downgrid(Z1, Z2, merge_or_replace, use_snowdrift);

	return Py_BuildValue("i", (int)ret);
}

static PyObject * Snowdrift_upgrid(SnowdriftDict *self, PyObject *args, PyObject *keywords)
{
	Snowdrift * const sd(self->snowdrift);

	// ======== Parse and typecheck the arguments
	PyArrayObject *Z2_py;
	PyArrayObject *Z1_py;
	Snowdrift::MergeOrReplace merge_or_replace = Snowdrift::MergeOrReplace::MERGE;

	static char const *keyword_list[] = {"Z2", "Z1", "merge_or_replace", NULL};
	if (!PyArg_ParseTupleAndKeywords(
		args, keywords, "OO|i",
		const_cast<char **>(keyword_list),
		&Z2_py, &Z1_py, &merge_or_replace)) return NULL;

	if (!check_dimensions(Z2_py, "Z2_py", NPY_DOUBLE, {sd->n2})) return NULL;
	if (!check_dimensions(Z1_py, "Z1_py", NPY_DOUBLE, {sd->n1, sd->num_hclass})) return NULL;

	auto Z2(py_to_blitz<double,1>(Z2_py));
	auto Z1(py_to_blitz_Z1(Z1_py));

	// ====== Upgrid!
	sd->upgrid(Z2, Z1, merge_or_replace);

	// Returns a Python None value
	// http://stackoverflow.com/questions/8450481/method-without-return-value-in-python-c-extension-module
	return Py_BuildValue("");
}



// static PyObject * Snowdrift_overlap(SnowdriftDict *self, PyObject *args)
// {
// 	// Arguments from Python
// 	PyArrayObject *densemat;
// 	
// 	/* Parse tuples separately since args will differ between C fcns */
// 	if (!PyArg_ParseTuple(args, "O", &densemat)) return NULL;
// 	if (!check_dimensions(densemat, NPY_DOUBLE, {sd->n1, sd->n2})) return NULL;
// 
// 
// 	
// 	/* Check that objects are 'double' type and vectors */
// 	if (!is_doublevector(densemat)) return NULL;
// 	
// 	snowdrift_overlap_c_(self->snowdrift_f,
// 		(double *)densemat->data,
// 		densemat->strides[0] / sizeof(double),
// 		densemat->strides[1] / sizeof(double));
// 
// 	// Returns a Python None value
// 	// http://stackoverflow.com/questions/8450481/method-without-return-value-in-python-c-extension-module
// 	return Py_BuildValue("");
// }




static PyMethodDef Snowdrift_methods[] = {
	{"init", (PyCFunction)Snowdrift_init, METH_KEYWORDS,
		"Set up elevation classes and land mask"},
//	{"info", (PyCFunction)Snowdrift_info, METH_VARARGS,
//		"Stuff a bunch of bounds, etc. into a dictionary"},
	{"upgrid", (PyCFunction)Snowdrift_upgrid, METH_VARARGS,
		"Convert from grid2 to grid1, simple overlap matrix multiplication"},
	{"downgrid", (PyCFunction)Snowdrift_downgrid, METH_KEYWORDS,
		"Convert from grid1 to grid2, simple overlap matrix multiplication"},
//	{"overlap", (PyCFunction)Snowdrift_overlap, METH_VARARGS,
//		"Obtain the overlap matrix (in dense form)"},
	{NULL}     /* Sentinel - marks the end of this structure */
};

PyTypeObject SnowdriftType = {
   PyObject_HEAD_INIT(NULL)
   0,                         /* ob_size */
   "Snowdrift",               /* tp_name */
   sizeof(SnowdriftDict),     /* tp_basicsize */
   0,                         /* tp_itemsize */
   (destructor)Snowdrift_dealloc, /* tp_dealloc */
   0,                         /* tp_print */
   0,                         /* tp_getattr */
   0,                         /* tp_setattr */
   0,                         /* tp_compare */
   0,                         /* tp_repr */
   0,                         /* tp_as_number */
   0,                         /* tp_as_sequence */
   0,                         /* tp_as_mapping */
   0,                         /* tp_hash */
   0,                         /* tp_call */
   0,                         /* tp_str */
   0,                         /* tp_getattro */
   0,                         /* tp_setattro */
   0,                         /* tp_as_buffer */
   Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags*/
   "Snowdrift object",        /* tp_doc */
   0,                         /* tp_traverse */
   0,                         /* tp_clear */
   0,                         /* tp_richcompare */
   0,                         /* tp_weaklistoffset */
   0,                         /* tp_iter */
   0,                         /* tp_iternext */
   Snowdrift_methods,         /* tp_methods */
//   Snowdrift_members,         /* tp_members */
   0,                         /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)Snowdrift__init,  /* tp_init */
   0,                         /* tp_alloc */
   (newfunc)Snowdrift_new    /* tp_new */
};

