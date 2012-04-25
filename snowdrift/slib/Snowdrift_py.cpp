#include <Python.h>
#include <arrayobject.h>
#include <math.h>
#include "snowdrift.h"
#include "pyutil.hpp"

using namespace giss;

// ========================================================================
/// Classmembers of the Python class
typedef struct {
	PyObject_HEAD
	giss::Snowdrift *snowdrift_f;	// Fortran-allocated snowdrift pointer
} SnowdriftDict;

// ========= class snowdrift.Snowdrift :

static PyObject *Snowdrift_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	SnowdriftDict *self;

	self = (SnowdriftDict *)type->tp_alloc(type, 0);

    if (self != NULL) {
		self->snowdrift_f = NULL;
    }

    return (PyObject *)self;
}

static int Snowdrift_init(SnowdriftDict *self, PyObject *args, PyObject *kwds)
{

	// Get arguments
	const char *fname;
	if (!PyArg_ParseTuple(args, "s", &fname)) {
		// Throw an exception...
		PyErr_SetString(PyExc_ValueError,
			"Snowdrift_init() called without a valid string as argument.");
		return 0;
	}

printf("Snowdrift_init(%s) called, snowdrift_f=%p\n", fname, self->snowdrift_f);

	// Instantiate pointer
	if (self->snowdrift_f) snowdrift_delete_c_(self->snowdrift_f);
	self->snowdrift_f = snowdrift_new_c_(fname, strlen(fname));


printf("snowdrift_f = %p\n", self->snowdrift_f);

	return 0;
}

static void Snowdrift_dealloc(SnowdriftDict *self)
{
	if (self->snowdrift_f) snowdrift_delete_c_(self->snowdrift_f);
	self->snowdrift_f = NULL;
	self->ob_type->tp_free((PyObject *)self);
}

//static PyMemberDef Snowdrift_members[] = {{NULL}};


static PyObject * Snowdrift_downgrid_snowdrift(SnowdriftDict *self, PyObject *args)
{
	// Get Arguments
	PyArrayObject *Z1;
	PyArrayObject *Z2;
// For some reason, this didn't work.  Maybe a reference count problem.
//	if (!PyArg_ParseTuple(args, "O!O!",
//		&PyArray_Type, &Z1,
//		&PyArray_Type, &Z2))
	if (!PyArg_ParseTuple(args, "OO", &Z1, &Z2))
	{
		return NULL;
	}


	// if (NULL == Z1)  return NULL;
	// if (NULL == Z2)  return NULL;
	
	/* Check that objects are 'double' type and vectors */
	if (!is_doublevector(Z1)) return NULL;
	if (!is_doublevector(Z2)) return NULL;

double *z1 = (double *)Z1->data;
int dim = Z1->dimensions[0];
printf("z1[%d] = %f %f %f...\n", dim, z1[0], z1[1], z1[2]);
printf("snowdrift_f = %p\n", self->snowdrift_f);
	
	int ret = snowdrift_downgrid_snowdrift_c_(self->snowdrift_f,
		(double *)Z1->data, 1, // Z1->dimensions[0],
		(double *)Z2->data, 1); //Z2->dimensions[0]);

	return Py_BuildValue("i", ret);
}

static PyObject * Snowdrift_upgrid(SnowdriftDict *self, PyObject *args)
{
	// Arguments from Python
	PyArrayObject *Z2;
	PyArrayObject *Z1;
	int merge_or_replace;

	/* Parse tuples separately since args will differ between C fcns */
#if 0
	if (!PyArg_ParseTuple(args, "O!O!",
		&PyArray_Type, &Z2,
		&PyArray_Type, &Z1)) return NULL;
#endif
	if (!PyArg_ParseTuple(args, "iOO", &merge_or_replace, &Z2, &Z1)) return NULL;
	
	/* Check that objects are 'double' type and vectors */
	if (!is_doublevector(Z2)) return NULL;
	if (!is_doublevector(Z1)) return NULL;
	
	snowdrift_upgrid_c_(self->snowdrift_f, merge_or_replace,
		(double *)Z2->data, 1, //Z2->dimensions[0],
		(double *)Z1->data, 1); //, Z1->dimensions[0]);

	// Returns a Python None value
	// http://stackoverflow.com/questions/8450481/method-without-return-value-in-python-c-extension-module
	return Py_BuildValue("");
}

static PyObject * Snowdrift_downgrid(SnowdriftDict *self, PyObject *args)
{
	// Arguments from Python
	PyArrayObject *Z2;
	PyArrayObject *Z1;
	int merge_or_replace;

	/* Parse tdownles separately since args will differ between C fcns */
#if 0
	if (!PyArg_ParseTuple(args, "O!O!",
		&PyArray_Type, &Z2,
		&PyArray_Type, &Z1)) return NULL;
#endif
	if (!PyArg_ParseTuple(args, "iOO", &merge_or_replace, &Z2, &Z1)) return NULL;
	
	/* Check that objects are 'double' type and vectors */
	if (!is_doublevector(Z2)) return NULL;
	if (!is_doublevector(Z1)) return NULL;
	
	snowdrift_downgrid_c_(self->snowdrift_f, merge_or_replace,
		(double *)Z2->data, 1, //Z2->dimensions[0],
		(double *)Z1->data, 1); //, Z1->dimensions[0]);

	// Returns a Python None value
	// http://stackoverflow.com/questions/8450481/method-without-return-value-in-python-c-extension-module
	return Py_BuildValue("");
}


static PyObject * Snowdrift_overlap(SnowdriftDict *self, PyObject *args)
{
	// Arguments from Python
	PyArrayObject *densemat;
	
	/* Parse tuples separately since args will differ between C fcns */
	if (!PyArg_ParseTuple(args, "O", &densemat)) return NULL;
	
	/* Check that objects are 'double' type and vectors */
	if (!is_doublevector(densemat)) return NULL;
	
	snowdrift_overlap_c_(self->snowdrift_f,
		(double *)densemat->data,
		densemat->strides[0] / sizeof(double),
		densemat->strides[1] / sizeof(double));

	// Returns a Python None value
	// http://stackoverflow.com/questions/8450481/method-without-return-value-in-python-c-extension-module
	return Py_BuildValue("");
}




static PyMethodDef Snowdrift_methods[] = {
	// {"__init__", Snowdrift_init
	{"downgrid_snowdrift", (PyCFunction)Snowdrift_downgrid_snowdrift, METH_VARARGS,
		"Convert from grid1 to grid2 using Snowdrift regridding method"},
	{"upgrid", (PyCFunction)Snowdrift_upgrid, METH_VARARGS,
		"Convert from grid2 to grid1, simple overlap matrix multiplication"},
	{"downgrid", (PyCFunction)Snowdrift_downgrid, METH_VARARGS,
		"Convert from grid1 to grid2, simple overlap matrix multiplication"},
	{"overlap", (PyCFunction)Snowdrift_overlap, METH_VARARGS,
		"Obtain the overlap matrix (in dense form)"},
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
   (initproc)Snowdrift_init,  /* tp_init */
   0,                         /* tp_alloc */
   (newfunc)Snowdrift_new    /* tp_new */
};

