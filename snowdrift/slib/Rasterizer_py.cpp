#include <Python.h>
#include <arrayobject.h>
#include <math.h>
#include "Rasterizer.hpp"
#include "Grid_py.hpp"
#include "Rasterizer_py.hpp"
#include "pyutil.hpp"

using namespace giss;

// ========================================================================

// ========= class snowdrift.Rasterizer :
static PyObject *Rasterizer_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	RasterizerDict *self;

	self = (RasterizerDict *)type->tp_alloc(type, 0);

    if (self != NULL) {
		self->rasterizer = NULL;
    }

    return (PyObject *)self;
}

static int Rasterizer__init(RasterizerDict *self, PyObject *args, PyObject *kwds)
{
	// Get arguments
	GridDict *grid;
	double x0, x1; int nx;	// Grid to rasterize to
	double y0, y1; int ny;

	if (!PyArg_ParseTuple(args, "Oddiddi", &grid,
		&x0, &x1, &nx,
		&y0, &y1, &ny))
	{
		// Throw an exception...
		PyErr_SetString(PyExc_ValueError,
			"Rasterizer__init() called without a valid string as argument.");
		return 0;
	}

	// Instantiate pointer
	if (self->rasterizer) delete self->rasterizer;
	self->rasterizer = new Rasterizer(grid->grid, x0, x1, nx, y0, y1, ny);
fprintf(stderr, "Rasterizer_new() returns %p\n", self->rasterizer);

	return 0;
}

static void Rasterizer_dealloc(RasterizerDict *self)
{
//fprintf(stderr, "Rasterizer_dealloc(%p)\n", self->rasterizer);
printf("Rasterizer_dealloc(self=%p, rasterizer=%p)\n", self, self->rasterizer);
	if (self->rasterizer) delete self->rasterizer;
	self->rasterizer = NULL;
	self->ob_type->tp_free((PyObject *)self);
}

//static PyMemberDef Rasterizer_members[] = {{NULL}};

PyObject *rasterize(PyObject *self, PyObject *args)
{
	// Get Arguments
	RasterizerDict *rast;
	PyArrayObject *data_py;
	PyArrayObject *out_py;
	if (!PyArg_ParseTuple(args, "OOO",
		&rast, &data_py, &out_py))
	{
		return NULL;
	}

	// --------- Check array sizes and types
	// out must have dimensions (nx, ny) and be double
	// data has single dimension, based on highest index possible
	// strides must be mulitple of sizeof(double)
//	if (!check_dimensions(data_py, "data_py", NPY_DOUBLE, {rast->rasterizer->
	auto data(py_to_blitz<double,1>(data_py));
	auto out(py_to_blitz<double,2>(out_py));

	rasterize(*rast->rasterizer, data, out);

	return Py_BuildValue("");
}

static PyMethodDef Rasterizer_methods[] = {
	{NULL}     /* Sentinel - marks the end of this structure */
};

PyTypeObject RasterizerType = {
   PyObject_HEAD_INIT(NULL)
   0,                         /* ob_size */
   "Rasterizer",               /* tp_name */
   sizeof(RasterizerDict),     /* tp_basicsize */
   0,                         /* tp_itemsize */
   (destructor)Rasterizer_dealloc, /* tp_dealloc */
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
   "Rasterizer object",        /* tp_doc */
   0,                         /* tp_traverse */
   0,                         /* tp_clear */
   0,                         /* tp_richcompare */
   0,                         /* tp_weaklistoffset */
   0,                         /* tp_iter */
   0,                         /* tp_iternext */
   Rasterizer_methods,         /* tp_methods */
//   Rasterizer_members,         /* tp_members */
   0,                         /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)Rasterizer__init,  /* tp_init */
   0,                         /* tp_alloc */
   (newfunc)Rasterizer_new    /* tp_new */
//   (freefunc)Rasterizer_free	/* tp_free */
};
