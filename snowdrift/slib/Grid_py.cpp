#include <Python.h>
#include <arrayobject.h>
#include <math.h>
#include "Grid.hpp"

using namespace giss;

// ========================================================================
/// Classmembers of the Python class
struct GridDict {
	PyObject_HEAD
	Grid *grid;	// The grid, in C++
};

// ========= class snowdrift.Grid :
static PyObject *Grid_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	GridDict *self;

printf("Grid_new() called\n");
	self = (GridDict *)type->tp_alloc(type, 0);

    if (self != NULL) {
		self->grid = NULL;
    }

    return (PyObject *)self;
}

static int Grid_init(GridDict *self, PyObject *args, PyObject *kwds)
{

	// Get arguments
	const char *fname;
	const char *vname;
	if (!PyArg_ParseTuple(args, "ss", &fname, &vname)) {
		// Throw an exception...
		PyErr_SetString(PyExc_ValueError,
			"Grid_init() called without a valid string as argument.");
		return 0;
	}

	// Instantiate pointer
	if (self->grid) delete self->grid;
	self->grid = Grid::from_netcdf(std::string(fname), std::string(vname)).release();
fprintf(stderr, "Grid_new() returns %p\n", self->grid);

	return 0;
}

static void Grid_dealloc(GridDict *self)
{
fprintf(stderr, "Grid_dealloc(%p)\n", self->grid);
	if (self->grid) delete self->grid;
	self->grid = NULL;
	self->ob_type->tp_free((PyObject *)self);
}

//static PyMemberDef Grid_members[] = {{NULL}};

static PyObject * Grid_rasterize(GridDict *self, PyObject *args)
{
	// Get Arguments
	PyArrayObject *data;
	PyArrayObject *out;
	double x0,x1,dx;
	double y0,y1,dy;
	if (!PyArg_ParseTuple(args, "ddddddOO",
		&x0, &x1, &dx, &y0, &y1, &dy,
		&data, &out))
	{
		return NULL;
	}

	// --------- Check array sizes and types
	// out must have dimensions (nx, ny) and be double
	// data has single dimension, based on highest index possible
	// strides must be mulitple of sizeof(double)


//fprintf(stderr, "Strides = %d %d\n", out->strides[0], out->strides[1]);
	self->grid->rasterize(x0, x1, dx, y0, y1, dy,
		(double *)data->data, data->strides[0] / sizeof(double),
		(double *)out->data,
		out->strides[0] / sizeof(double),
		out->strides[1] / sizeof(double));

	return Py_BuildValue("");
}

static PyMethodDef Grid_methods[] = {
	{"rasterize", (PyCFunction)Grid_rasterize, METH_VARARGS,
		"Regrid to a regular x/y grid for display ONLY"},
	{NULL}     /* Sentinel - marks the end of this structure */
};

PyTypeObject GridType = {
   PyObject_HEAD_INIT(NULL)
   0,                         /* ob_size */
   "Grid",               /* tp_name */
   sizeof(GridDict),     /* tp_basicsize */
   0,                         /* tp_itemsize */
   (destructor)Grid_dealloc, /* tp_dealloc */
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
   "Grid object",        /* tp_doc */
   0,                         /* tp_traverse */
   0,                         /* tp_clear */
   0,                         /* tp_richcompare */
   0,                         /* tp_weaklistoffset */
   0,                         /* tp_iter */
   0,                         /* tp_iternext */
   Grid_methods,         /* tp_methods */
//   Grid_members,         /* tp_members */
   0,                         /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)Grid_init,  /* tp_init */
   0,                         /* tp_alloc */
   (newfunc)Grid_new    /* tp_new */
};
