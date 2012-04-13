#include <Python.h>
#include <arrayobject.h>
#include <math.h>

/// Classmembers of the Python class
typedef struct {
	PyObject_HEAD
	void *snowdrift_f;	// Fortran-allocated snowdrift pointer
} SnowdriftDict;



// ============================ The functions
static PyObject *Snowdrift_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	SnowdriftDict *self;

	self = (SnowdriftDict *)type->tp_alloc(type, 0);

    if (self != NULL) {
		self->snowdrift_f = NULL;
    }

    return (PyObject *)self;

static int Snowdrift_init(SnowdriftDict *self, PyObject *args, PyObject *kwds)
{
	self->dict = PyDict_New();
	self->count = 0;

	const char *fname;
	if (!PyArg_ParseTuple(args, "s", &fname)) return NULL;

	if (self->snowdrift_f) snowdrift_delete_c(self->snowdrift_f);

	self->snowdrift_f = snowdrift_new_c(fname, strlen(fname));
	return 0;
}

static void Snowdrift_dealloc(SnowdriftDict *self)
{
	Py_XDECREF(self->dict);
	if (self->snowdritf_f) snowdrift_delete_c(self->snowdrift_f);
	self->snowdrift_f = NULL;
	self->ob_type->tp_free((PyObject *)self);
}

static PyMemberDef SnowdriftDict_members[] = {{NULL}};


/** Check that PyArrayObject is a double (Float) type and a vector
@return 0 if not a double vector, and also raise exception. */
int is_doublevector(PyArrayObject *vec)  {
	if (vec->descr->type_num != NPY_DOUBLE || vec->nd != 1)  {
		PyErr_SetString(PyExc_ValueError,
			"In is_doublevector: array must be of type Float and 1 dimensional (n).");
		return 0;
	}
	return 1;
}


static PyObject * Snowdrift_upgrid(SnowdriftDict *self, PyObject *args)
{
	PyArrayObject *vecin, *vecout;  // The python objects to be extracted from the args
	double *cin, *cout;             // The C vectors to be created to point to the 
	                                //   python vectors, cin and cout point to the row
	                                //   of vecin and vecout, respectively
	int i,j,n;
	const char *str;
	double dfac;
	
	/* Parse tuples separately since args will differ between C fcns */
	if (!PyArg_ParseTuple(args, "O!O!",
		&PyArray_Type, &Z2,
		&PyArray_Type, &Z1)) return NULL;
	if (NULL == Z2)  return NULL;
	if (NULL == Z1)  return NULL;
	
	// Print out input string
	printf("Input string: %s\n", str);
	
	/* Check that objects are 'double' type and vectors */
	if (!is_doublevector(Z2)) return NULL;
	if (!is_doublevector(Z1)) return NULL;
	
	snowdrift_upgrid_c(self->snowdrift_f,
		Z2->data, Z2->dimensions[0],
		Z1->data, Z1->dimensions[0]);

	return NULL;
}

static PyMethodDef Snowdrift_methods[] = {
	// {"__init__", Snowdrift_init
	{"downgrid", (PyCFunction)Snowdrift_downgrid, METH_VARARGS,
		"Convert from grid1 to grid2 using Snowdrift regridding method"},
	{"upgrid", (PyCFunction)Snowdrift_upgrid, METH_VARARGS,
		"Convert from grid2 to grid1, simple overlap matrix multiplication"},
	{NULL}     /* Sentinel - marks the end of this structure */
};

static PyTypeObject
SnowdriftType = {
   PyObject_HEAD_INIT(NULL)
   0,                         /* ob_size */
   "Snowdrift",               /* tp_name */
   sizeof(Snowdrift),         /* tp_basicsize */
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
   Snowdrift_members,         /* tp_members */
   0,                         /* tp_getset */
   0,                         /* tp_base */
   0,                         /* tp_dict */
   0,                         /* tp_descr_get */
   0,                         /* tp_descr_set */
   0,                         /* tp_dictoffset */
   (initproc)Snowdrift_init,  /* tp_init */
   0,                         /* tp_alloc */
   (newproc)Snowdrift_new,    /* tp_new */
};


void
initsnowdrift(void)
{
   PyObject* mod;

   // Create the module
   mod = Py_InitModule3("snowdrift", NULL, "Snowdrift Regridding Interface");
   if (mod == NULL) {
      return;
   }

   // Fill in some slots in the type, and make it ready
   SnowdriftType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&SnowdriftType) < 0) {
      return;
   }

   // Add the type to the module.
   Py_INCREF(&SnowdriftType);
   PyModule_AddObject(mod, "Snowdrift", (PyObject*)&SnowdriftType);
}
     

// ========================================================================
// /* ==== Set up the methods table ====================== */
// 
// static PyMethodDef ModuleMethods[] = { {NULL} };
// 
// /* ==== Initialize the C_test functions ====================== */
// // Module name must be _C_snowdrift in compile and linked 
// EXTERN_C PyMODINIT_FUNC initsnowdrift()
// {
//     // create a new module
//     PyObject *module = Py_InitModule("snowdrift", ModuleMethods);
// 	import_array();  // Must be present for NumPy.  Called first after above line.
//     PyObject *moduleDict = PyModule_GetDict(module);
// 
// 	// Create a new class
//     PyObject *classDict = PyDict_New();
//     PyObject *className = PyString_FromString("Snowdrift");
//     PyObject *snowdriftClass = PyClass_New(NULL, classDict, className);
//     PyDict_SetItemString(moduleDict, "Snowdrift", snowdriftClass);
//     Py_DECREF(classDict);
//     Py_DECREF(className);
//     Py_DECREF(snowdriftClass);
//     
//     /* add methods to class */
//     for (PyMethodDef *def = SnowdriftMethods; def->ml_name != NULL; def++) {
// 		PyObject *func = PyCFunction_New(def, NULL);
// 		PyObject *method = PyMethod_New(func, NULL, snowdriftClass);
// 		PyDict_SetItemString(classDict, def->ml_name, method);
// 		Py_DECREF(func);
// 		Py_DECREF(method);
//     }
// }
// 