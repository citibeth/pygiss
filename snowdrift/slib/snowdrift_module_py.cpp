#include <vector>
#include <Python.h>
#include <arrayobject.h>
#include <math.h>

extern PyTypeObject SnowdriftType;
extern PyTypeObject GridType;

extern PyTypeObject RasterizerType;
extern PyMethodDef Rasterizer_functions[];

// ===========================================================================

static PyMethodDef *snowdrift_function_sets[] = {
	Rasterizer_functions,
	NULL
};


extern "C"
void initsnowdrift(void)
{
   PyObject* mod;

	// Create array of all snowdrift functions
	int ndefs = 0;
	for (int i=0; snowdrift_function_sets[i] != NULL; ++i) {
		PyMethodDef *functions = snowdrift_function_sets[i];
		for (int j=0; functions[j].ml_name != NULL; ++j) {
			++ndefs;
		}
	}
	PyMethodDef *defs = new PyMethodDef[ndefs+1];	// We will give this memory to Py_InitModule3
	int k=0;
	for (int i=0; snowdrift_function_sets[i] != NULL; ++i) {
		PyMethodDef *functions = snowdrift_function_sets[i];
		for (int j=0; functions[j].ml_name != NULL; ++j) {
			defs[k++] = functions[j];
		}
	}
	const PyMethodDef dummy = {NULL};
	defs[k++] = dummy;


   // Create the module
   mod = Py_InitModule3("snowdrift", defs, "Snowdrift Regridding Interface");
   if (mod == NULL) {
      return;
   }

   // Fill in some slots in the type, and make it ready
//   SnowdriftType.tp_new = PyType_GenericNew;
   if (PyType_Ready(&SnowdriftType) < 0) {
      return;
   }

   // Add the type to the module.
   Py_INCREF(&SnowdriftType);
   PyModule_AddObject(mod, "Snowdrift", (PyObject*)&SnowdriftType);

   // ======== GridType
   if (PyType_Ready(&GridType) < 0) return;
   Py_INCREF(&GridType);
   PyModule_AddObject(mod, "Grid", (PyObject*)&GridType);

   // ======== RasterizerType
   if (PyType_Ready(&RasterizerType) < 0) return;
   Py_INCREF(&RasterizerType);
   PyModule_AddObject(mod, "Rasterizer", (PyObject*)&RasterizerType);


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