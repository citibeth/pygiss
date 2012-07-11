#include <Python.h>
#include <arrayobject.h>

int main(int argc, char **argv)
{

	printf("%03d,NPY_BOOL\n", NPY_BOOL);
	printf("%03d,NPY_BYTE\n", NPY_BYTE);
	printf("%03d,NPY_INT8\n", NPY_INT8);
	printf("%03d,NPY_SHORT\n", NPY_SHORT);
	printf("%03d,NPY_INT16\n", NPY_INT16);
	printf("%03d,NPY_INT\n", NPY_INT);
	printf("%03d,NPY_INT32\n", NPY_INT32);
	printf("%03d,NPY_LONG\n", NPY_LONG);
	printf("%03d,NPY_LONGLONG\n", NPY_LONGLONG);
	printf("%03d,NPY_INT64\n", NPY_INT64);
	printf("%03d,NPY_UBYTE\n", NPY_UBYTE);
	printf("%03d,NPY_UINT8\n", NPY_UINT8);
	printf("%03d,NPY_USHORT\n", NPY_USHORT);
	printf("%03d,NPY_UINT16\n", NPY_UINT16);
	printf("%03d,NPY_UINT\n", NPY_UINT);
	printf("%03d,NPY_UINT32\n", NPY_UINT32);
	printf("%03d,NPY_ULONG\n", NPY_ULONG);
	printf("%03d,NPY_ULONGLONG\n", NPY_ULONGLONG);
	printf("%03d,NPY_UINT64\n", NPY_UINT64);
	printf("%03d,NPY_HALF\n", NPY_HALF);
	printf("%03d,NPY_FLOAT16\n", NPY_FLOAT16);
	printf("%03d,NPY_FLOAT\n", NPY_FLOAT);
	printf("%03d,NPY_FLOAT32\n", NPY_FLOAT32);
	printf("%03d,NPY_DOUBLE\n", NPY_DOUBLE);
	printf("%03d,NPY_FLOAT64\n", NPY_FLOAT64);
	printf("%03d,NPY_LONGDOUBLE\n", NPY_LONGDOUBLE);
	printf("%03d,NPY_CFLOAT\n", NPY_CFLOAT);
	printf("%03d,NPY_COMPLEX64\n", NPY_COMPLEX64);
	printf("%03d,NPY_CDOUBLE\n", NPY_CDOUBLE);
	printf("%03d,NPY_COMPLEX128\n", NPY_COMPLEX128);
	printf("%03d,NPY_CLONGDOUBLE\n", NPY_CLONGDOUBLE);
	printf("%03d,NPY_DATETIME\n", NPY_DATETIME);
	printf("%03d,NPY_TIMEDELTA\n", NPY_TIMEDELTA);
	printf("%03d,NPY_STRING\n", NPY_STRING);
	printf("%03d,NPY_UNICODE\n", NPY_UNICODE);
	printf("%03d,NPY_OBJECT\n", NPY_OBJECT);
	printf("%03d,NPY_VOID\n", NPY_VOID);
	printf("%03d,NPY_INTP\n", NPY_INTP);
	printf("%03d,NPY_UINTP\n", NPY_UINTP);
#if 0
	printf("%03d,NPY_MASK\n", NPY_MASK);
	printf("%03d,NPY_DEFAULT_TYPE\n", NPY_DEFAULT_TYPE);
	printf("%03d,NPY_NTYPES\n", NPY_NTYPES);
	printf("%03d,NPY_NOTYPE\n", NPY_NOTYPE);
	printf("%03d,NPY_USERDEF\n", NPY_USERDEF);
#endif
}
