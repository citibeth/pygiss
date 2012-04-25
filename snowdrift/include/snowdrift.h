// C header file for relevant functions in snowdrift.f90

namespace giss {
	class Snowdrift;
	class SparseBuilder;
}

#ifdef __cplusplus
extern "C" {
#endif 

void snowdrift_delete_c_(giss::Snowdrift *);
giss::Snowdrift *snowdrift_new_c_(const char *, int);
void snowdrift_set_q_(giss::Snowdrift *, giss::SparseBuilder &QB);

int snowdrift_downgrid_snowdrift_c_(giss::Snowdrift *, double *, int, double *, int);
void snowdrift_upgrid_c_(giss::Snowdrift *, int, double *, int, double *, int);
void snowdrift_downgrid_c_(giss::Snowdrift *, int, double *, int, double *, int);
void snowdrift_overlap_c_(giss::Snowdrift *, double *, int, int);

#ifdef __cplusplus
}
#endif 
