// C header file for relevant functions in snowdrift.f90


#ifdef __cplusplus
extern "C" {
#endif 

void snowdrift_delete_c_(void *);
void *snowdrift_new_c_(const char *, int);
int snowdrift_downgrid_snowdrift_c_(void *, double *, int, double *, int);
void snowdrift_upgrid_c_(void *, int, double *, int, double *, int);
void snowdrift_downgrid_c_(void *, int, double *, int, double *, int);
void snowdrift_overlap_c_(void *, double *, int, int);

#ifdef __cplusplus
}
#endif 
