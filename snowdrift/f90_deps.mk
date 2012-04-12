# DO NOT EDIT --- auto-generated file
hsl_zd11_double.mod slib/hsl_zd11d.o: slib/hsl_zd11d.f90 
	$(FORTRAN_COMMAND_LINE) $<

ncoverlap_mod.mod slib/ncoverlap.o: slib/ncoverlap.F90 ncutil_mod.mod
	$(FORTRAN_COMMAND_LINE) $<

ncutil_mod.mod slib/ncutil_f.o: slib/ncutil_f.F90 
	$(FORTRAN_COMMAND_LINE) $<

snowdrift_mod.mod slib/snowdrift.o: slib/snowdrift.F90 ncoverlap_mod.mod ncutil_mod.mod hsl_zd11_double.mod sparsebuilder_mod.mod
	$(FORTRAN_COMMAND_LINE) $<

sparsebuilder_mod.mod slib/sparsebuilder.o: slib/sparsebuilder.F90 hsl_zd11_double.mod
	$(FORTRAN_COMMAND_LINE) $<

smain/test.o: smain/test.f90 
	$(FORTRAN_COMMAND_LINE) $<

smain/xy_snowdrift.o: smain/xy_snowdrift.f90 snowdrift_mod.mod ncutil_mod.mod
	$(FORTRAN_COMMAND_LINE) $<

