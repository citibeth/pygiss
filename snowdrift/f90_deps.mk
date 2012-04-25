# DO NOT EDIT --- auto-generated file
c_loc_x.mod slib/c_loc_x.o: slib/c_loc_x.f90 
	$(FORTRAN_COMMAND_LINE) $<

slib/eqp_x.o: slib/eqp_x.f90 qpt_x.mod
	$(FORTRAN_COMMAND_LINE) $<

hsl_zd11_double.mod slib/hsl_zd11d.o: slib/hsl_zd11d.f90 
	$(FORTRAN_COMMAND_LINE) $<

hsl_zd11_double_x.mod slib/hsl_zd11d_x.o: slib/hsl_zd11d_x.f90 hsl_zd11_double.mod c_loc_x.mod
	$(FORTRAN_COMMAND_LINE) $<

qpt_x.mod slib/qpt_x.o: slib/qpt_x.f90 hsl_zd11_double_x.mod c_loc_x.mod
	$(FORTRAN_COMMAND_LINE) $<

sparsebuilder_mod.mod slib/sparsebuilder.o: slib/sparsebuilder.F90 hsl_zd11_double.mod
	$(FORTRAN_COMMAND_LINE) $<

sparsecoord_mod.mod slib/sparsecoord.o: slib/sparsecoord.F90 hsl_zd11_double.mod
	$(FORTRAN_COMMAND_LINE) $<

smain/test.o: smain/test.f90 
	$(FORTRAN_COMMAND_LINE) $<

smain/xy_snowdrift.o: smain/xy_snowdrift.f90 
	$(FORTRAN_COMMAND_LINE) $<

