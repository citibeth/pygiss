# DO NOT EDIT --- auto-generated file
slib/eqp_f.o: slib/eqp_f.f90 qpt_x.mod
	$(FORTRAN_COMMAND_LINE) $<

hsl_zd11_double.mod slib/hsl_zd11d.o: slib/hsl_zd11d.f90 
	$(FORTRAN_COMMAND_LINE) $<

hsl_zd11_double_x.mod slib/hsl_zd11d_f.o: slib/hsl_zd11d_f.f90 hsl_zd11_double.mod c_loc_x.mod
	$(FORTRAN_COMMAND_LINE) $<

ncutil_mod.mod slib/ncutil_f.o: slib/ncutil_f.f90 
	$(FORTRAN_COMMAND_LINE) $<

qpt_x.mod slib/qpt_f.o: slib/qpt_f.f90 hsl_zd11_double_x.mod c_loc_x.mod
	$(FORTRAN_COMMAND_LINE) $<

c_loc_x.mod slib/x_loc_f.o: slib/x_loc_f.f90 
	$(FORTRAN_COMMAND_LINE) $<

smain/read_eqp.o: smain/read_eqp.f90 ncutil_mod.mod
	$(FORTRAN_COMMAND_LINE) $<

smain/test.o: smain/test.f90 
	$(FORTRAN_COMMAND_LINE) $<

smain/xy_snowdrift.o: smain/xy_snowdrift.f90 ncutil_mod.mod
	$(FORTRAN_COMMAND_LINE) $<

