subroutine read_eqp(fname, probx)

USE GALAHAD_EQP_double
use netcdf
use ncutil_mod

character(*) :: fname
type(QPT_problem_type) :: probx

integer :: nc
real*8, dimension(:), allocatable :: f_array
integer :: AinfoVar, HinfoVar
integer :: err

	call check(nf90_open("snowdrift.nc", nf90_NoWrite, nc))

	call nc_read_dim(nc, "eqp.m", probx%m)
	call nc_read_dim(nc, "eqp.n", probx%n)

	call nc_read_1d_array_double(nc, "eqp.f", f_array)
	probx%f = f_array(1)

	call nc_read_1d_array_double(nc, "eqp.G", probx%G)
	call nc_read_1d_array_double(nc, "eqp.C", probx%C)
	call nc_read_1d_array_double(nc, "eqp.X", probx%X)
	call nc_read_1d_array_double(nc, "eqp.Y", probx%Y)
	call nc_read_1d_array_double(nc, "eqp.Z", probx%Z)

	! ----------- Read the A Matrix
	call check(nf90_inq_varid(nc, "eqp.A.info", AinfoVar))
		call check(nf90_get_att(nc, AinfoVar, "m", probx%A%m))
		call check(nf90_get_att(nc, AinfoVar, "n", probx%A%n))
	call nc_read_dim(nc, "eqp.A.ne", probx%A%ne)
	call nc_read_1d_array_int(nc, "eqp.A.row", probx%A%row)
	call nc_read_1d_array_int(nc, "eqp.A.col", probx%A%col)
	call nc_read_1d_array_double(nc, "eqp.A.val", probx%A%val)
	call ZD11_put(probx%A%type, 'COORDINATE', err)

	! ----------- Read the H Matrix
	call check(nf90_inq_varid(nc, "eqp.H.info", HinfoVar))
		call check(nf90_get_att(nc, HinfoVar, "m", probx%H%m))
		call check(nf90_get_att(nc, HinfoVar, "n", probx%H%n))
	call nc_read_dim(nc, "eqp.H.ne", probx%H%ne)
	call nc_read_1d_array_int(nc, "eqp.H.row", probx%H%row)
	call nc_read_1d_array_int(nc, "eqp.H.col", probx%H%col)
	call nc_read_1d_array_double(nc, "eqp.H.val", probx%H%val)
	call ZD11_put(probx%H%type, 'COORDINATE', err)

	call check(nf90_close(nc))
end subroutine read_eqp

! =================================================================

program read_eqp_program

USE GALAHAD_EQP_double

IMPLICIT NONE

type(QPT_problem_type) :: prob
TYPE ( EQP_data_type ) :: data	! Used as tmp storage, we don't touch it.
TYPE ( EQP_control_type ) :: control
TYPE ( EQP_inform_type ) :: inform

	call read_eqp("snowdrift.nc", prob)
	prob%new_problem_structure = .TRUE.

	CALL EQP_initialize( data, control, inform ) ! Initialize control parameters
	control%print_level = 5

	print *,'m,n,A%m,A%n,A%ne',prob%m,prob%n,prob%A%m,prob%A%n,prob%A%ne
	print *,'H%m,H%n,H%ne',prob%H%m,prob%H%n,prob%H%ne

	control%SBLS_control%preconditioner = 1	! Random low-elevation points break this
!	control%SBLS_control%preconditioner = 0

	CALL EQP_solve( prob, data, control, inform) ! Solve


	IF ( inform%status /= 0 ) THEN	! Error
		WRITE( 6, "( ' QP_solve exit status = ', I6 ) " ) inform%status
		select case(inform%status)
!			case(-10)
!				write(6, "( ' inform%sils_factorize_status = ', I6 ) " ) inform%sils_factorize_status
			case(-10)
				write(6, "( ' inform%SBLS_inform%status = ', I6 ) " ) inform%SBLS_inform%status
				write(6, "( ' inform%SBLS_inform%SLS_inform%status = ', I6 ) " ) inform%SBLS_inform%SLS_inform%status

!			case(-32)
!				write(6, "( ' inform%PRESOLVE_inform%status = ', I6 ) " ) inform%PRESOLVE_inform%status
!			case(-35)
!				write(6, "( ' inform%QPC_inform%status = ', I6 ) " ) inform%QPC_inform%status
		end select
	else
		! ----------- Good return
		print *,'Successful QP Solve!'

		WRITE( 6, "( ' QP: ', I0, ' QPA iterations ', /, &
			' Optimal objective value =', &
			ES12.4  )" ) &
			inform%cg_iter, inform%obj

!		WRITE( 6, "( ' Optimal solution = ', ( 5ES12.4 ) )" ) prob%X

	end if
	CALL EQP_terminate( data, control, inform) ! delete internal workspace


end program read_eqp_program



