! TODO: 
! 1. Strides are not correctly used everywhere

module snowdrift_mod


use ncoverlap_mod
use ncutil_mod
use HSL_ZD11_double
use sparsebuilder_mod

implicit none

type snowdrift_t
	type(ncoverlap_t) :: overlap	! Stuff read from the overlap file
	type(ZD11_type) :: L		! Overlap matrix in used-grid-cell subspace
	type(ZD11_type) :: Q		! Quadratic form matrix to minimize
	type(ZD11_type) :: E		! Equality constraints LHS (at first same as L)
	real*8, dimension(:), allocatable :: grid1_total_coverage	! Total area of each gridcell in grid1 that overlaps SOMETHING in grid2 (indexed same as L, in projected space)
	real*8, dimension(:), allocatable :: grid2_total_coverage	! Total area of each gridcell in grid2 that overlaps SOMETHING in grid1 (indexed same as L, in projected space)

end type snowdrift_t

contains


! =====================================================

! Constructs the quadratic matrix to optimize in a QP problem,
! for an XY grid
! Assumes indexing scheme of Glimmer (fast-varying dimesion is x)
! @param x_center Center of each grid cell in the x direction
! @param active_cells Index of cells on the grid we're interested in.  Do not compute
!        for cells not in this list.  (Sorted)
subroutine add_smoothness_XY(Q, x_boundaries, y_boundaries, active_cells)
	type(sparsebuilder_t), intent(inout) :: Q
	real*8, dimension(:), intent(in) :: x_boundaries
	real*8, dimension(:), intent(in) :: y_boundaries
	integer, dimension(:), intent(in) :: active_cells

	real*8 :: center0_x2, center1_x2	! Temporary
	integer :: nx, ny			! Number of gridcells
	integer :: aci, di	! Indexing, looping
	integer :: x0i, y0i, x1i, y1i
	integer :: index0, index1	! 1-D versions of (x0i, y0i) and (x1i, y1i)
	logical :: err
	real*8 :: weight

	! Index differentials to use for computing discrete "derivative"
	integer, dimension(4), parameter :: derivative_x = (/-1, 1, 0, 0/)
	integer, dimension(4), parameter :: derivative_y = (/0, 0, -1, 1/)
	!real*8, dimension(4), parameter :: derivative_weights = (/1d0, 1d0, 1d0, 1d0/)

	real*8 :: deltax_x2, deltay_x2	! == (x1i,y1i) - (x0i,y0i)

	nx = size(x_boundaries) - 1
	ny = size(y_boundaries) - 1

	do aci=1,size(active_cells)
		index0 = active_cells(aci)
		y0i = (index0-1) / nx + 1			! Convert to 2D index
		x0i = (index0-1) - (y0i-1)*nx + 1

!print *,'index0',nx,index0,x0i,y0i

		do di=1,4
			x1i = x0i + derivative_x(di)
			if (x1i < 1 .or. x1i >nx) cycle
			center1_x2 = x_boundaries(x1i + 1) + x_boundaries(x1i)
			center0_x2 = x_boundaries(x0i + 1) + x_boundaries(x0i)
			deltax_x2 = center1_x2 - center0_x2

			y1i = y0i + derivative_y(di)
			if (y1i < 1 .or. y1i >ny) cycle
			center1_x2 = y_boundaries(y1i + 1) + y_boundaries(y1i)
			center0_x2 = y_boundaries(y0i + 1) + y_boundaries(y0i)
			deltay_x2 = center1_x2 - center0_x2

			! weight = 1 / |(x1,y1) - (x0,y0)|
			weight = 4d0 / (deltax_x2*deltax_x2 + deltay_x2*deltay_x2)

			! Add (Z1 - Z0)^2 to our objective function
			! (But the calls to sparsebuilder won't add anything if
			! index0 and index1 aren't both active cells.)
			index1 = (y1i-1) * nx + (x1i-1) + 1
!print *,'index0,index1',index0,index1

!weight=1d0
			err = sparsebuilder_add_byindex(Q, index0, index0, weight)
!print *,'err',err
			err = sparsebuilder_add_byindex(Q, index1, index1, weight)
			! The off-diagonal value is doubled...
			err = sparsebuilder_add_byindex(Q, index0, index1, -1d0 * weight)
		end do

	end do

end subroutine add_smoothness_XY
! ------------------------------------------------------------
! Builds sparse representation of the overlap matrix
! @param nco The netCDF overlap file
subroutine build_overlap(nco, L, grid1_total_coverage, grid2_total_coverage)
	use, intrinsic :: iso_c_binding
	use HSL_ZD11_double
	type(ncoverlap_t), intent(in) :: nco
	type(ZD11_type), intent(out) :: L
	type(sparsebuilder_t), pointer :: LB
	real*8, dimension(:), allocatable :: grid1_total_coverage
	real*8, dimension(:), allocatable :: grid2_total_coverage
	integer :: i
	logical :: err

	call sparsebuilder_new(LB, &
		size(nco%grid1%overlap_cells), size(nco%grid2%overlap_cells), 1, &
		0,0,0)
	call sparsebuilder_setindices(LB, nco%grid1%overlap_cells, nco%grid2%overlap_cells)
	do i=1,size(nco%area)
! See if maybe having widely varying coefficients in the matrix was
! the problem.  It was not.
!		if (nco%area(i) > ((5000*5000) * 1d-10)) then
			err = sparsebuilder_add_byindex(LB, &
				nco%overlap_cells(1,i), nco%overlap_cells(2,i), nco%area(i))
!		end if
	end do
	call sparsebuilder_render_coo_zd11(LB, L)
!print *,'XX3',size(L%val), L%m, L%n, L%ne,size(nco%area)

	! Sum over rows and cols
	call sparsebuilder_sum_per_row(LB, grid1_total_coverage)
	call sparsebuilder_sum_per_col(LB, grid2_total_coverage)

	call sparsebuilder_delete(LB)
end subroutine build_overlap
! ------------------------------------------------------------
subroutine snowdrift_init(snowdrift, fname, fname_len)
	integer, intent(in), value :: fname_len
	character(fname_len), intent(in) :: fname
	type(snowdrift_t), target, intent(out) :: snowdrift

	type(sparsebuilder_t), pointer :: QB
	type(ncoverlap_t), pointer :: nco   ! Name shortener
	! integer :: nc

print *,fname

	nco => snowdrift%overlap

	!call check(nf90_open(trim(fname), NF90_NOWRITE, nc))
	call ncoverlap_read(fname, fname_len, nco)
	!call check(nf90_close(nc))

	call build_overlap(nco, snowdrift%L, snowdrift%grid1_total_coverage, snowdrift%grid2_total_coverage)
!print *,'snowdrift%grid1_total_coverage',snowdrift%grid1_total_coverage
!print *,'snowdrift%grid2_total_coverage',snowdrift%grid2_total_coverage

!print *,'XX2',size(snowdrift%L%val), snowdrift%L%m, snowdrift%L%n, snowdrift%L%ne


	! Equality constraints
	snowdrift%E = snowdrift%L

	! -------- Build the quadratic matrix to minimize
	call sparsebuilder_new(QB, &
		size(nco%grid2%overlap_cells), size(nco%grid2%overlap_cells), 1, &
		MS_TRIANGULAR, TT_LOWER, 0)
	call sparsebuilder_setindices(QB, nco%grid2%overlap_cells, nco%grid2%overlap_cells)

	! --------- Smoothness condition is specific to ice grid type
	select case(nco%grid2%type)
		case(GT_XY)
			print *, 'Grid2 is of type GT_XY'
			call add_smoothness_XY(QB, &
				nco%grid2%xy%x_boundaries, nco%grid2%xy%y_boundaries, &
				nco%grid2%overlap_cells)
	end select

	call sparsebuilder_render_coo_zd11(QB, snowdrift%Q)
!print *,'snowdrift%Q%m,snowdrift%Q%n,snowdrift%Q%ne',snowdrift%Q%m,snowdrift%Q%n,snowdrift%Q%ne
!print *,'snowdrift%Q%row',snowdrift%Q%row
!print *,'snowdrift%Q%col',snowdrift%Q%col
!print *,'snowdrift%Q%val',snowdrift%Q%val
	call sparsebuilder_delete(QB)

end subroutine snowdrift_init

! --------------------------------------------------------------
! Moves data from grid1 to grid2
function snowdrift_downgrid_snowdrift(sd, Z1, Z1_stride, Z2, Z2_stride)
	USE GALAHAD_QP_double
	USE GALAHAD_QPT_double		! Debugging

	type(snowdrift_t) :: sd
	real*8, dimension(*) :: Z1, Z2
	integer, intent(in), value :: Z1_stride, Z2_stride
	logical :: snowdrift_downgrid_snowdrift

	integer :: nz1, nz2, index, j
	integer :: errcode
	real*8 :: nn, oo, pp, oo_by_pp, X0, X1, X2	! Temporary for back-filling data
	real*8, dimension(:), allocatable :: Z1_sub, Z2_sub
	real*8 :: val, epsilon
	integer :: time0_ms, time1_ms
	real*8 :: delta_time

	! --- Galahad Stuff
	INTEGER, PARAMETER :: wp = KIND( 1.0D+0 ) ! set precision
	REAL ( KIND = wp ), PARAMETER :: infinity = 10.0_wp ** 20
	TYPE ( QP_control_type ) :: control
	TYPE ( QP_inform_type ) :: inform
	INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_stat, B_stat
	TYPE ( QPT_problem_type ) :: p
	TYPE ( QP_data_type ) :: data	! Used as tmp storage, we don't touch it.

	! --------------------------------

	nz1 = size(sd%overlap%grid1%overlap_cells)		! # grid cells from grid1
	nz2 = size(sd%overlap%grid2%overlap_cells)		! # grid cells from grid2
	p%m = nz1		! # of rows in constraints matrix
	p%n = nz2		! H is square

	! Select out the items from Z1 that participate in downscaling
	allocate(Z1_sub(nz1))
	do j=1,nz1
		index = sd%overlap%grid1%overlap_cells(j)
		pp = sd%overlap%grid1%proj_area(j)
		nn = sd%overlap%grid1%native_area(j)

		Z1_sub(j) = (nn/pp) * Z1(index * Z1_stride)	! Scale to correct for area errors in projection
	end do

	! start problem data
	ALLOCATE( p%G( nz2 ), p%X_l( nz2 ), p%X_u( nz2 ) )
	ALLOCATE( p%C( nz1 ), p%C_l( nz1 ), p%C_u( nz1 ) )
	ALLOCATE( p%X( nz2 ), p%Y( nz1 ), p%Z( nz2 ) )
	ALLOCATE( B_stat( nz2 ), C_stat( nz1 ) )
	p%new_problem_structure = .TRUE.

	! Set up Matrices
	p%A = sd%E		! Equality constraints.  Jacobian


!print *,'XX1',size(sd%E%val), sd%E%m, sd%E%n, sd%E%ne
!print *,'XX1',size(p%A%val), p%A%m, p%A%n, p%A%ne

	p%H = sd%Q			! Quadratic term of objective function (Hessian)

	! Objective Function
	p%f = 0d0			! Constant term of objective function = 0
	p%G(:) = 0d0		! Linear term of objective function = 0


	! Constraints (rhs of p%A / sd%E)
	! If GALAHAD needs epsilon <> 0, then this will introduce
	! non-conservation.  If small enough, conservation can be
	! re-established through post-processing fudging.
	do j=1,nz1
		! Z1_sub is already in proj_area scaling
		val = Z1_sub(j) * sd%grid1_total_coverage(j)
		!epsilon = abs(p%C_l(j)) * 1d-20
		epsilon = 0
		p%C_l(j) = val - epsilon
		p%C_u(j) = val + epsilon
	end do

	! --------------- Set up the QP problem
	! constraint upper bound
	p%X_l(:) = -infinity
	p%X_u(:) = infinity

	! variable upper bound
	p%X = 0.0_wp ; p%Y = 0.0_wp ; p%Z = 0.0_wp ! start from zero

	! Start from (approx) result of simple reg
	call zd11_multiply_T(sd%L, Z1_sub, p%X)
	do j=1,nz2
		oo = sd%grid2_total_coverage(j)		! overlap area of this grid cell (in projected space)
		pp = sd%overlap%grid2%proj_area(j)
		nn = sd%overlap%grid2%native_area(j)
		p%X(j) = p%X(j) * pp / (oo * nn) !/ sd%overlap%grid2%native_area(j)
	end do

	! ------------ problem data complete, set up control
	CALL QP_initialize( data, control, inform ) ! Initialize control parameters
	control%infinity = infinity

	! Set infinity
	control%quadratic_programming_solver = 'qpa' ! use QPA.  (This is important in getting it to work at all).
!	control%scale = 7		! Sinkhorn-Knopp scaling: Breaks things!
!	control%scale = 1		! Fast and accurate for regridding
!	control%scale = 0		! No scaling: slow with proper derivative weights
	control%scale = 1

	control%generate_sif_file = .TRUE.

print *,'m,n,A%m,A%n,A%ne',p%m,p%n,p%A%m,p%A%n,p%A%ne
!print *,'p%A%row',p%A%row
!print *,'p%A%col',p%A%col
!print *,'p%A%val',p%A%val
!call QPT_A_from_C_to_S(p, errcode)
!print *,'errcode',errcode

print *
print *,'H%m,H%n,H%ne',p%H%m,p%H%n,p%H%ne
!print *,'p%H%row',p%H%row
!print *,'p%H%col',p%H%col
!print *,'p%H%val',p%H%val


	! Causes error on samples with lat/lon and cartesian grid
	!control%presolve = .TRUE.
	control%presolve = .FALSE.

	call system_clock(time0_ms)
	CALL QP_solve( p, data, control, inform, C_stat, B_stat ) ! Solve
	call system_clock(time1_ms)
	delta_time = time1_ms - time0_ms
	delta_time = delta_time * 1d-3
	write(6, "( 'QP_Solve took ', F6.3, ' seconds')") delta_time

!	inform%status = 0
	IF ( inform%status /= 0 ) THEN	! Error
		snowdrift_downgrid_snowdrift = .false.
		WRITE( 6, "( ' QP_solve exit status = ', I6 ) " ) inform%status
		select case(inform%status)
			case(-32)
				write(6, "( ' inform%PRESOLVE_inform%status = ', I6 ) " ) inform%PRESOLVE_inform%status
			case(-35)
				write(6, "( ' inform%QPC_inform%status = ', I6 ) " ) inform%QPC_inform%status
		end select
	else
		! ----------- Good return
		print *,'Successful QP Solve!'

		WRITE( 6, "( ' QP: ', I0, ' QPA iterations ', /, &
			' Optimal objective value =', &
			ES12.4  )" ) &
			inform%QPA_inform%iter, inform%obj

!		WRITE( 6, "( ' Optimal solution = ', ( 5ES12.4 ) )" ) p%X




		snowdrift_downgrid_snowdrift = .true.		

		! Merge result into Z2
		do j=1,nz2
			index = sd%overlap%grid2%overlap_cells(j)
			oo = sd%grid2_total_coverage(j)		! overlap area of this grid cell (in projected space)
			pp = sd%overlap%grid2%proj_area(j)
			nn = sd%overlap%grid2%native_area(j)
			X0 = Z2(index * Z2_stride)		! Original value
			X1 = p%X(j)			! Result of regridding

			oo_by_pp = oo/pp
			X2 = (1-oo_by_pp) * X0 + (pp/nn) * oo_by_pp * X1
			Z2(index * Z2_stride) = X2		! Put back final value
		end do
	end if
	CALL QP_terminate( data, control, inform) ! delete internal workspace

!	print *,'Z1',Z1
!	print *
!	print *,'Z2',Z2

end function snowdrift_downgrid_snowdrift

! @param merge_or_replace For GCM grid cells partially covered by ice cells:
!        1 = Merge upgrid value into GCM grid cell.
!        0 = Completely replace GCM grid cell value, scaling mass by
!            appropriate area
subroutine snowdrift_upgrid(sd, merge_or_replace, &
Z2_1d, Z2_stride, Z1_1d, Z1_stride)
	type(snowdrift_t), intent(in) :: sd
	integer, intent(in), value :: merge_or_replace
	real*8, dimension(*), intent(in) :: Z2_1d
	real*8, dimension(*), intent(out) :: Z1_1d
	integer, intent(in), value :: Z2_stride, Z1_stride

	real*8 :: nn, oo, pp, oo_by_pp
	real*8 :: X0, X1
	real*8 :: Lval, native_area_2
	integer :: nz1, nz2, index, j
	real*8, dimension(:), allocatable :: Stuff1_sub, Z2_sub

	nz1 = size(sd%overlap%grid1%overlap_cells)		! # grid cells from grid1
	nz2 = size(sd%overlap%grid2%overlap_cells)		! # grid cells from grid2

	! Select out the items from Z2 that participate in downscaling
	allocate(Z2_sub(nz2))
	do j=1,nz2
		index = sd%overlap%grid2%overlap_cells(j)
		pp = sd%overlap%grid2%proj_area(j)
		nn = sd%overlap%grid2%native_area(j)

		! Amount of stuff in each gridcell of grid2
		Z2_sub(j) = (nn/pp) * Z2_1d(index)
	end do

	! Multiply Stuff1_sub = L * Z2_sub
	allocate(Stuff1_sub(nz1))
	call zd11_multiply(sd%L, Z2_sub, Stuff1_sub)

	
	! Merge result into Z1
write (6,*) 'merge_or_repalce = ',merge_or_replace
	do j=1,nz1
		index = sd%overlap%grid1%overlap_cells(j)
		oo = sd%grid1_total_coverage(j)		! overlap area of this grid cell (in projected space)
		pp = sd%overlap%grid1%proj_area(j)
		nn = sd%overlap%grid1%native_area(j)
		X0 = Z1_1d(index) 		! Original value (amount of stuff = X0*nn)

		select case (merge_or_replace)
		case (0)		! Replace
			Z1_1d(index) = Stuff1_sub(j) * pp / (oo * nn)	! Stuff1_sub(j) * (pp/oo) / nn
		case (1)		! Merge
			oo_by_pp = oo/pp		! Fraction of this gridcell participating
			X1 = (1-oo_by_pp) * X0 + Stuff1_sub(j) / nn
			Z1_1d(index) = X1		! Put back final value
		end select
	end do

end subroutine snowdrift_upgrid






! @param merge_or_replace For GCM grid cells partially covered by ice cells:
!        1 = Merge upgrid value into GCM grid cell.
!        0 = Completely replace GCM grid cell value, scaling mass by
!            appropriate area
subroutine snowdrift_downgrid(sd, merge_or_replace, &
Z1_1d, Z1_stride, Z2_1d, Z2_stride)
	type(snowdrift_t), intent(in) :: sd
	integer, intent(in), value :: merge_or_replace
	real*8, dimension(*), intent(in) :: Z1_1d
	real*8, dimension(*), intent(out) :: Z2_1d
	integer, intent(in), value :: Z1_stride, Z2_stride

	real*8 :: nn, oo, pp, oo_by_pp
	real*8 :: X0, X1
	real*8 :: Lval, native_area_2
	integer :: nz2, nz1, index, j
	real*8, dimension(:), allocatable :: Stuff2_sub, Z1_sub

	nz2 = size(sd%overlap%grid2%overlap_cells)		! # grid cells from grid2
	nz1 = size(sd%overlap%grid1%overlap_cells)		! # grid cells from grid1

	! Select out the items from Z1 that participate in downscaling
	allocate(Z1_sub(nz1))
	do j=1,nz1
		index = sd%overlap%grid1%overlap_cells(j)
		pp = sd%overlap%grid1%proj_area(j)
		nn = sd%overlap%grid1%native_area(j)

		! Amount of stuff in each gridcell of grid1
		Z1_sub(j) = (nn/pp) * Z1_1d(index)
	end do

	! Multiply Stuff2_sub = L * Z1_sub
	allocate(Stuff2_sub(nz2))
	call zd11_multiply_t(sd%L, Z1_sub, Stuff2_sub)

	
	! Merge result into Z2
write (6,*) 'merge_or_repalce = ',merge_or_replace
	do j=1,nz2
		index = sd%overlap%grid2%overlap_cells(j)
		oo = sd%grid2_total_coverage(j)		! overlap area of this grid cell (in projected space)
		pp = sd%overlap%grid2%proj_area(j)
		nn = sd%overlap%grid2%native_area(j)
		X0 = Z2_1d(index) 		! Original value (amount of stuff = X0*nn)

		select case (merge_or_replace)
		case (0)		! Replace
			Z2_1d(index) = Stuff2_sub(j) * pp / (oo * nn)	! Stuff1_sub(j) * (pp/oo) / nn
		case (1)		! Merge
			oo_by_pp = oo/pp		! Fraction of this gridcell participating
			X1 = (1-oo_by_pp) * X0 + Stuff2_sub(j) / nn
			Z2_1d(index) = X1		! Put back final value
		end select
	end do

end subroutine snowdrift_downgrid




end module snowdrift_mod

! --------------------------------------------------------------
! This is for C interfacing

function snowdrift_new_c(fname, fname_len)
	use snowdrift_mod

	integer, intent(in), value :: fname_len
	character(fname_len), intent(in) :: fname
	type(c_ptr) :: snowdrift_new

	type(snowdrift_t), pointer :: sd

	allocate(sd)
	call snowdrift_init(sd, fname, fname_len)
	snowdrift_new = c_loc(sd)
end function snowdrift_new_c

subroutine snowdrift_delete_c(sd_c)
	use snowdrift_mod
	type(c_ptr), value :: sd_c

	type(snowdrift_t), pointer :: sd

	call c_f_pointer(sd_c, sd)
	deallocate(sd)
end subroutine snowdrift_delete_c

function snowdrift_downgrid_snowdrift_c(sd_c, Z1, Z1_stride, Z2, Z2_stride)
	use snowdrift_mod
	type(c_ptr), value :: sd_c
	real*8, dimension(*) :: Z1, Z2
	integer, intent(in), value :: Z1_stride, Z2_stride
	logical :: snowdrift_downgrid_snowdrift_c

	type(snowdrift_t), pointer :: sd

	call c_f_pointer(sd_c, sd)
	snowdrift_downgrid_snowdrift_c = snowdrift_downgrid_snowdrift(sd, Z1, Z1_stride, Z2, Z2_stride)
end function snowdrift_downgrid_snowdrift_c

subroutine snowdrift_upgrid_c(sd_c, merge_or_replace, Z2, Z2_stride, Z1, Z1_stride)
	use snowdrift_mod
	type(c_ptr), value :: sd_c
	integer, intent(in), value :: merge_or_replace
	real*8, dimension(*) :: Z2, Z1
	integer, intent(in), value :: Z2_stride, Z1_stride

	type(snowdrift_t), pointer :: sd

	call c_f_pointer(sd_c, sd)
	call snowdrift_upgrid(sd, merge_or_replace, Z2, Z2_stride, Z1, Z1_stride)
end subroutine snowdrift_upgrid_c

subroutine snowdrift_downgrid_c(sd_c, merge_or_replace, Z1, Z1_stride, Z2, Z2_stride)
	use snowdrift_mod
	type(c_ptr), value :: sd_c
	integer, intent(in), value :: merge_or_replace
	real*8, dimension(*) :: Z1, Z2
	integer, intent(in), value :: Z1_stride, Z2_stride

	type(snowdrift_t), pointer :: sd

	call c_f_pointer(sd_c, sd)
	call snowdrift_downgrid(sd, merge_or_replace, Z1, Z1_stride, Z2, Z2_stride)
end subroutine snowdrift_downgrid_c


! --------------------------------------------------------------

! Renders overlap matrix to a dense matrix
subroutine snowdrift_overlap_c(sd_c, densemat, row_stride, col_stride)
use snowdrift_mod
type(c_ptr), value :: sd_c
real*8, dimension(*) :: densemat
integer :: row_stride, col_stride

#if 0
	integer :: i
	integer :: index
	type(snowdrift_t), pointer :: sd

	call c_f_pointer(sd_c, sd)
	select case(STRING_get(sd%L%type))
		case('COORDINATE')
			ret(:) = 0
			do i=1,sd%L%ne
				index = row_stride * sd%L%row(i) + col_stride * sd%L%col(i)
				densemat(index) = sd%L%val(i)
			end do
		case DEFAULT
			write(6,*) 'hsl_zd11d.f90 only knows how to multiply by matrices of type COORDINATE'
			stop
	end select
#endif

end subroutine snowdrift_overlap_c
