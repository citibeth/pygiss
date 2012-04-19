! Moves data from grid1 to grid2
function snowdrift_downgrid_snowdrift(sd, Z1, Z1_stride, Z2, Z2_stride)
!	USE GALAHAD_QP_double
!	USE GALAHAD_QPT_double		! Debugging
	USE GALAHAD_EQP_double

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
!	INTEGER, ALLOCATABLE, DIMENSION( : ) :: C_stat, B_stat
	TYPE ( QPT_problem_type ) :: p
	TYPE ( EQP_data_type ) :: data	! Used as tmp storage, we don't touch it.
	TYPE ( EQP_control_type ) :: control
	TYPE ( EQP_inform_type ) :: inform

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
!	ALLOCATE( B_stat( nz2 ), C_stat( nz1 ) )
	p%new_problem_structure = .TRUE.

	! Set up Matrices
	p%A = sd%E		! Equality constraints.  Jacobian
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
		! We have "-val" here because the constraint is phrased as "Ax + c = 0"
		! See the GALAHAD documentation eqp.pdf
		p%C(j) = -val
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
	CALL EQP_initialize( data, control, inform ) ! Initialize control parameters
!	control%infinity = infinity

	! Set infinity
!	control%quadratic_programming_solver = 'qpa' ! use QPA.  (This is important in getting it to work at all).
!!	control%scale = 7		! Sinkhorn-Knopp scaling: Breaks things!
!!	control%scale = 1		! Fast and accurate for regridding
!!	control%scale = 0		! No scaling: slow with proper derivative weights
!	control%scale = 1

!	control%generate_sif_file = .TRUE.

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
	!!control%presolve = .TRUE.
	!control%presolve = .FALSE.


	! From Nick Gould:
	! When I run your examples, I find that it is best to use the
	!   control%SBLS_control%preconditioner = 1
	! This builds a "constraint preconditioner" in which the Hessian
	! of your quadratic is replaced by the identity matrix, and then
	! runs the projected CG algorithm. The number of nonzeros in the
	! factors appears to be O(n), e.g. for the 25km problem there are
	! 1412, 3528 entries in the preconditioning matrix and its factors
	! while if instead I use the full matrix
	!   control%SBLS_control%preconditioner = 2
	! there are 30563, 4565576 entries in the matrix and its factors
	! which looks like O(n^2) - this is slightly misleading as when
	! the Hessian or its approximation is diagonal, a more powerful
	! Schur-complement factorization is appropriate (and indeed used
	! by EQP). The preconditioner = 1 requires 13 conjugate-gradient
	! iterations that suggests the preconditioner is effective; of course
	! with preconditioner = 2 only 1 iteration is required, but the
	! factorization cost is far higher. The better preconditioned method
	! seems to take 0.06 seconds on my desktop machine for the 25km problem.
	control%SBLS_control%preconditioner = 1

	call system_clock(time0_ms)
	CALL EQP_solve( p, data, control, inform) ! Solve
	call system_clock(time1_ms)
	delta_time = time1_ms - time0_ms
	delta_time = delta_time * 1d-3
	write(6, "( 'EQP_Solve took ', F6.3, ' seconds')") delta_time

!	inform%status = 0
	IF ( inform%status /= 0 ) THEN	! Error
		snowdrift_downgrid_snowdrift = .false.
		WRITE( 6, "( ' QP_solve exit status = ', I6 ) " ) inform%status
!		select case(inform%status)
!			case(-32)
!				write(6, "( ' inform%PRESOLVE_inform%status = ', I6 ) " ) inform%PRESOLVE_inform%status
!			case(-35)
!				write(6, "( ' inform%QPC_inform%status = ', I6 ) " ) inform%QPC_inform%status
!		end select
	else
		! ----------- Good return
		print *,'Successful QP Solve!'

		WRITE( 6, "( ' QP: ', I0, ' QPA iterations ', /, &
			' Optimal objective value =', &
			ES12.4  )" ) &
			inform%cg_iter, inform%obj

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
	CALL EQP_terminate( data, control, inform) ! delete internal workspace

!	print *,'Z1',Z1
!	print *
!	print *,'Z2',Z2

end function snowdrift_downgrid_snowdrift
