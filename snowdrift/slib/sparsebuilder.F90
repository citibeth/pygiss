module sparsebuilder_mod
	use, intrinsic :: iso_c_binding
	implicit none

	integer, parameter :: MS_GENERAL=0, MS_SYMMETRIC=1, MS_HERMETIAN=2, MS_TRIANGULAR=3, MS_ANTI_SYMMETRIC=4, MS_DIAGONAL=5
	integer, parameter :: TT_GENERAL=0, TT_LOWER=1, TT_UPPER=2
	integer, parameter :: MDT_NON_UNIT=0, MDT_UNIT=1


	! From sparsebuilder_c.cpp
	type, bind(c) :: sparsebuilder_t
		integer(kind=c_int) :: nrow
		integer(kind=c_int) :: ncol
		integer(kind=c_int) :: matrix_structure
		integer(kind=c_int) :: triangular_type
		integer(kind=c_int) :: main_diagonal_type
		integer(kind=c_int) :: array_base

		! ... The rest will be opaque ...
	end type sparsebuilder_t


	! Distilled from Fortran Sparse BLAS
	type sparsematrix_t
		integer, dimension(9) :: descra
		real*8, dimension(:), allocatable  :: val
		integer, dimension(:), allocatable :: indx
		integer, dimension(:), allocatable :: jndx
	end type sparsematrix_t

	interface
		function sparsebuilder_new_0(nrow, ncol, array_base, &
			matrix_structure, triangular_type, main_diagonal_type) &
		bind(c)
			use, intrinsic :: iso_c_binding
			integer(kind=c_int), intent(in), value :: nrow		! don't need ", VALUE"
			integer(kind=c_int), intent(in), value :: ncol
			integer(kind=c_int), intent(in), value :: array_base
			integer(kind=c_int), intent(in), value :: matrix_structure
			integer(kind=c_int), intent(in), value :: triangular_type
			integer(kind=c_int), intent(in), value :: main_diagonal_type
			type(c_ptr) :: sparsebuilder_new_0
		end function sparsebuilder_new_0

		subroutine sparsebuilder_setindices(A, row_indices, col_indices) bind(c)
			use, intrinsic :: iso_c_binding
			import sparsebuilder_t
			type(sparsebuilder_t), intent(in) :: A
			integer(kind=c_int), intent(in) :: row_indices(*)
			integer(kind=c_int), intent(in) :: col_indices(*)
		end subroutine sparsebuilder_setindices


		subroutine sparsebuilder_delete_0(A) bind(c)
			use, intrinsic :: iso_c_binding
			import sparsebuilder_t
			type(sparsebuilder_t), intent(in) :: A
		end subroutine sparsebuilder_delete_0

		subroutine sparsebuilder_set(A, row, col, val) bind(c)
			use, intrinsic :: iso_c_binding
			import sparsebuilder_t
			type(sparsebuilder_t), intent(in) :: A
			integer(kind=c_int), intent(in), value :: row
			integer(kind=c_int), intent(in), value :: col
			real(kind=c_double), intent(in), value :: val
		end subroutine sparsebuilder_set

		subroutine sparsebuilder_add(A, row, col, val) bind(c)
			use, intrinsic :: iso_c_binding
			import sparsebuilder_t
			type(sparsebuilder_t), intent(in) :: A
			integer(kind=c_int), intent(in), value :: row
			integer(kind=c_int), intent(in), value :: col
			real(kind=c_double), intent(in), value :: val
		end subroutine sparsebuilder_add



		function sparsebuilder_set_byindex(A, row_index, col_index, val) bind(c)
			use, intrinsic :: iso_c_binding
			import sparsebuilder_t
			type(sparsebuilder_t), intent(in) :: A
			integer(kind=c_int), intent(in), value :: row_index
			integer(kind=c_int), intent(in), value :: col_index
			real(kind=c_double), intent(in), value :: val
			logical(kind=c_bool) :: sparsebuilder_set_byindex
		end function sparsebuilder_set_byindex

		function sparsebuilder_add_byindex(A, row_index, col_index, val) bind(c)
			use, intrinsic :: iso_c_binding
			import sparsebuilder_t
			type(sparsebuilder_t), intent(in) :: A
			integer(kind=c_int), intent(in), value :: row_index
			integer(kind=c_int), intent(in), value :: col_index
			real(kind=c_double), intent(in), value :: val
			logical(kind=c_bool) :: sparsebuilder_add_byindex
		end function sparsebuilder_add_byindex



		function sparsebuilder_nnz(A) bind(c)
			use, intrinsic :: iso_c_binding
			import sparsebuilder_t
			type(sparsebuilder_t), intent(in) :: A
			integer(kind=c_int) :: sparsebuilder_nnz
		end function sparsebuilder_nnz

		subroutine sparsebuilder_render_coo_0(A, val, indx, jndx) bind(c)
			use, intrinsic :: iso_c_binding
			import sparsebuilder_t
			type(sparsebuilder_t), intent(in) :: A
			type(c_ptr), intent(in), value :: val
			type(c_ptr), intent(in), value :: indx
			type(c_ptr), intent(in), value :: jndx
		end subroutine sparsebuilder_render_coo_0

		subroutine sparsebuilder_sum_per_row_0(A, sums) bind(c)
			use, intrinsic :: iso_c_binding
			import sparsebuilder_t
			type(sparsebuilder_t), intent(in) :: A
			type(c_ptr), intent(in), value :: sums
		end subroutine sparsebuilder_sum_per_row_0

		subroutine sparsebuilder_sum_per_col_0(A, sums) bind(c)
			use, intrinsic :: iso_c_binding
			import sparsebuilder_t
			type(sparsebuilder_t), intent(in) :: A
			type(c_ptr), intent(in), value :: sums
		end subroutine sparsebuilder_sum_per_col_0
	end interface

	contains

	subroutine sparsebuilder_new(fret, nrow, ncol, array_base, &
		matrix_structure, triangular_type, main_diagonal_type)
	use, intrinsic :: iso_c_binding
		type(sparsebuilder_t), pointer, intent(out) :: fret
		integer(kind=c_int), intent(in), value :: nrow		! don't need ", VALUE"
		integer(kind=c_int), intent(in), value :: ncol
		integer(kind=c_int), intent(in), value :: array_base
		integer(kind=c_int), intent(in), value :: matrix_structure
		integer(kind=c_int), intent(in), value :: triangular_type
		integer(kind=c_int), intent(in), value :: main_diagonal_type

		type(c_ptr) :: ret

		ret = sparsebuilder_new_0(nrow, ncol, array_base, &
			matrix_structure, triangular_type, main_diagonal_type)
		call c_f_pointer(ret, fret)
	end subroutine sparsebuilder_new


	subroutine sparsebuilder_delete(A)
		use, intrinsic :: iso_c_binding
		type(sparsebuilder_t), pointer, intent(inout) :: A

		call sparsebuilder_delete_0(A)
		nullify(A)
	end subroutine sparsebuilder_delete



	function cloc_int(A)
		use, intrinsic :: iso_c_binding
		integer, target :: A(*)
		type(c_ptr) :: cloc_int

		cloc_int = c_loc(A)
	end function cloc_int

	function cloc_real(A)
		use, intrinsic :: iso_c_binding
		real*8, target :: A(*)
		type(c_ptr) :: cloc_real

		cloc_real = c_loc(A)
	end function cloc_real

	! Allocate arrays Fortran-style for good integration
	subroutine sparsebuilder_render_coo(A, Aout)
		use, intrinsic :: iso_c_binding
		type(sparsebuilder_t), intent(in) :: A
		type(sparsematrix_t), intent(out), target :: Aout
		
		integer :: nnz

		nnz = sparsebuilder_nnz(A)
		Aout%descra(:) = 0
		Aout%descra(1) = A%matrix_structure
		Aout%descra(2) = A%triangular_type
		Aout%descra(3) = A%main_diagonal_type
		Aout%descra(4) = A%array_base
		Aout%descra(5) = 1		! No repeated values in columns of indx

		allocate(Aout%val(nnz))
		allocate(Aout%indx(nnz))
		allocate(Aout%jndx(nnz))

		! Hack around GNU compiler bug, crashes if we use c_loc() here
		call sparsebuilder_render_coo_0(A, cloc_real(Aout%val), cloc_int(Aout%indx), cloc_int(Aout%jndx))

	end subroutine sparsebuilder_render_coo


	! Renders into the HSL_ZD11 data structure used by GALAHAD
	subroutine sparsebuilder_render_coo_zd11(A, Aout)
		use HSL_ZD11_double
		use, intrinsic :: iso_c_binding

		integer :: nnz, err
		type(sparsebuilder_t), intent(in) :: A
		type(ZD11_type), intent(out), target :: Aout

		nnz = sparsebuilder_nnz(A)

		call ZD11_put(Aout%type, 'COORDINATE', err)
		Aout%ne = nnz
		Aout%m = A%nrow
		Aout%n = A%ncol

		nnz = sparsebuilder_nnz(A)
		allocate(Aout%val(nnz))
		allocate(Aout%row(nnz))
		allocate(Aout%col(nnz))

		! Hack around GNU compiler bug, crashes if we use c_loc() here
		call sparsebuilder_render_coo_0(A, cloc_real(Aout%val), cloc_int(Aout%row), cloc_int(Aout%col))

	end subroutine sparsebuilder_render_coo_zd11



	subroutine sparsebuilder_sum_per_row(A, sums)
		type(sparsebuilder_t), intent(in) :: A
		real*8, dimension(:), allocatable, intent(out) :: sums

		allocate(sums(A%nrow))
		call sparsebuilder_sum_per_row_0(A, cloc_real(sums))
	end subroutine sparsebuilder_sum_per_row

	subroutine sparsebuilder_sum_per_col(A, sums)
		type(sparsebuilder_t), intent(in) :: A
		real*8, dimension(:), allocatable, intent(out) :: sums

		allocate(sums(A%ncol))
		call sparsebuilder_sum_per_col_0(A, cloc_real(sums))
	end subroutine sparsebuilder_sum_per_col




end module sparsebuilder_mod

#if 0
program test
	use sparsematrix_mod
	use, intrinsic :: iso_c_binding
	use hsl_zd11_double
	implicit none
	type(sparsebuilder_t), pointer :: A	! sparsebuilder_t
	type(c_ptr) :: Ac
	type(sparsematrix_t), target :: Aout
	type(ZD11_type) :: Aout_zd11

	logical :: err

!	ac = sparsebuilder_new_0(4, 4, 1, &
	call sparsebuilder_new(A, 4, 4, 1, &
		MS_TRIANGULAR, TT_LOWER, 0)
!	print *,'ac',ac
!	call c_f_pointer(ac,A)

	call sparsebuilder_setindices(A, &
		(/ 17, 18, 22, 23 /), &
		(/ 17, 18, 22, 23 /))

	call sparsebuilder_add(A, 1, 2, 17d0)
	call sparsebuilder_add(A, 2, 1, 18d0)
	err = sparsebuilder_add_byindex(A, 17, 23, 44d0)

	call sparsebuilder_render_coo(A, Aout)
	call sparsebuilder_render_coo_zd11(A, Aout_zd11)


	print *,Aout%descra
	print *,Aout%val
	print *,Aout%indx
	print *,Aout%jndx
	print *

	print *,Aout_zd11%m, Aout_zd11%n,Aout_zd11%ne,Aout_zd11%id,Aout_zd11%type
	print *,Aout_zd11%row
	print *,Aout_zd11%col
	print *,Aout_zd11%ptr
	print *,Aout_zd11%val

	call sparsebuilder_delete(A)

end program test
#endif
