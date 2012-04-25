MODULE HSL_ZD11_double_x
	use, intrinsic :: iso_c_binding
	use HSL_ZD11_double

IMPLICIT NONE

	type ZD11_c
		! Pointer to the main struct (but we don't own this pointer)
		type(c_ptr) :: main			! ZD11_type *

		! Make portions of main available to C
	    type(c_ptr) :: m, n, ne		! int * (scalar)
		type(c_ptr) :: row			! int[m]
		type(c_ptr) :: col			! int[n]
		type(c_ptr) :: val			! double[ne]
	end type

CONTAINS

	! Computes mat * vec
	subroutine zd11_multiply(mat, vec, ret)
	implicit none
	type(zd11_type), intent(in) :: mat
	real*8, dimension(mat%n), intent(in) :: vec
	real*8, dimension(mat%m), intent(out) :: ret
	
		integer :: i
	
		select case(STRING_get(mat%type))
			case('COORDINATE')
				ret(:) = 0
				do i=1,mat%ne
					ret(mat%row(i)) = ret(mat%row(i)) + vec(mat%col(i)) * mat%val(i)
				end do
			case DEFAULT
				write(6,*) 'hsl_zd11d.f90 only knows how to multiply by matrices of type COORDINATE'
				stop
		end select
	
	end subroutine zd11_multiply
	
	! Computes mat^(transpose) * vec
	subroutine zd11_multiply_T(mat, vec, ret)
	type(zd11_type), intent(in) :: mat
	real*8, dimension(mat%m), intent(in) :: vec
	real*8, dimension(mat%n), intent(out) :: ret
	
		integer :: i
	
		select case(STRING_get(mat%type))
			case('COORDINATE')
				ret(:) = 0
				do i=1,mat%ne
					ret(mat%col(i)) = ret(mat%col(i)) + vec(mat%row(i)) * mat%val(i)
				end do
			case DEFAULT
				write(6,*) 'hsl_zd11d.f90 only knows how to multiply by matrices of type COORDINATE'
				stop
		end select
	
	end subroutine zd11_multiply_T

END MODULE HSL_ZD11_double_x

! ======================================================
! Functions to be called from C, so they're outside of a module

subroutine ZD11_c_init(self, main, m, n, ne)
use HSL_ZD11_double
use HSL_ZD11_double_x
use c_loc_x
use, intrinsic :: iso_c_binding
IMPLICIT NONE
type(ZD11_c) :: self
type(ZD11_type), pointer :: main
integer, value :: m, n, ne

	main%m = m
	main%n = n
	main%ne = ne

	self%main = c_loc(main)
	self%m = c_loc(main%m)
	self%n = c_loc(main%n)
	self%ne = c_loc(main%ne)
	allocate(main%row(ne))
	self%row = c_loc_array_int(main%row)
	allocate(main%col(ne))
	self%col = c_loc_array_int(main%col)
	allocate(main%val(ne))
	self%val = c_loc_array_double(main%val)

	call ZD11_put(main%type, 'COORDINATE')
end subroutine ZD11_c_init


!subroutine ZD11_c_destroy(self)
!type(ZD11_c) :: self
!	type(ZD11_type), pointer :: main
!	call c_f_pointer(self%main, main)
!	deallocate(main)
!subroutine ZD11_c_destroy(self)


function ZD11_put_type_c(self, string, l)
use HSL_ZD11_double
implicit none
type(ZD11_type) :: self				! ZD11_c *
character, dimension(*) :: string	! char *
integer :: l						! strlen(str)
integer :: ZD11_put_type_c

     if (allocated(self%type)) then
        deallocate(self%type,stat=ZD11_put_type_c)
        if (ZD11_put_type_c/=0) return
     end if
     allocate(self%type(l),stat=ZD11_put_type_c)
     if (ZD11_put_type_c/=0) return
     self%type(1:l) = string(1:l)
end function ZD11_put_type_c


		
