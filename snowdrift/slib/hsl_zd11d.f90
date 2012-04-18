! COPYRIGHT (c) 2006 Council for the Central Laboratory
!                    of the Research Councils
! Original date 21 February 2006. Version 1.0.0.
! 6 March 2007 Version 1.1.0. Argument stat made non-optional

MODULE HSL_ZD11_double

!  ==========================
!  Sparse matrix derived type
!  ==========================

  TYPE, PUBLIC :: ZD11_type
    INTEGER :: m, n, ne
    CHARACTER, ALLOCATABLE, DIMENSION(:) :: id, type
    INTEGER, ALLOCATABLE, DIMENSION(:) :: row, col, ptr
    REAL ( KIND( 1.0D+0 ) ), ALLOCATABLE, DIMENSION(:) :: val
  END TYPE

CONTAINS

   SUBROUTINE ZD11_put(array,string,stat)
     CHARACTER, allocatable :: array(:)
     CHARACTER(*), intent(in) ::  string
     INTEGER, intent(OUT) ::  stat

     INTEGER :: i,l

     l = len_trim(string)
     if (allocated(array)) then
        deallocate(array,stat=stat)
        if (stat/=0) return
     end if
     allocate(array(l),stat=stat)
     if (stat/=0) return
     do i = 1, l
       array(i) = string(i:i)
     end do

   END SUBROUTINE ZD11_put

   FUNCTION ZD11_get(array)
     CHARACTER, intent(in):: array(:)
     CHARACTER(size(array)) ::  ZD11_get
! Give the value of array to string.

     integer :: i
     do i = 1, size(array)
        ZD11_get(i:i) = array(i)
     end do

   END FUNCTION ZD11_get

!-*-*-*-*-  G A L A H A D -  S T R I N G _ g e t   F U N C T I O N  -*-*-*-*-

     FUNCTION STRING_get( array )

!  obtain the elements of a character array as a character variable

!  Dummy arguments

!  array - character array whose components hold the string
!  string_get - equivalent character string

     CHARACTER, INTENT( IN ), DIMENSION( : ) :: array
     CHARACTER( SIZE( array ) ) :: STRING_get

!  Local variables

     INTEGER :: i

     DO i = 1, SIZE( array )
        STRING_get( i : i ) = array( i )
     END DO

     RETURN

!  End of function STRING_get

     END FUNCTION STRING_get


! ---------------------------------------------------------
! Code below by Bob Fischer

! Computes mat * vec
subroutine zd11_multiply(mat, vec, ret)
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


! ! Renders to a dense matrix
! subroutine zd11_to_dense(mat, densemat, row_stride, col_stride)
! type(zd11_type), intent(in) :: mat
! real*8, dimension(*) :: densemat
! integer :: row_stride, col_stride
! 
! 	integer :: i
! 	integer :: index
! 
! 	select case(STRING_get(mat%type))
! 		case('COORDINATE')
! 			ret(:) = 0
! 			do i=1,mat%ne
! 				index = row_stride * mat%row(i) + col_stride * mat%col(i)
! 				densemat(index) = mat%val(i)
! 			end do
! 		case DEFAULT
! 			write(6,*) 'hsl_zd11d.f90 only knows how to multiply by matrices of type COORDINATE'
! 			stop
! 	end select
! 
! end subroutine zd11_to_dense


END MODULE HSL_ZD11_double


