subroutine test_dims2(ZG1)
implicit none
real*8,dimension(:,:), intent(in) :: ZG1

!	print *,'size',ZG1(1,2),size(ZG1,1), size(ZG1,2),size(ZG1)
	print *,'val',ZG1(1,2)

end subroutine test_dims2


program test
implicit none

	real*8,dimension(:,:), allocatable :: ZG1
	real*8,dimension(34,22) :: ZG2

	allocate(ZG1(11,21))
	ZG1(1,2) = 17
	ZG2(1,2) = 17

!	print *,'size',ZG2(1,2),size(ZG2,1), size(ZG2,2),size(ZG2)
	call test_dims2(ZG2)

	print *,'size',ZG1(1,2),size(ZG1,1), size(ZG1,2),size(ZG1)
	call test_dims2(ZG1)

end program test
