

program test
	use snowdrift_mod

	integer :: nx1, ny1
	real*8, dimension(:,:), allocatable, target :: ZG1
	real*8, dimension(:,:), allocatable :: ZG1a
	real*8, dimension(:), pointer :: ZG1_1d
	! Reconstructed coarse grid
	real*8, dimension(:,:), allocatable, target :: ZG1b
	real*8, dimension(:), pointer :: ZG1b_1d

	integer :: nx2, ny2
	real*8, dimension(:,:), allocatable, target :: ZG2
	real*8, dimension(:), pointer :: ZG2_1d


	integer :: ix, iy
	logical :: ok
	type(snowdrift_t) :: sd

	! Write output at end
	integer :: nc


	! ----------- Set up snowdrift regridding
	call snowdrift_init(sd, 'xy_overlap.nc', 13)


	! Changing native area has no effect (it should not)
	sd%overlap%grid1%native_area = sd%overlap%grid1%native_area * 1.1
	sd%overlap%grid2%native_area = sd%overlap%grid2%native_area * 1.1

	! ---------- Define a function on G1
	! (this doesn't happen in real life, it's already defined)
	nx1 = size(sd%overlap%grid1%xy%x_boundaries) - 1
	ny1 = size(sd%overlap%grid1%xy%y_boundaries) - 1
	print *,'nx1,ny1',nx1,ny1
	allocate(ZG1(nx1,ny1))
	allocate(ZG1b(nx1,ny1))
	do iy=1,nx1
		do ix=1,nx1
			ZG1(ix,iy) = -iy * 1.8d0 + ix
		end do
	end do

!	ZG1(1,1) = 10
!	ZG1(2,1) = 10
!	ZG1(1,2) = 11
!	ZG1(2,2) = 11

	! ----------- Allocate for G2 (regridded hi-res values)
	nx2 = size(sd%overlap%grid2%xy%x_boundaries) - 1
	ny2 = size(sd%overlap%grid2%xy%y_boundaries) - 1
	print *,'nx2,ny2',nx2,ny2
	allocate(ZG2(nx2,ny2))

	! ---------- Regrid!
	call c_f_pointer(c_loc(ZG1), ZG1_1d, (/ nx1 * ny1 /))
	call c_f_pointer(c_loc(ZG1b), ZG1b_1d, (/ nx1 * ny1 /))
	call c_f_pointer(c_loc(ZG2), ZG2_1d, (/ nx2 * ny2 /))

!	print *,ZG1_1d
	ok = snowdrift_downgrid(sd, ZG1_1d, 1, ZG2_1d, 1)

	if (ok) then
		call snowdrift_upgrid(sd, ZG2_1d, 1, ZG1b_1d, 1)
		call write_downgrid_solution('out.nc', ZG1, ZG2, ZG1b)
	end if


CONTAINS



subroutine write_downgrid_solution(fname, ZG1, ZG2, ZG1b)
use netcdf
use ncutil_mod

character(*), intent(in) :: fname
real*8,dimension(:,:) :: ZG1
real*8,dimension(:,:) :: ZG2
real*8,dimension(:,:) :: ZG1b

	integer, dimension(2) :: ZG1_dims, ZG2_dims
	integer :: ZG1_var, ZG1b_var, ZG2_var
	integer :: nc

print *,'ZG1 size',size(ZG1,1), size(ZG1,2)
print *,'ZG2 size',size(ZG2,1), size(ZG2,2)
print *,'ZG1b size',size(ZG1b,1), size(ZG1b,2)

	call check(nf90_create(trim(fname), NF90_64BIT_OFFSET, nc))

	! ---------- Define
	call check(nf90_def_dim(nc, 'ZG1.nx', size(ZG1,1), ZG1_dims(1)))
	call check(nf90_def_dim(nc, 'ZG1.ny', size(ZG1,2), ZG1_dims(2)))
	call check(nf90_def_var(nc, 'ZG1', NF90_DOUBLE, ZG1_dims, ZG1_var))

	call check(nf90_def_dim(nc, 'ZG2.nx', size(ZG2,1), ZG2_dims(1)))
	call check(nf90_def_dim(nc, 'ZG2.ny', size(ZG2,2), ZG2_dims(2)))
	call check(nf90_def_var(nc, 'ZG2', NF90_DOUBLE, ZG2_dims, ZG2_var))

	call check(nf90_def_var(nc, 'ZG1b', NF90_DOUBLE, ZG1_dims, ZG1b_var))

	! ----------- Write
	call check(nf90_enddef(nc))
	call check(nf90_put_var(nc, ZG1_var, ZG1))
	call check(nf90_put_var(nc, ZG2_var, ZG2))
	call check(nf90_put_var(nc, ZG1b_var, ZG1b))

	call check(nf90_close(nc))
end subroutine write_downgrid_solution


end program test
