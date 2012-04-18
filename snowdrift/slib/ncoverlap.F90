module ncoverlap_mod

use ncutil_mod
use netcdf

implicit none

integer, parameter :: GT_UNKNOWN=0, GT_LATLON=1, GT_XY=2

! ---------- Stuff read from netCDF overlap file

! Specialization for xy data type
type :: ncgrid_xy_t
	real*8, dimension(:), allocatable :: x_boundaries
	real*8, dimension(:), allocatable :: y_boundaries
end type ncgrid_xy_t

! Information related to one grid
type :: ncgrid_t
	integer, dimension(:), allocatable :: overlap_cells
	real*8, dimension(:), allocatable :: native_area
	real*8, dimension(:), allocatable :: proj_area
	integer :: type		! GT_*

	! Optional stuff (polymorphism of sorts)
	type(ncgrid_xy_t) :: xy
end type ncgrid_t

! The overall overlap file
type ncoverlap_t
	type(ncgrid_t) :: grid1, grid2
	integer, dimension(:,:), allocatable :: overlap_cells
	real*8, dimension(:), allocatable :: area
end type ncoverlap_t

! ========================================================
contains

! Reads info about one grid
! @param nc netCDF file handle
! @param name0 Name of the grid to read
! @param name_len Length of name0 string
! @param ncgrid Place for output
subroutine ncgrid_read(nc, name0, name_len, ncgrid)
	integer, intent(in) :: nc
	character(*), intent(in) :: name0
	integer, intent(in) :: name_len
	type(ncgrid_t), intent(inout) :: ncgrid

	integer, dimension(:), allocatable :: realized_cells
	real(8), dimension(:), allocatable :: realized_native_area, realized_proj_area
	real(8), dimension(:), allocatable :: full_native_area, full_proj_area
	integer :: min_realized, max_realized
	integer :: i, index

	character(100) :: name
	character(50) :: stype

	name(:) = name0(1:name_len)

	name(name_len+1:) = '.overlap_cells '
	call nc_read_1d_array_int(nc, name, ncgrid%overlap_cells)


	! ---- These variables are indexed by realized cells (not overlap cells)
	name(name_len+1:) = '.realized_cells '
	call nc_read_1d_array_int(nc, name, realized_cells)
	name(name_len+1:) = '.native_area '
	call nc_read_1d_array_double(nc, name, realized_native_area)
	name(name_len+1:) = '.proj_area '
	call nc_read_1d_array_double(nc, name, realized_proj_area)


	! -------- Convert above to being indexed by overlap cells
	! (assume realized_cells is sorted)
	min_realized = realized_cells(1)
	max_realized = realized_cells(size(realized_cells,1))
	allocate(full_native_area(min_realized:max_realized))
	allocate(full_proj_area(min_realized:max_realized))
	do i=1,size(realized_cells,1)
		index = realized_cells(i)
		full_native_area(index) = realized_native_area(i)
		full_proj_area(index) = realized_proj_area(i)
	end do
	allocate(ncgrid%native_area(size(ncgrid%overlap_cells,1)))
	allocate(ncgrid%proj_area(size(ncgrid%overlap_cells,1)))
	do i=1,size(ncgrid%overlap_cells,1)
		index = ncgrid%overlap_cells(i)
		ncgrid%native_area(i) = full_native_area(index)
		ncgrid%proj_area(i) = full_proj_area(index)
	end do


	name(name_len+1:) = '.info '
	call nc_read_attribute(nc, name, 'type', stype)

	ncgrid%type = GT_UNKNOWN
	select case (trim(stype))
		case('xy')
			ncgrid%type = GT_XY
			! allocate(ncgrid%xy)		! gfortran buggy with allocatable scalars
			name(name_len+1:) = '.x_boundaries '
			call nc_read_1d_array_double(nc, trim(name), ncgrid%xy%x_boundaries)
			name(name_len+1:) = '.y_boundaries '
			call nc_read_1d_array_double(nc, trim(name), ncgrid%xy%y_boundaries)
	end select

end subroutine ncgrid_read

! Reads entire overlap.nc file
! @param nc netCDF file handle
subroutine ncoverlap_read(fname, fname_len, ncoverlap)
	character(fname_len), intent(in) :: fname
	integer, intent(in) :: fname_len
	type(ncoverlap_t), intent(inout) :: ncoverlap

	integer :: nc

	call check(nf90_open(trim(fname), NF90_NOWRITE, nc))

	! ----- Read about the two grids individually
	call ncgrid_read(nc, 'grid1', 5, ncoverlap%grid1)
	call ncgrid_read(nc, 'grid2', 5, ncoverlap%grid2)

	! ----- Read about how they overlap
	call nc_read_2d_array_int(nc, 'overlap.overlap_cells', ncoverlap%overlap_cells)
	call nc_read_1d_array_double(nc, 'overlap.area', ncoverlap%area)

	call check(nf90_close(nc))
end subroutine ncoverlap_read

end module ncoverlap_mod

#if 0
program test

use ncoverlap_mod

	type(ncoverlap_t) :: nco
	call ncoverlap_read('overlap.nc', 10, nco)
!	call ncoverlap_read('overlap.nc')

	print *,nco%grid1%overlap_cells(1:5)
	print *,nco%grid1%native_area(1:5)
	print *,nco%grid1%proj_area(1:5)
	print *,nco%grid1%proj_area(1:5) / nco%grid1%native_area(1:5)
	print *

	print *,nco%grid2%overlap_cells(1:5)
	print *,nco%grid2%native_area(1:5)
	print *,nco%grid2%proj_area(1:5)
	print *,nco%grid2%proj_area(1:5) / nco%grid2%native_area(1:5)
	print *

	print *, nco%overlap_cells(:,1:5)
	print *, nco%area(1:5)



end program test
#endif
