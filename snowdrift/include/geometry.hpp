#pragma once

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Boolean_set_operations_2.h>
#include <CGAL/bounding_box.h>

namespace giss {

namespace gc {
	typedef CGAL::Exact_predicates_exact_constructions_kernel Kernel;
	typedef Kernel::Point_2                                   Point_2;
	typedef Kernel::Iso_rectangle_2                           Iso_rectangle_2;
	typedef CGAL::Polygon_2<Kernel>                           Polygon_2;
	typedef CGAL::Polygon_with_holes_2<Kernel> Polygon_with_holes_2;
}


/**
Computes the overlap area of two linear simple polygons. Uses BSO and then add the area of the polygon and substract
the area of its holes.
@param P The first polygon.
@param Q The second polygon.
@return The area of the overlap between P and Q.
@see acg.cs.tau.ac.il/courses/workshop/spring-2008/useful-routines.h
*/
template <class Kernel, class Container>
typename CGAL::Polygon_2<Kernel, Container>::FT
overlap_area(const CGAL::Polygon_2<Kernel, Container> &P, 
	const CGAL::Polygon_2<Kernel, Container> &Q)
{
	CGAL_precondition(P.is_simple());
	CGAL_precondition(Q.is_simple());

	typedef typename CGAL::Polygon_2<Kernel, Container>::FT FT;
	typedef CGAL::Polygon_with_holes_2<Kernel, Container> Polygon_with_holes_2;
	typedef std::list<Polygon_with_holes_2> Pol_list;
	Pol_list overlap;
	CGAL::intersection(P, Q, std::back_inserter(overlap));
	if (overlap.empty())
		return 0;
	
	// summing the areas and reducing the area of holes.
	FT result = 0;
	for (typename Pol_list::iterator it = overlap.begin(); it != overlap.end(); ++it)
	{
		Polygon_with_holes_2& cur = *it;
		result += cur.outer_boundary().area();
//std::cout << cur.outer_boundary().area() << std::endl;
		for (typename Polygon_with_holes_2::Hole_const_iterator jt = cur.holes_begin();
			jt != cur.holes_end(); ++jt)
		{
			result -= jt->area();
		}
	}	

	return result;
}


}
