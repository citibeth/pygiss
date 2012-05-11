/** C++ API for proj.4 Projection Library.
By Robert Fischer: robert.fischer@nasa.gov
April 5, 2012

This file is in the public domain.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 THE AUTHORS OR AUTHOR'S EMPLOYERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 DEALINGS IN THE SOFTWARE.
*/

#ifndef PROJPP_HPP
#define PROJPP_HPP

#include <proj_api.h>


#if 0
class ProjContext;

class ProjStatic {
public:
    ProjContext const defaultContext;

    ProjStatic();
};
extern ProjStatic projStatic;


class ProjContext {
    projCtx ctx;

public :


};

#endif

namespace giss {

class Proj {
    projPJ pj;

    explicit Proj(projPJ _pj) : pj(_pj) {}

public:
    Proj() : pj(0) {}

    friend int transform(Proj const &src, Proj const &dest,
        long point_count, int point_offset,
        double *x, double *y, double *z);


    // ------------------ Five Standard constructors/methods
    // See: http://www2.research.att.com/~bs/C++0xFAQ.html

    explicit Proj(std::string const &definition)
    {
        pj = pj_init_plus(definition.c_str());
        // pj_def = 0;
    }

    explicit Proj(char const *definition)
    {
        pj = pj_init_plus(definition);
    }

    ~Proj()
    {
        pj_free(pj);
        // if (pj_def) pj_dalloc(pj_def);
    }

    /** Transfer ownership (move) */
    Proj(Proj&& h) : pj{h.pj} //, pj_def{h.pj_def}
    {
        h.pj = 0;
        // h.pj_def = 0;
    }

    /** Transfer value */
    Proj& operator=(Proj&& h)
    {
        if (pj) pj_free(pj);
        pj = h.pj;
        h.pj = 0;
    }

    /** Copy constructor */
    Proj(const Proj &h)
    {
        char *pj_def = pj_get_def(h.pj, 0);
        pj = pj_init_plus(pj_def);
        pj_dalloc(pj_def);
    }

    // no copy with operator=()
    Proj& operator=(const Proj&) = delete;

    // --------------------------- Other Stuff


    /** Returns TRUE if the passed coordinate system is geographic
    (proj=latlong). */
    int is_latlong() const
        { return pj_is_latlong(pj); }


    /** Returns TRUE if the coordinate system is geocentric
    (proj=geocent). */
    int is_geocent() const
        { return pj_is_geocent(pj); }

    /** Returns the PROJ.4 initialization string suitable for use with
    pj_init_plus() that would produce this coordinate system, but with the
    definition expanded as much as possible (for instance +init= and
    +datum= definitions).
    @param options Unused at this point
    */
    std::string get_def(int options=0) const
    {
        char *pj_def = 0;
        pj_def = pj_get_def(pj, options);

        std::string ret = std::string(pj_def);
        pj_dalloc(pj_def);
        return ret;
    }


    /** Returns a new coordinate system definition which is the geographic
    coordinate (lat/long) system underlying pj_in. */
    Proj latlong_from_proj() const
    {
        return Proj(pj_latlong_from_proj(pj));
    }

};


inline int transform(Proj const &src, Proj const &dest,
    long point_count, int point_offset, double *x, double *y, double *z=0)
{
    return pj_transform(src.pj, dest.pj,
        point_count, point_offset, x, y, z);
}

inline int transform(Proj const &src, Proj const &dest,
    double x0, double y0, double &x1, double &y1)
{
    x1 = x0;
    y1 = y0;
    int ret = transform(src, dest, 1, 1, &x1, &y1);
    return ret;
}


}

#endif
