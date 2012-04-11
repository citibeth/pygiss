#pragma once

#include <cmath>


namespace giss {

const double D2R = M_PI / 180.0;
const double R2D = 180.0 / M_PI;

const double EQ_RAD = 6.371e6; /// Radius of the Earth (same as in ModelE)
//const double EQ_RAD = 6370997; /// Radius of the Earth (same as in proj.4, see src/pj_ellps.c)

};

