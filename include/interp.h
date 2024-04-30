#include <vector>
#include "ftk/ndarray.hh"

double triarea(double a, double b, double c);
double dist(double x0, double y0, double z0, double x1, double y1, double z1);
void barycent2d(double *p0, double *p1, double *p2, const double *v, double *lambda );
void interp2d(const double p[2], double v[2],const ftk::ndarray<float> &grad);