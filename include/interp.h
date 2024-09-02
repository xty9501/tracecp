#include <vector>
#include <array>
#include "ftk/ndarray.hh"
#include "ftk/numeric/inverse_linear_interpolation_solver.hh"
#include "ftk/numeric/linear_interpolation.hh"
#include <Eigen/Dense>
// 定义一个点
using Point = Eigen::Vector3d;

// 定义一个四面体
using Tetrahedron = std::array<Point, 4>;

double triarea(double a, double b, double c);
double dist(double x0, double y0, double z0, double x1, double y1, double z1);
void barycent2d(double *p0, double *p1, double *p2, const double *v, double *lambda );
bool solve_linear3x3(const double A[3][3], const double b[3], double x[3]);
bool compute_barycentric_coordinates(const double V[4][3], const double P[3], double lambda[4]);
void interp2d(const double p[2], double v[2],const ftk::ndarray<float> &grad);
void interp3d(const double p[3], double *v,const ftk::ndarray<float> &grad);
std::pair<Tetrahedron, Eigen::Vector4d> findTetrahedronVertices(double x, double y, double z);
std::pair<Tetrahedron, Eigen::Vector4d> findTetrahedronAndInterpolate(double x, double y, double z, const ftk::ndarray<float> &grad);
bool isPointInTetrahedron(const Point& p, const Tetrahedron& tetra, Eigen::Vector4d& bary_coords);
void interp3d_new(const double p[3],double *v, const ftk::ndarray<float> &grad);