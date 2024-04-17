#include "cp.hpp"

template<typename Type>
std::array<Type, 2> newRK4(const Type *x, const Type *v, const ftk::ndarray<double> &data, Type h, int DH, int DW, std::set<size_t>& lossless_index);

template<typename Type>
std::array<Type, 2> newRK4(const Type * x, const Type * v, const ftk::ndarray<double> &data,  Type h, const int DH, const int DW);

template<typename Type>
std::set<size_t> vertex_for_each_RK4(const Type *x, const Type *v, const ftk::ndarray<double> &data, double h, const int DH, const int DW);

template<typename Type>
std::array<Type, 2> rkf45(const Type * x, const Type * v, const ftk::ndarray<double> &data,  Type h, const int DH, const int DW,std::set<size_t>& lossless_index,double tolerance = 0.0005);

std::vector<std::array<double, 2>> trajectory(double *X_original,const std::array<double, 2>& initial_x, const double time_step, const int DH,const int DW, const std::unordered_map<int, critical_point_t>& critical_points, ftk::ndarray<double>& data  ,std::vector<int>& index, std::vector<double>& config, std::vector<record_t>& record,std::set<size_t>& lossless_index);

std::vector<std::array<double, 2>> trajectory(double *X_original,const std::array<double, 2>& initial_x, const double time_step, const int max_length, const int DH,const int DW, const std::unordered_map<int, critical_point_t>& critical_points, ftk::ndarray<double>& data  ,std::vector<int>& index);

std::vector<std::array<double, 2>> trajectory(double *X_original,const std::array<double, 2>& initial_x, const double time_step, const int max_length, const int DH,const int DW, const std::unordered_map<int, critical_point_t>& critical_points, ftk::ndarray<double>& data  ,std::vector<int>& index,std::set<size_t>& lossless_index);