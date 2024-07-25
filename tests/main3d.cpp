#include "utils.hpp"
#include <interp.h>
#include <utilsIO.h>
#include <ftk/numeric/print.hh>
#include <ftk/numeric/cross_product.hh>
#include <ftk/numeric/vector_norm.hh>
#include <ftk/numeric/linear_interpolation.hh>
#include <ftk/numeric/bilinear_interpolation.hh>
#include <ftk/numeric/inverse_linear_interpolation_solver.hh>
#include <ftk/numeric/inverse_bilinear_interpolation_solver.hh>
#include <ftk/numeric/gradient.hh>
#include <ftk/numeric/matrix_multiplication.hh>
#include <ftk/numeric/matrix_inverse.hh>
#include <ftk/numeric/clamp.hh>
// #include <ftk/numeric/eigen_solver2.hh>
#include <ftk/numeric/eigen_solver3.hh>
#include <ftk/numeric/linear_solver.hh>
#include <ftk/numeric/linear_solver1.hh>
#include <ftk/algorithms/cca.hh>
#include <ftk/geometry/cc2curves.hh>
#include <ftk/geometry/curve2tube.hh>
#include "ftk/ndarray.hh"
#include "ftk/numeric/critical_point_type.hh"
#include "ftk/numeric/critical_point_test.hh"
#include "cp.hpp"
#include <chrono>
#include "advect.hpp"
#include <math.h>
#include <Eigen/Dense>


// #include <hypermesh/ndarray.hh>
// #include <hypermesh/regular_simplex_mesh.hh>
#include <mutex>
// #include <hypermesh/ndarray.hh>
// #include <hypermesh/regular_simplex_mesh.hh>

#include <ftk/ndarray/ndarray_base.hh>
#include <unordered_map>
#include <queue>
#include <fstream>
#include <utility>
#include <set>

#include "sz_compress_cp_preserve_3d.hpp"
#include "sz_decompress_cp_preserve_3d.hpp"
#include "sz_lossless.hpp"

#include <omp.h>
using namespace std;


struct critical_point_t_3d {
  double x[3];
  int type;
  size_t simplex_id;
  double eigvalues[3];
  double eig_vec[3][3];
  critical_point_t_3d(){}
};

#define SINGULAR 0
#define STABLE_SOURCE 1
#define UNSTABLE_SOURCE 2
#define STABLE_REPELLING_SADDLE 3
#define UNSTABLE_REPELLING_SADDLE 4
#define STABLE_ATRACTTING_SADDLE  5
#define UNSTABLE_ATRACTTING_SADDLE  6
#define STABLE_SINK 7
#define UNSTABLE_SINK 8



template<typename T>
int
get_cp_type(const T X[4][3], const T U[4][3]){
  const T X_[3][3] = {
    {X[0][0] - X[3][0], X[1][0] - X[3][0], X[2][0] - X[3][0]}, 
    {X[0][1] - X[3][1], X[1][1] - X[3][1], X[2][1] - X[3][1]},
    {X[0][2] - X[3][2], X[1][2] - X[3][2], X[2][2] - X[3][2]}    
  };
  const T U_[3][3] = {
    {U[0][0] - U[3][0], U[1][0] - U[3][0], U[2][0] - U[3][0]}, 
    {U[0][1] - U[3][1], U[1][1] - U[3][1], U[2][1] - U[3][1]},
    {U[0][2] - U[3][2], U[1][2] - U[3][2], U[2][2] - U[3][2]}    
  };
  T inv_X_[3][3];
  ftk::matrix_inverse3x3(X_, inv_X_);
  T J[3][3];
  ftk::matrix3x3_matrix3x3_multiplication(inv_X_, U_, J);
  T P[4];
  ftk::characteristic_polynomial_3x3(J, P);
  std::complex<T> root[3];
  T disc = ftk::solve_cubic(P[2], P[1], P[0], root);
  if(fabs(disc) < std::numeric_limits<T>::epsilon()) return SINGULAR;
  int negative_real_parts = 0;
  for(int i=0; i<3; i++){
    negative_real_parts += (root[i].real() < 0);
  }
  switch(negative_real_parts){
    case 0:
      return (disc > 0) ? UNSTABLE_SOURCE : STABLE_SOURCE;
    case 1:
      return (disc > 0) ? UNSTABLE_REPELLING_SADDLE : STABLE_REPELLING_SADDLE;
    case 2:
      return (disc > 0) ? UNSTABLE_ATRACTTING_SADDLE : STABLE_ATRACTTING_SADDLE;
    case 3:
      return (disc > 0) ? UNSTABLE_SINK : STABLE_SINK;
    default:
      return SINGULAR;
  }
}


template<typename T_acc, typename T>
static inline void 
update_value(T_acc v[4][3], int local_id, int global_id, const T * U, const T * V, const T * W){
  v[local_id][0] = U[global_id];
  v[local_id][1] = V[global_id];
  v[local_id][2] = W[global_id];
}

template<typename T>
static inline void 
update_index_and_value(double v[4][3], int indices[4], int local_id, int global_id, const T * U, const T * V, const T * W){
  indices[local_id] = global_id;
  update_value(v, local_id, global_id, U, V, W);
}

void computeEigenvaluesAndEigenvectors(const double (&A)[3][3], double (&eigenvalues)[3], double (&eigenvectors)[3][3]) {
    // 将原生数组转换为 Eigen 矩阵
    Eigen::Matrix3d matrix;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            matrix(i, j) = A[i][j];

    // 使用 EigenSolver 计算特征值和特征向量
    Eigen::EigenSolver<Eigen::Matrix3d> solver(matrix);
    
    // 获取特征值
    Eigen::Vector3d eigenvalues_vec = solver.eigenvalues().real();
    
    // 获取特征向量
    Eigen::Matrix3d eigenvectors_mat = solver.eigenvectors().real();

    // 将 Eigen 的结果复制到原生数组
    for (int i = 0; i < 3; ++i) {
        eigenvalues[i] = eigenvalues_vec[i];
        for (int j = 0; j < 3; ++j) {
            eigenvectors[i][j] = eigenvectors_mat(i, j);
        }
    }
}

void 
check_simplex_seq(const double v[4][3], const double X[3][3], const int indices[4], int i, int j, int k, int simplex_id, std::unordered_map<int, critical_point_t_3d>& critical_points){
  double mu[4]; // check intersection
  double cond;
  // robust critical point test
//   bool succ = ftk::robust_critical_point_in_simplex3(vf, indices);
//   if (!succ) return;
  for (int i = 0; i < 4; i++) {
    if (v[i][0] == 0 && v[i][1] == 0 && v[i][2] == 0) {
      return;
    }
  }
  bool succ2 = ftk::inverse_lerp_s3v3(v, mu, &cond);
//   if(!succ2) ftk::clamp_barycentric<4>(mu);
  if (!succ2) return;
  double x[3]; // position
  ftk::lerp_s3v3(X, mu, x);
  critical_point_t_3d cp;
  cp.x[0] = k + x[0]; cp.x[1] = j + x[1]; cp.x[2] = i + x[2];
  cp.type = get_cp_type(X, v);
  cp.simplex_id = simplex_id;
  double J[3][3]; // jacobian
  double eigenvalues[3];
  double eigenvec[3][3];
  ftk::jacobian_3dsimplex(X, v, J);
  computeEigenvaluesAndEigenvectors(J, eigenvalues, eigenvec);
  //copy eigenvec to cp.eig_vec
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
        cp.eig_vec[i][j] = eigenvec[i][j];
        }
    }
    //copy eigenvalues to cp.eig
    for(int i=0; i<3; i++){
        cp.eigvalues[i] = eigenvalues[i];
    }
  critical_points[simplex_id] = cp;
}

template<typename T>
std::unordered_map<int, critical_point_t_3d>
compute_critical_points(const T * U, const T * V, const T * W, int r1, int r2, int r3){
  // check cp for all cells
  ptrdiff_t dim0_offset = r2*r3;
  ptrdiff_t dim1_offset = r3;
  ptrdiff_t cell_dim0_offset = (r2-1)*(r3-1);
  ptrdiff_t cell_dim1_offset = r3-1;
  size_t num_elements = r1*r2*r3;
  int indices[4] = {0};
  double v[4][3] = {0};
  double actual_coords[6][4][3];
  for(int i=0; i<6; i++){
    for(int j=0; j<4; j++){
      for(int k=0; k<3; k++){
        actual_coords[i][j][k] = tet_coords[i][j][k];
      }
    }
  }
  std::unordered_map<int, critical_point_t_3d> critical_points;
  for(int i=0; i<r1-1; i++){
    if(i%10==0) std::cout << i << " / " << r1-1 << std::endl;
    for(int j=0; j<r2-1; j++){
      for(int k=0; k<r3-1; k++){
        // order (reserved, z->x):
        // ptrdiff_t cell_offset = 6*(i*cell_dim0_offset + j*cell_dim1_offset + k);
        // ftk index
        ptrdiff_t cell_offset = 6*(i*dim0_offset + j*dim1_offset + k);
        // (ftk-0) 000, 001, 011, 111
        update_index_and_value(v, indices, 0, i*dim0_offset + j*dim1_offset + k, U, V, W);
        update_index_and_value(v, indices, 1, (i+1)*dim0_offset + j*dim1_offset + k, U, V, W);
        update_index_and_value(v,indices, 2, (i+1)*dim0_offset + (j+1)*dim1_offset + k, U, V, W);
        update_index_and_value(v, indices, 3, (i+1)*dim0_offset + (j+1)*dim1_offset + (k+1), U, V, W);
        check_simplex_seq(v, actual_coords[0], indices, i, j, k, cell_offset, critical_points);
        // (ftk-2) 000, 010, 011, 111
        update_index_and_value(v,indices, 1, i*dim0_offset + (j+1)*dim1_offset + k, U, V, W);
        check_simplex_seq(v, actual_coords[1], indices, i, j, k, cell_offset + 2, critical_points);
        // (ftk-1) 000, 001, 101, 111
        update_index_and_value(v,indices, 1, (i+1)*dim0_offset + j*dim1_offset + k, U, V, W);
        update_index_and_value(v, indices, 2, (i+1)*dim0_offset + j*dim1_offset + k+1, U, V, W);
        check_simplex_seq(v, actual_coords[2], indices, i, j, k, cell_offset + 1, critical_points);
        // (ftk-4) 000, 100, 101, 111
        update_index_and_value(v, indices, 1, i*dim0_offset + j*dim1_offset + k+1, U, V, W);
        check_simplex_seq(v, actual_coords[3], indices, i, j, k, cell_offset + 4, critical_points);
        // (ftk-3) 000, 010, 110, 111
        update_index_and_value(v,indices, 1, i*dim0_offset + (j+1)*dim1_offset + k, U, V, W);
        update_index_and_value(v,indices, 2, i*dim0_offset + (j+1)*dim1_offset + k+1, U, V, W);
        check_simplex_seq(v, actual_coords[4], indices, i, j, k, cell_offset + 3, critical_points);
        // (ftk-5) 000, 100, 110, 111
        update_index_and_value(v,indices, 1, i*dim0_offset + j*dim1_offset + k+1, U, V, W);
        check_simplex_seq(v, actual_coords[5], indices, i, j, k, cell_offset + 5, critical_points);
      }
    }
  }
  return critical_points; 
}



template<typename Container>
inline bool inside(const Container& x, int DH, int DW, int DD) {
  return (x[0] > 0 && x[0] < DH-1 && x[1] > 0 && x[1] < DW-1 && x[2] > 0 && x[2] < DD-1);
}
template<typename Type>
std::array<Type, 3> newRK4_3d(const Type * x, const Type * v, const ftk::ndarray<float> &data,  Type h, const int DH, const int DW, const int DD, std::set<size_t>& lossless_index,std::unordered_map<size_t, std::set<int>>& cellID_trajIDs_map,size_t trajID) {
  // x and y are positions, and h is the step size
  double rk1[3] = {0};
  const double p1[3] = {x[0], x[1],x[2]};

  auto coords = get_four_offsets(x, DW, DH,DD);
  for (auto offset:coords){
    lossless_index.insert(offset);
  }
  cellID_trajIDs_map[get_cell_offset_3d(x, DW, DH,DD)].insert(trajID);

  if(!inside(p1, DH, DW, DD)){
    //return std::array<Type, 2>{x[0], x[1]};
    return std::array<Type, 3>{-1, -1,-1};
  }
  interp3d(p1, rk1,data);
  coords = get_four_offsets(p1, DW, DH,DD);
  for (auto offset:coords){
    lossless_index.insert(offset);
  }
  //修改成3d
  cellID_trajIDs_map[get_cell_offset_3d(p1, DW, DH,DD)].insert(trajID);
  
  double rk2[3] = {0};
  const double p2[3] = {x[0] + 0.5 * h * rk1[0], x[1] + 0.5 * h * rk1[1],x[2] + 0.5 * h * rk1[2]};
  if (!inside(p2, DH, DW, DD)){
    //return std::array<Type, 2>{p1[0], p1[1]};
    return std::array<Type, 3>{-1, -1,-1};
  }
  interp3d(p2, rk2,data);
  coords = get_four_offsets(p2, DW, DH,DD);
  for (auto offset:coords){
    lossless_index.insert(offset);
  }
  cellID_trajIDs_map[get_cell_offset_3d(p2, DW, DH,DD)].insert(trajID);
  
  double rk3[3] = {0};
  const double p3[3] = {x[0] + 0.5 * h * rk2[0], x[1] + 0.5 * h * rk2[1],x[2] + 0.5 * h * rk2[2]};
  if (!inside(p3, DH, DW, DD)){
    return std::array<Type, 3>{-1, -1,-1};
  }
  interp3d(p3, rk3,data);
  coords = get_four_offsets(p3, DW, DH,DD);
  for (auto offset:coords){
    lossless_index.insert(offset);
  }
  cellID_trajIDs_map[get_cell_offset_3d(p3, DW, DH,DD)].insert(trajID);
  
  double rk4[3] = {0};
  const double p4[3] = {x[0] + h * rk3[0], x[1] + h * rk3[1],x[2] + h * rk3[2]};
  if (!inside(p4, DH, DW,DD)){
    return std::array<Type, 3>{-1, -1,-1};
  }
  interp3d(p4, rk4,data);
  coords = get_four_offsets(p4, DW, DH,DD);
  for (auto offset:coords){
    lossless_index.insert(offset);
  }
  cellID_trajIDs_map[get_cell_offset_3d(p4, DW, DH,DD)].insert(trajID);
  
  Type next_x = x[0] + h * (rk1[0] + 2 * rk2[0] + 2 * rk3[0] + rk4[0]) / 6.0;
  Type next_y = x[1] + h * (rk1[1] + 2 * rk2[1] + 2 * rk3[1] + rk4[1]) / 6.0;
  Type next_z = x[2] + h * (rk1[2] + 2 * rk2[2] + 2 * rk3[2] + rk4[2]) / 6.0;
  if (!inside(std::array<Type, 3>{next_x, next_y,next_z}, DH, DW,DD)){
    return std::array<Type, 3>{-1, -1,-1};
  }
  std::array<Type, 3> result = {next_x, next_y,next_z};
  coords = get_four_offsets(result, DW, DH, DD);
  for (auto offset:coords){
    lossless_index.insert(offset);
  }
  cellID_trajIDs_map[get_cell_offset_3d(result.data(), DW, DH,DD)].insert(trajID);
  return result;
}


// 缓存get_four_offsets结果的函数，需要加锁
template<typename Type>
void updateOffsetsAndMap(const Type* p, const int DW, const int DH, const int DD, std::set<size_t>& lossless_index, std::unordered_map<size_t, std::set<int>>& cellID_trajIDs_map, size_t trajID) {
    auto coords = get_four_offsets(p, DW, DH, DD);
    #pragma omp critical
    {
        for (auto offset : coords) {
            lossless_index.insert(offset);
            cellID_trajIDs_map[offset].insert(trajID);
        }
    }
}

// newRK4_3d 函数
//这个不能并行化
template<typename Type>
std::array<Type, 3> newRK4_3d_parallel(const Type* x, const Type* v, const ftk::ndarray<float>& data, Type h, const int DH, const int DW, const int DD, std::set<size_t>& lossless_index, std::unordered_map<size_t, std::set<int>>& cellID_trajIDs_map, size_t trajID) {
    double rk1[3] = {0}, rk2[3] = {0}, rk3[3] = {0}, rk4[3] = {0};
    const double p1[3] = {x[0], x[1], x[2]};

    if (!inside(p1, DH, DW, DD)) return {-1, -1, -1};
    updateOffsetsAndMap(p1, DW, DH, DD, lossless_index, cellID_trajIDs_map, trajID);
    interp3d(p1, rk1, data);

    const double p2[3] = {x[0] + 0.5 * h * rk1[0], x[1] + 0.5 * h * rk1[1], x[2] + 0.5 * h * rk1[2]};
    if (!inside(p2, DH, DW, DD)) return {-1, -1, -1};
    updateOffsetsAndMap(p2, DW, DH, DD, lossless_index, cellID_trajIDs_map, trajID);
    interp3d(p2, rk2, data);

    const double p3[3] = {x[0] + 0.5 * h * rk2[0], x[1] + 0.5 * h * rk2[1], x[2] + 0.5 * h * rk2[2]};
    if (!inside(p3, DH, DW, DD)) return {-1, -1, -1};
    updateOffsetsAndMap(p3, DW, DH, DD, lossless_index, cellID_trajIDs_map, trajID);
    interp3d(p3, rk3, data);

    const double p4[3] = {x[0] + h * rk3[0], x[1] + h * rk3[1], x[2] + h * rk3[2]};
    if (!inside(p4, DH, DW, DD)) return {-1, -1, -1};
    updateOffsetsAndMap(p4, DW, DH, DD, lossless_index, cellID_trajIDs_map, trajID);
    interp3d(p4, rk4, data);

    Type next_x = x[0] + h * (rk1[0] + 2 * rk2[0] + 2 * rk3[0] + rk4[0]) / 6.0;
    Type next_y = x[1] + h * (rk1[1] + 2 * rk2[1] + 2 * rk3[1] + rk4[1]) / 6.0;
    Type next_z = x[2] + h * (rk1[2] + 2 * rk2[2] + 2 * rk3[2] + rk4[2]) / 6.0;

    std::array<Type, 3> result = {next_x, next_y, next_z};
    if (!inside(result, DH, DW, DD)) return {-1, -1, -1};

    updateOffsetsAndMap(result.data(), DW, DH, DD, lossless_index, cellID_trajIDs_map, trajID);
    return result;
}


//并行化trajectory_3d可以
std::vector<std::array<double, 3>> trajectory_3d(double *X_original,const std::array<double, 3>& initial_x, const double time_step, const int max_length, const int DH,const int DW, const int DD,const std::unordered_map<int, critical_point_t_3d>& critical_points, ftk::ndarray<float>& data  ,std::vector<int>& index,std::set<size_t>& lossless_index,std::unordered_map<size_t, std::set<int>>& cellID_trajIDs_map, size_t trajID){
  std::vector<std::array<double, 3>> result;
  int flag = 0; // 1 means found, -1 means out of bound， 0 means reach max length
  int length = 0;
  result.push_back({X_original[0], X_original[1],X_original[2]}); //add original true position
  length ++;
  int orginal_offset = get_cell_offset_3d(X_original, DW, DH,DD);
  cellID_trajIDs_map[orginal_offset].insert(trajID); //add original cp to map 
  //printf("break1\n");
  

  std::array<double, 3> current_x = initial_x;

  //add original and initial_x position's offset
  auto ori_offset = get_four_offsets(X_original, DW, DH,DD);
  for (auto offset:ori_offset){
    lossless_index.insert(offset);
  }  

  if(!inside(current_x, DH, DW,DD)){
    //count_out_bound ++;
    flag = -1;
    result.push_back({-1, -1,-1});
    length ++;
    index.push_back(length);
    return result;
  }
  else{
    result.push_back(current_x); //add initial position(seed)
    length ++;
    cellID_trajIDs_map[get_cell_offset_3d(current_x.data(), DW, DH,DD)].insert(trajID); //add seed to map
    auto ini_offset = get_four_offsets(current_x, DW, DH,DD);
    for (auto offset:ini_offset){
      lossless_index.insert(offset);
    }
  }
  //printf("break2\n");
  //auto start_while = std::chrono::high_resolution_clock::now();
  double rk4_time_count = 0;
  while (flag == 0){
    //printf("current_x: %f %f %f\n", current_x[0], current_x[1],current_x[2]);
    if(!inside(current_x, DH, DW,DD)){
      //count_out_bound ++;
      //printf("out of bound!\n");
      flag = -1;
      result.push_back({-1, -1,-1});
      length ++;
      break;
    }
    if (length == max_length) {
      //printf("reach max length!\n");
      //count_limit ++;
      flag = 1;
      break;
    }

    double current_v[3] = {0};

    interp3d(current_x.data(), current_v,data); 

    //int current_offset = get_cell_offset(current_x.data(), DW, DH);    
    
    //auto rk4_start = std::chrono::high_resolution_clock::now();
    //std::array<double, 3> RK4result = newRK4_3d(current_x.data(), current_v, data, time_step, DH, DW,DD,lossless_index,cellID_trajIDs_map,trajID);
    std::array<double, 3> RK4result = newRK4_3d_parallel(current_x.data(), current_v, data, time_step, DH, DW,DD,lossless_index,cellID_trajIDs_map,trajID);
    //auto rk4_end = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double, std::milli> duration_rk4 = rk4_end - rk4_start;
    //rk4_time_count += duration_rk4.count();
    //printf("rk4result(next): %f %f %f\n", RK4result[0], RK4result[1],RK4result[2]);
    // rk4步骤似乎不需要将中间值加入到map中？
    // std::unordered_map<size_t, std::set<int>> temp_map;
    // std::array<double, 2> RK4result = newRK4(current_x.data(), current_v, data, time_step, DH, DW,lossless_index,temp_map,trajID);
    //std::array<double, 2> RK4result = rkf45(current_x.data(), current_v, data, time_step, DH, DW,lossless_index);
    
    if (RK4result[0] == -1 && RK4result[1] == -1 && RK4result[2] == -1){
      //count_out_bound ++;
      flag = -1;
      result.push_back({-1, -1,-1});
      length ++;
      break;
    }

    size_t current_offset = get_cell_offset_3d(RK4result.data(), DW, DH, DD);

    if (current_offset != orginal_offset){
      //moved to another cell
      auto it = critical_points.find(current_offset);
      if (it != critical_points.end()){
        auto cp = it->second;
        double error = 1e-2;
        if (cp.type < 3 || cp.type >6 && fabs(RK4result[0] - cp.x[0]) < error && fabs(RK4result[1] - cp.x[1]) < error){
          flag = 1; //found cp
          int cp_offset = get_cell_offset_3d(cp.x, DW, DH,DD);
          // first add rk4 position
          result.push_back({RK4result[0], RK4result[1],RK4result[2]});
          length++;
          cellID_trajIDs_map[current_offset].insert(trajID); //add element to map
          // then add cp position
          std::array<double, 3> true_cp = {cp.x[0], cp.x[1],cp.x[2]};
          result.push_back(true_cp);
          length++;
          cellID_trajIDs_map[get_cell_offset_3d(cp.x, DW, DH,DD)].insert(trajID); //add element to map
          index.push_back(length);
          return result;
        }
      }
    }
    current_x = RK4result;
    result.push_back(current_x);
    length++;
    cellID_trajIDs_map[current_offset].insert(trajID); //add element to map
  }
  //printf("break3\n");
  //auto end_while = std::chrono::high_resolution_clock::now();
  //std::chrono::duration<double, std::milli> duration_while = end_while - start_while;
  //printf("        while loop time: %f ms, rk4 total count %f\n", duration_while.count(), rk4_time_count);

  index.push_back(length);
  //printf("break4\n");
  return result;
}


std::vector<std::array<double, 3>> trajectory_3d_parallel(double *X_original, const std::array<double, 3>& initial_x, const double time_step, const int max_length, const int DH, const int DW, const int DD, const std::unordered_map<int, critical_point_t_3d>& critical_points, ftk::ndarray<float>& data, std::vector<int>& index, std::set<size_t>& lossless_index, std::unordered_map<size_t, std::set<int>>& cellID_trajIDs_map, size_t trajID) {
    std::vector<std::array<double, 3>> result;
    int flag = 0; // 1 means found, -1 means out of bound, 0 means reach max length
    int length = 0;
    result.push_back({X_original[0], X_original[1], X_original[2]}); // add original true position
    length++;
    int original_offset = get_cell_offset_3d(X_original, DW, DH, DD);
    #pragma omp critical
    {
        cellID_trajIDs_map[original_offset].insert(trajID); // add original cp to map
    }

    std::array<double, 3> current_x = initial_x;

    // add original and initial_x position's offset
    auto ori_offset = get_four_offsets(X_original, DW, DH, DD);
    #pragma omp critical
    {
        for (auto offset : ori_offset) {
            lossless_index.insert(offset);
        }
    }

    if (!inside(current_x, DH, DW, DD)) {
        flag = -1;
        result.push_back({-1, -1, -1});
        length++;
        #pragma omp critical
        {
            index.push_back(length);
        }
        return result;
    } else {
        result.push_back(current_x); // add initial position(seed)
        length++;
        #pragma omp critical
        {
            cellID_trajIDs_map[get_cell_offset_3d(current_x.data(), DW, DH, DD)].insert(trajID); // add seed to map
            auto ini_offset = get_four_offsets(current_x.data(), DW, DH, DD);
            for (auto offset : ini_offset) {
                lossless_index.insert(offset);
            }
        }
    }

    double rk4_time_count = 0;
    while (flag == 0) {
        if (!inside(current_x, DH, DW, DD)) {
            flag = -1;
            result.push_back({-1, -1, -1});
            length++;
            break;
        }
        if (length == max_length) {
            flag = 1;
            break;
        }

        double current_v[3] = {0};
        interp3d(current_x.data(), current_v, data);

        std::array<double, 3> RK4result = newRK4_3d_parallel(current_x.data(), current_v, data, time_step, DH, DW, DD, lossless_index, cellID_trajIDs_map, trajID);

        if (RK4result[0] == -1 && RK4result[1] == -1 && RK4result[2] == -1) {
            flag = -1;
            result.push_back({-1, -1, -1});
            length++;
            break;
        }

        size_t current_offset = get_cell_offset_3d(RK4result.data(), DW, DH, DD);

        if (current_offset != original_offset) {
            auto it = critical_points.find(current_offset);
            if (it != critical_points.end()) {
                auto cp = it->second;
                double error = 1e-3;
                if (cp.type < 3 || cp.type > 6 && fabs(RK4result[0] - cp.x[0]) < error && fabs(RK4result[1] - cp.x[1]) < error) {
                    flag = 1; // found cp
                    int cp_offset = get_cell_offset_3d(cp.x, DW, DH, DD);
                    result.push_back({RK4result[0], RK4result[1], RK4result[2]});
                    length++;
                    #pragma omp critical
                    {
                        cellID_trajIDs_map[current_offset].insert(trajID); // add element to map
                    }
                    std::array<double, 3> true_cp = {cp.x[0], cp.x[1], cp.x[2]};
                    result.push_back(true_cp);
                    length++;
                    #pragma omp critical
                    {
                        cellID_trajIDs_map[get_cell_offset_3d(cp.x, DW, DH, DD)].insert(trajID); // add element to map
                        index.push_back(length);
                    }
                    printf("reaching cp: %f, %f, %f\n", cp.x[0], cp.x[1], cp.x[2]);
                    return result;
                }
            }
        }
        current_x = RK4result;
        result.push_back(current_x);
        length++;
        #pragma omp critical
        {
            cellID_trajIDs_map[current_offset].insert(trajID); // add element to map
        }
    }

    #pragma omp critical
    {
        index.push_back(length);
    }

    return result;
}

int main(int argc, char ** argv){
    omp_set_num_threads(10);
    size_t num_elements = 0;
    float * U = readfile<float>(argv[1], num_elements);
    float * V = readfile<float>(argv[2], num_elements);
    float * W = readfile<float>(argv[3], num_elements);
    int r1 = atoi(argv[4]);
    int r2 = atoi(argv[5]);
    int r3 = atoi(argv[6]);
    double max_eb = atof(argv[7]);
    double h = 0.1;
    double eps = 0.01;
    int max_length = 1000;
    // cout << U[r2 + 3] << " " << U[3*r2 + 1] << endl;
    // transpose_2d(U, r1, r2);
    // cout << U[r2 + 3] << " " << U[3*r2 + 1] << endl;
    // transpose_2d(V, r1, r2);
    auto critical_points_0 = compute_critical_points(U, V, W, r1, r2, r3);
    cout << "critical points: " << critical_points_0.size() << endl;

    size_t result_size = 0;
    struct timespec start, end;
    int err = 0;
    err = clock_gettime(CLOCK_REALTIME, &start);
    cout << "start Compression\n";
    // unsigned char * result =  sz_compress_cp_preserve_3d_offline_log(U, V, W, r1, r2, r3, result_size, false, max_eb);
    unsigned char * result =  sz_compress_cp_preserve_3d_online_log(U, V, W, r1, r2, r3, result_size, false, max_eb);
    // exit(0);
    unsigned char * result_after_lossless = NULL;
    size_t lossless_outsize = sz_lossless_compress(ZSTD_COMPRESSOR, 3, result, result_size, &result_after_lossless);
    err = clock_gettime(CLOCK_REALTIME, &end);
    cout << "Compression time: " << (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000 << "s" << endl;
    cout << "Compressed size = " << lossless_outsize << ", ratio = " << (3*num_elements*sizeof(float)) * 1.0/lossless_outsize << endl;
    free(result);
    // exit(0);
    err = clock_gettime(CLOCK_REALTIME, &start);
    size_t lossless_output = sz_lossless_decompress(ZSTD_COMPRESSOR, result_after_lossless, lossless_outsize, &result, result_size);
    float * dec_U = NULL;
    float * dec_V = NULL;
    float * dec_W = NULL;
    sz_decompress_cp_preserve_3d_online_log<float>(result, r1, r2, r3, dec_U, dec_V, dec_W);
    err = clock_gettime(CLOCK_REALTIME, &end);
    cout << "Decompression time: " << (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec)/(double)1000000000 << "s" << endl;
    free(result_after_lossless);


    auto critical_points_out = compute_critical_points(dec_U,dec_V,dec_W, r1, r2, r3);
    cout << "critical points dec: " << critical_points_out.size() << endl;

    ftk::ndarray<float> grad_ori;
    grad_ori.reshape({3, static_cast<unsigned long>(r3),static_cast<unsigned long>(r2), static_cast<unsigned long>(r1)});//500,500,100
    refill_gradient_3d(0, r1, r2, r3, U, grad_ori);
    refill_gradient_3d(1, r1, r2, r3, V, grad_ori);
    refill_gradient_3d(2, r1, r2, r3, W, grad_ori);
    ftk::ndarray<float> grad_dec;
    grad_dec.reshape({3, static_cast<unsigned long>(r3),static_cast<unsigned long>(r2), static_cast<unsigned long>(r1)});//500,500,100
    refill_gradient_3d(0, r1, r2, r3, dec_U, grad_dec);
    refill_gradient_3d(1, r1, r2, r3, dec_V, grad_dec);
    refill_gradient_3d(2, r1, r2, r3, dec_W, grad_dec);
    printf("ori: 1,499,99,99 is %f\n",grad_ori(1,499,99,99));//(DD,DW,DH) (r3,r2,r1)
    printf("dec: 1,499,99,99 is %f\n",grad_dec(1,499,99,99));
    printf("ori: 1,38,48,58 is %f\n",grad_ori(1,38,48,58));
    printf("dec: 1,38,48,58 is %f\n",grad_dec(1,38,48,58));
    //test interp3d
    double test_pt[3] = {3.052532,277.932877,95.424966};//3.052532,277.932877,97.424966
    double test_out_val[3] = {0};

    auto interp_start = std::chrono::high_resolution_clock::now();
    interp3d(test_pt, test_out_val,grad_ori);
    auto interp_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_interp = interp_end - interp_start;
    printf("interp3d: %f %f %f\n", test_out_val[0], test_out_val[1],test_out_val[2]);
    printf("interp3d time: %f ms\n", duration_interp.count());



    //exit(0);

    //*************先计算一次整体的traj_ori 和traj_dec,后续只需增量修改*************

    //*************计算原始数据的traj_ori*************
    size_t found = 0;
    size_t outside = 0;
    size_t max_iter = 0;
    size_t total_saddle_count = 0;
    std::vector<std::vector<std::array<double, 3>>> trajs_ori;
    std::vector<int> index_ori;
    std::set<size_t> vertex_ori;
    std::unordered_map<size_t, std::set<int>> cellID_trajIDs_map_ori;
    std::atomic<size_t> trajID_counter(0);
    auto start1 = std::chrono::high_resolution_clock::now();

    std::vector<int> keys;
    for (const auto &p : critical_points_0) {
        keys.push_back(p.first);
    }
    printf("starting parallel for...\n");
    // for (const auto&p:critical_points_0){
    #pragma omp parallel for reduction(+:total_saddle_count)
    for (size_t i = 0; i < keys.size(); ++i) {
        int key = keys[i];
        printf("current key: %d,current thread: %d\n",key,omp_get_thread_num());
        auto &cp = critical_points_0[key];
        if (cp.type >=3 && cp.type <= 6){
            //auto start_six_traj = std::chrono::high_resolution_clock::now();
            total_saddle_count ++;
            //cout << "critical point saddle at " << cp.x[0] << " " << cp.x[1] << " " << cp.x[2] << " type: " << cp.type << endl;
            //print eigenvector
            // cout << "eigenvector: " << endl;
            // for(int i=0; i<3; i++){
            //     for(int j=0; j<3; j++){
            //         cout << cp.eig_vec[i][j] << " ";
            //     }
            //     cout << endl;
            // }
            //print eigenvalues
            // cout << "eigenvalues: " << endl;
            // for(int i=0; i<3; i++){
            //     cout << cp.eigvalues[i] << " ";
            // }
            // cout << endl;
            auto eigvec = cp.eig_vec;
            auto eigval = cp.eigvalues;
            auto pt = cp.x;
            //create 6x4 array of array
            std::array<std::array<double, 4>, 6> directions; //6 directions, first is direction(1 or -1), next 3 are seed point
            // if eigvalue is positive, then direction is 1, otherwise -1
            for (int i = 0; i < 3; i++){
                if (eigval[i] > 0){
                    directions[i][0] = 1;
                    directions[i][1] = eps * eigvec[i][0] + pt[0];
                    directions[i][2] = eps * eigvec[i][1] + pt[1];
                    directions[i][3] = eps * eigvec[i][2] + pt[2];
                    directions[i+3][0] = -1;
                    directions[i+3][1] = -1 * eps * eigvec[i][0] + pt[0];
                    directions[i+3][2] = -1 * eps * eigvec[i][1] + pt[1];
                    directions[i+3][3] = -1 * eps* eigvec[i][2] + pt[2];
                }
                else{
                    directions[i][0] = -1;
                    directions[i][1] = eps * eigvec[i][0] + pt[0];
                    directions[i][2] = eps * eigvec[i][1] + pt[1];
                    directions[i][3] = eps * eigvec[i][2] + pt[2];
                    directions[i+3][0] = 1;
                    directions[i+3][1] = -1 * eps * eigvec[i][0] + pt[0];
                    directions[i+3][2] = -1 * eps * eigvec[i][1] + pt[1];
                    directions[i+3][3] = -1 * eps * eigvec[i][2] + pt[2];
                }
            }

            //print directions
            // for (int i = 0; i < 6; i++){
            //     cout << "direction " << i << ": ";
            //     for (int j = 0; j < 4; j++){
            //         cout << directions[i][j] << " ";
            //     }
            //     cout << endl;
            // }
            
            for (int i = 0; i < 6; i++){
                //printf("direction %d: \n",i);
                auto direction = directions[i];  
                size_t trajID = trajID_counter++;
                //printf("current trajID: %d\n",trajID);
                std::vector<std::array<double, 3>> traj = trajectory_3d_parallel(pt, {direction[1], direction[2], direction[3]}, h * direction[0], max_length, r3,r2,r1, critical_points_0, grad_ori,index_ori,vertex_ori,cellID_trajIDs_map_ori,trajID);
                printf("threadID: %d, trajID: %d, seed pt: %f %f %f, end pt: %f %f %f\n",omp_get_thread_num(),trajID,direction[1], direction[2], direction[3],traj.back()[0],traj.back()[1],traj.back()[2]);
                #pragma omp critical
                {
                    trajs_ori.push_back(traj);
                }
                //print traj
                // printf("traj %d: \n",trajID);
                // for (auto p:traj){
                //     cout << p[0] << " " << p[1] << " " << p[2] << endl;
                // }
                //auto last = traj.back();
                //printf("trajs_ori_size: %d\n",trajs_ori.size());
            }
            //auto end_six_traj = std::chrono::high_resolution_clock::now();
            //std::chrono::duration<double, std::milli> duration_six_traj = end_six_traj - start_six_traj;
            //std::cout << "Elapsed time for 6 trajs in ms: " << duration_six_traj.count() << " ms" << std::endl;          
  
        }

    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end1 - start1;
    cout << "Elapsed time for calculate all traj once: " << elapsed.count() << "s" << endl;
    printf("total traj: %zu\n",trajs_ori.size());

    start1 = std::chrono::high_resolution_clock::now();
    for (auto traj:trajs_ori){
        auto last = traj.back();
        auto last_offset = get_cell_offset_3d(last.data(), r3, r2, r1);
        auto first = traj[0];
        // printf("first: %f %f %f, last: %f %f %f\n",first[0],first[1],first[2],last[0],last[1],last[2]);
        // printf("first: %f %f %f\n",first[0],first[1],first[2]);
        // printf("last: %f %f %f\n",last[0],last[1],last[2]);
        if (traj.size() == max_length){
            max_iter ++;
        }
        else if (last[0] == -1 && last[1] == -1 && last[2] == -1){
            outside ++;
        }
        else{
            //check if last point in critical_points_0

            for (auto cp:critical_points_0){
                auto cp_pt = cp.second.x;
                if (last[0] == cp_pt[0] && last[1] == cp_pt[1] && last[2] == cp_pt[2]){
                  found ++;
                  break;  
                }
            }

        }
        
    }
    end1 = std::chrono::high_resolution_clock::now();
    elapsed = end1 - start1;
    cout << "Elapsed time for check all traj once: " << elapsed.count() << "s" << endl;
    printf("found: %zu, outside: %zu, max_iter: %zu\n",found,outside,max_iter);
    printf("total traj: %zu\n",trajs_ori.size());
    printf("total critical points: %zu\n",critical_points_0.size());
    printf("total saddle points: %zu\n",total_saddle_count);
    printf("vertex_ori size: %zu\n",vertex_ori.size());

    //*************计算解压缩数据的traj_dec*************
    size_t found_dec = 0;
    size_t outside_dec = 0;
    size_t max_iter_dec = 0;
    size_t total_saddle_count_dec = 0;
    std::vector<std::vector<std::array<double, 3>>> trajs_dec;
    std::vector<int> index_dec;
    std::set<size_t> vertex_dec;
    std::unordered_map<size_t, std::set<int>> cellID_trajIDs_map_dec;
    for (const auto&p:critical_points_out){
        auto cp = p.second;
        if (cp.type >=3 && cp.type <= 6){
            total_saddle_count_dec ++;
            //cout << "critical point saddle at " << cp.x[0] << " " << cp.x[1] << " " << cp.x[2] << " type: " << cp.type << endl;
            //print eigenvector
            // cout << "eigenvector: " << endl;
            // for(int i=0; i<3; i++){
            //     for(int j=0; j<3; j++){
            //         cout << cp.eig_vec[i][j] << " ";
            //     }
            //     cout << endl;
            // }
            //print eigenvalues
            // cout << "eigenvalues: " << endl;
            // for(int i=0; i<3; i++){
            //     cout << cp.eigvalues[i] << " ";
            // }
            // cout << endl;
            auto eigvec = cp.eig_vec;
            auto eigval = cp.eigvalues;
            auto pt = cp.x;
            //create 6x4 array of array
            std::array<std::array<double, 4>, 6> directions; //6 directions, first is direction(1 or -1), next 3 are seed point
            // if eigvalue is positive, then direction is 1, otherwise -1
            for (int i = 0; i < 3; i++){
                if (eigval[i] > 0){
                    directions[i][0] = 1;
                    directions[i][1] = eps * eigvec[i][0] + pt[0];
                    directions[i][2] = eps * eigvec[i][1] + pt[1];
                    directions[i][3] = eps * eigvec[i][2] + pt[2];
                    directions[i+3][0] = -1;
                    directions[i+3][1] = -1 * eps * eigvec[i][0] + pt[0];
                    directions[i+3][2] = -1 * eps * eigvec[i][1] + pt[1];
                    directions[i+3][3] = -1 * eps* eigvec[i][2] + pt[2];
                }
                else{
                    directions[i][0] = -1;
                    directions[i][1] = eps * eigvec[i][0] + pt[0];
                    directions[i][2] = eps * eigvec[i][1] + pt[1];
                    directions[i][3] = eps * eigvec[i][2] + pt[2];
                    directions[i+3][0] = 1;
                    directions[i+3][1] = -1 * eps * eigvec[i][0] + pt[0];
                    directions[i+3][2] = -1 * eps * eigvec[i][1] + pt[1];
                    directions[i+3][3] = -1 * eps * eigvec[i][2] + pt[2];
                }
            }
            for (int i = 0; i < 6; i++){
                auto direction = directions[i];  
                size_t trajID = trajs_dec.size();
                std::vector<std::array<double, 3>> traj = trajectory_3d(pt, {direction[1], direction[2], direction[3]}, h * direction[0], max_length, r3,r2,r1, critical_points_out, grad_dec,index_dec,vertex_dec,cellID_trajIDs_map_dec,trajID);
                trajs_dec.push_back(traj);
            }
        }
    }


    for (auto traj:trajs_dec){
        auto last = traj.back();
        auto last_offset = get_cell_offset_3d(last.data(), r3, r2, r1);
        auto first = traj[0];
        for (auto cp:critical_points_out){
            auto cp_pt = cp.second.x;
            auto cell_offset = get_cell_offset_3d(cp_pt, r3, r2, r1);
            if (last_offset == cell_offset){
                found_dec ++;
                break;
            }
            else if (last[0] == -1 && last[1] == -1 && last[2] == -1){
                outside_dec ++;
                break;
            }
            else{
                max_iter_dec ++;
                break;
            }
        }
        
    }
    printf("found_dec: %zu, outside_dec: %zu, max_iter_dec: %zu\n",found_dec,outside_dec,max_iter_dec);
    printf("total traj_dec: %zu\n",trajs_dec.size());
    printf("total critical points_dec: %zu\n",critical_points_out.size());

    //check traj_ori and traj_dec, the first and second point should be the same
    for (int i = 0; i < trajs_ori.size(); i++){
        auto traj_ori = trajs_ori[i];
        auto traj_dec = trajs_dec[i];
        auto first_ori = traj_ori[0];
        auto first_dec = traj_dec[0];
        auto second_ori = traj_ori[1];
        auto second_dec = traj_dec[1];
        if (first_ori[0] != first_dec[0] || first_ori[1] != first_dec[1] || first_ori[2] != first_dec[2]){
            printf("first point not the same\n");
        }
        if (second_ori[0] != second_dec[0] || second_ori[1] != second_dec[1] || second_ori[2] != second_dec[2]){
            printf("second point not the same\n");
        }
    }
}