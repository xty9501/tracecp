#include "advect.hpp"
#include "cp.hpp"
#include "interp.h"
#include "utils.hpp"
#include "ftk/ndarray.hh"
#include <cmath>
#include <set>
#include <array>
#include<omp.h>
#include <mutex>

std::mutex mtx;

template<typename Type>
void updateOffsetsAndMap(const Type* p, const int DW, const int DH, std::set<size_t>& lossless_index, std::unordered_map<size_t, std::set<int>>& cellID_trajIDs_map, size_t trajID) {
  auto coords = get_three_offsets(p,DW,DH);
  #pragma omp critical
  {
    for (auto offset : coords){
      lossless_index.insert(offset);
      cellID_trajIDs_map[offset].insert(trajID);
    }
  }
}

// template<typename Type>
// void updateOffsets(const Type* p, const int DW, const int DH, std::set<size_t>& lossless_index) {
//   auto coords = get_three_offsets(p,DW,DH);
//   #pragma omp critical
//   {
//     for (auto offset : coords){
//       lossless_index.insert(offset);
//     }
//   }
// }

template<typename Type>
void updateOffsets(const Type* p, const int DW, const int DH, std::vector<std::set<size_t>>& thread_lossless_index,int thread_id) {
    auto coords = get_three_offsets(p, DW, DH);
    thread_lossless_index[thread_id].insert(coords.begin(), coords.end());
}

template<typename Type>
void updateOffsets_unorderset(const Type* p, const int DW, const int DH, std::vector<std::unordered_set<size_t>>& thread_unorderset, std::vector<std::vector<size_t>>& thread_lossless_index, int thread_id) {
    auto coords = get_three_offsets(p, DW, DH);
    // thread_lossless_index[thread_id].insert(coords.begin(), coords.end());
    for (auto offset : coords) {
        if (thread_unorderset[thread_id].find(offset) == thread_unorderset[thread_id].end()) {
            thread_unorderset[thread_id].insert(offset);
            thread_lossless_index[thread_id].push_back(offset);
        }
    }
}

template<typename Type>
std::array<Type, 2> newRK4(const Type * x, const Type * v, ftk::ndarray<float> &data,  Type h, const int DH, const int DW,std::set<size_t>& lossless_index) {
  // x and y are positions, and h is the step size
  double rk1[2] = {0};
  const double p1[2] = {x[0], x[1]};


  if(!inside(p1, DH, DW)){
    //return std::array<Type, 2>{x[0], x[1]};
    return std::array<Type, 2>{x[0], x[1]};
  }
  interp2d(p1, rk1,data);
  auto coords_p1 = get_three_offsets(p1, DW, DH);
  // for (auto offset:coords){
  //   lossless_index.insert(offset);
  // }
  
  double rk2[2] = {0};
  const double p2[2] = {x[0] + 0.5 * h * rk1[0], x[1] + 0.5 * h * rk1[1]};
  if (!inside(p2, DH, DW)){
    //return std::array<Type, 2>{p1[0], p1[1]};
    return std::array<Type, 2>{x[0], x[1]};
  }
  interp2d(p2, rk2,data);
  auto coords_p2 = get_three_offsets(p2, DW, DH);
  // for (auto offset:coords){
  //   lossless_index.insert(offset);
  // }
  
  double rk3[2] = {0};
  const double p3[2] = {x[0] + 0.5 * h * rk2[0], x[1] + 0.5 * h * rk2[1]};
  if (!inside(p3, DH, DW)){
    //return std::array<Type, 2>{p2[0], p2[1]};
    return std::array<Type, 2>{x[0], x[1]};
  }
  interp2d(p3, rk3,data);
  auto coords_p3 = get_three_offsets(p3, DW, DH);
  // for (auto offset:coords){
  //   lossless_index.insert(offset);
  // }
  
  double rk4[2] = {0};
  const double p4[2] = {x[0] + h * rk3[0], x[1] + h * rk3[1]};
  if (!inside(p4, DH, DW)){
    //return std::array<Type, 2>{p3[0], p3[1]};
    return std::array<Type, 2>{x[0], x[1]};
  }
  interp2d(p4, rk4,data);
  auto coords_p4 = get_three_offsets(p4, DW, DH);
  // for (auto offset:coords){
  //   lossless_index.insert(offset);
  // }
  
  Type next_x = x[0] + h * (rk1[0] + 2 * rk2[0] + 2 * rk3[0] + rk4[0]) / 6.0;
  Type next_y = x[1] + h * (rk1[1] + 2 * rk2[1] + 2 * rk3[1] + rk4[1]) / 6.0;
  // printf("shift: (%f, %f)\n", next_x - x[0], next_y - x[1]);
  // printf("coefficients: (%f,%f)\n",(rk1[0] + 2 * rk2[0] + 2 * rk3[0] + rk4[0]) / 6, (rk1[1] + 2 * rk2[1] + 2 * rk3[1] + rk4[1]) / 6);
  // printf("current h sign: %d\n", printsign(h));
  // printf("sign of coefficients x (%d,%d,%d,%d)\n", printsign(rk1[0]), printsign(rk2[0]), printsign(rk3[0]), printsign(rk4[0]));
  // printf("sign of coefficients y (%d,%d,%d,%d)\n", printsign(rk1[1]), printsign(rk2[1]), printsign(rk3[1]), printsign(rk4[1]));
  if (!inside(std::array<Type, 2>{next_x, next_y}, DH, DW)){
    //return std::array<Type, 2>{p4[0], p4[1]};
    return std::array<Type, 2>{x[0], x[1]};
  }
  std::array<Type, 2> result = {next_x, next_y};
  auto coords_final = get_three_offsets(result, DW, DH);
  lossless_index.insert(coords_p1.begin(), coords_p1.end());
  lossless_index.insert(coords_p2.begin(), coords_p2.end());
  lossless_index.insert(coords_p3.begin(), coords_p3.end());
  lossless_index.insert(coords_p4.begin(), coords_p4.end());
  lossless_index.insert(coords_final.begin(), coords_final.end());
  return result;
}


template<typename Type>
std::array<Type, 2> newRK4_parallel(const Type* x, const Type* v, const ftk::ndarray<float>& data, Type h, const int DH, const int DW, std::vector<std::set<size_t>>& thread_lossless_index,int thread_id) {
    // x and y are positions, and h is the step size
    double rk1[2] = {0};
    const double p1[2] = {x[0], x[1]};

    if (!inside(p1, DH, DW)) {
        return std::array<Type, 2>{x[0], x[1]};
    }
    //updateOffsets(p1, DW, DH, thread_lossless_index, thread_id);
    interp2d(p1, rk1, data);

    double rk2[2] = {0};
    const double p2[2] = {x[0] + 0.5 * h * rk1[0], x[1] + 0.5 * h * rk1[1]};
    if (!inside(p2, DH, DW)) {
        return std::array<Type, 2>{x[0], x[1]};
    }
    //updateOffsets(p2, DW, DH, thread_lossless_index, thread_id);
    interp2d(p2, rk2, data);

    double rk3[2] = {0};
    const double p3[2] = {x[0] + 0.5 * h * rk2[0], x[1] + 0.5 * h * rk2[1]};
    if (!inside(p3, DH, DW)) {
        return std::array<Type, 2>{x[0], x[1]};
    }
    //updateOffsets(p3, DW, DH,thread_lossless_index, thread_id);
    interp2d(p3, rk3, data);

    double rk4[2] = {0};
    const double p4[2] = {x[0] + h * rk3[0], x[1] + h * rk3[1]};
    if (!inside(p4, DH, DW)) {
        return std::array<Type, 2>{x[0], x[1]};
    }
    //updateOffsets(p4, DW, DH,thread_lossless_index, thread_id);
    interp2d(p4, rk4, data);

    Type next_x = x[0] + h * (rk1[0] + 2 * rk2[0] + 2 * rk3[0] + rk4[0]) / 6.0;
    Type next_y = x[1] + h * (rk1[1] + 2 * rk2[1] + 2 * rk3[1] + rk4[1]) / 6.0;
    if (!inside(std::array<Type, 2>{next_x, next_y}, DH, DW)) {
        return std::array<Type, 2>{x[0],x[1]};
    }
    std::array<Type, 2> result = {next_x, next_y};
    updateOffsets(p1, DW, DH, thread_lossless_index, thread_id);
    updateOffsets(p2, DW, DH, thread_lossless_index, thread_id);
    updateOffsets(p3, DW, DH, thread_lossless_index, thread_id);
    updateOffsets(p4, DW, DH, thread_lossless_index, thread_id);
    updateOffsets(result.data(), DW, DH,thread_lossless_index, thread_id);
    return result;
}


template<typename Type>
std::array<Type, 2> rkf45(const Type * x, const Type * v, const ftk::ndarray<float> &data,  Type h, const int DH, const int DW,std::set<size_t>& lossless_index,double tolerance) {
  //The Runge-Kutta-Fehlberg method

  double error = 0.0;
  bool acceptableError = false;

  // RKF45 系数
  const double c20 = 0.25, c21 = 0.25;
  const double c30 = 3.0/8.0, c31 = 3.0/32.0, c32 = 9.0/32.0;
  const double c40 = 12.0/13.0, c41 = 1932.0/2197.0, c42 = -7200.0/2197.0, c43 = 7296.0/2197.0;
  const double c50 = 1, c51 = 439.0/216.0, c52 = -8, c53 = 3680.0/513.0, c54 = -845.0/4104.0;
  const double c60 = 0.5, c61 = -8.0/27.0, c62 = 2, c63 = -3544.0/2565.0, c64 = 1859.0/4104.0, c65 = -11.0/40.0;

  // RK4 和 RK5 的权重不同
  const double a1 = 25.0/216.0, a3 = 1408.0/2565.0, a4 = 2197.0/4104.0, a5 = -1.0/5.0;
  const double b1 = 16.0/135.0, b3 = 6656.0/12825.0, b4 = 28561.0/56430.0, b5 = -9.0/50.0, b6 = 2.0/55.0;

  double k1[2] = {0};
  double k2[2] = {0};
  double k3[2] = {0};
  double k4[2] = {0};
  double k5[2] = {0};
  double k6[2] = {0};
  double RK4[2] = {0};
  double RK5[2] = {0};
  std::array<double, 2> p1 = {0, 0};
  std::array<double, 2> p2 = {0, 0};
  std::array<double, 2> p3 = {0, 0};
  std::array<double, 2> p4 = {0, 0};
  std::array<double, 2> p5 = {0, 0};
  std::array<double, 2> p6 = {0, 0};
  do{
    p1 = {x[0], x[1]};
    interp2d(p1.data(), k1,data);
    double xn = x[0] + h*c21*k1[0];
    double yn = x[1] + h*c21*k1[1];
    p2= {xn, yn};
    if (!inside(std::array<Type, 2>{xn, yn}, DH, DW)){
      return std::array<Type, 2>{-1, -1};
    }
    
    interp2d(p2.data(), k2,data);
    xn = x[0] + h*c31*k1[0] + h*c32*k2[0];
    yn = x[1] + h*c31*k1[1] + h*c32*k2[1];
    p3 = {xn, yn};

    if (!inside(std::array<Type, 2>{xn, yn}, DH, DW)){
      return std::array<Type, 2>{-1, -1};
    }
    
    interp2d(p3.data(), k3,data);
    xn = x[0] + h*c41*k1[0] + h*c42*k2[0] + h*c43*k3[0];
    yn = x[1] + h*c41*k1[1] + h*c42*k2[1] + h*c43*k3[1];
    p4 = {xn, yn};
    if (!inside(std::array<Type, 2>{xn, yn}, DH, DW)){
      return std::array<Type, 2>{-1, -1};
    }
    
    interp2d(p4.data(), k4,data);
    xn = x[0] + h*c51*k1[0] + h*c52*k2[0] + h*c53*k3[0] + h*c54*k4[0];
    yn = x[1] + h*c51*k1[1] + h*c52*k2[1] + h*c53*k3[1] + h*c54*k4[1];
    p5 = {xn, yn};
    if (!inside(std::array<Type, 2>{xn, yn}, DH, DW)){
      return std::array<Type, 2>{-1, -1};
    }
    
    interp2d(p5.data(), k5,data);
    xn = x[0] + h*c61*k1[0] + h*c62*k2[0] + h*c63*k3[0] + h*c64*k4[0] + h*c65*k5[0];
    yn = x[1] + h*c61*k1[1] + h*c62*k2[1] + h*c63*k3[1] + h*c64*k4[1] + h*c65*k5[1];
    p6 = {xn, yn};
    if (!inside(std::array<Type, 2>{xn, yn}, DH, DW)){
      return std::array<Type, 2>{-1, -1};
    }
    
    interp2d(p6.data(), k6,data);

    for (int i = 0; i < 2; i++){
      RK4[i] = x[i] + h*(a1*k1[i] + a3*k3[i] + a4*k4[i] + a5*k5[i]);
      RK5[i] = x[i] + h*(b1*k1[i] + b3*k3[i] + b4*k4[i] + b5*k5[i] + b6*k6[i]);
    }
    double R[2] = {std::abs(RK4[0] - RK5[0]), std::abs(RK4[1] - RK5[1])};
    error = std::max(R[0], R[1]);
    acceptableError = error <= tolerance;
    if (!acceptableError){
      h /= 2;
    }
    
  } while(!acceptableError);

  


  std::array<Type, 2> result = {x[0] + h*(b1*k1[0] + b3*k3[0] + b4*k4[0] + b5*k5[0] + b6*k6[0]), 
                               x[1] + h*(b1*k1[1] + b3*k3[1] + b4*k4[1] + b5*k5[1] + b6*k6[1])};
  if (!inside(result, DH, DW)){
    return std::array<Type, 2>{-1, -1};
  }
  //add the lossless index
  std::array<decltype(p1), 6> points = {p1, p2, p3, p4, p5, p6};
  for (auto point:points){
    auto coords = get_three_offsets(point, DW, DH);
    for (auto offset:coords){
      lossless_index.insert(offset);
    }
  }

  return result;

}


std::vector<std::array<double, 2>> trajectory_parallel(double *X_original,const std::array<double, 2>& initial_x, const double time_step, const int max_length, const int DH,const int DW, const std::unordered_map<size_t, critical_point_t>& critical_points, ftk::ndarray<float>& data ,std::vector<std::set<size_t>>& thread_lossless_index,int thread_id){
  std::vector<std::array<double, 2>> result;
  result.reserve(max_length);
  int flag = 0; // 1 means found, -1 means out of bound， 0 means reach max length
  int length = 0;
  result.push_back({X_original[0], X_original[1]}); //add original true position
  length ++;
  int orginal_offset = get_cell_offset(X_original, DW, DH);

  std::array<double, 2> current_x = initial_x;

  //add original and initial_x position's offset
  auto ori_offset = get_three_offsets(X_original, DW, DH);

  // for (auto offset:ori_offset){
  //   lossless_index.insert(offset);
  // }  
  if(!inside(current_x, DH, DW)){ //seed outside 
    return result;
  }

  result.push_back(current_x); //add initial position(seed)
  length ++;
  auto ini_offset = get_three_offsets(current_x, DW, DH);
  thread_lossless_index[thread_id].insert(ini_offset.begin(), ini_offset.end());
  auto original_offset = get_cell_offset(current_x.data(), DW, DH);


  while (result.size() < max_length) {
    double current_v[2] = {0};  
    std::array<double, 2> RK4result = newRK4_parallel(current_x.data(), current_v, data, time_step, DH, DW,thread_lossless_index,thread_id);
    size_t current_offset = get_cell_offset(RK4result.data(), DW, DH);

    if (!inside(RK4result, DH, DW)) { // Out of bound
        result.push_back(current_x);
        return result;
    }

    if (current_offset != original_offset) { // Moved to another cell
        auto it = critical_points.find(current_offset);
        if (it != critical_points.end()) {
            auto cp = it->second;
            double error = 1e-3;
            if (cp.type != SADDLE && fabs(RK4result[0] - cp.x[0]) < error && fabs(RK4result[1] - cp.x[1]) < error) {
                result.push_back({RK4result[0], RK4result[1]});
                result.push_back({cp.x[0], cp.x[1]}); // Add true critical point
                auto final_offset_rk = get_three_offsets(RK4result, DW, DH);
                thread_lossless_index[thread_id].insert(final_offset_rk.begin(), final_offset_rk.end());
                return result;
            }
        }
    }
    current_x = RK4result;
    result.push_back(current_x);
  }

  return result;
}


std::vector<std::array<double, 2>> trajectory(double *X_original,const std::array<double, 2>& initial_x, const double time_step, const int max_length, const int DH,const int DW, const std::unordered_map<size_t, critical_point_t>& critical_points, ftk::ndarray<float>& data,std::set<size_t>& lossless_index){
  std::vector<std::array<double, 2>> result;
  result.reserve(max_length);
  int flag = 0; // 1 means found, -1 means out of bound， 0 means reach max length
  int length = 0;
  result.push_back({X_original[0], X_original[1]}); //add original true position
  length ++;
  int orginal_offset = get_cell_offset(X_original, DW, DH);

  std::array<double, 2> current_x = initial_x;

  //add original and initial_x position's offset
  auto ori_offset = get_three_offsets(X_original, DW, DH);

  // for (auto offset:ori_offset){
  //   lossless_index.insert(offset);
  // }  
  if(!inside(current_x, DH, DW)){ //seed outside 
    return result;
  }

  result.push_back(current_x); //add initial position(seed)
  length ++;
  auto ini_offset = get_three_offsets(current_x, DW, DH);
  lossless_index.insert(ini_offset.begin(), ini_offset.end());
  auto original_offset = get_cell_offset(current_x.data(), DW, DH);


  while (result.size() < max_length) {
    double current_v[2] = {0};  
    std::array<double, 2> RK4result = newRK4(current_x.data(), current_v, data, time_step, DH, DW, lossless_index);
    size_t current_offset = get_cell_offset(RK4result.data(), DW, DH);

    if (!inside(RK4result, DH, DW)) { // Out of bound
        result.push_back(current_x);
        return result;
    }

    if (current_offset != original_offset) { // Moved to another cell
        auto it = critical_points.find(current_offset);
        if (it != critical_points.end()) {
            auto cp = it->second;
            double error = 1e-3;
            if (cp.type != SADDLE && fabs(RK4result[0] - cp.x[0]) < error && fabs(RK4result[1] - cp.x[1]) < error) {
                result.push_back({RK4result[0], RK4result[1]});
                result.push_back({cp.x[0], cp.x[1]}); // Add true critical point
                auto final_offset_rk = get_three_offsets(RK4result, DW, DH);
                lossless_index.insert(final_offset_rk.begin(), final_offset_rk.end());
                return result;
            }
        }
    }
    current_x = RK4result;
    result.push_back(current_x);
  }

  return result;
}


// MODIFIED, using unorderset + vector to store lossless index
template<typename Type>
std::array<Type, 2> newRK4(const Type * x, const Type * v, ftk::ndarray<float> &data,  Type h, const int DH, const int DW,std::unordered_set<size_t>& unorderset, std::vector<size_t>& lossless_index) {
  // x and y are positions, and h is the step size
  double rk1[2] = {0};
  const double p1[2] = {x[0], x[1]};


  if(!inside(p1, DH, DW)){
    //return std::array<Type, 2>{x[0], x[1]};
    return std::array<Type, 2>{x[0], x[1]};
  }
  interp2d(p1, rk1,data);
  auto coords_p1 = get_three_offsets(p1, DW, DH);
  // for (auto offset:coords){
  //   lossless_index.insert(offset);
  // }
  
  double rk2[2] = {0};
  const double p2[2] = {x[0] + 0.5 * h * rk1[0], x[1] + 0.5 * h * rk1[1]};
  if (!inside(p2, DH, DW)){
    //return std::array<Type, 2>{p1[0], p1[1]};
    return std::array<Type, 2>{x[0], x[1]};
  }
  interp2d(p2, rk2,data);
  auto coords_p2 = get_three_offsets(p2, DW, DH);
  // for (auto offset:coords){
  //   lossless_index.insert(offset);
  // }
  
  double rk3[2] = {0};
  const double p3[2] = {x[0] + 0.5 * h * rk2[0], x[1] + 0.5 * h * rk2[1]};
  if (!inside(p3, DH, DW)){
    //return std::array<Type, 2>{p2[0], p2[1]};
    return std::array<Type, 2>{x[0], x[1]};
  }
  interp2d(p3, rk3,data);
  auto coords_p3 = get_three_offsets(p3, DW, DH);
  // for (auto offset:coords){
  //   lossless_index.insert(offset);
  // }
  
  double rk4[2] = {0};
  const double p4[2] = {x[0] + h * rk3[0], x[1] + h * rk3[1]};
  if (!inside(p4, DH, DW)){
    //return std::array<Type, 2>{p3[0], p3[1]};
    return std::array<Type, 2>{x[0], x[1]};
  }
  interp2d(p4, rk4,data);
  auto coords_p4 = get_three_offsets(p4, DW, DH);
  // for (auto offset:coords){
  //   lossless_index.insert(offset);
  // }
  
  Type next_x = x[0] + h * (rk1[0] + 2 * rk2[0] + 2 * rk3[0] + rk4[0]) / 6.0;
  Type next_y = x[1] + h * (rk1[1] + 2 * rk2[1] + 2 * rk3[1] + rk4[1]) / 6.0;
  // printf("shift: (%f, %f)\n", next_x - x[0], next_y - x[1]);
  // printf("coefficients: (%f,%f)\n",(rk1[0] + 2 * rk2[0] + 2 * rk3[0] + rk4[0]) / 6, (rk1[1] + 2 * rk2[1] + 2 * rk3[1] + rk4[1]) / 6);
  // printf("current h sign: %d\n", printsign(h));
  // printf("sign of coefficients x (%d,%d,%d,%d)\n", printsign(rk1[0]), printsign(rk2[0]), printsign(rk3[0]), printsign(rk4[0]));
  // printf("sign of coefficients y (%d,%d,%d,%d)\n", printsign(rk1[1]), printsign(rk2[1]), printsign(rk3[1]), printsign(rk4[1]));
  if (!inside(std::array<Type, 2>{next_x, next_y}, DH, DW)){
    //return std::array<Type, 2>{p4[0], p4[1]};
    return std::array<Type, 2>{x[0], x[1]};
  }
  std::array<Type, 2> result = {next_x, next_y};
  auto coords_final = get_three_offsets(result, DW, DH);
  // lossless_index.insert(coords_p1.begin(), coords_p1.end());
  // lossless_index.insert(coords_p2.begin(), coords_p2.end());
  // lossless_index.insert(coords_p3.begin(), coords_p3.end());
  // lossless_index.insert(coords_p4.begin(), coords_p4.end());
  // lossless_index.insert(coords_final.begin(), coords_final.end());

  // check if contains, if not, insert
  for (auto offset:coords_p1){
    if (unorderset.find(offset) == unorderset.end()){
      unorderset.insert(offset);
      lossless_index.push_back(offset);
    }
  }
  for (auto offset:coords_p2){
    if (unorderset.find(offset) == unorderset.end()){
      unorderset.insert(offset);
      lossless_index.push_back(offset);
    }
  }
  for (auto offset:coords_p3){
    if (unorderset.find(offset) == unorderset.end()){
      unorderset.insert(offset);
      lossless_index.push_back(offset);
    }
  }
  for (auto offset:coords_p4){
    if (unorderset.find(offset) == unorderset.end()){
      unorderset.insert(offset);
      lossless_index.push_back(offset);
    }
  }
  for (auto offset:coords_final){
    if (unorderset.find(offset) == unorderset.end()){
      unorderset.insert(offset);
      lossless_index.push_back(offset);
    }
  }
  return result;
}

template<typename Type>
std::array<Type, 2> newRK4_parallel(const Type* x, const Type* v, const ftk::ndarray<float>& data, Type h, const int DH, const int DW, std::vector<std::unordered_set<size_t>>& thread_unorderset, std::vector<std::vector<size_t>>& thread_lossless_index, int thread_id) {
    // x and y are positions, and h is the step size
    double rk1[2] = {0};
    const double p1[2] = {x[0], x[1]};

    if (!inside(p1, DH, DW)) {
        return std::array<Type, 2>{x[0], x[1]};
    }
    //updateOffsets(p1, DW, DH, thread_lossless_index, thread_id);
    interp2d(p1, rk1, data);

    double rk2[2] = {0};
    const double p2[2] = {x[0] + 0.5 * h * rk1[0], x[1] + 0.5 * h * rk1[1]};
    if (!inside(p2, DH, DW)) {
        return std::array<Type, 2>{x[0], x[1]};
    }
    //updateOffsets(p2, DW, DH, thread_lossless_index, thread_id);
    interp2d(p2, rk2, data);

    double rk3[2] = {0};
    const double p3[2] = {x[0] + 0.5 * h * rk2[0], x[1] + 0.5 * h * rk2[1]};
    if (!inside(p3, DH, DW)) {
        return std::array<Type, 2>{x[0], x[1]};
    }
    //updateOffsets(p3, DW, DH,thread_lossless_index, thread_id);
    interp2d(p3, rk3, data);

    double rk4[2] = {0};
    const double p4[2] = {x[0] + h * rk3[0], x[1] + h * rk3[1]};
    if (!inside(p4, DH, DW)) {
        return std::array<Type, 2>{x[0], x[1]};
    }
    //updateOffsets(p4, DW, DH,thread_lossless_index, thread_id);
    interp2d(p4, rk4, data);

    Type next_x = x[0] + h * (rk1[0] + 2 * rk2[0] + 2 * rk3[0] + rk4[0]) / 6.0;
    Type next_y = x[1] + h * (rk1[1] + 2 * rk2[1] + 2 * rk3[1] + rk4[1]) / 6.0;
    if (!inside(std::array<Type, 2>{next_x, next_y}, DH, DW)) {
        return std::array<Type, 2>{x[0],x[1]};
    }
    std::array<Type, 2> result = {next_x, next_y};
    updateOffsets_unorderset(p1, DW, DH, thread_unorderset, thread_lossless_index, thread_id);
    updateOffsets_unorderset(p2, DW, DH, thread_unorderset, thread_lossless_index, thread_id);
    updateOffsets_unorderset(p3, DW, DH, thread_unorderset, thread_lossless_index, thread_id);
    updateOffsets_unorderset(p4, DW, DH, thread_unorderset, thread_lossless_index, thread_id);
    updateOffsets_unorderset(result.data(), DW, DH, thread_unorderset, thread_lossless_index, thread_id);
    return result;
}

std::vector<std::array<double, 2>> trajectory(double *X_original,const std::array<double, 2>& initial_x, const double time_step, const int max_length, const int DH,const int DW, const std::unordered_map<size_t, critical_point_t>& critical_points, ftk::ndarray<float>& data,std::unordered_set<size_t>& unorderset,std::vector<size_t>& lossless_index){
  std::vector<std::array<double, 2>> result;
  result.reserve(max_length);
  int flag = 0; // 1 means found, -1 means out of bound， 0 means reach max length
  int length = 0;
  result.push_back({X_original[0], X_original[1]}); //add original true position
  length ++;
  int orginal_offset = get_cell_offset(X_original, DW, DH);

  std::array<double, 2> current_x = initial_x;

  //add original and initial_x position's offset
  auto ori_offset = get_three_offsets(X_original, DW, DH);

  // for (auto offset:ori_offset){
  //   lossless_index.insert(offset);
  // }  
  if(!inside(current_x, DH, DW)){ //seed outside 
    return result;
  }

  result.push_back(current_x); //add initial position(seed)
  length ++;
  auto ini_offset = get_three_offsets(current_x, DW, DH);
  // lossless_index.insert(ini_offset.begin(), ini_offset.end());
  for (auto offset:ini_offset){
    if (unorderset.find(offset) == unorderset.end()){
      unorderset.insert(offset);
      lossless_index.push_back(offset);
    }
  }
  auto original_offset = get_cell_offset(current_x.data(), DW, DH);


  while (result.size() < max_length) {
    double current_v[2] = {0};  
    std::array<double, 2> RK4result = newRK4(current_x.data(), current_v, data, time_step, DH, DW, unorderset, lossless_index);
    size_t current_offset = get_cell_offset(RK4result.data(), DW, DH);

    if (!inside(RK4result, DH, DW)) { // Out of bound
        result.push_back(current_x);
        return result;
    }

    if (current_offset != original_offset) { // Moved to another cell
        auto it = critical_points.find(current_offset);
        if (it != critical_points.end()) {
            auto cp = it->second;
            double error = 1e-3;
            if (cp.type != SADDLE && fabs(RK4result[0] - cp.x[0]) < error && fabs(RK4result[1] - cp.x[1]) < error) {
                result.push_back({RK4result[0], RK4result[1]});
                result.push_back({cp.x[0], cp.x[1]}); // Add true critical point
                auto final_offset_rk = get_three_offsets(RK4result, DW, DH);
                // lossless_index.insert(final_offset_rk.begin(), final_offset_rk.end());
                for (auto offset:final_offset_rk){
                  if (unorderset.find(offset) == unorderset.end()){
                    unorderset.insert(offset);
                    lossless_index.push_back(offset);
                  }
                }
                return result;
            }
        }
    }
    current_x = RK4result;
    result.push_back(current_x);
  }

  return result;
}

std::vector<std::array<double, 2>> trajectory_parallel(double *X_original,const std::array<double, 2>& initial_x, const double time_step, const int max_length, const int DH,const int DW, const std::unordered_map<size_t, critical_point_t>& critical_points, ftk::ndarray<float>& data ,std::vector<std::unordered_set<size_t>>& thread_unorderset,std::vector<std::vector<size_t>> &thread_lossless_index,int thread_id){
  std::vector<std::array<double, 2>> result;
  result.reserve(max_length);
  int flag = 0; // 1 means found, -1 means out of bound， 0 means reach max length
  int length = 0;
  result.push_back({X_original[0], X_original[1]}); //add original true position
  length ++;
  int orginal_offset = get_cell_offset(X_original, DW, DH);

  std::array<double, 2> current_x = initial_x;

  //add original and initial_x position's offset
  auto ori_offset = get_three_offsets(X_original, DW, DH);

  // for (auto offset:ori_offset){
  //   lossless_index.insert(offset);
  // }  
  if(!inside(current_x, DH, DW)){ //seed outside 
    return result;
  }

  result.push_back(current_x); //add initial position(seed)
  length ++;
  auto ini_offset = get_three_offsets(current_x, DW, DH);
  // thread_lossless_index[thread_id].insert(ini_offset.begin(), ini_offset.end());
  for (auto offset:ini_offset){
    if (thread_unorderset[thread_id].find(offset) == thread_unorderset[thread_id].end()){
      thread_unorderset[thread_id].insert(offset);
      thread_lossless_index[thread_id].push_back(offset);
    }
  }
  auto original_offset = get_cell_offset(current_x.data(), DW, DH);


  while (result.size() < max_length) {
    double current_v[2] = {0};  
    std::array<double, 2> RK4result = newRK4_parallel(current_x.data(), current_v, data, time_step, DH, DW,thread_unorderset,thread_lossless_index,thread_id);
    size_t current_offset = get_cell_offset(RK4result.data(), DW, DH);

    if (!inside(RK4result, DH, DW)) { // Out of bound
        result.push_back(current_x);
        return result;
    }

    if (current_offset != original_offset) { // Moved to another cell
        auto it = critical_points.find(current_offset);
        if (it != critical_points.end()) {
            auto cp = it->second;
            double error = 1e-3;
            if (cp.type != SADDLE && fabs(RK4result[0] - cp.x[0]) < error && fabs(RK4result[1] - cp.x[1]) < error) {
                result.push_back({RK4result[0], RK4result[1]});
                result.push_back({cp.x[0], cp.x[1]}); // Add true critical point
                auto final_offset_rk = get_three_offsets(RK4result, DW, DH);
                // thread_lossless_index[thread_id].insert(final_offset_rk.begin(), final_offset_rk.end());
                for (auto offset:final_offset_rk){
                  if (thread_unorderset[thread_id].find(offset) == thread_unorderset[thread_id].end()){
                    thread_unorderset[thread_id].insert(offset);
                    thread_lossless_index[thread_id].push_back(offset);
                  }
                }
                return result;
            }
        }
    }
    current_x = RK4result;
    result.push_back(current_x);
  }

  return result;
}
