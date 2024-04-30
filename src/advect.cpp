#include "advect.hpp"
#include "cp.hpp"
#include "interp.h"
#include "utils.hpp"
#include "ftk/ndarray.hh"
#include <cmath>
#include <set>
#include <array>


template<typename Type>
std::array<Type, 2> newRK4(const Type * x, const Type * v, const ftk::ndarray<float> &data,  Type h, const int DH, const int DW,std::set<size_t>& lossless_index) {
  // x and y are positions, and h is the step size
  double rk1[2] = {0};
  const double p1[2] = {x[0], x[1]};

  auto coords = get_three_offsets(x, DW, DH);
  for (auto offset:coords){
    lossless_index.insert(offset);
  }

  if(!inside(p1, DH, DW)){
    //return std::array<Type, 2>{x[0], x[1]};
    return std::array<Type, 2>{-1, -1};
  }
  interp2d(p1, rk1,data);
  coords = get_three_offsets(p1, DW, DH);
  for (auto offset:coords){
    lossless_index.insert(offset);
  }
  
  double rk2[2] = {0};
  const double p2[2] = {x[0] + 0.5 * h * rk1[0], x[1] + 0.5 * h * rk1[1]};
  if (!inside(p2, DH, DW)){
    //return std::array<Type, 2>{p1[0], p1[1]};
    return std::array<Type, 2>{-1, -1};
  }
  interp2d(p2, rk2,data);
  coords = get_three_offsets(p2, DW, DH);
  for (auto offset:coords){
    lossless_index.insert(offset);
  }
  
  double rk3[2] = {0};
  const double p3[2] = {x[0] + 0.5 * h * rk2[0], x[1] + 0.5 * h * rk2[1]};
  if (!inside(p3, DH, DW)){
    //return std::array<Type, 2>{p2[0], p2[1]};
    return std::array<Type, 2>{-1, -1};
  }
  interp2d(p3, rk3,data);
  coords = get_three_offsets(p3, DW, DH);
  for (auto offset:coords){
    lossless_index.insert(offset);
  }
  
  double rk4[2] = {0};
  const double p4[2] = {x[0] + h * rk3[0], x[1] + h * rk3[1]};
  if (!inside(p4, DH, DW)){
    //return std::array<Type, 2>{p3[0], p3[1]};
    return std::array<Type, 2>{-1, -1};
  }
  interp2d(p4, rk4,data);
  coords = get_three_offsets(p4, DW, DH);
  for (auto offset:coords){
    lossless_index.insert(offset);
  }
  
  Type next_x = x[0] + h * (rk1[0] + 2 * rk2[0] + 2 * rk3[0] + rk4[0]) / 6.0;
  Type next_y = x[1] + h * (rk1[1] + 2 * rk2[1] + 2 * rk3[1] + rk4[1]) / 6.0;
  // printf("shift: (%f, %f)\n", next_x - x[0], next_y - x[1]);
  // printf("coefficients: (%f,%f)\n",(rk1[0] + 2 * rk2[0] + 2 * rk3[0] + rk4[0]) / 6, (rk1[1] + 2 * rk2[1] + 2 * rk3[1] + rk4[1]) / 6);
  // printf("current h sign: %d\n", printsign(h));
  // printf("sign of coefficients x (%d,%d,%d,%d)\n", printsign(rk1[0]), printsign(rk2[0]), printsign(rk3[0]), printsign(rk4[0]));
  // printf("sign of coefficients y (%d,%d,%d,%d)\n", printsign(rk1[1]), printsign(rk2[1]), printsign(rk3[1]), printsign(rk4[1]));
  if (!inside(std::array<Type, 2>{next_x, next_y}, DH, DW)){
    //return std::array<Type, 2>{p4[0], p4[1]};
    return std::array<Type, 2>{-1, -1};
  }
  std::array<Type, 2> result = {next_x, next_y};
  coords = get_three_offsets(result, DW, DH);
  for (auto offset:coords){
    lossless_index.insert(offset);
  }
  return result;
}

//overload newRK4 function
template<typename Type>
std::array<Type, 2> newRK4(const Type * x, const Type * v, const ftk::ndarray<float> &data,  Type h, const int DH, const int DW) {
  // x and y are positions, and h is the step size
  double rk1[2] = {0};
  const double p1[] = {x[0], x[1]};



  if(!inside(p1, DH, DW)){
    //return std::array<Type, 2>{x[0], x[1]};
    return std::array<Type, 2>{-1, -1};
  }
  interp2d(p1, rk1,data);

  
  double rk2[2] = {0};
  const double p2[] = {x[0] + 0.5 * h * rk1[0], x[1] + 0.5 * h * rk1[1]};
  if (!inside(p2, DH, DW)){
    //return std::array<Type, 2>{p1[0], p1[1]};
    return std::array<Type, 2>{-1, -1};
  }
  interp2d(p2, rk2,data);

  
  double rk3[2] = {0};
  const double p3[] = {x[0] + 0.5 * h * rk2[0], x[1] + 0.5 * h * rk2[1]};
  if (!inside(p3, DH, DW)){
    //return std::array<Type, 2>{p2[0], p2[1]};
    return std::array<Type, 2>{-1, -1};
  }
  interp2d(p3, rk3,data);

  
  double rk4[2] = {0};
  const double p4[] = {x[0] + h * rk3[0], x[1] + h * rk3[1]};
  if (!inside(p4, DH, DW)){
    //return std::array<Type, 2>{p3[0], p3[1]};
    return std::array<Type, 2>{-1, -1};
  }
  interp2d(p4, rk4,data);

  
  Type next_x = x[0] + h * (rk1[0] + 2 * rk2[0] + 2 * rk3[0] + rk4[0]) / 6;
  Type next_y = x[1] + h * (rk1[1] + 2 * rk2[1] + 2 * rk3[1] + rk4[1]) / 6;
  // printf("shift: (%f, %f)\n", next_x - x[0], next_y - x[1]);
  // printf("coefficients: (%f,%f)\n",(rk1[0] + 2 * rk2[0] + 2 * rk3[0] + rk4[0]) / 6, (rk1[1] + 2 * rk2[1] + 2 * rk3[1] + rk4[1]) / 6);
  // printf("current h sign: %d\n", printsign(h));
  // printf("sign of coefficients x (%d,%d,%d,%d)\n", printsign(rk1[0]), printsign(rk2[0]), printsign(rk3[0]), printsign(rk4[0]));
  // printf("sign of coefficients y (%d,%d,%d,%d)\n", printsign(rk1[1]), printsign(rk2[1]), printsign(rk3[1]), printsign(rk4[1]));
  if (!inside(std::array<Type, 2>{next_x, next_y}, DH, DW)){
    //return std::array<Type, 2>{p4[0], p4[1]};
    return std::array<Type, 2>{-1, -1};
  }
  std::array<Type, 2> result = {next_x, next_y};

  return result;
}

template<typename Type>
std::set<size_t> vertex_for_each_RK4(const Type *x, const Type *v, const ftk::ndarray<float> &data, double h, const int DH, const int DW) {
  std::set<size_t> result_set;
  double rk1[2] = {0};
  const double p1[] = {x[0], x[1]};

  auto coords = get_three_offsets(x, DW, DH);
  for (auto offset:coords){
    result_set.insert(offset);
  }

  if(!inside(p1, DH, DW)){
    //return std::array<Type, 2>{x[0], x[1]};
    return result_set;
  }
  interp2d(p1, rk1,data);
  coords = get_three_offsets(p1, DW, DH);
  for (auto offset:coords){
    result_set.insert(offset);
  }
  
  double rk2[2] = {0};
  const double p2[] = {x[0] + 0.5 * h * rk1[0], x[1] + 0.5 * h * rk1[1]};
  if (!inside(p2, DH, DW)){
    //return std::array<Type, 2>{p1[0], p1[1]};
    return result_set;
  }
  interp2d(p2, rk2,data);
  coords = get_three_offsets(p2, DW, DH);
  for (auto offset:coords){
    result_set.insert(offset);
  }
  
  double rk3[2] = {0};
  const double p3[] = {x[0] + 0.5 * h * rk2[0], x[1] + 0.5 * h * rk2[1]};
  if (!inside(p3, DH, DW)){
    //return std::array<Type, 2>{p2[0], p2[1]};
    return result_set;
  }
  interp2d(p3, rk3,data);
  coords = get_three_offsets(p3, DW, DH);
  for (auto offset:coords){
    result_set.insert(offset);
  }
  
  double rk4[2] = {0};
  const double p4[] = {x[0] + h * rk3[0], x[1] + h * rk3[1]};
  if (!inside(p4, DH, DW)){
    //return std::array<Type, 2>{p3[0], p3[1]};
    return result_set;
  }
  interp2d(p4, rk4,data);
  coords = get_three_offsets(p4, DW, DH);
  for (auto offset:coords){
    result_set.insert(offset);
  }
  
  Type next_x = x[0] + h * (rk1[0] + 2 * rk2[0] + 2 * rk3[0] + rk4[0]) / 6;
  Type next_y = x[1] + h * (rk1[1] + 2 * rk2[1] + 2 * rk3[1] + rk4[1]) / 6;
  // printf("shift: (%f, %f)\n", next_x - x[0], next_y - x[1]);
  // printf("coefficients: (%f,%f)\n",(rk1[0] + 2 * rk2[0] + 2 * rk3[0] + rk4[0]) / 6, (rk1[1] + 2 * rk2[1] + 2 * rk3[1] + rk4[1]) / 6);
  // printf("current h sign: %d\n", printsign(h));
  // printf("sign of coefficients x (%d,%d,%d,%d)\n", printsign(rk1[0]), printsign(rk2[0]), printsign(rk3[0]), printsign(rk4[0]));
  // printf("sign of coefficients y (%d,%d,%d,%d)\n", printsign(rk1[1]), printsign(rk2[1]), printsign(rk3[1]), printsign(rk4[1]));
  if (!inside(std::array<Type, 2>{next_x, next_y}, DH, DW)){
    //return std::array<Type, 2>{p4[0], p4[1]};
    return result_set;
  }
  std::array<Type, 2> result = {next_x, next_y};
  coords = get_three_offsets(result, DW, DH);
  for (auto offset:coords){
    result_set.insert(offset);
  }
  return result_set;
}

template std::set<unsigned long> vertex_for_each_RK4<double>(double const*, double const*, ftk::ndarray<float> const&, double, int, int);


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

std::vector<std::array<double, 2>> trajectory(double *X_original,const std::array<double, 2>& initial_x, const double time_step, const int DH,const int DW, const std::unordered_map<int, critical_point_t>& critical_points, ftk::ndarray<float>& data  ,std::vector<int>& index, std::vector<double>& config, std::vector<record_t>& record,std::set<size_t>& lossless_index){
  std::vector<std::array<double, 2>> result;
  int flag = 0; // 1 means found, -1 means out of bound， 0 means reach max length
  int length = 0;
  result.push_back({X_original[0], X_original[1]}); //add original true position
  length ++;
  int orginal_offset = get_cell_offset(X_original, DW, DH);

  std::array<double, 2> current_x;
  //copy initial_x to current_x
  current_x[0] = initial_x[0];
  current_x[1] = initial_x[1];

  //add original and initial_x position's offset
  auto ori_offset = get_three_offsets(X_original, DW, DH); //add cp offset
  for (auto offset:ori_offset){
    lossless_index.insert(offset);
  }
  auto ini_offset = get_three_offsets(initial_x, DW, DH); //add seed offset
  for (auto offset:ini_offset){
    lossless_index.insert(offset);
  }
  

  if(!inside(current_x, DH, DW)){
    //if seed is out of bound, return -1
    //count_out_bound ++;
    flag = -1;
    result.push_back({-1, -1});
    length ++;
    index.push_back(length);
    return result;
  }
  else{
    //seed inside the domain
    result.push_back(current_x); //add initial position(seed)
    length ++;
  }

  while (flag == 0){
    if(!inside(current_x, DH, DW)){
      //count_out_bound ++;
      flag = -1;
      result.push_back({-1, -1});
      length ++;
      break;
    }
    if (length == 2000) {
      //printf("reach max length!\n");
      //count_limit ++;
      // flag = 1;
      break;
    }

    double current_v[2] = {0};

    interp2d(current_x.data(), current_v,data); 

    //int current_offset = get_cell_offset(current_x.data(), DW, DH);    

    std::array<double, 2> RK4result = newRK4(current_x.data(), current_v, data, time_step, DH, DW,lossless_index);
    //std::array<double, 2> RK4result = rkf45(current_x.data(), current_v, data, time_step, DH, DW,lossless_index);
    
    if (RK4result[0] == -1 && RK4result[1] == -1){
      //count_out_bound ++;
      flag = -1;
      result.push_back({-1, -1});
      length ++;
      break;
    }

    int current_offset = get_cell_offset(RK4result.data(), DW, DH);

    if (current_offset != orginal_offset){
      //moved to another cell
        auto surrounding_cell = get_surrounding_cell(current_offset,RK4result, DW, DH);
        for (auto cell_offset:surrounding_cell){
          try{
              auto cp = critical_points.at(cell_offset);
              if (cp.type == SADDLE) break;
              //check if distance between current_x and cp.x is small enough
              double error = 1e-3;
              if (fabs(RK4result[0] - cp.x[0]) < error && fabs(RK4result[1] - cp.x[1]) < error){
                // if interpolated location is close to cp location, then find cp
                flag = 1;
                //count_found ++;
                //printf("found cp after %d iteration, type: %s\n",length, get_critical_point_type_string(cp.type).c_str());
                //printf("distance: %f\n",sqrt((initial_x[0]-current_x[0])*(initial_x[0]-current_x[0]) + (initial_x[1]-current_x[1])*(initial_x[1]-current_x[1])));
                //printf("start_id: %d, current_id: %d\n", orginal_offset,get_cell_offset(current_x.data(), DW, DH));
                //printf("start_values: (%f, %f), current_values: (%f, %f)\n", temp_v[0],temp_v[1],current_v[0],current_v[1]);
                //printf("start_position: (%f, %f), current_position: (%f, %f)\n", initial_x[0],initial_x[1],current_x[0],current_x[1]);

                //add to record
                int cp_offset = get_cell_offset(cp.x, DW, DH);
                record_t r(static_cast<double>(orginal_offset), static_cast<double>(cp_offset), config[2], config[0], config[1]);
                record.push_back(r);

                // first add rk4 position
                result.push_back({RK4result[0], RK4result[1]});
                length++;
                // then add cp position
                std::array<double, 2> true_cp = {cp.x[0], cp.x[1]};
                result.push_back(true_cp);
                length++;
                index.push_back(length);
                return result;
              }
              else{
                //not found cp in this cell

              }
          }
          catch(const std::out_of_range& e){
            // 键不存在，继续查找下一个键
          }
        }
    }
    current_x = RK4result;
    // printf("current_x: (%f, %f)\n", current_x[0], current_x[1]);
    result.push_back(current_x);
    length++;
  }
  // printf("length: %d\n", length);
  // printf("trajectory length: %ld\n", result.size());
  // printf("current_x: (%f, %f)\n", current_x[0], current_x[1]);
  
  // if (flag == 0){
  //   // printf("not found after %d iteration\n",length);
  //   // printf("start_id: %d, current_id: %d\n", orginal_offset,get_cell_offset(current_x.data(), DW, DH));
  //   // double temp_v[2] = {0};
  //   // interp2d(initial_x.data(), temp_v);
  //   // double current_v[2] = {0};
  //   // interp2d(current_x.data(), current_v);
  //   // printf("start_values: (%f, %f), current_values: (%f, %f)\n", temp_v[0],temp_v[1],current_v[0],current_v[1]);
  //   // printf("start_position: (%f, %f), current_position: (%f, %f)\n", initial_x[0],initial_x[1],current_x[0],current_x[1]);
  // }

  index.push_back(length);

  return result;
}


std::vector<std::array<double, 2>> trajectory(double *X_original,const std::array<double, 2>& initial_x, const double time_step, const int max_length, const int DH,const int DW, const std::unordered_map<int, critical_point_t>& critical_points, ftk::ndarray<float>& data  ,std::vector<int>& index){
  std::vector<std::array<double, 2>> result;
  int flag = 0; // 1 means found, -1 means out of bound， 0 means reach max length
  int length = 0;
  result.push_back({X_original[0], X_original[1]}); //add original true position
  length ++;
  int orginal_offset = get_cell_offset(X_original, DW, DH);

  std::array<double, 2> current_x = initial_x;
  

  if(!inside(current_x, DH, DW)){
    //count_out_bound ++;
    flag = -1;
    result.push_back({-1, -1});
    length ++;
    index.push_back(length);
    return result;
  }
  else{
    result.push_back(current_x); //add initial position(seed)
    length ++;
  }

  while (flag == 0){
    if(!inside(current_x, DH, DW)){
      //count_out_bound ++;
      flag = -1;
      result.push_back({-1, -1});
      length ++;
      break;
    }
    if (length == max_length) {
      //printf("reach max length!\n");
      //count_limit ++;
      // flag = 1;
      break;
    }

    double current_v[2] = {0};

    interp2d(current_x.data(), current_v,data); 

    //int current_offset = get_cell_offset(current_x.data(), DW, DH);    

    std::array<double, 2> RK4result = newRK4(current_x.data(), current_v, data, time_step, DH, DW); //用overload的RK4

    //std::array<double, 2> RK4result = rkf45(current_x.data(), current_v, data, time_step, DH, DW,lossless_index);
    
    if (RK4result[0] == -1 && RK4result[1] == -1){
      //count_out_bound ++;
      flag = -1;
      result.push_back({-1, -1});
      length ++;
      break;
    }

    int current_offset = get_cell_offset(RK4result.data(), DW, DH);

    if (current_offset != orginal_offset){
      //moved to another cell
        auto surrounding_cell = get_surrounding_cell(current_offset,RK4result, DW, DH);
        for (auto cell_offset:surrounding_cell){
          try{
              auto cp = critical_points.at(cell_offset);
              if (cp.type == SADDLE) break;
              //check if distance between current_x and cp.x is small enough
              double error = 1e-3;
              if (fabs(RK4result[0] - cp.x[0]) < error && fabs(RK4result[1] - cp.x[1]) < error){
                // if interpolated location is close to cp location, then find cp
                flag = 1;
                // first add rk4 position
                result.push_back({RK4result[0], RK4result[1]});
                length++;
                // then add cp position
                std::array<double, 2> true_cp = {cp.x[0], cp.x[1]};
                result.push_back(true_cp);
                length++;
                index.push_back(length);
                return result;
              }
              else{
                //not found cp in this cell

              }
          }
          catch(const std::out_of_range& e){
            // 键不存在，继续查找下一个键
          }
        }
    }
    current_x = RK4result;
    // printf("current_x: (%f, %f)\n", current_x[0], current_x[1]);
    result.push_back(current_x);
    length++;
  }
  index.push_back(length);

  return result;
}


std::vector<std::array<double, 2>> trajectory(double *X_original,const std::array<double, 2>& initial_x, const double time_step, const int max_length, const int DH,const int DW, const std::unordered_map<int, critical_point_t>& critical_points, ftk::ndarray<float>& data  ,std::vector<int>& index,std::set<size_t>& lossless_index){
  std::vector<std::array<double, 2>> result;
  int flag = 0; // 1 means found, -1 means out of bound， 0 means reach max length
  int length = 0;
  result.push_back({X_original[0], X_original[1]}); //add original true position
  length ++;
  int orginal_offset = get_cell_offset(X_original, DW, DH);

  std::array<double, 2> current_x = initial_x;

  //add original and initial_x position's offset
  auto ori_offset = get_three_offsets(X_original, DW, DH);
  for (auto offset:ori_offset){
    lossless_index.insert(offset);
  }
  auto ini_offset = get_three_offsets(initial_x, DW, DH);
  for (auto offset:ini_offset){
    lossless_index.insert(offset);
  }
  

  if(!inside(current_x, DH, DW)){
    //count_out_bound ++;
    flag = -1;
    result.push_back({-1, -1});
    length ++;
    index.push_back(length);
    return result;
  }
  else{
    result.push_back(current_x); //add initial position(seed)
    length ++;
  }

  while (flag == 0){
    if(!inside(current_x, DH, DW)){
      //count_out_bound ++;
      flag = -1;
      result.push_back({-1, -1});
      length ++;
      break;
    }
    if (length == 2000) {
      //printf("reach max length!\n");
      //count_limit ++;
      // flag = 1;
      break;
    }

    double current_v[2] = {0};

    interp2d(current_x.data(), current_v,data); 

    //int current_offset = get_cell_offset(current_x.data(), DW, DH);    

    std::array<double, 2> RK4result = newRK4(current_x.data(), current_v, data, time_step, DH, DW,lossless_index);
    //std::array<double, 2> RK4result = rkf45(current_x.data(), current_v, data, time_step, DH, DW,lossless_index);
    
    if (RK4result[0] == -1 && RK4result[1] == -1){
      //count_out_bound ++;
      flag = -1;
      result.push_back({-1, -1});
      length ++;
      break;
    }

    int current_offset = get_cell_offset(RK4result.data(), DW, DH);

    if (current_offset != orginal_offset){
      //moved to another cell
        auto surrounding_cell = get_surrounding_cell(current_offset,RK4result, DW, DH);
        for (auto cell_offset:surrounding_cell){
          try{
              auto cp = critical_points.at(cell_offset);
              if (cp.type == SADDLE) break;
              //check if distance between current_x and cp.x is small enough
              double error = 1e-3;
              if (fabs(RK4result[0] - cp.x[0]) < error && fabs(RK4result[1] - cp.x[1]) < error){
                // if interpolated location is close to cp location, then find cp
                flag = 1;

                // first add rk4 position
                result.push_back({RK4result[0], RK4result[1]});
                length++;
                // then add cp position
                std::array<double, 2> true_cp = {cp.x[0], cp.x[1]};
                result.push_back(true_cp);
                length++;
                index.push_back(length);
                return result;
              }
              else{
                //not found cp in this cell

              }
          }
          catch(const std::out_of_range& e){
            // 键不存在，继续查找下一个键
          }
        }
    }
    current_x = RK4result;
    // printf("current_x: (%f, %f)\n", current_x[0], current_x[1]);
    result.push_back(current_x);
    length++;
  }
  index.push_back(length);
  return result;
}
