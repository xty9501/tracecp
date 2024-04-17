#pragma once
#include <iostream>
#include <algorithm>
#include <unordered_map>
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
#include <ftk/numeric/eigen_solver2.hh>
#include <ftk/algorithms/cca.hh>
#include <ftk/geometry/cc2curves.hh>
#include <ftk/geometry/curve2tube.hh>
#include "ftk/ndarray.hh"
#include "ftk/numeric/critical_point_type.hh"
#include "ftk/numeric/critical_point_test.hh"

typedef struct record_t{
  double sid_start;
  double sid_end;
  double dir;
  double eig_vector_x;
  double eig_vector_y;
  record_t(double sid_start_, double sid_end_, double dir_, double eig_vector_x_, double eig_vector_y_){
    sid_start = sid_start_;
    sid_end = sid_end_;
    dir = dir_;
    eig_vector_x = eig_vector_x_;
    eig_vector_y = eig_vector_y_;
  }
  record_t(){}
}record_t;


typedef struct critical_point_t{
  double x[2];
  double eig_vec[2][2];
  double V[3][2];
  double X[3][2];
  double Jac[2][2];
  std::complex<double> eig[2];
  // double mu[3];
  int type;
  size_t simplex_id;
  critical_point_t(double* x_, double eig_[2], double eig_v[2][2], double Jac_[2][2], double V_[3][2],double X_[3][2], int t_, size_t simplex_id_){
    x[0] = x_[0];
    x[1] = x_[1];
    eig[0] = eig_[0];
    eig[1] = eig_[1];
    eig_vec[0][0] = eig_v[0][0];
    eig_vec[0][1] = eig_v[0][1];
    eig_vec[1][0] = eig_v[1][0];
    eig_vec[1][1] = eig_v[1][1];
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 2; j++) {
        V[i][j] = V_[i][j];
      }
    }
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 2; j++) {
        X[i][j] = X_[i][j];
      }
    }
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        Jac[i][j] = Jac_[i][j];
      }
    }
    // for (int i = 0; i < 3; i++) {
    //   mu[i] = mu_[i];
    // }
    type = t_;
    simplex_id = simplex_id_;
  }
  critical_point_t(){}
}critical_point_t;

#define DEFAULT_EB 1
#define SINGULAR 0
#define ATTRACTING 1 // 2 real negative eigenvalues
#define REPELLING 2 // 2 real positive eigenvalues
#define SADDLE 3// 1 real negative and 1 real positive
#define ATTRACTING_FOCUS 4 // complex with negative real
#define REPELLING_FOCUS 5 // complex with positive real
#define CENTER 6 // complex with 0 real

std::string get_critical_point_type_string(int type);

void refill_gradient(int id,const int DH,const int DW, const float* grad_tmp, ftk::ndarray<double>& grad);

template<typename T>
static void 
check_simplex_seq_saddle(const T v[3][2], const double X[3][2], const int indices[3], int i, int j, int simplex_id, std::unordered_map<int, critical_point_t>& critical_points);

template<typename T>
std::unordered_map<int, critical_point_t>
compute_critical_points(const T * U, const T * V, int r1, int r2);

