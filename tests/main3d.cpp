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
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

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
#define CPSZ_OMP_FLAG 1

#define TPSZ_OMP_FLAG 1 //usually set to 1

#define TRANSTER_TEST_FLAG 0
#define SPEED_TEST_FLAG 1
#define RECORD_INTERMEDIATE_TRAJECTORY 0
#define SAVE_OUTPUT_FILE 0
using namespace std;

template <class T>
void posix_write(std::string filename, T * data, size_t num_elements){
  int fd = open(filename.c_str(), O_CREAT|O_WRONLY|O_TRUNC, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  write(fd, data, num_elements * sizeof(T));
  fsync(fd);
  close(fd);
}

template <class T>
void posix_read(std::string filename, T * data, size_t num_elements){
  int fd = open(filename.c_str(), O_RDONLY);
  read(fd, data, num_elements * sizeof(T));
  close(fd);
}

typedef struct traj_config{
  double h;
  double eps;
  int max_length; 
} traj_config;

struct critical_point_t_3d {
  double x[3];
  int type;
  size_t simplex_id;
  double eigvalues[3];
  double eig_vec[3][3];
  critical_point_t_3d(){}
};

struct thresholds
{
  double threshold_div;
  double threshold_out;
  double threshold_max;
};
struct result_struct
{
  double pre_compute_cp_time_;
  double begin_cr_;
  double psnr_cpsz_overall_;
  int current_round_;
  double elapsed_alg_time_;
  double cpsz_comp_time_;
  double cpsz_decomp_time_;
  int rounds_;
  // pass trajID_need_fix_next_vec
  std::vector<int> trajID_need_fix_next_vec_;
  //trajID_need_fix_next_detail_vec
  std::vector<std::array<int,3>> trajID_need_fix_next_detail_vec_;
  std::array<int,3> origin_traj_detail_;

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

void record_criticalpoints(const std::string& prefix, const std::unordered_map<size_t, critical_point_t_3d>& cps, bool write_sid=false){
  std::vector<double> singular;
  std::vector<double> source;
  std::vector<double> saddle;
  std::vector<double> sink;
  for (const auto& cpd:cps){
    auto cp = cpd.second;
    if (cp.type == SINGULAR){
      singular.push_back(cp.x[0]);
      singular.push_back(cp.x[1]);
      singular.push_back(cp.x[2]);
    }
    else if (cp.type == STABLE_SOURCE || cp.type == UNSTABLE_SOURCE){
      source.push_back(cp.x[0]);
      source.push_back(cp.x[1]);
      source.push_back(cp.x[2]);
    }
    else if (cp.type >= 3 && cp.type <= 6){
      saddle.push_back(cp.x[0]);
      saddle.push_back(cp.x[1]);
      saddle.push_back(cp.x[2]);
    }
    else if (cp.type == STABLE_SINK || cp.type == UNSTABLE_SINK){
      sink.push_back(cp.x[0]);
      sink.push_back(cp.x[1]);
      sink.push_back(cp.x[2]);
    }
    else {
      continue;
    }
  }
  // std::string prefix = "../data/position";
  writefile((prefix + "_singular.dat").c_str(), singular.data(), singular.size());
  writefile((prefix + "_source.dat").c_str(), source.data(), source.size());
  writefile((prefix + "_saddle.dat").c_str(), saddle.data(), saddle.size());
  writefile((prefix + "_sink.dat").c_str(), sink.data(), sink.size());
  printf("Successfully write critical points to file %s\n", prefix.c_str());

}





double euclideanDistance(const array<double, 3>& p, const array<double, 3>& q) {
    return sqrt((p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) + (p[2] - q[2]) * (p[2] - q[2]));
}

double frechetDistance(const vector<array<double, 3>>& P, const vector<array<double, 3>>& Q) {
    int n = P.size();
    int m = Q.size();
    vector<vector<double>> dp(n, vector<double>(m, -1.0));

    // 初始化第一个元素
    dp[0][0] = euclideanDistance(P[0], Q[0]);

    // 计算第一列
    for (int i = 1; i < n; i++) {
        dp[i][0] = max(dp[i-1][0], euclideanDistance(P[i], Q[0]));
    }

    // 计算第一行
    for (int j = 1; j < m; j++) {
        dp[0][j] = max(dp[0][j-1], euclideanDistance(P[0], Q[j]));
    }

    for (int i = 1; i < n; i++) {
        for (int j = 1; j < m; j++) {
            dp[i][j] = max(min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]}), euclideanDistance(P[i], Q[j]));
        }
    }

    return dp[n-1][m-1];
}

double ESfrechetDistance(const vector<array<double, 3>>& P, const vector<array<double, 3>>& Q) {
    int n = P.size();
    int m = Q.size();
    vector<vector<double>> dp(n, vector<double>(m, -1.0));  // 初始化 DP 矩阵

    // 初始化第一个元素
    dp[0][0] = euclideanDistance(P[0], Q[0]);

    // 计算第一列
    for (int i = 1; i < n; i++) {
        dp[i][0] = max(dp[i-1][0], euclideanDistance(P[i], Q[0]));
    }

    // 计算第一行
    for (int j = 1; j < m; j++) {
        dp[0][j] = max(dp[0][j-1], euclideanDistance(P[0], Q[j]));
    }

    // 计算整个矩阵，并在 `P` 走到最后一行时提前返回结果
    for (int i = 1; i < n; i++) {
        for (int j = 1; j < m; j++) {
            dp[i][j] = max(min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]}), euclideanDistance(P[i], Q[j]));
        }
    }

    // `P` 走到头时返回最后一行中最小的 Frechet Distance
    double result = dp[n-1][0];  // 初始化为最后一行的第一个元素
    for (int j = 1; j < m; j++) {
        result = min(result, dp[n-1][j]);  // 找到最后一行的最小值
    }

    return result;
}


void calculateStatistics(const vector<double>& data, double& minVal, double& maxVal, double& medianVal, double& meanVal, double& stdevVal) {
    // 检查输入是否为空
    if (data.empty()) {
        cerr << "Error: The data set is empty!" << endl;
        return;
    }

    // 1. 计算最小值和最大值
    minVal = *min_element(data.begin(), data.end());
    maxVal = *max_element(data.begin(), data.end());

    // 2. 计算均值
    double sum = 0.0;
    for (double value : data) {
        sum += value;
    }
    meanVal = sum / data.size();

    // 3. 计算中位数
    vector<double> sortedData = data; // 创建数据的副本用于排序
    sort(sortedData.begin(), sortedData.end());
    size_t n = sortedData.size();
    
    if (n % 2 == 0) {
        medianVal = (sortedData[n / 2 - 1] + sortedData[n / 2]) / 2.0;
    } else {
        medianVal = sortedData[n / 2];
    }

    // 4. 计算标准差
    double varianceSum = 0.0;
    for (double value : data) {
        varianceSum += (value - meanVal) * (value - meanVal);
    }
    stdevVal = sqrt(varianceSum / data.size());
}

template<typename Type>
void verify(Type * ori_data, Type * data, size_t num_elements, double &nrmse){
    size_t i = 0;
    Type Max = 0, Min = 0, diffMax = 0;
    Max = ori_data[0];
    Min = ori_data[0];
    diffMax = fabs(data[0] - ori_data[0]);
    size_t k = 0;
    double sum1 = 0, sum2 = 0;
    for (i = 0; i < num_elements; i++){
        sum1 += ori_data[i];
        sum2 += data[i];
    }
    double mean1 = sum1/num_elements;
    double mean2 = sum2/num_elements;

    double sum3 = 0, sum4 = 0;
    double sum = 0, prodSum = 0, relerr = 0, abserr = 0;

    double maxpw_relerr = 0; 
    double maxpw_abserr = 0;
    for (i = 0; i < num_elements; i++){
        if (Max < ori_data[i]) Max = ori_data[i];
        if (Min > ori_data[i]) Min = ori_data[i];
        
        Type err = fabs(data[i] - ori_data[i]);
        if(ori_data[i]!=0 && fabs(ori_data[i])>1)
        {
            relerr = err/fabs(ori_data[i]);
            
            if(maxpw_relerr<relerr){
              maxpw_relerr = relerr;
            }
            if (maxpw_abserr < err){
              maxpw_abserr = err;
            }
        }

            

        if (diffMax < err)
            diffMax = err;
        prodSum += (ori_data[i]-mean1)*(data[i]-mean2);
        sum3 += (ori_data[i] - mean1)*(ori_data[i]-mean1);
        sum4 += (data[i] - mean2)*(data[i]-mean2);
        sum += err*err; 
    }
    double std1 = sqrt(sum3/num_elements);
    double std2 = sqrt(sum4/num_elements);
    double ee = prodSum/num_elements;
    double acEff = ee/std1/std2;

    double mse = sum/num_elements;
    double range = Max - Min;
    double psnr = 20*log10(range)-10*log10(mse);
    nrmse = sqrt(mse)/range;

    printf ("Min=%.20G, Max=%.20G, range=%.20G\n", Min, Max, range);
    printf ("Max absolute error = %.10f\n", diffMax);
    printf ("Max relative error = %f\n", diffMax/(Max-Min));
    printf ("Max pw relative error = %f\n", maxpw_relerr);
    printf ("Max pw absolute error = %f\n", maxpw_abserr);
    printf ("PSNR = %f, NRMSE= %.20G\n", psnr,nrmse);
    printf ("acEff=%f\n", acEff);   
}


std::array<double, 3> findLastNonNegativeOne(const std::vector<std::array<double, 3>>& vec) {
    for (auto it = vec.rbegin(); it != vec.rend(); ++it) {
        if ((*it)[0] != -1) {
            return *it;
        }
    }
    // 如果没有找到非 (-1, -1) 的点，可以返回一个特定的值，例如 {-1, -1}
    return vec.back();
}

bool LastTwoPointsAreEqual(const std::vector<std::array<double, 3>>& vec) {
    if (vec.size() < 2) return false;
    return vec[vec.size()-1] == vec[vec.size()-2];
}

bool SearchElementFromBack(const std::vector<std::array<double, 3>>& vec, const std::array<double, 3>& target) {
    for (auto it = vec.rbegin(); it != vec.rend(); ++it) {
        if (*it == target) {
            return true;
        }
    }
    return false;
}
int SearchElementFromBackIndex(const std::vector<std::array<double, 3>>& vec, const std::array<double, 3>& target) {
    for (int i = vec.size()-1; i >= 0; i--) {
        if (vec[i] == target) {
            return i;
        }
    }
    return -1;
}

// bool OutSameEdge(const std::array<double, 3>& p1, const std::array<double, 3>& p2, int DW, int DH, int DD){
//     std::string face1 = findClosestFace(p1, DW, DH, DD);
//     std::string face2 = findClosestFace(p2, DW, DH, DD);
//     return face1 == face2;
// }

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
    Eigen::Vector3d eigenvalues_vec = solver.eigenvalues().real().eval(); //似乎是要加上eval， 因为是lazy evaluation
    
    // 获取特征向量
    Eigen::Matrix3d eigenvectors_mat = solver.eigenvectors().real().eval();

    // 将 Eigen 的结果复制到原生数组
    for (int i = 0; i < 3; ++i) {
        eigenvalues[i] = eigenvalues_vec[i];
        for (int j = 0; j < 3; ++j) {
            eigenvectors[i][j] = eigenvectors_mat(i, j);
        }
    }
}

// void computeEigenvaluesAndEigenvectores_ftk(const double (&J)[3][3], double (&eigenvalues)[3], double (&eigenvectors)[3][3]){
//   ftk::solve_eigenvalues_symmetric3x3(J, eigenvalues);
//   ftk::solve_eigenvectors3x3(J, 3, eigenvalues, eigenvectors);
// }
void 
check_simplex_seq(const double v[4][3], const double X[3][3], const int indices[4], int i, int j, int k, size_t simplex_id, std::unordered_map<size_t, critical_point_t_3d>& critical_points){
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
  double threshold = 0.0;
  bool succ2 = ftk::inverse_lerp_s3v3(v, mu, &cond, threshold);
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
  if (cp.type >= 3 && cp.type <= 6){
    computeEigenvaluesAndEigenvectors(J, eigenvalues, eigenvec);
    //computeEigenvaluesAndEigenvectores_ftk(J, eigenvalues, eigenvec);
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
  }

  critical_points[simplex_id] = cp;
}

void 
omp_check_simplex_seq(const double v[4][3], const double X[3][3], const int indices[4], int i, int j, int k, size_t simplex_id, std::vector<critical_point_t_3d>& critical_points){
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
  double threshold = 0.0;
  bool succ2 = ftk::inverse_lerp_s3v3(v, mu, &cond, threshold);
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
  if (cp.type >= 3 && cp.type <= 6){
    computeEigenvaluesAndEigenvectors(J, eigenvalues, eigenvec);
    //computeEigenvaluesAndEigenvectores_ftk(J, eigenvalues, eigenvec);
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
  }

  critical_points[simplex_id] = cp;
}


template<typename T>
std::unordered_map<size_t, critical_point_t_3d>
compute_critical_points(const T * U, const T * V, const T * W, int r1, int r2, int r3){ //r1=DD,r2=DH,r3=DW
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
  std::unordered_map<size_t, critical_point_t_3d> critical_points;
  for(int i=1; i<r1-2; i++){
    if(i%10==0) std::cout << i << " / " << r1-1 << std::endl;
    for(int j=1; j<r2-2; j++){
      for(int k=1; k<r3-2; k++){
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


// template<typename T>
// std::unordered_map<size_t, critical_point_t_3d>
// omp_compute_critical_points(const T * U, const T * V, const T * W, int r1, int r2, int r3){
//   //set vector length to 6*r1*r2*r3
//   std::vector<critical_point_t_3d> critical_points_vec(6*r1*r2*r3);
//   // check cp for all cells
//   ptrdiff_t dim0_offset = r2*r3;
//   ptrdiff_t dim1_offset = r3;
//   ptrdiff_t cell_dim0_offset = (r2-1)*(r3-1);
//   ptrdiff_t cell_dim1_offset = r3-1;
//   size_t num_elements = r1*r2*r3;
//   int indices[4] = {0};
//   double v[4][3] = {0};
//   double actual_coords[6][4][3];
//   for(int i=0; i<6; i++){
//     for(int j=0; j<4; j++){
//       for(int k=0; k<3; k++){
//         actual_coords[i][j][k] = tet_coords[i][j][k];
//       }
//     }
//   }
//   #pragma omp parallel for collapse(3)
//   for(int i=1; i<r1-2; i++){
//     for(int j=1; j<r2-2; j++){
//       for(int k=1; k<r3-2; k++){
//         // order (reserved, z->x):
//         // ptrdiff_t cell_offset = 6*(i*cell_dim0_offset + j*cell_dim1_offset + k);
//         // ftk index
//         ptrdiff_t cell_offset = 6*(i*dim0_offset + j*dim1_offset + k);
//         // (ftk-0) 000, 001, 011, 111
//         update_index_and_value(v, indices, 0, i*dim0_offset + j*dim1_offset + k, U, V, W);
//         update_index_and_value(v, indices, 1, (i+1)*dim0_offset + j*dim1_offset + k, U, V, W);
//         update_index_and_value(v,indices, 2, (i+1)*dim0_offset + (j+1)*dim1_offset + k, U, V, W);
//         update_index_and_value(v, indices, 3, (i+1)*dim0_offset + (j+1)*dim1_offset + (k+1), U, V, W);
//         omp_check_simplex_seq(v, actual_coords[0], indices, i, j, k, cell_offset, critical_points_vec); 
//         // (ftk-2) 000, 010, 011, 111
//         update_index_and_value(v,indices, 1, i*dim0_offset + (j+1)*dim1_offset + k, U, V, W);
//         omp_check_simplex_seq(v, actual_coords[1], indices, i, j, k, cell_offset + 2, critical_points_vec);
//         // (ftk-1) 000, 001, 101, 111
//         update_index_and_value(v,indices, 1, (i+1)*dim0_offset + j*dim1_offset + k, U, V, W);
//         update_index_and_value(v, indices, 2, (i+1)*dim0_offset + j*dim1_offset + k+1, U, V, W);
//         omp_check_simplex_seq(v, actual_coords[2], indices, i, j, k, cell_offset + 1, critical_points_vec);
//         // (ftk-4) 000, 100, 101, 111
//         update_index_and_value(v, indices, 1, i*dim0_offset + j*dim1_offset + k+1, U, V, W);
//         omp_check_simplex_seq(v, actual_coords[3], indices, i, j, k, cell_offset + 4, critical_points_vec);
//         // (ftk-3) 000, 010, 110, 111
//         update_index_and_value(v,indices, 1, i*dim0_offset + (j+1)*dim1_offset + k, U, V, W);
//         update_index_and_value(v,indices, 2, i*dim0_offset + (j+1)*dim1_offset + k+1, U, V, W);
//         omp_check_simplex_seq(v, actual_coords[4], indices, i, j, k, cell_offset + 3, critical_points_vec);
//         // (ftk-5) 000, 100, 110, 111
//         update_index_and_value(v,indices, 1, i*dim0_offset + j*dim1_offset + k+1, U, V, W);
//         omp_check_simplex_seq(v, actual_coords[5], indices, i, j, k, cell_offset + 5, critical_points_vec);
//       }
//     }
//   }
//   //convert vector to unordered_map
//   std::unordered_map<size_t, critical_point_t_3d> critical_points;
//   for(int i=0; i<critical_points_vec.size(); i++){
//   }
//   return critical_points;
// }

template <typename T>
std::unordered_map<size_t, critical_point_t_3d> omp_compute_critical_points(const T* U, const T* V, const T* W, int r1, int r2, int r3) {
    ptrdiff_t dim0_offset = r2 * r3;
    ptrdiff_t dim1_offset = r3;

    // 使用线程数初始化一个 vector，用于保存每个线程的结果
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }
    std::vector<std::unordered_map<size_t, critical_point_t_3d>> thread_results(num_threads);

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num(); // 获取当前线程 ID
        std::unordered_map<size_t, critical_point_t_3d>& local_critical_points = thread_results[thread_id];

        #pragma omp for nowait
        for (int i = 1; i < r1 - 2; i++) {
            if (i % 10 == 0) std::cout << i << " / " << r1 - 1 << std::endl;
            for (int j = 1; j < r2 - 2; j++) {
                for (int k = 1; k < r3 - 2; k++) {
                    int indices[4] = {0};
                    double v[4][3] = {0};
                    double actual_coords[6][4][3];

                    for (int m = 0; m < 6; m++) {
                        for (int n = 0; n < 4; n++) {
                            for (int p = 0; p < 3; p++) {
                                actual_coords[m][n][p] = tet_coords[m][n][p];
                            }
                        }
                    }

                    ptrdiff_t cell_offset = 6 * (i * dim0_offset + j * dim1_offset + k);

                    // 更新顶点并检查每个 tetrahedron (simplex)
                    update_index_and_value(v, indices, 0, i * dim0_offset + j * dim1_offset + k, U, V, W);
                    update_index_and_value(v, indices, 1, (i + 1) * dim0_offset + j * dim1_offset + k, U, V, W);
                    update_index_and_value(v, indices, 2, (i + 1) * dim0_offset + (j + 1) * dim1_offset + k, U, V, W);
                    update_index_and_value(v, indices, 3, (i + 1) * dim0_offset + (j + 1) * dim1_offset + (k + 1), U, V, W);
                    check_simplex_seq(v, actual_coords[0], indices, i, j, k, cell_offset, local_critical_points);
                    
                    update_index_and_value(v, indices, 1, i * dim0_offset + (j + 1) * dim1_offset + k, U, V, W);
                    check_simplex_seq(v, actual_coords[1], indices, i, j, k, cell_offset + 2, local_critical_points);
                    
                    update_index_and_value(v, indices, 1, (i + 1) * dim0_offset + j * dim1_offset + k, U, V, W);
                    update_index_and_value(v, indices, 2, (i + 1) * dim0_offset + j * dim1_offset + k + 1, U, V, W);
                    check_simplex_seq(v, actual_coords[2], indices, i, j, k, cell_offset + 1, local_critical_points);
                    
                    update_index_and_value(v, indices, 1, i * dim0_offset + j * dim1_offset + k + 1, U, V, W);
                    check_simplex_seq(v, actual_coords[3], indices, i, j, k, cell_offset + 4, local_critical_points);
                    
                    update_index_and_value(v, indices, 1, i * dim0_offset + (j + 1) * dim1_offset + k, U, V, W);
                    update_index_and_value(v, indices, 2, i * dim0_offset + (j + 1) * dim1_offset + k + 1, U, V, W);
                    check_simplex_seq(v, actual_coords[4], indices, i, j, k, cell_offset + 3, local_critical_points);
                    
                    update_index_and_value(v, indices, 1, i * dim0_offset + j * dim1_offset + k + 1, U, V, W);
                    check_simplex_seq(v, actual_coords[5], indices, i, j, k, cell_offset + 5, local_critical_points);
                }
            }
        }
    }

    // 合并所有线程的结果，确保顺序一致
    std::unordered_map<size_t, critical_point_t_3d> critical_points;
    for (int tid = 0; tid < num_threads; ++tid) {
        critical_points.insert(thread_results[tid].begin(), thread_results[tid].end());
    }

    return critical_points;
}
template<typename Type>
void updateOffsets(const Type* p, const int DW, const int DH, const int DD, std::vector<std::set<size_t>>& thread_lossless_index,int thread_id) {
    auto coords = get_four_offsets(p, DW, DH, DD);
    thread_lossless_index[thread_id].insert(coords.begin(), coords.end());
}



template<typename Type>
std::array<Type, 3> newRK4_3d(const Type * x, const Type * v, const ftk::ndarray<float> &data,  Type h, const int DW, const int DH, const int DD, std::set<size_t>& lossless_index, bool verbose = false) {
  // x and y are positions, and h is the step size
  double rk1[3] = {0}, rk2[3] = {0}, rk3[3] = {0}, rk4[3] = {0};
  double p1[3] = {x[0], x[1],x[2]};
  double p2[3] = {0}, p3[3] = {0}, p4[3] = {0};
  std::array<Type, 3> result = {0,0,0};


  if(!inside_domain(p1, DH, DW, DD)){
    //return std::array<Type, 2>{x[0], x[1]};
    //return std::array<Type, 3>{-1, -1,-1};
    //改成如果出界就返回当前位置
    if (verbose){
    std::cout << "p1: " << p1[0] << " " << p1[1] << " " << p1[2] << std::endl;
    std::cout << "p2: " << p2[0] << " " << p2[1] << " " << p2[2] << std::endl;
    std::cout << "p3: " << p3[0] << " " << p3[1] << " " << p3[2] << std::endl;
    std::cout << "p4: " << p4[0] << " " << p4[1] << " " << p4[2] << std::endl;
    std::cout << "rk1: " << rk1[0] << " " << rk1[1] << " " << rk1[2] << std::endl;
    std::cout << "rk2: " << rk2[0] << " " << rk2[1] << " " << rk2[2] << std::endl;
    std::cout << "rk3: " << rk3[0] << " " << rk3[1] << " " << rk3[2] << std::endl;
    std::cout << "rk4: " << rk4[0] << " " << rk4[1] << " " << rk4[2] << std::endl;
    std::cout << "result: " << result[0] << " " << result[1] << " " << result[2] << std::endl;
    }
    return {x[0], x[1], x[2]};
  }
  interp3d_new(p1, rk1,data);
  auto coords_p1 = get_four_offsets(p1, DW, DH,DD);
  // for (auto offset:coords){
  //   lossless_index.insert(offset);
  // }
  
  // p2 = {x[0] + 0.5 * h * rk1[0], x[1] + 0.5 * h * rk1[1],x[2] + 0.5 * h * rk1[2]};
  p2[0] = x[0] + 0.5 * h * rk1[0];
  p2[1] = x[1] + 0.5 * h * rk1[1];
  p2[2] = x[2] + 0.5 * h * rk1[2];
  if (!inside_domain(p2, DH, DW, DD)){
    //return std::array<Type, 2>{p1[0], p1[1]};
    // return std::array<Type, 3>{-1, -1,-1};
    if (verbose){
    std::cout << "p1: " << p1[0] << " " << p1[1] << " " << p1[2] << std::endl;
    std::cout << "p2: " << p2[0] << " " << p2[1] << " " << p2[2] << std::endl;
    std::cout << "p3: " << p3[0] << " " << p3[1] << " " << p3[2] << std::endl;
    std::cout << "p4: " << p4[0] << " " << p4[1] << " " << p4[2] << std::endl;
    std::cout << "rk1: " << rk1[0] << " " << rk1[1] << " " << rk1[2] << std::endl;
    std::cout << "rk2: " << rk2[0] << " " << rk2[1] << " " << rk2[2] << std::endl;
    std::cout << "rk3: " << rk3[0] << " " << rk3[1] << " " << rk3[2] << std::endl;
    std::cout << "rk4: " << rk4[0] << " " << rk4[1] << " " << rk4[2] << std::endl;
    std::cout << "result: " << result[0] << " " << result[1] << " " << result[2] << std::endl;
    }
    lossless_index.insert(coords_p1.begin(), coords_p1.end());
    return {x[0], x[1], x[2]};
  }
  interp3d_new(p2, rk2,data);
  auto coords_p2 = get_four_offsets(p2, DW, DH,DD);
  // for (auto offset:coords_p2){
  //   lossless_index.insert(offset);
  // }
  
  // p3 = {x[0] + 0.5 * h * rk2[0], x[1] + 0.5 * h * rk2[1],x[2] + 0.5 * h * rk2[2]};
  p3[0] = x[0] + 0.5 * h * rk2[0];
  p3[1] = x[1] + 0.5 * h * rk2[1];
  p3[2] = x[2] + 0.5 * h * rk2[2];
  if (!inside_domain(p3, DH, DW, DD)){
    // return std::array<Type, 3>{-1, -1,-1};
    if (verbose){
    std::cout << "p1: " << p1[0] << " " << p1[1] << " " << p1[2] << std::endl;
    std::cout << "p2: " << p2[0] << " " << p2[1] << " " << p2[2] << std::endl;
    std::cout << "p3: " << p3[0] << " " << p3[1] << " " << p3[2] << std::endl;
    std::cout << "p4: " << p4[0] << " " << p4[1] << " " << p4[2] << std::endl;
    std::cout << "rk1: " << rk1[0] << " " << rk1[1] << " " << rk1[2] << std::endl;
    std::cout << "rk2: " << rk2[0] << " " << rk2[1] << " " << rk2[2] << std::endl;
    std::cout << "rk3: " << rk3[0] << " " << rk3[1] << " " << rk3[2] << std::endl;
    std::cout << "rk4: " << rk4[0] << " " << rk4[1] << " " << rk4[2] << std::endl;
    std::cout << "result: " << result[0] << " " << result[1] << " " << result[2] << std::endl;
    }
    lossless_index.insert(coords_p1.begin(), coords_p1.end());
    lossless_index.insert(coords_p2.begin(), coords_p2.end());
    return {x[0], x[1], x[2]};
  }
  interp3d_new(p3, rk3,data);
  auto coords_p3 = get_four_offsets(p3, DW, DH,DD);
  // for (auto offset:coords_p3){
  //   lossless_index.insert(offset);
  // }
  
  // p4 = {x[0] + h * rk3[0], x[1] + h * rk3[1],x[2] + h * rk3[2]};
  p4[0] = x[0] + h * rk3[0];
  p4[1] = x[1] + h * rk3[1];
  p4[2] = x[2] + h * rk3[2];
  if (!inside_domain(p4, DH, DW,DD)){
    // return std::array<Type, 3>{-1, -1,-1};
    if (verbose){
    std::cout << "p1: " << p1[0] << " " << p1[1] << " " << p1[2] << std::endl;
    std::cout << "p2: " << p2[0] << " " << p2[1] << " " << p2[2] << std::endl;
    std::cout << "p3: " << p3[0] << " " << p3[1] << " " << p3[2] << std::endl;
    std::cout << "p4: " << p4[0] << " " << p4[1] << " " << p4[2] << std::endl;
    std::cout << "rk1: " << rk1[0] << " " << rk1[1] << " " << rk1[2] << std::endl;
    std::cout << "rk2: " << rk2[0] << " " << rk2[1] << " " << rk2[2] << std::endl;
    std::cout << "rk3: " << rk3[0] << " " << rk3[1] << " " << rk3[2] << std::endl;
    std::cout << "rk4: " << rk4[0] << " " << rk4[1] << " " << rk4[2] << std::endl;
    std::cout << "result: " << result[0] << " " << result[1] << " " << result[2] << std::endl;
    }
    lossless_index.insert(coords_p1.begin(), coords_p1.end());
    lossless_index.insert(coords_p2.begin(), coords_p2.end());
    lossless_index.insert(coords_p3.begin(), coords_p3.end());
    return {x[0], x[1], x[2]};
  }
  interp3d_new(p4, rk4,data);
  auto coords_p4 = get_four_offsets(p4, DW, DH,DD);
  // for (auto offset:coords_p4){
  //   lossless_index.insert(offset);
  // }

  
  Type next_x = x[0] + h * (rk1[0] + 2 * rk2[0] + 2 * rk3[0] + rk4[0]) / 6.0;
  Type next_y = x[1] + h * (rk1[1] + 2 * rk2[1] + 2 * rk3[1] + rk4[1]) / 6.0;
  Type next_z = x[2] + h * (rk1[2] + 2 * rk2[2] + 2 * rk3[2] + rk4[2]) / 6.0;
  result[0] = next_x;
  result[1] = next_y;
  result[2] = next_z;
  if (!inside_domain(result, DH, DW,DD)){
    // return std::array<Type, 3>{-1, -1,-1};
    if (verbose){
    std::cout << "p1: " << p1[0] << " " << p1[1] << " " << p1[2] << std::endl;
    std::cout << "p2: " << p2[0] << " " << p2[1] << " " << p2[2] << std::endl;
    std::cout << "p3: " << p3[0] << " " << p3[1] << " " << p3[2] << std::endl;
    std::cout << "p4: " << p4[0] << " " << p4[1] << " " << p4[2] << std::endl;
    std::cout << "rk1: " << rk1[0] << " " << rk1[1] << " " << rk1[2] << std::endl;
    std::cout << "rk2: " << rk2[0] << " " << rk2[1] << " " << rk2[2] << std::endl;
    std::cout << "rk3: " << rk3[0] << " " << rk3[1] << " " << rk3[2] << std::endl;
    std::cout << "rk4: " << rk4[0] << " " << rk4[1] << " " << rk4[2] << std::endl;
    std::cout << "result: " << result[0] << " " << result[1] << " " << result[2] << std::endl;
    }
    lossless_index.insert(coords_p1.begin(), coords_p1.end());
    lossless_index.insert(coords_p2.begin(), coords_p2.end());
    lossless_index.insert(coords_p3.begin(), coords_p3.end());
    lossless_index.insert(coords_p4.begin(), coords_p4.end());
    return {x[0], x[1],x[2]};
  }
  auto coords_final = get_four_offsets(result, DW, DH, DD);
  lossless_index.insert(coords_p1.begin(), coords_p1.end());
  lossless_index.insert(coords_p2.begin(), coords_p2.end());
  lossless_index.insert(coords_p3.begin(), coords_p3.end());
  lossless_index.insert(coords_p4.begin(), coords_p4.end());
  lossless_index.insert(coords_final.begin(), coords_final.end());
  if (verbose){
    std::cout << "p1: " << p1[0] << " " << p1[1] << " " << p1[2] << std::endl;
    std::cout << "p2: " << p2[0] << " " << p2[1] << " " << p2[2] << std::endl;
    std::cout << "p3: " << p3[0] << " " << p3[1] << " " << p3[2] << std::endl;
    std::cout << "p4: " << p4[0] << " " << p4[1] << " " << p4[2] << std::endl;
    std::cout << "rk1: " << rk1[0] << " " << rk1[1] << " " << rk1[2] << std::endl;
    std::cout << "rk2: " << rk2[0] << " " << rk2[1] << " " << rk2[2] << std::endl;
    std::cout << "rk3: " << rk3[0] << " " << rk3[1] << " " << rk3[2] << std::endl;
    std::cout << "rk4: " << rk4[0] << " " << rk4[1] << " " << rk4[2] << std::endl;
    std::cout << "result: " << result[0] << " " << result[1] << " " << result[2] << std::endl;
  }

  return result;
}

// newRK4_3d 函数
template<typename Type>
std::array<Type, 3> newRK4_3d_parallel(const Type* x, const Type* v, const ftk::ndarray<float>& data, Type h, const int DW, const int DH, const int DD, std::vector<std::set<size_t>>& thread_lossless_index,int thread_id,bool verbose = false) {
    double rk1[3] = {0}, rk2[3] = {0}, rk3[3] = {0}, rk4[3] = {0};
    double p1[3] = {x[0], x[1], x[2]};
    double p2[3] = {0}, p3[3] = {0}, p4[3] = {0};
    std::array<Type, 3> result = {0, 0, 0};

    if (!inside_domain(p1, DH, DW, DD)){
        // return {-1, -1, -1};
        return {x[0], x[1], x[2]};
    } 
    // updateOffsets(p1, DW, DH, DD, thread_lossless_index, thread_id);
    interp3d_new(p1, rk1, data);

    // const double p2[3] = {x[0] + 0.5 * h * rk1[0], x[1] + 0.5 * h * rk1[1], x[2] + 0.5 * h * rk1[2]};
    p2[0] = x[0] + 0.5 * h * rk1[0];
    p2[1] = x[1] + 0.5 * h * rk1[1];
    p2[2] = x[2] + 0.5 * h * rk1[2];
    if (!inside_domain(p2, DH, DW, DD)) {
        // return {-1, -1, -1};
        updateOffsets(p1, DW, DH, DD, thread_lossless_index, thread_id);
        return {x[0], x[1], x[2]};
    }
    // updateOffsets(p2, DW, DH, DD, thread_lossless_index, thread_id);
    interp3d_new(p2, rk2, data);

    // const double p3[3] = {x[0] + 0.5 * h * rk2[0], x[1] + 0.5 * h * rk2[1], x[2] + 0.5 * h * rk2[2]};
    p3[0] = x[0] + 0.5 * h * rk2[0];
    p3[1] = x[1] + 0.5 * h * rk2[1];
    p3[2] = x[2] + 0.5 * h * rk2[2];
    if (!inside_domain(p3, DH, DW, DD)) {
        // return {-1, -1, -1};
        updateOffsets(p1, DW, DH, DD, thread_lossless_index, thread_id);
        updateOffsets(p2, DW, DH, DD, thread_lossless_index, thread_id);
        return {x[0], x[1], x[2]};
    }
    // updateOffsets(p3, DW, DH, DD, thread_lossless_index, thread_id);
    interp3d_new(p3, rk3, data);

    // const double p4[3] = {x[0] + h * rk3[0], x[1] + h * rk3[1], x[2] + h * rk3[2]};
    p4[0] = x[0] + h * rk3[0];
    p4[1] = x[1] + h * rk3[1];
    p4[2] = x[2] + h * rk3[2];
    if (!inside_domain(p4, DH, DW, DD)) {
        // return {-1, -1, -1};
        updateOffsets(p1, DW, DH, DD, thread_lossless_index, thread_id);
        updateOffsets(p2, DW, DH, DD, thread_lossless_index, thread_id);
        updateOffsets(p3, DW, DH, DD, thread_lossless_index, thread_id);
        return {x[0], x[1], x[2]};
    };
    // updateOffsets(p4, DW, DH, DD, thread_lossless_index, thread_id);
    interp3d_new(p4, rk4, data);

    Type next_x = x[0] + h * (rk1[0] + 2 * rk2[0] + 2 * rk3[0] + rk4[0]) / 6.0;
    Type next_y = x[1] + h * (rk1[1] + 2 * rk2[1] + 2 * rk3[1] + rk4[1]) / 6.0;
    Type next_z = x[2] + h * (rk1[2] + 2 * rk2[2] + 2 * rk3[2] + rk4[2]) / 6.0;
    result[0] = next_x;
    result[1] = next_y;
    result[2] = next_z;

    if (!inside_domain(result, DH, DW, DD)){
        // return {-1, -1, -1};
        updateOffsets(p1, DW, DH, DD, thread_lossless_index, thread_id);
        updateOffsets(p2, DW, DH, DD, thread_lossless_index, thread_id);
        updateOffsets(p3, DW, DH, DD, thread_lossless_index, thread_id);
        updateOffsets(p4, DW, DH, DD, thread_lossless_index, thread_id);
        return {x[0], x[1], x[2]};
    }
    updateOffsets(p1, DW, DH, DD, thread_lossless_index, thread_id);
    updateOffsets(p2, DW, DH, DD, thread_lossless_index, thread_id);
    updateOffsets(p3, DW, DH, DD, thread_lossless_index, thread_id);
    updateOffsets(p4, DW, DH, DD, thread_lossless_index, thread_id);
    updateOffsets(result.data(), DW, DH, DD, thread_lossless_index, thread_id);
    if (verbose){
        std::cout << "p1: " << p1[0] << " " << p1[1] << " " << p1[2] << std::endl;
        std::cout << "p2: " << p2[0] << " " << p2[1] << " " << p2[2] << std::endl;
        std::cout << "p3: " << p3[0] << " " << p3[1] << " " << p3[2] << std::endl;
        std::cout << "p4: " << p4[0] << " " << p4[1] << " " << p4[2] << std::endl;
        std::cout << "rk1: " << rk1[0] << " " << rk1[1] << " " << rk1[2] << std::endl;
        std::cout << "rk2: " << rk2[0] << " " << rk2[1] << " " << rk2[2] << std::endl;
        std::cout << "rk3: " << rk3[0] << " " << rk3[1] << " " << rk3[2] << std::endl;
        std::cout << "rk4: " << rk4[0] << " " << rk4[1] << " " << rk4[2] << std::endl;
        std::cout << "result: " << result[0] << " " << result[1] << " " << result[2] << std::endl;
    }
    return result;
}



std::vector<std::array<double, 3>> trajectory_3d_parallel(double *X_original, const std::array<double, 3>& initial_x, const double time_step, const int max_length, const int DW, const int DH, const int DD, const std::unordered_map<size_t, critical_point_t_3d>& critical_points, ftk::ndarray<float>& data,std::vector<std::set<size_t>>& thread_lossless_index,int thread_id) {
  std::vector<std::array<double, 3>> result;
  int flag = 0; // 1 means found, -1 means out of bound, 0 means reach max length
  int length = 0;
  result.push_back({X_original[0], X_original[1], X_original[2]}); // add original true position
  length++;
  //size_t original_offset = get_cell_offset_3d(X_original, DW, DH, DD);

  std::array<double, 3> current_x = initial_x;

  // add original and initial_x position's offset
  auto ori_offset = get_four_offsets(X_original, DW, DH, DD);
  //thread_lossless_index[thread_id].insert(ori_offset.begin(), ori_offset.end());

  if (!inside_domain(current_x, DH, DW, DD)) {
      return result;
  }
  
  result.push_back(current_x); // add initial position(seed)
  length++;
  auto ini_offset = get_four_offsets(current_x, DW, DH, DD);
  thread_lossless_index[thread_id].insert(ini_offset.begin(), ini_offset.end());
  auto original_offset = get_cell_offset_3d(current_x.data(), DW, DH, DD);
  while (result.size() < max_length){
    double current_v[3] = {0};
    std::array<double, 3> RK4result = newRK4_3d_parallel(current_x.data(), current_v, data, time_step, DW, DH, DD, thread_lossless_index, thread_id);
    size_t current_offset = get_cell_offset_3d(RK4result.data(), DW, DH, DD);

    if (!inside_domain(current_x, DH, DW, DD)) {
      result.push_back({current_x[0], current_x[1], current_x[2]});
      return result;
    }

    if (current_offset != original_offset){
      auto it = critical_points.find(current_offset);
      if (it != critical_points.end()){
        auto cp = it->second;
        double error = 1e-4;
        if ((cp.type < 3 || cp.type > 6) && (cp.type != 0) && fabs(RK4result[0] - cp.x[0]) < error && fabs(RK4result[1] - cp.x[1]) < error && fabs(RK4result[2] - cp.x[2]) < error){
          result.push_back({RK4result[0], RK4result[1], RK4result[2]});
          std::array<double, 3> true_cp = {cp.x[0], cp.x[1], cp.x[2]};
          result.push_back(true_cp);
          auto final_offset_rk = get_four_offsets(RK4result.data(), DW, DH, DD);
          // auto final_offset_cp = get_four_offsets(cp.x, DW, DH, DD);
          //printf("reaching cp: %f, %f, %f\n", cp.x[0], cp.x[1], cp.x[2]);
          thread_lossless_index[thread_id].insert(final_offset_rk.begin(), final_offset_rk.end());
          // thread_lossless_index[thread_id].insert(final_offset_cp.begin(), final_offset_cp.end());
          return result;
        }
      }
    }
    current_x = RK4result;
    result.push_back(current_x);
  }
  return result;
}

std::vector<std::array<double, 3>> trajectory_3d(double *X_original, const std::array<double, 3>& initial_x, const double time_step, const int max_length, const int DW, const int DH, const int DD, const std::unordered_map<size_t, critical_point_t_3d>& critical_points, ftk::ndarray<float>& data, std::set<size_t>& lossless_index) {
    std::vector<std::array<double, 3>> result;
    result.reserve(max_length);
    int flag = 0; // 1 means found, -1 means out of bound, 0 means reach max length
    int length = 0;
    result.push_back({X_original[0], X_original[1], X_original[2]}); // add original true position
    length++;
    //int original_offset = get_cell_offset_3d(X_original, DW, DH, DD);

    std::array<double, 3> current_x = initial_x;

    // add original and initial_x position's offset
    auto ori_offset = get_four_offsets(X_original, DW, DH, DD);
    // lossless_index.insert(ori_offset.begin(), ori_offset.end());

    if (!inside_domain(current_x, DH, DW, DD)) { //seed out of bound
        return result;
    }
    result.push_back(current_x); // add initial position(seed)
    length++;
    auto ini_offset = get_four_offsets(current_x, DW, DH, DD);
    lossless_index.insert(ini_offset.begin(), ini_offset.end());
    auto original_offset = get_cell_offset_3d(current_x.data(), DW, DH, DD);
    while(result.size() < max_length){
      double current_v[3] = {0};
      std::array<double, 3> RK4result = newRK4_3d(current_x.data(), current_v, data, time_step, DW, DH, DD, lossless_index);
      size_t current_offset = get_cell_offset_3d(RK4result.data(), DW, DH, DD);

      if (!inside_domain(RK4result, DH, DW, DD)) {
        result.push_back(current_x);
        return result;
      }

      if (current_offset != original_offset){
        //move to a new cell
        auto it = critical_points.find(current_offset);
        if (it != critical_points.end()){
          auto cp = it->second;
          double error = 1e-4;
          if ((cp.type < 3 || cp.type > 6) && fabs(RK4result[0] - cp.x[0]) < error && fabs(RK4result[1] - cp.x[1]) < error && fabs(RK4result[2] - cp.x[2]) < error){
            result.push_back(RK4result);
            result.push_back({cp.x[0],cp.x[1],cp.x[2]}); //add critical point
            auto final_offset_rk = get_four_offsets(RK4result.data(), DW, DH, DD);
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

void final_check(float *U, float *V, float *W, int r1, int r2, int r3, double eb, int obj,traj_config t_config,int num_thread,int total_num_thread, std::set<size_t> vertex_need_to_lossless, thresholds th,result_struct results,std::string eb_type, std::string file_out_dir=""){
  // 统计lossless占比
  std::vector<float> lossless_data;
  for (auto o:vertex_need_to_lossless){
    lossless_data.push_back(U[o]);
    lossless_data.push_back(V[o]);
    lossless_data.push_back(W[o]);
  }
  printf("lossless_data size: %ld\n", lossless_data.size());
  size_t lossless_value_size = vertex_need_to_lossless.size() * 3 * sizeof(float);
  unsigned char * lossless_value = NULL;
  size_t lossless_value_size_after_zstd = sz_lossless_compress(ZSTD_COMPRESSOR, 3, (unsigned char *)lossless_data.data(), lossless_value_size, &lossless_value);
  printf("lossless_value using zstd size: %zu\n", lossless_value_size_after_zstd);
  free(lossless_value);

  unsigned char * final_result = NULL;
  size_t final_result_size = 0;
  auto tpsz_comp_start = std::chrono::high_resolution_clock::now();
  if (eb_type == "rel"){
    if (CPSZ_OMP_FLAG == 0){
      final_result = sz_compress_cp_preserve_3d_record_vertex(U, V, W, r1, r2, r3, final_result_size, false, eb, vertex_need_to_lossless);
    }
    else{
      //use omp version
      float * dec_U_inplace = NULL;
      float * dec_V_inplace = NULL;
      float * dec_W_inplace = NULL;
      final_result = omp_sz_compress_cp_preserve_3d_record_vertex(U, V, W, r1, r2, r3, final_result_size, eb, vertex_need_to_lossless, num_thread, dec_U_inplace, dec_V_inplace, dec_W_inplace);
      free(dec_U_inplace);
      free(dec_V_inplace);
      free(dec_W_inplace);
    }

  }
  else if (eb_type == "abs"){
    if (CPSZ_OMP_FLAG == 0){
      final_result = sz_compress_cp_preserve_3d_online_abs_record_vertex(U, V, W, r1, r2, r3, final_result_size, eb, vertex_need_to_lossless);
    }
    else{
      //use omp version
      std::vector<bool> cp_exist = {};
      float * dec_U_inplace = NULL;
      float * dec_V_inplace = NULL;
      float * dec_W_inplace = NULL;
      final_result = omp_sz_compress_cp_preserve_3d_online_abs_record_vertex(U, V, W, r1, r2, r3, final_result_size, eb, vertex_need_to_lossless, num_thread, dec_U_inplace, dec_V_inplace, dec_W_inplace,cp_exist);
      free(dec_U_inplace);
      free(dec_V_inplace);
      free(dec_W_inplace);
    }
  }
  auto tpsz_comp_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> tpsz_comp_time = tpsz_comp_end - tpsz_comp_start;
  unsigned char * result_after_zstd = NULL;
  size_t result_after_zstd_size = sz_lossless_compress(ZSTD_COMPRESSOR, 3, final_result, final_result_size, &result_after_zstd);
  //write to file
  std::string file_name = "/home/mxi235/data/parallel_result/output_" + eb_type + std::to_string(total_num_thread) + "_threads";
  if (SAVE_OUTPUT_FILE == 1){
    //file_out_dir + num_threads+ .tpsz
    posix_write((file_name + ".tpsz").c_str(), result_after_zstd, result_after_zstd_size);
    free(result_after_zstd);
    result_after_zstd = (unsigned char *)malloc(result_after_zstd_size);
    //read from file
    // unsigned char * result_after_zstd = NULL;
    // size_t result_after_zstd_size = 0;
    auto read_time_start = std::chrono::high_resolution_clock::now();
    posix_read<unsigned char>((file_name + ".tpsz").c_str(), result_after_zstd, result_after_zstd_size);
    auto read_time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> read_time = read_time_end - read_time_start;
    printf("read time = %f\n", read_time.count());
  }

  free(final_result);
  auto tpsz_decomp_start = std::chrono::high_resolution_clock::now();
  size_t zstd_decompressed_size = sz_lossless_decompress(ZSTD_COMPRESSOR, result_after_zstd, result_after_zstd_size, &final_result, final_result_size);
  float * final_dec_U = NULL;
  float * final_dec_V = NULL;
  float * final_dec_W = NULL;
  if (eb_type == "rel"){
    if(CPSZ_OMP_FLAG == 0){
      sz_decompress_cp_preserve_3d_record_vertex<float>(final_result, r1, r2, r3, final_dec_U, final_dec_V, final_dec_W);
    }
    else{
      //use omp version
      omp_sz_decompress_cp_preserve_3d_record_vertex<float>(final_result, r1, r2, r3,final_dec_U, final_dec_V, final_dec_W);
    }
  }
  else if (eb_type == "abs"){
    if (CPSZ_OMP_FLAG == 0){
      sz_decompress_cp_preserve_3d_online_abs_record_vertex<float>(final_result, r1, r2, r3, final_dec_U, final_dec_V, final_dec_W);
    }
    else{
      //use omp version
      omp_sz_decompress_cp_preserve_3d_online_abs_record_vertex<float>(final_result, r1, r2, r3,final_dec_U, final_dec_V, final_dec_W);
    }
  }
  auto tpsz_decomp_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> tpsz_decomp_time = tpsz_decomp_end - tpsz_decomp_start;
  printf("verifying...\n");
  double nrmse_u, nrmse_v, nrmse_w,psnr_overall,psnr_cpsz_overall;
  verify(U, final_dec_U, r1*r2*r3, nrmse_u);
  verify(V, final_dec_V, r1*r2*r3, nrmse_v);
  verify(W, final_dec_W, r1*r2*r3, nrmse_w);
  psnr_cpsz_overall = 20 * log10(sqrt(3) / sqrt(nrmse_u * nrmse_u + nrmse_v * nrmse_v + nrmse_w * nrmse_w));
  //printf("nrmse_u=%f, nrmse_v=%f, nrmse_w=%f, psnr_cpsz_overall=%f\n", nrmse_u, nrmse_v, nrmse_w, psnr_cpsz_overall);
  printf("====================================\n");
  printf("orginal trajs detail: outside: %d, max_iter: %d, reach cp %d\n",results.origin_traj_detail_[0],results.origin_traj_detail_[1],results.origin_traj_detail_[2]);
  printf("pre-compute cp time :%f\n",results.pre_compute_cp_time_);
  printf("BEGIN Compression ratio = %f\n", results.begin_cr_);
  printf("PSNR_overall_cpsz = %f\n", results.psnr_cpsz_overall_);
  printf("FINAL Compression ratio = %f\n", (3*r1*r2*r3*sizeof(float)) * 1.0/result_after_zstd_size);
  printf("percentage of lossless data size: %f\n", lossless_value_size_after_zstd * 1.0 / result_after_zstd_size);
  printf("PSNR_overall = %f\n", psnr_cpsz_overall);
  printf("comp time = %f\n", results.cpsz_comp_time_);
  printf("decomp time = %f\n", results.cpsz_decomp_time_);
  printf("algorithm time = %f\n", results.elapsed_alg_time_);
  printf("tpsz comp time = %f\n", tpsz_comp_time.count());
  printf("tpsz total time = %f\n", results.elapsed_alg_time_ + tpsz_comp_time.count() + results.pre_compute_cp_time_);
  printf("tpsz decompress time = %f\n", tpsz_decomp_time.count());
  printf("rounds = %d\n", results.rounds_);
  for (int i = 0; i < results.trajID_need_fix_next_detail_vec_.size(); i++){
    printf("trajID_need_fix_next: %d, wrong outside: %d, wrong max_iter: %d, wrong cp: %d\n", results.trajID_need_fix_next_vec_[i], results.trajID_need_fix_next_detail_vec_[i][0], results.trajID_need_fix_next_detail_vec_[i][1], results.trajID_need_fix_next_detail_vec_[i][2]);
  }
  printf("====================================\n");
  //检查vertex_need_to_lossless对应的点是否一致
  for(auto p:vertex_need_to_lossless){
    if(U[p] != final_dec_U[p] || V[p] != final_dec_V[p] || W[p] != final_dec_W[p]){
      printf("vertex is diff: index=%d, ori_u=%f, dec_u=%f, ori_v=%f, dec_v=%f, ori_w=%f, dec_w=%f\n", p, U[p], final_dec_U[p], V[p], final_dec_V[p], W[p], final_dec_W[p]);
      exit(0);
    }
  }
  //检查trajectories是否满足要求
  ftk::ndarray<float> grad_final;
  grad_final.reshape({3, static_cast<unsigned long>(r3),static_cast<unsigned long>(r2), static_cast<unsigned long>(r1)});
  refill_gradient_3d(0, r1, r2, r3, final_dec_U, grad_final);
  refill_gradient_3d(1, r1, r2, r3, final_dec_V, grad_final);
  refill_gradient_3d(2, r1, r2, r3, final_dec_W, grad_final);

  ftk::ndarray<float> grad_ori;
  grad_ori.reshape({3, static_cast<unsigned long>(r3),static_cast<unsigned long>(r2), static_cast<unsigned long>(r1)});
  refill_gradient_3d(0, r1, r2, r3, U, grad_ori);
  refill_gradient_3d(1, r1, r2, r3, V, grad_ori);
  refill_gradient_3d(2, r1, r2, r3, W, grad_ori);

  omp_set_num_threads(num_thread);
  //auto critical_points_final = compute_critical_points(final_dec_U, final_dec_V, final_dec_W, r1, r2, r3);
  auto critical_points_final = omp_compute_critical_points(final_dec_U, final_dec_V, final_dec_W, r1, r2, r3);
  auto critical_points_ori = omp_compute_critical_points(U, V, W, r1, r2, r3);
  if (critical_points_final.size() != critical_points_ori.size()){
    printf("critical points size not equal!\n");
    exit(0);
  }
  for (auto p:critical_points_ori){
    //find the same point in critical_points_out
    auto it = critical_points_ori.find(p.first);
    if (it != critical_points_ori.end()){
      auto cp_ori = it->second;
      auto cp_final = p.second;
      if (cp_ori.type != cp_final.type || cp_ori.x[0] != cp_final.x[0] || cp_ori.x[1] != cp_final.x[1] || cp_ori.x[2] != cp_final.x[2]){
        printf("critical points not same pos or type!\n");
        exit(0);
      }
    }
    else{
      printf("critical points not equal!\n");
      exit(0);
    }

  }
  std::vector<std::vector<std::array<double, 3>>> trajs_final;
  std::vector<int> length_index_final;
  std::vector<int> keys;
  
  std::vector<std::set<size_t>> thread_lossless_index(num_thread);
  for (const auto &p : critical_points_final) {
      if (p.second.type >= 3 && p.second.type <= 6) keys.push_back(p.first); //如果是saddle点，就加入到keys中
  }

  std::vector<std::vector<std::array<double, 3>>> final_check_ori(keys.size() * 6);
  std::vector<std::vector<std::array<double, 3>>> final_check_dec(keys.size() * 6);

  bool terminate = false;

  #pragma omp parallel for
  for (size_t j = 0; j < keys.size(); ++j) {
      int key = keys[j];
      // printf("current key: %d,current thread: %d\n",key,omp_get_thread_num());
      auto &cp = critical_points_final[key];
      if (cp.type >=3 && cp.type <= 6){ //only for saddle points
          int thread_id = omp_get_thread_num();
          //printf("current thread: %d, current saddle: %d\n",thread_id,key);
          auto eigvec = cp.eig_vec;
          auto eigval = cp.eigvalues;
          auto pt = cp.x;
          //create 6x4 array of array
          std::array<std::array<double, 4>, 6> directions; //6 directions, first is direction(1 or -1), next 3 are seed point
          // if eigvalue is positive, then direction is 1, otherwise -1
          for (int i = 0; i < 3; i++){
              if (eigval[i] > 0){
                  directions[i][0] = 1;
                  directions[i][1] = t_config.eps * eigvec[i][0] + pt[0];
                  directions[i][2] = t_config.eps * eigvec[i][1] + pt[1];
                  directions[i][3] = t_config.eps * eigvec[i][2] + pt[2];
                  directions[i+3][0] = 1;
                  directions[i+3][1] = -1 * t_config.eps * eigvec[i][0] + pt[0];
                  directions[i+3][2] = -1 * t_config.eps * eigvec[i][1] + pt[1];
                  directions[i+3][3] = -1 * t_config.eps* eigvec[i][2] + pt[2];
              }
              else{
                  directions[i][0] = -1;
                  directions[i][1] = t_config.eps * eigvec[i][0] + pt[0];
                  directions[i][2] = t_config.eps * eigvec[i][1] + pt[1];
                  directions[i][3] = t_config.eps * eigvec[i][2] + pt[2];
                  directions[i+3][0] = -1;
                  directions[i+3][1] = -1 * t_config.eps * eigvec[i][0] + pt[0];
                  directions[i+3][2] = -1 * t_config.eps * eigvec[i][1] + pt[1];
                  directions[i+3][3] = -1 * t_config.eps * eigvec[i][2] + pt[2];
              }
          }          
          for (int k = 0; k < 6; k++){
              //printf("direction %d: \n",i);
              std::array<double, 3> seed = {directions[k][1], directions[k][2], directions[k][3]};
              auto direction = directions[k];  
              //printf("current trajID: %d\n",trajID);
              std::vector<std::array<double, 3>> result_return_ori = trajectory_3d_parallel(pt, seed, t_config.h * directions[k][0], t_config.max_length, r3,r2,r1, critical_points_final, grad_ori,thread_lossless_index,thread_id);
              std::vector<std::array<double, 3>> result_return_dec = trajectory_3d_parallel(pt, seed, t_config.h * directions[k][0], t_config.max_length, r3,r2,r1, critical_points_final, grad_final,thread_lossless_index,thread_id);
              final_check_ori[j*6+k] = result_return_ori;
              final_check_dec[j*6+k] = result_return_dec;
              if (LastTwoPointsAreEqual(result_return_ori)){
                if (euclideanDistance(result_return_ori.back(),result_return_dec.back()) > th.threshold_out){
                  printf("some trajectories not fixed!(case 0-0)\n");
                  printf("ori first point: %f, %f, %f ,dec first point: %f, %f, %f\n",result_return_ori[0][0],result_return_ori[0][1],result_return_ori[0][2],result_return_dec[0][0],result_return_dec[0][1],result_return_dec[0][2]);
                  printf("ori last-1 point: %f, %f, %f ,dec last-1 point: %f, %f, %f\n",result_return_ori[result_return_ori.size()-2][0],result_return_ori[result_return_ori.size()-2][1],result_return_ori[result_return_ori.size()-2][2],result_return_dec[result_return_dec.size()-2][0],result_return_dec[result_return_dec.size()-2][1],result_return_dec[result_return_dec.size()-2][2]);
                  printf("ori last point: %f, %f, %f ,dec last point: %f, %f, %f\n",result_return_ori.back()[0],result_return_ori.back()[1],result_return_ori.back()[2],result_return_dec.back()[0],result_return_dec.back()[1],result_return_dec.back()[2]);
                  printf("ori length: %d, dec length: %d\n",result_return_ori.size(),result_return_dec.size());
                  terminate = true;
                }
              }
              else if (result_return_ori.size() == t_config.max_length){
                if (euclideanDistance(result_return_ori.back(),result_return_dec.back()) >th.threshold_max){
                  printf("some trajectories not fixed!(case 0-1),trajID = %d\n",j*6+k);
                  printf("ori first point: %f, %f, %f ,dec first point: %f, %f, %f\n",result_return_ori[0][0],result_return_ori[0][1],result_return_ori[0][2],result_return_dec[0][0],result_return_dec[0][1],result_return_dec[0][2]);
                  printf("ori last-1 point: %f, %f, %f ,dec last-1 point: %f, %f, %f\n",result_return_ori[result_return_ori.size()-2][0],result_return_ori[result_return_ori.size()-2][1],result_return_ori[result_return_ori.size()-2][2],result_return_dec[result_return_dec.size()-2][0],result_return_dec[result_return_dec.size()-2][1],result_return_dec[result_return_dec.size()-2][2]);
                  printf("ori last point: %f, %f, %f ,dec last point: %f, %f, %f\n",result_return_ori.back()[0],result_return_ori.back()[1],result_return_ori.back()[2],result_return_dec.back()[0],result_return_dec.back()[1],result_return_dec.back()[2]);
                  printf("ori length: %d, dec length: %d\n",result_return_ori.size(),result_return_dec.size());
                  terminate = true;
                }
              }
              else{
                    //inside and not reach max, ie found cp
                    if (get_cell_offset_3d(result_return_ori.back().data(), r3,r2,r1) != get_cell_offset_3d(result_return_dec.back().data(), r3,r2,r1)){
                      printf("some trajectories not fixed!(case 0-2)\n");
                      printf("ori first point: %f, %f, %f ,dec first point: %f, %f, %f\n",result_return_ori[0][0],result_return_ori[0][1],result_return_ori[0][2],result_return_dec[0][0],result_return_dec[0][1],result_return_dec[0][2]);
                      printf("ori last-1 point: %f, %f, %f ,dec last-1 point: %f, %f, %f\n",result_return_ori[result_return_ori.size()-2][0],result_return_ori[result_return_ori.size()-2][1],result_return_ori[result_return_ori.size()-2][2],result_return_dec[result_return_dec.size()-2][0],result_return_dec[result_return_dec.size()-2][1],result_return_dec[result_return_dec.size()-2][2]);
                      printf("ori last point: %f, %f, %f ,dec last point: %f, %f, %f\n",result_return_ori.back()[0],result_return_ori.back()[1],result_return_ori.back()[2],result_return_dec.back()[0],result_return_dec.back()[1],result_return_dec.back()[2]);
                      printf("ori length: %d, dec length: %d\n",result_return_ori.size(),result_return_dec.size());
                      terminate = true;
                    }
                  }
          }     
      }
    }
  if(terminate){
    printf("some trajectories not fixed!\n");
  }
  printf("all passed!\n");

  //计算frechet distance
  int numTrajectories = final_check_ori.size();
  vector<double> frechetDistances(numTrajectories, -1);
  auto frechetDis_time_start = std::chrono::high_resolution_clock::now();
  double max_distance = -1.0;
  int max_index = -1;
  #pragma omp parallel for
  for (int i = 0; i < numTrajectories; i++){
    auto traj1 = final_check_ori[i];
    auto traj2 = final_check_dec[i];
    frechetDistances[i] = frechetDistance(traj1,traj2);
    #pragma omp critical
    {
      if(frechetDistances[i] > max_distance){
        max_distance = frechetDistances[i];
        max_index = i;
      }
    }
  }
  auto frechetDis_time_end = std::chrono::high_resolution_clock::now();
  //calculate time in second
  std::chrono::duration<double> frechetDis_time = frechetDis_time_end - frechetDis_time_start;
  printf("frechet distance time: %f\n",frechetDis_time.count());
  //print max distance ans start point, end point
  printf("max_distance index: %d\n",max_index);
  printf("max_distance start point ori: %f, %f, %f\n",final_check_ori[max_index][0][0],final_check_ori[max_index][0][1],final_check_ori[max_index][0][2]);
  printf("max_distance start point dec: %f, %f, %f\n",final_check_dec[max_index][0][0],final_check_dec[max_index][0][1],final_check_dec[max_index][0][2]);
  printf("max_distance end point ori: %f, %f, %f\n",final_check_ori[max_index].back()[0],final_check_ori[max_index].back()[1],final_check_ori[max_index].back()[2]);
  printf("max_distance end point dec: %f, %f, %f\n",final_check_dec[max_index].back()[0],final_check_dec[max_index].back()[1],final_check_dec[max_index].back()[2]);
  printf("dec length: %d, ori length: %d\n",final_check_dec[max_index].size(),final_check_ori[max_index].size());
  printf("max_distance: %f\n",max_distance);
  std::vector<std::vector<std::array<double, 3>>> max_distance_single_trajs(2);
  max_distance_single_trajs[0] = final_check_ori[max_index];
  max_distance_single_trajs[1] = final_check_dec[max_index];

  //计算统计量
  double minVal, maxVal, medianVal, meanVal, stdevVal;
  calculateStatistics(frechetDistances, minVal, maxVal, medianVal, meanVal, stdevVal);
  printf("Statistics data===============\n");
  printf("min: %f\n", minVal);
  printf("max: %f\n", maxVal);
  printf("median: %f\n", medianVal);
  printf("mean: %f\n", meanVal);
  printf("stdev: %f\n", stdevVal);
  printf("Statistics data===============\n");
  free(final_result);
  free(result_after_zstd);
  free(final_dec_U);
  free(final_dec_V);
  free(final_dec_W);
  if(file_out_dir != ""){
    save_trajs_to_binary_3d(final_check_ori, file_out_dir + "ori_traj_3d.bin");
    save_trajs_to_binary_3d(final_check_dec, file_out_dir + "dec_traj_3d.bin");
    save_trajs_to_binary_3d(max_distance_single_trajs, file_out_dir + "max_distance_single_traj_3d.bin");
  }
  // 
  //以下代码用来统计omp-cpsz的解压时间,用不同的线程数
  //run with n_thread = 1,8,27,64,125
  // std::vector<int> thread_nums = {125,64,27,8,1};
  // for (int i = 0; i < thread_nums.size(); i++){
  //   int total_thread = thread_nums[i];
  //     float * test_u_dec = NULL;
  //     float * test_v_dec = NULL;
  //     float * test_w_dec = NULL;
  //     printf("*************************\n");
  //     printf("begin omp-cpsz decompression with record vertices using #thread = %d\n",total_thread);
  //     float * dec_U_inplace = NULL;
  //     float * dec_V_inplace = NULL;
  //     float * dec_W_inplace = NULL;
  //     if (eb_type == "rel"){
  //       if (total_thread == 1){
  //       final_result = sz_compress_cp_preserve_3d_record_vertex(U,V,W, r1, r2, r3, final_result_size,false, eb,vertex_need_to_lossless);
  //       }
  //       else{
  //       final_result = omp_sz_compress_cp_preserve_3d_record_vertex(U, V, W, r1, r2, r3, final_result_size, eb, vertex_need_to_lossless, total_thread, dec_U_inplace, dec_V_inplace, dec_W_inplace);
  //       free(dec_U_inplace);
  //       free(dec_V_inplace);
  //       free(dec_W_inplace);
  //       }

  //     }
  //     else if (eb_type == "abs"){
  //       std::vector<bool> cp_exist = {};
  //       if (total_thread == 1){
  //         final_result = sz_compress_cp_preserve_3d_online_abs_record_vertex(U,V,W, r1, r2, r3, final_result_size, eb,vertex_need_to_lossless);
  //       }
  //       else{
  //         final_result = omp_sz_compress_cp_preserve_3d_online_abs_record_vertex(U,V,W, r1, r2, r3, final_result_size, eb,vertex_need_to_lossless, total_thread, dec_U_inplace, dec_V_inplace, dec_W_inplace,cp_exist);
  //         free(dec_U_inplace);
  //         free(dec_V_inplace);
  //         free(dec_W_inplace);
  //       }

  //     }
      
  //     result_after_zstd_size = sz_lossless_compress(ZSTD_COMPRESSOR, 3, final_result, final_result_size, &result_after_zstd);
  //     free(final_result);
  //     auto omp_with_vertices_decomp_start = std::chrono::high_resolution_clock::now();
  //     zstd_decompressed_size = sz_lossless_decompress(ZSTD_COMPRESSOR, result_after_zstd, result_after_zstd_size, &final_result, final_result_size);
  //     if (eb_type == "rel"){
  //       if (total_thread == 1){
  //         sz_decompress_cp_preserve_3d_record_vertex<float>(final_result, r1, r2, r3,test_u_dec, test_v_dec, test_w_dec);
  //       }
  //       else{
  //         omp_sz_decompress_cp_preserve_3d_record_vertex<float>(final_result, r1, r2, r3,test_u_dec, test_v_dec, test_w_dec);
  //       }

  //     }
  //     else if (eb_type == "abs"){
  //       if (total_thread == 1){
  //         sz_decompress_cp_preserve_3d_online_abs_record_vertex<float>(final_result, r1, r2, r3,test_u_dec, test_v_dec, test_w_dec);
  //       }
  //       else{
  //         omp_sz_decompress_cp_preserve_3d_online_abs_record_vertex<float>(final_result, r1, r2, r3,test_u_dec, test_v_dec, test_w_dec);
  //       }
  //     }
  //     auto omp_with_vertices_decomp_end = std::chrono::high_resolution_clock::now();
  //     printf("*************************\n");
  //     printf("omp tpsz with vertices decompression time: %f, using %d threads\n",std::chrono::duration<double>(omp_with_vertices_decomp_end - omp_with_vertices_decomp_start).count(),total_thread);
  //     printf("*************************\n");
  //     free(final_result);
  //     free(result_after_zstd);
  //     free(test_u_dec);
  //     free(test_v_dec);
  //     free(test_w_dec);
  // }


}
int main(int argc, char ** argv){
    //bool write_flag = true;
    // 计时用
    std::vector<double> compare_time_vec;
    std::vector<double> index_time_vec;
    std::vector<double> re_cal_trajs_time_vec;
    std::vector<int> trajID_need_fix_next_vec;
    std::vector<std::array<int,3>> trajID_need_fix_next_detail_vec; //0:outside, 1.reach max iter, 2.find cp
    std::array<int,3> origin_traj_detail;
    std::set<size_t> final_vertex_need_to_lossless; //最终需要lossless的点的index
    bool stop = false; //算法停止flag
    std::vector<int> fixed_cpsz_trajID;
    size_t num_elements = 0;
    float * U = readfile<float>(argv[1], num_elements);
    float * V = readfile<float>(argv[2], num_elements);
    float * W = readfile<float>(argv[3], num_elements);
    int r1 = atoi(argv[4]);
    int r2 = atoi(argv[5]);
    int r3 = atoi(argv[6]);
    double h = atof(argv[7]);
    double eps = atof(argv[8]);
    int max_length = atoi(argv[9]);
    double max_eb = atof(argv[10]);
    // int obj = atoi(argv[11]);
    std::string eb_type = argv[11];
    int obj = 0;
    int total_thread = atoi(argv[12]);
    int thread_num_compression = total_thread;
    int thread_num_fix = total_thread;

    double threshold = atof(argv[13]);
    double threshold_outside = atof(argv[14]);
    double threshold_max_iter = atof(argv[15]);
    double next_index_coeff;
    if (argc == 17){
      next_index_coeff = atof(argv[16]);
    }
    else{
      next_index_coeff = 1.0;
    }
    printf("next_index_coeff: %f\n",next_index_coeff);

    // these two flags are used to control whether use the saved compressed data,to save computation time
    int readout_flag = 0;
    int writeout_flag = 1;

    // if(SPEED_TEST_FLAG == 1){
    //   if (total_thread >=125){
    //     thread_num_compression = 125;
    //     thread_num_fix = total_thread;
    //   }
    //   else if (total_thread >= 64){
    //     thread_num_compression = 64;
    //     thread_num_fix = total_thread;
    //   }
    //   else if (total_thread >= 27){
    //     thread_num_compression = 27;
    //     thread_num_fix = total_thread;
    //   }
    // }

    thread_num_compression = total_thread;
    thread_num_fix = total_thread;
    
    std::chrono::duration<double> cpsz_comp_duration;
    std::chrono::duration<double> cpsz_decomp_duration;
    
    std::string file_out_dir = "";
    if (argc == 18){
      file_out_dir = argv[17];
    }
    // int obj = 0;
    omp_set_num_threads(total_thread);
    traj_config t_config = {h, eps, max_length};

    float * dec_U = NULL;
    float * dec_V = NULL;
    float * dec_W = NULL;
    // pre-compute critical points
    auto cp_cal_start = std::chrono::high_resolution_clock::now();
    //auto critical_points_0 = compute_critical_points(U, V, W, r1, r2, r3); //r1=DD,r2=DH,r3=DW
    cout << "using omp to compute critical points" << endl;
    auto critical_points_0 = omp_compute_critical_points(U, V, W, r1, r2, r3);
    auto cp_cal_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cp_cal_duration = cp_cal_end - cp_cal_start;
    cout << "critical points #: " << critical_points_0.size() << endl;
    printf("pre-compute cp detail time: %f\n", cp_cal_duration.count());
    
    if (file_out_dir != ""){
    record_criticalpoints(file_out_dir + "critical_points_ori", critical_points_0);
    }

    double begin_cr = 0;

    size_t result_size = 0;
    struct timespec start, end;
    if (readout_flag == 1){
      dec_U = readfile<float>("/home/mxi235/data/temp_data/dec_U.bin", num_elements);
      dec_V = readfile<float>("/home/mxi235/data/temp_data/dec_V.bin", num_elements);
      dec_W = readfile<float>("/home/mxi235/data/temp_data/dec_W.bin", num_elements);
    }
    
    else{
      std::cout << "start Compression\n";
      unsigned char * result;
      auto comp_time_start = std::chrono::high_resolution_clock::now();
      if(eb_type == "rel"){
        if (CPSZ_OMP_FLAG == 0){
          result =  sz_compress_cp_preserve_3d_online_log(U, V, W, r1, r2, r3, result_size, false, max_eb);
        }
        else{
          //use omp version
          std::set<size_t> empty_set;
          float * dec_U_inplace = NULL;
          float * dec_V_inplace = NULL;
          float * dec_W_inplace = NULL;
          result = omp_sz_compress_cp_preserve_3d_record_vertex(U, V, W, r1, r2, r3, result_size, max_eb, empty_set, thread_num_compression, dec_U_inplace, dec_V_inplace, dec_W_inplace);
          free(dec_U_inplace);
          free(dec_V_inplace);
          free(dec_W_inplace);
        }
      }
      else if(eb_type == "abs"){
        if (CPSZ_OMP_FLAG == 0){
          result = sz_compress_cp_preserve_3d_online_abs_record_vertex(U, V, W, r1, r2, r3, result_size, max_eb);
        }
        else{
          //use omp version
          std::set<size_t> empty_set;
          float * dec_U_inplace = NULL;
          float * dec_V_inplace = NULL;
          float * dec_W_inplace = NULL;
          std::vector<bool> cp_exist = {};
          result = omp_sz_compress_cp_preserve_3d_online_abs_record_vertex(U,V,W, r1, r2, r3, result_size, max_eb,empty_set, thread_num_compression, dec_U_inplace, dec_V_inplace, dec_W_inplace,cp_exist);
          double a,b,c;
          verify(U, dec_U_inplace, r1*r2*r3, a);
          verify(V, dec_V_inplace, r1*r2*r3, b);
          verify(W, dec_W_inplace, r1*r2*r3, c);
          printf("####check inplace above\n");
          free(dec_U_inplace);
          free(dec_V_inplace);
          free(dec_W_inplace);
        }
      }
      else{
        printf("not support eb_type\n");
        exit(0);
      }
      unsigned char * result_after_lossless = NULL;
      size_t lossless_outsize = sz_lossless_compress(ZSTD_COMPRESSOR, 3, result, result_size, &result_after_lossless);

      auto comp_time_end = std::chrono::high_resolution_clock::now();
      cpsz_comp_duration = comp_time_end - comp_time_start;
      printf("compression time: %f\n", cpsz_comp_duration.count());

      begin_cr = (3*num_elements*sizeof(float)) * 1.0/lossless_outsize;
      cout << "Compressed size = " << lossless_outsize << ", ratio = " << (3*num_elements*sizeof(float)) * 1.0/lossless_outsize << endl;
      free(result);
      // exit(0);
      auto decomp_time_start = std::chrono::high_resolution_clock::now();
      size_t lossless_output = sz_lossless_decompress(ZSTD_COMPRESSOR, result_after_lossless, lossless_outsize, &result, result_size);
      if (eb_type == "rel"){
        if(CPSZ_OMP_FLAG == 0){
          sz_decompress_cp_preserve_3d_online_log<float>(result, r1, r2, r3, dec_U, dec_V, dec_W);
        }
        else{
          //use omp version
          omp_sz_decompress_cp_preserve_3d_record_vertex<float>(result, r1, r2, r3,dec_U, dec_V, dec_W);
        }
      }
      else if (eb_type == "abs"){
        if (CPSZ_OMP_FLAG == 0){
          sz_decompress_cp_preserve_3d_online_abs_record_vertex<float>(result, r1, r2, r3, dec_U, dec_V, dec_W);
        }
        else{
          //use omp version
          omp_sz_decompress_cp_preserve_3d_online_abs_record_vertex<float>(result, r1, r2, r3,dec_U, dec_V, dec_W);
        }
      }
      
      auto decomp_time_end = std::chrono::high_resolution_clock::now();
      cpsz_decomp_duration = decomp_time_end - decomp_time_start;
      printf("decompression time: %f\n", cpsz_decomp_duration.count());
      free(result_after_lossless);
    }
    // exit(0);
    printf("verifying...\n");
    double nrmse_u, nrmse_v, nrmse_w,psnr_overall,psnr_cpsz_overall;
    verify(U, dec_U, r1*r2*r3, nrmse_u);
    verify(V, dec_V, r1*r2*r3, nrmse_v);
    verify(W, dec_W, r1*r2*r3, nrmse_w);
    psnr_cpsz_overall = 20 * log10(sqrt(3) / sqrt(nrmse_u * nrmse_u + nrmse_v * nrmse_v + nrmse_w * nrmse_w));
    
    

    // if (writeout_flag == 1){
    //   writefile<float>("/home/mxi235/data/temp_data/dec_U.bin", dec_U, num_elements);
    //   writefile<float>("/home/mxi235/data/temp_data/dec_V.bin", dec_V, num_elements);
    //   writefile<float>("/home/mxi235/data/temp_data/dec_W.bin", dec_W, num_elements);
    //   printf("success write out\n");
    // }




    omp_set_num_threads(total_thread);
    //auto critical_points_out = compute_critical_points(dec_U,dec_V,dec_W, r1, r2, r3);
    cout << "using omp to compute critical points" << endl;
    auto critical_points_out = omp_compute_critical_points(dec_U, dec_V, dec_W, r1, r2, r3);
    cout << "critical points dec: " << critical_points_out.size() << endl;

    
    if (critical_points_0.size() != critical_points_out.size()){
    printf("critical_points_ori size: %ld, critical_points_out size: %ld\n", critical_points_0.size(), critical_points_out.size());
    printf("critical points size not equal\n");
    exit(0);
    }
    //check two critical points are the same
    for (auto p:critical_points_0){
      //find the same point in critical_points_out
      auto it = critical_points_out.find(p.first);
      if (it != critical_points_out.end()){
        auto cp_ori = p.second;
        auto cp_dec = it->second;
        if (cp_ori.x[0] != cp_dec.x[0] || cp_ori.x[1] != cp_dec.x[1] || cp_ori.x[2] != cp_dec.x[2] || cp_ori.type != cp_dec.type){
          printf("critical points not same pos or type\n");
          printf("ori: %.10f, %.10f, %.10f, dec: %.10f, %.10f, %.10f\n", cp_ori.x[0], cp_ori.x[1], cp_ori.x[2], cp_dec.x[0], cp_dec.x[1], cp_dec.x[2]);
          //print u,v,w
        }
      }
    }

    //replace the points on surface with original data
    // for (int i = 0; i < r1; i++){
    //   for (int j = 0; j < r2; j++){
    //     for (int k = 0; k < r3; k++){
    //       if (i == 0 || i == r3-1 || j == 0 || j == r2-1 || k == 0 || k == r1-1){
    //         dec_U[k + r3 * j + r3 * r2 * i] = U[k + r3 * j + r3 * r2 * i];
    //         dec_V[k + r3 * j + r3 * r2 * i] = V[k + r3 * j + r3 * r2 * i];
    //         dec_W[k + r3 * j + r3 * r2 * i] = W[k + r3 * j + r3 * r2 * i];
    //       }
    //     }
    //   }
    // }
    // for (int i = 0; i < r1; i++){
    //   for (int j = 0; j < r2; j++){
    //     for (int k = 0; k < r3; k++){
    //       if (i == 1 || i == r3-2 || j == 1 || j == r2-2 || k == 1 || k == r1-2){
    //         dec_U[k + r3 * j + r3 * r2 * i] = U[k + r3 * j + r3 * r2 * i];
    //         dec_V[k + r3 * j + r3 * r2 * i] = V[k + r3 * j + r3 * r2 * i];
    //         dec_W[k + r3 * j + r3 * r2 * i] = W[k + r3 * j + r3 * r2 * i];
    //       }
    //     }
    //   }
    // }

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

    
    //exit(0);

    //*************先计算一次整体的traj_ori 和traj_dec,后续只需增量修改*************
    auto start_alg_time = std::chrono::high_resolution_clock::now();
    //*************计算原始数据的traj_ori*************
    size_t total_saddle_count = 0;
    size_t total_traj_count = 0;
    size_t total_traj_reach_cp = 0;
    std::set<size_t> vertex_ori;
    // std::unordered_map<size_t, std::set<int>> cellID_trajIDs_map_ori;
    auto start1 = std::chrono::high_resolution_clock::now();

    std::vector<int> keys;
    for (const auto &p : critical_points_0) {
        if (p.second.type >= 3 && p.second.type <= 6) keys.push_back(p.first); //如果是saddle点，就加入到keys中
    }
    printf("keys size(# of saddle): %ld\n", keys.size());
    std::vector<double> trajID_direction_vector(keys.size() * 6, 0);
    std::vector<std::vector<std::array<double, 3>>> trajs_ori(keys.size() * 6);//指定长度为saddle的个数*6，因为每个saddle有6个方向
    printf("trajs_ori size: %ld\n", trajs_ori.size());
      // /*这里一定要加上去，不然由于动态扩容会产生额外开销*/
    size_t expected_size = max_length * 1 + 1;
    for (auto& traj : trajs_ori) {
        traj.reserve(expected_size); // 预分配容量
    }
    std::vector<std::set<size_t>> thread_lossless_index(thread_num_fix);
    // for (const auto&p:critical_points_0){
    #pragma omp parallel for reduction(+:total_saddle_count,total_traj_count,total_traj_reach_cp) 
    for (size_t j = 0; j < keys.size(); ++j) {
        int key = keys[j];
        // printf("current key: %d,current thread: %d\n",key,omp_get_thread_num());
        auto &cp = critical_points_0[key];
        if (cp.type >=3 && cp.type <= 6){ //only for saddle points
            total_saddle_count ++;
            int thread_id = omp_get_thread_num();
            //printf("current thread: %d, current saddle: %d\n",thread_id,key);
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
                    directions[i+3][0] = 1;
                    directions[i+3][1] = -1 * eps * eigvec[i][0] + pt[0];
                    directions[i+3][2] = -1 * eps * eigvec[i][1] + pt[1];
                    directions[i+3][3] = -1 * eps* eigvec[i][2] + pt[2];
                }
                else{
                    directions[i][0] = -1;
                    directions[i][1] = eps * eigvec[i][0] + pt[0];
                    directions[i][2] = eps * eigvec[i][1] + pt[1];
                    directions[i][3] = eps * eigvec[i][2] + pt[2];
                    directions[i+3][0] = -1;
                    directions[i+3][1] = -1 * eps * eigvec[i][0] + pt[0];
                    directions[i+3][2] = -1 * eps * eigvec[i][1] + pt[1];
                    directions[i+3][3] = -1 * eps * eigvec[i][2] + pt[2];
                }
            }          
            for (int k = 0; k < 6; k++){
                //printf("direction %d: \n",i);
                std::array<double, 3> seed = {directions[k][1], directions[k][2], directions[k][3]};
                auto direction = directions[k];  
                //printf("current trajID: %d\n",trajID);
                std::vector<std::array<double, 3>> result_return = trajectory_3d_parallel(pt, seed, h * directions[k][0], max_length, r3,r2,r1, critical_points_0, grad_ori,thread_lossless_index,thread_id);
                // printf("threadID: %d, trajID: %d, seed pt: %f %f %f, end pt: %f %f %f\n",omp_get_thread_num(),trajID,direction[1], direction[2], direction[3],traj.back()[0],traj.back()[1],traj.back()[2]);
                trajs_ori[j*6 + k] = result_return;
                trajID_direction_vector[j*6 + k] = directions[k][0];
                total_traj_count ++;
            }     
        }

    }
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> traj_ori_begin_elapsed = end1 - start1;
    cout << "Elapsed time for calculate all ori traj once: " << traj_ori_begin_elapsed.count() << "s" << endl;
    printf("total critical points: %zu\n",critical_points_0.size());
    printf("total saddle points: %zu\n",total_saddle_count);
    printf("total traj: %zu\n",total_traj_count);

    //start1 = std::chrono::high_resolution_clock::now();
    // for (auto traj:trajs_ori){
    //     auto last = traj.back();
    //     auto last_offset = get_cell_offset_3d(last.data(), r3, r2, r1);
    //     auto first = traj[0];
    //     // printf("first: %f %f %f, last: %f %f %f\n",first[0],first[1],first[2],last[0],last[1],last[2]);
    //     // printf("first: %f %f %f\n",first[0],first[1],first[2]);
    //     // printf("last: %f %f %f\n",last[0],last[1],last[2]);
    //     if (traj.size() == max_length){
    //         max_iter ++;
    //     }
    //     else if (last[0] == -1 && last[1] == -1 && last[2] == -1){
    //         outside ++;
    //     }
    //     else{
    //         //check if last point in critical_points_0
    //         for (auto cp:critical_points_0){
    //             auto cp_pt = cp.second.x;
    //             if (last[0] == cp_pt[0] && last[1] == cp_pt[1] && last[2] == cp_pt[2]){
    //               found ++;
    //               break;  
    //             }
    //         }
    //     }
    // }
    // end1 = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double>elapsed = end1 - start1;
    //cout << "Elapsed time for check all traj once: " << elapsed.count() << "s" << endl;
    printf("total traj: %zu\n",trajs_ori.size());
    printf("total critical points: %zu\n",critical_points_0.size());
    printf("total saddle points: %zu\n",total_saddle_count);
    printf("vertex_ori size: %zu\n",vertex_ori.size());


    //*************计算解压缩数据的traj_dec*************
    total_saddle_count = 0,total_traj_count = 0;
    std::set<size_t> vertex_dec;
    // std::unordered_map<size_t, std::set<int>> cellID_trajIDs_map_dec;
    std::vector<int> keys_dec;
    start1 = std::chrono::high_resolution_clock::now();
    for (const auto &p : critical_points_out) {
      if (p.second.type >= 3 && p.second.type <= 6) keys_dec.push_back(p.first);
        // keys_dec.push_back(p.first);
    }
    printf("keys_dec size(# of saddle): %ld\n", keys_dec.size());
    std::vector<double> trajID_direction_vector_dec(keys_dec.size() * 6, 0);//不需要
    std::vector<std::vector<std::array<double, 3>>> trajs_dec(keys_dec.size() * 6);//指定长度为saddle的个数*6，因为每个saddle有6个方向
    printf("trajs_dec size: %ld\n", trajs_dec.size());
    // /*这里一定要加上去，不然由于动态扩容会产生额外开销*/
    expected_size = max_length * 1 + 1;
    for (auto& traj : trajs_dec) {
        traj.reserve(expected_size); // 预分配容量
    }
    thread_lossless_index.clear();
    thread_lossless_index.resize(thread_num_fix);
    #pragma omp parallel for reduction(+:total_saddle_count,total_traj_count,total_traj_reach_cp) 
    for (size_t j = 0; j < keys_dec.size(); ++j) {
        int key = keys_dec[j];
        // printf("current key: %d,current thread: %d\n",key,omp_get_thread_num());
        auto &cp = critical_points_out[key];
        if (cp.type >=3 && cp.type <= 6){
            //auto start_six_traj = std::chrono::high_resolution_clock::now();
            total_saddle_count ++;
            int thread_id = omp_get_thread_num();
            //printf("current thread: %d, current saddle: %d\n",thread_id,key);
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
                    directions[i+3][0] = 1;
                    directions[i+3][1] = -1 * eps * eigvec[i][0] + pt[0];
                    directions[i+3][2] = -1 * eps * eigvec[i][1] + pt[1];
                    directions[i+3][3] = -1 * eps* eigvec[i][2] + pt[2];
                }
                else{
                    directions[i][0] = -1;
                    directions[i][1] = eps * eigvec[i][0] + pt[0];
                    directions[i][2] = eps * eigvec[i][1] + pt[1];
                    directions[i][3] = eps * eigvec[i][2] + pt[2];
                    directions[i+3][0] = -1;
                    directions[i+3][1] = -1 * eps * eigvec[i][0] + pt[0];
                    directions[i+3][2] = -1 * eps * eigvec[i][1] + pt[1];
                    directions[i+3][3] = -1 * eps * eigvec[i][2] + pt[2];
                }
            }          
            for (int k = 0; k < 6; k++){
                //printf("direction %d: \n",i);
                std::array<double, 3> seed = {directions[k][1], directions[k][2], directions[k][3]};
                auto direction = directions[k];  
                //printf("current trajID: %d\n",trajID);
                std::vector<std::array<double, 3>> result_return = trajectory_3d_parallel(pt, seed, h * directions[k][0], max_length, r3,r2,r1, critical_points_out, grad_dec,thread_lossless_index,thread_id);
                // printf("threadID: %d, trajID: %d, seed pt: %f %f %f, end pt: %f %f %f\n",omp_get_thread_num(),trajID,direction[1], direction[2], direction[3],traj.back()[0],traj.back()[1],traj.back()[2]);
                trajs_dec[j*6 + k] = result_return;
                total_traj_count ++;
            }     
        }

    }
    
    end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> traj_dec_begin_elapsed = end1 - start1;
    cout << "Elapsed time for calculate all dec traj once: " << traj_dec_begin_elapsed.count() << "s" << endl;
    printf("total traj dec: %zu\n",trajs_dec.size());
    printf("total critical points: %zu\n",critical_points_out.size());
    printf("total saddle points: %zu\n",total_saddle_count);
    printf("vertex_ori size: %zu\n",vertex_ori.size());
    // for (int i = 0; i < trajs_ori.size(); i++){
    //     auto traj_ori = trajs_ori[i];
    //     auto traj_dec = trajs_dec[i];
    //     auto first_ori = traj_ori[0];
    //     auto first_dec = traj_dec[0];
    //     auto second_ori = traj_ori[1];
    //     auto second_dec = traj_dec[1];
    //     if (first_ori[0] != first_dec[0] || first_ori[1] != first_dec[1] || first_ori[2] != first_dec[2]){
    //         printf("first point not the same\n");
    //         exit(0);
    //     }
    //     if (second_ori[0] != second_dec[0] || second_ori[1] != second_dec[1] || second_ori[2] != second_dec[2]){
    //         printf("second point not the same\n");
    //         exit(0);
    //     }
    // }

    if (file_out_dir != ""){
      save_trajs_to_binary_3d(trajs_dec, file_out_dir + "cpsz_traj_3d.bin");
    }

    //计算frechet distance between trajs_ori and trajs_dec(cpsz)

    vector<double> frechetDistances_cpsz(trajs_ori.size(), -1);
    double max_distance_cpsz = -1.0;
    int max_index_cpsz = -1;
    #pragma omp parallel for
    for (int i = 0; i < trajs_ori.size(); i++){
      auto traj1 = trajs_ori[i];
      auto traj2 = trajs_dec[i];
      frechetDistances_cpsz[i] = frechetDistance(traj1,traj2);
      #pragma omp critical
      {
        if(frechetDistances_cpsz[i] > max_distance_cpsz){
          max_distance_cpsz = frechetDistances_cpsz[i];
          max_index_cpsz = i;
        }
      }
    }
      //计算统计量
    double minVal_cpsz, maxVal_cpsz, medianVal_cpsz, meanVal_cpsz, stdevVal_cpsz;
    calculateStatistics(frechetDistances_cpsz, minVal_cpsz, maxVal_cpsz, medianVal_cpsz, meanVal_cpsz, stdevVal_cpsz);
    printf("Statistics data cpsz===============\n");
    printf("index(same with ori indedx): %d\n", max_index_cpsz);
    printf("min: %f\n", minVal_cpsz);
    printf("max: %f\n", maxVal_cpsz);
    printf("median: %f\n", medianVal_cpsz);
    printf("mean: %f\n", meanVal_cpsz);
    printf("stdev: %f\n", stdevVal_cpsz);
    printf("Statistics data cpsz===============\n");

    if (file_out_dir != ""){
        std::vector<std::vector<std::array<double, 3>>> max_distance_single_trajs_cpsz(2);
        max_distance_single_trajs_cpsz[0] = trajs_ori[max_index_cpsz];
        max_distance_single_trajs_cpsz[1] = trajs_dec[max_index_cpsz];
        save_trajs_to_binary_3d(max_distance_single_trajs_cpsz, file_out_dir + "max_distance_single_traj_3d_cpsz.bin");

    }
    // exit(0);
    


    // int last_two_pt_same = 0;
    // int max_size = 0;
    // int reach_cp = 0;
    // for (auto p:trajs_ori){
    //     //print all trajs that last two points are equal
    //     if (p.size() > 1){
    //         auto last = p.back();
    //         auto last2 = p[p.size()-2];
    //         if (p.size() == max_length){
    //             max_size ++;
    //         }
    //         else if (last[0] == last2[0] && last[1] == last2[1] && last[2] == last2[2]){
    //             //printf("last pt (%f %f %f)\n",last[0],last[1],last[2]);
    //             last_two_pt_same ++;
    //         }
    //         else {
    //             reach_cp ++;
    //         }
    //     }
    //     if (p.size() <=1){
    //       //print all pts in p
    //       printf("====================================\n");
    //       for (auto pt:p){
    //         printf("pt: %f %f %f\n",pt[0],pt[1],pt[2]);
    //       }
    //     }
        
    // }
    // printf("last two pt same: %d\n",last_two_pt_same);
    // printf("max size: %d\n",max_size);
    // printf("reach cp: %d\n",reach_cp);
    // printf("total traj: %d\n",trajs_ori.size());
    // exit(0);

    // 计算哪里有问题（init queue）
    std::set<size_t> trajID_need_fix;
    auto init_queue_start = std::chrono::high_resolution_clock::now();
    int num_outside = 0;
    int num_max_iter = 0;
    int num_find_cp = 0;
    int wrong_num_outside = 0;
    int wrong_num_max_iter = 0;
    int wrong_num_find_cp = 0;
    std::vector<std::set<size_t>> local_trajID_need_fix(thread_num_fix);
    // switch (obj)
    // {
    // case 0:
    #pragma omp parallel for reduction(+:num_outside, num_max_iter, num_find_cp, wrong_num_outside, wrong_num_max_iter, wrong_num_find_cp)
    for(size_t i =0; i< trajs_ori.size(); ++i){
      const auto& t1 = trajs_ori[i];
      const auto& t2 = trajs_dec[i];
      bool cond1 = get_cell_offset_3d(t1.back().data(), r3, r2, r1) == get_cell_offset_3d(t2.back().data(), r3, r2, r1);
      bool cond2 = t1.size() == t_config.max_length;
      //bool f_dist = frechetDistance(t1, t2) >= threshold;

      if (LastTwoPointsAreEqual(t1)){
        num_outside ++;
        //ori inside
        if (!LastTwoPointsAreEqual(t2)){
          //dec outside
          wrong_num_outside ++;
          // trajID_need_fix.insert(i);
          local_trajID_need_fix[omp_get_thread_num()].insert(i);
        }
        else{
          //dec outside
          if ((euclideanDistance(t1.back(), t2.back()) > threshold_outside) && (frechetDistance(t1, t2) >= threshold)){
            wrong_num_outside ++;
            // trajID_need_fix.insert(i);
            local_trajID_need_fix[omp_get_thread_num()].insert(i);
          }
        }
      }
      else if (cond2){
        num_max_iter ++;
        //ori reach max
        if (t2.size() != t_config.max_length){
          //dec not reach max, add
          wrong_num_max_iter ++;
          // trajID_need_fix.insert(i);
          local_trajID_need_fix[omp_get_thread_num()].insert(i);
        }
        else{
          //dec reach max, need to check distance
          if ((euclideanDistance(t1.back(), t2.back()) > threshold_max_iter) && (frechetDistance(t1, t2) >= threshold)){
            wrong_num_max_iter ++;
            // trajID_need_fix.insert(i);
            local_trajID_need_fix[omp_get_thread_num()].insert(i);
          }
        }
      }
      else{
        //reach cp
        num_find_cp ++;
        if(!cond1 || (frechetDistance(t1, t2) >= threshold)){
          wrong_num_find_cp ++;
          // trajID_need_fix.insert(i);
          local_trajID_need_fix[omp_get_thread_num()].insert(i);  
        }
      }
    }
    //break;
    // }

    //汇总local_trajID_need_fix
    for (auto& s:local_trajID_need_fix){
      trajID_need_fix.insert(s.begin(),s.end());
    }


    auto init_queue_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> init_queue_elapsed = init_queue_end - init_queue_start;
    printf("Elapsed time for init queue: %f s\n", init_queue_elapsed.count());
    if (trajID_need_fix.size() == 0){
      printf("no need to fix\n");
      if (file_out_dir != ""){
        save_trajs_to_binary_3d(trajs_ori, file_out_dir + "3d_trajs_ori.bin");
        save_trajs_to_binary_3d(trajs_dec, file_out_dir + "3d_trajs_dec.bin");
      }
      exit(0);
    }
    else{
      printf("need to fix: %zu\n",trajID_need_fix.size());
      bool write_cpsz_wrong_trajs_flag = true;
      if (write_cpsz_wrong_trajs_flag && file_out_dir != ""){
        std::vector<std::vector<std::array<double, 3>>> wrong_trajs_ori;
        std::vector<std::vector<std::array<double, 3>>> wrong_trajs_dec;
        for (auto trajID:trajID_need_fix){
          wrong_trajs_ori.push_back(trajs_ori[trajID]);
          wrong_trajs_dec.push_back(trajs_dec[trajID]);
          fixed_cpsz_trajID.push_back(trajID);
        }
        save_trajs_to_binary_3d(wrong_trajs_ori, file_out_dir + "3d_wrong_trajs_ori.bin");
        save_trajs_to_binary_3d(wrong_trajs_dec, file_out_dir + "3d_wrong_trajs_dec.bin");
      }
      trajID_need_fix_next_vec.push_back(trajID_need_fix.size());
      trajID_need_fix_next_detail_vec.push_back({wrong_num_outside, wrong_num_max_iter, wrong_num_find_cp});
      origin_traj_detail = {num_outside, num_max_iter, num_find_cp};
      printf("original traj has: %d outside, %d max_iter, %d find_cp\n",num_outside, num_max_iter, num_find_cp);
    }

    //*************开始修复轨迹*************
    int current_round = 0;
    do
    {
      printf("begin fix traj,current_round: %d\n", current_round);
      if (current_round >=10){
        printf("current_round >= 10, exit\n");
        exit(0);
      }
      if (file_out_dir != ""){
        std::vector<std::vector<std::array<double, 3>>> wrong_trajs_cpsz_intermidiate;
        for (const auto& trajID:trajID_need_fix){
          wrong_trajs_cpsz_intermidiate.push_back(trajs_dec[trajID]);
        }
        printf("writing intermidiate wrong trajs to file...\n");
        save_trajs_to_binary_3d(wrong_trajs_cpsz_intermidiate, file_out_dir + "state_" + std::to_string(current_round) + "wrong_trajs_cpsz_3d.bin");
      }
      current_round ++;
      std::set<size_t> trajID_need_fix_next;
      //fix trajecotry
      auto index_time_start = std::chrono::high_resolution_clock::now();
      std::vector<size_t> trajID_need_fix_vector(trajID_need_fix.begin(),trajID_need_fix.end()); //set to vector
      printf("current iteration size: %zu\n",trajID_need_fix_vector.size());

      std::vector<std::set<size_t>> local_all_vertex_for_all_diff_traj(thread_num_fix);

      omp_lock_t lock;
      omp_init_lock(&lock);
      #pragma omp parallel for schedule(dynamic)
      for (size_t i=0;i<trajID_need_fix_vector.size(); ++i){
        auto current_trajID = trajID_need_fix_vector[i];
        // for (const auto& trajID:trajID_need_fix){
        //   size_t current_trajID = trajID;
        bool success = false;
        auto t1 = trajs_ori[current_trajID];
        auto t2 = trajs_dec[current_trajID];
        // if(current_trajID == 4128 || current_trajID == 378){
        //   printf("when fix, found ID=4128 or 378\n");
        //   printf("t1 size: %zu, t2 size: %zu\n",t1.size(),t2.size());
        //   printf("t1 first: %f %f %f, t1 last: %f %f %f\n",t1[0][0],t1[0][1],t1[0][2],t1.back()[0],t1.back()[1],t1.back()[2]);
        //   printf("t2 first: %f %f %f, t2 last: %f %f %f\n",t2[0][0],t2[0][1],t2[0][2],t2.back()[0],t2.back()[1],t2.back()[2]);
        // }
        int start_fix_index = 0;
        // int end_fix_index = t1.size() - 1;
        int end_fix_index = 1;
        int thread_id = omp_get_thread_num();

        //find the first different point
        int changed = 0;
        for (size_t j = start_fix_index; j < std::min(t1.size(),t2.size()); ++j){
        //for (size_t j = start_fix_index; j < max_index; ++j){
          auto p1 = t1[j];
          auto p2 = t2[j];
          if (j < t1.size() - 1 && j < t2.size() - 1){
            double dist = euclideanDistance(p1, p2);
            //auto p1_offset = get_cell_offset(p1.data(), DW, DH);
            //auto p2_offset = get_cell_offset(p2.data(), DW, DH);
            if ((dist > threshold)){
              end_fix_index = j;
              //end_fix_index = t1.size() - 1;
              changed = 1;
              break;
            }
          }
        }
        // if (t1.size() == t_config.max_length){
        // end_fix_index = t1.size() / 2; //从中间开始fix
        // }
        // end_fix_index = std::min(end_fix_index, static_cast<int>(t1.size()) - 1);
        end_fix_index = std::min(end_fix_index, static_cast<int>(t1.size())); //t1.size() - 1;

        do
        {
          double direction = trajID_direction_vector[current_trajID];
          std::set<size_t> temp_vertexID; //存储经过的点对应的vertexID
          std::set<size_t> temp_vertexID_check;
          auto temp_trajs_ori = trajectory_3d(t1[0].data(), t1[1], h * direction, end_fix_index, r3, r2, r1, critical_points_0, grad_ori,temp_vertexID);
          for (auto o:temp_vertexID){ //用原始数据更新dec_U,dec_V,dec_W
            dec_U[o] = U[o];
            dec_V[o] = V[o];
            dec_W[o] = W[o];
            int x = o % r3; //o 转化为坐标
            int y = (o / r3) % r2;
            int z = o / (r3 * r2);
            grad_dec(0, x, y, z) = U[o];//更新grad_dec
            grad_dec(1, x, y, z) = V[o];
            grad_dec(2, x, y, z) = W[o];
            local_all_vertex_for_all_diff_traj[thread_id].insert(o);
          }
          auto temp_trajs_check = trajectory_3d(t1[0].data(), t1[1], h * direction, end_fix_index, r3, r2, r1, critical_points_0, grad_dec,temp_vertexID_check);
          if (LastTwoPointsAreEqual(t1)){
            //ori outside
            if (!LastTwoPointsAreEqual(temp_trajs_check)){
              //dec inside
              success = false;
            }
            else{
              //dec outside
              if ((euclideanDistance(t1.back(), temp_trajs_check.back()) < threshold_outside) && (frechetDistance(t1, temp_trajs_check) < threshold)){
                success = true;
              }
            }
          }
          else if (t1.size() == t_config.max_length){
            if ((temp_trajs_check.size() == t_config.max_length)){
              // if ((euclideanDistance(t1.back(), temp_trajs_check.back()) >=threshold_max_iter) || (frechetDistance(t1, temp_trajs_check) >= threshold)){
              //   success = false;
              // }
              // else{
              //   success = true;
              // }
              if ((euclideanDistance(t1.back(), temp_trajs_check.back()) <=threshold_max_iter) && (frechetDistance(t1, temp_trajs_check) <= threshold)){
                success = true;
              }
            }
          }
          else{
            //reach cp
            if ((get_cell_offset_3d(t1.back().data(), r3, r2, r1) == get_cell_offset_3d(temp_trajs_check.back().data(), r3, r2, r1)) && (frechetDistance(t1, temp_trajs_check) < threshold)){
              success = true;
            }
          }
          if (end_fix_index >=t1.size()){
            success = true;
          }
          else{
            end_fix_index = std::min(end_fix_index + static_cast<int>(next_index_coeff*t_config.max_length),static_cast<int>(t1.size()));
          }
        
        }while(success == false);


        // while(!success){
        //   double direction = trajID_direction_vector[current_trajID];
        //   std::set<size_t> temp_vertexID; //存储经过的点对应的vertexID
        //   std::set<size_t> temp_var;
        //   // std::unordered_map<size_t, double> rollback_dec_u;
        //   // std::unordered_map<size_t, double> rollback_dec_v;
        //   // std::unordered_map<size_t, double> rollback_dec_w;
        //   //计算一次rk4直到终点，得到经过的cellID， 然后替换数据
        //   // end_fix_index = std::min(end_fix_index,t_config.max_length); 
        //   auto temp_trajs_ori = trajectory_3d(t1[0].data(), t1[1], h * direction, end_fix_index, r3, r2, r1, critical_points_0, grad_ori,temp_vertexID);
        //   //printf("end_fix_index for temp_trajs_ori: %d\n",end_fix_index);
        //   //此时temp_trajs_ori中存储的是从起点到分岔点经过的vertex
        //   // for (auto o:temp_vertexID){
        //   //   rollback_dec_u[o] = dec_U[o];//存储需要回滚的数据
        //   //   rollback_dec_v[o] = dec_V[o];
        //   //   rollback_dec_w[o] = dec_W[o];
        //   //   local_all_vertex_for_all_diff_traj[thread_id].insert(o);//存储经过的vertex到local_all_vertex_for_all_diff_traj
        //   // }
        //   //这里o越界了
        //   for (auto o:temp_vertexID){ //用原始数据更新dec_U,dec_V,dec_W
        //     dec_U[o] = U[o];
        //     dec_V[o] = V[o];
        //     dec_W[o] = W[o];
        //     int x = o % r3; //o 转化为坐标
        //     int y = (o / r3) % r2;
        //     int z = o / (r3 * r2);
        //     grad_dec(0, x, y, z) = U[o];//更新grad_dec
        //     grad_dec(1, x, y, z) = V[o];
        //     grad_dec(2, x, y, z) = W[o];
        //     local_all_vertex_for_all_diff_traj[thread_id].insert(o);
        //   }
        //   //此时数据更新了，如果此时计算从起点到分岔点的轨迹(使用dec），应该是一样的
        //   //std::vector<int> temp_index_test;
        //   //std::set<size_t> temp_vertexID_test; //存储经过的点对应的vertexID
        //   //auto temp_trajs_test = trajectory_3d(t1[0].data(), t1[1], h * direction, end_fix_index, r3, r2, r1, critical_points_out, grad_dec,temp_index_test,temp_vertexID_test);
        //   //auto temp_trajs_check = trajectory_3d(current_divergence_pos.data(), current_divergence_pos, h * direction, t1.size()-end_fix_index+2, r3, r2, r1, critical_points_out, grad_dec,temp_index_check,temp_var);
        //   auto temp_trajs_check = trajectory_3d(t1[0].data(), t1[1], h * direction, end_fix_index, r3, r2, r1, critical_points_0, grad_dec,temp_var);
        //   if (LastTwoPointsAreEqual(t1)){
        //     //ori outside
        //     if (!LastTwoPointsAreEqual(temp_trajs_check)){
        //       //dec inside
        //       success = false;
        //     }
        //     else{
        //       //dec outside
        //       if ((euclideanDistance(t1.back(), temp_trajs_check.back()) < threshold_outside) && (frechetDistance(t1, temp_trajs_check) < threshold)){
        //         success = true;
        //       }
        //     }
        //   }
        //   else if (t1.size() == t_config.max_length){
        //     if ((temp_trajs_check.size() == t_config.max_length)){
        //       if ((euclideanDistance(t1.back(), temp_trajs_check.back()) >=threshold_max_iter) || (frechetDistance(t1, temp_trajs_check) >= threshold)){
        //         success = false;
        //       }
        //       else{
        //         success = true;
        //       }
        //     }
        //   }
        //   else{
        //     //reach cp
        //     if ((get_cell_offset_3d(t1.back().data(), r3, r2, r1) == get_cell_offset_3d(temp_trajs_check.back().data(), r3, r2, r1)) && (frechetDistance(t1, temp_trajs_check) < threshold)){
        //       success = true;
        //     }
        //   }
        //   if (!success){
        //     //线程争抢可能导致没发fix
        //     //rollback
        //     // for (auto o:temp_vertexID){
        //     //   dec_U[o] = rollback_dec_u[o];
        //     //   dec_V[o] = rollback_dec_v[o];
        //     //   dec_W[o] = rollback_dec_w[o];
        //     //   int x = o % r3; //o 转化为坐标
        //     //   int y = (o / r3) % r2;
        //     //   int z = o / (r3 * r2);
        //     //   grad_dec(0, x, y, z) = dec_U[o];//回退grad_dec
        //     //   grad_dec(1, x, y, z) = dec_V[o];
        //     //   grad_dec(2, x, y, z) = dec_W[o];
        //     //   local_all_vertex_for_all_diff_traj[thread_id].erase(o); //删除local_all_vertex_for_all_diff_traj中的vertex
        //     // }
        //     if (end_fix_index >= static_cast<int>(t1.size())){
        //       printf("t_config.max_length: %d,end_fix_index%d\n",t_config.max_length,end_fix_index);
        //       printf("error: current end_fix_index is %d, current ID: %zu\n",end_fix_index,current_trajID);
        //       printf("ori first: (%f %f %f), temp_trajs_ori first: (%f %f %f), temp_trajs_check first: (%f %f %f)\n",t1[0][0],t1[0][1],t1[0][2],temp_trajs_ori[0][0],temp_trajs_ori[0][1],temp_trajs_ori[0][2],temp_trajs_check[0][0],temp_trajs_check[0][1],temp_trajs_check[0][2]);
        //       printf("ori second: (%f %f %f), temp_trajs_ori second: (%f %f %f), temp_trajs_check second: (%f %f %f)\n",t1[1][0],t1[1][1],t1[1][2],temp_trajs_ori[1][0],temp_trajs_ori[1][1],temp_trajs_ori[1][2],temp_trajs_check[1][0],temp_trajs_check[1][1],temp_trajs_check[1][2]);
        //       printf("ori last-2: (%f %f %f), temp_trajs_ori last-2: (%f %f %f), temp_trajs_check last-2: (%f %f %f)\n",t1[t1.size()-3][0],t1[t1.size()-3][1],t1[t1.size()-3][2],temp_trajs_ori[temp_trajs_ori.size()-3][0],temp_trajs_ori[temp_trajs_ori.size()-3][1],temp_trajs_ori[temp_trajs_ori.size()-3][2],temp_trajs_check[temp_trajs_check.size()-3][0],temp_trajs_check[temp_trajs_check.size()-3][1],temp_trajs_check[temp_trajs_check.size()-3][2]);
        //       printf("ori last-1: (%f %f %f), temp_trajs_ori last-1: (%f %f %f), temp_trajs_check last-1: (%f %f %f)\n",t1[t1.size()-2][0],t1[t1.size()-2][1],t1[t1.size()-2][2],temp_trajs_ori[temp_trajs_ori.size()-2][0],temp_trajs_ori[temp_trajs_ori.size()-2][1],temp_trajs_ori[temp_trajs_ori.size()-2][2],temp_trajs_check[temp_trajs_check.size()-2][0],temp_trajs_check[temp_trajs_check.size()-2][1],temp_trajs_check[temp_trajs_check.size()-2][2]);
        //       printf("ori last: (%f %f %f), temp_trajs_ori last: (%f %f %f), temp_trajs_check last: (%f %f %f)\n",t1[t1.size()-1][0],t1[t1.size()-1][1],t1[t1.size()-1][2],temp_trajs_ori[temp_trajs_ori.size()-1][0],temp_trajs_ori[temp_trajs_ori.size()-1][1],temp_trajs_ori[temp_trajs_ori.size()-1][2],temp_trajs_check[temp_trajs_check.size()-1][0],temp_trajs_check[temp_trajs_check.size()-1][1],temp_trajs_check[temp_trajs_check.size()-1][2]);
        //       printf("t1 size: %zu, temp_trajs_ori size: %zu, temp_trajs_check size: %zu\n",t1.size(),temp_trajs_ori.size(),temp_trajs_check.size());
        //       // if (current_round >=6){
        //       //   std::array<double,3> v = {0,0,0};
        //       //   std::set<size_t> lossless;
        //       //   double direction = trajID_direction_vector[current_trajID];
        //       //   auto rk4_ori = newRK4_3d(t1[t1.size()-2].data(),v.data(),grad_ori,t_config.h * direction ,r3,r2,r1,lossless,true);
        //       //   auto rk4_check = newRK4_3d(temp_trajs_check[temp_trajs_check.size()-2].data(),v.data(),grad_dec,t_config.h * direction,r3,r2,r1,lossless,true);
        //       //   printf("done\n");
        //       // }
        //       break;
        //     }
        //     end_fix_index = std::min(end_fix_index + static_cast<int>(0.005*t_config.max_length), static_cast<int>(t1.size()));
        //   }
        //   else{
        //     //成功修正当前trajectory
        //     //printf("fix traj %zu successfully\n",current_trajID);
        //     trajs_dec[current_trajID] = temp_trajs_check;
        //     break;
        //   }
        // }
      
      
      } 
      omp_destroy_lock(&lock);
    
      //汇总all_vertex_for_all_diff_traj
      // printf("merging all_vertex_for_all_diff_traj...\n");
      for (const auto& local_set:local_all_vertex_for_all_diff_traj){
        // printf("local_set size: %zu\n",local_set.size());
        final_vertex_need_to_lossless.insert(local_set.begin(),local_set.end());
      }
      printf("final_vertex_need_to_lossless size: %zu\n",final_vertex_need_to_lossless.size());
      
      auto index_time_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_index_time = index_time_end - index_time_start;
      index_time_vec.push_back(elapsed_index_time.count());
      //此时dec_U,dec_V,dec_W已经更新，需要重新计算所有的trajectory
      printf("recalculating trajectories for updated decompressed data...\n");
      //get trajectories for updated decompressed data
      auto recalc_trajs_start = std::chrono::high_resolution_clock::now();

      //用原来的trajs_dec替换
      trajs_dec.resize(keys_dec.size() * 6);
      for (auto& traj : trajs_dec) {
        traj.resize(t_config.max_length, {0.0, 0.0, 0.0});
      }

      std::vector<std::set<size_t>> thread_lossless_index_dec_next(thread_num_fix);
      #pragma omp parallel for
      for (size_t j = 0; j < keys_dec.size(); ++j) {
          int key = keys_dec[j];
          auto &cp = critical_points_out[key];
          if (cp.type >=3 && cp.type <= 6){
              int thread_id = omp_get_thread_num();
              auto eigvec = cp.eig_vec;
              auto eigval = cp.eigvalues;
              auto pt = cp.x;
              std::array<std::array<double, 4>, 6> directions; //6 directions, first is direction(1 or -1), next 3 are seed point
              for (int i = 0; i < 3; i++){
                  if (eigval[i] > 0){
                      directions[i][0] = 1;
                      directions[i][1] = eps * eigvec[i][0] + pt[0];
                      directions[i][2] = eps * eigvec[i][1] + pt[1];
                      directions[i][3] = eps * eigvec[i][2] + pt[2];
                      directions[i+3][0] = 1;
                      directions[i+3][1] = -1 * eps * eigvec[i][0] + pt[0];
                      directions[i+3][2] = -1 * eps * eigvec[i][1] + pt[1];
                      directions[i+3][3] = -1 * eps* eigvec[i][2] + pt[2];
                  }
                  else{
                      directions[i][0] = -1;
                      directions[i][1] = eps * eigvec[i][0] + pt[0];
                      directions[i][2] = eps * eigvec[i][1] + pt[1];
                      directions[i][3] = eps * eigvec[i][2] + pt[2];
                      directions[i+3][0] = -1;
                      directions[i+3][1] = -1 * eps * eigvec[i][0] + pt[0];
                      directions[i+3][2] = -1 * eps * eigvec[i][1] + pt[1];
                      directions[i+3][3] = -1 * eps * eigvec[i][2] + pt[2];
                  }
              }          
              for (int k = 0; k < 6; k++){
                  std::array<double,3> seed = {directions[k][1], directions[k][2], directions[k][3]};
                  std::vector<std::array<double, 3>> result_return = trajectory_3d_parallel(pt, seed, h * directions[k][0], max_length, r3,r2,r1, critical_points_out, grad_dec,thread_lossless_index_dec_next,thread_id);
                  trajs_dec[j*6 + k] = result_return;
              }
          }
      }

      if(RECORD_INTERMEDIATE_TRAJECTORY == 1){
        if (file_out_dir != ""){
          std::vector<std::vector<std::array<double, 3>>> trajs_dec_intermidiate;
          for (const auto& traj:trajs_dec){
            trajs_dec_intermidiate.push_back(traj);
          }
          save_trajs_to_binary_3d(trajs_dec_intermidiate, file_out_dir + "state_" + std::to_string(current_round) + "trajs_dec_3d.bin");
        }
      }

    
      auto recalc_trajs_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_recalc_trajs = recalc_trajs_end - recalc_trajs_start;
      re_cal_trajs_time_vec.push_back(elapsed_recalc_trajs.count());
      //compare the new trajectories with the old ones
      printf("comparing the new trajectories with the old ones to see if all trajectories are fixed ...\n");

      int wrong = 0;
      int wrong_num_outside = 0;
      int wrong_num_max_iter = 0;
      int wrong_num_find_cp = 0;
      //这个for现在也要并行了，因为frechetDistance算的慢
      std::vector<std::set<size_t>> local_trajID_need_fix_next(thread_num_fix);
      auto compare_traj_start = std::chrono::high_resolution_clock::now();
      #pragma omp parallel for reduction(+:wrong, wrong_num_outside, wrong_num_max_iter, wrong_num_find_cp)
      for (size_t i =0; i< trajs_ori.size(); ++i){
        auto t1 = trajs_ori[i];
        auto t2 = trajs_dec[i];
        bool cond1 = get_cell_offset_3d(t1.back().data(), r3, r2, r1) == get_cell_offset_3d(t2.back().data(), r3, r2, r1);
        bool cond2 = t1.size() == t_config.max_length;
        bool cond3 = t2.size() == t_config.max_length;
        // bool f_dis = frechetDistance(t1, t2) >= threshold;
        if (LastTwoPointsAreEqual(t1)){
          //ori outside
          if (!LastTwoPointsAreEqual(t2)){
            //dec inside
            wrong ++;
            wrong_num_outside ++;
            // trajID_need_fix_next.insert(i);
            local_trajID_need_fix_next[omp_get_thread_num()].insert(i);
          }
          else{
            //dec outside
            if ((euclideanDistance(t1.back(), t2.back()) > threshold_outside) || (frechetDistance(t1, t2) >= threshold)){
              wrong ++;
              wrong_num_outside ++;
              // trajID_need_fix_next.insert(i);
              local_trajID_need_fix_next[omp_get_thread_num()].insert(i);
            }
          }
        }
        else if (cond2){
          if (t2.size() != t_config.max_length){
            wrong ++;
            wrong_num_max_iter ++;
            // trajID_need_fix_next.insert(i);
            local_trajID_need_fix_next[omp_get_thread_num()].insert(i);
          }
          else{
            // if ((euclideanDistance(t1.back(), t2.back()) >= threshold_max_iter) || (frechetDistance(t1, t2) >= threshold)){
            //   wrong ++;
            //   wrong_num_max_iter ++;
            //   // trajID_need_fix_next.insert(i);
            //   local_trajID_need_fix_next[omp_get_thread_num()].insert(i);
            // }
            if (!((euclideanDistance(t1.back(), t2.back()) <= threshold_max_iter) && (frechetDistance(t1, t2) <= threshold))){
              wrong ++;
              wrong_num_max_iter ++;
              // trajID_need_fix_next.insert(i);
              local_trajID_need_fix_next[omp_get_thread_num()].insert(i);
            }
          }
        }
        else{
          //reach cp
          if(!cond1 || (frechetDistance(t1, t2) >= threshold)){
            wrong ++;
            wrong_num_find_cp ++;
            // trajID_need_fix_next.insert(i);
            local_trajID_need_fix_next[omp_get_thread_num()].insert(i);
          }
        }     

      }
      //汇总local_trajID_need_fix_next
      for (auto& s:local_trajID_need_fix_next){
        trajID_need_fix_next.insert(s.begin(),s.end());
      }
      
      printf("wrong: %d, wrong_num_outside: %d, wrong_num_max_iter: %d, wrong_num_find_cp: %d\n",wrong,wrong_num_outside,wrong_num_max_iter,wrong_num_find_cp);
      auto compare_traj_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> compare_traj_elapsed = compare_traj_end - compare_traj_start;
      compare_time_vec.push_back(compare_traj_elapsed.count());
      
      if(trajID_need_fix_next.size() == 0){ //不需要下一轮的修复
        stop = true;
        printf("All trajectories are fixed!\n");
      }
      else{
        //printf("trajID_need_fix_next size: %ld\n", trajID_need_fix_next.size());
        trajID_need_fix_next_vec.push_back(trajID_need_fix_next.size());
        trajID_need_fix_next_detail_vec.push_back({wrong_num_outside, wrong_num_max_iter, wrong_num_find_cp});
        trajID_need_fix.clear();
        for(auto o:trajID_need_fix_next){
          trajID_need_fix.insert(o);
        }
        trajID_need_fix_next.clear();
      }


      if(current_round >= 9){
        //print first of the trajID_need_fix
        int first_trajID_need_fix = *trajID_need_fix.begin();
        printf("first trajID_need_fix: %d\n",first_trajID_need_fix);
        auto t1 = trajs_ori[first_trajID_need_fix];
        auto t2 = trajs_dec[first_trajID_need_fix];
        double direction = trajID_direction_vector[first_trajID_need_fix];
        printf("h * direction: %f\n",h * direction);
        for (size_t i = 0; i < t1.size(); ++i){
          printf("ori: %f %f %f, dec: %f %f %f\n",t1[i][0],t1[i][1],t1[i][2],t2[i][0],t2[i][1],t2[i][2]);
          std::array<double,3> v = {0,0,0};
          std::set<size_t> lossless;
          auto rk4_ori = newRK4_3d(t1[i].data(),v.data(),grad_ori,t_config.h * direction,r3,r2,r1,lossless,true);
          auto rk4_dec = newRK4_3d(t2[i].data(),v.data(),grad_dec,t_config.h * direction,r3,r2,r1,lossless,true);
          printf("~~~~~\n");
          auto rk4_ori_neg_h = newRK4_3d(t1[i].data(),v.data(),grad_ori,-t_config.h * direction,r3,r2,r1,lossless,true);
          auto rk4_dec_neg_h = newRK4_3d(t2[i].data(),v.data(),grad_dec,-t_config.h * direction,r3,r2,r1,lossless,true);
          printf("====================================\n");
          }
        printf("ori last-2: %f %f %f, dec last-2: %f %f %f\n",t1[t1.size()-3][0],t1[t1.size()-3][1],t1[t1.size()-3][2],t2[t2.size()-3][0],t2[t2.size()-3][1],t2[t2.size()-3][2]);
        printf("ori last-1: %f %f %f, dec last-1: %f %f %f\n",t1[t1.size()-2][0],t1[t1.size()-2][1],t1[t1.size()-2][2],t2[t2.size()-2][0],t2[t2.size()-2][1],t2[t2.size()-2][2]);
        printf("ori last: %f %f %f, dec last: %f %f %f\n",t1[t1.size()-1][0],t1[t1.size()-1][1],t1[t1.size()-1][2],t2[t2.size()-1][0],t2[t2.size()-1][1],t2[t2.size()-1][2]);
        
        // printf("last-2 rk4:\n");
        // std::array<double,3> v = {0,0,0};
        // std::set<size_t> lossless;
        // auto rk4_ori = newRK4_3d(t1[t1.size()-2].data(),v.data(),grad_ori,t_config.h,r3,r2,r1,lossless,true);
        // auto rk4_dec = newRK4_3d(t2[t2.size()-2].data(),v.data(),grad_dec,t_config.h,r3,r2,r1,lossless,true);
        //   exit(0);
      }
    
    }while (!stop);
    auto end_alg_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_alg_time = end_alg_time - start_alg_time;
    //printf("traj_ori_begain time: %f\n", traj_ori_begin_elapsed.count());
    //printf("traj_dec_begin time: %f\n", traj_dec_begin_elapsed.count()); 
    
    printf("total round: %d\n", current_round);
    // for (auto t:index_time_vec){
    //   printf("index_time: %f\n", t);
    // }
    printf("Total time (excclude cpsz time): %f\n", elapsed_alg_time.count());
    printf("traj_begin(ori+dec) time: %f\n", traj_ori_begin_elapsed.count() + traj_dec_begin_elapsed.count());
    printf("compare & init_queue time: %f\n", init_queue_elapsed.count());
    printf("sum of index_time: %f\n", std::accumulate(index_time_vec.begin(), index_time_vec.end(), 0.0));
    // for (auto t:re_cal_trajs_time_vec){
    //   printf("re_cal_trajs_time: %f\n", t);
    // }
    printf("sum of re_cal_trajs_time: %f\n", std::accumulate(re_cal_trajs_time_vec.begin(), re_cal_trajs_time_vec.end(), 0.0));
    // for (auto t:compare_time_vec){
    //   printf("compare & update_queue_time: %f\n", t);
    // }
    printf("sum of compare & update_queue_time: %f\n", std::accumulate(compare_time_vec.begin(), compare_time_vec.end(), 0.0));

    for(auto t:trajID_need_fix_next_vec){
      printf("trajID_need_fix_next: %d\n", t);
    }
  
    // printf("====================================\n");
    // printf("BEGIN Compression Ratio: %f\n", begin_cr);
    // printf("psnr_overall_cpsz: %f\n",psnr_cpsz_overall);
    // printf("%d\n",current_round);
    // printf("%f\n",elapsed_alg_time.count());
    // printf("%f\n",traj_ori_begin_elapsed.count() + traj_dec_begin_elapsed.count());
    // printf("%f\n",init_queue_elapsed.count());
    // printf("%f\n",std::accumulate(index_time_vec.begin(), index_time_vec.end(), 0.0));
    // printf("%f\n",std::accumulate(re_cal_trajs_time_vec.begin(), re_cal_trajs_time_vec.end(), 0.0));
    // printf("%f\n",std::accumulate(compare_time_vec.begin(), compare_time_vec.end(), 0.0));
    // struct result_struct
    // {
    //   double begin_cr_ = begin_cr;
    //   double psnr_cpsz_overall_ = psnr_cpsz_overall;
    //   int current_round_ = current_round;
    //   double elapsed_alg_time_ = elapsed_alg_time.count();
    //   double cpsz_comp_time_ = cpsz_comp_duration.count();
    //   double cpsz_decomp_time_ = cpsz_decomp_duration.count();
    //   // pass trajID_need_fix_next_vec
    //   std::vector<int> trajID_need_fix_next_vec_ = trajID_need_fix_next_vec;
    //   //trajID_need_fix_next_detail_vec
    //   std::vector<std::vector<int>> trajID_need_fix_next_detail_vec_ = trajID_need_fix_next_detail_vec;
    // };
    
    // now final check
    // printf("verifying the final decompressed data...\n");
    // printf("BEGIN Compression Ratio: %f\n", begin_cr);
    printf("final checking...\n");
    bool write_flag = true;
    thresholds th = {threshold, threshold_outside, threshold_max_iter};
    result_struct results ={cp_cal_duration.count(),begin_cr,psnr_cpsz_overall,current_round,elapsed_alg_time.count(),cpsz_comp_duration.count(),cpsz_decomp_duration.count(),current_round,trajID_need_fix_next_vec,trajID_need_fix_next_detail_vec,origin_traj_detail};
    final_check(U, V, W, r1, r2, r3, max_eb,obj,t_config,thread_num_compression,total_thread,final_vertex_need_to_lossless,th,results,eb_type,file_out_dir);   
}