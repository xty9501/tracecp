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
#include <ftk/numeric/eigen_solver2.hh>
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
#include <list>
#include <set>


#include "sz_cp_preserve_utils.hpp"
#include "sz_compress_cp_preserve_2d.hpp"
#include "sz_decompress_cp_preserve_2d.hpp"
#include "sz_lossless.hpp"
#include <iostream> 
#include <algorithm>
#include<omp.h>

#define CPSZ_OMP_FLAG 0

std::array<double, 2> findLastNonNegativeOne(const std::vector<std::array<double, 2>>& vec) {
    for (auto it = vec.rbegin(); it != vec.rend(); ++it) {
        if ((*it)[0] != -1 && (*it)[1] != -1) {
            return *it;
        }
    }
    // 如果没有找到非 (-1, -1) 的点，返回最后一个点
    return vec.back();
}


double euclideanDistance(const std::array<double, 2>& p1, const std::array<double, 2>& p2) {
    return std::sqrt(std::pow(p1[0] - p2[0], 2) + std::pow(p1[1] - p2[1], 2));
}

double MaxeuclideanDistance(const std::vector<std::array<double, 2>>& vec1, const std::vector<std::array<double, 2>>& vec2) {
    double max_distance = 0;
    for (int i = 0; i < vec1.size(); i++) {
        double distance = euclideanDistance(vec1[i], vec2[i]);
        if (distance > max_distance) {
            max_distance = distance;
        }
    }
    return max_distance;
}

//EDR distance return a int, range from 0 to max(seq1.size(), seq2.size()), '
//if two point's distance is less than threshold, consider them as the same point
double calculateEDR2D(const std::vector<std::array<double, 2>>& seq1, const std::vector<std::array<double, 2>>& seq2, double threshold) {
    size_t len1 = seq1.size();
    size_t len2 = seq2.size();

    // Create a 2D vector to store distances, initialize with zeros
    std::vector<std::vector<double>> dp(len1 + 1, std::vector<double>(len2 + 1, 0));

    // Initialize the first column and first row
    for (size_t i = 0; i <= len1; ++i) {
        dp[i][0] = i;
    }
    for (size_t j = 0; j <= len2; ++j) {
        dp[0][j] = j;
    }

    // Fill the dp matrix
    for (size_t i = 1; i <= len1; ++i) {
        for (size_t j = 1; j <= len2; ++j) {
            // Use Euclidean distance and compare with threshold
            double cost = euclideanDistance(seq1[i - 1], seq2[j - 1]) <= threshold ? 0 : 1;
            dp[i][j] = std::min({
                dp[i - 1][j] + 1,    // Deletion
                dp[i][j - 1] + 1,    // Insertion
                dp[i - 1][j - 1] + cost  // Substitution or match
            });
        }
    }

    return dp[len1][len2];
}

//function to calculate the angle between two vectors
double angleBetweenVectors(const std::array<double, 2>& velocity1, const std::array<double, 2>& velocity2) {
    double dotProduct = velocity1[0] * velocity2[0] + velocity1[1] * velocity2[1];
    double magnitude1 = sqrt(velocity1[0] * velocity1[0] + velocity1[1] * velocity1[1]);
    double magnitude2 = sqrt(velocity2[0] * velocity2[0] + velocity2[1] * velocity2[1]);
    return acos(dotProduct / (magnitude1 * magnitude2));
}

double frechetDistance(const vector<array<double, 2>>& P, const vector<array<double, 2>>& Q) {
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
    //return early when hit the boundary of the matrix
}

double ESfrechetDistance(const vector<array<double, 2>>& P, const vector<array<double, 2>>& Q) {
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

    // 计算最后一行的最小值
    // double result = dp[n-1][0];
    // for (int j = 1; j < m; j++) {
    //     if (dp[n-1][j] < result) {
    //         result = dp[n-1][j];
    //     }
    // }
    // 计算最后一列的最小值
    double result = dp[0][m-1];  // 初始化为最后一列的第一个元素
    for (int i = 1; i < n; i++) {
        if (dp[i][m-1] < result) {
            result = dp[i][m-1];  // 找到最后一列的最小值
        }
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

std::vector<size_t> get_surrouding_3x3_vertex_index(const double x, const double y, const int DW, const int DH) {
    std::vector<size_t> result;
    int x0 = floor(x);
    int y0 = floor(y);
    for (int i = -1; i <= 2; i++) {
        for (int j = -1; j <= 2; j++) {
            int x1 = x0 + i;
            int y1 = y0 + j;
            if (x1 >= 0 && x1 < DW && y1 >= 0 && y1 < DH) {
                result.push_back(y1 * DW + x1);
            }
        }
    }
    
}

std::pair<std::array<double, 2>, std::array<double, 2>> findLastTwoNonNegativeOne(const std::vector<std::array<double, 2>>& vec) {
    std::array<double, 2> last_valid_point = {-1, -1};
    std::array<double, 2> second_last_valid_point = {-1, -1};

    int valid_count = 0;  // 计数找到的非 (-1, -1) 的点
    
    // 遍历输入向量，查找最后两个非 (-1, -1) 的点
    for (auto it = vec.rbegin(); it != vec.rend(); ++it) {
        if ((*it)[0] != -1 && (*it)[1] != -1) {
            if (valid_count == 0) {
                last_valid_point = *it;  // 记录最后一个非 (-1, -1) 的点
            } else if (valid_count == 1) {
                second_last_valid_point = *it;  // 记录倒数第二个非 (-1, -1) 的点
                break;  // 找到两个点后，退出循环
            }
            valid_count++;
        }
    }

    return {second_last_valid_point, last_valid_point};  // 返回一对坐标
}

bool LastTwoPointsAreEqual(const std::vector<std::array<double, 2>>& vec) {
    if (vec.size() < 2) return false;
    return vec[vec.size()-1] == vec[vec.size()-2];
    // const auto& last = vec[vec.size()-1];      // 最后一个点
    // const auto& second_last = vec[vec.size()-2]; // 倒数第二个点

    // return std::fabs(last[0] - second_last[0]) < 1e-4 && 
    //        std::fabs(last[1] - second_last[1]) < 1e-4;
}

bool SearchElementFromBack(const std::vector<std::array<double, 2>>& vec, const std::array<double, 2>& target) {
    for (auto it = vec.rbegin(); it != vec.rend(); ++it) {
        if (*it == target) {
            return true;
        }
    }
    return false;
}


typedef struct traj_config{
  double h;
  double eps;
  int max_length; 
  double next_index_coeff;
} traj_config;


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
    double sum = 0, prodSum = 0, relerr = 0;

    double maxpw_relerr = 0; 
    for (i = 0; i < num_elements; i++){
        if (Max < ori_data[i]) Max = ori_data[i];
        if (Min > ori_data[i]) Min = ori_data[i];
        
        Type err = fabs(data[i] - ori_data[i]);
        if(ori_data[i]!=0 && fabs(ori_data[i])>1)
        {
            relerr = err/fabs(ori_data[i]);
            if(maxpw_relerr<relerr)
                maxpw_relerr = relerr;
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
    printf ("PSNR = %f, NRMSE= %.20G\n", psnr,nrmse);
    printf ("acEff=%f\n", acEff);   
}

bool check_pt_is_cp(const std::array<double, 2>& pt, const std::unordered_map<size_t, critical_point_t>& cps){
  for (const auto& cpd:cps){
    auto cp = cpd.second;
    if (std::abs(cp.x[0] - pt[0]) < 1e-3 && std::abs(cp.x[1] - pt[1]) < 1e-3){
      return true;
    }
  }
  return false;
}

double vector_field_resolution = std::numeric_limits<double>::max();
uint64_t vector_field_scaling_factor = 1;
// int DW = 128, DH = 128;// the dimensionality of the data is DW*DH

ftk::simplicial_regular_mesh m(2);
std::mutex mutex;

size_t global_count = 0;


inline bool file_exists(const std::string& filename) {
    std::ifstream f(filename.c_str());
    return f.good();
}



// template<typename Container>
// bool inside(const Container& x, int DH, int DW) {
//   if (x[0] <=0 || x[0] > DW-1 || x[1] <= 0 || x[1] > DH-1) return false;
//   return true;
// }

void record_criticalpoints(const std::string& prefix, const std::unordered_map<size_t, critical_point_t>& cps, bool write_sid=false){
  std::vector<double> singular;
  std::vector<double> attracting;
  std::vector<double> repelling;
  std::vector<double> saddle;
  std::vector<double> attracting_focus;
  std::vector<double> repelling_focus;
  std::vector<double> center;
  for (const auto& cpd:cps){
    auto cp = cpd.second;
    if (cp.type == SINGULAR){
      singular.push_back(cp.x[0]);
      singular.push_back(cp.x[1]);
    }
    else if (cp.type == ATTRACTING){
      attracting.push_back(cp.x[0]);
      attracting.push_back(cp.x[1]);
    }
    else if (cp.type == REPELLING){
      repelling.push_back(cp.x[0]);
      repelling.push_back(cp.x[1]);
    }
    else if (cp.type == SADDLE){
      saddle.push_back(cp.x[0]);
      saddle.push_back(cp.x[1]);
    }
    else if (cp.type == ATTRACTING_FOCUS){
      attracting_focus.push_back(cp.x[0]);
      attracting_focus.push_back(cp.x[1]);
    }
    else if (cp.type == REPELLING_FOCUS){
      repelling_focus.push_back(cp.x[0]);
      repelling_focus.push_back(cp.x[1]);
    }
    else if (cp.type == CENTER){
      center.push_back(cp.x[0]);
      center.push_back(cp.x[1]);
    }
    else {
      continue;
    }
  }
  // std::string prefix = "../data/position";
  writefile((prefix + "_singular.dat").c_str(), singular.data(), singular.size());
  writefile((prefix + "_attracting.dat").c_str(), attracting.data(), attracting.size());
  writefile((prefix + "_repelling.dat").c_str(), repelling.data(), repelling.size());
  writefile((prefix + "_saddle.dat").c_str(), saddle.data(), saddle.size());
  writefile((prefix + "_attracting_focus.dat").c_str(), attracting_focus.data(), attracting_focus.size());
  writefile((prefix + "_repelling_focus.dat").c_str(), repelling_focus.data(), repelling_focus.size());
  writefile((prefix + "_center.dat").c_str(), center.data(), center.size());
  printf("Successfully write critical points to file %s\n", prefix.c_str());
  
}






template<typename T>
void write_current_state_data(std::string file_path, const T * U, const T * V, T *dec_U, T *dec_V, std::vector<std::vector<std::array<double, 2>>> &trajs_dec,std::vector<int> &index_dec, size_t r1, size_t r2, int NUM_ITER){
  
  // write decompress traj
  std::string dec_traj_file = file_path + "dec_traj_iteration_" + std::to_string(NUM_ITER) + ".bin" + ".out";
  write_trajectory(trajs_dec, dec_traj_file.c_str());
  printf("Successfully write decompressed trajectory to file, total trajectory: %ld\n",trajs_dec.size());
  //write decompress index
  std::string index_file = file_path + "index_iteration_" + std::to_string(NUM_ITER) + ".bin" + ".out";
  writefile(index_file.c_str(), index_dec.data(), index_dec.size());
  printf("Successfully write orginal index to file, total index: %ld\n",index_dec.size());

  //write dec data
  std::string decU_file = file_path + "dec_u_iteration_" + std::to_string(NUM_ITER) + ".bin" + ".out";
  writefile(decU_file.c_str(), dec_U, r1*r2);
  std::string decV_file = file_path + "dec_v_iteration_" + std::to_string(NUM_ITER) + ".bin" + ".out";
  writefile(decV_file.c_str(), dec_V, r1*r2);
}


template<typename T>
void
fix_traj_v2(T * U, T * V,size_t r1, size_t r2, double max_pwr_eb,traj_config t_config,int totoal_thread, int obj,std::string eb_type,double threshold,double threshold_outside, double threshold_max_iter, std::string file_dir = ""){
  
  //控制写入文件的flag
  bool write_flag = true;
  
  int DW = r2;
  int DH = r1;
  bool stop = false;
  std::set<size_t> all_vertex_for_all_diff_traj;



  int NUM_ITER = 0;
  // double threshold = 0.5;
  // double threshold_outside = 0.5;
  // double threshold_max_iter = 0.5;

  //std::map<size_t, double> trajID_direction_map; //用来存储每个traj对应的方向
  //std::set<size_t> current_diff_traj_id;
  // std::unordered_map<size_t,int> current_diff_traj_id;
  //std::set<size_t> last_diff_traj_id;
  // std::unordered_map<size_t,int> last_diff_traj_id;
  std::vector<double> compare_time_vec;
  std::vector<double> index_time_vec;
  std::vector<double> re_cal_trajs_time_vec;
  std::vector<int> trajID_need_fix_next_vec;
  std::vector<std::array<int,3>> trajID_need_fix_next_detail_vec; //0:outside, 1.reach max iter, 2.find cp
  std::array<int,3> origin_traj_detail;
  std::vector<int> fixed_cpsz_trajID;
  //get cp for original data
  auto critical_points_ori = compute_critical_points(U, V, r1, r2);
  if (file_dir != ""){
    record_criticalpoints(file_dir + "critical_points_ori", critical_points_ori);
  }
  //exit(0);

  auto cpsz_comp_start = std::chrono::high_resolution_clock::now();
  size_t result_size = 0;
  unsigned char * result = NULL;
  double current_pwr_eb = 0;
  if(eb_type == "rel"){
    //****original version of cpsz********
    if (CPSZ_OMP_FLAG == 0){
      result = sz_compress_cp_preserve_2d_fix(U, V, r1, r2, result_size, false, max_pwr_eb, current_pwr_eb);
    }
    else{
      //use omp version
      std::set<size_t> empty_set;
      float * dec_inplace_U = NULL;
      float * dec_inplace_V = NULL;
      result = omp_sz_compress_cp_preserve_2d_record_vertex(U, V, r1, r2,result_size,false,max_pwr_eb,empty_set,totoal_thread,dec_inplace_U,dec_inplace_V,eb_type);
    }
    
  }
  else if (eb_type == "abs"){
    if (CPSZ_OMP_FLAG == 0){
      //****** cpsz with absolute error bound ******
      result = sz_compress_cp_preserve_2d_online_abs_record_vertex(U, V, r1, r2, result_size, false, max_pwr_eb);
    }
    else{
      std::set<size_t> empty_set;
      float * dec_inplace_U = NULL;
      float * dec_inplace_V = NULL;
      result = omp_sz_compress_cp_preserve_2d_record_vertex(U, V, r1, r2,result_size,false,max_pwr_eb,empty_set,totoal_thread,dec_inplace_U,dec_inplace_V,eb_type);
    }
    
  }
  else{
    printf("eb_type must be rel or abs\n");
    exit(0);
  }

  //***** use cpsz+st2 **********
  //result = sz_compress_cp_preserve_2d_st2_fix(U, V, r1, r2, result_size, false, max_pwr_eb, current_pwr_eb, critical_points_ori);
  
  //****** cpsz with absolute error bound ******
  
  
  unsigned char * result_after_lossless = NULL;
  size_t lossless_outsize = sz_lossless_compress(ZSTD_COMPRESSOR, 3, result, result_size, &result_after_lossless);
  free(result);

  auto cpsz_comp_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> cpsz_comp_duration = cpsz_comp_end - cpsz_comp_start;
  printf("cpsz Compress time: %f\n", cpsz_comp_duration.count());

  auto cpsz_decomp_start = std::chrono::high_resolution_clock::now();
  size_t lossless_output = sz_lossless_decompress(ZSTD_COMPRESSOR, result_after_lossless, lossless_outsize, &result, result_size);
  float * dec_U = NULL;
  float * dec_V = NULL;
  if (eb_type == "rel"){
    if (CPSZ_OMP_FLAG == 0){
      sz_decompress_cp_preserve_2d_online<float>(result, r1,r2, dec_U, dec_V); // use cpsz
    }
    else{
      omp_sz_decompress_cp_preserve_2d_online(result, r1,r2, dec_U, dec_V);
    }
  }
  else if (eb_type == "abs"){
    if (CPSZ_OMP_FLAG == 0){
      sz_decompress_cp_preserve_2d_online_record_vertex<float>(result, r1,r2, dec_U, dec_V); // use cpsz
    }
    else{
      omp_sz_decompress_cp_preserve_2d_online(result, r1,r2, dec_U, dec_V);
    }
    
  }
  //sz_decompress_cp_preserve_2d_online<float>(result, r1,r2, dec_U, dec_V); // use cpsz
  // calculate compression ratio
  printf("BEGIN Compressed size(original) = %zu, ratio = %f\n", lossless_outsize, (2*r1*r2*sizeof(float)) * 1.0/lossless_outsize);
  double cr_ori = (2*r1*r2*sizeof(float)) * 1.0/lossless_outsize; 

  auto cpsz_decomp_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> cpsz_decomp_duration = cpsz_decomp_end - cpsz_decomp_start;
  printf("cpsz Decompress time: %f\n", cpsz_decomp_duration.count());

  // exit(0);
  if (WRITE_OUT_EB == 1){
    exit(0);
  }


  double nrmse_u, nrmse_v,psnr_overall,psnr_cpsz_overall;
  verify(U, dec_U, r1*r2, nrmse_u);
  verify(V, dec_V, r1*r2, nrmse_v);
  psnr_cpsz_overall = 20 * log10(sqrt(2) / sqrt(nrmse_u*nrmse_u + nrmse_v*nrmse_v));
  printf("cpsz overall PSNR: %f\n", psnr_cpsz_overall);

  if (file_dir != ""){
    //get which vertex is lossless for cpsz
    std::vector<size_t> lossless_vertex_cpsz;
    for (size_t i = 0; i < r1*r2; ++i){
      if (U[i] == dec_U[i] && V[i] == dec_V[i]){
        lossless_vertex_cpsz.push_back(i);
      }
    }
    //write to binary
    writefile((file_dir + "lossless_vertex_cpsz.bin").c_str(),lossless_vertex_cpsz.data(),lossless_vertex_cpsz.size());
  }



  //get cp for decompressed data
  auto critical_points_dec = compute_critical_points(dec_U, dec_V, r1, r2);
  //check two have same cp
  //check if all cretical_points_ori in critical_points_dec
  for (const auto& cpd:critical_points_ori){
    auto cp = cpd.second;
    if (critical_points_dec.find(cpd.first) == critical_points_dec.end()){
      printf("critical point %ld not in critical_points_dec\n", cpd.first);
      printf("critical point %ld, x: %f, y: %f, type: %d\n", cpd.first, cp.x[0], cp.x[1], cp.type);
    }
  }
  //检查多出来的cp
  for (const auto& cpd:critical_points_dec){
    auto cp = cpd.second;
    if (critical_points_ori.find(cpd.first) == critical_points_ori.end()){
      printf("critical point %ld not in critical_points_ori\n", cpd.first);
      printf("critical point %ld, x: %f, y: %f, type: %d\n", cpd.first, cp.x[0], cp.x[1], cp.type);
      //print cp.X matrix,which is 3x2
      // printf("cp.X matrix:\n");
      // for (int i = 0; i < 3; ++i){
      //   for (int j = 0; j < 2; ++j){
      //     printf("%f ", cp.X[i][j]);
      //   }
      //   printf("\n");
      // }
      printf("cp.V matrix:\n");
      for (int i = 0; i < 3; ++i){
        for (int j = 0; j < 2; ++j){
          printf("%f ", cp.V[i][j]);
        }
        printf("\n");
      }
    }
  }
    
  if (critical_points_ori.size() != critical_points_dec.size()){
    printf("critical_points_ori size: %ld, critical_points_dec size: %ld\n", critical_points_ori.size(), critical_points_dec.size());
    printf("critical points size not equal\n");
    exit(0);
  }
  //check if all key in critical_points_ori is in critical_points_dec
  for (const auto& cpd:critical_points_ori){
    if (critical_points_dec.find(cpd.first) == critical_points_dec.end()){
      printf("critical point %ld not in critical_points_dec\n", cpd.first);
      printf("critical point %ld, x: %f, y: %f, type: %d\n", cpd.first, cpd.second.x[0], cpd.second.x[1], cpd.second.type);
    }
  }
  printf("critical_points_ori size: %ld, critical_points_dec size: %ld\n", critical_points_ori.size(), critical_points_dec.size());

  // check if all cp are exactly the same
  for (auto p : critical_points_ori){
    auto cp_ori = p.second.x;
    auto cp_dec = critical_points_dec[p.first].x;
    if (cp_ori[0] != cp_dec[0] || cp_ori[1] != cp_dec[1]){
      printf("critical point not equal\n");
      printf("cp_ori: %f, %f\n", cp_ori[0], cp_ori[1]);
      printf("cp_dec: %f, %f\n", cp_dec[0], cp_dec[1]);
      exit(0);
    }
  }

  // exit(0);

  //replace all vertex in the boundary to dec_U, dec_V
  // for (size_t i = 0; i < r1; ++i){
  //   dec_U[i * DW] = U[i * DW];
  //   dec_U[i * DW + DW - 1] = U[i * DW + DW - 1];
  //   dec_V[i * DW] = V[i * DW];
  //   dec_V[i * DW + DW - 1] = V[i * DW + DW - 1];
    //
    // dec_U[i * DW + 1] = U[i * DW + 1];
    // dec_U[i * DW + DW - 2] = U[i * DW + DW - 2];
    // dec_V[i * DW + 1] = V[i * DW + 1];
    // dec_V[i * DW + DW - 2] = V[i * DW + DW - 2];
  // }
  // for (size_t i = 0; i < r2; ++i){
  //   dec_U[i] = U[i];
  //   dec_U[(DH - 1) * DW + i] = U[(DH - 1) * DW + i];
  //   dec_V[i] = V[i];
  //   dec_V[(DH - 1) * DW + i] = V[(DH - 1) * DW + i];
    //
    // dec_U[DW + i] = U[DW + i];
    // dec_U[(DH - 2) * DW + i] = U[(DH - 2) * DW + i];
    // dec_V[DW + i] = V[DW + i];
    // dec_V[(DH - 2) * DW + i] = V[(DH - 2) * DW + i];
  // }
  //get grad for original data
  ftk::ndarray<float> grad_ori;
  grad_ori.reshape({2, static_cast<unsigned long>(r2), static_cast<unsigned long>(r1)});
  refill_gradient(0, r1, r2, U, grad_ori);
  refill_gradient(1, r1, r2, V, grad_ori);
  //get grad for decompressed data
  ftk::ndarray<float> grad_dec;
  grad_dec.reshape({2, static_cast<unsigned long>(r2), static_cast<unsigned long>(r1)});
  refill_gradient(0, r1, r2, dec_U, grad_dec);
  refill_gradient(1, r1, r2, dec_V, grad_dec);

  auto total_time_start = std::chrono::high_resolution_clock::now();

  printf("first time calculate trajectory\n");
  //*************先计算一次整体的traj_ori 和traj_dec,后续只需增量修改*************
  //get trajectory for original data
  
  std::set<size_t> vertex_ori;
  // std::unordered_map<size_t,std::vector<size_t>> vertex_ori_map;
  printf("calculating trajectory for original data...\n");
  // 统计耗时
  auto start = std::chrono::high_resolution_clock::now();
  size_t ori_saddle_count = 0;
  
  //小地方，可以并行计算
  std::vector<size_t> keys;
  for (const auto&p : critical_points_ori){
    if (p.second.type == SADDLE){
      keys.push_back(p.first);
    }
  }
  printf("keys size(# of saddle): %ld\n", keys.size());
  std::vector<double> trajID_direction_vector(keys.size() * 4, 0);
  std::vector<std::vector<std::array<double, 2>>> trajs_ori(keys.size() * 4);//指定长度为saddle的个数*4，因为每个saddle有4个方向
  printf("trajs_ori size: %ld\n", trajs_ori.size());
  // /*这里一定要加上去，不然由于动态扩容会产生额外开销*/
  size_t expected_size = t_config.max_length * 1 + 1;
  for (auto& traj : trajs_ori) {
      traj.reserve(expected_size); // 预分配容量
  }


  std::vector<std::set<size_t>> thread_lossless_index(totoal_thread);
  #pragma omp parallel for reduction(+:ori_saddle_count)
  // for(const auto& p:critical_points_ori){
  for(size_t j = 0; j < keys.size(); ++j){
    auto key = keys[j];
    auto &cp = critical_points_ori[key];
    //auto cp = p.second;
    if (cp.type != SADDLE){
      printf("not saddle point\n");
      exit(0);
    }
    int thread_id = omp_get_thread_num();
    //printf("key: %ld, thread: %d\n", key, thread_id);
    ori_saddle_count ++;
    auto eigvec = cp.eig_vec;
    auto eigval = cp.eig;
    auto pt = cp.x;
    std::array<std::array<double, 3>, 4> directions;//first is direction(1 or -1), next 2 are seed point
    for (int i = 0; i < 2; ++i){
            if (eigval[i].real() > 0){
              directions[i][0] = 1;
              directions[i][1] = t_config.eps * eigvec[i][0] + pt[0];
              directions[i][2] = t_config.eps * eigvec[i][1] + pt[1];
              directions[i+2][0] = 1;
              directions[i+2][1] = -1 * t_config.eps * eigvec[i][0] + pt[0];
              directions[i+2][2] = -1 * t_config.eps * eigvec[i][1] + pt[1];
            }
            else{
              directions[i][0] = -1;
              directions[i][1] = t_config.eps * eigvec[i][0] + pt[0];
              directions[i][2] = t_config.eps * eigvec[i][1] + pt[1];
              directions[i+2][0] = -1;
              directions[i+2][1] = -1 * t_config.eps * eigvec[i][0] + pt[0];
              directions[i+2][2] = -1 * t_config.eps * eigvec[i][1] + pt[1];
            }
    }


    for (int k = 0; k < 4; ++k) {
      std::vector<std::array<double, 2>> result_return;
      //check if inside
      std::array<double, 2> seed = {directions[k][1], directions[k][2]};
      //size_t current_traj_index = trajID_counter++;
      result_return = trajectory_parallel(pt, seed, directions[k][0] * t_config.h, t_config.max_length, DH, DW, critical_points_ori, grad_ori,thread_lossless_index,thread_id);
      trajs_ori[j *4 + k] = result_return;
      trajID_direction_vector[j*4 + k] = directions[k][0];

    }
  }  
  // 这里只是计算trajs，不需要算vertex其实
  // for (const auto& local_set:thread_lossless_index){
  //   vertex_ori.insert(local_set.begin(), local_set.end());
  // }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> traj_ori_begin_elapsed = end - start;
  printf("Time for original trajectory calculation: %f\n", traj_ori_begin_elapsed.count());
  printf("ori_cp_count: %ld\n", critical_points_ori.size());
  printf("ori_saddle_count: %ld\n", ori_saddle_count);
  printf("size of trajs_ori: %ld\n", trajs_ori.size());
  printf("size of vertex_ori: %ld\n", vertex_ori.size());
  //print how many element in trajID_direction_map
  printf("size of trajID_direction_vector: %ld\n", trajID_direction_vector.size());


  

    //*************计算解压缩数据的traj_dec*************
  std::set<size_t> vertex_dec;
  printf("calculating trajectory for decompressed data...\n");
  start = std::chrono::high_resolution_clock::now();
  size_t dec_saddle_count = 0;

  std::vector<size_t> keys_dec;
  for (const auto&p : critical_points_dec){
    if (p.second.type == SADDLE){
      keys_dec.push_back(p.first);
    }
  }
  printf("keys_dec size(# of saddle): %ld\n", keys_dec.size());
  std::vector<std::vector<std::array<double, 2>>> trajs_dec(keys_dec.size() * 4);//指定长度为saddle的个数*4，因为每个saddle有4个方向
  printf("trajs_dec size: %ld\n", trajs_dec.size());
  // /*这里一定要加上去，不然由于动态扩容会产生额外开销*/
  for (auto& traj : trajs_dec) {
      traj.reserve(expected_size); // 预分配容量
  }


  std::vector<std::set<size_t>> thread_lossless_index_dec(totoal_thread);
  #pragma omp parallel for reduction(+:dec_saddle_count)
  for(size_t j = 0; j < keys.size(); ++j){
    auto key = keys[j];
    auto &cp = critical_points_dec[key];
    //auto cp = p.second;
    if (cp.type != SADDLE){
      printf("not saddle point\n");
      exit(0);
    }
    int thread_id = omp_get_thread_num();
    dec_saddle_count ++;
    auto eigvec = cp.eig_vec;
    auto eigval = cp.eig;
    auto pt = cp.x;
    std::array<std::array<double, 3>, 4> directions;//first is direction(1 or -1), next 2 are seed point
    for (int i = 0; i < 2; i++){
            if (eigval[i].real() > 0){
              directions[i][0] = 1;
              directions[i][1] = t_config.eps * eigvec[i][0] + pt[0];
              directions[i][2] = t_config.eps * eigvec[i][1] + pt[1];
              directions[i+2][0] = 1;
              directions[i+2][1] = -1 * t_config.eps * eigvec[i][0] + pt[0];
              directions[i+2][2] = -1 * t_config.eps * eigvec[i][1] + pt[1];
            }
            else{
              directions[i][0] = -1;
              directions[i][1] = t_config.eps * eigvec[i][0] + pt[0];
              directions[i][2] = t_config.eps * eigvec[i][1] + pt[1];
              directions[i+2][0] = -1;
              directions[i+2][1] = -1 * t_config.eps * eigvec[i][0] + pt[0];
              directions[i+2][2] = -1 * t_config.eps * eigvec[i][1] + pt[1];
            }
    }

    for (int i = 0; i < 4; i ++) {
      std::vector<std::array<double, 2>> result_return;
      //check if inside
      std::array<double, 2> seed = {directions[i][1], directions[i][2]};
      if(!inside(seed,DH, DW)){
        printf("seed is outside\n");
        printf("seed: (%f,%f)\n", seed[0], seed[1]);
        //exit(0);
      }
      // size_t current_traj_index = trajs_dec.size();
      // trajs_dec.push_back(result_return);
      result_return = trajectory_parallel(pt, seed, directions[i][0] * t_config.h, t_config.max_length, DH, DW, critical_points_dec, grad_dec,thread_lossless_index_dec,thread_id);
      trajs_dec[j * 4 + i] = result_return;
    }

  }

  //这里不需要计算vertex，因为只是计算trajs
  // for (const auto& local_set:thread_lossless_index_dec){
  //   vertex_dec.insert(local_set.begin(), local_set.end());
  // }
  end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> traj_dec_begin_elapsed = end - start;
  printf("Time for decompressed trajectory calculation: %f\n", traj_dec_begin_elapsed.count());
  // printf("dec_saddle_count: %ld\n", dec_saddle_count);
  // printf("size of trajs_dec: %ld\n", trajs_dec.size());
  // printf("size of vertex_dec: %ld\n", vertex_dec.size());

  //write cpsz trajectory to file
  if(file_dir != ""){
    save_trajs_to_binary(trajs_dec, file_dir + "cpsz_traj.bin");
    //write cpsz data
    writefile((file_dir + "decU_cpsz.bin").c_str(), dec_U, r1*r2);
    writefile((file_dir + "decV_cpsz.bin").c_str(), dec_V, r1*r2);
  }
  
  // exit(0);

  // optimization: for each saddle point, find the adjacent 12 triangles, and lossless store the index of the vertices
  // for (auto cp: critical_points_ori) {
  //   if (cp.second.type != SADDLE) continue;
  //   auto pt = cp.second.x;
  //   auto vertexs = get_surrouding_3x3_vertex_index(pt[0],pt[1], DW, DH);
  //   for (auto v: vertexs) {
  //     // vertex_ori.insert(v);
  //     all_vertex_for_all_diff_traj.insert(v);
  //     dec_U[v] = U[v];
  //     dec_V[v] = V[v];
  //     int x = v % DW;
  //     int y = v / DW;
  //     grad_dec(0, x, y) = U[v];
  //     grad_dec(1, x, y) = V[v];
  //   }
  // }
  
  //计算cpsz的frechet distance
  vector<double> frechetDistances_cpsz(trajs_ori.size(), -1);
  double max_distance_cpsz = -1.0;
  int max_index_cpsz = -1;
  #pragma omp parallel for
  for (int i = 0; i < trajs_ori.size(); i++) {
    auto t1 = trajs_ori[i];
    auto t2 = trajs_dec[i];
    frechetDistances_cpsz[i] = frechetDistance(t1, t2);
    #pragma omp critical
    {
      if (frechetDistances_cpsz[i] > max_distance_cpsz){
        max_distance_cpsz = frechetDistances_cpsz[i];
        max_index_cpsz = i;
      }
    }
  }
  double minVal_cpsz, maxVal_cpsz, medianVal_cpsz, meanVal_cpsz, stdevVal_cpsz;
  calculateStatistics(frechetDistances_cpsz, minVal_cpsz, maxVal_cpsz, medianVal_cpsz, meanVal_cpsz, stdevVal_cpsz);
  printf("Statistics data cpsz frechet distance===============\n");

  printf("min: %f\n", minVal_cpsz);
  printf("max: %f\n", maxVal_cpsz);
  printf("median: %f\n", medianVal_cpsz);
  printf("mean: %f\n", meanVal_cpsz);
  printf("stdev: %f\n", stdevVal_cpsz);
  printf("max index: %d,second index: %d, third index: %d\n", max_index_cpsz);
  printf("Statistics data cpsz frechet distance===============\n");


  // 计算哪里有问题（init queue）
  std::set<size_t> trajID_need_fix = {};
  auto init_queue_start = std::chrono::high_resolution_clock::now();
  int num_outside = 0;
  int num_max_iter = 0;
  int num_find_cp = 0;
  int wrong_num_outside = 0;
  int wrong_num_max_iter = 0;
  int wrong_num_find_cp = 0;
  std::vector<std::set<size_t>> local_trajID_need_fix(totoal_thread);

  // switch(obj){
  //   case 0:
  omp_set_num_threads(totoal_thread);
  #pragma omp parallel for reduction(+:num_outside, num_max_iter, num_find_cp, wrong_num_outside, wrong_num_max_iter, wrong_num_find_cp)
  for(size_t i =0; i< trajs_ori.size(); ++i){
    //printf("processing traj %ld by thread %d\n", i, omp_get_thread_num());
    auto t1 = trajs_ori[i];
    auto t2 = trajs_dec[i];
    bool cond1 = get_cell_offset(t1.back().data(), DW, DH) == get_cell_offset(t2.back().data(), DW, DH); //ori and dec reach same cp
    bool cond2 = (t1.size() == t_config.max_length); //ori reach max
    //bool f_dist = frechetDistance(t1, t2) >= threshold;
    if (LastTwoPointsAreEqual(t1)){
      num_outside ++;
      //ori outside
      if (!LastTwoPointsAreEqual(t2)){
        //dec inside
        wrong_num_outside ++;
        local_trajID_need_fix[omp_get_thread_num()].insert(i);
      }
      else{
        //dec outside
        //if (euclideanDistance(t1.back(), t2.back()) >= threshold_outside){
        if ( euclideanDistance(t1.back(), t2.back()) >= threshold_outside && frechetDistance(t1, t2) >= threshold){
          wrong_num_outside ++;
          local_trajID_need_fix[omp_get_thread_num()].insert(i);
        }
      }
    }
    else if (cond2){
      num_max_iter ++;
      //ori reach max
      //这里的判断是｜｜ 不是&&！！
      // if ((euclideanDistance(t1.back(), t2.back()) > threshold_max_iter)|| (t2.size() != t_config.max_length)){
      if (t2.size() != t_config.max_length){
        //dec not reach max, add
        wrong_num_max_iter ++;
        local_trajID_need_fix[omp_get_thread_num()].insert(i);
      }
      else{
        //dec reach max, need to check distance
        // if (euclideanDistance(t1.back(), t2.back()) >= threshold_max_iter && f_dist){
        if (euclideanDistance(t1.back(), t2.back()) >= threshold_max_iter && frechetDistance(t1, t2) >= threshold){
        //if (ESfrechetDistance(t1, t2) >= threshold){
          wrong_num_max_iter ++;
          local_trajID_need_fix[omp_get_thread_num()].insert(i);
        }
      }
    }
    else {
      num_find_cp ++;
      //reach cp
      if (!cond1 || frechetDistance(t1, t2) >= threshold){
        wrong_num_find_cp ++;
        local_trajID_need_fix[omp_get_thread_num()].insert(i);
      }
    }
  }
  //     break;
  // }

  //汇总local_trajID_need_fix
  for (const auto& local_set:local_trajID_need_fix){
    trajID_need_fix.insert(local_set.begin(), local_set.end());
  }


  auto init_queue_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> init_queue_elapsed = init_queue_end - init_queue_start;
  printf("Time for init queue: %f\n", init_queue_elapsed.count());

  if (trajID_need_fix.size() == 0){
    stop = true;
    printf("No need to fix!\n");
    if (file_dir != ""){
      // printf("writing original and decompressed trajs to file...\n");
      printf("writing original and decompressed trajs to file...\n");
      save_trajs_to_binary(trajs_ori, file_dir + "ori_traj.bin");
      save_trajs_to_binary(trajs_dec, file_dir + "dec_traj.bin");

    }
    exit(0);
  }
  else{
    printf("trajID_need_fix size: %ld\n", trajID_need_fix.size());
    bool write_flag = true;
    if (write_flag && file_dir != ""){
      std::vector<std::vector<std::array<double, 2>>> wrong_trajs_ori;
      std::vector<std::vector<std::array<double, 2>>> wrong_trajs_cpsz;
      for (const auto& trajID:trajID_need_fix){
        wrong_trajs_ori.push_back(trajs_ori[trajID]);
        wrong_trajs_cpsz.push_back(trajs_dec[trajID]);
        fixed_cpsz_trajID.push_back(trajID);
      }
      save_trajs_to_binary(wrong_trajs_ori, file_dir + "wrong_trajs_ori.bin");
      save_trajs_to_binary(wrong_trajs_cpsz, file_dir + "wrong_trajs_cpsz.bin");

    }
    trajID_need_fix_next_vec.push_back(trajID_need_fix.size());
    trajID_need_fix_next_detail_vec.push_back({wrong_num_outside, wrong_num_max_iter, wrong_num_find_cp});
    origin_traj_detail = {num_outside, num_max_iter, num_find_cp};
  }

  auto fix_traj_start = std::chrono::high_resolution_clock::now();
  //exit(0);
  //*************开始修复轨迹*************   
  int current_round = 0;
  do
  {
    if (write_flag && file_dir != "" && (current_round % 2) == 0){
      std::vector<std::vector<std::array<double, 2>>> wrong_trajs_cpsz_intermidiate;
      for (const auto& trajID:trajID_need_fix){
        wrong_trajs_cpsz_intermidiate.push_back(trajs_dec[trajID]);
      }
      // printf("writing original and decompressed trajs to file...\n");
      printf("writing intermidiate wrong trajs to file...\n");
      save_trajs_to_binary(wrong_trajs_cpsz_intermidiate, file_dir + "state_" + std::to_string(current_round) + "wrong_trajs_cpsz.bin");
    }
    if (current_round >=30){
      printf("current_round >= 30, exit\n");
      exit(0);
    }
    printf("begin fix traj,current_round: %d\n", current_round++);
    std::set<size_t> trajID_need_fix_next;
    //fix trajectory
    auto index_time_start = std::chrono::high_resolution_clock::now();
    //convert trajID_need_fix to vector
    std::vector<size_t> trajID_need_fix_vector(trajID_need_fix.begin(), trajID_need_fix.end());
    printf("current iteration size: %ld\n", trajID_need_fix_vector.size());
    
    std::vector<std::set<size_t>> local_all_vertex_for_all_diff_traj(totoal_thread);

    omp_lock_t lock;
    omp_init_lock(&lock);  // 初始化锁
    #pragma omp parallel for schedule(dynamic)
    for (size_t i=0;i<trajID_need_fix_vector.size(); ++i){
      auto current_trajID = trajID_need_fix_vector[i];
      auto ori_t = trajs_ori[current_trajID];
      std::string ori_type = (ori_t.size() == t_config.max_length) ? "max_iter" : (LastTwoPointsAreEqual(ori_t) ? "outside" : "find_cp");
      //printf("processing current_trajID: %ld by thread:%d, ori_type: %s\n", current_trajID, omp_get_thread_num(), ori_type.c_str());
      bool success = false;
      auto& t1 = trajs_ori[current_trajID];
      auto& t2 = trajs_dec[current_trajID];
      int start_fix_index = 0;
      //int end_fix_index = t1.size() - 1;
      int end_fix_index = 1;
      int thread_id = omp_get_thread_num();

      //find the first different point
      int changed = 0;
      //for (size_t j = start_fix_index; j < std::min(t1.size(),t2.size()); ++j){
      for (size_t j = start_fix_index; j < t1.size(); ++j){
      //for (size_t j = start_fix_index; j < max_index; ++j){
        auto& p1 = t1[j];
        auto& p2 = t2[j];
        if (j < t1.size() - 1 && j < t2.size() - 1){
          double dist = sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2));
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

      // if (changed == 0){
      //   if (euclideanDistance(t1.back(), t2.back()) >= threshold){
      //     // end_fix_index = t1.size() - 1;
      //   }
      // }

      // if (t1.size() == t_config.max_length){
      //   end_fix_index = t1.size() / 2; //从中间开始fix
      // }
      // end_fix_index = std::min(end_fix_index, static_cast<int>(t1.size())); //t1.size() - 1;

      do
      {
        double direction = trajID_direction_vector[current_trajID];
        std::set<size_t> temp_vertexID;//临时变量记录经过的vertexID
        std::set<size_t> temp_vertexID_check;
        // end_fix_index = std::min(end_fix_index,t_config.max_length);
        auto temp_trajs_ori = trajectory(t1[0].data(), t1[1], direction * t_config.h, end_fix_index, DH, DW, critical_points_ori, grad_ori,temp_vertexID);
        //omp_set_lock(&lock);
        // #pragma omp critical
        // {
        for (auto o:temp_vertexID){ // 更新dec_U, dec_V
          dec_U[o] = U[o];
          dec_V[o] = V[o];
          //同时更新grad_dec
          //o 转化为坐标
          int x = o % DW;
          int y = o / DW;
          grad_dec(0, x, y) = U[o];
          grad_dec(1, x, y) = V[o];
          local_all_vertex_for_all_diff_traj[thread_id].insert(o);
        }
        // }
        //omp_unset_lock(&lock);
        auto temp_trajs_check = trajectory(t1[0].data(), t1[1], direction * t_config.h, end_fix_index, DH, DW, critical_points_dec, grad_dec,temp_vertexID_check);
        //case 0
        if (LastTwoPointsAreEqual(t1)){
          //ori outside
          if (!LastTwoPointsAreEqual(temp_trajs_check)){
            //dec inside
            success = false;
          }
          else{
            //dec outside
            success = (euclideanDistance(t1.back(), temp_trajs_check.back()) < threshold_outside && frechetDistance(t1, temp_trajs_check) < threshold);
          }
        }
        else if (t1.size() == t_config.max_length){
          //ori reach max
          //这里的判断是｜｜ 不是&&！！
          //success = ((euclideanDistance(t1.back(), temp_trajs_check.back()) < threshold_max_iter) || (temp_trajs_check.size() == t_config.max_length));
          if ((temp_trajs_check.size() == t_config.max_length)){
            // if (euclideanDistance(t1.back(), temp_trajs_check.back()) >= threshold_max_iter || frechetDistance(t1, temp_trajs_check) >= threshold){
            //change:没走到的话，不判断末尾距离，只判断ESfrechet distance
            if (euclideanDistance(t1.back(), temp_trajs_check.back()) >= threshold_max_iter && frechetDistance(t1, temp_trajs_check) >= threshold){
            //if (ESfrechetDistance(t1, temp_trajs_check) >= threshold){
              success = false;
            }
            else{
              success = true;
            }
          }
        }
        else{
          //reach cp
          if (get_cell_offset(t1.back().data(), DW, DH) == get_cell_offset(temp_trajs_check.back().data(), DW, DH) && frechetDistance(t1, temp_trajs_check) < threshold){
            success = true;
          }
        }

        if (end_fix_index >= t1.size()){
          success = true;
        }
        else{
          //printf("INCREASE END_FIX_INDEX!! current_trajID: %ld, end_fix_index: %d, t1.size(): %zu\n", current_trajID, end_fix_index, t1.size());
          end_fix_index = std::min(end_fix_index + static_cast<int>(t_config.next_index_coeff*t_config.max_length),static_cast<int>(t1.size()));
        }
      } while (success == false);
      
      //printf("processed current_trajID: %ld by thread:%d, ori_type: %s, end_fix_index: %d ori_len: %d\n", current_trajID, omp_get_thread_num(), ori_type.c_str(), end_fix_index, t1.size());
    
      // while (!success){
      //   //printf("thread: %d, current_trajID: %ld, current_end_fix_index: %d\n", thread_id, current_trajID, end_fix_index);
      //   //cout << "thread: " << thread_id << ", current_trajID: " << current_trajID << ", current_end_fix_index: " << end_fix_index << endl;
      //   double direction = trajID_direction_vector[current_trajID];
      //   //std::unordered_map<size_t, std::set<int>> temp_cellID_trajIDs_map;//需要临时的一个map统计修正的traj经过那些cellID
      //   std::set<size_t> temp_vertexID;//临时变量记录经过的vertexID
      //   std::set<size_t> temp_vertexID_check;
      //   // std::unordered_map<size_t,double> rollback_dec_u; //记录需要回滚的dec数据
      //   // std::unordered_map<size_t,double> rollback_dec_v;
      //   // std::set<size_t> rollback_vertexID;
      //   //计算一次rk4直到终止点，统计经过的cellID，然后替换数据
        
      //   /* 使用简单的版本*/
      //   auto temp_trajs_ori_time_start = std::chrono::high_resolution_clock::now();
      //   end_fix_index = std::min(end_fix_index,t_config.max_length);
      //   auto temp_trajs_ori = trajectory(t1[0].data(), t1[1], direction * t_config.h, end_fix_index, DH, DW, critical_points_ori, grad_ori,temp_vertexID);
        
      //   //auto temp_trajs_ori_time_end = std::chrono::high_resolution_clock::now();

      //   //此时temp_vertexID记录了从起点到分岔点需要经过的所有vertex
      //   // for (auto o:temp_vertexID){
      //   //   rollback_dec_u[o] = dec_U[o];
      //   //   rollback_dec_v[o] = dec_V[o];
      //   //   local_all_vertex_for_all_diff_traj[thread_id].insert(o);
      //   //   // rollback_vertexID.insert(o);
      //   // }
      //   // auto current_divergence_pos = temp_trajs_ori.back();
      //   //printf("current_divergence_pos ori data(temp_trajs_ori last element): (%f,%f)\n", current_divergence_pos[0], current_divergence_pos[1]);
      //   //printf("current_end_fix_index: %d, t1.size(): %zu, t2.size(): %zu\n", end_fix_index, t1.size(), t2.size());
      //   omp_set_lock(&lock);
      //   for (auto o:temp_vertexID){ // 更新dec_U, dec_V
      //     dec_U[o] = U[o];
      //     dec_V[o] = V[o];
      //     //同时更新grad_dec
      //     //o 转化为坐标
      //     int x = o % DW;
      //     int y = o / DW;
      //     grad_dec(0, x, y) = U[o];
      //     grad_dec(1, x, y) = V[o];
      //     local_all_vertex_for_all_diff_traj[thread_id].insert(o);
      //   }
      //   omp_unset_lock(&lock);
      //   //检查是不是修正成功
      //   /*
      //   1. 这里使用简单的traj计算，避免过度开销
      //   2. 不从0开始，而是从end_fix_index开始，走max_length-end_fix_index+2步
      //   */
      //   //auto temp_debug = trajectory(t2[0].data(), t2[1],direction * t_config.h,end_fix_index, DH, DW, critical_points_dec, grad_dec,temp_index_check);

      //   auto temp_trajs_check = trajectory(t1[0].data(), t1[1], direction * t_config.h, end_fix_index, DH, DW, critical_points_dec, grad_dec,temp_vertexID_check);

      //   // switch (obj)
      //   // {
      //   //   case 0:
      //   if (LastTwoPointsAreEqual(t1)){
      //     //ori outside
      //     if (!LastTwoPointsAreEqual(temp_trajs_check)){
      //       //dec inside
      //       success = false;
      //     }
      //     else{
      //       //dec outside
      //       success = (euclideanDistance(t1.back(), temp_trajs_check.back()) < threshold_outside && frechetDistance(t1, temp_trajs_check) < threshold);
      //     }
      //   }
      //   else if (t1.size() == t_config.max_length){
      //     //ori reach max
      //     //这里的判断是｜｜ 不是&&！！
      //     //success = ((euclideanDistance(t1.back(), temp_trajs_check.back()) < threshold_max_iter) || (temp_trajs_check.size() == t_config.max_length));
      //     if ((temp_trajs_check.size() == t_config.max_length)){
      //       // if (euclideanDistance(t1.back(), temp_trajs_check.back()) >= threshold_max_iter || frechetDistance(t1, temp_trajs_check) >= threshold){
      //       //change:没走到的话，判断末端距离
      //       if (euclideanDistance(t1.back(), temp_trajs_check.back()) >= threshold_max_iter && ESfrechetDistance(t1, temp_trajs_check) >= threshold){
      //         success = false;
      //       }
      //       else{
      //         success = true;
      //       }
      //     }
      //     // if (!success){
      //     //   printf("phase2 (check succc) ID: %ld, t1 size: %ld, t2 size: %ld,distance: %f\n", current_trajID, t1.size(), temp_trajs_check.size(), euclideanDistance(t1.back(), temp_trajs_check.back()));
      //     // }
      //   }
      //   else{
      //     //reach cp
      //     if (get_cell_offset(t1.back().data(), DW, DH) == get_cell_offset(temp_trajs_check.back().data(), DW, DH) && frechetDistance(t1, temp_trajs_check) < threshold){
      //       success = true;
      //     }
      //   }
      //   //     break;
      //   // }
      
      //   if (!success){
      //     //rollback
      //     // for (auto o:temp_vertexID){
      //     //   dec_U[o] = rollback_dec_u[o];
      //     //   dec_V[o] = rollback_dec_v[o];
      //     //   int x = o % DW;
      //     //   int y = o / DW;
      //     //   grad_dec(0, x, y) = dec_U[o];
      //     //   grad_dec(1, x, y) = dec_V[o];
      //     //   local_all_vertex_for_all_diff_traj[thread_id].erase(o);
      //     //   // rollback_vertexID.erase(o);
      //     // }
      //     if (end_fix_index >= static_cast<int>(t1.size()) || end_fix_index >= t_config.max_length){ //(t1.size()-1)
      //       printf("t_config.max_length: %d,end_fix_index%d\n", t_config.max_length, end_fix_index);
      //       printf("error: current end_fix_index is %d, current ID: %ld\n", end_fix_index, current_trajID);
      //       //print all t1,temp_trajs_ori,temp_trajs_check
      //       // for (size_t i = 0; i < t1.size(); ++i){
      //       //   printf("t1: (%f,%f), temp_trajs_ori: (%f,%f), temp_trajs_check: (%f,%f)\n", t1[i][0], t1[i][1], temp_trajs_ori[i][0], temp_trajs_ori[i][1], temp_trajs_check[i][0], temp_trajs_check[i][1]);
      //       // }
      //       printf("ori first: (%f,%f), temp_trajs_ori first: (%f,%f), temp_trajs_check first: (%f,%f)\n", t1[0][0], t1[0][1],temp_trajs_ori[0][0], temp_trajs_ori[0][1],temp_trajs_check[0][0], temp_trajs_check[0][1]);
      //       printf("ori second: (%f,%f), temp_trajs_ori second: (%f,%f), temp_trajs_check second: (%f,%f)\n", t1[1][0], t1[1][1],temp_trajs_ori[1][0], temp_trajs_ori[1][1],temp_trajs_check[1][0], temp_trajs_check[1][1]);
      //       printf("ori last-2 : (%f,%f), temp_trajs_ori last-2: (%f,%f), temp_trajs_check last-2: (%f,%f)\n", t1[t1.size()-3][0], t1[t1.size()-3][1],temp_trajs_ori[temp_trajs_ori.size()-3][0], temp_trajs_ori[temp_trajs_ori.size()-3][1],temp_trajs_check[temp_trajs_check.size()-3][0], temp_trajs_check[temp_trajs_check.size()-3][1]);
      //       printf("ori last-1 : (%f,%f), temp_trajs_ori last-1: (%f,%f), temp_trajs_check last-1: (%f,%f)\n", t1[t1.size()-2][0], t1[t1.size()-2][1],temp_trajs_ori[temp_trajs_ori.size()-2][0], temp_trajs_ori[temp_trajs_ori.size()-2][1],temp_trajs_check[temp_trajs_check.size()-2][0], temp_trajs_check[temp_trajs_check.size()-2][1]);
      //       printf("ori last : (%f,%f), temp_trajs_ori last: (%f,%f), temp_trajs_check last: (%f,%f)\n", t1.back()[0], t1.back()[1],temp_trajs_ori.back()[0], temp_trajs_ori.back()[1],temp_trajs_check.back()[0], temp_trajs_check.back()[1]);
      //       printf("t1 size %zu, temp_trajs_ori size: %zu, temp_trajs_check size: %zu\n", t1.size(), temp_trajs_ori.size(), temp_trajs_check.size());
      //       //printf("check temp_trajs_check last is cp: %d\n",check_pt_is_cp(temp_trajs_check.back(),critical_points_dec));//(0.001559,442.715650)
      //       //trajID_need_fix_next.insert(current_trajID);
      //       //trajs_dec[current_trajID] = temp_trajs_check;
      //       success = true;  
      //     }
      //     // end_fix_index = std::min(end_fix_index + 10,static_cast<int>(t1.size()));
      //     else{
      //       end_fix_index = std::min(end_fix_index + static_cast<int>(0.005*t_config.max_length),static_cast<int>(t1.size())); //t1.size() - 1
      //     } 
      //   }
      //   else{
      //     //修正成功当前trajectory
      //     //printf("fix traj %zu successfully\n",current_trajID);
      //     //更新trajs_dec
      //     trajs_dec[current_trajID] = temp_trajs_check;
      //     // local_all_vertex_for_all_diff_traj[thread_id].insert(rollback_vertexID.begin(), rollback_vertexID.end()); 
      //     success = true;     
      //   }
      // }
      // printf("current_trajID: %ld fixed successfully\n", current_trajID);
    } 
    printf("all threads have finished fixing traj\n");
    omp_destroy_lock(&lock);

    //汇总all_vertex_for_all_diff_traj
    for (const auto& local_set:local_all_vertex_for_all_diff_traj){
      all_vertex_for_all_diff_traj.insert(local_set.begin(), local_set.end());
    }
    printf("all_vertex_for_all_diff_traj size: %ld\n", all_vertex_for_all_diff_traj.size());
    
    
    auto index_time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> index_time_elapsed = index_time_end - index_time_start;
    index_time_vec.push_back(index_time_elapsed.count());
    //此时dec_u,v 已经更新，重新计算所有的trajectory
    printf("recalculating trajectory for decompressed data...\n");
    //check if need to fix next round
    //get trajectory for decompressed data
    auto recal_trajs_start = std::chrono::high_resolution_clock::now();

    //用原来的trajs_dec 替换
    //set trajs_dec to empty
    trajs_dec.resize(keys_dec.size() * 4);
    for (auto& traj : trajs_dec) {
        traj.resize(expected_size,{0.0,0.0});
    }

    // std::vector<std::vector<std::array<double, 2>>> trajs_dec_next(keys.size() * 4);
    // for (auto& traj : trajs_dec_next) {
    //     traj.reserve(expected_size); // 预分配容量
    // }
    std::vector<std::set<size_t>> thread_index_dec_next(totoal_thread);
    std::set<size_t> vertex_dec_next;
    #pragma omp parallel for
    // for(const auto& p:critical_points_dec){
    for(size_t j = 0; j < keys_dec.size(); ++j){
      auto key = keys_dec[j];
      auto &cp = critical_points_dec[key];
      int thread_id = omp_get_thread_num();
      // auto cp = p.second;
      auto eigvec = cp.eig_vec;
      auto eigval = cp.eig;
      auto pt = cp.x;
      std::array<std::array<double, 3>, 4> directions;//first is direction(1 or -1), next 2 are seed point
      for (int i = 0; i < 2; i++){
          if (eigval[i].real() > 0){
            directions[i][0] = 1;
            directions[i][1] = t_config.eps * eigvec[i][0] + pt[0];
            directions[i][2] = t_config.eps * eigvec[i][1] + pt[1];
            directions[i+2][0] = 1;
            directions[i+2][1] = -1 * t_config.eps * eigvec[i][0] + pt[0];
            directions[i+2][2] = -1 * t_config.eps * eigvec[i][1] + pt[1];
          }
          else{
            directions[i][0] = -1;
            directions[i][1] = t_config.eps * eigvec[i][0] + pt[0];
            directions[i][2] = t_config.eps * eigvec[i][1] + pt[1];
            directions[i+2][0] = -1;
            directions[i+2][1] = -1 * t_config.eps * eigvec[i][0] + pt[0];
            directions[i+2][2] = -1 * t_config.eps * eigvec[i][1] + pt[1];
          }
      }

      for (int k = 0; k < 4; k++) {
        std::array<double, 2> seed = {directions[k][1], directions[k][2]};  
        std::vector<std::array<double, 2>> result_return;
        double pos[2] = {cp.x[0], cp.x[1]};
        // result_return = trajectory(pos, seed, directions[i][0] * t_config.h, t_config.max_length, DH, DW, critical_points_dec, grad_dec, index_dec_next);
        // trajs_dec_next.push_back(result_return); 
        result_return = trajectory_parallel(pt, seed, directions[k][0] * t_config.h, t_config.max_length, DH, DW, critical_points_dec, grad_dec,thread_index_dec_next,thread_id);
        trajs_dec[j *4 + k] = result_return;
      }
    
    }
  
    auto recal_trajs_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_recal = recal_trajs_end - recal_trajs_start;
    re_cal_trajs_time_vec.push_back(elapsed_recal.count());
    //compare the trajectory
    printf("comparing.... double check if all trajectories are fixed...,total trajectory: %ld\n",trajs_ori.size());
    // size_t outside_count = 0;
    // size_t hit_max_iter = 0;
    size_t wrong = 0;
    // size_t correct = 0;
    int wrong_num_outside = 0;
    int wrong_num_max_iter = 0;
    int wrong_num_find_cp = 0;
    //这个for现在也要并行了，因为frechetDistance算的慢
    std::vector<std::set<size_t>> local_trajID_need_fix_next(totoal_thread);
    auto comp_start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for reduction(+:wrong_num_outside, wrong_num_max_iter, wrong_num_find_cp,wrong)
    for(size_t i=0;i < trajs_ori.size(); ++i){
      auto t1 = trajs_ori[i];
      auto t2 = trajs_dec[i];
      bool cond1 = get_cell_offset(t1.back().data(), DW, DH) == get_cell_offset(t2.back().data(), DW, DH);
      bool cond2 = (t1.size() == t_config.max_length);
      bool cond3 = t2.size() == t_config.max_length;
      // bool f_dis = frechetDistance(t1, t2) >= threshold;
      // bool cond4 = t1.back()[0] == -1;
      // bool cond5 = t2.back()[0] == -1;
      // switch (obj)
      // {
      // case 0:
      if (LastTwoPointsAreEqual(t1)){
        if (!LastTwoPointsAreEqual(t2)){
          wrong ++;
          wrong_num_outside ++;
          local_trajID_need_fix_next[omp_get_thread_num()].insert(i);
        }
        else{
          if (euclideanDistance(t1.back(), t2.back()) > threshold_outside || frechetDistance(t1, t2) >= threshold){
            wrong ++;
            wrong_num_outside ++;
            local_trajID_need_fix_next[omp_get_thread_num()].insert(i);
          }
        }
      }
      else if (cond2){
        // ori reach max
        //这里的判断是｜｜ 不是&&！！
        // if ((euclideanDistance(t1.back(), t2.back()) > threshold_max_iter) || (t2.size() != t_config.max_length)){
        if (t2.size() != t_config.max_length){
          //dec not reach max, add
          wrong ++;
          wrong_num_max_iter ++;
          local_trajID_need_fix_next[omp_get_thread_num()].insert(i);
        }
        else{
          //dec reach max, need to check distance
          //change:没走到的话，判断末端距离+edr
          // if (euclideanDistance(t1.back(), t2.back()) >= threshold_max_iter || f_dis){
          if (euclideanDistance(t1.back(), t2.back()) >= threshold_max_iter || frechetDistance(t1, t2) >= threshold){
          //if (ESfrechetDistance(t1, t2) >= threshold){
            wrong ++;
            wrong_num_max_iter ++;
            local_trajID_need_fix_next[omp_get_thread_num()].insert(i);
          }
        }
      }
      else{
        num_find_cp ++;
        if (!cond1 || frechetDistance(t1, t2) >= threshold){
          wrong ++;
          wrong_num_find_cp ++;
          local_trajID_need_fix_next[omp_get_thread_num()].insert(i);
        }
      }
      //   break;
      // }

    }
    //汇总local_trajID_need_fix_next
    for (const auto& local_set:local_trajID_need_fix_next){
      trajID_need_fix_next.insert(local_set.begin(), local_set.end());
    }
    printf("wrong: %ld\n", wrong);
    auto comp_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> comp_elapsed = comp_end - comp_start;
    compare_time_vec.push_back(comp_elapsed.count());

    if(trajID_need_fix_next.size() == 0){
      stop = true;
      printf("All trajectories are fixed!\n");
    }
    else{
      //printf("trajID_need_fix_next size: %ld\n", trajID_need_fix_next.size());
      trajID_need_fix_next_vec.push_back(trajID_need_fix_next.size());
      trajID_need_fix_next_detail_vec.push_back({wrong_num_outside, wrong_num_max_iter, wrong_num_find_cp});
      //清空trajID_need_fix，然后把trajID_need_fix_next赋值给trajID_need_fix
      trajID_need_fix.clear();
      //printf("before change trajID_need_fix size(should be 0): %ld\n", trajID_need_fix.size());
      for(auto o:trajID_need_fix_next){
        trajID_need_fix.insert(o);
      }


      trajID_need_fix_next.clear();
    }
    

  } while (stop == false);
  
  auto fix_traj_end = std::chrono::high_resolution_clock::now(); 
  std::chrono::duration<double> fix_traj_elapsed = fix_traj_end - fix_traj_start;
  printf("fix_traj time: %f\n", fix_traj_elapsed.count());
  auto total_time_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_fix = total_time_end - total_time_start;
  printf("total round: %d\n", current_round);
  printf("Total Time(exclude cpsz time & preparation): %f\n", elapsed_fix.count());
  //print each time in comp_time_vec
  // printf("traj_ori_begain time: %f\n", traj_ori_begin_elapsed.count());
  // printf("traj_dec_begin time: %f\n", traj_dec_begin_elapsed.count()); 
  printf("traj_begin(ori+dec) time: %f\n", (traj_ori_begin_elapsed.count() + traj_dec_begin_elapsed.count()));
  printf("compare & init_queue time: %f\n", init_queue_elapsed.count());
  // for (auto t:index_time_vec){
  //   printf("index_time: %f\n", t);
  // }
  printf("sum of index_time: %f\n", std::accumulate(index_time_vec.begin(), index_time_vec.end(), 0.0));
  // for (auto t:re_cal_trajs_time_vec){
  //   printf("re_cal_trajs_time: %f\n", t);
  // }
  printf("sum of re_cal_trajs_time: %f\n", std::accumulate(re_cal_trajs_time_vec.begin(), re_cal_trajs_time_vec.end(), 0.0));
  // for (auto t:compare_time_vec){
  //   printf("compare & update_queue_time: %f\n", t);
  // }
  printf("sum of compare & update_queue_time: %f\n", std::accumulate(compare_time_vec.begin(), compare_time_vec.end(), 0.0));

  printf("origin_traj_detail: outside: %d, max_iter: %d, find_cp: %d\n", origin_traj_detail[0], origin_traj_detail[1], origin_traj_detail[2]);
  for (auto t:trajID_need_fix_next_vec){
    printf("trajID_need_fix_next: %d\n", t);
  }
  for(auto t:trajID_need_fix_next_detail_vec){
    printf("trajID_need_fix_next_detail: wrong outside: %d, wrong max_iter: %d, wrong find_cp: %d\n", t[0], t[1], t[2]);
  }


  //so far, all trajectories should be fixed
  printf("all_vertex_for_all_diff_traj size: %ld\n", all_vertex_for_all_diff_traj.size());

  //check compression ratio
  auto tpsz_comp_time_start = std::chrono::high_resolution_clock::now();
  unsigned char * final_result = NULL;
  size_t final_result_size = 0;
  if(eb_type == "rel"){
    if (CPSZ_OMP_FLAG == 0){
      final_result = sz_compress_cp_preserve_2d_record_vertex(U, V, r1, r2, final_result_size, false, max_pwr_eb, all_vertex_for_all_diff_traj);
    }
    else{
      float * dec_inplace_U = NULL;
      float * dec_inplace_V = NULL;
      final_result = omp_sz_compress_cp_preserve_2d_record_vertex(U, V, r1, r2, final_result_size, false, max_pwr_eb, all_vertex_for_all_diff_traj,totoal_thread,dec_inplace_U,dec_inplace_V,eb_type);
      free(dec_inplace_U);
      free(dec_inplace_V);
    }
  

  }
  else if(eb_type == "abs"){
    if (CPSZ_OMP_FLAG == 0){
    final_result = sz_compress_cp_preserve_2d_online_abs_record_vertex(U, V, r1, r2, final_result_size, true, max_pwr_eb, all_vertex_for_all_diff_traj);
    final_result = sz_compress_cp_preserve_2d_online_abs_record_vertex(U, V, r1, r2, final_result_size, true, max_pwr_eb, all_vertex_for_all_diff_traj);
      final_result = sz_compress_cp_preserve_2d_online_abs_record_vertex(U, V, r1, r2, final_result_size, true, max_pwr_eb, all_vertex_for_all_diff_traj);
    }
    else{
      float * dec_inplace_U = NULL;
      float * dec_inplace_V = NULL;
      final_result = omp_sz_compress_cp_preserve_2d_record_vertex(U, V, r1, r2, final_result_size, true, max_pwr_eb, all_vertex_for_all_diff_traj,totoal_thread,dec_inplace_U,dec_inplace_V,eb_type);
      free(dec_inplace_U);
      free(dec_inplace_V);
    }
    
  }
  auto tpsz_comp_time_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> tpsz_comp_duration = tpsz_comp_time_end - tpsz_comp_time_start;

  printf("checkpt1\n");
  unsigned char * result_after_zstd = NULL;
  size_t result_after_zstd_size = sz_lossless_compress(ZSTD_COMPRESSOR, 3, final_result, final_result_size, &result_after_zstd);


  free(final_result);
  auto tpsz_decomp_start = std::chrono::high_resolution_clock::now();
  size_t zstd_decompressed_size = sz_lossless_decompress(ZSTD_COMPRESSOR, result_after_zstd, result_after_zstd_size, &final_result, final_result_size);
  //printf("final lossless output size %zu, final_result_size %zu\n",final_lossless_output,final_result_size);//should be same with cpsz出来的大小
  // printf("checkpt3\n");
  // free(final_result);
  float * final_dec_U = NULL;
  float * final_dec_V = NULL;
  if (CPSZ_OMP_FLAG == 0){
    sz_decompress_cp_preserve_2d_online_record_vertex<float>(final_result, r1, r2, final_dec_U, final_dec_V);
  }
  else{
    omp_sz_decompress_cp_preserve_2d_online(final_result, r1, r2, final_dec_U, final_dec_V);
  }
  
  auto tpsz_decomp_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> tpsz_decomp_duration = tpsz_decomp_end - tpsz_decomp_start;

  printf("verifying...\n");
 
  verify(U, final_dec_U, r1*r2,nrmse_u);
  verify(V, final_dec_V, r1*r2,nrmse_v);

  psnr_overall = 20 * log10(sqrt(2) / sqrt(nrmse_u*nrmse_u + nrmse_v*nrmse_v));
  //printf("nrmse_u: %f, nrmse_v: %f, psnr_overall: %f\n", nrmse_u, nrmse_v, psnr_overall);
  printf("====================================\n");
  printf("original trajs detail: outside: %d, max_iter: %d, find_cp: %d\n", origin_traj_detail[0], origin_traj_detail[1], origin_traj_detail[2]);
  printf("BEGIN Compression ratio = %f\n", cr_ori);
  printf("psnr_overall_cpsz: %f\n",psnr_cpsz_overall);
  printf("FINAL Compressed ratio = %f\n",(2*r1*r2*sizeof(float)) * 1.0/result_after_zstd_size);
  printf("psnr_overall: %f\n", psnr_overall);

  printf("comp time cpsz: %f\n", cpsz_comp_duration.count());
  printf("decomp time cpsz: %f\n", cpsz_decomp_duration.count());
  printf("fix time tpsz: %f\n",traj_dec_begin_elapsed.count() + traj_ori_begin_elapsed.count() + init_queue_elapsed.count() + fix_traj_elapsed.count());
  printf("comp Total time tpsz: %f\n",traj_dec_begin_elapsed.count() + traj_ori_begin_elapsed.count() + init_queue_elapsed.count() + fix_traj_elapsed.count() + tpsz_comp_duration.count());
  printf("decomp time tpsz: %f\n", tpsz_decomp_duration.count());

  
  printf("%d\n",current_round);
  for (int i = 0; i < trajID_need_fix_next_vec.size(); ++i){
    printf("trajID_need_fix_next: %d, wrong outside: %d, wrong max_iter: %d, wrong find_cp: %d\n", trajID_need_fix_next_vec[i], trajID_need_fix_next_detail_vec[i][0], trajID_need_fix_next_detail_vec[i][1], trajID_need_fix_next_detail_vec[i][2]);
  }
  // for (auto t:trajID_need_fix_next_vec){
  //   printf("trajID_need_fix_next: %d\n", t);
  // }
  // for(auto t:trajID_need_fix_next_detail_vec){
  //   printf("trajID_need_fix_next_detail: wrong outside: %d, wrong max_iter: %d, wrong find_cp: %d\n", t[0], t[1], t[2]);
  // }
  
  // printf("%f\n",(traj_ori_begin_elapsed.count() + traj_dec_begin_elapsed.count()));
  // printf("%f\n",init_queue_elapsed.count());
  // printf("%f\n",std::accumulate(index_time_vec.begin(), index_time_vec.end(), 0.0));
  // printf("%f\n",std::accumulate(re_cal_trajs_time_vec.begin(), re_cal_trajs_time_vec.end(), 0.0));
  // printf("%f\n",std::accumulate(compare_time_vec.begin(), compare_time_vec.end(), 0.0));
  // printf("%f\n",result_after_zstd_size, (2*r1*r2*sizeof(float)) * 1.0/result_after_zstd_size);
  printf("====================================\n");
  printf("# of all_vertex_for_all_diff_traj: %ld\n", all_vertex_for_all_diff_traj.size());
  
  //检查all_vertex_for_all_diff_traj对应的点是否一致
  for (auto o:all_vertex_for_all_diff_traj){
    if (U[o] != final_dec_U[o] || V[o] != final_dec_V[o]){
      printf("vertex %ld is diff\n", o);
      printf("U: %f, final_dec_U: %f\n", U[o], final_dec_U[o]);
      printf("V: %f, final_dec_V: %f\n", V[o], final_dec_V[o]);
      exit(0);
    }
  }

  printf("checkpt4\n");
  //now check if all trajectories are correct
  ftk::ndarray<float> grad_final;
  grad_final.reshape({2, static_cast<unsigned long>(r2), static_cast<unsigned long>(r1)});
  refill_gradient(0, r1, r2, final_dec_U, grad_final);
  refill_gradient(1, r1, r2, final_dec_V, grad_final);
  auto critical_points_final =compute_critical_points(final_dec_U, final_dec_V, r1, r2);
  
  //check if all critical points are exactly preserved
  // for (auto cp:critical_points_final){
  //   auto ID = cp.first;
  //   auto coord = cp.second.x;
  //   for (auto cp_ori:critical_points_ori){
  //     if (cp_ori.first == ID){
  //       if (cp_ori.second.x[0] != coord[0] || cp_ori.second.x[1] != coord[1] || cp_ori.second.type != cp.second.type){
  //         printf("critical point %ld is diff\n", ID);
  //         printf("ori: (%f,%f), dec: (%f,%f)\n", cp_ori.second.x[0], cp_ori.second.x[1], coord[0], coord[1]);
  //         exit(0);
  //       }
  //     }
  //   }
  // }

 // 并行check************************************************

  std::vector<size_t> keys_final_check;
  for (const auto&p : critical_points_final){
    if (p.second.type == SADDLE){
      keys_final_check.push_back(p.first);
    }
  }
  printf("keys size(# of saddle): %ld\n", keys.size());
  std::vector<std::vector<std::array<double, 2>>> trajs_final_check(keys.size() * 4);//指定长度为saddle的个数*4，因为每个saddle有4个方向
  printf("trajs_ori size: %ld\n", trajs_final_check.size());
  // /*这里一定要加上去，不然由于动态扩容会产生额外开销*/
  for (auto& traj : trajs_final_check) {
      // traj.reserve(expected_size,{0.0, 0.0}); // 预分配容量
      traj.resize(expected_size, {0.0, 0.0});
  }

  //std::atomic<size_t> trajID_counter(0);
  std::vector<std::set<size_t>> thread_lossless_index_final_check(totoal_thread);
  omp_set_num_threads(totoal_thread);
  #pragma omp parallel for
  // for(const auto& p:critical_points_ori){
  for(size_t j = 0; j < keys.size(); ++j){
    auto key = keys[j];
    auto &cp = critical_points_final[key];
    //auto cp = p.second;
    if (cp.type != SADDLE){
      printf("not saddle point\n");
      exit(0);
    }
    int thread_id = omp_get_thread_num();
    //printf("key: %ld, thread: %d\n", key, thread_id);
    ori_saddle_count ++;
    auto eigvec = cp.eig_vec;
    auto eigval = cp.eig;
    auto pt = cp.x;
    std::array<std::array<double, 3>, 4> directions;//first is direction(1 or -1), next 2 are seed point
    for (int i = 0; i < 2; ++i){
            if (eigval[i].real() > 0){
              directions[i][0] = 1;
              directions[i][1] = t_config.eps * eigvec[i][0] + pt[0];
              directions[i][2] = t_config.eps * eigvec[i][1] + pt[1];
              directions[i+2][0] = 1;
              directions[i+2][1] = -1 * t_config.eps * eigvec[i][0] + pt[0];
              directions[i+2][2] = -1 * t_config.eps * eigvec[i][1] + pt[1];
            }
            else{
              directions[i][0] = -1;
              directions[i][1] = t_config.eps * eigvec[i][0] + pt[0];
              directions[i][2] = t_config.eps * eigvec[i][1] + pt[1];
              directions[i+2][0] = -1;
              directions[i+2][1] = -1 * t_config.eps * eigvec[i][0] + pt[0];
              directions[i+2][2] = -1 * t_config.eps * eigvec[i][1] + pt[1];
            }
    }


    for (int k = 0; k < 4; ++k) {
      std::vector<std::array<double, 2>> result_return;
      //check if inside
      std::array<double, 2> seed = {directions[k][1], directions[k][2]};
      //size_t current_traj_index = trajID_counter++;
      result_return = trajectory_parallel(pt, seed, directions[k][0] * t_config.h, t_config.max_length, DH, DW, critical_points_final, grad_final,thread_lossless_index_final_check,thread_id);
      trajs_final_check[j *4 + k] = result_return;

    }
  }

  //if file_dir is empty, then not write
  if (file_dir != ""){
  // printf("writing original and decompressed trajs to file...\n");
    save_trajs_to_binary(trajs_ori, file_dir + "ori_traj.bin");
    save_trajs_to_binary(trajs_final_check, file_dir + "dec_traj.bin");
    writefile((file_dir + "decU.bin").c_str(), final_dec_U, r1*r2);
    writefile((file_dir + "decV.bin").c_str(), final_dec_V, r1*r2);
    std::vector<std::vector<std::array<double, 2>>> fixed_cpsz_trajs;
    for (auto o:fixed_cpsz_trajID){
      fixed_cpsz_trajs.push_back(trajs_dec[o]);
    }
    save_trajs_to_binary(fixed_cpsz_trajs, file_dir + "fixed_cpsz_traj.bin");
  }


  //check if all trajectories are fixed
  // 这里由于改了frechetDistance的计算方式，所以check的时候也要计算frechetDistance，所以很慢，所以要并行
  size_t terminate = 0;
  #pragma omp parallel for reduction(+:terminate)
  for (size_t i = 0; i < trajs_final_check.size(); ++i){
    auto t1 = trajs_ori[i];
    auto t2 = trajs_final_check[i];
    if (LastTwoPointsAreEqual(t1)){
      //ori outside
      if (euclideanDistance(t1.back(),t2.back()) > threshold_outside){
        printf("some trajectories not fixed(case0-0)\n");
        printf("ori length: %zu, dec length: %zu\n", t1.size(), t2.size());
        printf("ori first %f,%f, dec first %f,%f\n", t1[0][0], t1[0][1], t2[0][0], t2[0][1]);
        printf("ori last-1 %f,%f, dec last-1 %f,%f\n", t1[t1.size()-2][0], t1[t1.size()-2][1], t2[t2.size()-2][0], t2[t2.size()-2][1]);
        printf("ori last %f,%f, dec last %f,%f\n", t1.back()[0], t1.back()[1], t2.back()[0], t2.back()[1]);
        terminate++;
      }
    }
    else if (t1.size() == t_config.max_length){
      //ori reach max, dec should satisfy distance
      //change: 没走到的话，不判断末尾距离，而是判断ESfrechet distance
      if(euclideanDistance(t1.back(),t2.back()) > threshold_max_iter){
      //if (ESfrechetDistance(t1, t2) >= threshold){
        printf("some trajectories not fixed(case0-1)\n");
        printf("trajID: %ld\n", i);
        printf("ori length: %zu, dec length: %zu\n", t1.size(), t2.size());
        printf("ori first %f,%f, dec first %f,%f\n", t1[0][0], t1[0][1], t2[0][0], t2[0][1]);
        printf("ori second %f,%f, dec second %f,%f\n", t1[1][0], t1[1][1], t2[1][0], t2[1][1]);
        printf("ori last-1 %f,%f, dec last-1 %f,%f\n", t1[t1.size()-2][0], t1[t1.size()-2][1], t2[t2.size()-2][0], t2[t2.size()-2][1]);
        printf("ori last %f,%f, dec last %f,%f,dist: %f\n", t1.back()[0], t1.back()[1], t2.back()[0], t2.back()[1], euclideanDistance(t1.back(),t2.back()));
        terminate++;
        //print all t1,t2 when different
        for (int i = 0; i < t_config.max_length; ++i){
          if (t1[i][0] != t2[i][0] || t1[i][1] != t2[i][1]){
            printf("different start from indx: %d, ori: (%f,%f), dec: (%f,%f)\n", i, t1[i][0], t1[i][1], t2[i][0], t2[i][1]);
            // break;
            exit(0);
          }
        }

      }
    }
    else{
      //ori inside, not reach max, ie found cp
      if (get_cell_offset(t1.back().data(), DW, DH) != get_cell_offset(t2.back().data(), DW, DH)){
        printf("some trajectories not fixed(case0-2)\n");
        printf("ori length: %zu, dec length: %zu\n", t1.size(), t2.size());
        printf("ori first %f,%f, dec first %f,%f\n", t1[0][0], t1[0][1], t2[0][0], t2[0][1]);
        printf("ori last-1 %f,%f, dec last-1 %f,%f\n", t1[t1.size()-2][0], t1[t1.size()-2][1], t2[t2.size()-2][0], t2[t2.size()-2][1]);
        printf("ori last %f,%f, dec last %f,%f\n", t1.back()[0], t1.back()[1], t2.back()[0], t2.back()[1]);
        terminate++;
      }
    }
  }
  if (terminate > 0){
    printf("some trajectories not fixed\n");
    exit(0);
  }
  else{
    printf("all passed!\n");
    printf("all_vertex_for_all_diff_traj size: %ld\n", all_vertex_for_all_diff_traj.size());
  }
  //把all_vertex_for_all_diff_traj这个set里的东西写出来
  std::vector<size_t> temp_vec(all_vertex_for_all_diff_traj.begin(), all_vertex_for_all_diff_traj.end());
  if(file_dir != ""){
    writefile((file_dir + "all_vertex_for_all_diff_traj.bin").c_str(), temp_vec.data(), temp_vec.size());
    printf("write all_vertex_for_all_diff_traj to file sussfully\n");
  }

  
  // exit(0);



  // std::vector<std::vector<std::array<double, 2>>> final_check_ori;
  // std::vector<std::vector<std::array<double, 2>>> final_check_dec;
  // std::vector<int> index_tmp;
  // std::vector<int> index_final;
  // std::set<size_t> vertex_final;
  // for(const auto& p:critical_points_final){
  //   auto cp = p.second;
  //   if (cp.type == SADDLE){
  //     auto eigvec = cp.eig_vec;
  //     auto eigval = cp.eig;
  //     auto pt = cp.x;
  //     std::array<std::array<double, 3>, 4> directions;//first is direction(1 or -1), next 2 are seed point
  //     for (int i = 0; i < 2; i++){
  //         if (eigval[i].real() > 0){
  //           directions[i][0] = 1;
  //           directions[i][1] = t_config.eps * eigvec[i][0] + pt[0];
  //           directions[i][2] = t_config.eps * eigvec[i][1] + pt[1];
  //           directions[i+2][0] = 1;
  //           directions[i+2][1] = -1 * t_config.eps * eigvec[i][0] + pt[0];
  //           directions[i+2][2] = -1 * t_config.eps * eigvec[i][1] + pt[1];
  //         }
  //         else{
  //           directions[i][0] = -1;
  //           directions[i][1] = t_config.eps * eigvec[i][0] + pt[0];
  //           directions[i][2] = t_config.eps * eigvec[i][1] + pt[1];
  //           directions[i+2][0] = -1;
  //           directions[i+2][1] = -1 * t_config.eps * eigvec[i][0] + pt[0];
  //           directions[i+2][2] = -1 * t_config.eps * eigvec[i][1] + pt[1];
  //         }
  //     }

  //     for (int i = 0; i < 4; i ++) {
  //       std::vector<std::array<double, 2>> result_return;
  //       //check if inside
  //       std::array<double, 2> seed = {directions[i][1], directions[i][2]};
  //       double pos[2] = {cp.x[0], cp.x[1]};
  //       std::set<size_t> temp1,temp2;
  //       std::vector<std::array<double, 2>> result_return_ori = trajectory(pos, seed, directions[i][0] * t_config.h, t_config.max_length, DH, DW, critical_points_final, grad_ori, index_tmp,temp1);
  //       std::vector<std::array<double, 2>> result_return_dec = trajectory(pos, seed, directions[i][0] * t_config.h, t_config.max_length, DH, DW, critical_points_final, grad_final, index_final,temp2);
  //       final_check_ori.push_back(result_return_ori);
  //       final_check_dec.push_back(result_return_dec);
  //       bool terminate = false;
  //       switch (obj)
  //       {
  //       case 0:
  //         // if (LastTwoPointsAreEqual(result_return_ori)){
  //         //   //outside
  //         //   if(!SearchElementFromBack(result_return_dec, result_return_ori.back())){
  //         //     printf("some trajectories not fixed(case0-0)\n");
  //         //     printf("ori length: %zu, dec length: %zu\n", result_return_ori.size(), result_return_dec.size());
  //         //     printf("ori first %f,%f, dec first %f,%f\n", result_return_ori[0][0], result_return_ori[0][1], result_return_dec[0][0], result_return_dec[0][1]);
  //         //     printf("ori last-1 %f,%f, dec last-1 %f,%f\n", result_return_ori[result_return_ori.size()-2][0], result_return_ori[result_return_ori.size()-2][1], result_return_dec[result_return_dec.size()-2][0], result_return_dec[result_return_dec.size()-2][1]);
  //         //     printf("ori last %f,%f, dec last %f,%f\n", result_return_ori.back()[0], result_return_ori.back()[1], result_return_dec.back()[0], result_return_dec.back()[1]);
  //         //     exit(0);
  //         //   }
  //         // }
  //         if (LastTwoPointsAreEqual(result_return_ori)){
  //           if (euclideanDistance(result_return_ori.back(),result_return_dec.back()) > threshold_outside){
  //             printf("some trajectories not fixed(case0-0)\n");
  //             printf("ori length: %zu, dec length: %zu\n", result_return_ori.size(), result_return_dec.size());
  //             printf("ori first %f,%f, dec first %f,%f\n", result_return_ori[0][0], result_return_ori[0][1], result_return_dec[0][0], result_return_dec[0][1]);
  //             printf("ori last-1 %f,%f, dec last-1 %f,%f\n", result_return_ori[result_return_ori.size()-2][0], result_return_ori[result_return_ori.size()-2][1], result_return_dec[result_return_dec.size()-2][0], result_return_dec[result_return_dec.size()-2][1]);
  //             printf("ori last %f,%f, dec last %f,%f\n", result_return_ori.back()[0], result_return_ori.back()[1], result_return_dec.back()[0], result_return_dec.back()[1]);
  //             terminate = true;
  //           }
  //         }
  //         else if (result_return_ori.size() == t_config.max_length){
  //           if (euclideanDistance(result_return_ori.back(),result_return_dec.back()) > threshold_max_iter){
  //             printf("some trajectories not fixed(case0-1)\n");
  //             printf("ori length: %zu, dec length: %zu\n", result_return_ori.size(), result_return_dec.size());
  //             printf("ori first %f,%f, dec first %f,%f\n", result_return_ori[0][0], result_return_ori[0][1], result_return_dec[0][0], result_return_dec[0][1]);
  //             printf("ori last-1 %f,%f, dec last-1 %f,%f\n", result_return_ori[result_return_ori.size()-2][0], result_return_ori[result_return_ori.size()-2][1], result_return_dec[result_return_dec.size()-2][0], result_return_dec[result_return_dec.size()-2][1]);
  //             printf("ori last %f,%f, dec last %f,%f\n", result_return_ori.back()[0], result_return_ori.back()[1], result_return_dec.back()[0], result_return_dec.back()[1]);
  //             printf("distance: %f\n", euclideanDistance(result_return_ori.back(),result_return_dec.back()));
  //             //print all result_return_ori, result_return_dec
  //             for (size_t i = 0; i < result_return_ori.size(); ++i){
  //               printf("ori: (%f,%f), dec: (%f,%f)\n", result_return_ori[i][0], result_return_ori[i][1], result_return_dec[i][0], result_return_dec[i][1]);
  //             }
  //             printf("====================================\n");
  //             terminate = true;
  //           }
  //         }
  //         else{
  //           //inside and not reach max, ie found cp
  //           if(get_cell_offset(result_return_ori.back().data(), DW, DH) != get_cell_offset(result_return_dec.back().data(), DW, DH)){
  //             printf("some trajectories not fixed(case0-2)\n");
  //             printf("ori length: %zu, dec length: %zu\n", result_return_ori.size(), result_return_dec.size());
  //             printf("ori first %f,%f, dec first %f,%f\n", result_return_ori[0][0], result_return_ori[0][1], result_return_dec[0][0], result_return_dec[0][1]);
  //             printf("ori last-1 %f,%f, dec last-1 %f,%f\n", result_return_ori[result_return_ori.size()-2][0], result_return_ori[result_return_ori.size()-2][1], result_return_dec[result_return_dec.size()-2][0], result_return_dec[result_return_dec.size()-2][1]);
  //             printf("ori last %f,%f, dec last %f,%f\n", result_return_ori.back()[0], result_return_ori.back()[1], result_return_dec.back()[0], result_return_dec.back()[1]);
  //             terminate = true;
  //           }
  //         }
  //         break;
        
  //       case 2:
  //         // if (result_return_ori.back()[0] == -1){
  //         //   if(result_return_dec.back()[0] != -1){
  //         //     printf("some trajectories not fixed(case2-0)\n");
  //         //     printf("ori length: %zu, dec length: %zu\n", result_return_ori.size(), result_return_dec.size());
  //         //     printf("ori first %f,%f, dec first %f,%f\n", result_return_ori[0][0], result_return_ori[0][1], result_return_dec[0][0], result_return_dec[0][1]);
  //         //     printf("ori last-1 %f,%f, dec last-1 %f,%f\n", result_return_ori[result_return_ori.size()-2][0], result_return_ori[result_return_ori.size()-2][1], result_return_dec[result_return_dec.size()-2][0], result_return_dec[result_return_dec.size()-2][1]);
  //         //     printf("ori last %f,%f, dec last %f,%f\n", result_return_ori.back()[0], result_return_ori.back()[1], result_return_dec.back()[0], result_return_dec.back()[1]);
  //         //     exit(0);
  //         //   }
  //         //   auto ori_last_inside = findLastNonNegativeOne(result_return_ori);
  //         //   auto dec_last_inside = findLastNonNegativeOne(result_return_dec);
  //         //   for(int j = result_return_dec.size()-1; j >= 0; --j){
  //         //     if (get_cell_offset(ori_last_inside.data(), DW, DH) == get_cell_offset(result_return_dec[j].data(), DW, DH)){
  //         //       break;
  //         //     }
  //         //     if (j == 0){ //not found
  //         //       printf("some trajectories not fixed(case2-1)\n");
  //         //       printf("ori length: %zu, dec length: %zu\n", result_return_ori.size(), result_return_dec.size());
  //         //       printf("ori first %f,%f, dec first %f,%f\n", result_return_ori[0][0], result_return_ori[0][1], result_return_dec[0][0], result_return_dec[0][1]);
  //         //       printf("ori last-1 %f,%f, dec last-1 %f,%f\n", result_return_ori[result_return_ori.size()-2][0], result_return_ori[result_return_ori.size()-2][1], result_return_dec[result_return_dec.size()-2][0], result_return_dec[result_return_dec.size()-2][1]);
  //         //       printf("ori last %f,%f, dec last %f,%f\n", result_return_ori.back()[0], result_return_ori.back()[1], result_return_dec.back()[0], result_return_dec.back()[1]);
  //         //       exit(0);
  //         //     }
  //         //   }
  //         // }
  //         if (LastTwoPointsAreEqual(result_return_ori)){
  //           //outside
  //           if(!SearchElementFromBack(result_return_dec, result_return_ori.back())){
  //             printf("some trajectories not fixed(case0-0)\n");
  //             printf("ori length: %zu, dec length: %zu\n", result_return_ori.size(), result_return_dec.size());
  //             printf("ori first %f,%f, dec first %f,%f\n", result_return_ori[0][0], result_return_ori[0][1], result_return_dec[0][0], result_return_dec[0][1]);
  //             printf("ori last-1 %f,%f, dec last-1 %f,%f\n", result_return_ori[result_return_ori.size()-2][0], result_return_ori[result_return_ori.size()-2][1], result_return_dec[result_return_dec.size()-2][0], result_return_dec[result_return_dec.size()-2][1]);
  //             printf("ori last %f,%f, dec last %f,%f\n", result_return_ori.back()[0], result_return_ori.back()[1], result_return_dec.back()[0], result_return_dec.back()[1]);
  //             exit(0);
  //           }
  //         }
  //         else if (result_return_ori.size() == t_config.max_length){
  //           if (result_return_dec.size() != t_config.max_length){
  //             printf("some trajectories not fixed(case2-2)\n");
  //             printf("ori length: %zu, dec length: %zu\n", result_return_ori.size(), result_return_dec.size());
  //             printf("ori first %f,%f, dec first %f,%f\n", result_return_ori[0][0], result_return_ori[0][1], result_return_dec[0][0], result_return_dec[0][1]);
  //             printf("ori last-1 %f,%f, dec last-1 %f,%f\n", result_return_ori[result_return_ori.size()-2][0], result_return_ori[result_return_ori.size()-2][1], result_return_dec[result_return_dec.size()-2][0], result_return_dec[result_return_dec.size()-2][1]);
  //             printf("ori last %f,%f, dec last %f,%f\n", result_return_ori.back()[0], result_return_ori.back()[1], result_return_dec.back()[0], result_return_dec.back()[1]);
  //             exit(0);
  //           }
  //         }
  //         else{
  //           //inside and not reach max, ie found cp
  //           if (get_cell_offset(result_return_ori.back().data(), DW, DH) != get_cell_offset(result_return_dec.back().data(), DW, DH)){
  //             printf("some trajectories not fixed(case2-3\n");
  //             printf("ori length: %zu, dec length: %zu\n", result_return_ori.size(), result_return_dec.size());
  //             printf("ori first %f,%f, dec first %f,%f\n", result_return_ori[0][0], result_return_ori[0][1], result_return_dec[0][0], result_return_dec[0][1]);
  //             printf("ori last-1 %f,%f, dec last-1 %f,%f\n", result_return_ori[result_return_ori.size()-2][0], result_return_ori[result_return_ori.size()-2][1], result_return_dec[result_return_dec.size()-2][0], result_return_dec[result_return_dec.size()-2][1]);
  //             printf("ori last %f,%f, dec last %f,%f\n", result_return_ori.back()[0], result_return_ori.back()[1], result_return_dec.back()[0], result_return_dec.back()[1]);
  //             exit(0);
  //           }
  //         }
  //         break;
  //       }


  //     }
  //   }
  // }
  // if (!terminate){
  // printf("all pass!\n");
  // }
  // else{
  //   printf("some trajectories not fixed\n");
  // }
  //calculate frechet distance
  int numTrajectories = trajs_ori.size();
  vector<double> frechetDistances(numTrajectories, -1);
  auto frechetDis_time_start = std::chrono::high_resolution_clock::now();
  double max_distance = -1.0;
  int max_index = -1;
  #pragma omp parallel for
  for (int i = 0; i < numTrajectories; i++) {
    auto t1 = trajs_ori[i];
    auto t2 = trajs_final_check[i];
    // //remove (-1,-1) from t1 and t2,from the end
    // while (t1.back()[0] == -1){
    //   t1.pop_back();
    // }
    // while (t2.back()[0] == -1){
    //   t2.pop_back();
    // }
    frechetDistances[i] = frechetDistance(t1, t2);
    #pragma omp critical
    {
      if (frechetDistances[i] > max_distance){
        max_distance = frechetDistances[i];
        max_index = i;
      }
    }
  }
  auto frechetDis_time_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> frechetDis_time_elapsed = frechetDis_time_end - frechetDis_time_start;
  printf("frechetDis_time: %f\n", frechetDis_time_elapsed.count());
  //print max distance and start point， end point
  printf("max_distance start point ori: %f,%f\n", trajs_ori[max_index][0][0], trajs_ori[max_index][0][1]);
  printf("max_distance start point dec: %f,%f\n", trajs_final_check[max_index][0][0], trajs_final_check[max_index][0][1]);
  printf("max_distance end point -1 ori: %f,%f\n", trajs_ori[max_index][trajs_ori[max_index].size()-2][0], trajs_ori[max_index][trajs_ori[max_index].size()-2][1]);
  printf("max_distance end point -1 dec: %f,%f\n", trajs_final_check[max_index][trajs_final_check[max_index].size()-2][0], trajs_final_check[max_index][trajs_final_check[max_index].size()-2][1]);
  printf("max_distance end point ori: %f,%f\n", trajs_ori[max_index].back()[0], trajs_ori[max_index].back()[1]);
  printf("max_distance end point dec: %f,%f\n", trajs_final_check[max_index].back()[0], trajs_final_check[max_index].back()[1]);
  printf("dec length: %zu, ori length: %zu\n", trajs_final_check[max_index].size(), trajs_ori[max_index].size());
  //calculate statistics
  double minVal, maxVal, medianVal, meanVal, stdevVal;
  calculateStatistics(frechetDistances, minVal, maxVal, medianVal, meanVal, stdevVal);

  std::vector<std::vector<std::array<double, 2>>> max_distance_traj;
  max_distance_traj.push_back(trajs_ori[max_index]);
  max_distance_traj.push_back(trajs_final_check[max_index]);
  if (file_dir != ""){
    save_trajs_to_binary(max_distance_traj, file_dir + "max_distance_traj.bin");
  }

  //print boxplot data
  printf("Statistics data===============\n");
  printf("min: %f\n", minVal);
  printf("max: %f\n", maxVal);
  printf("median: %f\n", medianVal);
  printf("mean: %f\n", meanVal);
  printf("stdev: %f\n", stdevVal);
  printf("Statistics data===============\n");


  free(result_after_zstd);

}



int main(int argc, char **argv){
  ftk::ndarray<float> grad; //grad是三纬，第一个纬度是2，代表着u或者v，第二个纬度是DH，第三个纬度是DW
  ftk::ndarray<float> grad_out;
  size_t num = 0;
  float * u = readfile<float>(argv[1], num);
  float * v = readfile<float>(argv[2], num);
  int DW = atoi(argv[3]); //1800
  int DH = atoi(argv[4]); //1200
  double h = atof(argv[5]); //h 0.05
  double eps = atof(argv[6]); //eps 0.01
  int max_length = atoi(argv[7]); //max_length 2000
  double max_eb = atof(argv[8]); //max_eb 0.01
  //traj_config t_config = {0.05, 0.01, 2000};

  //double max_eb = 0.01;
  int obj = 0;
  std::string eb_type = argv[9];
  //int obj = atoi(argv[9]);
  int total_thread = atoi(argv[10]);
  double threshold = atof(argv[11]);
  double threshold_outside = atof(argv[12]);
  double threshold_max_iter = atof(argv[13]);
  double next_index_coeff;
  if (argc == 15){
    next_index_coeff = atof(argv[14]);
  }
  else{
    next_index_coeff = 1.0;
  }
  printf("next_index_coeff: %f\n", next_index_coeff);
  std::string file_out_dir;
  if (argc == 16){
    file_out_dir = argv[15];
  }
  else{
    file_out_dir = "";
  }
  traj_config t_config = {h, eps, max_length,next_index_coeff};
  
  omp_set_num_threads(total_thread);
  /*
  objectives0: only garantee those trajectories that reach cp are correct
  objectives1: object0 + garantee those trajectories that reach max_length has same ending cell +
                those trajectories that outside the domain has same ending cell

  */
  //fix_traj(u, v,DH, DW, max_eb, t_config, obj);
  fix_traj_v2(u, v,DH, DW, max_eb, t_config, total_thread,obj,eb_type,threshold,threshold_outside,threshold_max_iter, file_out_dir);

  free(u);
  free(v);

}





