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
#define CPSZ_OMP_FLAG 1

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

// 使用动态规划计算Fréchet距离，并使用OpenMP进行并行化
double frechetDistance(const vector<array<double, 2>>& P, const vector<array<double, 2>>& Q) {
    int n = P.size();
    int m = Q.size();
    vector<vector<double>> dp(n, vector<double>(m, -1.0));

    // 初始化第一个元素
    dp[0][0] = euclideanDistance(P[0], Q[0]);

    // 计算第一列
    #pragma omp parallel for
    for (int i = 1; i < n; i++) {
        dp[i][0] = max(dp[i-1][0], euclideanDistance(P[i], Q[0]));
    }

    // 计算第一行
    #pragma omp parallel for
    for (int j = 1; j < m; j++) {
        dp[0][j] = max(dp[0][j-1], euclideanDistance(P[0], Q[j]));
    }

    // 使用OpenMP并行化主循环
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < n; i++) {
        for (int j = 1; j < m; j++) {
            dp[i][j] = max(min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]}), euclideanDistance(P[i], Q[j]));
        }
    }

    return dp[n-1][m-1];
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
naive_method(T * U, T * V,size_t r1, size_t r2, double max_pwr_eb,traj_config t_config,int totoal_thread, int obj,std::string eb_type, std::string file_dir = ""){
  //bool write_flag = true;
  int DW = r2;
  int DH = r1;
  bool stop = false;
  std::set<size_t> all_vertex_for_all_diff_traj;

  //first add all vertex that in the boundary
  // for (size_t i = 0; i < r1; ++i){
  //   all_vertex_for_all_diff_traj.insert(i * DW);
  //   all_vertex_for_all_diff_traj.insert(i * DW + DW - 1);
    // all_vertex_for_all_diff_traj.insert(i * DW + 1);
    // all_vertex_for_all_diff_traj.insert(i * DW + DW - 2);
  // }
  // for (size_t i = 0; i < r2; ++i){
  //   all_vertex_for_all_diff_traj.insert(i);
  //   all_vertex_for_all_diff_traj.insert((DH - 1) * DW + i);
    // all_vertex_for_all_diff_traj.insert(DW + i);
    // all_vertex_for_all_diff_traj.insert((DH - 2) * DW + i);
  // }

  int NUM_ITER = 0;
  const int MAX_ITER = 2000;
  double threshold = 1.4142;
  double threshold_outside = 5;
  double threshold_max_iter = 10;
  //std::map<size_t, double> trajID_direction_map; //用来存储每个traj对应的方向
  //std::set<size_t> current_diff_traj_id;
  // std::unordered_map<size_t,int> current_diff_traj_id;
  //std::set<size_t> last_diff_traj_id;
  // std::unordered_map<size_t,int> last_diff_traj_id;
  std::vector<double> compare_time_vec;
  std::vector<double> index_time_vec;
  std::vector<double> re_cal_trajs_time_vec;
  std::vector<int> trajID_need_fix_next_vec;
  
  size_t result_size = 0;
  unsigned char * result = NULL;
  double current_pwr_eb = 0;
  if (eb_type == "rel"){
    if (CPSZ_OMP_FLAG == 0)
    result = sz_compress_cp_preserve_2d_fix(U, V, r1, r2, result_size, false, max_pwr_eb, current_pwr_eb);
    else{
      float * dec_U_inplace = NULL;
      float * dec_V_inplace = NULL;
      std::set<size_t> empty_set;
      result = omp_sz_compress_cp_preserve_2d_record_vertex(U, V, r1, r2, result_size, false, max_pwr_eb, empty_set, totoal_thread,dec_U_inplace,dec_V_inplace,eb_type);
      free(dec_U_inplace);
      free(dec_V_inplace);
    }
  }
  else if (eb_type == "abs"){
    if (CPSZ_OMP_FLAG == 0){
      result = sz_compress_cp_preserve_2d_online_abs_record_vertex(U, V, r1, r2, result_size, false, max_pwr_eb);
    }
    else{
      //use omp version
      float * dec_U_inplace = NULL;
      float * dec_V_inplace = NULL;
      std::set<size_t> empty_set;
      result = omp_sz_compress_cp_preserve_2d_record_vertex(U, V, r1, r2, result_size, false, max_pwr_eb, empty_set, totoal_thread,dec_U_inplace,dec_V_inplace,eb_type);
      free(dec_U_inplace);
      free(dec_V_inplace);
    }
    
  }
  
  unsigned char * result_after_lossless = NULL;
  size_t lossless_outsize = sz_lossless_compress(ZSTD_COMPRESSOR, 3, result, result_size, &result_after_lossless);
  free(result);
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
      sz_decompress_cp_preserve_2d_online_record_vertex<float>(result, r1,r2, dec_U, dec_V); 
    }
    else{
      omp_sz_decompress_cp_preserve_2d_online(result, r1,r2, dec_U, dec_V);
    }
    
  }
  
  // calculate compression ratio
  printf("BEGIN Compressed size(original) = %zu, ratio = %f\n", lossless_outsize, (2*r1*r2*sizeof(float)) * 1.0/lossless_outsize);
  double cr_ori = (2*r1*r2*sizeof(float)) * 1.0/lossless_outsize; 

  auto compute_cp_start = std::chrono::high_resolution_clock::now();
  //get cp for original data
  auto critical_points_ori = compute_critical_points(U, V, r1, r2);
  //get cp for decompressed data
  auto critical_points_dec = compute_critical_points(dec_U, dec_V, r1, r2);
  auto compute_cp_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> compute_cp_elapsed = compute_cp_end - compute_cp_start;


  auto total_time_start = std::chrono::high_resolution_clock::now();
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

  

  printf("first time calculate trajectory\n");
  //*************先计算一次整体的traj_ori 和traj_dec,后续只需增量修改*************
  //get trajectory for original data
  
  std::vector<int> index_ori;
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

  //std::atomic<size_t> trajID_counter(0);
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
      //result_return = trajectory(pt, seed, directions[i][0] * t_config.h, t_config.max_length, DH, DW, critical_points_ori, grad_ori, index_ori, vertex_ori, cellID_trajIDs_map_ori, current_traj_index);
      result_return = trajectory_parallel(pt, seed, directions[k][0] * t_config.h, t_config.max_length, DH, DW, critical_points_ori, grad_ori,thread_lossless_index,thread_id);
      trajs_ori[j *4 + k] = result_return;
      trajID_direction_vector[j*4 + k] = directions[k][0];

    }
  }  
  // naive方法要记录ori所涉及的所有vertex
  for (const auto& local_set:thread_lossless_index){
    vertex_ori.insert(local_set.begin(), local_set.end());
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> traj_ori_begin_elapsed = end - start;
  printf("Time for original trajectory calculation: %f\n", traj_ori_begin_elapsed.count()); 
  printf("ori_saddle_count: %ld\n", ori_saddle_count);
  printf("size of trajs_ori: %ld\n", trajs_ori.size());
  printf("size of vertex_ori: %ld\n", vertex_ori.size());
  //print how many element in trajID_direction_map
  printf("size of trajID_direction_vector: %ld\n", trajID_direction_vector.size());
  // if (std::find(trajID_direction_vector.begin(), trajID_direction_vector.end(), 0.0) != trajID_direction_vector.end()) {
  //       std::cout << "Vector contains 0" << std::endl;
  //   } else {
  //       std::cout << "Vector does not contain 0" << std::endl;
  //   }

  //把vertex_ori中的点搞到压缩中
  size_t result_size_naive = 0;
  unsigned char * result_naive = NULL;
  if (eb_type == "rel"){
    if(CPSZ_OMP_FLAG == 0){
      result_naive = sz_compress_cp_preserve_2d_fix(U, V, r1, r2, result_size_naive, false, max_pwr_eb, current_pwr_eb, vertex_ori);
    }
    else{
      float * dec_U_inplace = NULL;
      float * dec_V_inplace = NULL;
      result_naive = omp_sz_compress_cp_preserve_2d_record_vertex(U, V, r1, r2, result_size_naive, false, max_pwr_eb, vertex_ori, totoal_thread,dec_U_inplace,dec_V_inplace,eb_type);
      free(dec_U_inplace);
      free(dec_V_inplace);
    }
    
  }
  else if (eb_type == "abs"){
    if(CPSZ_OMP_FLAG == 0){
    result_naive = sz_compress_cp_preserve_2d_online_abs_record_vertex(U, V, r1, r2, result_size_naive, false, max_pwr_eb, vertex_ori);
    }
    else{
      float * dec_U_inplace = NULL;
      float * dec_V_inplace = NULL;
      result_naive = omp_sz_compress_cp_preserve_2d_record_vertex(U, V, r1, r2, result_size_naive, false, max_pwr_eb, vertex_ori, totoal_thread,dec_U_inplace,dec_V_inplace,eb_type);
      free(dec_U_inplace);
      free(dec_V_inplace);
    }
  }
  
  unsigned char * result_after_lossless_naive = NULL;
  size_t lossless_outsize_naive = sz_lossless_compress(ZSTD_COMPRESSOR, 3, result_naive, result_size_naive, &result_after_lossless_naive);
  
  auto total_time_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> total_time_elapsed = total_time_end - total_time_start;
  printf("tpsz naive method total time: %f\n", total_time_elapsed.count());
  
  auto decompress_start = std::chrono::high_resolution_clock::now();
  size_t lossless_output_naive = sz_lossless_decompress(ZSTD_COMPRESSOR, result_after_lossless_naive, lossless_outsize_naive, &result_naive, result_size_naive);
  float * dec_U_naive = NULL;
  float * dec_V_naive = NULL;
  if (eb_type == "rel"){
    if (CPSZ_OMP_FLAG == 0){
      sz_decompress_cp_preserve_2d_online<float>(result, r1,r2, dec_U_naive, dec_V_naive); // use cpsz
    }
    else{
      omp_sz_decompress_cp_preserve_2d_online(result, r1,r2, dec_U_naive, dec_V_naive);
    }
    
  }
  else if (eb_type == "abs"){
    if (CPSZ_OMP_FLAG == 0){
    sz_decompress_cp_preserve_2d_online_record_vertex<float>(result, r1,r2, dec_U_naive, dec_V_naive); 
    }
    else{
      omp_sz_decompress_cp_preserve_2d_online(result, r1,r2, dec_U_naive, dec_V_naive);
    }
  }
  
  // calculate compression ratio
  auto decompress_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> decompress_elapsed = decompress_end - decompress_start;
  printf("tpsz naive method Decompress time: %f\n", decompress_elapsed.count());
  printf("Compression ratio for naive method: %f\n", lossless_outsize_naive, (2*r1*r2*sizeof(float)) * 1.0/lossless_outsize_naive);
  double nrmse_u, nrmse_v, psnr_overall;
  verify(U, dec_U_naive, r1*r2, nrmse_u);
  verify(V, dec_V_naive, r1*r2, nrmse_v);
  psnr_overall = 20 * log10(sqrt(3) / sqrt(nrmse_u*nrmse_u + nrmse_v*nrmse_v));
  printf("nrmse_u: %f, nrmse_v: %f, psnr_overall: %f\n", nrmse_u, nrmse_v, psnr_overall);
  exit(0);


  printf("checkpt4\n");
  //now check if all trajectories are correct
  ftk::ndarray<float> grad_final;
  grad_final.reshape({2, static_cast<unsigned long>(r2), static_cast<unsigned long>(r1)});
  refill_gradient(0, r1, r2, dec_U_naive, grad_final);
  refill_gradient(1, r1, r2, dec_V_naive, grad_final);
  auto critical_points_final =compute_critical_points(dec_U_naive, dec_V_naive, r1, r2);
  std::vector<std::vector<std::array<double, 2>>> final_check_ori;
  std::vector<std::vector<std::array<double, 2>>> final_check_dec;
  std::set<size_t> vertex_final;
  for(const auto& p:critical_points_final){
    auto cp = p.second;
    if (cp.type == SADDLE){
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
        double pos[2] = {cp.x[0], cp.x[1]};
        std::set<size_t> temp1,temp2;
        std::vector<std::array<double, 2>> result_return_ori = trajectory(pos, seed, directions[i][0] * t_config.h, t_config.max_length, DH, DW, critical_points_final, grad_ori,temp1);
        std::vector<std::array<double, 2>> result_return_dec = trajectory(pos, seed, directions[i][0] * t_config.h, t_config.max_length, DH, DW, critical_points_final, grad_final,temp2);
        final_check_ori.push_back(result_return_ori);
        final_check_dec.push_back(result_return_dec);
        switch (obj)
        {
        case 0:
          // if (LastTwoPointsAreEqual(result_return_ori)){
          //   //outside
          //   if(!SearchElementFromBack(result_return_dec, result_return_ori.back())){
          //     printf("some trajectories not fixed(case0-0)\n");
          //     printf("ori length: %zu, dec length: %zu\n", result_return_ori.size(), result_return_dec.size());
          //     printf("ori first %f,%f, dec first %f,%f\n", result_return_ori[0][0], result_return_ori[0][1], result_return_dec[0][0], result_return_dec[0][1]);
          //     printf("ori last-1 %f,%f, dec last-1 %f,%f\n", result_return_ori[result_return_ori.size()-2][0], result_return_ori[result_return_ori.size()-2][1], result_return_dec[result_return_dec.size()-2][0], result_return_dec[result_return_dec.size()-2][1]);
          //     printf("ori last %f,%f, dec last %f,%f\n", result_return_ori.back()[0], result_return_ori.back()[1], result_return_dec.back()[0], result_return_dec.back()[1]);
          //     exit(0);
          //   }
          // }
          if (LastTwoPointsAreEqual(result_return_ori)){
            if (euclideanDistance(result_return_ori.back(),result_return_dec.back()) > threshold_outside){
              printf("some trajectories not fixed(case0-0)\n");
              printf("ori length: %zu, dec length: %zu\n", result_return_ori.size(), result_return_dec.size());
              printf("ori first %f,%f, dec first %f,%f\n", result_return_ori[0][0], result_return_ori[0][1], result_return_dec[0][0], result_return_dec[0][1]);
              printf("ori last-1 %f,%f, dec last-1 %f,%f\n", result_return_ori[result_return_ori.size()-2][0], result_return_ori[result_return_ori.size()-2][1], result_return_dec[result_return_dec.size()-2][0], result_return_dec[result_return_dec.size()-2][1]);
              printf("ori last %f,%f, dec last %f,%f\n", result_return_ori.back()[0], result_return_ori.back()[1], result_return_dec.back()[0], result_return_dec.back()[1]);
              exit(0);
            }
          }
          else if (result_return_ori.size() == t_config.max_length){
            if (euclideanDistance(result_return_ori.back(),result_return_dec.back()) > threshold_max_iter){
              printf("some trajectories not fixed(case0-1)\n");
              printf("ori length: %zu, dec length: %zu\n", result_return_ori.size(), result_return_dec.size());
              printf("ori first %f,%f, dec first %f,%f\n", result_return_ori[0][0], result_return_ori[0][1], result_return_dec[0][0], result_return_dec[0][1]);
              printf("ori last-1 %f,%f, dec last-1 %f,%f\n", result_return_ori[result_return_ori.size()-2][0], result_return_ori[result_return_ori.size()-2][1], result_return_dec[result_return_dec.size()-2][0], result_return_dec[result_return_dec.size()-2][1]);
              printf("ori last %f,%f, dec last %f,%f\n", result_return_ori.back()[0], result_return_ori.back()[1], result_return_dec.back()[0], result_return_dec.back()[1]);
              exit(0);
            }
          }
          else{
            //inside and not reach max, ie found cp
            if(get_cell_offset(result_return_ori.back().data(), DW, DH) != get_cell_offset(result_return_dec.back().data(), DW, DH)){
              printf("some trajectories not fixed(case0-2)\n");
              printf("ori length: %zu, dec length: %zu\n", result_return_ori.size(), result_return_dec.size());
              printf("ori first %f,%f, dec first %f,%f\n", result_return_ori[0][0], result_return_ori[0][1], result_return_dec[0][0], result_return_dec[0][1]);
              printf("ori last-1 %f,%f, dec last-1 %f,%f\n", result_return_ori[result_return_ori.size()-2][0], result_return_ori[result_return_ori.size()-2][1], result_return_dec[result_return_dec.size()-2][0], result_return_dec[result_return_dec.size()-2][1]);
              printf("ori last %f,%f, dec last %f,%f\n", result_return_ori.back()[0], result_return_ori.back()[1], result_return_dec.back()[0], result_return_dec.back()[1]);
              exit(0);
            }
          }
          break;
        
        case 2:
          // if (result_return_ori.back()[0] == -1){
          //   if(result_return_dec.back()[0] != -1){
          //     printf("some trajectories not fixed(case2-0)\n");
          //     printf("ori length: %zu, dec length: %zu\n", result_return_ori.size(), result_return_dec.size());
          //     printf("ori first %f,%f, dec first %f,%f\n", result_return_ori[0][0], result_return_ori[0][1], result_return_dec[0][0], result_return_dec[0][1]);
          //     printf("ori last-1 %f,%f, dec last-1 %f,%f\n", result_return_ori[result_return_ori.size()-2][0], result_return_ori[result_return_ori.size()-2][1], result_return_dec[result_return_dec.size()-2][0], result_return_dec[result_return_dec.size()-2][1]);
          //     printf("ori last %f,%f, dec last %f,%f\n", result_return_ori.back()[0], result_return_ori.back()[1], result_return_dec.back()[0], result_return_dec.back()[1]);
          //     exit(0);
          //   }
          //   auto ori_last_inside = findLastNonNegativeOne(result_return_ori);
          //   auto dec_last_inside = findLastNonNegativeOne(result_return_dec);
          //   for(int j = result_return_dec.size()-1; j >= 0; --j){
          //     if (get_cell_offset(ori_last_inside.data(), DW, DH) == get_cell_offset(result_return_dec[j].data(), DW, DH)){
          //       break;
          //     }
          //     if (j == 0){ //not found
          //       printf("some trajectories not fixed(case2-1)\n");
          //       printf("ori length: %zu, dec length: %zu\n", result_return_ori.size(), result_return_dec.size());
          //       printf("ori first %f,%f, dec first %f,%f\n", result_return_ori[0][0], result_return_ori[0][1], result_return_dec[0][0], result_return_dec[0][1]);
          //       printf("ori last-1 %f,%f, dec last-1 %f,%f\n", result_return_ori[result_return_ori.size()-2][0], result_return_ori[result_return_ori.size()-2][1], result_return_dec[result_return_dec.size()-2][0], result_return_dec[result_return_dec.size()-2][1]);
          //       printf("ori last %f,%f, dec last %f,%f\n", result_return_ori.back()[0], result_return_ori.back()[1], result_return_dec.back()[0], result_return_dec.back()[1]);
          //       exit(0);
          //     }
          //   }
          // }
          if (LastTwoPointsAreEqual(result_return_ori)){
            //outside
            if(!SearchElementFromBack(result_return_dec, result_return_ori.back())){
              printf("some trajectories not fixed(case0-0)\n");
              printf("ori length: %zu, dec length: %zu\n", result_return_ori.size(), result_return_dec.size());
              printf("ori first %f,%f, dec first %f,%f\n", result_return_ori[0][0], result_return_ori[0][1], result_return_dec[0][0], result_return_dec[0][1]);
              printf("ori last-1 %f,%f, dec last-1 %f,%f\n", result_return_ori[result_return_ori.size()-2][0], result_return_ori[result_return_ori.size()-2][1], result_return_dec[result_return_dec.size()-2][0], result_return_dec[result_return_dec.size()-2][1]);
              printf("ori last %f,%f, dec last %f,%f\n", result_return_ori.back()[0], result_return_ori.back()[1], result_return_dec.back()[0], result_return_dec.back()[1]);
              exit(0);
            }
          }
          else if (result_return_ori.size() == t_config.max_length){
            if (result_return_dec.size() != t_config.max_length){
              printf("some trajectories not fixed(case2-2)\n");
              printf("ori length: %zu, dec length: %zu\n", result_return_ori.size(), result_return_dec.size());
              printf("ori first %f,%f, dec first %f,%f\n", result_return_ori[0][0], result_return_ori[0][1], result_return_dec[0][0], result_return_dec[0][1]);
              printf("ori last-1 %f,%f, dec last-1 %f,%f\n", result_return_ori[result_return_ori.size()-2][0], result_return_ori[result_return_ori.size()-2][1], result_return_dec[result_return_dec.size()-2][0], result_return_dec[result_return_dec.size()-2][1]);
              printf("ori last %f,%f, dec last %f,%f\n", result_return_ori.back()[0], result_return_ori.back()[1], result_return_dec.back()[0], result_return_dec.back()[1]);
              exit(0);
            }
          }
          else{
            //inside and not reach max, ie found cp
            if (get_cell_offset(result_return_ori.back().data(), DW, DH) != get_cell_offset(result_return_dec.back().data(), DW, DH)){
              printf("some trajectories not fixed(case2-3\n");
              printf("ori length: %zu, dec length: %zu\n", result_return_ori.size(), result_return_dec.size());
              printf("ori first %f,%f, dec first %f,%f\n", result_return_ori[0][0], result_return_ori[0][1], result_return_dec[0][0], result_return_dec[0][1]);
              printf("ori last-1 %f,%f, dec last-1 %f,%f\n", result_return_ori[result_return_ori.size()-2][0], result_return_ori[result_return_ori.size()-2][1], result_return_dec[result_return_dec.size()-2][0], result_return_dec[result_return_dec.size()-2][1]);
              printf("ori last %f,%f, dec last %f,%f\n", result_return_ori.back()[0], result_return_ori.back()[1], result_return_dec.back()[0], result_return_dec.back()[1]);
              exit(0);
            }
          }
          break;
        }


      }
    }
  }
  printf("all pass!\n");
  //calculate frechet distance
  int numTrajectories = final_check_ori.size();
  vector<double> frechetDistances(numTrajectories, -1);
  auto frechetDis_time_start = std::chrono::high_resolution_clock::now();
  double max_distance = -1.0;
  int max_index = -1;
  #pragma omp parallel for
  for (int i = 0; i < numTrajectories; i++) {
    auto t1 = final_check_ori[i];
    auto t2 = final_check_dec[i];
    //remove (-1,-1) from t1 and t2,from the end
    while (t1.back()[0] == -1){
      t1.pop_back();
    }
    while (t2.back()[0] == -1){
      t2.pop_back();
    }
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
  printf("max_distance start point ori: %f,%f\n", final_check_ori[max_index][0][0], final_check_ori[max_index][0][1]);
  printf("max_distance start point dec: %f,%f\n", final_check_dec[max_index][0][0], final_check_dec[max_index][0][1]);
  printf("max_distance end point ori: %f,%f\n", final_check_ori[max_index].back()[0], final_check_ori[max_index].back()[1]);
  printf("max_distance end point dec: %f,%f\n", final_check_dec[max_index].back()[0], final_check_dec[max_index].back()[1]);
  printf("dec length: %zu, ori length: %zu\n", final_check_dec[max_index].size(), final_check_ori[max_index].size());
  //calculate statistics
  double minVal, maxVal, medianVal, meanVal, stdevVal;
  calculateStatistics(frechetDistances, minVal, maxVal, medianVal, meanVal, stdevVal);

  //print boxplot data
  printf("Statistics data===============\n");
  printf("min: %f\n", minVal);
  printf("max: %f\n", maxVal);
  printf("median: %f\n", medianVal);
  printf("mean: %f\n", meanVal);
  printf("stdev: %f\n", stdevVal);
  printf("Statistics data===============\n");
  
  



  //if file_dir is empty, then not write
  if (file_dir != ""){
    // printf("writing original and decompressed trajs to file...\n");
    save_trajs_to_binary(final_check_ori, file_dir + "ori_traj.bin");
    save_trajs_to_binary(final_check_dec, file_dir + "dec_traj.bin");
  }


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
  traj_config t_config = {h, eps, max_length};
  //double max_eb = 0.01;
  //int objectives = 0;
  int obj = 0;
  std::string eb_type = argv[9];
  int total_thread = atoi(argv[10]);
  std::string file_out_dir ="";
  if (argc == 12){
    file_out_dir = argv[11];
  }
  
  omp_set_num_threads(total_thread);
  /*
  objectives0: only garantee those trajectories that reach cp are correct
  objectives1: object0 + garantee those trajectories that reach max_length has same ending cell +
                those trajectories that outside the domain has same ending cell

  */
  //fix_traj(u, v,DH, DW, max_eb, t_config, obj);
  naive_method(u, v,DH, DW, max_eb, t_config, total_thread,obj,eb_type, file_out_dir);

  free(u);
  free(v);

}





