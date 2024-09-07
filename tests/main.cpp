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

std::array<double,2> calcululateIntersections(const std::array<double,2>& p1, const std::array<double,2>& p2){
  double slope = (p2[1] - p1[1]) / (p2[0] - p1[0]);
  double y_intercept = p1[1] - slope * p1[0];
  double x_intercept = -y_intercept / slope;
  return {x_intercept, y_intercept};
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

typedef struct traj_config{
  double h;
  double eps;
  int max_length; 
} traj_config;


template<typename Type>
void verify(Type * ori_data, Type * data, size_t num_elements){
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
    double nrmse = sqrt(mse)/range;

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
fix_traj_v2(T * U, T * V,size_t r1, size_t r2, double max_pwr_eb,traj_config t_config,int totoal_thread, int obj, std::string file_dir = ""){
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
  result = sz_compress_cp_preserve_2d_fix(U, V, r1, r2, result_size, false, max_pwr_eb, current_pwr_eb);
  unsigned char * result_after_lossless = NULL;
  size_t lossless_outsize = sz_lossless_compress(ZSTD_COMPRESSOR, 3, result, result_size, &result_after_lossless);
  size_t lossless_output = sz_lossless_decompress(ZSTD_COMPRESSOR, result_after_lossless, lossless_outsize, &result, result_size);
  float * dec_U = NULL;
  float * dec_V = NULL;
  sz_decompress_cp_preserve_2d_online<float>(result, r1,r2, dec_U, dec_V); // use cpsz
  // calculate compression ratio
  printf("BEGIN Compressed size(original) = %zu, ratio = %f\n", lossless_outsize, (2*r1*r2*sizeof(float)) * 1.0/lossless_outsize);
  double cr_ori = (2*r1*r2*sizeof(float)) * 1.0/lossless_outsize; 
  // init a unordered_map, key is the index of cell, value is an array which contains the index of trajectory that goes through this cell
  std::unordered_map<size_t, std::set<int>> cellID_trajIDs_map_ori;
  std::unordered_map<size_t, std::set<int>> cellID_trajIDs_map_dec;

  //get cp for original data
  auto critical_points_ori = compute_critical_points(U, V, r1, r2);
  //get cp for decompressed data
  auto critical_points_dec = compute_critical_points(dec_U, dec_V, r1, r2);

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
  // 这里只是计算trajs，不需要算vertex其实
  // for (const auto& local_set:thread_lossless_index){
  //   vertex_ori.insert(local_set.begin(), local_set.end());
  // }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> traj_ori_begin_elapsed = end - start;
  printf("Time for original trajectory calculation: %f\n", traj_ori_begin_elapsed.count()); 
  printf("ori_saddle_count: %ld\n", ori_saddle_count);
  printf("size of trajs_ori: %ld\n", trajs_ori.size());
  printf("size of vertex_ori: %ld\n", vertex_ori.size());
  //print how many element in trajID_direction_map
  printf("size of trajID_direction_vector: %ld\n", trajID_direction_vector.size());
  if (std::find(trajID_direction_vector.begin(), trajID_direction_vector.end(), 0.0) != trajID_direction_vector.end()) {
        std::cout << "Vector contains 0" << std::endl;
    } else {
        std::cout << "Vector does not contain 0" << std::endl;
    }

  
  //get trajectory for decompressed data
  std::vector<int> index_dec;
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

  //std::atomic<size_t> trajID_counter(0);
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
      // result_return = trajectory(pt, seed, directions[i][0]* t_config.h, t_config.max_length, DH, DW, critical_points_dec, grad_dec, index_dec, vertex_dec, cellID_trajIDs_map_dec, current_traj_index);
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
  printf("dec_saddle_count: %ld\n", dec_saddle_count);
  printf("size of trajs_dec: %ld\n", trajs_dec.size());
  printf("size of vertex_dec: %ld\n", vertex_dec.size());

  //检查一下这两个trajs_ori和trajs_dec是前三个坐标一致，为了验证顺序是否一致（不算时间）
  // for(size_t i = 0; i < trajs_ori.size(); ++i){
  //   auto t1 = trajs_ori[i];
  //   auto t2 = trajs_dec[i];
  //   if (t1[0][0] != t2[0][0] || t1[0][1] != t2[0][1] || t1[1][0] != t2[1][0] || t1[1][1] != t2[1][1]){
  //     printf("Trajectory ori %ld is diff to dec\n", i);
  //     printf("first ori:(%f,%f), second ori(%f,%f): last ori:(%f,%f)\n",t1[0][0],t1[0][1],t1[1][0],t1[1][1],t1.back()[0],t1.back()[1]);
  //     printf("first dec:(%f,%f), second dec(%f,%f): last dec:(%f,%f)\n",t2[0][0],t2[0][1],t2[1][0],t2[1][1],t2.back()[0],t2.back()[1]);
  //     printf("ori length: %zu, dec length: %zu\n", t1.size(), t2.size());
  //     exit(0);
  //   }
  // }

  // 计算哪里有问题（init queue）
  std::set<size_t> trajID_need_fix = {};
  auto init_queue_start = std::chrono::high_resolution_clock::now();

  switch(obj){
    case 0:
      for(size_t i =0; i< trajs_ori.size(); ++i){
        auto t1 = trajs_ori[i];
        auto t2 = trajs_dec[i];
        bool cond1 = get_cell_offset(t1.back().data(), DW, DH) == get_cell_offset(t2.back().data(), DW, DH); //ori and dec reach same cp
        bool cond2 = t1.size() == t_config.max_length; //ori reach max
        bool cond3 = t1.back()[0] == -1; //ori outside
        bool cond4 = t2.back()[0] == -1; //dec outside
        // if (!cond2 && !cond3 && !cond1){ //ori 找到了cp，但是dec和ori不一致
        //   trajID_need_fix.insert(i);
        // }
        // if (cond3 && !cond4){ //ori outside, dec inside
        //   trajID_need_fix.insert(i);
        // }
        if (cond3){
          //ori outside
          auto last_inside_ori = findLastNonNegativeOne(t1);
          auto last_inside_dec = findLastNonNegativeOne(t2);
          //每一个t2从后往前找，看有没有cell跟last_inside_ori一样的，如果没有，就insert
          if (!cond4){ //没出去就直接加了
            trajID_need_fix.insert(i);
          }
          for(int j = t2.size()-1; j >= 0; --j){
            if (get_cell_offset(last_inside_ori.data(), DW, DH) == get_cell_offset(t2[j].data(), DW, DH)){
              break;
            }
            if (j == 0){
              trajID_need_fix.insert(i);
            }
          }
          // if (get_cell_offset(last_inside_ori.data(), DW, DH) != get_cell_offset(last_inside_dec.data(), DW, DH)){
          //   trajID_need_fix.insert(i);
          //   //printf("ori pos: (%f, %f), dec pos: (%f, %f), dist: %f\n", last_inside[0], last_inside[1], last_inside_dec[0], last_inside_dec[1], euclideanDistance(last_inside, last_inside_dec));
          // }
        }
        else if (cond2){
          //ori reach max,dont care dec
          continue;
        }
        else if (!cond2 && !cond3){
          //ori inside, not reach max, ie found cp
          if (!cond1){
            trajID_need_fix.insert(i);
            //printf("ori found cp but different cell with dec, ori pos: (%f, %f), dec pos: (%f, %f)\n", t1.back()[0], t1.back()[1], t2.back()[0], t2.back()[1]);
          }
        }
      }
      break;
    case 1:
      for(size_t i=0;i<trajs_ori.size(); ++i){
        auto t1 = trajs_ori[i];
        auto t2 = trajs_dec[i];
        bool cond2 = t1.size() == t_config.max_length;
        bool cond3 = t2.size() == t_config.max_length;
        bool cond4 = t1.back()[0] == -1;
        bool cond5 = t2.back()[0] == -1;
        if(!cond2 && !cond4){ //inside and not reach max, ie found cp
          std::array<double, 2>  ori_last_inside = findLastNonNegativeOne(t1);
          std::array<double, 2>  dec_last_inside = findLastNonNegativeOne(t2);
          if (get_cell_offset(ori_last_inside.data(), DW, DH) != get_cell_offset(dec_last_inside.data(), DW, DH)){
            trajID_need_fix.insert(i);
        
          }

        }
        else if (cond2){ //original reach limit
          //找到dec最后一个非(-1,-1)的点
          std::array<double, 2>  dec_last_inside = findLastNonNegativeOne(t2);
          if (get_cell_offset(t1.back().data(), DW, DH) != get_cell_offset(dec_last_inside.data(), DW, DH)){
            trajID_need_fix.insert(i);
          }
        }
        else if (cond4){// original outside
          //遍历找到t1最后一个非(-1,-1)的点
          std::array<double, 2>  ori_last_inside = findLastNonNegativeOne(t1);
          std::array<double, 2>  dec_last_inside = findLastNonNegativeOne(t2);
          if (get_cell_offset(ori_last_inside.data(), DW, DH) != get_cell_offset(dec_last_inside.data(), DW, DH)){
            trajID_need_fix.insert(i);
          }
        }
      }
      break;
    case 2:
      //  original | dec
      //  outside  | outside (could go different direction)
      //  max_iter | max_iter (could be different)
      //  reach cp | reach same cp
      for(size_t i=0;i<trajs_ori.size(); ++i){
        auto t1 = trajs_ori[i];
        auto t2 = trajs_dec[i];
        bool cond2 = t1.size() == t_config.max_length;
        bool cond3 = t2.size() == t_config.max_length;
        bool cond4 = t1.back()[0] == -1;
        bool cond5 = t2.back()[0] == -1;
        // if (cond4){
        //   //ori outside
        //   auto last_inside = findLastNonNegativeOne(t1);
        //   auto last_inside_dec = findLastNonNegativeOne(t2);
        //   if (get_cell_offset(last_inside.data(), DW, DH) != get_cell_offset(last_inside_dec.data(), DW, DH)){
        //     trajID_need_fix.insert(i);
        //   }
        // }

        if(cond4){
          //ori outside
          auto last_inside_ori = findLastNonNegativeOne(t1);
          auto last_inside_dec = findLastNonNegativeOne(t2);
          //每一个t2从后往前找，看有没有cell跟last_inside_ori一样的，如果没有，就insert
          if (!cond5){ //没出去就直接加了
            trajID_need_fix.insert(i);
          }
          for(int j = t2.size()-1; j >= 0; --j){
            if (get_cell_offset(last_inside_ori.data(), DW, DH) == get_cell_offset(t2[j].data(), DW, DH)){
              break;
            }
            if (j == 0){
              trajID_need_fix.insert(i);
            }
          }
        }


        else if(cond2){
          //ori reach max
          if (!cond3){
            trajID_need_fix.insert(i);
          }
        }
        else if (!cond2 && !cond4){
          //ori inside, not reach max, ie found cp
          std::array<double, 2>  ori_last_inside = findLastNonNegativeOne(t1);
          std::array<double, 2>  dec_last_inside = findLastNonNegativeOne(t2);
          if (get_cell_offset(ori_last_inside.data(), DW, DH) != get_cell_offset(dec_last_inside.data(), DW, DH)){
            trajID_need_fix.insert(i);
          }
        }
      }
      break;
  }


  auto init_queue_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> init_queue_elapsed = init_queue_end - init_queue_start;

  if (trajID_need_fix.size() == 0){
    stop = true;
    printf("No need to fix!\n");
    exit(0);
  }
  else{
    printf("trajID_need_fix size: %ld\n", trajID_need_fix.size());
    trajID_need_fix_next_vec.push_back(trajID_need_fix.size());
  }

  int current_round = 0;
  double temp_trajs_ori_time = 0;
  do
  {
    printf("begin fix traj,current_round: %d\n", current_round++);
    std::set<size_t> trajID_need_fix_next;
    //fix trajectory
    auto index_time_start = std::chrono::high_resolution_clock::now();
    //这里不太好并行
    //每个线程拿一个trajid
    //convert trajID_need_fix to vector
    std::vector<size_t> trajID_need_fix_vector(trajID_need_fix.begin(), trajID_need_fix.end());
    printf("current iteration size: %ld\n", trajID_need_fix_vector.size());
    
    std::vector<std::set<size_t>> local_all_vertex_for_all_diff_traj(totoal_thread);

    #pragma omp parallel for
    for (size_t i=0;i<trajID_need_fix_vector.size(); ++i){
      auto current_trajID = trajID_need_fix_vector[i];
      // for (const auto& trajID:trajID_need_fix){
      //   size_t current_trajID = trajID;
      bool success = false;
      auto t1 = trajs_ori[current_trajID];
      auto t2 = trajs_dec[current_trajID];
      int start_fix_index = 0;
      int end_fix_index = t1.size() - 1;
      double threshold = 1.4142;
      
      int thread_id = omp_get_thread_num();

      //find the first different point
      int changed = 0;
      for (size_t j = start_fix_index; j < std::min(t1.size(),t2.size()); ++j){
      //for (size_t j = start_fix_index; j < max_index; ++j){
        auto p1 = t1[j];
        auto p2 = t2[j];
        if (p1[0] > 0 && p2[0] > 0){
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
      end_fix_index = std::min(end_fix_index, static_cast<int>(t1.size() - 1));
    
      while (!success){
        double direction = trajID_direction_vector[current_trajID];
        std::vector<int> temp_index_ori;
        std::vector<int> temp_index_check;
        //std::unordered_map<size_t, std::set<int>> temp_cellID_trajIDs_map;//需要临时的一个map统计修正的traj经过那些cellID
        std::set<size_t> temp_vertexID;//临时变量记录经过的vertexID
        std::set<size_t> temp_vertexID_check;
        std::unordered_map<size_t,double> rollback_dec_u; //记录需要回滚的dec数据
        std::unordered_map<size_t,double> rollback_dec_v;
        //计算一次rk4直到终止点，统计经过的cellID，然后替换数据
        /* 使用简单的版本*/
        auto temp_trajs_ori_time_start = std::chrono::high_resolution_clock::now();
        end_fix_index = std::min(end_fix_index + 1,t_config.max_length);
        auto temp_trajs_ori = trajectory(t1[0].data(), t1[1], direction * t_config.h, end_fix_index, DH, DW, critical_points_ori, grad_ori,temp_index_ori, temp_vertexID);
        
        //auto temp_trajs_ori_time_end = std::chrono::high_resolution_clock::now();

        //此时temp_vertexID记录了从起点到分岔点需要经过的所有vertex
        for (auto o:temp_vertexID){
          rollback_dec_u[o] = dec_U[o];
          rollback_dec_v[o] = dec_V[o];
          local_all_vertex_for_all_diff_traj[thread_id].insert(o);
        }
        auto current_divergence_pos = temp_trajs_ori.back();
        //printf("current_divergence_pos ori data(temp_trajs_ori last element): (%f,%f)\n", current_divergence_pos[0], current_divergence_pos[1]);
        //printf("current_end_fix_index: %d, t1.size(): %zu, t2.size(): %zu\n", end_fix_index, t1.size(), t2.size());
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
        }
        //检查是不是修正成功
        //std::unordered_map<size_t, std::set<int>> temp_cellID_trajIDs_map_dec;
        //std::set<size_t> temp_var;
        //auto temp_trajs_check = trajectory(t2[0].data(), t2[1], direction * t_config.h, t_config.max_length, DH, DW, critical_points_dec, grad_dec,temp_index, temp_var,temp_cellID_trajIDs_map_dec,current_trajID);
        //auto temp_trajs_check = trajectory(t2[0].data(), t2[1], direction * t_config.h, t_config.max_length, DH, DW, critical_points_dec, grad_dec,temp_index, temp_var,temp_cellID_trajIDs_map_dec,current_trajID);
        /*
        1. 这里使用简单的traj计算，避免过度开销
        2. 不从0开始，而是从end_fix_index开始，走max_length-end_fix_index+2步
        */

        //auto temp_debug = trajectory(t2[0].data(), t2[1],direction * t_config.h,end_fix_index, DH, DW, critical_points_dec, grad_dec,temp_index_check);
        
        //printf("current_divergence_pos dec data: (%f,%f)\n", temp_debug.back()[0], temp_debug.back()[1]);
        //auto temp_trajs_check = trajectory(t2[0].data(), t2[1], direction * t_config.h, t_config.max_length, DH, DW, critical_points_dec, grad_dec,temp_index_check);
        // auto temp_trajs_check = trajectory(current_divergence_pos.data(), current_divergence_pos, direction * t_config.h, t1.size()-end_fix_index+2, DH, DW, critical_points_dec, grad_dec,temp_index_check);
        auto temp_trajs_check = trajectory(t1[0].data(), t1[1], direction * t_config.h, t_config.max_length, DH, DW, critical_points_dec, grad_dec,temp_index_check,temp_vertexID_check);
        //printf("temp_trajs_check last element: (%f,%f), max_iter-end_fix + 2 = %d\n", temp_trajs_check.back()[0], temp_trajs_check.back()[1], t_config.max_length-end_fix_index+2);
        //printf("ori last element: (%f,%f)\n", t1.back()[0], t1.back()[1]);

      
        //success = (get_cell_offset(findLastNonNegativeOne(t1).data(), DW, DH) == get_cell_offset(findLastNonNegativeOne(temp_trajs_check).data(), DW, DH));
        switch (obj)
        {
          case 0:
            if (t1.back()[0] == -1){
              if (temp_trajs_check.back()[0] != -1){
                success = false;
              }
              //ori outside, dec last non(-1,-1) point should be close
              auto last_inside_ori = findLastNonNegativeOne(t1);
              for(int j = temp_trajs_check.size()-1; j >= 0; --j){
                if (get_cell_offset(last_inside_ori.data(), DW, DH) == get_cell_offset(temp_trajs_check[j].data(), DW, DH)){
                  success = true;
                  break;
                }
                if (j == 0){//not found
                  success = false;
                }
              }
              //success = (get_cell_offset(findLastNonNegativeOne(t1).data(), DW, DH) == get_cell_offset(findLastNonNegativeOne(temp_trajs_check).data(), DW, DH));
            }
            else if (t1.size() == t_config.max_length){
              //ori reach max, dont care dec
              success = true;
              }
            else if (t1.size() != t_config.max_length && t1.back()[0] != -1){
              //ori inside, not reach max, ie found cp
              success = (get_cell_offset(t1.back().data(), DW, DH) == get_cell_offset(temp_trajs_check.back().data(), DW, DH));
            }
            break;
          case 1:
            success = (get_cell_offset(findLastNonNegativeOne(t1).data(), DW, DH) == get_cell_offset(findLastNonNegativeOne(temp_trajs_check).data(), DW, DH));
            break;
          case 2:
            // if (t1.back()[0] == -1){
            //   //ori outside, dec last non(-1,-1) point should be close
            //   success = (get_cell_offset(findLastNonNegativeOne(t1).data(), DW, DH) == get_cell_offset(findLastNonNegativeOne(temp_trajs_check).data(), DW, DH));
            // }
            if(t1.back()[0] == -1){
              if (temp_trajs_check.back()[0] != -1){
                success = false;
              }
              //ori outside, dec last non(-1,-1) point should be close
              auto last_inside_ori = findLastNonNegativeOne(t1);
              for(int j = temp_trajs_check.size()-1; j >= 0; --j){
                if (get_cell_offset(last_inside_ori.data(), DW, DH) == get_cell_offset(temp_trajs_check[j].data(), DW, DH)){
                  success = true;
                  break;
                }
                if (j == 0){//not found
                  success = false;
                }
              }
            }
            else if (t1.size() == t_config.max_length){
              //ori reach max, dec should reach max as well
              success = (temp_trajs_check.size() == t_config.max_length);
            }
            else if (t1.size() != t_config.max_length && t1.back()[0] != -1){
              //ori inside, not reach max, ie found cp
              success = (get_cell_offset(t1.back().data(), DW, DH) == get_cell_offset(temp_trajs_check.back().data(), DW, DH));
            }
            break;
        }
      
        if (!success){
          //rollback
          //printf("trajID: %ld not fixed, current end_fix_index: %d\n", current_trajID, end_fix_index);
          // if(end_fix_index >=(t1.size()-1)){
          //   printf("BEFORE ERASE: local_all_vertex_for_all_diff_traj size: %ld\n", local_all_vertex_for_all_diff_traj[thread_id].size());
          //   printf("temp_vertexID size: %ld\n", temp_vertexID.size());
          // }
          for (auto o:temp_vertexID){
            dec_U[o] = rollback_dec_u[o];
            dec_V[o] = rollback_dec_v[o];
            int x = o % DW;
            int y = o / DW;
            grad_dec(0, x, y) = dec_U[o];
            grad_dec(1, x, y) = dec_V[o];
            local_all_vertex_for_all_diff_traj[thread_id].erase(o);
          }
          if (end_fix_index >= t_config.max_length){ //(t1.size()-1)

            printf("t_config.max_length: %d,end_fix_index%d\n", t_config.max_length, end_fix_index);
            printf("error: current end_fix_index is %d, current ID: %ld\n", end_fix_index, current_trajID);
            //print all t1,temp_trajs_ori,temp_trajs_check
            // for (size_t i = 0; i < t1.size(); ++i){
            //   printf("t1: (%f,%f), temp_trajs_ori: (%f,%f), temp_trajs_check: (%f,%f)\n", t1[i][0], t1[i][1], temp_trajs_ori[i][0], temp_trajs_ori[i][1], temp_trajs_check[i][0], temp_trajs_check[i][1]);
            // }
            printf("ori first: (%f,%f), temp_trajs_ori first: (%f,%f), temp_trajs_check first: (%f,%f)\n", t1[0][0], t1[0][1],temp_trajs_ori[0][0], temp_trajs_ori[0][1],temp_trajs_check[0][0], temp_trajs_check[0][1]);
            printf("ori second: (%f,%f), temp_trajs_ori second: (%f,%f), temp_trajs_check second: (%f,%f)\n", t1[1][0], t1[1][1],temp_trajs_ori[1][0], temp_trajs_ori[1][1],temp_trajs_check[1][0], temp_trajs_check[1][1]);
            printf("ori last-2 : (%f,%f), temp_trajs_ori last-2: (%f,%f), temp_trajs_check last-2: (%f,%f)\n", t1[t1.size()-3][0], t1[t1.size()-3][1],temp_trajs_ori[temp_trajs_ori.size()-3][0], temp_trajs_ori[temp_trajs_ori.size()-3][1],temp_trajs_check[temp_trajs_check.size()-3][0], temp_trajs_check[temp_trajs_check.size()-3][1]);
            printf("ori last-1 : (%f,%f), temp_trajs_ori last-1: (%f,%f), temp_trajs_check last-1: (%f,%f)\n", t1[t1.size()-2][0], t1[t1.size()-2][1],temp_trajs_ori[temp_trajs_ori.size()-2][0], temp_trajs_ori[temp_trajs_ori.size()-2][1],temp_trajs_check[temp_trajs_check.size()-2][0], temp_trajs_check[temp_trajs_check.size()-2][1]);
            printf("ori last : (%f,%f), temp_trajs_ori last: (%f,%f), temp_trajs_check last: (%f,%f)\n", t1.back()[0], t1.back()[1],temp_trajs_ori.back()[0], temp_trajs_ori.back()[1],temp_trajs_check.back()[0], temp_trajs_check.back()[1]);
            printf("t1 size %zu, temp_trajs_ori size: %zu, temp_trajs_check size: %zu\n", t1.size(), temp_trajs_ori.size(), temp_trajs_check.size());
            printf("check temp_trajs_check last is cp: %d\n",check_pt_is_cp(temp_trajs_check.back(),critical_points_dec));//(0.001559,442.715650)
            //trajID_need_fix_next.insert(current_trajID);
            //trajs_dec[current_trajID] = temp_trajs_check;
            break;  
          }
          end_fix_index = std::min(end_fix_index + 10,t_config.max_length);
          //end_fix_index = std::min(end_fix_index + 10,static_cast<int>(t1.size() - 1));
        }
        else{
          //修正成功当前trajectory
          //printf("fix traj %zu successfully\n",current_trajID);
          //更新trajs_dec
          trajs_dec[current_trajID] = temp_trajs_check;       
        }
      }
    } 

    //汇总all_vertex_for_all_diff_traj
    for (const auto& local_set:local_all_vertex_for_all_diff_traj){
      all_vertex_for_all_diff_traj.insert(local_set.begin(), local_set.end());
    }
    
    
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

    // std::vector<std::vector<std::array<double, 2>>> trajs_dec_next(keys.size() * 4);
    // for (auto& traj : trajs_dec_next) {
    //     traj.reserve(expected_size); // 预分配容量
    // }
    std::vector<std::set<size_t>> thread_index_dec_next(totoal_thread);
    std::set<size_t> vertex_dec_next;
    //这个容易并行
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
    size_t outside_count = 0;
    size_t hit_max_iter = 0;
    size_t wrong = 0;
    size_t correct = 0;
    //这个for不需要并行
    auto comp_start = std::chrono::high_resolution_clock::now();
    for(size_t i=0;i < trajs_ori.size(); ++i){
      auto t1 = trajs_ori[i];
      auto t2 = trajs_dec[i];
      bool cond1 = get_cell_offset(t1.back().data(), DW, DH) == get_cell_offset(t2.back().data(), DW, DH);
      bool cond2 = t1.size() == t_config.max_length;
      bool cond3 = t2.size() == t_config.max_length;
      bool cond4 = t1.back()[0] == -1;
      bool cond5 = t2.back()[0] == -1;
      switch (obj)
      {
      case 0:
        if (cond4){ // original outside
          auto ori_last_inside = findLastNonNegativeOne(t1);
          auto dec_last_inside = findLastNonNegativeOne(t2);
          if (!cond5){ //dec inside
            wrong ++;
            trajID_need_fix_next.insert(i);
          }
          for(int j = t2.size()-1; j >= 0; --j){
            if (get_cell_offset(ori_last_inside.data(), DW, DH) == get_cell_offset(t2[j].data(), DW, DH)){
              break;
            }
            if (j == 0){ //not found
              wrong ++;
              trajID_need_fix_next.insert(i);
            }
          }
          // if (get_cell_offset(ori_last_inside.data(), DW, DH) != get_cell_offset(dec_last_inside.data(), DW, DH)){
          //   wrong ++;
          //   trajID_need_fix_next.insert(i);
          // }
        }
        else if (cond2){ //original reach limit
          continue;
        }
        else if (!cond2 && !cond4){ //inside and not reach max, ie found cp
          if (!cond1){
            wrong ++;
            trajID_need_fix_next.insert(i);
          }
        }
        break;
      
      case 1:
        if (cond4){ // original outside
          auto ori_last_inside = findLastNonNegativeOne(t1);
          auto dec_last_inside = findLastNonNegativeOne(t2);
          if (get_cell_offset(ori_last_inside.data(), DW, DH) != get_cell_offset(dec_last_inside.data(), DW, DH)){
            wrong ++;
            trajID_need_fix_next.insert(i);
            // printf("Trajectory %ld is wrong!!!\n", i);
            // printf("first ori:(%f,%f), second ori(%f,%f): last ori:(%f,%f)\n",t1[0][0],t1[0][1],t1[1][0],t1[1][1],t1.back()[0],t1.back()[1]);
            // printf("first dec:(%f,%f), second dec(%f,%f): last dec:(%f,%f)\n",t2[0][0],t2[0][1],t2[1][0],t2[1][1],t2.back()[0],t2.back()[1]);
            // printf("ori length: %zu, dec length: %zu\n", t1.size(), t2.size());
          }
        }
        else if (cond2){ //original reach limit
          auto dec_last_inside = findLastNonNegativeOne(t2);
          if (get_cell_offset(t1.back().data(), DW, DH) != get_cell_offset(dec_last_inside.data(), DW, DH)){
            wrong ++;
            trajID_need_fix_next.insert(i);
            // printf("Trajectory %ld is wrong!!!\n", i);
            // printf("first ori:(%f,%f), second ori(%f,%f): last ori:(%f,%f)\n",t1[0][0],t1[0][1],t1[1][0],t1[1][1],t1.back()[0],t1.back()[1]);
            // printf("first dec:(%f,%f), second dec(%f,%f): last dec:(%f,%f)\n",t2[0][0],t2[0][1],t2[1][0],t2[1][1],t2.back()[0],t2.back()[1]);
            // printf("ori length: %zu, dec length: %zu\n", t1.size(), t2.size());
          }
        }
        else if (!cond2 && !cond4){ //inside and not reach max, ie found cp
          auto ori_last_inside = findLastNonNegativeOne(t1);
          auto dec_last_inside = findLastNonNegativeOne(t2);
          if (get_cell_offset(ori_last_inside.data(), DW, DH) != get_cell_offset(dec_last_inside.data(), DW, DH)){
            wrong ++;
            trajID_need_fix_next.insert(i);
            // printf("Trajectory %ld is wrong!!!\n", i);
            // printf("first ori:(%f,%f), second ori(%f,%f): last ori:(%f,%f)\n",t1[0][0],t1[0][1],t1[1][0],t1[1][1],t1.back()[0],t1.back()[1]);
            // printf("first dec:(%f,%f), second dec(%f,%f): last dec:(%f,%f)\n",t2[0][0],t2[0][1],t2[1][0],t2[1][1],t2.back()[0],t2.back()[1]);
            // printf("ori length: %zu, dec length: %zu\n", t1.size(), t2.size());
          }
        }
        break;
      
      case 2:
        // if (cond2 && cond3){
        //   continue;
        // }
        // else if (cond4 && cond5){
        //   continue;
        // }
        // else{
        //   if (!cond1){
        //     wrong ++;
        //     trajID_need_fix_next.insert(i);
        //     // printf("Trajectory %ld is wrong!!!\n", i);
        //     // printf("first ori:(%f,%f), second ori(%f,%f): last ori:(%f,%f)\n",t1[0][0],t1[0][1],t1[1][0],t1[1][1],t1.back()[0],t1.back()[1]);
        //     // printf("first dec:(%f,%f), second dec(%f,%f): last dec:(%f,%f)\n",t2[0][0],t2[0][1],t2[1][0],t2[1][1],t2.back()[0],t2.back()[1]);
        //     // printf("ori length: %zu, dec length: %zu\n", t1.size(), t2.size());
        //   }
        // }
        if (cond4){
          //ori outside
          // auto last_inside = findLastNonNegativeOne(t1);
          // auto last_inside_dec = findLastNonNegativeOne(t2);
          // if (get_cell_offset(last_inside.data(), DW, DH) != get_cell_offset(last_inside_dec.data(), DW, DH)){
          //   wrong ++;
          //   trajID_need_fix_next.insert(i);
          // }
          auto ori_last_inside = findLastNonNegativeOne(t1);
          auto dec_last_inside = findLastNonNegativeOne(t2);
          if (!cond5){ //dec inside
            wrong ++;
            trajID_need_fix_next.insert(i);
          }
          for(int j = t2.size()-1; j >= 0; --j){
            if (get_cell_offset(ori_last_inside.data(), DW, DH) == get_cell_offset(t2[j].data(), DW, DH)){
              break;
            }
            if (j == 0){ //not found
              wrong ++;
              trajID_need_fix_next.insert(i);
            }
          }
        }
        else if (cond2){
          //ori reach max, dec should reach max as well
          if (t2.size() != t_config.max_length){
            wrong ++;
            trajID_need_fix_next.insert(i);
          }
        }
        else if (!cond2 && !cond4){
          //ori inside, not reach max, ie found cp
          if (get_cell_offset(t1.back().data(), DW, DH) != get_cell_offset(t2.back().data(), DW, DH)){
            wrong ++;
            trajID_need_fix_next.insert(i);
          }
        }
        break;

      }

    }
    //printf("wrong: %ld\n", wrong);
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
      //清空trajID_need_fix，然后把trajID_need_fix_next赋值给trajID_need_fix
      trajID_need_fix.clear();
      //printf("before change trajID_need_fix size(should be 0): %ld\n", trajID_need_fix.size());
      for(auto o:trajID_need_fix_next){
        trajID_need_fix.insert(o);
      }
      if (trajID_need_fix.size() < 5){
        for (auto o:trajID_need_fix){
          printf("trajID %d\n", o);
          printf("ori first: (%f,%f), dec first: (%f,%f)\n", trajs_ori[o][0][0], trajs_ori[o][0][1], trajs_dec[o][0][0], trajs_dec[o][0][1]);
          printf("ori last-1: (%f,%f), dec last-1: (%f,%f)\n", trajs_ori[o][trajs_ori[o].size()-2][0], trajs_ori[o][trajs_ori[o].size()-2][1], trajs_dec[o][trajs_dec[o].size()-2][0], trajs_dec[o][trajs_dec[o].size()-2][1]);
          printf("ori last: (%f,%f), dec last: (%f,%f)\n", trajs_ori[o].back()[0], trajs_ori[o].back()[1], trajs_dec[o].back()[0], trajs_dec[o].back()[1]);
          printf("ori size: %zu, dec size: %zu\n", trajs_ori[o].size(), trajs_dec[o].size());
        }
      }
      trajID_need_fix_next.clear();
      //printf("after change trajID_need_fix size(should be 1+ wrong): %ld\n", trajID_need_fix.size());
    }
    

  } while (stop == false);
  
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

  for(auto t:trajID_need_fix_next_vec){
    printf("trajID_need_fix_next: %d\n", t);
  }


  //so far, all trajectories should be fixed
  printf("all_vertex_for_all_diff_traj size: %ld\n", all_vertex_for_all_diff_traj.size());

  //check compression ratio
  unsigned char * final_result = NULL;
  size_t final_result_size = 0;
  final_result = sz_compress_cp_preserve_2d_record_vertex(U, V, r1, r2, final_result_size, false, max_pwr_eb, all_vertex_for_all_diff_traj);

  printf("checkpt1\n");
  unsigned char * result_after_zstd = NULL;
  size_t result_after_zstd_size = sz_lossless_compress(ZSTD_COMPRESSOR, 3, final_result, final_result_size, &result_after_zstd);
  printf("checkpt2\n");
  printf("BEGIN Compression ratio = %f\n", cr_ori);
  printf("FINAL Compressed size = %zu, ratio = %f\n", result_after_zstd_size, (2*r1*r2*sizeof(float)) * 1.0/result_after_zstd_size);
  printf("====================================\n");
  printf("%d\n",current_round);
  printf("%f\n",elapsed_fix.count());
  printf("%f\n",(traj_ori_begin_elapsed.count() + traj_dec_begin_elapsed.count()));
  printf("%f\n",init_queue_elapsed.count());
  printf("%f\n",std::accumulate(index_time_vec.begin(), index_time_vec.end(), 0.0));
  printf("%f\n",std::accumulate(re_cal_trajs_time_vec.begin(), re_cal_trajs_time_vec.end(), 0.0));
  printf("%f\n",std::accumulate(compare_time_vec.begin(), compare_time_vec.end(), 0.0));
  printf("%f\n",result_after_zstd_size, (2*r1*r2*sizeof(float)) * 1.0/result_after_zstd_size);
  printf("====================================\n");
  free(final_result);
  size_t zstd_decompressed_size = sz_lossless_decompress(ZSTD_COMPRESSOR, result_after_zstd, result_after_zstd_size, &final_result, final_result_size);
  //printf("final lossless output size %zu, final_result_size %zu\n",final_lossless_output,final_result_size);//should be same with cpsz出来的大小
  // printf("checkpt3\n");
  // free(final_result);
  float * final_dec_U = NULL;
  float * final_dec_V = NULL;
  sz_decompress_cp_preserve_2d_online_record_vertex<float>(final_result, r1, r2, final_dec_U, final_dec_V);
  printf("verifying...\n");
  //verify(U, final_dec_U, r1*r2);
  double lossless_sum_u = 0;
  for(auto p:all_vertex_for_all_diff_traj){
    lossless_sum_u += U[p];
  }
  printf("lossless_sum_u_for_all_vertex_for_all_diff_traj: %f\n", lossless_sum_u);
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
  std::vector<std::vector<std::array<double, 2>>> final_check_ori;
  std::vector<std::vector<std::array<double, 2>>> final_check_dec;
  std::vector<int> index_tmp;
  std::vector<int> index_final;
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
        std::vector<std::array<double, 2>> result_return_ori = trajectory(pos, seed, directions[i][0] * t_config.h, t_config.max_length, DH, DW, critical_points_final, grad_ori, index_tmp);
        std::vector<std::array<double, 2>> result_return_dec = trajectory(pos, seed, directions[i][0] * t_config.h, t_config.max_length, DH, DW, critical_points_final, grad_final, index_final);
        final_check_ori.push_back(result_return_ori);
        final_check_dec.push_back(result_return_dec);
        switch (obj)
        {
        case 0:
          if (result_return_ori.back()[0] == -1){
            if(result_return_dec.back()[0] != -1){
              printf("some trajectories not fixed(case0-0)\n");
              printf("ori length: %zu, dec length: %zu\n", result_return_ori.size(), result_return_dec.size());
              printf("ori first %f,%f, dec first %f,%f\n", result_return_ori[0][0], result_return_ori[0][1], result_return_dec[0][0], result_return_dec[0][1]);
              printf("ori last-1 %f,%f, dec last-1 %f,%f\n", result_return_ori[result_return_ori.size()-2][0], result_return_ori[result_return_ori.size()-2][1], result_return_dec[result_return_dec.size()-2][0], result_return_dec[result_return_dec.size()-2][1]);
              printf("ori last %f,%f, dec last %f,%f\n", result_return_ori.back()[0], result_return_ori.back()[1], result_return_dec.back()[0], result_return_dec.back()[1]);
              exit(0);
            }
            auto ori_last_inside = findLastNonNegativeOne(result_return_ori);
            auto dec_last_inside = findLastNonNegativeOne(result_return_dec);
            for(int j = result_return_dec.size()-1; j >= 0; --j){
              if (get_cell_offset(ori_last_inside.data(), DW, DH) == get_cell_offset(result_return_dec[j].data(), DW, DH)){
                break;
              }
              if (j == 0){ //not found
                printf("some trajectories not fixed(case0-1)\n");
                printf("ori length: %zu, dec length: %zu\n", result_return_ori.size(), result_return_dec.size());
                printf("ori first %f,%f, dec first %f,%f\n", result_return_ori[0][0], result_return_ori[0][1], result_return_dec[0][0], result_return_dec[0][1]);
                printf("ori last-1 %f,%f, dec last-1 %f,%f\n", result_return_ori[result_return_ori.size()-2][0], result_return_ori[result_return_ori.size()-2][1], result_return_dec[result_return_dec.size()-2][0], result_return_dec[result_return_dec.size()-2][1]);
                printf("ori last %f,%f, dec last %f,%f\n", result_return_ori.back()[0], result_return_ori.back()[1], result_return_dec.back()[0], result_return_dec.back()[1]);
                exit(0);
              }
            }
            // if (get_cell_offset(ori_last_inside.data(), DW, DH) != get_cell_offset(dec_last_inside.data(), DW, DH)){
            //   printf("some trajectories not fixed(case0-0)\n");
            //   printf("ori length: %zu, dec length: %zu\n", result_return_ori.size(), result_return_dec.size());
            //   printf("ori first %f,%f, dec first %f,%f\n", result_return_ori[0][0], result_return_ori[0][1], result_return_dec[0][0], result_return_dec[0][1]);
            //   printf("ori last-1 %f,%f, dec last-1 %f,%f\n", result_return_ori[result_return_ori.size()-2][0], result_return_ori[result_return_ori.size()-2][1], result_return_dec[result_return_dec.size()-2][0], result_return_dec[result_return_dec.size()-2][1]);
            //   printf("ori last %f,%f, dec last %f,%f\n", result_return_ori.back()[0], result_return_ori.back()[1], result_return_dec.back()[0], result_return_dec.back()[1]);
            //   exit(0);
            // }
          }
          else if (result_return_ori.size() == t_config.max_length){
            continue;
          }
          else if (result_return_ori.size() != t_config.max_length && result_return_ori.back()[0] != -1){
            //inside and not reach max, ie found cp
            if (get_cell_offset(result_return_ori.back().data(), DW, DH) != get_cell_offset(result_return_dec.back().data(), DW, DH)){
              printf("some trajectories not fixed(case0-2\n");
              printf("ori length: %zu, dec length: %zu\n", result_return_ori.size(), result_return_dec.size());
              printf("ori first %f,%f, dec first %f,%f\n", result_return_ori[0][0], result_return_ori[0][1], result_return_dec[0][0], result_return_dec[0][1]);
              printf("ori last-1 %f,%f, dec last-1 %f,%f\n", result_return_ori[result_return_ori.size()-2][0], result_return_ori[result_return_ori.size()-2][1], result_return_dec[result_return_dec.size()-2][0], result_return_dec[result_return_dec.size()-2][1]);
              printf("ori last %f,%f, dec last %f,%f\n", result_return_ori.back()[0], result_return_ori.back()[1], result_return_dec.back()[0], result_return_dec.back()[1]);
              exit(0);
            }
          }
          break;
        
        case 2:
          if (result_return_ori.back()[0] == -1){
            if(result_return_dec.back()[0] != -1){
              printf("some trajectories not fixed(case2-0)\n");
              printf("ori length: %zu, dec length: %zu\n", result_return_ori.size(), result_return_dec.size());
              printf("ori first %f,%f, dec first %f,%f\n", result_return_ori[0][0], result_return_ori[0][1], result_return_dec[0][0], result_return_dec[0][1]);
              printf("ori last-1 %f,%f, dec last-1 %f,%f\n", result_return_ori[result_return_ori.size()-2][0], result_return_ori[result_return_ori.size()-2][1], result_return_dec[result_return_dec.size()-2][0], result_return_dec[result_return_dec.size()-2][1]);
              printf("ori last %f,%f, dec last %f,%f\n", result_return_ori.back()[0], result_return_ori.back()[1], result_return_dec.back()[0], result_return_dec.back()[1]);
              exit(0);
            }
            auto ori_last_inside = findLastNonNegativeOne(result_return_ori);
            auto dec_last_inside = findLastNonNegativeOne(result_return_dec);
            for(int j = result_return_dec.size()-1; j >= 0; --j){
              if (get_cell_offset(ori_last_inside.data(), DW, DH) == get_cell_offset(result_return_dec[j].data(), DW, DH)){
                break;
              }
              if (j == 0){ //not found
                printf("some trajectories not fixed(case2-1)\n");
                printf("ori length: %zu, dec length: %zu\n", result_return_ori.size(), result_return_dec.size());
                printf("ori first %f,%f, dec first %f,%f\n", result_return_ori[0][0], result_return_ori[0][1], result_return_dec[0][0], result_return_dec[0][1]);
                printf("ori last-1 %f,%f, dec last-1 %f,%f\n", result_return_ori[result_return_ori.size()-2][0], result_return_ori[result_return_ori.size()-2][1], result_return_dec[result_return_dec.size()-2][0], result_return_dec[result_return_dec.size()-2][1]);
                printf("ori last %f,%f, dec last %f,%f\n", result_return_ori.back()[0], result_return_ori.back()[1], result_return_dec.back()[0], result_return_dec.back()[1]);
                exit(0);
              }
            }
            // if (get_cell_offset(ori_last_inside.data(), DW, DH) != get_cell_offset(dec_last_inside.data(), DW, DH)){
            //   printf("some trajectories not fixed(case2-0)\n");
            //   printf("ori length: %zu, dec length: %zu\n", result_return_ori.size(), result_return_dec.size());
            //   printf("ori first %f,%f, dec first %f,%f\n", result_return_ori[0][0], result_return_ori[0][1], result_return_dec[0][0], result_return_dec[0][1]);
            //   printf("ori last-1 %f,%f, dec last-1 %f,%f\n", result_return_ori[result_return_ori.size()-2][0], result_return_ori[result_return_ori.size()-2][1], result_return_dec[result_return_dec.size()-2][0], result_return_dec[result_return_dec.size()-2][1]);
            //   printf("ori last %f,%f, dec last %f,%f\n", result_return_ori.back()[0], result_return_ori.back()[1], result_return_dec.back()[0], result_return_dec.back()[1]);
            //   exit(0);
            // }
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
          else if (result_return_ori.size() != t_config.max_length && result_return_ori.back()[0] != -1){
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
    // save_trajs_to_binary(final_check_ori, file_dir + "ori_traj.bin");
    // save_trajs_to_binary(final_check_dec, file_dir + "dec_traj.bin");
  }

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
  traj_config t_config = {h, eps, max_length};
  //double max_eb = 0.01;
  //int objectives = 0;
  int obj = atoi(argv[9]);
  int total_thread = atoi(argv[10]);
  std::string file_out_dir = argv[11];
  omp_set_num_threads(total_thread);
  /*
  objectives0: only garantee those trajectories that reach cp are correct
  objectives1: object0 + garantee those trajectories that reach max_length has same ending cell +
                those trajectories that outside the domain has same ending cell

  */
  //fix_traj(u, v,DH, DW, max_eb, t_config, obj);
  fix_traj_v2(u, v,DH, DW, max_eb, t_config, total_thread,obj, file_out_dir);

  free(u);
  free(v);

}





