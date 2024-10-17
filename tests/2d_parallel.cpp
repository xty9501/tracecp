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

// 比较两个 double 类型的坐标是否相同
bool arePointsEqual(const double a[2], const double b[2], double epsilon = 1e-6) {
  return std::fabs(a[0] - b[0]) < epsilon && std::fabs(a[1] - b[1]) < epsilon;
}

// 检查两个 unordered_map 是否具有相同的键，并且 critical_point 的 x 坐标一致
bool haveSameCriticalPointCoordinates(const std::unordered_map<size_t, critical_point_t>& map1, const std::unordered_map<size_t, critical_point_t>& map2) {
  // 1. 检查两个 map 的大小是否相同
  if (map1.size() != map2.size()) {
    return false;
  }

  // 2. 遍历 map1，检查 map2 是否具有相同键并且 x 坐标一致
  for (const auto& pair : map1) {
    size_t key = pair.first;
    const critical_point_t& cp1 = pair.second;

    // 检查 map2 中是否有相同的键
    auto it = map2.find(key);
    if (it == map2.end()) {
      return false;  // 如果 map2 中没有这个键，则不一致
    }

    // 检查 x 坐标是否一致
    const critical_point_t& cp2 = it->second;
    if (!arePointsEqual(cp1.x, cp2.x)) {
      return false;  // 如果 x 坐标不一致，则不一致
    }
  }

  // 3. 所有键和 x 坐标都一致
  return true;
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
    double sum = 0, prodSum = 0, relerr = 0;
    size_t max_index = -1;
    size_t max_rel_index = -1;
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
                max_rel_index = i;
        }

        if (diffMax < err){
            max_index = i;
            diffMax = err;
        }
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
    printf ("Max absolute error index: %ld\n", max_index);
    printf ("Max relative error = %f\n", diffMax/(Max-Min));
    printf ("Max relative error index: %ld\n", max_rel_index);
    printf ("Max pw relative error = %f\n", maxpw_relerr);
    printf ("PSNR = %f, NRMSE= %.20G\n", psnr,nrmse);
    printf ("acEff=%f\n", acEff);   
}

int main(int argc, char **argv){
    ftk::ndarray<float> grad; //grad是三纬，第一个纬度是2，代表着u或者v，第二个纬度是DH，第三个纬度是DW
    ftk::ndarray<float> grad_out;
    size_t num = 0;
    float * U = readfile<float>(argv[1], num);
    float * V = readfile<float>(argv[2], num);
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
    std::string file_out_dir;
    if (argc == 15){
    file_out_dir = argv[14];
    }
    else{
    file_out_dir = "";
    }
    /*
    double current_pwr_eb = 0;
    omp_set_num_threads(total_thread);
    size_t result_size = 0;
    unsigned char * result = NULL;
    std::set<size_t> all_vertex_for_all_diff_traj = {};
    auto cpsz_comp_start = std::chrono::high_resolution_clock::now();
    float * dec_inplace_U = NULL;
    float * dec_inplace_V = NULL;
    result = omp_sz_compress_cp_preserve_2d_record_vertex(U,V, DH, DW, result_size, false, max_eb,all_vertex_for_all_diff_traj, total_thread, dec_inplace_U, dec_inplace_V,eb_type);
    unsigned char * result_after_lossless = NULL;
    size_t lossless_outsize = sz_lossless_compress(ZSTD_COMPRESSOR, 3, result, result_size, &result_after_lossless);
    free(result);
    auto cpsz_comp_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpsz_comp_duration = cpsz_comp_end - cpsz_comp_start;
    printf("cpsz Compress time: %f\n", cpsz_comp_duration.count());

    // decomp
    auto cpsz_decomp_start = std::chrono::high_resolution_clock::now();
    size_t lossless_output = sz_lossless_decompress(ZSTD_COMPRESSOR, result_after_lossless, lossless_outsize, &result, result_size);
    float * dec_U = NULL;
    float * dec_V = NULL;
    omp_sz_decompress_cp_preserve_2d_online<float>(result, DH,DW, dec_U, dec_V); // use cpsz
    // calculate compression ratio
    printf("ori data size: %zu, lossless_output size: %zu\n", (2*DH*DW*sizeof(float)), lossless_output);
    printf("BEGIN Compressed size(original) = %zu, ratio = %f\n", lossless_outsize, (2*DH*DW*sizeof(float)) * 1.0/lossless_outsize);
    double cr_ori = (2*DH*DW*sizeof(float)) * 1.0/lossless_outsize; 

    auto cpsz_decomp_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpsz_decomp_duration = cpsz_decomp_end - cpsz_decomp_start;
    printf("cpsz Decompress time: %f\n", cpsz_decomp_duration.count());
    // check cp for original data and decompressed data
    auto critical_points_ori = compute_critical_points(U,V, DH,DW);
    auto critical_points_dec = compute_critical_points(dec_U, dec_V, DH,DW);
    auto critical_points_dec_inplace = compute_critical_points(dec_inplace_U, dec_inplace_V, DH,DW);
    if (critical_points_ori.size() != critical_points_dec.size()) {
        printf("Error: the number of critical points are different\n");
        printf("ori size: %ld, dec size: %ld\n", critical_points_ori.size(), critical_points_dec.size());
    }
    else{
        printf("same number of critical points, # cp = %ld\n", critical_points_ori.size());
    }
    // if (haveSameCriticalPointCoordinates(critical_points_ori, critical_points_dec)) {
    //     std::cout << "The two maps have the same critical point coordinates." << std::endl;
    // } else {
    //     std::cout << "The two maps do not have the same critical point coordinates." << std::endl;
    // }

    for (auto p : critical_points_ori) {
      size_t key = p.first;
      if (critical_points_dec.find(key) == critical_points_dec.end()) {
          printf("Error: key %ld not in dec\n", key);
      }
      else{
        // check if the critical point is the same
        auto cp_ori = p.second.x;
        auto cp_dec = critical_points_dec[key].x;
        if (cp_ori[0] != cp_dec[0] || cp_ori[1] != cp_dec[1]) {
            printf("Error: key %ld, cp is not the same\n", key);
            printf("ori: x: %f, y: %f\n", cp_ori[0], cp_ori[1]);
            printf("dec: x: %f, y: %f\n", cp_dec[0], cp_dec[1]);
        }
      }
    }
    */


    printf("=============check parallel=======================\n");
    size_t result_size = 0;
    unsigned char * result = NULL;
    float * dec_inplace_U = NULL;
    float * dec_inplace_V = NULL;
    std::set<size_t> all_vertex_for_all_diff_traj = {};
    auto cpsz_parallel_comp_start = std::chrono::high_resolution_clock::now();
    result = omp_sz_compress_cp_preserve_2d_record_vertex(U,V, DH, DW, result_size, false, max_eb,all_vertex_for_all_diff_traj,total_thread, dec_inplace_U, dec_inplace_V,eb_type);
    unsigned char * result_after_lossless = NULL;
    size_t lossless_outsize = sz_lossless_compress(ZSTD_COMPRESSOR, 3, result, result_size, &result_after_lossless);
    free(result);
    auto cpsz_parallel_comp_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpsz_parallel_comp_duration = cpsz_parallel_comp_end - cpsz_parallel_comp_start;
    printf("cpsz parallel Compress time: %f\n", cpsz_parallel_comp_duration.count());
    exit(0);
    size_t lossless_output = sz_lossless_decompress(ZSTD_COMPRESSOR, result_after_lossless, lossless_outsize, &result, result_size);
    float * dec_U = NULL;
    float * dec_V = NULL;
    omp_sz_decompress_cp_preserve_2d_online<float>(result, DH,DW, dec_U, dec_V);
    auto critical_points_ori = compute_critical_points(U, V,DH,DW);
    auto critical_points_dec = compute_critical_points(dec_inplace_U, dec_inplace_V, DH,DW);
    double nrmse = 0;
    // verify(U, dec_inplace_U, DH*DW, nrmse);
    // verify(V, dec_inplace_V, DH*DW, nrmse);
    verify(U, dec_U, DH*DW, nrmse);
    // exit(0);
    verify(V, dec_V, DH*DW, nrmse);
    
    if (critical_points_ori.size() != critical_points_dec.size()) {
        printf("Error: the number of critical points are different\n");
        printf("ori size: %ld, dec size: %ld\n", critical_points_ori.size(), critical_points_dec.size());
    }
    else{
        printf("same number of critical points, # cp = %ld\n", critical_points_ori.size());
    }
    for (auto p : critical_points_ori) {
      size_t key = p.first;
      if (critical_points_dec.find(key) == critical_points_dec.end()) {
          printf("Error: key %ld not in dec\n", key);
          printf("ori: x: %f, y: %f\n", p.second.x[0], p.second.x[1]);
      }
      else{
        // check if the critical point is the same
        auto cp_ori = p.second.x;
        auto cp_dec = critical_points_dec[key].x;
        if (cp_ori[0] != cp_dec[0] || cp_ori[1] != cp_dec[1]) {
            printf("Error: key %ld, cp is not the same\n", key);
            printf("ori: x: %f, y: %f\n", cp_ori[0], cp_ori[1]);
            printf("dec: x: %f, y: %f\n", cp_dec[0], cp_dec[1]);
            int floor_x = floor(cp_ori[0]);
            int floor_y = floor(cp_ori[1]);
            printf("U_left_bottom: %f, U_right_bottom: %f, U_left_top: %f, U_right_top: %f\n", U[floor_y*DW+floor_x], U[floor_y*DW+floor_x+1], U[(floor_y+1)*DW+floor_x], U[(floor_y+1)*DW+floor_x+1]);
            printf("dec_U_left_bottom: %f, dec_U_right_bottom: %f, dec_U_left_top: %f, dec_U_right_top: %f\n", dec_U[floor_y*DW+floor_x], dec_U[floor_y*DW+floor_x+1], dec_U[(floor_y+1)*DW+floor_x], dec_U[(floor_y+1)*DW+floor_x+1]);
            printf("dec_inplace_U_left_bottom: %f, dec_inplace_U_right_bottom: %f, dec_inplace_U_left_top: %f, dec_inplace_U_right_top: %f\n", dec_inplace_U[floor_y*DW+floor_x], dec_inplace_U[floor_y*DW+floor_x+1], dec_inplace_U[(floor_y+1)*DW+floor_x], dec_inplace_U[(floor_y+1)*DW+floor_x+1]);
            printf("V_left_bottom: %f, V_right_bottom: %f, V_left_top: %f, V_right_top: %f\n", V[floor_y*DW+floor_x], V[floor_y*DW+floor_x+1], V[(floor_y+1)*DW+floor_x], V[(floor_y+1)*DW+floor_x+1]);
            printf("dec_V_left_bottom: %f, dec_V_right_bottom: %f, dec_V_left_top: %f, dec_V_right_top: %f\n", dec_V[floor_y*DW+floor_x], dec_V[floor_y*DW+floor_x+1], dec_V[(floor_y+1)*DW+floor_x], dec_V[(floor_y+1)*DW+floor_x+1]);
            printf("dec_inplace_V_left_bottom: %f, dec_inplace_V_right_bottom: %f, dec_inplace_V_left_top: %f, dec_inplace_V_right_top: %f\n", dec_inplace_V[floor_y*DW+floor_x], dec_inplace_V[floor_y*DW+floor_x+1], dec_inplace_V[(floor_y+1)*DW+floor_x], dec_inplace_V[(floor_y+1)*DW+floor_x+1]);
        }
      }
    }
    //print compression ratio
    printf("parallel compression ratio: %f\n", (2*DH*DW*sizeof(float)) * 1.0/lossless_outsize);

    printf("=============check serial=======================\n");
    //check 串行和结果是否一致

    critical_points_ori = compute_critical_points(U, V,DH,DW);


    size_t result_size_serial = 0;
    unsigned char * result_serial = NULL; 
    if(eb_type == "rel"){
      //****original version of cpsz********
      result_serial = sz_compress_cp_preserve_2d_fix(U, V, DH, DW, result_size_serial, false, max_eb);
    }
    else if (eb_type == "abs"){
      //****** cpsz with absolute error bound ******
      result_serial = sz_compress_cp_preserve_2d_online_abs_record_vertex(U, V, DH, DW, result_size_serial, false,max_eb);
    }
    // result = sz_compress_cp_preserve_2d_fix(U, V, DH, DW, result_size, false, max_eb);
    unsigned char * result_after_lossless_serial = NULL;
    size_t lossless_outsize_serial = sz_lossless_compress(ZSTD_COMPRESSOR, 3, result_serial, result_size_serial, &result_after_lossless_serial);
    free(result_serial);
    size_t lossless_output_serial = sz_lossless_decompress(ZSTD_COMPRESSOR, result_after_lossless_serial, lossless_outsize_serial, &result_serial, result_size_serial);
    
    float * dec_U_serial = NULL;
    float * dec_V_serial = NULL;
    sz_decompress_cp_preserve_2d_online<float>(result_serial, DH,DW, dec_U_serial, dec_V_serial); // use cpsz
    
    auto critical_points_dec_serial = compute_critical_points(dec_U_serial, dec_V_serial, DH,DW);
    if (critical_points_ori.size() != critical_points_dec_serial.size()) {
        printf("Error: the number of critical points are different\n");
        printf("ori size: %ld, dec size: %ld\n", critical_points_ori.size(), critical_points_dec_serial.size());
    }
    else{
        printf("same number of critical points, # cp = %ld\n", critical_points_ori.size());
    }
    for (auto p : critical_points_ori) {
      size_t key = p.first;
      if (critical_points_dec_serial.find(key) == critical_points_dec_serial.end()) {
          printf("Error: key %ld not in dec\n", key);
      }
      else{
        // check if the critical point is the same
        auto cp_ori = p.second.x;
        auto cp_dec = critical_points_dec_serial[key].x;
        if (cp_ori[0] != cp_dec[0] || cp_ori[1] != cp_dec[1]) {
            printf("Error: key %ld, cp is not the same\n", key);
            printf("ori: x: %f, y: %f\n", cp_ori[0], cp_ori[1]);
            printf("dec: x: %f, y: %f\n", cp_dec[0], cp_dec[1]);
        }
      }
    }
    verify(U, dec_U_serial, DH*DW, nrmse);
    verify(V, dec_V_serial, DH*DW, nrmse);
    // print compression ratio
    printf("serial compression ratio: %f\n", (2*DH*DW*sizeof(float)) * 1.0/lossless_outsize_serial);


    // //determine which key dose not in ori
    // for (auto cp: critical_points_dec) {
    //     if (critical_points_ori.find(cp.first) == critical_points_ori.end()) {
    //         printf("key %ld not in ori\n", cp.first);
    //         printf("x: %f, y: %f, type: %d\n", cp.second.x[0], cp.second.x[1], cp.second.type);
    //     }
    // }



    // exit(0);
    // printf("critical points dec inplace: %ld\n", critical_points_dec_inplace.size());
    // writefile("dec_inplace_U", dec_inplace_U, DH*DW);
    // writefile("dec_inplace_V", dec_inplace_V, DH*DW);
    // writefile("dec_U", dec_U, DH*DW);
    // writefile("dec_V", dec_V, DH*DW);
    // //print first 10 different critical points key
    // int count = 0;

    // double nrmse_u;
    // verify(U, dec_U, DW*DH, nrmse_u);

    // // printf("same number of critical points\n");
    // exit(0);
}