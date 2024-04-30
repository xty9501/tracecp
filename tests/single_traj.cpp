#include "utils.hpp"
#include "cp.hpp"
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


#include "sz_cp_preserve_utils.hpp"
#include "sz_compress_cp_preserve_2d.hpp"
#include "sz_decompress_cp_preserve_2d.hpp"
#include "sz_lossless.hpp"
#include <iostream> 
#include <algorithm>



int main(int argc, char **argv){
    size_t num = 0;
    double eps = 0.01;
    double time_step = 0.05;
    ftk::ndarray<float> grad; //grad是三纬，第一个纬度是2，代表着u或者v，第二个纬度是DH，第三个纬度是DW
    ftk::ndarray<float> grad_out;
    float * u = readfile<float>(argv[1], num);
    float * v = readfile<float>(argv[2], num);
    float * dec_u = readfile<float>(argv[3], num);
    float * dec_v = readfile<float>(argv[4], num);
    int DW = atoi(argv[5]); //1800
    int DH = atoi(argv[6]); //1200
    double x_start = atof(argv[7]); 
    double y_start = atof(argv[8]);
    double max_eb = 0.01;
    grad.reshape({2, static_cast<unsigned long>(DW), static_cast<unsigned long>(DH)});
    refill_gradient(0, DH, DW, u, grad);
    refill_gradient(1, DH, DW, v, grad);
    grad_out.reshape({2, static_cast<unsigned long>(DW), static_cast<unsigned long>(DH)});
    refill_gradient(0, DH, DW, dec_u, grad_out);
    refill_gradient(1, DH, DW, dec_v, grad_out);

    std::vector<std::vector<std::array<double, 2>>> traj_ori;
    std::vector<std::vector<std::array<double, 2>>> traj_dec;
    std::vector<int> traj_index_ori;
    std::vector<int> traj_index_dec;
    std::vector<std::array<double,2>> X_all_direction; 

  //check cp
    auto critical_points_0 =compute_critical_points(u, v, DH, DW);
    auto critical_points_dec =compute_critical_points(dec_u, dec_v, DH, DW);
    double start_c[2];
    start_c[0] = x_start; start_c[1] = y_start;
    size_t simplex_offset = get_cell_offset(start_c, DW, DH);
    if(critical_points_0.find(simplex_offset) != critical_points_0.end()){
        std::cout<<"simplex_offset is a critical point"<<std::endl;
        if(critical_points_dec.find(simplex_offset) != critical_points_dec.end()){
            std::cout<<"simplex_offset is a critical point in dec"<<std::endl;
            printf("original eigenvector1: %f, %f\n", critical_points_0[simplex_offset].eig_vec[0][0], critical_points_0[simplex_offset].eig_vec[0][1]);
            printf("original eigenvector2: %f, %f\n", critical_points_0[simplex_offset].eig_vec[1][0], critical_points_0[simplex_offset].eig_vec[1][1]);
            printf("decompressed eigenvector1: %f, %f\n", critical_points_dec[simplex_offset].eig_vec[0][0], critical_points_dec[simplex_offset].eig_vec[0][1]);
            printf("decompressed eigenvector2: %f, %f\n", critical_points_dec[simplex_offset].eig_vec[1][0], critical_points_dec[simplex_offset].eig_vec[1][1]);
            //get seed
            auto cp = critical_points_0[simplex_offset];
            auto cp_dec = critical_points_dec[simplex_offset];
            double seed_coord0[2] = {cp.x[0] + eps*cp.eig_vec[0][0], cp.x[1] + eps*cp.eig_vec[0][1]};
            double seed_coord1[2] = {cp.x[0] - eps*cp.eig_vec[0][0], cp.x[1] - eps*cp.eig_vec[0][1]};
            double seed_coord2[2] = {cp.x[0] + eps*cp.eig_vec[1][0], cp.x[1] + eps*cp.eig_vec[1][1]};
            double seed_coord3[2] = {cp.x[0] - eps*cp.eig_vec[1][0], cp.x[1] - eps*cp.eig_vec[1][1]};
            printf("cp coord: %f, %f\n", cp.x[0], cp.x[1]);
            printf("seed coord 0: %f, %f\n", seed_coord0[0], seed_coord0[1]);
            printf("seed coord 1: %f, %f\n", seed_coord1[0], seed_coord1[1]);
            printf("seed coord 2: %f, %f\n", seed_coord2[0], seed_coord2[1]);
            printf("seed coord 3: %f, %f\n", seed_coord3[0], seed_coord3[1]);

            //now get trajectory
            X_all_direction.push_back({cp.x[0] + eps*cp.eig_vec[0][0], cp.x[1] + eps*cp.eig_vec[0][1]});
            X_all_direction.push_back({cp.x[0] - eps*cp.eig_vec[0][0], cp.x[1] - eps*cp.eig_vec[0][1]});
            X_all_direction.push_back({cp.x[0] + eps*cp.eig_vec[1][0], cp.x[1] + eps*cp.eig_vec[1][1]});
            X_all_direction.push_back({cp.x[0] - eps*cp.eig_vec[1][0], cp.x[1] - eps*cp.eig_vec[1][1]});
            double original_cp_coord[2] = {cp.x[0], cp.x[1]};
            for (int i = 0; i < 4; i ++) {
                if (i == 0 || i ==1){
                    std::vector<std::array<double, 2>> result = trajectory(original_cp_coord, X_all_direction[i],time_step,2000,DH,DW,critical_points_0,grad,traj_index_ori);
                    traj_ori.push_back(result);
                    std::vector<std::array<double, 2>> result_dec = trajectory(original_cp_coord, X_all_direction[i],time_step,2000,DH,DW,critical_points_0,grad_out,traj_index_dec);
                    traj_dec.push_back(result_dec);
                }
                else{
                    std::vector<std::array<double, 2>> result  = trajectory(original_cp_coord, X_all_direction[i],-time_step,2000,DH,DW,critical_points_0,grad,traj_index_ori);
                    traj_ori.push_back(result);
                    std::vector<std::array<double, 2>> result_dec = trajectory(original_cp_coord, X_all_direction[i],-time_step,2000,DH,DW,critical_points_0,grad_out,traj_index_dec);
                    traj_dec.push_back(result_dec);
                }
            }
        }
        else{
        std::cout<<"simplex_offset is not a critical point in dec"<<std::endl;
        exit(0);
        }
    }
    else{
        std::cout<<"simplex_offset is not a critical point"<<std::endl;
        exit(0);
    }

    //print first trajectory 20 points
    for (int i = 0; i < 4; i++){
        printf("%d th direction\n", i);
        for (int j = 0; j < 20; j++){
            printf("ori traj %d, %d: %f, %f\n", i, j, traj_ori[i][j][0], traj_ori[i][j][1]);
            printf("dec traj %d, %d: %f, %f\n", i, j, traj_dec[i][j][0], traj_dec[i][j][1]);
        }
        printf("ori traj last element: %f, %f\n", traj_ori[i][traj_ori[i].size()-1][0], traj_ori[i][traj_ori[i].size()-1][1]);
        printf("dec traj last element: %f, %f\n", traj_dec[i][traj_dec[i].size()-1][0], traj_dec[i][traj_dec[i].size()-1][1]);
    }

    //write dec trajectory to file
    std::string base = "/Users/mingzexia/Documents/Github/tracecp/data/";
    std::string filename = base +  "single_traj_" + std::string(argv[7]) + "_" + std::string(argv[8]) + ".bin";
    write_trajectory(traj_ori, filename);
    std::string filename_dec = base + "single_traj_dec_" + std::string(argv[7]) + "_" + std::string(argv[8]) + ".bin";
    write_trajectory(traj_dec, filename_dec);
    //write index to file
    std::string filename_index = base + "single_traj_index_" + std::string(argv[7]) + "_" + std::string(argv[8]) + ".bin";
    writefile(filename_index.c_str(), traj_index_ori.data(), traj_index_ori.size());
    std::string filename_index_dec = base + "single_traj_index_dec_" + std::string(argv[7]) + "_" + std::string(argv[8]) + ".bin";
    writefile(filename_index_dec.c_str(), traj_index_dec.data(), traj_index_dec.size());


}