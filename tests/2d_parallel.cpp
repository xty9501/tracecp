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
    std::string file_out_dir;
    if (argc == 15){
    file_out_dir = argv[14];
    }
    else{
    file_out_dir = "";
    }
    double current_pwr_eb = 0;
    omp_set_num_threads(total_thread);
    size_t result_size = 0;
    unsigned char * result = NULL;
    std::set<size_t> all_vertex_for_all_diff_traj = {};
    result = omp_sz_compress_cp_preserve_2d_fix(u, v, DH, DW, result_size, false, max_eb,all_vertex_for_all_diff_traj, total_thread);
    
}