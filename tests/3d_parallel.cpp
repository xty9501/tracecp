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

//using namespace std;

template<typename T>
static inline void 
update_index_and_value(double v[4][3], int local_id, int global_id, const T * U, const T * V, const T * W){
	v[local_id][0] = U[global_id];
	v[local_id][1] = V[global_id];
	v[local_id][2] = W[global_id];
}

template<typename T>
static int 
check_cp(T v[4][3]){
	double mu[4]; // check intersection
	double cond;
	double threshold = 0.0;
	bool succ = ftk::inverse_lerp_s3v3(v, mu, &cond, threshold);
	if(!succ) return -1;
	return 1;
}

template<typename T>
static vector<bool> 
compute_cp(const T * U, const T * V, const T * W, int r1, int r2, int r3){
	// check cp for all cells
	vector<bool> cp_exist(6*(r1-1)*(r2-1)*(r3-1), 0);
	ptrdiff_t dim0_offset = r2*r3;
	ptrdiff_t dim1_offset = r3;
	ptrdiff_t cell_dim0_offset = (r2-1)*(r3-1);
	ptrdiff_t cell_dim1_offset = r3-1;
	double v[4][3] = {0};
	for(int i=0; i<r1-1; i++){
		if(i%10==0) std::cout << i << " / " << r1-1 << std::endl;
		for(int j=0; j<r2-1; j++){
			for(int k=0; k<r3-1; k++){
				// order (reserved, z->x):
				ptrdiff_t cell_offset = 6*(i*cell_dim0_offset + j*cell_dim1_offset + k);
				// (ftk-0) 000, 001, 011, 111
				update_index_and_value(v, 0, i*dim0_offset + j*dim1_offset + k, U, V, W);
				update_index_and_value(v, 1, (i+1)*dim0_offset + j*dim1_offset + k, U, V, W);
				update_index_and_value(v, 2, (i+1)*dim0_offset + (j+1)*dim1_offset + k, U, V, W);
				update_index_and_value(v, 3, (i+1)*dim0_offset + (j+1)*dim1_offset + (k+1), U, V, W);
				cp_exist[cell_offset] = (check_cp(v) == 1);
				// (ftk-2) 000, 010, 011, 111
				update_index_and_value(v, 1, i*dim0_offset + (j+1)*dim1_offset + k, U, V, W);
				cp_exist[cell_offset + 1] = (check_cp(v) == 1);
				// (ftk-1) 000, 001, 101, 111
				update_index_and_value(v, 1, (i+1)*dim0_offset + j*dim1_offset + k, U, V, W);
				update_index_and_value(v, 2, (i+1)*dim0_offset + j*dim1_offset + k+1, U, V, W);
				cp_exist[cell_offset + 2] = (check_cp(v) == 1);
				// (ftk-4) 000, 100, 101, 111
				update_index_and_value(v, 1, i*dim0_offset + j*dim1_offset + k+1, U, V, W);
				cp_exist[cell_offset + 3] = (check_cp(v) == 1);
				// (ftk-3) 000, 010, 110, 111
				update_index_and_value(v, 1, i*dim0_offset + (j+1)*dim1_offset + k, U, V, W);
				update_index_and_value(v, 2, i*dim0_offset + (j+1)*dim1_offset + k+1, U, V, W);
				cp_exist[cell_offset + 4] = (check_cp(v) == 1);
				// (ftk-5) 000, 100, 110, 111
				update_index_and_value(v, 1, i*dim0_offset + j*dim1_offset + k+1, U, V, W);
				cp_exist[cell_offset + 5] = (check_cp(v) == 1);
			}
		}
	}	
	return cp_exist;	
}


template<typename T>
static std::vector<bool> 
omp_compute_cp(const T * U, const T * V, const T * W, int r1, int r2, int r3){
    // Number of cells in each dimension
    int cells_r1 = r1 - 1;
    int cells_r2 = r2 - 1;
    int cells_r3 = r3 - 1;
    // Total number of cells
    size_t total_cells = static_cast<size_t>(cells_r1) * cells_r2 * cells_r3;
    // Initialize cp_exist vector
    std::vector<bool> cp_exist(6 * total_cells, false);
    ptrdiff_t dim0_offset = r2 * r3;
    ptrdiff_t dim1_offset = r3;
    ptrdiff_t cell_dim0_offset = (r2 - 1) * (r3 - 1);
    ptrdiff_t cell_dim1_offset = r3 - 1;
    // Parallelize the outer loops using OpenMP
    #pragma omp parallel for collapse(2) schedule(static)
    for(int i = 0; i < r1 - 1; i++){
        for(int j = 0; j < r2 - 1; j++){
            for(int k = 0; k < r3 - 1; k++){
                double v[4][3];  // Thread-private variable
                // Calculate the cell offset
                ptrdiff_t cell_idx = i * cell_dim0_offset + j * cell_dim1_offset + k;
                ptrdiff_t cell_offset = 6 * cell_idx;
                // (ftk-0) 000, 001, 011, 111
                update_index_and_value(v, 0, i * dim0_offset + j * dim1_offset + k, U, V, W);
                update_index_and_value(v, 1, (i + 1) * dim0_offset + j * dim1_offset + k, U, V, W);
                update_index_and_value(v, 2, (i + 1) * dim0_offset + (j + 1) * dim1_offset + k, U, V, W);
                update_index_and_value(v, 3, (i + 1) * dim0_offset + (j + 1) * dim1_offset + (k + 1), U, V, W);
                cp_exist[cell_offset] = (check_cp(v) == 1);
                // (ftk-2) 000, 010, 011, 111
                update_index_and_value(v, 1, i * dim0_offset + (j + 1) * dim1_offset + k, U, V, W);
                cp_exist[cell_offset + 1] = (check_cp(v) == 1);
                // (ftk-1) 000, 001, 101, 111
                update_index_and_value(v, 1, (i + 1) * dim0_offset + j * dim1_offset + k, U, V, W);
                update_index_and_value(v, 2, (i + 1) * dim0_offset + j * dim1_offset + (k + 1), U, V, W);
                cp_exist[cell_offset + 2] = (check_cp(v) == 1);
                // (ftk-4) 000, 100, 101, 111
                update_index_and_value(v, 1, i * dim0_offset + j * dim1_offset + (k + 1), U, V, W);
                cp_exist[cell_offset + 3] = (check_cp(v) == 1);
                // (ftk-3) 000, 010, 110, 111
                update_index_and_value(v, 1, i * dim0_offset + (j + 1) * dim1_offset + k, U, V, W);
                update_index_and_value(v, 2, i * dim0_offset + (j + 1) * dim1_offset + (k + 1), U, V, W);
                cp_exist[cell_offset + 4] = (check_cp(v) == 1);
                // (ftk-5) 000, 100, 110, 111
                update_index_and_value(v, 1, i * dim0_offset + j * dim1_offset + (k + 1), U, V, W);
                cp_exist[cell_offset + 5] = (check_cp(v) == 1);
            }
        }
    }
    return cp_exist;
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
    double abserr = 0;

    double maxpw_relerr = 0; 
    double maxpw_abserr = 0;
    for (i = 0; i < num_elements; i++){
        if (Max < ori_data[i]) Max = ori_data[i];
        if (Min > ori_data[i]) Min = ori_data[i];
        
        Type err = fabs(data[i] - ori_data[i]);
        if(ori_data[i]!=0 && fabs(ori_data[i])>1)
        {
            relerr = err/fabs(ori_data[i]);
            if(maxpw_relerr<relerr)
                maxpw_relerr = relerr;
            abserr = err;
            if(maxpw_abserr<abserr)
                maxpw_abserr = abserr;
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
    printf("Max pw absolute error = %f\n", maxpw_abserr);
    printf ("PSNR = %f, NRMSE= %.20G\n", psnr,nrmse);
    printf ("acEff=%f\n", acEff);   
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

    double threshold = atof(argv[13]);
    double threshold_outside = atof(argv[14]);
    double threshold_max_iter = atof(argv[15]);

    // these two flags are used to control whether use the saved compressed data,to save computation time
    int readout_flag = 0;
    int writeout_flag = 0;
    int cp_flag = 1;

    std::chrono::duration<double> cpsz_comp_duration;
    std::chrono::duration<double> cpsz_decomp_duration;
    
    std::string file_out_dir = "";
    // if (argc == 16){
    // file_out_dir = argv[16];
    // }
    // int obj = 0;
    omp_set_num_threads(total_thread);
    float * dec_U = NULL;
    float * dec_V = NULL;
    float * dec_W = NULL;
    // pre-compute critical points
    // auto cp_cal_start = std::chrono::high_resolution_clock::now();
    // auto critical_points_0 = compute_critical_points(U, V, W, r1, r2, r3); //r1=DD,r2=DH,r3=DW
    // auto cp_cal_end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> cp_cal_duration = cp_cal_end - cp_cal_start;
    // cout << "critical points #: " << critical_points_0.size() << endl;
    
    double begin_cr = 0;

    size_t result_size = 0;
    struct timespec start, end;
    if (readout_flag == 1){
    dec_U = readfile<float>("/home/mxi235/data/temp_data/dec_U.bin", num_elements);
    dec_V = readfile<float>("/home/mxi235/data/temp_data/dec_V.bin", num_elements);
    dec_W = readfile<float>("/home/mxi235/data/temp_data/dec_W.bin", num_elements);
    }
    
    else{
        cout << "start Compression\n";
        unsigned char * result;
        auto comp_time_start = std::chrono::high_resolution_clock::now();

        float * dec_U_inplace = NULL;
        float * dec_V_inplace = NULL;
        float * dec_W_inplace = NULL;
        if(eb_type == "abs"){
        result = omp_sz_compress_cp_preserve_3d_online_abs_record_vertex(U,V,W, r1, r2, r3, result_size, max_eb,final_vertex_need_to_lossless, total_thread, dec_U_inplace, dec_V_inplace, dec_W_inplace);
        
        //单线程
        //result = sz_compress_cp_preserve_3d_online_abs_record_vertex(U,V,W, r1, r2, r3, result_size, max_eb,final_vertex_need_to_lossless);
        }
        else{
            printf("not support this eb_type\n");
            exit(0);
        }

        unsigned char * result_after_lossless = NULL;
        size_t lossless_outsize = sz_lossless_compress(ZSTD_COMPRESSOR, 3, result, result_size, &result_after_lossless);

        auto comp_time_end = std::chrono::high_resolution_clock::now();
        cpsz_comp_duration = comp_time_end - comp_time_start;
        printf("Compress time: %f\n", cpsz_comp_duration.count());
        begin_cr = (3*num_elements*sizeof(float)) * 1.0/lossless_outsize;
        cout << "Compressed size = " << lossless_outsize << ", ratio = " << (3*num_elements*sizeof(float)) * 1.0/lossless_outsize << endl;
        free(result);

        //decompression
        size_t lossless_output = sz_lossless_decompress(ZSTD_COMPRESSOR, result_after_lossless, lossless_outsize, &result, result_size);
        if (eb_type == "abs"){
            omp_sz_decompress_cp_preserve_3d_online_abs_record_vertex<float>(result, r1, r2, r3,dec_U, dec_V, dec_W);

            //单线程
            //sz_decompress_cp_preserve_3d_online_abs_record_vertex<float>(result, r1, r2, r3,dec_U, dec_V, dec_W);
        }
        else{
            printf("not support this eb_type\n");
            exit(0);
        }
        //now verify the decompressed data
        double nrmse_u, nrmse_v, nrmse_w;
        verify(U, dec_U, r1*r2*r3, nrmse_u);
        printf("====================================\n");
        verify(V, dec_V, r1*r2*r3, nrmse_v);
        printf("====================================\n");
        verify(W, dec_W, r1*r2*r3, nrmse_w);
        printf("====================================\n");

        //now check the critical points
        auto cp_exist_ori = omp_compute_cp(U, V, W, r1, r2, r3);
        auto cp_exist_dec = omp_compute_cp(dec_U, dec_V, dec_W, r1, r2, r3);
        printf("ori cp #: %ld, dec cp #: %ld\n", std::count(cp_exist_ori.begin(), cp_exist_ori.end(), true), std::count(cp_exist_dec.begin(), cp_exist_dec.end(), true));
    
        //check cp_exist_inplace
        // auto cp_exist_dec_inplace = compute_cp(dec_U_inplace, dec_V_inplace, dec_W_inplace, r1, r2, r3);
        // printf("dec inplace cp #: %ld\n", std::count(cp_exist_dec_inplace.begin(), cp_exist_dec_inplace.end(), true));
        // exit(0);    

        // if dec has more cp than ori, then we need to check the difference
        if (std::count(cp_exist_dec.begin(), cp_exist_dec.end(), true) > std::count(cp_exist_ori.begin(), cp_exist_ori.end(), true)){
            for (size_t i = 0; i < cp_exist_ori.size(); i++){
                if (cp_exist_ori[i] != cp_exist_dec[i]){
                    //coonvert key to coordinate
                    int x = i/(r2*r3);
                    int y = (i%(r2*r3))/r3;
                    int z = (i%(r2*r3))%r3;
                    printf("key %ld, x: %d, y: %d, z: %d\n", i, x, y, z);
                }
            }
        }
        // if dec has same number of cp with ori, check if they are the same
        else{
            for (size_t i = 0; i < cp_exist_ori.size(); i++){
                if (cp_exist_ori[i] != cp_exist_dec[i]){
                    //coonvert key to coordinate
                    int x = i/(r2*r3);
                    int y = (i%(r2*r3))/r3;
                    int z = (i%(r2*r3))%r3;
                    printf("key %ld, x: %d, y: %d, z: %d\n", i, x, y, z);
                }
            }
        }
    }
    

}