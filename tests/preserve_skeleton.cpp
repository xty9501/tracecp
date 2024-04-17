
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
#include <ftk/numeric/clamp.hh>
#include <ftk/numeric/eigen_solver2.hh>
#include <ftk/algorithms/cca.hh>
#include <ftk/geometry/cc2curves.hh>
#include <ftk/geometry/curve2tube.hh>
#include "ftk/ndarray.hh"
#include "ftk/numeric/critical_point_type.hh"
#include "ftk/numeric/critical_point_test.hh"
#include <chrono>

#include "sz_cp_preserve_utils.hpp"
#include "sz_compress_cp_preserve_2d.hpp"
#include "sz_decompress_cp_preserve_2d.hpp"
#include "sz_lossless.hpp"
#include <iostream> 
using namespace std;

double vector_field_resolution = std::numeric_limits<double>::max();
uint64_t vector_field_scaling_factor = 1;
// int DW = 128, DH = 128;// the dimensionality of the data is DW*DH
ftk::ndarray<double> grad; //grad是三纬，第一个纬度是2，代表着u或者v，第二个纬度是DH，第三个纬度是DW
ftk::simplicial_regular_mesh m(2);


size_t global_count = 0;


typedef struct critical_point_t{
  double x[2];
  double eig_vec[2][2];
  double V[3][2];
  double X[3][2];
  // double mu[3];
  int type;
  size_t simplex_id;
  critical_point_t(double* x_, double eig_v[2][2], double V_[3][2],double X_[3][2], int t_, size_t simplex_id_){
    x[0] = x_[0];
    x[1] = x_[1];
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
    // for (int i = 0; i < 3; i++) {
    //   mu[i] = mu_[i];
    // }
    type = t_;
    simplex_id = simplex_id_;
  }
  critical_point_t(){}
}critical_point_t;



// std::unordered_map<int, critical_point_t> critical_points;
 
// #define DEFAULT_EB 1
#define SINGULAR 0
#define ATTRACTING 1 // 2 real negative eigenvalues
#define REPELLING 2 // 2 real positive eigenvalues
#define SADDLE 3// 1 real negative and 1 real positive
#define ATTRACTING_FOCUS 4 // complex with negative real
#define REPELLING_FOCUS 5 // complex with positive real
#define CENTER 6 // complex with 0 real

std::string get_critical_point_type_string(int type){
  switch(type){
    case 0:
      return "SINGULAR";
    case 1:
      return "ATTRACTING";
    case 2:
      return "REPELLING";
    case 3:
      return "SADDLE";
    case 4:
      return "ATTRACTING_FOCUS";
    case 5:
      return "REPELLING_FOCUS";
    case 6:
      return "CENTER";
    default:
      return "INVALID";
  }
}


void refill_gradient(int id,const int DH,const int DW, const float* grad_tmp){
  const float * grad_tmp_pos = grad_tmp;
  for (int i = 0; i < DH; i ++) {
    for (int j = 0; j < DW; j ++) {
      grad(id, j, i) = *(grad_tmp_pos ++);
    }
  }
}



template<typename T_fp>
static void 
check_simplex_seq_saddle(const T_fp vf[3][2], const double v[3][2], const double X[3][2], const int indices[3], int i, int j, int simplex_id, std::unordered_map<int, critical_point_t>& critical_points){
  // robust critical point test
  // bool succ = ftk::robust_critical_point_in_simplex2(vf, indices);
  // if (!succ) return;
  double mu[3]; // check intersection
  double cond;
  for(int i=0; i<3; i++){ //skip if any of the vertex is 0
    if((v[i][0] == 0) && (v[i][1] == 0)){
      return;
    }
  }
  bool succ2 = ftk::inverse_lerp_s2v2(v, mu, &cond);
  // if (!succ2) ftk::clamp_barycentric<3>(mu);
  if (!succ2) return;
  double x[2]; // position
  ftk::lerp_s2v2(X, mu, x);
  double J[2][2]; // jacobian
  ftk::jacobian_2dsimplex2(X, v, J);  
  int cp_type = 0;
  std::complex<double> eig[2];
  double delta = ftk::solve_eigenvalues2x2(J, eig);
  // if(fabs(delta) < std::numeric_limits<double>::epsilon())
  if (delta >= 0) { // two real roots
    if (eig[0].real() * eig[1].real() < 0) {
      cp_type = SADDLE;
    } else if (eig[0].real() < 0) {
      cp_type = ATTRACTING;
    }
    else if (eig[0].real() > 0){
      cp_type = REPELLING;
    }
    else cp_type = SINGULAR;
  } else { // two conjugate roots
    if (eig[0].real() < 0) {
      cp_type = ATTRACTING_FOCUS;
    } else if (eig[0].real() > 0) {
      cp_type = REPELLING_FOCUS;
    } else 
      cp_type = CENTER;
  }
  // if (cp_type == SINGULAR) return;
  critical_point_t cp;
  // critical_point_t cp(x, eig_vec,v,X, cp_type);
  cp.x[0] = j + x[0]; cp.x[1] = i + x[1];
  double eig_vec[2][2]={0};
  double eig_r[2];
  eig_r[0] = eig[0].real(), eig_r[1] = eig[1].real();
  ftk::solve_eigenvectors2x2(J, 2, eig_r, eig_vec);
  cp.eig_vec[0][0] = eig_vec[0][0]; cp.eig_vec[0][1] = eig_vec[0][1];
  cp.eig_vec[1][0] = eig_vec[1][0]; cp.eig_vec[1][1] = eig_vec[1][1];
  cp.V[0][0] = v[0][0]; cp.V[0][1] = v[0][1];
  cp.V[1][0] = v[1][0]; cp.V[1][1] = v[1][1];
  cp.V[2][0] = v[2][0]; cp.V[2][1] = v[2][1];
  cp.X[0][0] = X[0][0]; cp.X[0][1] = X[0][1];
  cp.X[1][0] = X[1][0]; cp.X[1][1] = X[1][1];
  cp.X[2][0] = X[2][0]; cp.X[2][1] = X[2][1];
  cp.type = cp_type;
  cp.simplex_id = simplex_id;
  // cp.v = mu[0]*v[0][0] + mu[1]*v[1][0] + mu[2]*v[2][0];
  // cp.u = mu[0]*v[0][1] + mu[1]*v[1][1] + mu[2]*v[2][1];
  critical_points[simplex_id] = cp;


  
  
}

template<typename T>
std::unordered_map<int, critical_point_t>
compute_critical_points(const T * U, const T * V, int r1, int r2, uint64_t vector_field_scaling_factor){
  // check cp for all cells
  using T_fp = int64_t;
  size_t num_elements = r1*r2;
  T_fp * U_fp = (T_fp *) malloc(num_elements * sizeof(T_fp));
  T_fp * V_fp = (T_fp *) malloc(num_elements * sizeof(T_fp));
  for(int i=0; i<num_elements; i++){
    U_fp[i] = U[i]*vector_field_scaling_factor;
    V_fp[i] = V[i]*vector_field_scaling_factor;
  }
  int indices[3] = {0};
  // __int128 vf[4][3] = {0};
	double X1[3][2] = {
		{0, 0},
		{0, 1},
		{1, 1}
	};
	double X2[3][2] = {
		{0, 0},
		{1, 0},
		{1, 1}
	};
  int64_t vf[3][2] = {0};
  double v[3][2] = {0};
  std::unordered_map<int, critical_point_t> critical_points;
	for(int i=1; i<r1-2; i++){
    if(i%100==0) std::cout << i << " / " << r1-1 << std::endl;
		for(int j=1; j<r2-2; j++){
      ptrdiff_t cell_offset = 2*(i * (r2-1) + j);
			indices[0] = i*r2 + j;
			indices[1] = (i+1)*r2 + j;
			indices[2] = (i+1)*r2 + (j+1); 
			// cell index 0
			for(int p=0; p<3; p++){
				vf[p][0] = U_fp[indices[p]];
				vf[p][1] = V_fp[indices[p]];
				v[p][0] = U[indices[p]];
				v[p][1] = V[indices[p]];
			}
      
      check_simplex_seq_saddle(vf, v, X1, indices, i, j, cell_offset, critical_points);
			// cell index 1
			indices[1] = i*r2 + (j+1);
			vf[1][0] = U_fp[indices[1]], vf[1][1] = V_fp[indices[1]];
			v[1][0] = U[indices[1]], v[1][1] = V[indices[1]];			
      check_simplex_seq_saddle(vf, v, X2, indices, i, j, cell_offset + 1, critical_points);
		}
	}
  free(U_fp);
  free(V_fp);
  return critical_points; 
}



template<typename T, typename T_fp>
static int64_t 
convert_to_fixed_point(const T * U, const T * V, size_t num_elements, T_fp * U_fp, T_fp * V_fp, T_fp& range, int type_bits=63){
	double vector_field_resolution = 0;
	int64_t vector_field_scaling_factor = 1;
	for (int i=0; i<num_elements; i++){
		double min_val = std::max(fabs(U[i]), fabs(V[i]));
		vector_field_resolution = std::max(vector_field_resolution, min_val);
	}
	int vbits = std::ceil(std::log2(vector_field_resolution));
	int nbits = (type_bits - 3) / 2;
	vector_field_scaling_factor = 1 << (nbits - vbits);
	std::cerr << "resolution=" << vector_field_resolution 
	<< ", factor=" << vector_field_scaling_factor 
	<< ", nbits=" << nbits << ", vbits=" << vbits << ", shift_bits=" << nbits - vbits << std::endl;
	int64_t max = std::numeric_limits<int64_t>::min();
	int64_t min = std::numeric_limits<int64_t>::max();
	printf("max = %lld, min = %lld\n", max, min);
	for(int i=0; i<num_elements; i++){
		U_fp[i] = U[i] * vector_field_scaling_factor;
		V_fp[i] = V[i] * vector_field_scaling_factor;
		max = std::max(max, U_fp[i]);
		max = std::max(max, V_fp[i]);
		min = std::min(min, U_fp[i]);
		min = std::min(min, V_fp[i]);
	}
	printf("max = %lld, min = %lld\n", max, min);
	range = max - min;
	return vector_field_scaling_factor;
}

// template<typename T_data>
// unsigned char *
// preserve_skeleton(T_data * U, T_data * V, int r1, int r2, double max_pwr_eb, size_t num_elements, size_t & result_size){
//     using T = int64_t;
//     T * U_fp = (T *) malloc(num_elements*sizeof(T));
//     T * V_fp = (T *) malloc(num_elements*sizeof(T));
//     T range = 0;
//     T vector_field_scaling_factor = convert_to_fixed_point(U, V, num_elements, U_fp, V_fp, range);
//     printf("fixed point range = %lld\n", range);
//     int * eb_quant_index = (int *) malloc(num_elements*sizeof(int));
// 	int * data_quant_index = (int *) malloc(2*num_elements*sizeof(int));
// 	int * eb_quant_index_pos = eb_quant_index;
// 	int * data_quant_index_pos = data_quant_index;
//     // next, row by row
// 	const int base = 2;
// 	const double log_of_base = log2(base);
// 	const int capacity = 65536;
// 	const int intv_radius = (capacity >> 1);
// 	T max_eb = range * max_pwr_eb;
//     unpred_vec<T_data> unpred_data;
// }

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


int main(int argc, char **argv){
    size_t num_elements = 0; 
    float * U = readfile<float>(argv[1], num_elements);
    float * V = readfile<float>(argv[2], num_elements);
    int DW = atoi(argv[3]); 
    int DH = atoi(argv[4]); 
    double max_eb = atof(argv[5]); 
    std::string option = argv[6];


	// //get critical points
	// const int type_bits = 63;
	// double vector_field_resolution = 0;
	// uint64_t vector_field_scaling_factor = 1;
	// for (int i=0; i<num_elements; i++){
	// 	double min_val = std::max(fabs(U[i]), fabs(V[i]));
	// 	vector_field_resolution = std::max(vector_field_resolution, min_val);
	// }
	// int vbits = std::ceil(std::log2(vector_field_resolution));
	// int nbits = (type_bits - 3) / 2;
	// vector_field_scaling_factor = 1 << (nbits - vbits);
	// auto critical_points_0 =compute_critical_points(U, V, DH, DW, vector_field_scaling_factor);



    if (option == "normal"){
    printf("************\n");
    //正常压缩
    size_t result_size = 0;
    unsigned char * result = NULL;
    //result = sz_compress_cp_preserve_sos_2d_online_fp(U, V, DH,DW, result_size, false, max_eb); // use cpsz-sos
    result = sz_compress_cp_preserve_2d_online(U, V, DH,DW, result_size, false, max_eb); // use cpsz
    unsigned char * result_after_lossless = NULL;
    size_t lossless_outsize = sz_lossless_compress(ZSTD_COMPRESSOR, 3, result, result_size, &result_after_lossless);
    cout << "Compressed size(original) = " << lossless_outsize << ", ratio = " << (2*num_elements*sizeof(float)) * 1.0/lossless_outsize << endl;
    
    //正常压缩后的数据解压
    free(result);
    size_t lossless_output = sz_lossless_decompress(ZSTD_COMPRESSOR, result_after_lossless, lossless_outsize, &result, result_size);
    float * dec_U = NULL;
    float * dec_V = NULL;
    // sz_decompress_cp_preserve_2d_online_fp<float>(result, DH,DW, dec_U, dec_V); // use cpsz-sos
    sz_decompress_cp_preserve_2d_online<float>(result, DH,DW, dec_U, dec_V); // use cpsz
    printf("verifying...\n");
    verify(U, dec_U, num_elements);

    writefile((string(argv[1]) + ".out").c_str(), dec_U, num_elements);
    writefile((string(argv[2]) + ".out").c_str(), dec_V, num_elements);
    printf("written to %s.out and %s.out\n", argv[1], argv[2]);
    free(result);
    free(dec_U);
    free(dec_V);

    }

    else if (option == "lossless_trajectory"){
    size_t result_size = 0;
    printf("************\n");
    //read index_need_lossless.bin
    size_t *lossless_index = NULL;
    size_t lossless_index_size = 0;
    lossless_index = readfile<size_t>("../small_data/index_need_lossless.bin", lossless_index_size);
    printf("number of index need to lossless = %ld\n", lossless_index_size);

    //convert to unordered_map
    std::unordered_map<size_t, size_t> lossless_index_map;
    for (int i = 0; i < lossless_index_size; i++){
        lossless_index_map[lossless_index[i]] = i;
    }
    printf("number of index need to lossless = %ld\n", lossless_index_map.size());
    size_t result_size_test =0;
    unsigned char * result_test = NULL;
    // lossless trajectory压缩
    //result_test = compress_lossless_index(U, V, lossless_index_map, DH,DW, result_size_test, false, max_eb); //cpsz-sos
    result_test = sz_compress_cp_preserve_2d_online(U, V, DH,DW, result_size_test, false, max_eb,lossless_index_map); //cpsz
    unsigned char * result_after_lossless_test = NULL;
    size_t lossless_outsize_test = sz_lossless_compress(ZSTD_COMPRESSOR, 3, result_test, result_size_test, &result_after_lossless_test); 
    cout << "Compressed size(lossless store trajectory) = " << lossless_outsize_test << ", ratio = " << (2*num_elements*sizeof(float)) * 1.0/lossless_outsize_test << endl;
    printf("result_size = %zu, result_size_test = %zu\n", result_size, result_size_test);
    // lossless trajectory压缩后的数据解压
    free(result_test);
    size_t lossless_output_test = sz_lossless_decompress(ZSTD_COMPRESSOR, result_after_lossless_test, lossless_outsize_test, &result_test, result_size_test);
    float * dec_U_test = NULL;
    float * dec_V_test = NULL;
    //sz_decompress_cp_preserve_2d_online_fp<float>(result_test, DH,DW, dec_U_test, dec_V_test); //use cpsz-sos
    sz_decompress_cp_preserve_2d_online<float>(result_test, DH,DW, dec_U_test, dec_V_test); //use cpsz
    printf("verifying...\n");
    verify(U, dec_U_test, num_elements);


    //write dec_U_test to file
    writefile((string(argv[1]) + ".test").c_str(), dec_U_test, num_elements);
    writefile((string(argv[2]) + ".test").c_str(), dec_V_test, num_elements);

    free(result_test);
    free(dec_U_test);
    free(dec_V_test);
    free(result_after_lossless_test);
    free(lossless_index);
    free(U);
    free(V);
    }
    
    else if (option == "new_method"){
      size_t result_size = 0;
      printf("************\n");
      //read index_need_lossless.bin
      size_t *lossless_index = NULL;
      size_t lossless_index_size = 0;
      lossless_index = readfile<size_t>("../small_data/index_need_lossless.bin", lossless_index_size);
      printf("number of index need to lossless = %ld\n", lossless_index_size);
      //convert to unordered_map
      std::unordered_map<size_t, size_t> lossless_index_map;
      for (int i = 0; i < lossless_index_size; i++){
          lossless_index_map[lossless_index[i]] = i;
      }

      //read traj_cells.bin
      std::vector<std::vector<size_t>> traj_cells = readVectorOfVector<size_t>("../small_data/traj_cells.bin");
      



    }
    else {
      printf("option only support normal and lossless_trajectory\n");
    }
	

	
    exit(0);
}