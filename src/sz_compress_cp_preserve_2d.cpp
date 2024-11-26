#include "sz_cp_preserve_utils.hpp"
#include "sz_compress_3d.hpp"
#include "sz_compress_cp_preserve_2d.hpp"
#include "sz_def.hpp"
#include "sz_compression_utils.hpp"
#include <cassert>
#include <iostream>
#include <set>
#include "cp.hpp"
#include <omp.h>
#include "utilsIO.h"


inline std::set<size_t> convert_simplexID_to_coords(const std::set<size_t>& simplex, const int DW, const int DH){
  std::set<size_t> result;
  for (const auto& simplex_ID : simplex){
    if (simplex_ID % 2 == 0){
    //upper
      size_t x = simplex_ID / 2 % (DW-1);
      size_t y = simplex_ID / 2 / (DW-1);
      result.insert(y * DW + x); //left low
      result.insert(y * DW + DW + x); //left up
      result.insert(y * DW + DW + x + 1); //right up
    }
    else{
      size_t x = simplex_ID / 2 % (DW-1);
      size_t y = simplex_ID / 2 / (DW-1);
      result.insert(y * DW + x); //left low
      result.insert(y * DW + x + 1); //right low
      result.insert(y * DW + DW + x + 1); //right up
    }    
  }
  return result;
}


// template<typename Type>
// void writefile(const char * file, Type * data, size_t num_elements){
// 	std::ofstream fout(file, std::ios::binary);
// 	fout.write(reinterpret_cast<const char*>(&data[0]), num_elements*sizeof(Type));
// 	fout.close();
// }

// template
// void writefile(const char * file,  int * data, size_t num_elements);

// maximal error bound to keep the sign of u0v1 - u0v2 + u1v2 - u1v0 + u2v0 - u2v1
template<typename T>
inline double max_eb_to_keep_sign_det2x2(const T u0, const T u1, const T u2, const T v0, const T v1, const T v2){
  double positive = 0;
  double negative = 0;
  accumulate(u0*v1, positive, negative);
  accumulate(-u0*v2, positive, negative);
  accumulate(u1*v2, positive, negative);
  accumulate(-u1*v0, positive, negative);
  accumulate(u2*v0, positive, negative);
  accumulate(-u2*v1, positive, negative);
  return max_eb_to_keep_sign(positive, negative, 2);
}

template<typename T>
inline double max_eb_to_keep_sign_2d_offline_2(const T u0, const T u1, const int degree=2){
  double positive = 0;
  double negative = 0;
  accumulate(u0, positive, negative);
  accumulate(u1, positive, negative);
  return max_eb_to_keep_sign(positive, negative, degree);
}

template<typename T>
inline double max_eb_to_keep_sign_2d_offline_4(const T u0, const T u1, const T u2, const T u3, const int degree=2){
  double positive = 0;
  double negative = 0;
  accumulate(u0, positive, negative);
  accumulate(u1, positive, negative);
  accumulate(u2, positive, negative);
  accumulate(u3, positive, negative);
  return max_eb_to_keep_sign(positive, negative, degree);
}

// det(c) = (x0 - x2)*(y1 - y2) - (x1 - x2)*(y0 - y2)
// c0 = (y1 - y2) / det(c)   c1 = -(y0 - y2) / det(c)
// c1 = -(x1 - x2) / det(c)  c3 = (x0 - x2) / det(c)
template<typename T>
inline void get_adjugate_matrix_for_position(const T x0, const T x1, const T x2, const T y0, const T y1, const T y2, T c[4]){
  T determinant = (x0 - x2)*(y1 - y2) - (x1 - x2)*(y0 - y2);
  c[0] = (y1 - y2) / determinant;
  c[1] = -(y0 - y2) / determinant;
  c[2] = -(x1 - x2) / determinant;
  c[3] = (x0 - x2) / determinant;
  // printf("%.4g, %.2g %.2g %.2g %.2g\n", determinant, c[0], c[1], c[2], c[3]);
  // exit(0);
}

// accumulate positive and negative in (a + b + c ...)^2
template<typename T>
inline void accumulate_in_square(const std::vector<T>& coeff, double& positive, double& negative){
  for(int i=0; i<coeff.size(); i++){
    for(int j=0; j<coeff.size(); j++){
      accumulate(coeff[i]*coeff[j], positive, negative);
    }
  }
}

// maximal error bound to keep the sign of B^2 - 4C
// where  B = - (c0 * (u0 - u2) + c1 * (u1 - u2) + c2 * (v0 - v2) + c3 * (v1 - v2))
//        C = det2x2 = u0v1 - u0v2 + u1v2 - u1v0 + u2v0 - u2v1
template<typename T>
inline double max_eb_to_keep_sign_eigen_delta_2(const T u0, const T u1, const T u2, const T v0, const T v1, const T v2,
  const T x0, const T x1, const T x2, const T y0, const T y1, const T y2){
  double eb = 1;
  T c[4] = {0};
  {
    get_adjugate_matrix_for_position(x0, x1, x2, y0, y1, y2, c);
    // keep sign for B
    double positive = 0;
    double negative = 0;
    accumulate(c[0]*u0, positive, negative);
    accumulate(c[1]*u1, positive, negative);
    accumulate(-(c[0] + c[1])*u2, positive, negative);
    accumulate(c[2]*v0, positive, negative);
    accumulate(c[3]*v1, positive, negative);
    accumulate(-(c[2] + c[3])*v2, positive, negative);
    eb = max_eb_to_keep_sign(positive, negative, 1);
    // keep sign for C
    eb = MINF(eb, max_eb_to_keep_sign_det2x2(u0, u1, u2, v0, v1, v2));
  }
  T m = c[1]*c[2] - c[0]*c[3];
  T C = (-m) * (u0*v1 - u0*v2 + u1*v2 - u1*v0 + u2*v0 - u2*v1);
  if(C == 0) return 0;
  if(C < 0) return eb;
  {
    std::vector<T> coeff(6);
    coeff[0] = c[0]*u0;
    coeff[1] = c[1]*u1;
    coeff[2] = - (c[1] + c[0])*u2;
    coeff[3] = c[2]*v0;
    coeff[4] = c[3]*v1;
    coeff[5] = - (c[3] + c[2])*v2;
    // keep sign for B^2 - 4*C
    double positive = 0;
    double negative = 0;
    accumulate_in_square(coeff, positive, negative);
    accumulate(-4*m*u1*v0, positive, negative);
    accumulate(4*m*u2*v0, positive, negative);
    accumulate(4*m*u0*v1, positive, negative);
    accumulate(-4*m*u2*v1, positive, negative);
    accumulate(-4*m*u0*v2, positive, negative);
    accumulate(4*m*u1*v2, positive, negative);
    eb = MINF(eb, max_eb_to_keep_sign(positive, negative, 2));
  }
  return eb;
}

template<typename T>
inline double max_eb_to_keep_position_and_type(const T u0, const T u1, const T u2, const T v0, const T v1, const T v2,
											const T x0, const T x1, const T x2, const T y0, const T y1, const T y2){
	double u0v1 = u0 * v1;
	double u1v0 = u1 * v0;
	double u0v2 = u0 * v2;
	double u2v0 = u2 * v0;
	double u1v2 = u1 * v2;
	double u2v1 = u2 * v1;
	double det = u0v1 - u1v0 + u1v2 - u2v1 + u2v0 - u0v2;
	double eb = 0;
	if(det != 0){
		bool f1 = (det / (u2v0 - u0v2) >= 1);
		bool f2 = (det / (u1v2 - u2v1) >= 1); 
		bool f3 = (det / (u0v1 - u1v0) >= 1); 
		if(f1 && f2 && f3){
			// critical point found
			eb = 1;
			double eb1 = MINF(max_eb_to_keep_sign_2d_offline_2(u2v0, -u0v2), max_eb_to_keep_sign_2d_offline_4(u0v1, -u1v0, u1v2, -u2v1));
			double eb2 = MINF(max_eb_to_keep_sign_2d_offline_2(u1v2, -u2v1), max_eb_to_keep_sign_2d_offline_4(u0v1, -u1v0, u2v0, -u0v2));
			double eb3 = MINF(max_eb_to_keep_sign_2d_offline_2(u0v1, -u1v0), max_eb_to_keep_sign_2d_offline_4(u1v2, -u2v1, u2v0, -u0v2));
			double eb4 = MINF(eb3, max_eb_to_keep_sign_eigen_delta_2(u0, u1, u2, v0, v1, v2, x0, x1, x2, y0, y1, y2));
			eb = MINF(eb1, eb2);
			eb = MINF(eb, eb4);
		}
		else{
			// no critical point
			eb = 0;
			if(!f1){
				double eb_cur = MINF(max_eb_to_keep_sign_2d_offline_2(u2v0, -u0v2), max_eb_to_keep_sign_2d_offline_4(u0v1, -u1v0, u1v2, -u2v1));
				// double eb_cur = MINF(max_eb_to_keep_sign_2(u2, u0, v2, v0), max_eb_to_keep_sign_4(u0, u1, u2, v0, v1, v2));
				eb = MAX(eb, eb_cur);
			}
			if(!f2){
				double eb_cur = MINF(max_eb_to_keep_sign_2d_offline_2(u1v2, -u2v1), max_eb_to_keep_sign_2d_offline_4(u0v1, -u1v0, u2v0, -u0v2));
				// double eb_cur = MINF(max_eb_to_keep_sign_2(u1, u2, v1, v2), max_eb_to_keep_sign_4(u2, u0, u1, v2, v0, v1));
				eb = MAX(eb, eb_cur);
			}
			if(!f3){
				double eb_cur = MINF(max_eb_to_keep_sign_2d_offline_2(u0v1, -u1v0), max_eb_to_keep_sign_2d_offline_4(u1v2, -u2v1, u2v0, -u0v2));
				// double eb_cur = MINF(max_eb_to_keep_sign_2(u0, u1, v0, v1), max_eb_to_keep_sign_4(u1, u2, u0, v1, v2, v0));
				eb = MAX(eb, eb_cur);
			}
			// eb = MINF(eb, DEFAULT_EB);
		}
	}
	return eb;
}

// // compression with pre-computed error bounds
// template<typename T>
// unsigned char *
// sz_compress_cp_preserve_2d_offline(const T * U, const T * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb){

// 	size_t num_elements = r1 * r2;
// 	double * eb = (double *) malloc(num_elements * sizeof(double));
// 	for(int i=0; i<num_elements; i++) eb[i] = max_pwr_eb;
// 	const T * U_pos = U;
// 	const T * V_pos = V;
// 	double * eb_pos = eb;
// 	// coordinates for triangle_coordinates
// 	const T X_upper[3][2] = {{0, 0}, {1, 0}, {1, 1}};
// 	const T X_lower[3][2] = {{0, 0}, {0, 1}, {1, 1}};
// 	const size_t offset_upper[3] = {0, r2, r2+1};
// 	const size_t offset_lower[3] = {0, 1, r2+1};
// 	printf("compute eb\n");
// 	for(int i=0; i<r1-1; i++){
// 		const T * U_row_pos = U_pos;
// 		const T * V_row_pos = V_pos;
// 		double * eb_row_pos = eb_pos;
// 		for(int j=0; j<r2-1; j++){
// 			for(int k=0; k<2; k++){
// 				auto X = (k == 0) ? X_upper : X_lower;
// 				auto offset = (k == 0) ? offset_upper : offset_lower;
// 				// reversed order!
// 				double max_cur_eb = max_eb_to_keep_position_and_type(U_row_pos[offset[0]], U_row_pos[offset[1]], U_row_pos[offset[2]],
// 					V_row_pos[offset[0]], V_row_pos[offset[1]], V_row_pos[offset[2]], X[0][1], X[1][1], X[2][1],
// 					X[0][0], X[1][0], X[2][0]);
// 				eb_row_pos[offset[0]] = MINF(eb_row_pos[offset[0]], max_cur_eb);
// 				eb_row_pos[offset[1]] = MINF(eb_row_pos[offset[1]], max_cur_eb);
// 				eb_row_pos[offset[2]] = MINF(eb_row_pos[offset[2]], max_cur_eb);
// 			}
// 			U_row_pos ++;
// 			V_row_pos ++;
// 			eb_row_pos ++;
// 		}
// 		U_pos += r2;
// 		V_pos += r2;
// 		eb_pos += r2;
// 	}
// 	printf("compute eb done\n");
// 	double * eb_u = (double *) malloc(num_elements * sizeof(double));
// 	double * eb_v = (double *) malloc(num_elements * sizeof(double));
// 	int * eb_quant_index = (int *) malloc(2*num_elements*sizeof(int));
// 	int * eb_quant_index_pos = eb_quant_index;
// 	const int base = 4;
// 	double log2_of_base = log2(base);
// 	const double threshold = std::numeric_limits<double>::epsilon();
// 	for(int i=0; i<num_elements; i++){
// 		eb_u[i] = fabs(U[i]) * eb[i];
// 		*(eb_quant_index_pos ++) = eb_exponential_quantize(eb_u[i], base, log2_of_base, threshold);
// 		// *(eb_quant_index_pos ++) = eb_linear_quantize(eb_u[i], 1e-2);
// 		if(eb_u[i] < threshold) eb_u[i] = 0;
// 	}
// 	for(int i=0; i<num_elements; i++){
// 		eb_v[i] = fabs(V[i]) * eb[i];
// 		*(eb_quant_index_pos ++) = eb_exponential_quantize(eb_v[i], base, log2_of_base, threshold);
// 		// *(eb_quant_index_pos ++) = eb_linear_quantize(eb_v[i], 1e-2);
// 		if(eb_v[i] < threshold) eb_v[i] = 0;
// 	}
// 	free(eb);
// 	printf("quantize eb done\n");
// 	unsigned char * compressed_eb = (unsigned char *) malloc(2*num_elements*sizeof(int));
// 	unsigned char * compressed_eb_pos = compressed_eb; 
// 	Huffman_encode_tree_and_data(2*1024, eb_quant_index, 2*num_elements, compressed_eb_pos);
// 	size_t compressed_eb_size = compressed_eb_pos - compressed_eb;
// 	size_t compressed_u_size = 0;
// 	size_t compressed_v_size = 0;
// 	unsigned char * compressed_u = sz_compress_2d_with_eb(U, eb_u, r1, r2, compressed_u_size);
// 	unsigned char * compressed_v = sz_compress_2d_with_eb(V, eb_v, r1, r2, compressed_v_size);
// 	printf("eb_size = %ld, u_size = %ld, v_size = %ld\n", compressed_eb_size, compressed_u_size, compressed_v_size);
// 	free(eb_u);
// 	free(eb_v);
// 	compressed_size = sizeof(int) + sizeof(size_t) + compressed_eb_size + sizeof(size_t) + compressed_u_size + sizeof(size_t) + compressed_v_size;
// 	unsigned char * compressed = (unsigned char *) malloc(compressed_size);
// 	unsigned char * compressed_pos = compressed;
// 	write_variable_to_dst(compressed_pos, base);
// 	write_variable_to_dst(compressed_pos, threshold);
// 	write_variable_to_dst(compressed_pos, compressed_eb_size);
// 	write_variable_to_dst(compressed_pos, compressed_u_size);
// 	write_variable_to_dst(compressed_pos, compressed_v_size);
// 	write_array_to_dst(compressed_pos, compressed_eb, compressed_eb_size);
// 	write_array_to_dst(compressed_pos, compressed_u, compressed_u_size);
// 	printf("compressed_pos = %ld\n", compressed_pos - compressed);
// 	write_array_to_dst(compressed_pos, compressed_v, compressed_v_size);
// 	free(compressed_eb);
// 	free(compressed_u);
// 	free(compressed_v);
// 	return compressed;
// }

// template
// unsigned char *
// sz_compress_cp_preserve_2d_offline(const float * U, const float * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb);

// template
// unsigned char *
// sz_compress_cp_preserve_2d_offline(const double * U, const double * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb);

// // compression with pre-computed error bounds in logarithmic domain
// template<typename T>
// unsigned char *
// sz_compress_cp_preserve_2d_offline_log(const T * U, const T * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb){

// 	size_t num_elements = r1 * r2;
// 	double * eb = (double *) malloc(num_elements * sizeof(double));
// 	for(int i=0; i<num_elements; i++) eb[i] = max_pwr_eb;
// 	const T * U_pos = U;
// 	const T * V_pos = V;
// 	double * eb_pos = eb;
// 	// coordinates for triangle_coordinates
// 	const T X_upper[3][2] = {{0, 0}, {1, 0}, {1, 1}};
// 	const T X_lower[3][2] = {{0, 0}, {0, 1}, {1, 1}};
// 	const size_t offset_upper[3] = {0, r2, r2+1};
// 	const size_t offset_lower[3] = {0, 1, r2+1};
// 	// printf("compute eb\n");
// 	for(int i=0; i<r1-1; i++){
// 		const T * U_row_pos = U_pos;
// 		const T * V_row_pos = V_pos;
// 		double * eb_row_pos = eb_pos;
// 		for(int j=0; j<r2-1; j++){
// 			for(int k=0; k<2; k++){
// 				auto X = (k == 0) ? X_upper : X_lower;
// 				auto offset = (k == 0) ? offset_upper : offset_lower;
// 				// reversed order!
// 				double max_cur_eb = max_eb_to_keep_position_and_type(U_row_pos[offset[0]], U_row_pos[offset[1]], U_row_pos[offset[2]],
// 					V_row_pos[offset[0]], V_row_pos[offset[1]], V_row_pos[offset[2]], X[0][1], X[1][1], X[2][1],
// 					X[0][0], X[1][0], X[2][0]);
// 				eb_row_pos[offset[0]] = MINF(eb_row_pos[offset[0]], max_cur_eb);
// 				eb_row_pos[offset[1]] = MINF(eb_row_pos[offset[1]], max_cur_eb);
// 				eb_row_pos[offset[2]] = MINF(eb_row_pos[offset[2]], max_cur_eb);
// 			}
// 			U_row_pos ++;
// 			V_row_pos ++;
// 			eb_row_pos ++;
// 		}
// 		U_pos += r2;
// 		V_pos += r2;
// 		eb_pos += r2;
// 	}
// 	// writefile("eb_2d_decoupled.dat", eb, num_elements);
// 	// printf("compute eb done\n");
// 	size_t sign_map_size = (num_elements - 1)/8 + 1;
// 	unsigned char * sign_map_compressed = (unsigned char *) malloc(2*sign_map_size);
// 	unsigned char * sign_map_compressed_pos = sign_map_compressed;
// 	unsigned char * sign_map = (unsigned char *) malloc(num_elements*sizeof(unsigned char));
// 	// Note the convert function has address auto increment
// 	T * log_U = log_transform(U, sign_map, num_elements);
// 	convertIntArray2ByteArray_fast_1b_to_result_sz(sign_map, num_elements, sign_map_compressed_pos);
// 	T * log_V = log_transform(V, sign_map, num_elements);
// 	convertIntArray2ByteArray_fast_1b_to_result_sz(sign_map, num_elements, sign_map_compressed_pos);
// 	free(sign_map);
// 	// transfrom eb to log(1 + eb) and the quantize
// 	int * eb_quant_index = (int *) malloc(num_elements*sizeof(int));
// 	int * eb_quant_index_pos = eb_quant_index;
// 	const int base = 2;
// 	double log2_of_base = log2(base);
// 	const double threshold = std::numeric_limits<double>::epsilon();
// 	for(int i=0; i<num_elements; i++){
// 		eb[i] = log2(1 + eb[i]);
// 		*(eb_quant_index_pos ++) = eb_exponential_quantize(eb[i], base, log2_of_base, threshold);
// 		// *(eb_quant_index_pos ++) = eb_linear_quantize(eb[i], 5e-3);
// 	}
// 	// printf("quantize eb done\n");
// 	unsigned char * compressed_eb = (unsigned char *) malloc(num_elements*sizeof(int));
// 	unsigned char * compressed_eb_pos = compressed_eb; 
// 	Huffman_encode_tree_and_data(2*1024, eb_quant_index, num_elements, compressed_eb_pos);
// 	size_t compressed_eb_size = compressed_eb_pos - compressed_eb;
// 	size_t compressed_u_size = 0;
// 	size_t compressed_v_size = 0;
// 	unsigned char * compressed_u = sz_compress_2d_with_eb(log_U, eb, r1, r2, compressed_u_size);
// 	free(log_U);
// 	unsigned char * compressed_v = sz_compress_2d_with_eb(log_V, eb, r1, r2, compressed_v_size);
// 	free(log_V);
// 	// printf("eb_size = %ld, log_u_size = %ld, log_v_size = %ld\n", compressed_eb_size, compressed_u_size, compressed_v_size);
// 	free(eb);
// 	compressed_size = sizeof(int) + 2*sign_map_size + sizeof(size_t) + sizeof(double) + compressed_eb_size + sizeof(size_t) + compressed_u_size + sizeof(size_t) + compressed_v_size;
// 	unsigned char * compressed = (unsigned char *) malloc(compressed_size);
// 	unsigned char * compressed_pos = compressed;
// 	write_variable_to_dst(compressed_pos, base);
// 	write_variable_to_dst(compressed_pos, threshold);
// 	write_variable_to_dst(compressed_pos, compressed_eb_size);
// 	write_variable_to_dst(compressed_pos, compressed_u_size);
// 	write_variable_to_dst(compressed_pos, compressed_v_size);
// 	write_array_to_dst(compressed_pos, compressed_eb, compressed_eb_size);
// 	write_array_to_dst(compressed_pos, sign_map_compressed, 2*sign_map_size);
// 	// printf("before data: %ld\n", compressed_pos - compressed);
// 	write_array_to_dst(compressed_pos, compressed_u, compressed_u_size);
// 	write_array_to_dst(compressed_pos, compressed_v, compressed_v_size);
// 	free(sign_map_compressed);
// 	free(compressed_eb);
// 	free(compressed_u);
// 	free(compressed_v);
// 	return compressed;
// }

// template
// unsigned char *
// sz_compress_cp_preserve_2d_offline_log(const float * U, const float * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb);

// template
// unsigned char *
// sz_compress_cp_preserve_2d_offline_log(const double * U, const double * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb);

typedef struct conditions_2d{
	bool computed;
	bool singular;
	bool flags[3];
	conditions_2d(){
		computed = false;
		singular = false;
		for(int i=0; i<3; i++){
			flags[i] = false;
		}
	}
}conditions_2d;

// maximal error bound to keep the sign of A*(1 + e_1) + B*(1 + e_2) + C
template<typename T>
inline double max_eb_to_keep_sign_2d_online(const T A, const T B, const T C=0){
	double fabs_sum = (fabs(A) + fabs(B));
	if(fabs_sum == 0) return 0;
	return fabs(A + B + C) / fabs_sum;
}

/* 
absolute error bound version.
*/

// maximal absolute error bound to keep the sign of A*e_1 + B*e_2 + C
template<typename T>
inline double max_eb_to_keep_sign_2d_online_abs(const T A, const T B, const T C=0){
	return fabs(C) / (fabs(A) + fabs(B));
}


/*
triangle mesh x0, x1, x2, derive absolute cp-preserving eb for x2 given x0, x1
*/
template<typename T>
double 
derive_cp_eb_for_positions_online_abs(const T u0, const T u1, const T u2, const T v0, const T v1, const T v2, const T c[4]){//, conditions_2d& cond){
	double M0 = u2*v0 - u0*v2;
  double M1 = u1*v2 - u2*v1;
  double M2 = u0*v1 - u1*v0;
  double M = M0 + M1 + M2;
	if(M == 0) return 0;
	bool f1 = (M0 == 0) || (M / M0 >= 1);
	bool f2 = (M1 == 0) || (M / M1 >= 1); 
	bool f3 = (M2 == 0) || (M / M2 >= 1);
	double eb = 0;
	if(f1 && f2 && f3){
		// cp exists
		return 0;
	}
	else{
		eb = 0;
		if(!f1){
			// M0(M1 + M2)
			// M0: (u2+e1)*v0 - u0(v2+e2)
			// double cur_eb = MINF(max_eb_to_keep_sign_2d_online(u2*v0, -u0*v2), max_eb_to_keep_sign_2d_online(-u2*v1, u1*v2, u0*v1 - u1*v0));
			double cur_eb = MINF(max_eb_to_keep_sign_2d_online_abs(v0, -u0, u2*v0 - u0*v2), max_eb_to_keep_sign_2d_online_abs(-v1, u1, u1*v2 - u2*v1 + u0*v1 - u1*v0));
			eb = MAX(eb, cur_eb);
		}
		if(!f2){
			// M1(M0 + M2)
			// double cur_eb = MINF(max_eb_to_keep_sign_2d_online(-u2*v1, u1*v2), max_eb_to_keep_sign_2d_online(u2*v0, -u0*v2, u0*v1 - u1*v0));
			double cur_eb = MINF(max_eb_to_keep_sign_2d_online_abs(-v1, u1, u1*v2 - u2*v1), max_eb_to_keep_sign_2d_online_abs(v0, -u0, u2*v0 - u0*v2 + u0*v1 - u1*v0));
			eb = MAX(eb, cur_eb);				
		}
		if(!f3){
			// M2(M0 + M1)
			// double cur_eb = max_eb_to_keep_sign_2d_online(u2*v0 - u2*v1, u1*v2 - u0*v2);
			double cur_eb = max_eb_to_keep_sign_2d_online_abs(v0 - v1, u1 - u0, u2*v0 - u0*v2 + u1*v2 - u2*v1);
			eb = MAX(eb, cur_eb);				
		}
	}
	return eb;
}

// W0 + W1 = u1v2 - u2v1 + u2v0 - u0v2
// W1 + W2 = u2v0 - u0v2 + u0v1 - u1v0
// W2 + W0 = u0v1 - u1v0 + u1v2 - u2v1
template<typename T>
inline double max_eb_to_keep_position_online(const T u0v1, const T u1v0, const T u1v2, const T u2v1, const T u2v0, const T u0v2){
	double eb = MINF(max_eb_to_keep_sign_2d_online(-u2v1, u1v2), max_eb_to_keep_sign_2d_online(u2v0, -u0v2));
	// eb = MINF(eb, max_eb_to_keep_sign_2d_online(u2v0, -u0v2, u0v1 - u1v0));
	// eb = MINF(eb, max_eb_to_keep_sign_2d_online(-u2v1, u1v2, u0v1 - u1v0));
	// eb = MINF(eb, max_eb_to_keep_sign_2d_online(u2v0 - u2v1, u1v2 - u0v2));
	return eb;
}

// maximal error bound to keep the sign of B^2 - 4C
// where  B = - (c0 * (u0 - u2) + c1 * (u1 - u2) + c2 * (v0 - v2) + c3 * (v1 - v2))
//        C = det2x2 = u0v1 - u0v2 + u1v2 - u1v0 + u2v0 - u2v1
template<typename T>
inline double max_eb_to_keep_type_online(const T u0, const T u1, const T u2, const T v0, const T v1, const T v2, const T c[4]){
	double eb = 1;
	{
		// keep sign for C
		eb = MINF(eb, max_eb_to_keep_sign_2d_online(u2*v0 - u2*v1, u1*v2 - u0*v2, u0*v1 - u1*v0));
	}
	T m = c[1]*c[2] - c[0]*c[3];
	T C = (-m) * (u0*v1 - u0*v2 + u1*v2 - u1*v0 + u2*v0 - u2*v1);
	if(C <= 0) return eb;
	{
	    // keep sign for B
	    // coeff[0] = c[0]*u0;
	    // coeff[1] = c[1]*u1;
	    // coeff[2] = - (c[1] + c[0])*u2;
	    // coeff[3] = c[2]*v0;
	    // coeff[4] = c[3]*v1;
	    // coeff[5] = - (c[3] + c[2])*v2;
	    eb = max_eb_to_keep_sign_2d_online(-c[0]*u2 - c[1]*u2, -c[2]*v2 - c[3]*v2, c[0]*u0 + c[1]*u1 + c[2]*v0 + c[3]*v1);
	}
	{
		// Note that meaning of B in the rhs changes here
		// keep sign for B^2 - 4*C
		// B = A*(1+e_1) + B*(1+e_2) + C
		// C = D*(1+e_1) + E*(1+e_2) + F
		double A = -c[0]*u2 - c[1]*u2, B = -c[2]*v2 - c[3]*v2, C = c[0]*u0 + c[1]*u1 + c[2]*v0 + c[3]*v1;
		double D = (-m)*(u2*v0 - u2*v1), E = (-m)*(u1*v2 - u0*v2), F = (-m)*(u0*v1 - u1*v0);
		// B = A*e_1 + B*e_2 + C'
		// C = D*e_1 + E*e_2 + F'
		C += A + B, F += D + E;
		// B^2 - 4C = (A*e_1 + B*e_2)^2 + (2AC' - 4D)e_1 + (2BC' - 4E)e_2 + C'^2 - 4F'
		double delta = C*C - 4*F;
		if(delta == 0) return 0;
		else if(delta > 0){
			// (|2AC' - 4D| + |2BC' - 4E|)* -e + delta > 0
			if((fabs(2*A*C - 4*D) + fabs(2*B*C - 4*E)) == 0) eb = 1;
			else eb = MINF(eb, delta/(fabs(2*A*C - 4*D) + fabs(2*B*C - 4*E)));
		}
		else{
			// (|A| + |B|)^2*e^2 + (|2AC' - 4D| + |2BC' - 4E|)*e + delta < 0
			double a = (fabs(A) + fabs(B))*(fabs(A) + fabs(B));
			double b = fabs(2*A*C - 4*D) + fabs(2*B*C - 4*E);
			double c = delta;
			// if(b*b - 4*a*c < 0){
			// 	printf("impossible as a*c is always less than 0\n");
			// 	exit(0);
			// }
			eb = MINF(eb, (-b + sqrt(b*b - 4*a*c))/(2*a));

			// check four corners
			// double e1[2] = {-1, 1};
			// double e2[2] = {-1, 1};
			// double c = delta;
			// for(int i=0; i<2; i++){
			// 	for(int j=0; j<2; j++){
			// 		double a = (e1[i] * A + e2[j] * B) * (e1[i] * A + e2[j] * B);
			// 		double b = (2*A*C - 4*D) * e1[i] + (2*B*C - 4*E) * e2[j];
			// 		if(a == 0) eb = MINF(eb, 1);
			// 		else eb = MINF(eb, (-b + sqrt(b*b - 4*a*c))/(2*a));
			// 	}
			// }
		}
	}
	return eb;
}

/*
triangle mesh x0, x1, x2, derive cp-preserving eb for x2 given x0, x1
*/
template<typename T>
double 
derive_cp_eb_for_positions_online(const T u0, const T u1, const T u2, const T v0, const T v1, const T v2, const T c[4]){//, conditions_2d& cond){
	// if(!cond.computed){
	//     double M0 = u2*v0 - u0*v2;
	//     double M1 = u1*v2 - u2*v1;
	//     double M2 = u0*v1 - u1*v0;
	//     double M = M0 + M1 + M2;
	//     cond.singular = (M == 0);
	//     if(cond.singular) return 0;
	//     cond.flags[0] = (M0 == 0) || (M / M0 >= 1);
	//     cond.flags[1] = (M1 == 0) || (M / M1 >= 1);
	//     cond.flags[2] = (M2 == 0) || (M / M2 >= 1);
	//     cond.computed = true;
	// }
	// else{
	//     if(cond.singular) return 0;
	// }
	// const bool * flag = cond.flags;
	// bool f1 = flag[0];
	// bool f2 = flag[1]; 
	// bool f3 = flag[2];
	double M0 = u2*v0 - u0*v2;
    double M1 = u1*v2 - u2*v1;
    double M2 = u0*v1 - u1*v0;
    double M = M0 + M1 + M2;
	if(M == 0) return 0;
	bool f1 = (M0 == 0) || (M / M0 >= 1);
	bool f2 = (M1 == 0) || (M / M1 >= 1); 
	bool f3 = (M2 == 0) || (M / M2 >= 1);
	double eb = 0;
	if(f1 && f2 && f3){
		return 0;
		// eb = max_eb_to_keep_position_online(u0v1, u1v0, u1v2, u2v1, u2v0, u0v2);
		eb = MINF(max_eb_to_keep_position_online(u0*v1, u1*v0, u1*v2, u2*v1, u2*v0, u0*v2), 
			max_eb_to_keep_type_online(u0, u1, u2, v0, v1, v2, c));
	}
	else{
		eb = 0;
		if(!f1){
			// W1(W0 + W2)
			double cur_eb = MINF(max_eb_to_keep_sign_2d_online(u2*v0, -u0*v2), max_eb_to_keep_sign_2d_online(-u2*v1, u1*v2, u0*v1 - u1*v0));
			eb = MAX(eb, cur_eb);
		}
		if(!f2){
			// W0(W1 + W2)
			double cur_eb = MINF(max_eb_to_keep_sign_2d_online(-u2*v1, u1*v2), max_eb_to_keep_sign_2d_online(u2*v0, -u0*v2, u0*v1 - u1*v0));
			eb = MAX(eb, cur_eb);				
		}
		if(!f3){
			// W2(W0 + W1)
			double cur_eb = max_eb_to_keep_sign_2d_online(u2*v0 - u2*v1, u1*v2 - u0*v2);
			eb = MAX(eb, cur_eb);				
		}
	}
	return eb;
}

template<typename T>
inline bool 
inbound(T index, T lb, T ub){
	return (index >= lb) && (index < ub);
}

template<typename T>
unsigned char *
sz_compress_cp_preserve_2d_online(const T * U, const T * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb,const std::unordered_map <size_t, size_t> &lossless_index,const std::unordered_map <size_t, size_t> &index_need_to_fix){
	size_t num_elements = r1 * r2;
	T * decompressed_U = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_U, U, num_elements*sizeof(T));
	T * decompressed_V = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_V, V, num_elements*sizeof(T));
	int * eb_quant_index = (int *) malloc(2*num_elements*sizeof(int));
	int * data_quant_index = (int *) malloc(2*num_elements*sizeof(int));
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	// next, row by row
	const int base = 4;
	const double log_of_base = log2(base);
	const int capacity = 65536;
	const int intv_radius = (capacity >> 1);
	unpred_vec<T> unpred_data;
	T * U_pos = decompressed_U;
	T * V_pos = decompressed_V;
	// offsets to get six adjacent triangle indices
	// the 7-th rolls back to T0
	/*
			T3	T4
		T2	X 	T5
		T1	T0(T6)
	*/
	const int offsets[7] = {
		-(int)r2, -(int)r2 - 1, -1, (int)r2, (int)r2+1, 1, -(int)r2
	};
	const T x[6][3] = {
		{1, 0, 1},
		{0, 0, 1},
		{0, 1, 1},
		{0, 1, 0},
		{1, 1, 0},
		{1, 0, 0}
	};
	const T y[6][3] = {
		{0, 0, 1},
		{0, 1, 1},
		{0, 1, 0},
		{1, 1, 0},
		{1, 0, 0},
		{1, 0, 1}
	};
	T inv_C[6][4]; //这东西用不到
	for(int i=0; i<6; i++){
		get_adjugate_matrix_for_position(x[i][0], x[i][1], x[i][2], y[i][0], y[i][1], y[i][2], inv_C[i]); //算周围6个三角形的 determinant相关
	}
	int index_offset[6][2][2];
	for(int i=0; i<6; i++){
		for(int j=0; j<2; j++){
			index_offset[i][j][0] = x[i][j] - x[i][2];
			index_offset[i][j][1] = y[i][j] - y[i][2];
		}
	}
	double threshold = std::numeric_limits<double>::epsilon();
	// conditions_2d cond;
	for(int i=0; i<r1; i++){
		// printf("start %d row\n", i);
		T * cur_U_pos = U_pos;
		T * cur_V_pos = V_pos;
		for(int j=0; j<r2; j++){
			size_t vertex_index = i * r2 + j;
			double required_eb;
			if(lossless_index.find(vertex_index) != lossless_index.end()){
				required_eb = 0;
			}
			else{
				required_eb = max_pwr_eb;
				// derive eb given six adjacent triangles
				for(int k=0; k<6; k++){
					bool in_mesh = true;
					for(int p=0; p<2; p++){
						// reserved order!
						if(!(in_range(i + index_offset[k][p][1], (int)r1) && in_range(j + index_offset[k][p][0], (int)r2))){
							in_mesh = false;
							break;
						}
					}
					if(in_mesh){
						required_eb = MINF(required_eb, derive_cp_eb_for_positions_online(cur_U_pos[offsets[k]], cur_U_pos[offsets[k+1]], cur_U_pos[0],
							cur_V_pos[offsets[k]], cur_V_pos[offsets[k+1]], cur_V_pos[0], inv_C[k]));
					}
				}
			}
			if(required_eb > 0){
				bool unpred_flag = false;
				T decompressed[2];
				// compress U and V
				for(int k=0; k<2; k++){
					T * cur_data_pos = (k == 0) ? cur_U_pos : cur_V_pos;
					T cur_data = *cur_data_pos;
					double abs_eb = fabs(cur_data) * required_eb;
					eb_quant_index_pos[k] = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
					// eb_quant_index_pos[k] = eb_linear_quantize(abs_eb, 1e-3);
					if(eb_quant_index_pos[k] > 0){
						// get adjacent data and perform Lorenzo
						/*
							d2 X
							d0 d1
						*/
						T d0 = (i && j) ? cur_data_pos[-1 - r2] : 0;
						T d1 = (i) ? cur_data_pos[-r2] : 0;
						T d2 = (j) ? cur_data_pos[-1] : 0;
						T pred = d1 + d2 - d0;
						double diff = cur_data - pred;
						double quant_diff = fabs(diff) / abs_eb + 1;
						if(quant_diff < capacity){
							quant_diff = (diff > 0) ? quant_diff : -quant_diff;
							int quant_index = (int)(quant_diff/2) + intv_radius;
							data_quant_index_pos[k] = quant_index;
							decompressed[k] = pred + 2 * (quant_index - intv_radius) * abs_eb; 
							// check original data
							if(fabs(decompressed[k] - cur_data) >= abs_eb){
								unpred_flag = true;
								break;
							}
						}
						else{
							unpred_flag = true;
							break;
						}
					}
					else unpred_flag = true;
				}
				if(unpred_flag){
					// recover quant index
					*(eb_quant_index_pos ++) = 0;
					*(eb_quant_index_pos ++) = 0;
					*(data_quant_index_pos ++) = intv_radius;
					*(data_quant_index_pos ++) = intv_radius;
					unpred_data.push_back(*cur_U_pos);
					unpred_data.push_back(*cur_V_pos);
				}
				else{
					eb_quant_index_pos += 2;
					data_quant_index_pos += 2;
					// assign decompressed data
					*cur_U_pos = decompressed[0];
					*cur_V_pos = decompressed[1];
				}
			}
			else{
				// record as unpredictable data
				*(eb_quant_index_pos ++) = 0;
				*(eb_quant_index_pos ++) = 0;
				*(data_quant_index_pos ++) = intv_radius;
				*(data_quant_index_pos ++) = intv_radius;
				unpred_data.push_back(*cur_U_pos);
				unpred_data.push_back(*cur_V_pos);
			}
			cur_U_pos ++, cur_V_pos ++;
		}
		U_pos += r2;
		V_pos += r2;
	}
	free(decompressed_U);
	free(decompressed_V);
	printf("offsets eb_q, data_q, unpred: %ld %ld %ld\n", eb_quant_index_pos - eb_quant_index, data_quant_index_pos - data_quant_index, unpred_data.size());
	unsigned char * compressed = (unsigned char *) malloc(2*num_elements*sizeof(T));
	unsigned char * compressed_pos = compressed;
	write_variable_to_dst(compressed_pos, base);
	write_variable_to_dst(compressed_pos, threshold);
	write_variable_to_dst(compressed_pos, intv_radius);
	size_t unpredictable_count = unpred_data.size();
	write_variable_to_dst(compressed_pos, unpredictable_count);
	write_array_to_dst(compressed_pos, (T *)&unpred_data[0], unpredictable_count);	
	Huffman_encode_tree_and_data(2*1024, eb_quant_index, 2*num_elements, compressed_pos);
	free(eb_quant_index);
	Huffman_encode_tree_and_data(2*capacity, data_quant_index, 2*num_elements, compressed_pos);
	printf("pos = %ld\n", compressed_pos - compressed);
	free(data_quant_index);
	compressed_size = compressed_pos - compressed;
	return compressed;	
}

template
unsigned char *
sz_compress_cp_preserve_2d_online(const float * U, const float * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb,const std::unordered_map <size_t, size_t> &lossless_index,const std::unordered_map <size_t, size_t> &index_need_to_fix);

template
unsigned char *
sz_compress_cp_preserve_2d_online(const double * U, const double * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb,const std::unordered_map <size_t, size_t> &lossless_index,const std::unordered_map <size_t, size_t> &index_need_to_fix);

template<typename T>
unsigned char *
sz_compress_cp_preserve_2d_fix(const T * U, const T * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb,double modified_eb,const std::set<size_t> &index_need_to_fix){
	
	std::vector<float> eb_result(r1*r2, 0);

	size_t num_elements = r1 * r2;
	T * decompressed_U = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_U, U, num_elements*sizeof(T));
	T * decompressed_V = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_V, V, num_elements*sizeof(T));
	int * eb_quant_index = (int *) malloc(2*num_elements*sizeof(int));
	int * data_quant_index = (int *) malloc(2*num_elements*sizeof(int));
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	// next, row by row
	const int base = 4;
	const double log_of_base = log2(base);
	const int capacity = 65536;
	const int intv_radius = (capacity >> 1);
	unpred_vec<T> unpred_data;
	T * U_pos = decompressed_U;
	T * V_pos = decompressed_V;
	// offsets to get six adjacent triangle indices
	// the 7-th rolls back to T0
	/*
			T3	T4
		T2	X 	T5
		T1	T0(T6)
	*/
	const int offsets[7] = {
		-(int)r2, -(int)r2 - 1, -1, (int)r2, (int)r2+1, 1, -(int)r2
	};
	const T x[6][3] = {
		{1, 0, 1},
		{0, 0, 1},
		{0, 1, 1},
		{0, 1, 0},
		{1, 1, 0},
		{1, 0, 0}
	};
	const T y[6][3] = {
		{0, 0, 1},
		{0, 1, 1},
		{0, 1, 0},
		{1, 1, 0},
		{1, 0, 0},
		{1, 0, 1}
	};
	T inv_C[6][4];
	for(int i=0; i<6; i++){
		get_adjugate_matrix_for_position(x[i][0], x[i][1], x[i][2], y[i][0], y[i][1], y[i][2], inv_C[i]);
	}
	int index_offset[6][2][2];
	for(int i=0; i<6; i++){
		for(int j=0; j<2; j++){
			index_offset[i][j][0] = x[i][j] - x[i][2];
			index_offset[i][j][1] = y[i][j] - y[i][2];
		}
	}
	double threshold = std::numeric_limits<double>::epsilon();
	// conditions_2d cond;
	for(int i=0; i<r1; i++){ //DH
		// printf("start %d row\n", i);
		T * cur_U_pos = U_pos;
		T * cur_V_pos = V_pos;
		for(int j=0; j<r2; j++){ //DW
			size_t vertex_index = i * r2 + j;
			double required_eb;
			// if(index_need_to_fix.find(vertex_index) != index_need_to_fix.end()){
			// 	required_eb = modified_eb;
			// }
			// else{
			required_eb = max_pwr_eb;
			// derive eb given six adjacent triangles
			for(int k=0; k<6; k++){
				bool in_mesh = true;
				for(int p=0; p<2; p++){
					// reserved order!
					if(!(in_range(i + index_offset[k][p][1], (int)r1) && in_range(j + index_offset[k][p][0], (int)r2))){
						in_mesh = false;
						break;
					}
				}
				if(in_mesh){
					required_eb = MINF(required_eb, derive_cp_eb_for_positions_online(cur_U_pos[offsets[k]], cur_U_pos[offsets[k+1]], cur_U_pos[0],
						cur_V_pos[offsets[k]], cur_V_pos[offsets[k+1]], cur_V_pos[0], inv_C[k]));
				}
			}
			// }

			if(index_need_to_fix.find(vertex_index) != index_need_to_fix.end()){
				required_eb = MINF(required_eb, modified_eb);
			}

			if(required_eb > 0){
				if(WRITE_OUT_EB == 1){
					eb_result[vertex_index] = required_eb;
				}
				
				bool unpred_flag = false;
				T decompressed[2];
				// compress U and V
				for(int k=0; k<2; k++){
					T * cur_data_pos = (k == 0) ? cur_U_pos : cur_V_pos;
					T cur_data = *cur_data_pos;
					double abs_eb = fabs(cur_data) * required_eb;
					eb_quant_index_pos[k] = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
					// eb_quant_index_pos[k] = eb_linear_quantize(abs_eb, 1e-3);
					if(eb_quant_index_pos[k] > 0){
						// get adjacent data and perform Lorenzo
						/*
							d2 X
							d0 d1
						*/
						T d0 = (i && j) ? cur_data_pos[-1 - r2] : 0;
						T d1 = (i) ? cur_data_pos[-r2] : 0;
						T d2 = (j) ? cur_data_pos[-1] : 0;
						T pred = d1 + d2 - d0;
						double diff = cur_data - pred;
						double quant_diff = fabs(diff) / abs_eb + 1;
						if(quant_diff < capacity){
							quant_diff = (diff > 0) ? quant_diff : -quant_diff;
							int quant_index = (int)(quant_diff/2) + intv_radius;
							data_quant_index_pos[k] = quant_index;
							decompressed[k] = pred + 2 * (quant_index - intv_radius) * abs_eb; 
							// check original data
							if(fabs(decompressed[k] - cur_data) >= abs_eb){
								unpred_flag = true;
								break;
							}
						}
						else{
							unpred_flag = true;
							break;
						}
					}
					else unpred_flag = true;
				}
				if(unpred_flag){
					// recover quant index
					*(eb_quant_index_pos ++) = 0;
					*(eb_quant_index_pos ++) = 0;
					*(data_quant_index_pos ++) = intv_radius;
					*(data_quant_index_pos ++) = intv_radius;
					unpred_data.push_back(*cur_U_pos);
					unpred_data.push_back(*cur_V_pos);
				}
				else{
					eb_quant_index_pos += 2;
					data_quant_index_pos += 2;
					// assign decompressed data
					*cur_U_pos = decompressed[0];
					*cur_V_pos = decompressed[1];
				}
			}
			else{
				// record as unpredictable data
				*(eb_quant_index_pos ++) = 0;
				*(eb_quant_index_pos ++) = 0;
				*(data_quant_index_pos ++) = intv_radius;
				*(data_quant_index_pos ++) = intv_radius;
				unpred_data.push_back(*cur_U_pos);
				unpred_data.push_back(*cur_V_pos);
			}
			cur_U_pos ++, cur_V_pos ++;
		}
		U_pos += r2;
		V_pos += r2;
	}
	if(WRITE_OUT_EB == 1){
		writefile("/home/mxi235/data/eb_result/eb_result_rel.bin", &eb_result[0], eb_result.size());
		writefile("/home/mxi235/data/eb_result/rel_dec_U.bin", decompressed_U, num_elements);
		writefile("/home/mxi235/data/eb_result/rel_dec_V.bin", decompressed_V, num_elements);
	}
	free(decompressed_U);
	free(decompressed_V);
	printf("offsets eb_q, data_q, unpred: %ld %ld %ld\n", eb_quant_index_pos - eb_quant_index, data_quant_index_pos - data_quant_index, unpred_data.size());
	unsigned char * compressed = (unsigned char *) malloc(2*num_elements*sizeof(T));
	unsigned char * compressed_pos = compressed;
	write_variable_to_dst(compressed_pos, base);
	write_variable_to_dst(compressed_pos, threshold);
	write_variable_to_dst(compressed_pos, intv_radius);
	size_t unpredictable_count = unpred_data.size();
	write_variable_to_dst(compressed_pos, unpredictable_count);
	write_array_to_dst(compressed_pos, (T *)&unpred_data[0], unpredictable_count);	
	Huffman_encode_tree_and_data(2*1024, eb_quant_index, 2*num_elements, compressed_pos);
	free(eb_quant_index);
	Huffman_encode_tree_and_data(2*capacity, data_quant_index, 2*num_elements, compressed_pos);
	printf("pos = %ld\n", compressed_pos - compressed);
	free(data_quant_index);
	compressed_size = compressed_pos - compressed;
	return compressed;	
}

template
unsigned char *
sz_compress_cp_preserve_2d_fix(const float * U, const float * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb,double modified_eb,const std::set<size_t> &index_need_to_fix);

// template
// unsigned char *
// sz_compress_cp_preserve_2d_fix(const double * U, const double * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb,double modified_eb,const std::set<size_t> &index_need_to_fix);

template<typename T>
unsigned char *
sz_compress_cp_preserve_2d_record_vertex(const T * U, const T * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb,const std::set<size_t> &index_need_to_fix){
	size_t num_elements = r1 * r2;
	unsigned char * bitmap;
	if (index_need_to_fix.size() != 0){
		printf("parpare bitmap\n");
		//准备bitmap#####################
		bitmap = (unsigned char *) malloc(num_elements*sizeof(unsigned char));
		if (bitmap == NULL) {
		fprintf(stderr, "Failed to allocate memory for bitmap\n");
		exit(1);
		}
		// set all to 0
		// memset(bitmap, 0, num_elements * sizeof(T));
		memset(bitmap, 0, num_elements * sizeof(unsigned char));
		//set index_need_to_fix to 1
		for(auto it = index_need_to_fix.begin(); it != index_need_to_fix.end(); ++it){
			assert(*it < num_elements);
			bitmap[*it] = 1;
		}
		size_t intArrayLength = num_elements;
		// 准备输出的长度
		size_t num_bytes = (intArrayLength % 8 == 0) ? intArrayLength / 8 : intArrayLength / 8 + 1;
		// unsigned char *compressedArray = new unsigned char[num_bytes];
		// unsigned char *compressed_pos = compressedArray;  // 指针指向压缩数组的开始
		// convertIntArray2ByteArray_fast_1b_to_result_sz(bitmap, intArrayLength, compressed_pos);
		//准备bitmap#####################
	}


	T * decompressed_U = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_U, U, num_elements*sizeof(T));
	T * decompressed_V = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_V, V, num_elements*sizeof(T));
	int * eb_quant_index = (int *) malloc(2*num_elements*sizeof(int));
	int * data_quant_index = (int *) malloc(2*num_elements*sizeof(int));
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	// next, row by row
	const int base = 4;
	const double log_of_base = log2(base);
	const int capacity = 65536;
	const int intv_radius = (capacity >> 1);
	unpred_vec<T> unpred_data;
	T * U_pos = decompressed_U;
	T * V_pos = decompressed_V;
	// offsets to get six adjacent triangle indices
	// the 7-th rolls back to T0
	/*
			T3	T4
		T2	X 	T5
		T1	T0(T6)
	*/
	const int offsets[7] = {
		-(int)r2, -(int)r2 - 1, -1, (int)r2, (int)r2+1, 1, -(int)r2
	};
	const T x[6][3] = {
		{1, 0, 1},
		{0, 0, 1},
		{0, 1, 1},
		{0, 1, 0},
		{1, 1, 0},
		{1, 0, 0}
	};
	const T y[6][3] = {
		{0, 0, 1},
		{0, 1, 1},
		{0, 1, 0},
		{1, 1, 0},
		{1, 0, 0},
		{1, 0, 1}
	};
	T inv_C[6][4];
	for(int i=0; i<6; i++){
		get_adjugate_matrix_for_position(x[i][0], x[i][1], x[i][2], y[i][0], y[i][1], y[i][2], inv_C[i]);
	}
	int index_offset[6][2][2];
	for(int i=0; i<6; i++){
		for(int j=0; j<2; j++){
			index_offset[i][j][0] = x[i][j] - x[i][2];
			index_offset[i][j][1] = y[i][j] - y[i][2];
		}
	}
	double threshold = std::numeric_limits<double>::epsilon();
	// conditions_2d cond;
	for(int i=0; i<r1; i++){ //DH
		// printf("start %d row\n", i);
		T * cur_U_pos = U_pos;
		T * cur_V_pos = V_pos;
		for(int j=0; j<r2; j++){ //DW
			size_t vertex_index = i * r2 + j;
			double required_eb;
			// if(index_need_to_fix.find(vertex_index) != index_need_to_fix.end()){
			// 	required_eb = modified_eb;
			// }
			// else{
			required_eb = max_pwr_eb;
			// derive eb given six adjacent triangles
			for(int k=0; k<6; k++){
				bool in_mesh = true;
				for(int p=0; p<2; p++){
					// reserved order!
					if(!(in_range(i + index_offset[k][p][1], (int)r1) && in_range(j + index_offset[k][p][0], (int)r2))){
						in_mesh = false;
						break;
					}
				}
				if(in_mesh){
					required_eb = MINF(required_eb, derive_cp_eb_for_positions_online(cur_U_pos[offsets[k]], cur_U_pos[offsets[k+1]], cur_U_pos[0],
						cur_V_pos[offsets[k]], cur_V_pos[offsets[k+1]], cur_V_pos[0], inv_C[k]));
				}
			}
			// }

			// if(index_need_to_fix.find(vertex_index) != index_need_to_fix.end()){
			// 	required_eb = MINF(required_eb, modified_eb);
			// }
			//record vertex不需要给另外的eb

			if(required_eb > 0){
				bool unpred_flag = false;
				T decompressed[2];
				// compress U and V
				for(int k=0; k<2; k++){
					T * cur_data_pos = (k == 0) ? cur_U_pos : cur_V_pos;
					T cur_data = *cur_data_pos;
					double abs_eb = fabs(cur_data) * required_eb;
					eb_quant_index_pos[k] = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
					// eb_quant_index_pos[k] = eb_linear_quantize(abs_eb, 1e-3);
					if(eb_quant_index_pos[k] > 0){
						// get adjacent data and perform Lorenzo
						/*
							d2 X
							d0 d1
						*/
						T d0 = (i && j) ? cur_data_pos[-1 - r2] : 0;
						T d1 = (i) ? cur_data_pos[-r2] : 0;
						T d2 = (j) ? cur_data_pos[-1] : 0;
						T pred = d1 + d2 - d0;
						double diff = cur_data - pred;
						double quant_diff = fabs(diff) / abs_eb + 1;
						if(quant_diff < capacity){
							quant_diff = (diff > 0) ? quant_diff : -quant_diff;
							int quant_index = (int)(quant_diff/2) + intv_radius;
							data_quant_index_pos[k] = quant_index;
							decompressed[k] = pred + 2 * (quant_index - intv_radius) * abs_eb; 
							// check original data
							if(fabs(decompressed[k] - cur_data) >= abs_eb){
								unpred_flag = true;
								break;
							}
						}
						else{
							unpred_flag = true;
							break;
						}
					}
					else unpred_flag = true;
				}
				if(unpred_flag){
					// recover quant index
					*(eb_quant_index_pos ++) = 0;
					*(eb_quant_index_pos ++) = 0;
					*(data_quant_index_pos ++) = intv_radius;
					*(data_quant_index_pos ++) = intv_radius;
					unpred_data.push_back(*cur_U_pos);
					unpred_data.push_back(*cur_V_pos);
				}
				else{
					eb_quant_index_pos += 2;
					data_quant_index_pos += 2;
					// assign decompressed data
					*cur_U_pos = decompressed[0];
					*cur_V_pos = decompressed[1];
				}
			}
			else{
				// record as unpredictable data
				*(eb_quant_index_pos ++) = 0;
				*(eb_quant_index_pos ++) = 0;
				*(data_quant_index_pos ++) = intv_radius;
				*(data_quant_index_pos ++) = intv_radius;
				unpred_data.push_back(*cur_U_pos);
				unpred_data.push_back(*cur_V_pos);
			}
			cur_U_pos ++, cur_V_pos ++;
		}
		U_pos += r2;
		V_pos += r2;
	}
	printf("ready to free decompressed_U &V\n");
	free(decompressed_U);
	free(decompressed_V);
	printf("offsets eb_q, data_q, unpred: %ld %ld %ld\n", eb_quant_index_pos - eb_quant_index, data_quant_index_pos - data_quant_index, unpred_data.size());
	unsigned char * compressed = (unsigned char *) malloc(2*num_elements*sizeof(T));
	unsigned char * compressed_pos = compressed;
	// 写index_need_to_fix的size
	write_variable_to_dst(compressed_pos, index_need_to_fix.size()); //size_t, index_need_to_fix的大小
	if (index_need_to_fix.size() != 0){
		convertIntArray2ByteArray_fast_1b_to_result_sz(bitmap, num_elements, compressed_pos);
		printf("bitmap pos = %ld\n", compressed_pos - compressed);
		//再写index_need_to_fix对应U和V的数据
		for (auto it = index_need_to_fix.begin(); it != index_need_to_fix.end(); it++){
			write_variable_to_dst(compressed_pos, U[*it]); //T, index_need_to_fix对应的U的值
		}
		for (auto it = index_need_to_fix.begin(); it != index_need_to_fix.end(); it++){
			write_variable_to_dst(compressed_pos, V[*it]); //T, index_need_to_fix对应的V的值
		}
		printf("index_need_to_fix lossless data pos = %ld\n", compressed_pos - compressed);
	}


	write_variable_to_dst(compressed_pos, base); //int
	write_variable_to_dst(compressed_pos, threshold); //double
	write_variable_to_dst(compressed_pos, intv_radius); //int
	size_t unpredictable_count = unpred_data.size();
	write_variable_to_dst(compressed_pos, unpredictable_count); //int, unpred_data的大小
	write_array_to_dst(compressed_pos, (T *)&unpred_data[0], unpredictable_count);	
	Huffman_encode_tree_and_data(2*1024, eb_quant_index, 2*num_elements, compressed_pos);
	free(eb_quant_index);
	Huffman_encode_tree_and_data(2*capacity, data_quant_index, 2*num_elements, compressed_pos);
	printf("pos = %ld\n", compressed_pos - compressed);//pos = 41812580
	free(data_quant_index);
	compressed_size = compressed_pos - compressed;
	return compressed;	
}

template
unsigned char *
sz_compress_cp_preserve_2d_record_vertex(const float * U, const float * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb,const std::set<size_t> &index_need_to_fix);

template<typename T>
unsigned char *
sz_compress_cp_preserve_2d_st2_fix(const T * U, const T * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb,double modified_eb,std::unordered_map<size_t, critical_point_t> & critical_points,const std::set<size_t> &index_need_to_fix){
	size_t num_elements = r1 * r2;
	T * decompressed_U = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_U, U, num_elements*sizeof(T));
	T * decompressed_V = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_V, V, num_elements*sizeof(T));
	int * eb_quant_index = (int *) malloc(2*num_elements*sizeof(int)); //这个index在cpszsos中是一倍，为什么？
	int * data_quant_index = (int *) malloc(2*num_elements*sizeof(int));
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	//build cp_exist vector
	std::vector<bool> cp_exist(2*(r1-1)*(r2-1), 0);
	for(auto it = critical_points.begin(); it != critical_points.end(); it++){
		cp_exist[it->first] = 1;
	}
	// next, row by row
	const int base = 4;
	const double log_of_base = log2(base);
	const int capacity = 65536;
	const int intv_radius = (capacity >> 1);
	unpred_vec<T> unpred_data;
	T * U_pos = decompressed_U;
	T * V_pos = decompressed_V;
	// offsets to get six adjacent triangle indices
	// the 7-th rolls back to T0
	/*
			T3	T4
		T2	X 	T5
		T1	T0(T6)
	*/
	const int offsets[7] = {
		-(int)r2, -(int)r2 - 1, -1, (int)r2, (int)r2+1, 1, -(int)r2
	};
	const T x[6][3] = {
		{1, 0, 1},
		{0, 0, 1},
		{0, 1, 1},
		{0, 1, 0},
		{1, 1, 0},
		{1, 0, 0}
	};
	const T y[6][3] = {
		{0, 0, 1},
		{0, 1, 1},
		{0, 1, 0},
		{1, 1, 0},
		{1, 0, 0},
		{1, 0, 1}
	};
	T inv_C[6][4];
	for(int i=0; i<6; i++){
		get_adjugate_matrix_for_position(x[i][0], x[i][1], x[i][2], y[i][0], y[i][1], y[i][2], inv_C[i]);
	}
	int index_offset[6][2][2];
	for(int i=0; i<6; i++){
		for(int j=0; j<2; j++){
			index_offset[i][j][0] = x[i][j] - x[i][2];
			index_offset[i][j][1] = y[i][j] - y[i][2];
		}
	}
	int cell_offset[6] = {
		-2*((int)r2-1)-1, -2*((int)r2-1)-2, -1, 0, 1, -2*((int)r2-1)
	};
	double threshold = std::numeric_limits<double>::epsilon();
	// conditions_2d cond;
	for(int i=0; i<r1; i++){ //DH
		// printf("start %d row\n", i);
		T * cur_U_pos = U_pos;
		T * cur_V_pos = V_pos;
		for(int j=0; j<r2; j++){ //DW
			size_t vertex_index = i * r2 + j;
			double required_eb;
			bool unpred_flag = false;
			bool verification_flag = false;
			required_eb = max_pwr_eb;
			// derive eb given six adjacent triangles
			for(int k=0; k<6; k++){
				bool in_mesh = true;
				for(int p=0; p<2; p++){
					// reserved order!
					if(!(in_range(i + index_offset[k][p][1], (int)r1) && in_range(j + index_offset[k][p][0], (int)r2))){
						in_mesh = false;
						break;
					}
				}
				if(in_mesh){
					// required_eb = MINF(required_eb, derive_cp_eb_for_positions_online(cur_U_pos[offsets[k]], cur_U_pos[offsets[k+1]], cur_U_pos[0],
					// 	cur_V_pos[offsets[k]], cur_V_pos[offsets[k+1]], cur_V_pos[0], inv_C[k]));
					bool original_has_cp = cp_exist[2*(i*(r2-1) + j) + cell_offset[k]];
					if (original_has_cp){
						unpred_flag = true;
						verification_flag = true;
						break;
					}
				}
			}
			T decompressed[2];

			int n_try = 0;
			while(!verification_flag){
				unpred_flag = false;
				for(int k=0; k<2; k++){
					T * cur_data_pos = (k == 0) ? cur_U_pos : cur_V_pos;
					T cur_data = *cur_data_pos;
					double abs_eb = fabs(cur_data) * required_eb;
					eb_quant_index_pos[k] = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);					
					// eb_quant_index_pos[k] = eb_linear_quantize(abs_eb, 1e-3);
					if(eb_quant_index_pos[k] > 0){
						// get adjacent data and perform Lorenzo
						/*
							d2 X
							d0 d1
						*/
						T d0 = (i && j) ? cur_data_pos[-1 - r2] : 0;
						T d1 = (i) ? cur_data_pos[-r2] : 0;
						T d2 = (j) ? cur_data_pos[-1] : 0;
						T pred = d1 + d2 - d0;
						double diff = cur_data - pred;
						double quant_diff = fabs(diff) / abs_eb + 1;
						if(quant_diff < capacity){
							quant_diff = (diff > 0) ? quant_diff : -quant_diff;
							int quant_index = (int)(quant_diff/2) + intv_radius;
							data_quant_index_pos[k] = quant_index;
							decompressed[k] = pred + 2 * (quant_index - intv_radius) * abs_eb; 
							// check original data
							if(fabs(decompressed[k] - cur_data) >= abs_eb){
								unpred_flag = true;
								break;
							}
						}
						else{
							unpred_flag = true;
							break;
						}
					}
					else unpred_flag = true;
				}
				if(unpred_flag) break;
				//verify cp in six adjacent triangles
				verification_flag = true;
				for(int k=0; k<6; k++){
					bool in_mesh = true;
					for(int p=0; p<2; p++){
						// reserved order!
						if(!(in_range(i + index_offset[k][p][1], (int)r1) && in_range(j + index_offset[k][p][0], (int)r2))){
							in_mesh = false;
							break;
						}
					}
					if(in_mesh){
						int indices[3];
						for(int p=0; p<2; p++){
							indices[p] = (i + index_offset[k][p][1])*r2 + (j + index_offset[k][p][0]);
						}
						indices[2] = i*r2 + j;

						double current_v[3][2];
						for(int p=0; p<2; p++){
							current_v[p][0] = decompressed_U[indices[p]];
							current_v[p][1] = decompressed_V[indices[p]];
						}
						current_v[2][0] = decompressed[0]; //我擦，这里需要用的是decompressed（预测值），而不是decompressed_U，但是另外两个是decompressed_U
						current_v[2][1] = decompressed[1];
						bool decompressed_has_cp = (check_cp(current_v) == 1);
						if (decompressed_has_cp){
							verification_flag = false;
							break;
						}
					}
				}
				//relax error bound
				required_eb /= 2;
				n_try ++;
				if ((!verification_flag) && (n_try >=3)){
					unpred_flag = true;
					verification_flag = true;
				}
			}
			if(unpred_flag){
				// recover quant index
				*(eb_quant_index_pos ++) = 0;
				*(eb_quant_index_pos ++) = 0;
				*(data_quant_index_pos ++) = intv_radius;
				*(data_quant_index_pos ++) = intv_radius;
				unpred_data.push_back(*cur_U_pos);
				unpred_data.push_back(*cur_V_pos);
			}
			else{
				eb_quant_index_pos += 2;
				data_quant_index_pos += 2;
				// assign decompressed data
				*cur_U_pos = decompressed[0];
				*cur_V_pos = decompressed[1];
			}
			cur_U_pos ++, cur_V_pos ++;
		}
		U_pos += r2;
		V_pos += r2;
	}
	free(decompressed_U);
	free(decompressed_V);
	printf("offsets eb_q, data_q, unpred: %ld %ld %ld\n", eb_quant_index_pos - eb_quant_index, data_quant_index_pos - data_quant_index, unpred_data.size());
	unsigned char * compressed = (unsigned char *) malloc(2*num_elements*sizeof(T));
	unsigned char * compressed_pos = compressed;
	write_variable_to_dst(compressed_pos, base);
	write_variable_to_dst(compressed_pos, threshold);
	write_variable_to_dst(compressed_pos, intv_radius);
	size_t unpredictable_count = unpred_data.size();
	write_variable_to_dst(compressed_pos, unpredictable_count);
	write_array_to_dst(compressed_pos, (T *)&unpred_data[0], unpredictable_count);	
	Huffman_encode_tree_and_data(2*1024, eb_quant_index, 2*num_elements, compressed_pos);
	free(eb_quant_index);
	Huffman_encode_tree_and_data(2*capacity, data_quant_index, 2*num_elements, compressed_pos);
	printf("pos = %ld\n", compressed_pos - compressed);
	free(data_quant_index);
	compressed_size = compressed_pos - compressed;
	return compressed;	
}

template
unsigned char *
sz_compress_cp_preserve_2d_st2_fix(const float * U, const float * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb,double modified_eb,std::unordered_map<size_t, critical_point_t> & critical_points,const std::set<size_t> &index_need_to_fix);

// abs error bound version
template<typename T>
unsigned char *
sz_compress_cp_preserve_2d_online_abs_record_vertex(const T * U, const T * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb, const std::set<size_t> &index_need_to_fix){
	std::vector<float> eb_result(r1*r2, 0);
	size_t num_elements = r1 * r2;
	size_t intArrayLength = num_elements;
	size_t num_bytes = (intArrayLength % 8 == 0) ? intArrayLength / 8 : intArrayLength / 8 + 1;
	unsigned char * bitmap;
	if(index_need_to_fix.size() != 0){
		printf("parpare bitmap\n");
		//准备bitmap#####################
		bitmap = (unsigned char *) malloc(num_elements*sizeof(unsigned char));
		if (bitmap == NULL) {
		fprintf(stderr, "Failed to allocate memory for bitmap\n");
		exit(1);
		}
		// set all to 0
		// memset(bitmap, 0, num_elements * sizeof(T));
		memset(bitmap, 0, num_elements * sizeof(unsigned char));
		//set index_need_to_fix to 1
		for(auto it = index_need_to_fix.begin(); it != index_need_to_fix.end(); ++it){
			assert(*it < num_elements);
			bitmap[*it] = 1;
		}
		//准备bitmap#####################
	}
	T * decompressed_U = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_U, U, num_elements*sizeof(T));
	T * decompressed_V = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_V, V, num_elements*sizeof(T));
	int * eb_quant_index = (int *) malloc(2*num_elements*sizeof(int));
	int * data_quant_index = (int *) malloc(2*num_elements*sizeof(int));
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	// next, row by row
	const int base = 4;
	const double log_of_base = log2(base);
	const int capacity = 65536;
	const int intv_radius = (capacity >> 1);
	unpred_vec<T> unpred_data;
	T * U_pos = decompressed_U;
	T * V_pos = decompressed_V;
	// offsets to get six adjacent triangle indices
	// the 7-th rolls back to T0
	/*
			T3	T4
		T2	X 	T5
		T1	T0(T6)
	*/
	const int offsets[7] = {
		-(int)r2, -(int)r2 - 1, -1, (int)r2, (int)r2+1, 1, -(int)r2
	};
	const T x[6][3] = {
		{1, 0, 1},
		{0, 0, 1},
		{0, 1, 1},
		{0, 1, 0},
		{1, 1, 0},
		{1, 0, 0}
	};
	const T y[6][3] = {
		{0, 0, 1},
		{0, 1, 1},
		{0, 1, 0},
		{1, 1, 0},
		{1, 0, 0},
		{1, 0, 1}
	};
	T inv_C[6][4];
	for(int i=0; i<6; i++){
		get_adjugate_matrix_for_position(x[i][0], x[i][1], x[i][2], y[i][0], y[i][1], y[i][2], inv_C[i]);
	}
	int index_offset[6][2][2];
	for(int i=0; i<6; i++){
		for(int j=0; j<2; j++){
			index_offset[i][j][0] = x[i][j] - x[i][2];
			index_offset[i][j][1] = y[i][j] - y[i][2];
		}
	}
	double threshold = std::numeric_limits<double>::epsilon();
	// conditions_2d cond;
	for(int i=0; i<r1; i++){
		// printf("start %d row\n", i);
		T * cur_U_pos = U_pos;
		T * cur_V_pos = V_pos;
		for(int j=0; j<r2; j++){
			double required_eb = max_pwr_eb;
			// derive eb given six adjacent triangles
			if((cur_U_pos[0] == 0) || (cur_V_pos[0] == 0)){
				required_eb = 0;
			}
			else{
				for(int k=0; k<6; k++){
					bool in_mesh = true;
					for(int p=0; p<2; p++){
						// reserved order!
						if(!(in_range(i + index_offset[k][p][1], (int)r1) && in_range(j + index_offset[k][p][0], (int)r2))){
							in_mesh = false;
							break;
						}
					}
					if(in_mesh){
						// derive abs eb
						required_eb = MINF(required_eb, derive_cp_eb_for_positions_online_abs(cur_U_pos[offsets[k]], cur_U_pos[offsets[k+1]], cur_U_pos[0],
							cur_V_pos[offsets[k]], cur_V_pos[offsets[k+1]], cur_V_pos[0], inv_C[k]));
					}
				}				
			}
			if(required_eb > 0){
				if(WRITE_OUT_EB ==1){
					eb_result[i*r2 + j] = required_eb;
				}
				bool unpred_flag = false;
				T decompressed[2];
				// compress U and V
				for(int k=0; k<2; k++){
					T * cur_data_pos = (k == 0) ? cur_U_pos : cur_V_pos;
					T cur_data = *cur_data_pos;
					double abs_eb = required_eb;
					eb_quant_index_pos[k] = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
					// eb_quant_index_pos[k] = eb_linear_quantize(abs_eb, 1e-3);
					if(eb_quant_index_pos[k] > 0){
						// get adjacent data and perform Lorenzo
						/*
							d2 X
							d0 d1
						*/
						T d0 = (i && j) ? cur_data_pos[-1 - r2] : 0;
						T d1 = (i) ? cur_data_pos[-r2] : 0;
						T d2 = (j) ? cur_data_pos[-1] : 0;
						T pred = d1 + d2 - d0;
						double diff = cur_data - pred;
						double quant_diff = fabs(diff) / abs_eb + 1;
						if(quant_diff < capacity){
							quant_diff = (diff > 0) ? quant_diff : -quant_diff;
							int quant_index = (int)(quant_diff/2) + intv_radius;
							data_quant_index_pos[k] = quant_index;
							decompressed[k] = pred + 2 * (quant_index - intv_radius) * abs_eb; 
							// check original data
							if(fabs(decompressed[k] - cur_data) >= abs_eb){
								unpred_flag = true;
								break;
							}
						}
						else{
							unpred_flag = true;
							break;
						}
					}
					else unpred_flag = true;
				}
				if(unpred_flag){
					// recover quant index
					*(eb_quant_index_pos ++) = 0;
					*(eb_quant_index_pos ++) = 0;
					*(data_quant_index_pos ++) = intv_radius;
					*(data_quant_index_pos ++) = intv_radius;
					unpred_data.push_back(*cur_U_pos);
					unpred_data.push_back(*cur_V_pos);
				}
				else{
					eb_quant_index_pos += 2;
					data_quant_index_pos += 2;
					// assign decompressed data
					*cur_U_pos = decompressed[0];
					*cur_V_pos = decompressed[1];
				}
			}
			else{
				// record as unpredictable data
				*(eb_quant_index_pos ++) = 0;
				*(eb_quant_index_pos ++) = 0;
				*(data_quant_index_pos ++) = intv_radius;
				*(data_quant_index_pos ++) = intv_radius;
				unpred_data.push_back(*cur_U_pos);
				unpred_data.push_back(*cur_V_pos);
			}
			cur_U_pos ++, cur_V_pos ++;
		}
		U_pos += r2;
		V_pos += r2;
	}
	if(WRITE_OUT_EB == 1){
		writefile("/home/mxi235/data/eb_result/eb_result_abs.bin", &eb_result[0], eb_result.size());
		writefile("/home/mxi235/data/eb_result/abs_dec_U.bin", decompressed_U, num_elements);
		writefile("/home/mxi235/data/eb_result/abs_dec_V.bin", decompressed_V, num_elements);
	}
	free(decompressed_U);
	free(decompressed_V);
	printf("offsets eb_q, data_q, unpred: %ld %ld %ld\n", eb_quant_index_pos - eb_quant_index, data_quant_index_pos - data_quant_index, unpred_data.size());
	unsigned char * compressed = (unsigned char *) malloc(3*num_elements*sizeof(T));
	unsigned char * compressed_pos = compressed;
	//写index_need_to_fix的大小
	write_variable_to_dst(compressed_pos, index_need_to_fix.size());
	printf("index_need_to_fix pos = %ld\n", compressed_pos - compressed);
	printf("index_need_to_fix size = %ld\n", index_need_to_fix.size());
	if(index_need_to_fix.size() != 0){
		//修改：先写bitmap
		// write_variable_to_dst(compressed_pos,num_elements); // 处理后的bitmap的长度 size_t
		//write_array_to_dst(compressed_pos, compressedArray, num_bytes);
		convertIntArray2ByteArray_fast_1b_to_result_sz(bitmap, num_elements, compressed_pos);
		printf("bitmap pos = %ld\n", compressed_pos - compressed);
		//再写index_need_to_fix对应U和V的数据
		for (auto it = index_need_to_fix.begin(); it != index_need_to_fix.end(); it++){
			write_variable_to_dst(compressed_pos, U[*it]); //T, index_need_to_fix对应的U的值
		}
		for (auto it = index_need_to_fix.begin(); it != index_need_to_fix.end(); it++){
			write_variable_to_dst(compressed_pos, V[*it]); //T, index_need_to_fix对应的V的值
		}
		printf("index_need_to_fix lossless data pos = %ld\n", compressed_pos - compressed);
	}
	write_variable_to_dst(compressed_pos, base);
	write_variable_to_dst(compressed_pos, threshold);
	write_variable_to_dst(compressed_pos, intv_radius);
	size_t unpredictable_count = unpred_data.size();
	write_variable_to_dst(compressed_pos, unpredictable_count);
	write_array_to_dst(compressed_pos, (T *)&unpred_data[0], unpredictable_count);	
	Huffman_encode_tree_and_data(2*1024, eb_quant_index, 2*num_elements, compressed_pos);
	free(eb_quant_index);
	Huffman_encode_tree_and_data(2*capacity, data_quant_index, 2*num_elements, compressed_pos);
	printf("pos = %ld\n", compressed_pos - compressed);
	free(data_quant_index);
	compressed_size = compressed_pos - compressed;
	return compressed;	
}

template
unsigned char *
sz_compress_cp_preserve_2d_online_abs_record_vertex(const float * U, const float * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb,const std::set<size_t> &index_need_to_fix);

template
unsigned char *
sz_compress_cp_preserve_2d_online_abs_record_vertex(const double * U, const double * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb,const std::set<size_t> &index_need_to_fix);


template<typename T>
static int 
check_cp_numeric(T v[3][2]){
  T mu[3]; // check intersection
  bool succ2 = ftk::inverse_lerp_s2v2(v, mu);
  if (!succ2) return -1;
	return 1;
}

template<typename T>
static vector<bool> 
compute_cp_numeric(const T * U, const T * V, int r1, int r2){
	// check cp for all cells
	vector<bool> cp_exist(2*(r1-1)*(r2-1), 0);
	for(int i=0; i<r1-1; i++){
		for(int j=0; j<r2-1; j++){
			int indices[3];
			indices[0] = i*r2 + j;
			indices[1] = (i+1)*r2 + j;
			indices[2] = (i+1)*r2 + (j+1); 
			T vf[3][2];
			// cell index 0
			for(int p=0; p<3; p++){
				vf[p][0] = U[indices[p]];
				vf[p][1] = V[indices[p]];
			}
			cp_exist[2*(i * (r2-1) + j)] = (check_cp_numeric(vf) == 1);
			// cell index 1
			indices[1] = i*r2 + (j+1);
			vf[1][0] = U[indices[1]];
			vf[1][1] = V[indices[1]];
			cp_exist[2*(i * (r2-1) + j) + 1] = (check_cp_numeric(vf) == 1);
		}
	}
	return cp_exist;	
}


// TODO:这个还没修改适配
template<typename T>
unsigned char *
sz_compress_cp_preserve_2d_online_abs_relax_FN(const T * U, const T * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb){
	size_t num_elements = r1 * r2;
	T * decompressed_U = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_U, U, num_elements*sizeof(T));
	T * decompressed_V = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_V, V, num_elements*sizeof(T));
	int * eb_quant_index = (int *) malloc(2*num_elements*sizeof(int));
	int * data_quant_index = (int *) malloc(2*num_elements*sizeof(int));
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	// next, row by row
	const int base = 4;
	const double log_of_base = log2(base);
	const int capacity = 65536;
	const int intv_radius = (capacity >> 1);
	unpred_vec<T> unpred_data;
	T * U_pos = decompressed_U;
	T * V_pos = decompressed_V;
	// offsets to get six adjacent triangle indices
	// the 7-th rolls back to T0
	/*
			T3	T4
		T2	X 	T5
		T1	T0(T6)
	*/
	const int offsets[7] = {
		-(int)r2, -(int)r2 - 1, -1, (int)r2, (int)r2+1, 1, -(int)r2
	};
	const T x[6][3] = {
		{1, 0, 1},
		{0, 0, 1},
		{0, 1, 1},
		{0, 1, 0},
		{1, 1, 0},
		{1, 0, 0}
	};
	const T y[6][3] = {
		{0, 0, 1},
		{0, 1, 1},
		{0, 1, 0},
		{1, 1, 0},
		{1, 0, 0},
		{1, 0, 1}
	};
	int index_offset[6][2][2];
	for(int i=0; i<6; i++){
		for(int j=0; j<2; j++){
			index_offset[i][j][0] = x[i][j] - x[i][2];
			index_offset[i][j][1] = y[i][j] - y[i][2];
		}
	}
	// offset relative to 2*(i*r2 + j)
	// note: width for cells is 2*(r2-1)
	int cell_offset[6] = {
		-2*((int)r2-1)-1, -2*((int)r2-1)-2, -1, 0, 1, -2*((int)r2-1)
	};
	double threshold = std::numeric_limits<double>::epsilon();
	// check cp for all cells
	vector<bool> cp_exist = compute_cp_numeric(U, V, r1, r2);
	int count = 0;
	int max_count = 1;
	for(int i=0; i<r1; i++){
		// printf("start %d row\n", i);
		T * cur_U_pos = U_pos;
		T * cur_V_pos = V_pos;
		for(int j=0; j<r2; j++){
			double abs_eb = max_pwr_eb;
			bool unpred_flag = false;
			T decompressed[2];
			// compress data and then verify
			bool verification_flag = false;
			if((*cur_U_pos == 0) && (*cur_V_pos == 0)){
				verification_flag = true;
				unpred_flag = true;
			}
			else{
				// check if cp exists in adjacent cells
				for(int k=0; k<6; k++){
					bool in_mesh = true;
					for(int p=0; p<2; p++){
						// reserved order!
						if(!(in_range(i + index_offset[k][p][1], (int)r1) && in_range(j + index_offset[k][p][0], (int)r2))){
							in_mesh = false;
							break;
						}
					}
					if(in_mesh){
						bool original_has_cp = cp_exist[2*(i*(r2-1) + j) + cell_offset[k]];
						if(original_has_cp){
							unpred_flag = true;
							verification_flag = true;
							break;
						}
					}
				}
			}
			while(!verification_flag){
				*eb_quant_index_pos = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
				unpred_flag = false;
				// compress U and V
				for(int k=0; k<2; k++){
					T * cur_data_pos = (k == 0) ? cur_U_pos : cur_V_pos;
					T cur_data = *cur_data_pos;
					// get adjacent data and perform Lorenzo
					/*
						d2 X
						d0 d1
					*/
					T d0 = (i && j) ? cur_data_pos[-1 - r2] : 0;
					T d1 = (i) ? cur_data_pos[-r2] : 0;
					T d2 = (j) ? cur_data_pos[-1] : 0;
					T pred = d1 + d2 - d0;
					T diff = cur_data - pred;
					T quant_diff = std::abs(diff) / abs_eb + 1;
					if(quant_diff < capacity){
						quant_diff = (diff > 0) ? quant_diff : -quant_diff;
						int quant_index = (int)(quant_diff/2) + intv_radius;
						data_quant_index_pos[k] = quant_index;
						decompressed[k] = pred + 2 * (quant_index - intv_radius) * abs_eb; 
					}
					else{
						unpred_flag = true;
						break;
					}
				}
				if(unpred_flag) break;
				// verify cp in six adjacent triangles
				verification_flag = true;
				for(int k=0; k<6; k++){
					bool in_mesh = true;
					for(int p=0; p<2; p++){
						// reserved order!
						if(!(in_range(i + index_offset[k][p][1], (int)r1) && in_range(j + index_offset[k][p][0], (int)r2))){
							in_mesh = false;
							break;
						}
					}
					if(in_mesh){
						T vf[3][2];
						vf[0][0] = cur_U_pos[offsets[k]];
						vf[1][0] = cur_U_pos[offsets[k+1]];
						vf[2][0] = decompressed[0];
						vf[0][1] = cur_V_pos[offsets[k]];
						vf[1][1] = cur_V_pos[offsets[k+1]];
						vf[2][1] = decompressed[1];
						bool decompressed_has_cp = (check_cp_numeric(vf) == 1);
						if(decompressed_has_cp){
							verification_flag = false;
							break;
						}
					}
				}
				// relax error bound
				abs_eb /= 2;
				count ++;
				if((!verification_flag) && (count > max_count)){
					unpred_flag = true;
					verification_flag = true;					
				}
			}
			if(unpred_flag){
				// recover quant index
				*(eb_quant_index_pos ++) = 0;
				*(eb_quant_index_pos ++) = 0;
				*(data_quant_index_pos ++) = intv_radius;
				*(data_quant_index_pos ++) = intv_radius;
				unpred_data.push_back(*cur_U_pos);
				unpred_data.push_back(*cur_V_pos);
			}
			else{
				eb_quant_index_pos[1] = eb_quant_index_pos[0];
				eb_quant_index_pos += 2;
				data_quant_index_pos += 2;
				*cur_U_pos = decompressed[0];
				*cur_V_pos = decompressed[1];
			}
			cur_U_pos ++, cur_V_pos ++;
		}
		U_pos += r2;
		V_pos += r2;
	}
	free(decompressed_U);
	free(decompressed_V);
	printf("offsets eb_q, data_q, unpred: %ld %ld %ld\n", eb_quant_index_pos - eb_quant_index, data_quant_index_pos - data_quant_index, unpred_data.size());
	unsigned char * compressed = (unsigned char *) malloc(2*num_elements*sizeof(T));
	unsigned char * compressed_pos = compressed;
	write_variable_to_dst(compressed_pos, base);
	write_variable_to_dst(compressed_pos, threshold);
	write_variable_to_dst(compressed_pos, intv_radius);
	size_t unpredictable_count = unpred_data.size();
	write_variable_to_dst(compressed_pos, unpredictable_count);
	write_array_to_dst(compressed_pos, (T *)&unpred_data[0], unpredictable_count);	
	Huffman_encode_tree_and_data(2*1024, eb_quant_index, 2*num_elements, compressed_pos);
	free(eb_quant_index);
	Huffman_encode_tree_and_data(2*capacity, data_quant_index, 2*num_elements, compressed_pos);
	printf("pos = %ld\n", compressed_pos - compressed);
	free(data_quant_index);
	compressed_size = compressed_pos - compressed;
	return compressed;	
}

template
unsigned char *
sz_compress_cp_preserve_2d_online_abs_relax_FN(const float * U, const float * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb);

template
unsigned char *
sz_compress_cp_preserve_2d_online_abs_relax_FN(const double * U, const double * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb);


// 计算块的起始位置和大小的函数
std::tuple<int, int, int, int> calculate_block(int n, int m, int num_threads, int thread_id) {
    int t = std::sqrt(num_threads);
	int rows_per_block = n / t;
    int cols_per_block = m / t;
    int extra_rows = n % t;
    int extra_cols = m % t;

    int row_id = thread_id / t;
    int col_id = thread_id % t;

    int start_row = row_id * rows_per_block + std::min(row_id, extra_rows);
    int start_col = col_id * cols_per_block + std::min(col_id, extra_cols);

    int block_height = rows_per_block + (row_id < extra_rows ? 1 : 0);
    int block_width = cols_per_block + (col_id < extra_cols ? 1 : 0);

    return std::make_tuple(start_row, start_col, block_height, block_width);
}

template<typename T>
unsigned char *
omp_sz_compress_cp_preserve_2d_record_vertex(const T * U, const T * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb,const std::set<size_t> &index_need_to_fix,int n_threads, T * &decompressed_U_ptr, T * &decompressed_V_ptr,std::string eb_type){
	size_t num_elements = r1 * r2;
	size_t intArrayLength = num_elements;
	size_t num_bytes = (intArrayLength % 8 == 0) ? intArrayLength / 8 : intArrayLength / 8 + 1;
	unsigned char * bitmap;
	if(index_need_to_fix.size() != 0){
		printf("parpare bitmap\n");
		//准备bitmap#####################
		bitmap = (unsigned char *) malloc(num_elements*sizeof(unsigned char));
		if (bitmap == NULL) {
		fprintf(stderr, "Failed to allocate memory for bitmap\n");
		exit(1);
		}
		// set all to 0
		// memset(bitmap, 0, num_elements * sizeof(T));
		memset(bitmap, 0, num_elements * sizeof(unsigned char));
		//set index_need_to_fix to 1
		for(auto it = index_need_to_fix.begin(); it != index_need_to_fix.end(); ++it){
			assert(*it < num_elements);
			bitmap[*it] = 1;
		}
		//准备bitmap#####################
	}
	auto prepare_comp_start = std::chrono::high_resolution_clock::now();
	//确定线程数是不是平方数
	int num_threads = n_threads;
	if (sqrt(num_threads) != (int)sqrt(num_threads)){
		printf("The number of threads must be a square number!\n");
		exit(0);
	}
	T * decompressed_U = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_U, U, num_elements*sizeof(T));
	T * decompressed_V = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_V, V, num_elements*sizeof(T));

	// T * decompressed_U_2 = (T *) malloc(num_elements*sizeof(T));
	// memcpy(decompressed_U_2, U, num_elements*sizeof(T));
	// T * decompressed_V_2 = (T *) malloc(num_elements*sizeof(T));
	// memcpy(decompressed_V_2, V, num_elements*sizeof(T));

	int * eb_quant_index = (int *) malloc(2*num_elements*sizeof(int));
	int * data_quant_index = (int *) malloc(2*num_elements*sizeof(int));
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	const int base = 4;
	const double log_of_base = log2(base);
	const int capacity = 65536;
	const int intv_radius = (capacity >> 1);
	std::vector<unpred_vec<T>> unpred_data_thread(num_threads);
	// T * U_pos = decompressed_U;
	// T * V_pos = decompressed_V;
	// size_t stateNum_eb_quant = 2*1024;
	// size_t stateNum_data_quant = 2*capacity;
	// HuffmanTree * huffmanTree_eb_quant = createHuffmanTree(stateNum_eb_quant);
	// HuffmanTree * huffmanTree_data_quant = createHuffmanTree(stateNum_data_quant);
	// offsets to get six adjacent triangle indices
	// the 7-th rolls back to T0
	/*
			T3	T4
		T2	X 	T5
		T1	T0(T6)
	*/
	const int offsets[7] = {
		-(int)r2, -(int)r2 - 1, -1, (int)r2, (int)r2+1, 1, -(int)r2
	};
	const T x[6][3] = {
		{1, 0, 1},
		{0, 0, 1},
		{0, 1, 1},
		{0, 1, 0},
		{1, 1, 0},
		{1, 0, 0}
	};
	const T y[6][3] = {
		{0, 0, 1},
		{0, 1, 1},
		{0, 1, 0},
		{1, 1, 0},
		{1, 0, 0},
		{1, 0, 1}
	};
	T inv_C[6][4];
	for(int i=0; i<6; i++){
		get_adjugate_matrix_for_position(x[i][0], x[i][1], x[i][2], y[i][0], y[i][1], y[i][2], inv_C[i]);
	}
	int index_offset[6][2][2];
	for(int i=0; i<6; i++){
		for(int j=0; j<2; j++){
			index_offset[i][j][0] = x[i][j] - x[i][2];
			index_offset[i][j][1] = y[i][j] - y[i][2];
		}
	}
	double threshold = std::numeric_limits<double>::epsilon();
	int n = r1, m = r2;



	int t = sqrt(num_threads);
    // 计算每个块的大小
    int block_height = n / t;
    int block_width = m / t;

    // 处理余数情况（如果 n 或 m 不能被 t 整除）
    int remaining_rows = n % t;
    int remaining_cols = m % t;

    // 存储划分线的位置
    std::vector<int> dividing_rows;
    std::vector<int> dividing_cols;
    for (int i = 1; i < t; ++i) {
        dividing_rows.push_back(i * block_height);
        dividing_cols.push_back(i * block_width);
    }

	std::vector<bool> is_dividing_row(n, false);
	std::vector<bool> is_dividing_col(m, false);
	for (int i = 0; i < dividing_rows.size(); ++i) {
		is_dividing_row[dividing_rows[i]] = true;
	}
	for (int i = 0; i < dividing_cols.size(); ++i) {
		is_dividing_col[dividing_cols[i]] = true;
	}

	//总数据块数
	int total_blocks = num_threads;

	/*
	//统计总在划分线上的数据点数
	int num_dividing_points = 0;
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			if (is_dividing_line[i * m + j]) {
				num_dividing_points++;
			}
		}
	}
	int suppose_num_dividing_points = (sqrt(num_threads) -1) * (n + m) - (sqrt(num_threads) - 1) * (sqrt(num_threads) - 1);
	printf("num_dividing_points = %d, suppose to be = %d\n", num_dividing_points,suppose_num_dividing_points);
	//计算块内的数据的数量
	int num_block_points = 0;
	#pragma omp parallel for schedule(dynamic) reduction(+:num_block_points)
	for (int block_id = 0; block_id < total_blocks; ++block_id){
		int block_row = block_id / t;
		int block_col = block_id % t;
		// 计算块的起始和结束行列索引
		int start_row = block_row * block_height;
		int end_row = (block_row + 1) * block_height;
		if (block_row == t - 1) {
			end_row += remaining_rows;
		}
		int start_col = block_col * block_width;
		int end_col = (block_col + 1) * block_width;
		if (block_col == t - 1) {
			end_col += remaining_cols;
		}
		// 处理块内的数据点
		for(int i=start_row; i<end_row; ++i){
			// 跳过划分线上的行
			if (std::find(dividing_rows.begin(), dividing_rows.end(), i) != dividing_rows.end()) {
				continue;
			}
			for (int j = start_col; j<end_col; ++j){
				// 跳过划分线上的列
				if (is_dividing_line[i * m + j]) {
					continue;
				}
				num_block_points ++;
			}
		}
	}
	printf("num_block_points = %d\n", num_block_points);
	printf("sum of num_block_points + num_dividing_points = %d, suppose to be = %d\n", num_block_points + num_dividing_points, num_elements);
	// exit(0);
	*/
	size_t processed_points_count = 0;
	size_t processed_block_count = 0;
	size_t processed_edge_row_count = 0;
	size_t processed_edge_col_count = 0;
	size_t processed_dot_count = 0;
	std::vector<int> proccessed_points(num_elements, 0);

	auto prepare_comp_end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> prepare_comp_time = prepare_comp_end - prepare_comp_start;
	printf("prepare_comp_time = %f\n", prepare_comp_time.count());
	auto comp_start = std::chrono::high_resolution_clock::now();
	// 并行处理区域，不包括划分线上的数据点
	omp_set_num_threads(num_threads);
	#pragma omp parallel for //reduction(+:processed_block_count)
	for (int block_id = 0; block_id < total_blocks; ++block_id){
		int block_row = block_id / t;
		int block_col = block_id % t;

		// 计算块的起始和结束行列索引
		int start_row = block_row * block_height;
		int end_row = (block_row + 1) * block_height;
		if (block_row == t - 1) {
			end_row += remaining_rows;
		}
		int start_col = block_col * block_width;
		int end_col = (block_col + 1) * block_width;
		if (block_col == t - 1) {
			end_col += remaining_cols;
		}
		
		//print out each thread process which block
		//printf("Thread %d process block %d, start_row = %d, end_row = %d, start_col = %d, end_col = %d\n", omp_get_thread_num(), block_id, start_row, end_row, start_col, end_col);


		//shift eb_quant_index_pos and data_quant_index_pos to the corresponding position
		// eb_quant_index_pos += 2*(start_row * m + start_col);
		// data_quant_index_pos += 2*(start_row * m + start_col);

		// 处理块内的数据点
		for(int i=start_row; i<end_row; ++i){
			// 跳过划分线上的行
			// if (std::find(dividing_rows.begin(), dividing_rows.end(), i) != dividing_rows.end()) {
			// 	continue;
			// }
			if (is_dividing_row[i]) {
				continue;
			}
			// T * cur_U_pos = U_pos + i * r2 + start_col;
			// T * cur_V_pos = V_pos + i * r2 + start_col;

			for (int j = start_col; j<end_col; ++j){
				// 跳过划分线上的列
				// if (is_dividing_line[i * m + j]) {
				// 	continue;
				// }
				if (is_dividing_col[j]) {
					continue;
				}
				
				//processed_block_count ++;
				// 指针计算
				// T * cur_U_pos = U_pos + i * r2 + j;
				// T * cur_V_pos = V_pos + i * r2 + j;
				// 以下使用索引而非指针递增
				size_t position_idx = (i * r2 + j);

				// #pragma omp critical
				// {
				// 	proccessed_points[position_idx] ++;
				// }

				double required_eb;
				required_eb = max_pwr_eb;
				// derive eb given six adjacent triangles
				for (int k = 0; k < 6; k++) {
					bool in_mesh = true;
					for (int p = 0; p < 2; p++) {
						//reserved order!
						//if (!(in_range(i + index_offset[k][p][1], (int)end_row) && in_range(j + index_offset[k][p][0], (int)end_col))) {
						// if (!(in_range(i + index_offset[k][p][1], (int)r1) && in_range(j + index_offset[k][p][0], (int)r2)) || 
						// 	!(in_local_range(i + index_offset[k][p][1], (int)start_row, (int)end_row) && in_local_range(j + index_offset[k][p][0], (int)start_col, (int)end_col))) {
						if (!(in_range(i + index_offset[k][p][1], (int)r1) && in_range(j + index_offset[k][p][0], (int)r2))){
							in_mesh = false;
							break;
						}
						// if (!(in_local_range(i + index_offset[k][p][1], (int)start_row, (int)end_row) && in_local_range(j + index_offset[k][p][0], (int)start_col, (int)end_col))) {
						// 	in_mesh = false;
						// 	break;
						// }
					}
					if (in_mesh) {
						double update_eb;
						(eb_type == "abs") ? update_eb = derive_cp_eb_for_positions_online_abs(decompressed_U[position_idx + offsets[k]], decompressed_U[position_idx + offsets[k + 1]], decompressed_U[position_idx],
							decompressed_V[position_idx + offsets[k]], decompressed_V[position_idx + offsets[k + 1]], decompressed_V[position_idx], inv_C[k]) :
							update_eb = derive_cp_eb_for_positions_online(decompressed_U[position_idx + offsets[k]], decompressed_U[position_idx + offsets[k + 1]], decompressed_U[position_idx],
								decompressed_V[position_idx + offsets[k]], decompressed_V[position_idx + offsets[k + 1]], decompressed_V[position_idx], inv_C[k]);
						required_eb = MINF(required_eb, update_eb);
						// required_eb = MINF(required_eb, derive_cp_eb_for_positions_online(decompressed_U[position_idx + offsets[k]], decompressed_U[position_idx + offsets[k + 1]], decompressed_U[position_idx], 
						// decompressed_V[position_idx + offsets[k]], decompressed_V[position_idx + offsets[k + 1]],decompressed_V[position_idx],inv_C[k]));
					}
				}
				// if (position_idx == 1863*3600+1801){
				// 	printf("now i = %d, j = %d\n", i, j);
				// 	printf("required_eb = %f\n", required_eb);
				// }

				if(required_eb >0){
					bool unpred_flag = false;
					T decompressed[2];
					int temp_eb_quant_index[2];
					// compress U and V
					for (int k = 0; k < 2; k++) {
						// T * cur_data_pos = (k == 0) ? cur_U_pos : cur_V_pos;
						T * cur_data_field = (k == 0) ? decompressed_U : decompressed_V;
						// T cur_data = *cur_data_pos;
						double abs_eb;
						(eb_type == "abs") ? abs_eb = required_eb : abs_eb = fabs(cur_data_field[position_idx]) * required_eb;
						//double abs_eb = fabs(cur_data_field[position_idx]) * required_eb;

						// eb_quant_index_pos[2*(i*r2 + j) + k] = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
						eb_quant_index[2*position_idx + k] = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
						temp_eb_quant_index[k] = eb_quant_index[2*position_idx + k];
						if (eb_quant_index[2*position_idx + k] > 0) {
							// get adjacent data and perform Lorenzo
							// T d0 = (i != start_row && j != start_col) ? cur_data_field[position_idx -1 - r2] : 0; //问题：这里偏移量对不对？
							// T d1 = (i != start_row) ? cur_data_field[position_idx-r2] : 0;
							// T d2 = (j != start_col) ? cur_data_field[position_idx-1] : 0;
							// T d0 = (i && j ) ? cur_data_field[position_idx -1 - r2] : 0; //问题：这里偏移量对不对？
							T d0 = ((i != 0 && j != 0) && (i - start_row > 1 && j - start_col > 1)) ? cur_data_field[position_idx -1 - r2] : 0; //问题：这里偏移量对不对？
							// T d1 = (i) ? cur_data_field[position_idx-r2] : 0;
							T d1 = (i != 0 && i - start_row > 1) ? cur_data_field[position_idx-r2] : 0;
							// T d2 = (j) ? cur_data_field[position_idx-1] : 0;
							T d2 = (j != 0 && j - start_col > 1) ? cur_data_field[position_idx-1] : 0;
							T pred = d1 + d2 - d0;
							double diff = cur_data_field[position_idx] - pred;
							double quant_diff = fabs(diff) / abs_eb + 1;
							if (quant_diff < capacity) {
								quant_diff = (diff > 0) ? quant_diff : -quant_diff;
								int quant_index = (int)(quant_diff / 2) + intv_radius;
								// data_quant_index_pos[2*(i*r2 + j) + k] = quant_index;
								data_quant_index[2*position_idx + k] = quant_index;
								decompressed[k] = pred + 2 * (quant_index - intv_radius) * abs_eb;
								// check original data
								// if (position_idx == 1863*3600+1801){
								// 	printf("=======\n");
								// 	printf("now i = %d,start_row = %d, j = %d, start_col = %d\n", i, start_row, j, start_col);
								// 	printf("i = %d, j = %d, k = %d, d0 = %f, d1 = %f, d2 = %f, pred = %f, eb = %f, data_quant_index = %d, decompressed = %f\n", i, j, k, d0, d1, d2, pred, abs_eb, quant_index, decompressed[k]);
								// 	printf("abs_eb = %f, diff = %f, quant_diff = %f\n", abs_eb, diff, quant_diff);
								// 	printf("decompressed[k] = %f, cur_data_field[position_idx] = %f\n", decompressed[k], cur_data_field[position_idx]);
								// }

								if (fabs(decompressed[k] - cur_data_field[position_idx]) >= abs_eb) {
									unpred_flag = true;
									break;
								}
							}
							else {

								unpred_flag = true;
								break;
							}
						
						}
						else {

							unpred_flag = true;
							}
					}
					
				
					if(unpred_flag){
						// // recover quant index
						// *(eb_quant_index_pos ++) = 0;
						// *(eb_quant_index_pos ++) = 0;
						// *(data_quant_index_pos ++) = intv_radius;
						// *(data_quant_index_pos ++) = intv_radius;
						// unpred_data_thread[omp_get_thread_num()].push_back(*cur_U_pos);
						// unpred_data_thread[omp_get_thread_num()].push_back(*cur_V_pos);
						// printf("pos_idx = %ld\n", position_idx);
						//pos_idx = 101372
						eb_quant_index[2*position_idx] = 0;
						eb_quant_index[2*position_idx + 1] = 0;
						data_quant_index[2*position_idx] = intv_radius;
						data_quant_index[2*position_idx + 1] = intv_radius;
						unpred_data_thread[omp_get_thread_num()].push_back(decompressed_U[position_idx]);
						unpred_data_thread[omp_get_thread_num()].push_back(decompressed_V[position_idx]);
						// if (position_idx == 1863*3600+1801){
						// 	printf("id 1863*3600+1801 is unpredict\n");
						// }
					}
					else{
						// eb_quant_index_pos += 2;
						// data_quant_index_pos += 2;
						// eb_quant_index[2*position_idx] = temp_eb_quant_index[0];
						// eb_quant_index[2*position_idx + 1] = temp_eb_quant_index[1];
						// assign decompressed data
						// *cur_U_pos = decompressed[0];
						// *cur_V_pos = decompressed[1];
						decompressed_U[position_idx] = decompressed[0];
						decompressed_V[position_idx] = decompressed[1];
						// if (position_idx == 1863*3600+1801){
						// 	printf("id 1863*3600+1801 use predict data\n");
						// }
					}
				}
				else{
					// req_eb <= 0 直接lossless record
					// record as unpredictable data
					// *(eb_quant_index_pos ++) = 0;
					// *(eb_quant_index_pos ++) = 0;
					// *(data_quant_index_pos ++) = intv_radius;
					// *(data_quant_index_pos ++) = intv_radius;
					eb_quant_index[2*position_idx] = 0;
					eb_quant_index[2*position_idx + 1] = 0;
					data_quant_index[2*position_idx] = intv_radius;
					data_quant_index[2*position_idx + 1] = intv_radius;
					unpred_data_thread[omp_get_thread_num()].push_back(decompressed_U[position_idx]);
					unpred_data_thread[omp_get_thread_num()].push_back(decompressed_V[position_idx]);
					// if (position_idx == 1863*3600+1801){
					// 	printf("id 1863*3600+1801 is unpredict and req_eb <=0\n");
					// 	printf("req_eb = %f,decompressed_U[position_idx] = %f, decompressed_V[position_idx] = %f,U[position_idx] = %f, V[position_idx] = %f\n", required_eb, decompressed_U[position_idx], decompressed_V[position_idx], U[position_idx], V[position_idx]);
					// }
				}
				// cur_U_pos ++, cur_V_pos ++;
			}
			// U_pos += r2 - (end_col - start_col);//问题：这里偏移量对不对？
			// V_pos += r2 - (end_col - start_col);
		}
	}

	// print all unpred_size for each thread
	// for (auto s :unpred_data_thread){
	// 	printf("vector size = %ld\n", s.size());
	// }
	// merge unpred_data_thread
	// for (auto s : unpred_data_thread) {
	// 	unpred_data.insert(unpred_data.end(), s.begin(), s.end());
	// }

	/*
	//优化：目前数据总量对不上。。。。
	//处理block内的数据
	omp_set_num_threads(num_threads);
	#pragma omp parallel for reduction(+:processed_block_count)
	for (int blockID = 0; blockID < num_threads; blockID++){
		auto [start_row, start_col, block_height, block_width] = calculate_block(n, m, num_threads, blockID);
		printf("threadID = %d, start_row = %d, start_col = %d, block_height = %d, block_width = %d\n", blockID, start_row, start_col, block_height, block_width);
		// process inner block
		for (int i = start_row + 1; i < start_row + block_height -1; ++i){
			for (int j = start_col+1; j < start_col + block_width - 1; ++j){
				processed_block_count ++;
			}
		}
	}
	printf("processed_block_count = %ld\n", processed_block_count);
	//处理边界 行
	// omp_set_num_threads((t+1)*t);
	#pragma omp parallel for reduction(+:processed_edge_row_count)
	for (int lineID = 0; lineID < num_threads; lineID++){
		auto [start_row, start_col, block_height, block_width] = calculate_block(n, m, num_threads, lineID);
		if (start_row == 0){
			// 最上层，顺带处理了
			for (int j = start_col + 1; j < start_col + block_width - 1; ++j) {
            	processed_edge_row_count++;  // 处理第一行
			}
		}
		// 处理最后一行
		for (int j = start_col + 1; j < start_col + block_width - 1; ++j) {
			processed_edge_row_count++;  // 处理最后一行
		}
		printf("threadID: %d, block_width = %d,\n", lineID, block_width);
	}
	printf("processed_row_count = %ld\n", processed_edge_row_count);
	//处理边界列
	#pragma omp parallel for reduction(+:processed_edge_col_count)
	for (int lineID = 0; lineID <num_threads; lineID++){
		auto [start_row, start_col, block_height, block_width] = calculate_block(n, m, num_threads, lineID);
		if(start_col == 0){
			for (int i = start_row + 1; i < start_row + block_height - 1; ++i) {
				processed_edge_col_count++;
			}
		}
		// 处理最后一列
		for (int i = start_row + 1; i < start_row + block_height - 1; ++i) {
			processed_edge_col_count++;
		}
	}
	printf("processed_col_count = %ld\n", processed_edge_col_count);
	exit(0);
	*/

	//printf("dividing_rows.size() = %ld, dividing_cols.size() = %ld\n", dividing_rows.size(), dividing_cols.size());
	//printf("dividing_rows[0], dividing_rows[1] = %d, %d\n", dividing_rows[0], dividing_rows[1]);
	//优化处理线和点
	//先处理横着的线（行）

	std::vector<std::vector<T>> unpred_data_row((t-1)*t);
	omp_set_num_threads((t-1)*t);
	#pragma omp parallel for collapse(2) //reduction(+:processed_edge_row_count)
	for(int i : dividing_rows){
		for (int j = -1; j < (int)dividing_cols.size();j++){
			//printf("j = %d, dividing_col[j] = %d\n", j, dividing_cols[j]);
			int thread_id = omp_get_thread_num();

			int start_col = (j == -1) ? 0 : dividing_cols[j];
			int end_col = (j == dividing_cols.size() - 1) ? m : dividing_cols[j+1];
			//printf("threadID = %d, row = %d, start_col = %d, end_col = %d\n", thread_id, i, start_col, end_col);
			//处理线上的数据
			for (int c = start_col; c < end_col; ++c){
				if (std::find(dividing_cols.begin(), dividing_cols.end(), c) != dividing_cols.end()) {
					//c is a dividing point
					continue;
				}
				//processed_edge_row_count ++;
				size_t position_idx = (i * r2 + c);
				// #pragma omp critical
				// {
				// 	proccessed_points[position_idx] ++;
				// }
				double required_eb;
				required_eb = max_pwr_eb;
				// derive eb given six adjacent triangles
				for (int k = 0; k < 6; k++) {
					bool in_mesh = true;
					for (int p = 0; p < 2; p++) {
						//reserved order!
						//if (!(in_range(i + index_offset[k][p][1], (int)end_row) && in_range(j + index_offset[k][p][0], (int)end_col))) {
						// if (!(in_range(i + index_offset[k][p][1], (int)r1) && in_range(j + index_offset[k][p][0], (int)r2)) ||
						// 	(c - start_col == 1) || (c - start_col == end_col - start_col - 1)){
						if (!(in_range(i + index_offset[k][p][1], (int)r1) && in_range(c + index_offset[k][p][0], (int)r2))){
							in_mesh = false;
							break;
						}
						// if (!(in_local_range(j + index_offset[k][p][0], (int)start_col, (int)end_col))) {
						// 	in_mesh = false;
						// 	break;
						// }
					}
					if (in_mesh) {
						double update_eb;
						(eb_type == "abs") ? update_eb = derive_cp_eb_for_positions_online_abs(decompressed_U[position_idx + offsets[k]], decompressed_U[position_idx + offsets[k + 1]], decompressed_U[position_idx],
							decompressed_V[position_idx + offsets[k]], decompressed_V[position_idx + offsets[k + 1]], decompressed_V[position_idx], inv_C[k]) :
							update_eb = derive_cp_eb_for_positions_online(decompressed_U[position_idx + offsets[k]], decompressed_U[position_idx + offsets[k + 1]], decompressed_U[position_idx],
								decompressed_V[position_idx + offsets[k]], decompressed_V[position_idx + offsets[k + 1]], decompressed_V[position_idx], inv_C[k]);
						required_eb = MINF(required_eb, update_eb);
						// required_eb = MINF(required_eb, derive_cp_eb_for_positions_online(decompressed_U[position_idx + offsets[k]], decompressed_U[position_idx + offsets[k + 1]], decompressed_U[position_idx], 
						// decompressed_V[position_idx + offsets[k]], decompressed_V[position_idx + offsets[k + 1]],decompressed_V[position_idx],inv_C[k]));
					}
				}

				if(required_eb >0){
					bool unpred_flag = false;
					T decompressed[2];
					int temp_eb_quant_index[2];
					// compress U and V
					for (int k = 0; k < 2; k++) {
						// T * cur_data_pos = (k == 0) ? cur_U_pos : cur_V_pos;
						T * cur_data_field = (k == 0) ? decompressed_U : decompressed_V;
						// T cur_data = *cur_data_pos;
						double abs_eb;
						(eb_type == "abs") ? abs_eb = required_eb : abs_eb = fabs(cur_data_field[position_idx]) * required_eb;
						//double abs_eb = fabs(cur_data_field[position_idx]) * required_eb;

						// eb_quant_index_pos[2*(i*r2 + j) + k] = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
						eb_quant_index[2*position_idx + k] = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
						temp_eb_quant_index[k] = eb_quant_index[2*position_idx + k];
						if (eb_quant_index[2*position_idx + k] > 0) {
							// get adjacent data and perform Lorenzo
							//use 1d Lorenzo
							//T d0 = (c) ? cur_data_field[position_idx -1] : 0;
							T d0 = (c && (c - start_col > 1)) ? cur_data_field[position_idx -1] : 0; 
							T pred = d0;
							double diff = cur_data_field[position_idx] - pred;
							double quant_diff = fabs(diff) / abs_eb + 1;
							if (quant_diff < capacity) {
								quant_diff = (diff > 0) ? quant_diff : -quant_diff;
								int quant_index = (int)(quant_diff / 2) + intv_radius;
								// data_quant_index_pos[2*(i*r2 + j) + k] = quant_index;
								data_quant_index[2*position_idx + k] = quant_index;
								decompressed[k] = pred + 2 * (quant_index - intv_radius) * abs_eb;
								// check original data
								// if (position_idx == 2506800){
								// 	printf("=======\n");
								// 	printf("i = %d, j = %d, k = %d, d0 = %f, d1 = %f, d2 = %f, pred = %f, eb = %f, data_quant_index = %d, decompressed = %f\n", i, j, k, d0, d1, d2, pred, abs_eb, quant_index, decompressed[k]);
								// 	printf("decompressed[k] = %f, cur_data_field[position_idx] = %f\n", decompressed[k], cur_data_field[position_idx]);
								// }
								if (fabs(decompressed[k] - cur_data_field[position_idx]) >= abs_eb) {
									unpred_flag = true;
									break;
								}
							}
							else {
								unpred_flag = true;
								break;
							}
						}
						else {
							unpred_flag = true;
							}
					}
					
				
					if(unpred_flag){
						eb_quant_index[2*position_idx] = 0;
						eb_quant_index[2*position_idx + 1] = 0;
						data_quant_index[2*position_idx] = intv_radius;
						data_quant_index[2*position_idx + 1] = intv_radius;
						unpred_data_row[thread_id].push_back(decompressed_U[position_idx]);
						unpred_data_row[thread_id].push_back(decompressed_V[position_idx]);
					}
					else{
						decompressed_U[position_idx] = decompressed[0];
						decompressed_V[position_idx] = decompressed[1];
					}
				}
				else{
					eb_quant_index[2*position_idx] = 0;
					eb_quant_index[2*position_idx + 1] = 0;
					data_quant_index[2*position_idx] = intv_radius;
					data_quant_index[2*position_idx + 1] = intv_radius;
					unpred_data_row[thread_id].push_back(decompressed_U[position_idx]);
					unpred_data_row[thread_id].push_back(decompressed_V[position_idx]);
				}
			}
		}
	}

	//处理竖着的线（列）
	std::vector<std::vector<T>> unpred_data_col((t-1)*t);
	omp_set_num_threads((t-1)*t);
	#pragma omp parallel for collapse(2) //reduction(+:processed_edge_col_count)
	for(int j : dividing_cols){
		for (int i = -1; i < (int)dividing_rows.size();i++){
			//printf("j = %d, dividing_col[j] = %d\n", j, dividing_cols[j]);
			int thread_id = omp_get_thread_num();
			int start_row = (i == -1) ? 0 : dividing_rows[i];
			int end_row = (i == dividing_rows.size() - 1) ? n : dividing_rows[i+1];
			//printf("threadID = %d, col = %d, start_row = %d, end_row = %d\n", thread_id, j, start_row, end_row);
			//处理线上的数据
			for (int r = start_row; r < end_row; ++r){
				if (std::find(dividing_rows.begin(), dividing_rows.end(), r) != dividing_rows.end()) {
					//k is a dividing point
					continue;
				}
				//processed_edge_col_count ++;
				size_t position_idx = (r * r2 + j);
				// #pragma omp critical
				// {
				// 	proccessed_points[position_idx] ++;
				// }

				// if (position_idx == 1864*3600+1800){
				// 	printf("now i = %d, j = %d is on vetical line\n", r, j);
				// }
				double required_eb;
				required_eb = max_pwr_eb;
				// derive eb given six adjacent triangles
				for (int k = 0; k < 6; k++) {
					bool in_mesh = true;
					for (int p = 0; p < 2; p++) {
						//reserved order!
						//if (!(in_range(i + index_offset[k][p][1], (int)end_row) && in_range(j + index_offset[k][p][0], (int)end_col))) {
						// if (!(in_range(i + index_offset[k][p][1], (int)r1) && in_range(j + index_offset[k][p][0], (int)r2)) ||
						// 	(r - start_row == 1) || (r - start_row == end_row - start_row - 1)){
						if (!(in_range(r + index_offset[k][p][1], (int)r1) && in_range(j + index_offset[k][p][0], (int)r2))){
							in_mesh = false;
							// if (position_idx == 1864*3600+1800){
							// 	printf("out of mesh!!, r = %d, index_offset[k][p][1] = %d, r1 = %d, j = %d, index_offset[k][p][0] = %d, r2 = %d\n", r, index_offset[k][p][1], r1, j, index_offset[k][p][0], r2);
							// }
							break;
						}
						// if (!(in_local_range(i + index_offset[k][p][1], (int)start_row, (int)end_row))) {
						// 	in_mesh = false;
						// 	break;
						// }
					}
					if (in_mesh) {
						// if (position_idx == 1864*3600+1800){
						// 	printf("in_mesh!!\n");
						// }
						double update_eb;
						(eb_type == "abs") ? update_eb = derive_cp_eb_for_positions_online_abs(decompressed_U[position_idx + offsets[k]], decompressed_U[position_idx + offsets[k + 1]], decompressed_U[position_idx],
							decompressed_V[position_idx + offsets[k]], decompressed_V[position_idx + offsets[k + 1]], decompressed_V[position_idx], inv_C[k]) :
							update_eb = derive_cp_eb_for_positions_online(decompressed_U[position_idx + offsets[k]], decompressed_U[position_idx + offsets[k + 1]], decompressed_U[position_idx],
								decompressed_V[position_idx + offsets[k]], decompressed_V[position_idx + offsets[k + 1]], decompressed_V[position_idx], inv_C[k]);
						required_eb = MINF(required_eb, update_eb);
						// required_eb = MINF(required_eb, derive_cp_eb_for_positions_online(decompressed_U[position_idx + offsets[k]], decompressed_U[position_idx + offsets[k + 1]], decompressed_U[position_idx], 
						// decompressed_V[position_idx + offsets[k]], decompressed_V[position_idx + offsets[k + 1]],decompressed_V[position_idx],inv_C[k]));
					}
				}
				// if (position_idx == 1864*3600+1800){
				// 	printf("now i = %d, j = %d, required_eb = %f\n", r, j, required_eb);
				// }

				if(required_eb >0){
					bool unpred_flag = false;
					T decompressed[2];
					int temp_eb_quant_index[2];
					// compress U and V
					for (int k = 0; k < 2; k++) {
						// T * cur_data_pos = (k == 0) ? cur_U_pos : cur_V_pos;
						T * cur_data_field = (k == 0) ? decompressed_U : decompressed_V;
						// T cur_data = *cur_data_pos;
						double abs_eb;
						(eb_type == "abs") ? abs_eb = required_eb : abs_eb = fabs(cur_data_field[position_idx]) * required_eb;
						//double abs_eb = fabs(cur_data_field[position_idx]) * required_eb;

						// eb_quant_index_pos[2*(i*r2 + j) + k] = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
						eb_quant_index[2*position_idx + k] = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
						temp_eb_quant_index[k] = eb_quant_index[2*position_idx + k];
						if (eb_quant_index[2*position_idx + k] > 0) {
							// get adjacent data and perform Lorenzo
							//use 1d Lorenzo
							//T d0 = (r) ? cur_data_field[position_idx - r2] : 0;
							T d0 = (r && (r - start_row > 1)) ? cur_data_field[position_idx - r2] : 0; 
							T pred = d0;
							double diff = cur_data_field[position_idx] - pred;
							double quant_diff = fabs(diff) / abs_eb + 1;
							if (quant_diff < capacity) {
								quant_diff = (diff > 0) ? quant_diff : -quant_diff;
								int quant_index = (int)(quant_diff / 2) + intv_radius;
								// data_quant_index_pos[2*(i*r2 + j) + k] = quant_index;
								data_quant_index[2*position_idx + k] = quant_index;
								decompressed[k] = pred + 2 * (quant_index - intv_radius) * abs_eb;
								// check original data
								// if (position_idx == 2506800){
								// 	printf("=======\n");
								// 	printf("i = %d, j = %d, k = %d, d0 = %f, d1 = %f, d2 = %f, pred = %f, eb = %f, data_quant_index = %d, decompressed = %f\n", i, j, k, d0, d1, d2, pred, abs_eb, quant_index, decompressed[k]);
								// 	printf("decompressed[k] = %f, cur_data_field[position_idx] = %f\n", decompressed[k], cur_data_field[position_idx]);
								// }

								if (fabs(decompressed[k] - cur_data_field[position_idx]) >= abs_eb) {
									unpred_flag = true;
									break;
								}
							}
							else {

								unpred_flag = true;
								break;
							}
						
						}
						else {
							unpred_flag = true;
							}
					}
					
				
					if(unpred_flag){
						eb_quant_index[2*position_idx] = 0;
						eb_quant_index[2*position_idx + 1] = 0;
						data_quant_index[2*position_idx] = intv_radius;
						data_quant_index[2*position_idx + 1] = intv_radius;
						unpred_data_col[thread_id].push_back(decompressed_U[position_idx]);
						unpred_data_col[thread_id].push_back(decompressed_V[position_idx]);
						
					}
					else{
						decompressed_U[position_idx] = decompressed[0];
						decompressed_V[position_idx] = decompressed[1];
					}
				}
				else{
					eb_quant_index[2*position_idx] = 0;
					eb_quant_index[2*position_idx + 1] = 0;
					data_quant_index[2*position_idx] = intv_radius;
					data_quant_index[2*position_idx + 1] = intv_radius;
					unpred_data_col[thread_id].push_back(decompressed_U[position_idx]);
					unpred_data_col[thread_id].push_back(decompressed_V[position_idx]);
				}
			}
		}
	}
	
	std::vector<T> unpred_dot;
	//处理点,串行
	for(int i : dividing_rows){
		for(int j : dividing_cols){
				processed_dot_count ++;
				size_t position_idx = (i * r2 + j);
				proccessed_points[position_idx] ++;

				//lossless record
				eb_quant_index[2*position_idx] = 0;
				eb_quant_index[2*position_idx + 1] = 0;
				data_quant_index[2*position_idx] = intv_radius;
				data_quant_index[2*position_idx + 1] = intv_radius;
				unpred_dot.push_back(decompressed_U[position_idx]);
				unpred_dot.push_back(decompressed_V[position_idx]);
		}
	}

	auto comp_end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> comp_time = comp_end - comp_start;
	printf("derivation time = %f\n", comp_time.count());
	// printf("processed_block_count = %ld, processed_edge_row_count = %ld, processed_edge_col_count = %ld, processed_dot_count%ld\n", processed_block_count, processed_edge_row_count, processed_edge_col_count,processed_dot_count);
	// printf("number of processed points = %ld\n", processed_block_count + processed_edge_row_count + processed_edge_col_count + processed_dot_count);
	// printf("total number of points = %ld\n", num_elements);
	// //check if all points are processed and only processed once
	// for (int i = 0; i < num_elements; i++){
	// 	if (proccessed_points[i] != 1){
	// 		printf("point %d is processed %d times\n", i, proccessed_points[i]);
	// 		exit(0);
	// 	}
	// }
	// exit(0);

/*
	//目前已经处理完了每个块的数据，现在要特殊处理划分线上的数据
	//串行处理划分线上的数据
	unpred_vec<T> unpred_data_dividing;
	for(int i = 0; i < r1; i++){
		for (int j = 0 ; j < r2; j++){
			if (is_dividing_line[i * m + j]){
				// T * cur_U_pos = U_pos + i * r2 + j;
				// T * cur_V_pos = V_pos + i * r2 + j;
				size_t position_idx = (i * r2 + j);

				double required_eb;
				required_eb = max_pwr_eb;
				// derive eb given six adjacent triangles
				for (int k = 0; k < 6; k++) {
					bool in_mesh = true;
					for (int p = 0; p < 2; p++) {
						//reserved order!
						if (!(in_range(i + index_offset[k][p][1], (int)r1) && in_range(j + index_offset[k][p][0], (int)r2))) {
							in_mesh = false;
							break;
						}
					}
					if (in_mesh) {
						double updated_eb;
						(eb_type == "abs") ? updated_eb = derive_cp_eb_for_positions_online_abs(decompressed_U[position_idx + offsets[k]], decompressed_U[position_idx + offsets[k + 1]], decompressed_U[position_idx],
							decompressed_V[position_idx + offsets[k]], decompressed_V[position_idx + offsets[k + 1]], decompressed_V[position_idx], inv_C[k]) :
							updated_eb = derive_cp_eb_for_positions_online(decompressed_U[position_idx + offsets[k]], decompressed_U[position_idx + offsets[k + 1]], decompressed_U[position_idx],
							decompressed_V[position_idx + offsets[k]], decompressed_V[position_idx + offsets[k + 1]], decompressed_V[position_idx], inv_C[k]);
						required_eb = MINF(required_eb, updated_eb);

						// required_eb = MINF(required_eb, derive_cp_eb_for_positions_online(decompressed_U[position_idx + offsets[k]], decompressed_U[position_idx + offsets[k + 1]], decompressed_U[position_idx],
						// 	decompressed_V[position_idx + offsets[k]], decompressed_V[position_idx + offsets[k + 1]], decompressed_V[position_idx], inv_C[k]));
					}
				}

				if(required_eb > 0){
					bool unpred_flag = false;
					T decompressed[2];
					T temp_eb_quant_index[2];
					// compress U and V
					for (int k = 0; k < 2; k++) {
						T * cur_data_field = (k == 0) ? decompressed_U : decompressed_V;
						// T cur_data = *cur_data_pos;
						//double abs_eb = fabs(cur_data_field[position_idx]) * required_eb;
						double abs_eb;
						(eb_type == "abs") ? abs_eb = required_eb : abs_eb = fabs(cur_data_field[position_idx]) * required_eb;
						// eb_quant_index_pos[2*(i*r2 + j) + k] = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
						eb_quant_index[2*position_idx + k] = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
						// if (eb_quant_index_pos[2*(i*r2 + j) + k] > 0) {
						temp_eb_quant_index[k] = eb_quant_index[2*position_idx + k];
						if (eb_quant_index[2*position_idx + k] > 0) {
							// get adjacent data and perform Lorenzo
							T d0 = (i != 0 && j != 0) ? cur_data_field[position_idx -1 - r2] : 0; //问题：这里偏移量对不对？
							T d1 = (i != 0) ? cur_data_field[position_idx-r2] : 0;
							T d2 = (j != 0) ? cur_data_field[position_idx-1] : 0;
							T pred = d1 + d2 - d0;
							double diff = cur_data_field[position_idx] - pred;
							double quant_diff = fabs(diff) / abs_eb + 1;
							if (quant_diff < capacity) {
								quant_diff = (diff > 0) ? quant_diff : -quant_diff;
								int quant_index = (int)(quant_diff / 2) + intv_radius;
								// data_quant_index_pos[2*(i*r2 + j) + k] = quant_index;
								data_quant_index[2*position_idx + k] = quant_index;
								decompressed[k] = pred + 2 * (quant_index - intv_radius) * abs_eb;
								// check original data
								if (fabs(decompressed[k] - cur_data_field[position_idx]) >= abs_eb) { //cuode
									unpred_flag = true;
									break;
								}
							}
							else {
								unpred_flag = true;
								break;
							}
						
						}
						else unpred_flag = true;
					}
				
					if(unpred_flag){
						// recover quant index
						eb_quant_index[2*position_idx] = 0;
						eb_quant_index[2*position_idx + 1] = 0;
						data_quant_index[2*position_idx] = intv_radius;
						data_quant_index[2*position_idx + 1] = intv_radius;
						unpred_data_dividing.push_back(decompressed_U[position_idx]);
						unpred_data_dividing.push_back(decompressed_V[position_idx]);

					}
					else{
						// eb_quant_index[2*position_idx] = temp_eb_quant_index[0];
						// eb_quant_index[2*position_idx + 1] = temp_eb_quant_index[1];
						decompressed_U[position_idx] = decompressed[0];
						decompressed_V[position_idx] = decompressed[1];
					}
				}
				else{
					// record as unpredictable data
					// *(eb_quant_index_pos ++) = 0;
					// *(eb_quant_index_pos ++) = 0;
					// *(data_quant_index_pos ++) = intv_radius;
					// *(data_quant_index_pos ++) = intv_radius;
					eb_quant_index[2*position_idx] = 0;
					eb_quant_index[2*position_idx + 1] = 0;
					data_quant_index[2*position_idx] = intv_radius;
					data_quant_index[2*position_idx + 1] = intv_radius;
					unpred_data_dividing.push_back(decompressed_U[position_idx]);
					unpred_data_dividing.push_back(decompressed_V[position_idx]);
				}
			}
		}
	}
	
*/


	
	// free(decompressed_U);
	// free(decompressed_V);
	//set decompressed_U_ptr and decompressed_V_ptr
	decompressed_U_ptr = decompressed_U;
	decompressed_V_ptr = decompressed_V;

	auto write_variables_start = std::chrono::high_resolution_clock::now();

	unsigned char * compressed = (unsigned char *) malloc(2*num_elements*sizeof(T));
	unsigned char * compressed_pos = compressed;
	//write size of index_need_to_fix
	write_variable_to_dst(compressed_pos, index_need_to_fix.size());
	// if size of index_need_to_fix is not 0, write bitmap
	if(index_need_to_fix.size() != 0){
		convertIntArray2ByteArray_fast_1b_to_result_sz(bitmap, num_elements, compressed_pos);
		//write index_need_to_fix's data(U and V)
		for(auto it = index_need_to_fix.begin(); it != index_need_to_fix.end(); ++it){
			write_variable_to_dst(compressed_pos, U[*it]);
		}
		for(auto it = index_need_to_fix.begin(); it != index_need_to_fix.end(); ++it){
			write_variable_to_dst(compressed_pos, V[*it]);
		}
	}
	
	//write number of threads
	write_variable_to_dst(compressed_pos, num_threads);
	printf("num_threads = %d,pos = %ld\n", num_threads, compressed_pos - compressed);

	//写每个block的unpred_data size
	for (auto threadID = 0; threadID < num_threads; threadID++){
		write_variable_to_dst(compressed_pos, unpred_data_thread[threadID].size());
		//printf("thread %d, unpred_data size = %ld,maxvalue = %f\n", threadID, unpred_data_thread[threadID].size(), *std::max_element(unpred_data_thread[threadID].begin(), unpred_data_thread[threadID].end()));
	}

	//写每个block的unpred_data
	for (auto threadID = 0; threadID < num_threads; threadID++){
		write_array_to_dst(compressed_pos, (T *)&unpred_data_thread[threadID][0], unpred_data_thread[threadID].size());
	}

	//写每个row的unpred_data size
	for (auto threadID = 0; threadID < (t-1)*t; threadID++){
		write_variable_to_dst(compressed_pos, unpred_data_row[threadID].size());
		//printf("thread %d, unpred_row size = %ld,maxvalue = %f\n", threadID, unpred_data_row[threadID].size(), (unpred_data_row[threadID].size() == 0)? 0:*std::max_element(unpred_data_row[threadID].begin(), unpred_data_row[threadID].end()) );
	}
	//写每个row的unpred_data
	for (auto threadID = 0; threadID < (t-1)*t; threadID++){
		write_array_to_dst(compressed_pos, (T *)&unpred_data_row[threadID][0], unpred_data_row[threadID].size());
	}

	//写每个col的unpred_data size
	for (auto threadID = 0; threadID < (t-1)*t; threadID++){
		write_variable_to_dst(compressed_pos, unpred_data_col[threadID].size());
	}

	//写每个col的unpred_data
	for (auto threadID = 0; threadID < (t-1)*t; threadID++){
		write_array_to_dst(compressed_pos, (T *)&unpred_data_col[threadID][0], unpred_data_col[threadID].size());
	}

	//写dot的unpred_data size
	write_variable_to_dst(compressed_pos,unpred_dot.size());
	//写dot的unpred_data
	write_array_to_dst(compressed_pos, (T *)&unpred_dot[0], unpred_dot.size());

	auto write_variables_end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> write_variable_time = write_variables_end - write_variables_start;
	printf("write variables time: %f\n",write_variable_time.count());
	//printf("pos after write all upredict= %ld\n", compressed_pos - compressed);
	write_variable_to_dst(compressed_pos, base);
	//printf("base = %d,pos = %ld\n", base, compressed_pos - compressed);
	write_variable_to_dst(compressed_pos, threshold);
	//printf("threshold = %d ,pos = %ld%f\n", threshold, compressed_pos - compressed);
	write_variable_to_dst(compressed_pos, intv_radius);
	//printf("intv_radius = %d,pos = %ld\n", intv_radius, compressed_pos - compressed);
	//unpredictable data size is sum of all thread's unpredictable data size

	/*
	// //write dividing line data
	// write_variable_to_dst(compressed_pos, (size_t)unpred_data_dividing.size());
	// //printf("comp unpred_data_dividing_count = %ld\n", unpred_data_dividing.size());
	// write_array_to_dst(compressed_pos, (T *)&unpred_data_dividing[0], unpred_data_dividing.size());
	//printf("comp pos after all upredict_dividing = %ld\n", compressed_pos - compressed);
	*/
	

	// size_t * freq = (size_t *) calloc(num_threads * 4 * 1024, sizeof(size_t));
	// size_t * freq = (size_t *) malloc(num_threads* 4*1024*sizeof(size_t));
	// memset(freq, 0, num_threads* 4*1024*sizeof(size_t));

	//std::cout<<"eb max = "<<*std::max_element(eb_quant_index, eb_quant_index + 2*num_elements)<<std::endl;
	//std::cout<<"eb min = "<<*std::min_element(eb_quant_index, eb_quant_index + 2*num_elements)<<std::endl;

	/*
	// serial huffman****************************************************
	auto serial_eb_quant_huffman_start = std::chrono::high_resolution_clock::now();
	omp_Huffman_encode_tree_and_data(2*1024, eb_quant_index, 2*num_elements, compressed_pos,freq,num_threads);
	auto serial_eb_quant_huffman_end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> serial_eb_quant_huffman_time = serial_eb_quant_huffman_end - serial_eb_quant_huffman_start;
	std::cout<<"serial_eb_quant_huffman_time = "<<serial_eb_quant_huffman_time.count()<<std::endl;
	//Huffman_encode_tree_and_data(2*1024, eb_quant_index, 2*num_elements, compressed_pos);
	// serial huffman****************************************************
	*/

	//naive parallel huffman****************************************************
	auto parallel_eb_quant_huffman_start = std::chrono::high_resolution_clock::now();
	std::vector<std::vector<unsigned char>> compressed_buffers(num_threads);
	std::vector<size_t> compressed_sizes(num_threads);
	//resize
	for (int i = 0; i < num_threads; i++){
		compressed_buffers[i].resize(2*num_elements / num_threads);
	}
	#pragma omp parallel for num_threads(num_threads)
	for (int i = 0; i < num_threads; i++){
		size_t start_pos = i * num_elements / num_threads;
		size_t end_pos = (i == num_threads - 1) ? num_elements : (i + 1) * num_elements / num_threads;
		unsigned char *local_compressed_pos = compressed_buffers[i].data();
		size_t local_compressed_size = 0;
		Huffman_encode_tree_and_data(2*1024,eb_quant_index + 2*start_pos, 2*(end_pos - start_pos), local_compressed_pos,local_compressed_size);
		compressed_sizes[i] = local_compressed_size;
	}

	// write compressed_sizes first
	for (int i = 0; i < num_threads; i++){
		write_variable_to_dst(compressed_pos, compressed_sizes[i]);
		//printf("comp thread %d, compressed_size = %ld\n", i, compressed_sizes[i]);
	}
	//merge compressed_buffers write to compressed
	for (int i = 0; i < num_threads; i++){
		memcpy(compressed_pos, compressed_buffers[i].data(), compressed_sizes[i]);
		compressed_pos += compressed_sizes[i];
	}
	auto parallel_eb_quant_huffman_end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> parallel_eb_quant_huffman_time = parallel_eb_quant_huffman_end - parallel_eb_quant_huffman_start;
	std::cout<<"parallel_eb_quant_huffman_time = "<<parallel_eb_quant_huffman_time.count()<<std::endl;
	//naive parallel huffman eb quant****************************************************
	
	//writefile("comp_eb_quant_index.txt",eb_quant_index, 2*num_elements);
	printf("done with eb_quant huffman\n");
	//printf("comp eb max = %d\n", *std::max_element(eb_quant_index, eb_quant_index + 2*num_elements));
	//printf("comp eb min = %d\n", *std::min_element(eb_quant_index, eb_quant_index + 2*num_elements));
	free(eb_quant_index);
	// free(freq);
	// freq = NULL;
	//freq = (size_t *) calloc(num_threads * 4 * capacity*sizeof(size_t), sizeof(size_t));
	// freq = (size_t *) malloc(num_threads* 4*capacity*sizeof(size_t));
	// memset(freq, 0, num_threads* 4*capacity*sizeof(size_t));
	//std::cout<<"quant max = "<<*std::max_element(data_quant_index, data_quant_index + 2*num_elements)<<std::endl;
	//std::cout<<"quant min = "<<*std::min_element(data_quant_index, data_quant_index + 2*num_elements)<<std::endl;
	//omp_Huffman_encode_tree_and_data(capacity, data_quant_index, 2*num_elements, compressed_pos,freq, num_threads);
	
	/*
	// serial huffman data quant ****************************************************
	auto data_quant_huffman_start = std::chrono::high_resolution_clock::now();
	Huffman_encode_tree_and_data(capacity, data_quant_index, 2*num_elements, compressed_pos);
	auto data_quant_huffman_end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> data_quant_huffman_time = data_quant_huffman_end - data_quant_huffman_start;
	std::cout<<"serial data_quant_huffman_time = "<<data_quant_huffman_time.count()<<std::endl;
	// serial huffman data quant ****************************************************
	*/

	// parallel huffman data quant ****************************************************
	auto parallel_data_quant_huffman_start = std::chrono::high_resolution_clock::now();
	printf("qqqq_num_threads = %d\n", num_threads);	
	std::vector<std::vector<unsigned char>> compressed_buffers_data_quant(num_threads);
	std::vector<size_t> compressed_sizes_data_quant(num_threads);
	//resize
	auto resize_start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < num_threads; i++){
		compressed_buffers_data_quant[i].resize(2*num_elements);
	}
	auto resize_end = std::chrono::high_resolution_clock::now();
	printf("resize time = %f\n", std::chrono::duration<double>(resize_end - resize_start).count());
	#pragma omp parallel for num_threads(num_threads)
	for (int i = 0; i < num_threads; i++){
		size_t start_pos = i * num_elements / num_threads;
		size_t end_pos = (i == num_threads - 1) ? num_elements : (i + 1) * num_elements / num_threads;
		unsigned char *local_compressed_pos = compressed_buffers_data_quant[i].data();
		size_t local_compressed_size = 0;
		Huffman_encode_tree_and_data(capacity,data_quant_index + 2*start_pos, 2*(end_pos - start_pos), local_compressed_pos,local_compressed_size);
		compressed_sizes_data_quant[i] = local_compressed_size;
	}
	//write compressed_sizes_data_quant first
	for (int i = 0; i < num_threads; i++){
		write_variable_to_dst(compressed_pos, compressed_sizes_data_quant[i]);
		//printf("comp thread %d, compressed_size = %ld\n", i, compressed_sizes_data_quant[i]);
	}

	//merge compressed_buffers_data_quant write to compressed
	for (int i = 0; i < num_threads; i++){
		memcpy(compressed_pos, compressed_buffers_data_quant[i].data(), compressed_sizes_data_quant[i]);
		compressed_pos += compressed_sizes_data_quant[i];
	}
	auto parallel_data_quant_huffman_end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> parallel_data_quant_huffman_time = parallel_data_quant_huffman_end - parallel_data_quant_huffman_start;
	std::cout<<"parallel_data_quant_huffman_time = "<<parallel_data_quant_huffman_time.count()<<std::endl;
	// parallel huffman data quant ****************************************************
	printf("comp data max = %d\n", *std::max_element(data_quant_index, data_quant_index + 2*num_elements));
	printf("comp data min = %d\n", *std::min_element(data_quant_index, data_quant_index + 2*num_elements));
	free(data_quant_index);
	compressed_size = compressed_pos - compressed;
	return compressed;
}

template
unsigned char *
omp_sz_compress_cp_preserve_2d_record_vertex(const float * U, const float * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb,const std::set<size_t> &index_need_to_fix,int num_threads,float * &decompressed_U_ptr, float * &decompressed_V_ptr,std::string eb_type);

template
unsigned char *
omp_sz_compress_cp_preserve_2d_record_vertex(const double * U, const double * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb,const std::set<size_t> &index_need_to_fix,int num_threads,double * &decompressed_U_ptr, double * &decompressed_V_ptr,std::string eb_type);