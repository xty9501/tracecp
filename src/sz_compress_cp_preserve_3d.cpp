#include "sz_cp_preserve_utils.hpp"
#include "sz_compress_3d.hpp"
#include "sz_compress_cp_preserve_3d.hpp"
#include "sz_def.hpp"
#include "sz_compression_utils.hpp"
#include <unordered_map>
#include <cassert>
#include <ftk/numeric/inverse_linear_interpolation_solver.hh>
#include <chrono>



template<typename Type>
void writefile(const char * file, Type * data, size_t num_elements){
	std::ofstream fout(file, std::ios::binary);
	fout.write(reinterpret_cast<const char*>(&data[0]), num_elements*sizeof(Type));
	fout.close();
}

template<typename T>
inline double 
max_eb_to_keep_position_and_type_3d_offline(const T u0, const T u1, const T u2, const T u3, const T v0, const T v1, const T v2, const T v3,
	const T w0, const T w1, const T w2, const T w3){
	double u3v1w2 = u3*v1*w2, u3v2w1 = u3*v2*w1, u3v2w0 = u3*v2*w0, u3v0w2 = u3*v0*w2, u3v0w1 = u3*v0*w1, u3v1w0 = u3*v1*w0;
	double u1v3w2 = u1*v3*w2, u2v3w1 = u2*v3*w1, u2v3w0 = u2*v3*w0, u0v3w2 = u0*v3*w2, u0v3w1 = u0*v3*w1, u1v3w0 = u1*v3*w0;
	double u1v2w3 = u1*v2*w3, u2v1w3 = u2*v1*w3, u0v2w3 = u0*v2*w3, u2v0w3 = u2*v0*w3, u0v1w3 = u0*v1*w3, u1v0w3 = u1*v0*w3;
	double u0v1w2 = u0*v1*w2, u0v2w1 = u0*v2*w1, u1v2w0 = u1*v2*w0, u1v0w2 = u1*v0*w2, u2v0w1 = u2*v0*w1, u2v1w0 = u2*v1*w0;
	double u3_0 = - u3v1w2 + u3v2w1, u3_1 = - u3v2w0 + u3v0w2, u3_2 = - u3v0w1 + u3v1w0;
	double v3_0 = u1v3w2 - u2v3w1, v3_1 = u2v3w0 - u0v3w2, v3_2 = u0v3w1 - u1v3w0;
	double w3_0 = - u1v2w3 + u2v1w3, w3_1 = u0v2w3 - u2v0w3, w3_2 = - u0v1w3 + u1v0w3;
	double c_4 = u0v1w2 - u0v2w1 + u1v2w0 - u1v0w2 + u2v0w1 - u2v1w0;
	double M0 = u3_0 + v3_0 + w3_0;
	double M1 = u3_1 + v3_1 + w3_1;
	double M2 = u3_2 + v3_2 + w3_2;
	double M3 = c_4;
	double M = M0 + M1 + M2 + M3;
	if(M == 0) return 0;
	bool flag[4];
	flag[0] = (M0 == 0) || (M / M0 > 1);
	flag[1] = (M1 == 0) || (M / M1 > 1);
	flag[2] = (M2 == 0) || (M / M2 > 1);
	flag[3] = (M3 == 0) || (M / M3 > 1);
	if(flag[0] && flag[1] && flag[2] && flag[3]){
		// cp found
		if(same_direction(u0, u1, u2, u3) || same_direction(v0, v1, v2, v3) || same_direction(w0, w1, w2, w3)) return 1;
		return 0;
	}
	else{
		float eb = 0;
		double positive_m0 = 0, negative_m0 = 0;
		{
			// M0
			accumulate(- u3v1w2, positive_m0, negative_m0);
			accumulate(u3v2w1, positive_m0, negative_m0);
			accumulate(u1v3w2, positive_m0, negative_m0);
			accumulate(- u2v3w1, positive_m0, negative_m0);
			accumulate(- u1v2w3, positive_m0, negative_m0);
			accumulate(u2v1w3, positive_m0, negative_m0);
		}
		double positive_m1 = 0, negative_m1 = 0;
		{
			// M1
			accumulate(- u3v2w0, positive_m1, negative_m1);
			accumulate(u3v0w2, positive_m1, negative_m1);
			accumulate(u2v3w0, positive_m1, negative_m1);
			accumulate(- u0v3w2, positive_m1, negative_m1);
			accumulate(u0v2w3, positive_m1, negative_m1);
			accumulate(- u2v0w3, positive_m1, negative_m1);
		}
		double positive_m2 = 0, negative_m2 = 0;
		{
			// M2
			accumulate(- u3v0w1, positive_m2, negative_m2);
			accumulate(u3v1w0, positive_m2, negative_m2);
			accumulate(u0v3w1, positive_m2, negative_m2);
			accumulate(- u1v3w0, positive_m2, negative_m2);
			accumulate(- u0v1w3, positive_m2, negative_m2);
			accumulate(u1v0w3, positive_m2, negative_m2);			
		}
		double positive_m3 = 0, negative_m3 = 0;
		{
			// M3
			accumulate(u0v1w2, positive_m3, negative_m3);
			accumulate(- u0v2w1, positive_m3, negative_m3);
			accumulate(u1v2w0, positive_m3, negative_m3);
			accumulate(- u1v0w2, positive_m3, negative_m3);
			accumulate(u2v0w1, positive_m3, negative_m3);
			accumulate(- u2v1w0, positive_m3, negative_m3);
		}
		float p_m0 = positive_m0, p_m1 = positive_m1, p_m2 = positive_m2, p_m3 = positive_m3;
		float n_m0 = negative_m0, n_m1 = negative_m1, n_m2 = negative_m2, n_m3 = negative_m3;
		if(!flag[0]){
			float eb1 = max_eb_to_keep_sign_3d_offline(p_m0, n_m0);
			float eb2 = max_eb_to_keep_sign_3d_offline(p_m1 + p_m2 + p_m3, n_m1 + n_m2 + n_m3);
			eb = MAX(eb, MIN(eb1, eb2));
		}
		if(!flag[1]){
			float eb1 = max_eb_to_keep_sign_3d_offline(p_m1, n_m1);
			float eb2 = max_eb_to_keep_sign_3d_offline(p_m0 + p_m2 + p_m3, n_m0 + n_m2 + n_m3);
			eb = MAX(eb, MIN(eb1, eb2));
		}
		if(!flag[2]){
			float eb1 = max_eb_to_keep_sign_3d_offline(p_m2, n_m2);
			float eb2 = max_eb_to_keep_sign_3d_offline(p_m0 + p_m1 + p_m3, n_m0 + n_m1 + n_m3);
			eb = MAX(eb, MIN(eb1, eb2));
		}
		if(!flag[3]){
			float eb1 = max_eb_to_keep_sign_3d_offline(p_m3, n_m3);
			float eb2 = max_eb_to_keep_sign_3d_offline(p_m0 + p_m1 + p_m2, n_m0 + n_m1 + n_m2);
			eb = MAX(eb, MIN(eb1, eb2));
		}
		return eb;
	}
}

template<typename T>
unsigned char *
sz_compress_cp_preserve_3d_offline_log(const T * U, const T * V, const T * W, size_t r1, size_t r2, size_t r3, size_t& compressed_size, bool transpose, double max_pwr_eb){

	size_t num_elements = r1 * r2 * r3;
	size_t dim0_offset = r2 * r3;
	size_t dim1_offset = r3;
	// compute eb
	double * eb_offline = (double *) malloc(num_elements*sizeof(double));
	for(int i=0; i<num_elements; i++){
		eb_offline[i] = max_pwr_eb;
	}
	const int coordinates[6][4][3] = {
		// offset = 0, 0, 0
		{
			{0, 0, 1},
			{0, 1, 1},
			{1, 1, 1},
			{0, 0, 0}
		},
		{
			{0, 1, 0},
			{0, 1, 1},
			{1, 1, 1},
			{0, 0, 0}
		},
		{
			{0, 0, 1},
			{1, 0, 1},
			{1, 1, 1},
			{0, 0, 0}
		},
		{
			{1, 0, 0},
			{1, 0, 1},
			{1, 1, 1},
			{0, 0, 0}
		},
		{
			{0, 1, 0},
			{1, 1, 0},
			{1, 1, 1},
			{0, 0, 0}
		},
		{
			{1, 0, 0},
			{1, 1, 0},
			{1, 1, 1},
			{0, 0, 0}
		}
	};
	int index_offset[6][3][3];
	for(int i=0; i<6; i++){
		for(int j=0; j<3; j++){
			for(int k=0; k<3; k++){
				index_offset[i][j][k] = coordinates[i][j][k] - coordinates[i][3][k];
			}
		}
	}
	ptrdiff_t offset[6][3];
	for(int i=0; i<6; i++){
		for(int x=0; x<3; x++){
			offset[i][x] = (coordinates[i][x][0] - coordinates[i][3][0]) + (coordinates[i][x][1] - coordinates[i][3][1]) * dim1_offset + (coordinates[i][x][2] - coordinates[i][3][2]) * dim0_offset;
		}
	}
	const T * pre_U_pos = U;
	const T * pre_V_pos = V;
	const T * pre_W_pos = W;
	double * eb_offline_compute_pos = eb_offline;
	{
		const T * cur_U_pos = pre_U_pos;
		const T * cur_V_pos = pre_V_pos;
		const T * cur_W_pos = pre_W_pos;
		for(int i=0; i<r1 - 1; i++){
			for(int j=0; j<r2 - 1; j++){
				for(int k=0; k<r3 - 1; k++){
					for(int n=0; n<6; n++){
						double cur_eb = max_eb_to_keep_position_and_type_3d_offline(
								cur_U_pos[offset[n][0]], cur_U_pos[offset[n][1]], cur_U_pos[offset[n][2]], *cur_U_pos,
								cur_V_pos[offset[n][0]], cur_V_pos[offset[n][1]], cur_V_pos[offset[n][2]], *cur_V_pos,
								cur_W_pos[offset[n][0]], cur_W_pos[offset[n][1]], cur_W_pos[offset[n][2]], *cur_W_pos);
						eb_offline_compute_pos[0] = MIN(eb_offline_compute_pos[0], cur_eb);
						for(int ind=0; ind<3; ind++){
							eb_offline_compute_pos[offset[n][ind]] = MIN(eb_offline_compute_pos[offset[n][ind]], cur_eb);
						}
					}
					eb_offline_compute_pos ++;
					cur_U_pos ++, cur_V_pos ++, cur_W_pos ++;
				}
				// skip the last element
				eb_offline_compute_pos ++;
				cur_U_pos ++, cur_V_pos ++, cur_W_pos ++;
			}
			// skip the last row
			eb_offline_compute_pos += r3;
			cur_U_pos += r3, cur_V_pos += r3, cur_W_pos += r3;
		}
	}
	// writefile("eb_3d_offline.dat", eb_offline, num_elements);
	// exit(0);
	size_t sign_map_size = (num_elements - 1)/8 + 1;
	unsigned char * sign_map_compressed = (unsigned char *) malloc(3*sign_map_size);
	unsigned char * sign_map_compressed_pos = sign_map_compressed;
	unsigned char * sign_map = (unsigned char *) malloc(num_elements*sizeof(unsigned char));
	// Note the convert function has address auto increment
	T * log_U = log_transform(U, sign_map, num_elements);
	convertIntArray2ByteArray_fast_1b_to_result_sz(sign_map, num_elements, sign_map_compressed_pos);
	T * log_V = log_transform(V, sign_map, num_elements);
	convertIntArray2ByteArray_fast_1b_to_result_sz(sign_map, num_elements, sign_map_compressed_pos);
	T * log_W = log_transform(W, sign_map, num_elements);
	convertIntArray2ByteArray_fast_1b_to_result_sz(sign_map, num_elements, sign_map_compressed_pos);
	free(sign_map);

	T * decompressed_U = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_U, U, num_elements*sizeof(T));
	T * decompressed_V = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_V, V, num_elements*sizeof(T));
	T * decompressed_W = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_W, W, num_elements*sizeof(T));

	int * data_quant_index = (int *) malloc(3*num_elements*sizeof(int));
	int * data_quant_index_pos = data_quant_index;
	// next, row by row
	const int capacity = 65536;
	const int intv_radius = (capacity >> 1);
	int * eb_quant_index = (int *) malloc(num_elements*sizeof(int));
	int * eb_quant_index_pos = eb_quant_index;
	const int base = 2;
	const double log_of_base = log2(base);
	// offsets to get 24 adjacent simplex indices
	// x -> z, high -> low
	// current data would always be the last index, i.e. x[i][3]
	T * cur_log_U_pos = log_U;
	T * cur_log_V_pos = log_V;
	T * cur_log_W_pos = log_W;
	T * cur_U_pos = decompressed_U;
	T * cur_V_pos = decompressed_V;
	T * cur_W_pos = decompressed_W;
	unpred_vec<T> eb_zero_data = unpred_vec<T>();
	ptrdiff_t max_pointer_pos = num_elements;
	std::unordered_map<int, vector<bool>> flags;
	double threshold = std::numeric_limits<float>::epsilon();
	int eb_quant_index_max = (int) (log2(1.0 / threshold)/log_of_base) + 1;

	const double * eb_offline_pos = eb_offline;
	for(int i=0; i<r1; i++){
		// printf("start %d row\n", i);
		for(int j=0; j<r2; j++){
			for(int k=0; k<r3; k++){
				double required_eb = *(eb_offline_pos++);
				// derive eb given 24 adjacent simplex
				if(required_eb > 0){
					bool unpred_flag = false;
					T decompressed[3];
					double abs_eb = log2(1 + required_eb);
					*eb_quant_index_pos = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
					if(*eb_quant_index_pos > 0){
						// compress vector fields
						T * log_data_pos[3] = {cur_log_U_pos, cur_log_V_pos, cur_log_W_pos};
						T * data_pos[3] = {cur_U_pos, cur_V_pos, cur_W_pos};
						for(int p=0; p<3; p++){
							T * cur_log_data_pos = log_data_pos[p];
							T cur_data = *cur_log_data_pos;
							// get adjacent data and perform Lorenzo
							/*
								d6	X
								d4	d5
								d2	d3
								d0	d1
							*/
							T d0 = (i && j && k) ? cur_log_data_pos[- dim0_offset - dim1_offset - 1] : 0;
							T d1 = (i && j) ? cur_log_data_pos[- dim0_offset - dim1_offset] : 0;
							T d2 = (i && k) ? cur_log_data_pos[- dim0_offset - 1] : 0;
							T d3 = (i) ? cur_log_data_pos[- dim0_offset] : 0;
							T d4 = (j && k) ? cur_log_data_pos[- dim1_offset - 1] : 0;
							T d5 = (j) ? cur_log_data_pos[- dim1_offset] : 0;
							T d6 = (k) ? cur_log_data_pos[- 1] : 0;
							T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
							double diff = cur_data - pred;
							double quant_diff = fabs(diff) / abs_eb + 1;
							if(quant_diff < capacity){
								quant_diff = (diff > 0) ? quant_diff : -quant_diff;
								int quant_index = (int)(quant_diff/2) + intv_radius;
								data_quant_index_pos[p] = quant_index;
								decompressed[p] = pred + 2 * (quant_index - intv_radius) * abs_eb; 
								// check original data
								if(fabs(decompressed[p] - cur_data) >= abs_eb){
									unpred_flag = true;
									break;
								}
							}
							else{
								unpred_flag = true;
								break;
							}
						}
					}
					else unpred_flag = true;
					if(unpred_flag){
						*(eb_quant_index_pos ++) = eb_quant_index_max;
						eb_zero_data.push_back(*cur_U_pos);
						eb_zero_data.push_back(*cur_V_pos);
						eb_zero_data.push_back(*cur_W_pos);
					}
					else{
						eb_quant_index_pos ++;
						data_quant_index_pos += 3;
						*cur_log_U_pos = decompressed[0];
						*cur_log_V_pos = decompressed[1];
						*cur_log_W_pos = decompressed[2];
						*cur_U_pos = (*cur_U_pos > 0) ? exp2(*cur_log_U_pos) : -exp2(*cur_log_U_pos);
						*cur_V_pos = (*cur_V_pos > 0) ? exp2(*cur_log_V_pos) : -exp2(*cur_log_V_pos);
						*cur_W_pos = (*cur_W_pos > 0) ? exp2(*cur_log_W_pos) : -exp2(*cur_log_W_pos);
					}
				}
				else{
					// record as unpredictable data
					*(eb_quant_index_pos ++) = 0;
					eb_zero_data.push_back(*cur_U_pos);
					eb_zero_data.push_back(*cur_V_pos);
					eb_zero_data.push_back(*cur_W_pos);
				}
				cur_log_U_pos ++, cur_log_V_pos ++, cur_log_W_pos ++;
				cur_U_pos ++, cur_V_pos ++, cur_W_pos ++;
			}
		}
	}
	free(eb_offline);
	free(log_U);
	free(log_V);
	free(log_W);
	free(decompressed_U);
	free(decompressed_V);
	free(decompressed_W);
	// printf("offset eb_q, data_q, unpred: %ld %ld %ld\n", eb_quant_index_pos - eb_quant_index, data_quant_index_pos - data_quant_index, eb_zero_data.size());
	unsigned char * compressed = (unsigned char *) malloc(3*num_elements*sizeof(T));
	unsigned char * compressed_pos = compressed;
	write_variable_to_dst(compressed_pos, base);
	write_variable_to_dst(compressed_pos, intv_radius);
	write_array_to_dst(compressed_pos, sign_map_compressed, 3*sign_map_size);
	free(sign_map_compressed);
	size_t unpredictable_count = eb_zero_data.size();
	write_variable_to_dst(compressed_pos, unpredictable_count);
	write_array_to_dst(compressed_pos, (T *)&eb_zero_data[0], unpredictable_count);	
	// printf("eb_zero_data size = %ld\n", unpredictable_count*sizeof(T));
	// store out range information
	unsigned char * tmp = compressed_pos;
	size_t eb_quant_num = eb_quant_index_pos - eb_quant_index;
	write_variable_to_dst(compressed_pos, eb_quant_num);
	Huffman_encode_tree_and_data(2*256, eb_quant_index, num_elements, compressed_pos);
	// printf("eb_quant_index size = %ld\n", compressed_pos - tmp);
	free(eb_quant_index);
	tmp = compressed_pos;
	size_t data_quant_num = data_quant_index_pos - data_quant_index;
	write_variable_to_dst(compressed_pos, data_quant_num);
	Huffman_encode_tree_and_data(2*capacity, data_quant_index, data_quant_num, compressed_pos);
	// printf("data_quant_index size = %ld\n", compressed_pos - tmp);
	free(data_quant_index);
	compressed_size = compressed_pos - compressed;
	return compressed;	
}

template
unsigned char *
sz_compress_cp_preserve_3d_offline_log(const float * U, const float * V, const float * W, size_t r1, size_t r2, size_t r3, size_t& compressed_size, bool transpose, double max_pwr_eb);

template
unsigned char *
sz_compress_cp_preserve_3d_offline_log(const double * U, const double * V, const double * W, size_t r1, size_t r2, size_t r3, size_t& compressed_size, bool transpose, double max_pwr_eb);

typedef struct conditions_3d{
	bool computed;
	bool singular;
	bool flags[4];
	conditions_3d(){
		computed = false;
		singular = false;
		for(int i=0; i<4; i++){
			flags[i] = false;
		}
	}
}conditions_3d;

// maximal error bound to keep the sign of A*(1 + e_1) + B*(1 + e_2) + C*(1+e_3) + D
template<typename T>
inline double max_eb_to_keep_sign_3d_online(const T A, const T B, const T C, const T D=0){
	if((A == 0) && (B == 0) && (C == 0)) return 1;
	return fabs(A + B + C + D) / (fabs(A) + fabs(B) + fabs(C));
}



template<typename T>
double 
max_eb_to_keep_position_and_type_3d_online(const T u0, const T u1, const T u2, const T u3, const T v0, const T v1, const T v2, const T v3,
	const T w0, const T w1, const T w2, const T w3){
	//det = -u2 v1 w0 + u3 v1 w0 + u1 v2 w0 - u3 v2 w0 - u1 v3 w0 + u2 v3 w0 + 
	//  u2 v0 w1 - u3 v0 w1 - u0 v2 w1 + u3 v2 w1 + u0 v3 w1 - u2 v3 w1 - 
	//  u1 v0 w2 + u3 v0 w2 + u0 v1 w2 - u3 v1 w2 - u0 v3 w2 + u1 v3 w2 + 
	//  u1 v0 w3 - u2 v0 w3 - u0 v1 w3 + u2 v1 w3 + u0 v2 w3 - u1 v2 w3
	//    = P0 + P1 + P2 + P3
	// mu0 = (u3 v2 w1 - u2 v3 w1 - u3 v1 w2 + u1 v3 w2 + u2 v1 w3 - u1 v2 w3) / det = P0/(P1 + P2 + P3 + P4)
	// mu1 = (-u3 v2 w0 + u2 v3 w0 + u3 v0 w2 - u0 v3 w2 - u2 v0 w3 + u0 v2 w3) / det = P1/(P1 + P2 + P3 + P4)
	// mu2 = (u3 v1 w0 - u1 v3 w0 - u3 v0 w1 + u0 v3 w1 + u1 v0 w3 - u0 v1 w3) / det = P2/(P1 + P2 + P3 + P4)
	// mu3 = (-u2 v1 w0 + u1 v2 w0 + u2 v0 w1 - u0 v2 w1 - u1 v0 w2 + u0 v1 w2) / det = P3/(P1 + P2 + P3 + P4)
	// cond.computed = false;
	// if(!cond.computed){
	//     double M0 = -u1*v2*w3 + u1*v3*w2 - u2*v3*w1 + u2*v1*w3 - u3*v1*w2 + u3*v2*w1;
	//     double M1 = -u0*v3*w2 + u0*v2*w3 - u2*v0*w3 + u2*v3*w0 - u3*v2*w0 + u3*v0*w2;
	//     double M2 = -u0*v1*w3 + u0*v3*w1 - u1*v3*w0 + u1*v0*w3 - u3*v0*w1 + u3*v1*w0;
	//     double M3 = u0*v1*w2 - u0*v2*w1 + u1*v2*w0 - u1*v0*w2 + u2*v0*w1 - u2*v1*w0;
	//     double M = M0 + M1 + M2 + M3;
	//     cond.singular = (M == 0);
	//     if(cond.singular) return 0;
	//     cond.flags[0] = (M0 == 0) || (M / M0 > 1);
	//     cond.flags[1] = (M1 == 0) || (M / M1 > 1);
	//     cond.flags[2] = (M2 == 0) || (M / M2 > 1);
	//     cond.flags[3] = (M3 == 0) || (M / M3 > 1);
	//     cond.computed = true;
	// }
	// else{
	//     if(cond.singular) return 0;
	// }
	// const bool * flag = cond.flags;
	double u3_0 = - u3*v1*w2 + u3*v2*w1, u3_1 = - u3*v2*w0 + u3*v0*w2, u3_2 = - u3*v0*w1 + u3*v1*w0;
	double v3_0 = u1*v3*w2 - u2*v3*w1, v3_1 = u2*v3*w0 - u0*v3*w2, v3_2 = u0*v3*w1 - u1*v3*w0;
	double w3_0 = - u1*v2*w3 + u2*v1*w3, w3_1 = u0*v2*w3 - u2*v0*w3, w3_2 = - u0*v1*w3 + u1*v0*w3;
	double c_4 = u0*v1*w2 - u0*v2*w1 + u1*v2*w0 - u1*v0*w2 + u2*v0*w1 - u2*v1*w0;
	double M0 = u3_0 + v3_0 + w3_0;
	double M1 = u3_1 + v3_1 + w3_1;
	double M2 = u3_2 + v3_2 + w3_2;
	double M3 = c_4;
	double M = M0 + M1 + M2 + M3;
	if(M == 0){
		if(same_direction(u0, u1, u2, u3) || same_direction(v0, v1, v2, v3) || same_direction(w0, w1, w2, w3)) return 1;
		return 0;
	}
	bool flag[4];
	flag[0] = (M0 == 0) || (M / M0 > 1);
	flag[1] = (M1 == 0) || (M / M1 > 1);
	flag[2] = (M2 == 0) || (M / M2 > 1);
	flag[3] = (M3 == 0) || (M / M3 > 1);
	if(flag[0] && flag[1] && flag[2] && flag[3]){
		// cp found
		return 0;
		double eb = 1;
		double cur_eb = 0;
		eb = MIN(eb, max_eb_to_keep_sign_3d_online(u3_0, v3_0, w3_0));
		eb = MIN(eb, max_eb_to_keep_sign_3d_online(u3_1, v3_1, w3_1));
		eb = MIN(eb, max_eb_to_keep_sign_3d_online(u3_2, v3_2, w3_2));
		eb = MIN(eb, max_eb_to_keep_sign_3d_online(u3_0 + u3_1 + u3_2, v3_0 + v3_1 + v3_2, w3_0 + w3_1 + w3_2));
		return eb;
	}
	else{
		double eb = 0;
		double cur_eb = 0;
		if(!flag[0]){
			cur_eb = MIN(max_eb_to_keep_sign_3d_online(u3_0, v3_0, w3_0), 
					max_eb_to_keep_sign_3d_online(u3_1 + u3_2, v3_1 + v3_2, w3_1 + w3_2, c_4));
			eb = MAX(eb, cur_eb);
		}
		if(!flag[1]){
			cur_eb = MIN(max_eb_to_keep_sign_3d_online(u3_1, v3_1, w3_1), 
					max_eb_to_keep_sign_3d_online(u3_0 + u3_2, v3_0 + v3_2, w3_0 + w3_2, c_4));
			eb = MAX(eb, cur_eb);
		}
		if(!flag[2]){
			cur_eb = MIN(max_eb_to_keep_sign_3d_online(u3_2, v3_2, w3_2), 
					max_eb_to_keep_sign_3d_online(u3_0 + u3_1, v3_0 + v3_1, w3_0 + w3_1, c_4));
			eb = MAX(eb, cur_eb);
		}
		if(!flag[3]){
			cur_eb = max_eb_to_keep_sign_3d_online(u3_0 + u3_1 + u3_2, v3_0 + v3_1 + v3_2, w3_0 + w3_1 + w3_2);
			eb = MAX(eb, cur_eb);
		}
		return eb;
	}
}

template<typename T>
unsigned char *
sz_compress_cp_preserve_3d_online_log(const T * U, const T * V, const T * W, size_t r1, size_t r2, size_t r3, size_t& compressed_size, bool transpose, double max_pwr_eb){

	size_t num_elements = r1 * r2 * r3;
	size_t sign_map_size = (num_elements - 1)/8 + 1;
	unsigned char * sign_map_compressed = (unsigned char *) malloc(3*sign_map_size);
	unsigned char * sign_map_compressed_pos = sign_map_compressed;
	unsigned char * sign_map = (unsigned char *) malloc(num_elements*sizeof(unsigned char));
	// Note the convert function has address auto increment
	T * log_U = log_transform(U, sign_map, num_elements);
	convertIntArray2ByteArray_fast_1b_to_result_sz(sign_map, num_elements, sign_map_compressed_pos);
	T * log_V = log_transform(V, sign_map, num_elements);
	convertIntArray2ByteArray_fast_1b_to_result_sz(sign_map, num_elements, sign_map_compressed_pos);
	T * log_W = log_transform(W, sign_map, num_elements);
	convertIntArray2ByteArray_fast_1b_to_result_sz(sign_map, num_elements, sign_map_compressed_pos);
	free(sign_map);

	T * decompressed_U = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_U, U, num_elements*sizeof(T));
	T * decompressed_V = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_V, V, num_elements*sizeof(T));
	T * decompressed_W = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_W, W, num_elements*sizeof(T));

	int * eb_quant_index = (int *) malloc(num_elements*sizeof(int));
	int * data_quant_index = (int *) malloc(3*num_elements*sizeof(int));
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	// next, row by row
	const int base = 2;
	const double log_of_base = log2(base);
	const int capacity = 65536;
	const int intv_radius = (capacity >> 1);
	size_t dim0_offset = r2 * r3;
	size_t dim1_offset = r3;
	// offsets to get 24 adjacent simplex indices
	// x -> z, high -> low
	// current data would always be the last index, i.e. x[i][3]
	const int coordinates[24][4][3] = {
		// offset = 0, 0, 0
		{
			{0, 0, 1},
			{0, 1, 1},
			{1, 1, 1},
			{0, 0, 0}
		},
		{
			{0, 1, 0},
			{0, 1, 1},
			{1, 1, 1},
			{0, 0, 0}
		},
		{
			{0, 0, 1},
			{1, 0, 1},
			{1, 1, 1},
			{0, 0, 0}
		},
		{
			{1, 0, 0},
			{1, 0, 1},
			{1, 1, 1},
			{0, 0, 0}
		},
		{
			{0, 1, 0},
			{1, 1, 0},
			{1, 1, 1},
			{0, 0, 0}
		},
		{
			{1, 0, 0},
			{1, 1, 0},
			{1, 1, 1},
			{0, 0, 0}
		},
		// offset = -1, 0, 0
		{
			{0, 0, 0},
			{1, 0, 1},
			{1, 1, 1},
			{1, 0, 0}
		},
		{
			{0, 0, 0},
			{1, 1, 0},
			{1, 1, 1},
			{1, 0, 0}
		},
		// offset = 0, -1, 0
		{
			{0, 0, 0},
			{0, 1, 1},
			{1, 1, 1},
			{0, 1, 0}
		},
		{
			{0, 0, 0},
			{1, 1, 0},
			{1, 1, 1},
			{0, 1, 0}
		},
		// offset = -1, -1, 0
		{
			{0, 0, 0},
			{0, 1, 0},
			{1, 1, 1},
			{1, 1, 0}
		},
		{
			{0, 0, 0},
			{1, 0, 0},
			{1, 1, 1},
			{1, 1, 0}
		},
		// offset = 0, 0, -1
		{
			{0, 0, 0},
			{0, 1, 1},
			{1, 1, 1},
			{0, 0, 1}
		},
		{
			{0, 0, 0},
			{1, 0, 1},
			{1, 1, 1},
			{0, 0, 1}
		},
		// offset = -1, 0, -1
		{
			{0, 0, 0},
			{0, 0, 1},
			{1, 1, 1},
			{1, 0, 1}
		},
		{
			{0, 0, 0},
			{1, 0, 0},
			{1, 1, 1},
			{1, 0, 1}
		},
		// offset = 0, -1, -1
		{
			{0, 0, 0},
			{0, 0, 1},
			{1, 1, 1},
			{0, 1, 1}
		},
		{
			{0, 0, 0},
			{0, 1, 0},
			{1, 1, 1},
			{0, 1, 1}
		},
		// offset = -1, -1, -1
		{
			{0, 0, 0},
			{0, 0, 1},
			{0, 1, 1},
			{1, 1, 1}
		},
		{
			{0, 0, 0},
			{0, 1, 0},
			{0, 1, 1},
			{1, 1, 1}
		},
		{
			{0, 0, 0},
			{0, 0, 1},
			{1, 0, 1},
			{1, 1, 1}
		},
		{
			{0, 0, 0},
			{1, 0, 0},
			{1, 0, 1},
			{1, 1, 1}
		},
		{
			{0, 0, 0},
			{0, 1, 0},
			{1, 1, 0},
			{1, 1, 1}
		},
		{
			{0, 0, 0},
			{1, 0, 0},
			{1, 1, 0},
			{1, 1, 1}
		}
	};
	ptrdiff_t simplex_offset[24];
	{
		ptrdiff_t * simplex_offset_pos = simplex_offset;
		ptrdiff_t base = 0;
		// offset = 0, 0, 0
		for(int i=0; i<6; i++){
			*(simplex_offset_pos++) = i;
		}
		// offset = -1, 0, 0
		base = -6*dim0_offset;
		*(simplex_offset_pos++) = base + 3;
		*(simplex_offset_pos++) = base + 5;
		// offset = 0, -1, 0
		base = -6*dim1_offset;
		*(simplex_offset_pos++) = base + 1;
		*(simplex_offset_pos++) = base + 4;
		// offset = -1, -1, 0
		base = -6*dim0_offset - 6*dim1_offset;
		*(simplex_offset_pos++) = base + 4;
		*(simplex_offset_pos++) = base + 5;
		// offset = 0, 0, -1
		base = -6;
		*(simplex_offset_pos++) = base + 0;
		*(simplex_offset_pos++) = base + 2;
		// offset = -1, 0, -1
		base = -6*dim0_offset - 6;
		*(simplex_offset_pos++) = base + 2;
		*(simplex_offset_pos++) = base + 3;
		// offset = 0, -1, -1
		base = -6*dim1_offset - 6;
		*(simplex_offset_pos++) = base + 0;
		*(simplex_offset_pos++) = base + 1;
		// offset = -1, -1, -1
		base = -6*dim0_offset - 6*dim1_offset - 6;
		for(int i=0; i<6; i++){
			*(simplex_offset_pos++) = base + i;
		}
	}
	int index_offset[24][3][3];
	for(int i=0; i<24; i++){
		for(int j=0; j<3; j++){
			for(int k=0; k<3; k++){
				index_offset[i][j][k] = coordinates[i][j][k] - coordinates[i][3][k];
			}
		}
	}
	ptrdiff_t offset[24][3];
	for(int i=0; i<24; i++){
		for(int x=0; x<3; x++){
			// offset[i][x] = (coordinates[i][x][0] - coordinates[i][3][0]) * dim0_offset + (coordinates[i][x][1] - coordinates[i][3][1]) * dim1_offset + (coordinates[i][x][2] - coordinates[i][3][2]);
			offset[i][x] = (coordinates[i][x][0] - coordinates[i][3][0]) + (coordinates[i][x][1] - coordinates[i][3][1]) * dim1_offset + (coordinates[i][x][2] - coordinates[i][3][2]) * dim0_offset;
		}
	}
	T * cur_log_U_pos = log_U;
	T * cur_log_V_pos = log_V;
	T * cur_log_W_pos = log_W;
	T * cur_U_pos = decompressed_U;
	T * cur_V_pos = decompressed_V;
	T * cur_W_pos = decompressed_W;
	unpred_vec<T> eb_zero_data = unpred_vec<T>();
	ptrdiff_t max_pointer_pos = num_elements;
	double threshold = std::numeric_limits<float>::epsilon();
	int eb_quant_index_max = (int) (log2(1.0 / threshold)/log_of_base) + 1;

	// double * eb = (double *) malloc(sizeof(double)*num_elements);
	// int eb_index = 0;
	// int est_outrange = num_elements * 0.1;
	// unsigned char * outrange_sign = (unsigned char *) malloc(est_outrange);
	// int * outrange_exp = (int *) malloc(est_outrange*sizeof(int));
	// unsigned char * outrange_residue = (unsigned char *) malloc(est_outrange*sizeof(T));
	// unsigned char * outrange_sign_pos = outrange_sign;
	// int * outrange_exp_pos = outrange_exp; 
	// unsigned char * outrange_residue_pos = outrange_residue;
	// int outrange_pos = 0;
	// unpred_vec<float> outrange_data = unpred_vec<float>();
	// record flags
	for(int i=0; i<r1; i++){
		// printf("start %d row\n", i);
		for(int j=0; j<r2; j++){
			for(int k=0; k<r3; k++){
				double required_eb = max_pwr_eb;
				// derive eb given 24 adjacent simplex
				for(int n=0; n<24; n++){
					bool in_mesh = true;
					for(int p=0; p<3; p++){
						// reversed order!
						if(!(in_range(i + index_offset[n][p][2], (int)r1) && in_range(j + index_offset[n][p][1], (int)r2) && in_range(k + index_offset[n][p][0], (int)r3))){
							in_mesh = false;
							break;
						}
					}
					if(in_mesh){
						int index = simplex_offset[n] + i*6*dim0_offset + j*6*dim1_offset + k*6; // TODO: define index for each simplex
						required_eb = MIN(required_eb, max_eb_to_keep_position_and_type_3d_online(
							cur_U_pos[offset[n][0]], cur_U_pos[offset[n][1]], cur_U_pos[offset[n][2]], *cur_U_pos,
							cur_V_pos[offset[n][0]], cur_V_pos[offset[n][1]], cur_V_pos[offset[n][2]], *cur_V_pos,
							cur_W_pos[offset[n][0]], cur_W_pos[offset[n][1]], cur_W_pos[offset[n][2]], *cur_W_pos));
					}
				}
				// eb[eb_index ++] = required_eb;
				if(required_eb < 1e-6) required_eb = 0;
				if(required_eb > 0){
					bool unpred_flag = false;
					T decompressed[3];
					double abs_eb = log2(1 + required_eb);
					*eb_quant_index_pos = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
					if(*eb_quant_index_pos > 0){
						// compress vector fields
						T * log_data_pos[3] = {cur_log_U_pos, cur_log_V_pos, cur_log_W_pos};
						T * data_pos[3] = {cur_U_pos, cur_V_pos, cur_W_pos};
						for(int p=0; p<3; p++){
							T * cur_log_data_pos = log_data_pos[p];
							T cur_data = *cur_log_data_pos;
							// get adjacent data and perform Lorenzo
							/*
								d6	X
								d4	d5
								d2	d3
								d0	d1
							*/
							T d0 = (i && j && k) ? cur_log_data_pos[- dim0_offset - dim1_offset - 1] : 0;
							T d1 = (i && j) ? cur_log_data_pos[- dim0_offset - dim1_offset] : 0;
							T d2 = (i && k) ? cur_log_data_pos[- dim0_offset - 1] : 0;
							T d3 = (i) ? cur_log_data_pos[- dim0_offset] : 0;
							T d4 = (j && k) ? cur_log_data_pos[- dim1_offset - 1] : 0;
							T d5 = (j) ? cur_log_data_pos[- dim1_offset] : 0;
							T d6 = (k) ? cur_log_data_pos[- 1] : 0;
							T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
							double diff = cur_data - pred;
							double quant_diff = fabs(diff) / abs_eb + 1;
							if(quant_diff < capacity){
								quant_diff = (diff > 0) ? quant_diff : -quant_diff;
								int quant_index = (int)(quant_diff/2) + intv_radius;
								data_quant_index_pos[p] = quant_index;
								decompressed[p] = pred + 2 * (quant_index - intv_radius) * abs_eb; 
								// check original data
								if(fabs(decompressed[p] - cur_data) >= abs_eb){
									unpred_flag = true;
									break;
								}
							}
							else{
								unpred_flag = true;
								break;
							}
						}
					}
					else unpred_flag = true;
					if(unpred_flag){
						*(eb_quant_index_pos ++) = eb_quant_index_max;
						// int outrange_residue_len = exp_offset<double>() - getExponent(required_eb) + 2;
						// *cur_U_pos = out_of_range_data_encode(*cur_U_pos, outrange_residue_len, outrange_sign_pos, outrange_exp_pos, outrange_residue_pos, outrange_pos);
						// *cur_V_pos = out_of_range_data_encode(*cur_V_pos, outrange_residue_len, outrange_sign_pos, outrange_exp_pos, outrange_residue_pos, outrange_pos);
						// *cur_W_pos = out_of_range_data_encode(*cur_W_pos, outrange_residue_len, outrange_sign_pos, outrange_exp_pos, outrange_residue_pos, outrange_pos);
						// *cur_log_U_pos = log2(fabs(*cur_U_pos));
						// *cur_log_V_pos = log2(fabs(*cur_V_pos));
						// *cur_log_W_pos = log2(fabs(*cur_W_pos));
						// printf("outrange_residue_len = %d\n", outrange_residue_pos - outrange_residue);

						eb_zero_data.push_back(*cur_U_pos);
						eb_zero_data.push_back(*cur_V_pos);
						eb_zero_data.push_back(*cur_W_pos);
					}
					else{
						eb_quant_index_pos ++;
						data_quant_index_pos += 3;
						*cur_log_U_pos = decompressed[0];
						*cur_log_V_pos = decompressed[1];
						*cur_log_W_pos = decompressed[2];
						*cur_U_pos = (*cur_U_pos > 0) ? exp2(*cur_log_U_pos) : -exp2(*cur_log_U_pos);
						*cur_V_pos = (*cur_V_pos > 0) ? exp2(*cur_log_V_pos) : -exp2(*cur_log_V_pos);
						*cur_W_pos = (*cur_W_pos > 0) ? exp2(*cur_log_W_pos) : -exp2(*cur_log_W_pos);
					}
				}
				else{
					// record as unpredictable data
					*(eb_quant_index_pos ++) = 0;
					eb_zero_data.push_back(*cur_U_pos);
					eb_zero_data.push_back(*cur_V_pos);
					eb_zero_data.push_back(*cur_W_pos);
				}
				cur_log_U_pos ++, cur_log_V_pos ++, cur_log_W_pos ++;
				cur_U_pos ++, cur_V_pos ++, cur_W_pos ++;
			}
		}
	}
	// printf("%d %d\n", eb_index, num_elements);
	// writefile("eb_3d.dat", eb, num_elements);
	// free(eb);
	// if(outrange_pos) outrange_residue_pos ++;
	free(log_U);
	free(log_V);
	free(log_W);
	free(decompressed_U);
	free(decompressed_V);
	free(decompressed_W);
	printf("offset eb_q, data_q, unpred: %ld %ld %ld\n", eb_quant_index_pos - eb_quant_index, data_quant_index_pos - data_quant_index, eb_zero_data.size());
	unsigned char * compressed = (unsigned char *) malloc(3*num_elements*sizeof(T));
	unsigned char * compressed_pos = compressed;
	write_variable_to_dst(compressed_pos, base);
	write_variable_to_dst(compressed_pos, intv_radius);
	write_array_to_dst(compressed_pos, sign_map_compressed, 3*sign_map_size);
	free(sign_map_compressed);
	size_t unpredictable_count = eb_zero_data.size();
	write_variable_to_dst(compressed_pos, unpredictable_count);
	write_array_to_dst(compressed_pos, (T *)&eb_zero_data[0], unpredictable_count);	
	printf("eb_zero_data size = %ld\n", unpredictable_count*sizeof(T));
	// store out range information
	unsigned char * tmp = compressed_pos;
	// size_t outrange_count = outrange_sign_pos - outrange_sign;
	// write_variable_to_dst(compressed_pos, outrange_count);
	// convertIntArray2ByteArray_fast_1b_to_result_sz(outrange_sign, outrange_count, compressed_pos);
	// unsigned char * tmp2 = compressed_pos;
	// Huffman_encode_tree_and_data(2*(exp_offset<T>() + 1), outrange_exp, outrange_count, compressed_pos);
	// unsigned char * tmp3 = compressed_pos;
	// write_array_to_dst(compressed_pos, outrange_residue, outrange_residue_pos - outrange_residue);
	// printf("outrange count = %ld, outrange_exp_size = %ld, outrange_residue_size = %ld\n", outrange_count, tmp3 - tmp2, outrange_residue_pos - outrange_residue);
	// printf("outrange size = %ld\n", compressed_pos - tmp);
	// free(outrange_sign);
	// free(outrange_exp);
	// free(outrange_residue);
	tmp = compressed_pos;
	size_t eb_quant_num = eb_quant_index_pos - eb_quant_index;
	write_variable_to_dst(compressed_pos, eb_quant_num);
	Huffman_encode_tree_and_data(2*256, eb_quant_index, num_elements, compressed_pos);
	printf("eb_quant_index size = %ld\n", compressed_pos - tmp);
	free(eb_quant_index);
	tmp = compressed_pos;
	size_t data_quant_num = data_quant_index_pos - data_quant_index;
	write_variable_to_dst(compressed_pos, data_quant_num);
	Huffman_encode_tree_and_data(2*capacity, data_quant_index, data_quant_num, compressed_pos);
	printf("data_quant_index size = %ld\n", compressed_pos - tmp);
	free(data_quant_index);
	compressed_size = compressed_pos - compressed;
	return compressed;	
}

template
unsigned char *
sz_compress_cp_preserve_3d_online_log(const float * U, const float * V, const float * W, size_t r1, size_t r2, size_t r3, size_t& compressed_size, bool transpose, double max_pwr_eb);

template
unsigned char *
sz_compress_cp_preserve_3d_online_log(const double * U, const double * V, const double * W, size_t r1, size_t r2, size_t r3, size_t& compressed_size, bool transpose, double max_pwr_eb);


typedef struct Tet{
	int vertex[4];
}Tet;

template<typename T>
std::vector<Tet> construct_tets(int n, const T * data, int m, const int * tets_ind, std::vector<std::vector<std::pair<int, int>>>& point_tets){
	std::vector<Tet> tets;
	point_tets.clear();
	for(int i=0; i<n; i++){
		point_tets.push_back(std::vector<std::pair<int, int>>());
	}
	const int * tets_ind_pos = tets_ind;
	for(int i=0; i<m; i++){
		Tet t;
		for(int j=0; j<4; j++){
			int ind = *(tets_ind_pos ++);
			t.vertex[j] = ind;
			point_tets[ind].push_back(make_pair(i, j));
		}
		tets.push_back(t);
	}
	return tets;
}

template<typename T>
unsigned char *
sz_compress_cp_preserve_3d_unstructured(int n, const T * points, const T * data, int m, const int * tets_ind, size_t& compressed_size, double max_pwr_eb){
	// map point to adjcent tets and position in that tets
	std::vector<std::vector<std::pair<int, int>>> point_tets;
	size_t sign_map_size = (3*n - 1)/8 + 1;
	unsigned char * sign_map = (unsigned char *) malloc(3*n*sizeof(unsigned char));
	T * log_data = log_transform(data, sign_map, 3*n);
	std::vector<Tet> tets = construct_tets(n, data, m, tets_ind, point_tets);
	unsigned char * sign_map_compressed = (unsigned char *) malloc(sign_map_size);
	unsigned char * sign_map_compressed_pos = sign_map_compressed;
	convertIntArray2ByteArray_fast_1b_to_result_sz(sign_map, 3*n, sign_map_compressed_pos);
	free(sign_map);

	unpred_vec<T> eb_zero_data = unpred_vec<T>();
	double threshold = std::numeric_limits<float>::epsilon();
	const int base = 2;
	const double log_of_base = log2(base);
	const int capacity = 65536;
	const int intv_radius = (capacity >> 1);
	const int eb_quant_index_max = (int) (log2(1.0 / threshold)/log_of_base) + 1;
	
	T * dec_data = (T *) malloc(3*n*sizeof(T));
	memcpy(dec_data, data, 3*n*sizeof(T));
	T * dec_data_pos = dec_data;
	T * log_data_pos = log_data;
	int * eb_quant_index = (int *) malloc(n*sizeof(int));
	int * data_quant_index = (int *) malloc(3*n*sizeof(int));
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	int count = 0;
	for(int i=0; i<n; i++){
		double required_eb = max_pwr_eb;
		auto adj_tets_ids = point_tets[i];
		for(const auto& id:adj_tets_ids){
			auto t = tets[id.first];
			int pos = id.second;
			std::vector<int> inds{0, 1, 2, 3};
			inds.erase(inds.begin() + pos);
			const int data_offset[4] = {t.vertex[inds[0]]*3, t.vertex[inds[1]]*3, t.vertex[inds[2]]*3, t.vertex[pos]*3};
			required_eb = MIN(required_eb, max_eb_to_keep_position_and_type_3d_online(
				dec_data[data_offset[0]], dec_data[data_offset[1]], dec_data[data_offset[2]], dec_data[data_offset[3]],
				dec_data[data_offset[0] + 1], dec_data[data_offset[1] + 1], dec_data[data_offset[2] + 1], dec_data[data_offset[3] + 1],
				dec_data[data_offset[0] + 2], dec_data[data_offset[1] + 2], dec_data[data_offset[2] + 2], dec_data[data_offset[3] + 2]));
		}
		if(required_eb < 1e-10) required_eb = 0;
		if(required_eb > 0){
			bool unpred_flag = false;
			double abs_eb = log2(1 + required_eb);
			*eb_quant_index_pos = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
			T decompressed[3];
			if(*eb_quant_index_pos > 0){
				// compress vector fields
				for(int p=0; p<3; p++){
					T * cur_log_data_pos = log_data_pos + p;
					T cur_data = *cur_log_data_pos;
					T pred = (i) ? cur_log_data_pos[-3] : 0;
					double diff = cur_data - pred;
					double quant_diff = fabs(diff) / abs_eb + 1;
					if(quant_diff < capacity){
						quant_diff = (diff > 0) ? quant_diff : -quant_diff;
						int quant_index = (int)(quant_diff/2) + intv_radius;
						data_quant_index_pos[p] = quant_index;
						decompressed[p] = pred + 2 * (quant_index - intv_radius) * abs_eb; 
						// check original data
						if(fabs(decompressed[p] - cur_data) >= abs_eb){
							unpred_flag = true;
							break;
						}
					}
					else{
						unpred_flag = true;
						break;
					}
				}
			}
			else unpred_flag = true;
			if(unpred_flag){
				*(eb_quant_index_pos ++) = 0;
				eb_zero_data.push_back(dec_data_pos[0]);
				eb_zero_data.push_back(dec_data_pos[1]);
				eb_zero_data.push_back(dec_data_pos[2]);
			}
			else{
				eb_quant_index_pos ++;
				data_quant_index_pos += 3;
				log_data_pos[0] = decompressed[0];
				log_data_pos[1] = decompressed[1];
				log_data_pos[2] = decompressed[2];
				dec_data_pos[0] = (dec_data_pos[0] > 0) ? exp2(log_data_pos[0]) : -exp2(log_data_pos[0]);
				dec_data_pos[1] = (dec_data_pos[1] > 0) ? exp2(log_data_pos[1]) : -exp2(log_data_pos[1]);
				dec_data_pos[2] = (dec_data_pos[2] > 0) ? exp2(log_data_pos[2]) : -exp2(log_data_pos[2]);
			}
		}
		else{
			// record as unpredictable data
			count ++;
			*(eb_quant_index_pos ++) = 0;
			eb_zero_data.push_back(dec_data_pos[0]);
			eb_zero_data.push_back(dec_data_pos[1]);
			eb_zero_data.push_back(dec_data_pos[2]);
		}
		log_data_pos += 3;
		dec_data_pos += 3;
	}
	// printf("eb 0 count = %d\n", count);
	free(dec_data);
	free(log_data);
	// printf("offset eb_q, data_q, unpred: %ld %ld %ld\n", eb_quant_index_pos - eb_quant_index, data_quant_index_pos - data_quant_index, eb_zero_data.size());
	unsigned char * compressed = (unsigned char *) malloc(3*n*sizeof(T));
	unsigned char * compressed_pos = compressed;
	write_variable_to_dst(compressed_pos, base);
	write_variable_to_dst(compressed_pos, intv_radius);
	write_array_to_dst(compressed_pos, sign_map_compressed, sign_map_size);
	free(sign_map_compressed);
	// printf("sign map size = %ld\n", sign_map_size);
	size_t unpredictable_count = eb_zero_data.size();
	write_variable_to_dst(compressed_pos, unpredictable_count);
	write_array_to_dst(compressed_pos, (T *)&eb_zero_data[0], unpredictable_count);	
	// printf("eb_zero_data size = %ld\n", unpredictable_count*sizeof(T));
	// store out range information
	unsigned char * tmp = compressed_pos;
	size_t eb_quant_num = eb_quant_index_pos - eb_quant_index;
	write_variable_to_dst(compressed_pos, eb_quant_num);
	Huffman_encode_tree_and_data(2*256, eb_quant_index, n, compressed_pos);
	// printf("eb_quant_index size = %ld\n", compressed_pos - tmp);
	free(eb_quant_index);
	tmp = compressed_pos;
	size_t data_quant_num = data_quant_index_pos - data_quant_index;
	write_variable_to_dst(compressed_pos, data_quant_num);
	Huffman_encode_tree_and_data(2*capacity, data_quant_index, data_quant_num, compressed_pos);
	// printf("data_quant_index size = %ld\n", compressed_pos - tmp);
	free(data_quant_index);
	compressed_size = compressed_pos - compressed;
	// printf("size = %ld\n", compressed_size);
	return compressed;	
}

template
unsigned char *
sz_compress_cp_preserve_3d_unstructured(int n, const float * points, const float * data, int m, const int * tets_ind, size_t& compressed_size, double max_pwr_eb);

template
unsigned char *
sz_compress_cp_preserve_3d_unstructured(int n, const double * points, const double * data, int m, const int * tets_ind, size_t& compressed_size, double max_pwr_eb);


template<typename T>
unsigned char *
sz_compress_cp_preserve_3d_record_vertex(const T * U, const T * V, const T * W, size_t r1, size_t r2, size_t r3, size_t& compressed_size, bool transpose, double max_pwr_eb,const std::set<size_t>& index_need_to_lossless){

	size_t num_elements = r1 * r2 * r3;
	//准备bitmap
	unsigned char * bitmap = (unsigned char *) malloc(num_elements*sizeof(unsigned char));
	if(bitmap == NULL){
		printf("Error: bitmap malloc failed\n");
		exit(1);
	}
	//set all to 0
	memset(bitmap, 0, num_elements*sizeof(unsigned char));
	//set the index to 1
	for(auto it = index_need_to_lossless.begin(); it != index_need_to_lossless.end(); it++){
		assert(*it < num_elements);
		bitmap[*it] = 1;
	}
	size_t intArrayLength = num_elements;
	//输出的长度num_bytes
	size_t num_bytes = (intArrayLength % 8 == 0) ? intArrayLength / 8 : intArrayLength / 8 + 1;
	//准备完成
	size_t sign_map_size = (num_elements - 1)/8 + 1;
	unsigned char * sign_map_compressed = (unsigned char *) malloc(3*sign_map_size);
	unsigned char * sign_map_compressed_pos = sign_map_compressed;
	unsigned char * sign_map = (unsigned char *) malloc(num_elements*sizeof(unsigned char));
	// Note the convert function has address auto increment
	T * log_U = log_transform(U, sign_map, num_elements);
	convertIntArray2ByteArray_fast_1b_to_result_sz(sign_map, num_elements, sign_map_compressed_pos);
	T * log_V = log_transform(V, sign_map, num_elements);
	convertIntArray2ByteArray_fast_1b_to_result_sz(sign_map, num_elements, sign_map_compressed_pos);
	T * log_W = log_transform(W, sign_map, num_elements);
	convertIntArray2ByteArray_fast_1b_to_result_sz(sign_map, num_elements, sign_map_compressed_pos);
	free(sign_map);

	T * decompressed_U = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_U, U, num_elements*sizeof(T));
	T * decompressed_V = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_V, V, num_elements*sizeof(T));
	T * decompressed_W = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_W, W, num_elements*sizeof(T));

	int * eb_quant_index = (int *) malloc(num_elements*sizeof(int));
	int * data_quant_index = (int *) malloc(3*num_elements*sizeof(int));
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	// next, row by row
	const int base = 2;
	const double log_of_base = log2(base);
	const int capacity = 65536;
	const int intv_radius = (capacity >> 1);
	size_t dim0_offset = r2 * r3;
	size_t dim1_offset = r3;
	// offsets to get 24 adjacent simplex indices
	// x -> z, high -> low
	// current data would always be the last index, i.e. x[i][3]
	const int coordinates[24][4][3] = {
		// offset = 0, 0, 0
		{
			{0, 0, 1},
			{0, 1, 1},
			{1, 1, 1},
			{0, 0, 0}
		},
		{
			{0, 1, 0},
			{0, 1, 1},
			{1, 1, 1},
			{0, 0, 0}
		},
		{
			{0, 0, 1},
			{1, 0, 1},
			{1, 1, 1},
			{0, 0, 0}
		},
		{
			{1, 0, 0},
			{1, 0, 1},
			{1, 1, 1},
			{0, 0, 0}
		},
		{
			{0, 1, 0},
			{1, 1, 0},
			{1, 1, 1},
			{0, 0, 0}
		},
		{
			{1, 0, 0},
			{1, 1, 0},
			{1, 1, 1},
			{0, 0, 0}
		},
		// offset = -1, 0, 0
		{
			{0, 0, 0},
			{1, 0, 1},
			{1, 1, 1},
			{1, 0, 0}
		},
		{
			{0, 0, 0},
			{1, 1, 0},
			{1, 1, 1},
			{1, 0, 0}
		},
		// offset = 0, -1, 0
		{
			{0, 0, 0},
			{0, 1, 1},
			{1, 1, 1},
			{0, 1, 0}
		},
		{
			{0, 0, 0},
			{1, 1, 0},
			{1, 1, 1},
			{0, 1, 0}
		},
		// offset = -1, -1, 0
		{
			{0, 0, 0},
			{0, 1, 0},
			{1, 1, 1},
			{1, 1, 0}
		},
		{
			{0, 0, 0},
			{1, 0, 0},
			{1, 1, 1},
			{1, 1, 0}
		},
		// offset = 0, 0, -1
		{
			{0, 0, 0},
			{0, 1, 1},
			{1, 1, 1},
			{0, 0, 1}
		},
		{
			{0, 0, 0},
			{1, 0, 1},
			{1, 1, 1},
			{0, 0, 1}
		},
		// offset = -1, 0, -1
		{
			{0, 0, 0},
			{0, 0, 1},
			{1, 1, 1},
			{1, 0, 1}
		},
		{
			{0, 0, 0},
			{1, 0, 0},
			{1, 1, 1},
			{1, 0, 1}
		},
		// offset = 0, -1, -1
		{
			{0, 0, 0},
			{0, 0, 1},
			{1, 1, 1},
			{0, 1, 1}
		},
		{
			{0, 0, 0},
			{0, 1, 0},
			{1, 1, 1},
			{0, 1, 1}
		},
		// offset = -1, -1, -1
		{
			{0, 0, 0},
			{0, 0, 1},
			{0, 1, 1},
			{1, 1, 1}
		},
		{
			{0, 0, 0},
			{0, 1, 0},
			{0, 1, 1},
			{1, 1, 1}
		},
		{
			{0, 0, 0},
			{0, 0, 1},
			{1, 0, 1},
			{1, 1, 1}
		},
		{
			{0, 0, 0},
			{1, 0, 0},
			{1, 0, 1},
			{1, 1, 1}
		},
		{
			{0, 0, 0},
			{0, 1, 0},
			{1, 1, 0},
			{1, 1, 1}
		},
		{
			{0, 0, 0},
			{1, 0, 0},
			{1, 1, 0},
			{1, 1, 1}
		}
	};
	ptrdiff_t simplex_offset[24];
	{
		ptrdiff_t * simplex_offset_pos = simplex_offset;
		ptrdiff_t base = 0;
		// offset = 0, 0, 0
		for(int i=0; i<6; i++){
			*(simplex_offset_pos++) = i;
		}
		// offset = -1, 0, 0
		base = -6*dim0_offset;
		*(simplex_offset_pos++) = base + 3;
		*(simplex_offset_pos++) = base + 5;
		// offset = 0, -1, 0
		base = -6*dim1_offset;
		*(simplex_offset_pos++) = base + 1;
		*(simplex_offset_pos++) = base + 4;
		// offset = -1, -1, 0
		base = -6*dim0_offset - 6*dim1_offset;
		*(simplex_offset_pos++) = base + 4;
		*(simplex_offset_pos++) = base + 5;
		// offset = 0, 0, -1
		base = -6;
		*(simplex_offset_pos++) = base + 0;
		*(simplex_offset_pos++) = base + 2;
		// offset = -1, 0, -1
		base = -6*dim0_offset - 6;
		*(simplex_offset_pos++) = base + 2;
		*(simplex_offset_pos++) = base + 3;
		// offset = 0, -1, -1
		base = -6*dim1_offset - 6;
		*(simplex_offset_pos++) = base + 0;
		*(simplex_offset_pos++) = base + 1;
		// offset = -1, -1, -1
		base = -6*dim0_offset - 6*dim1_offset - 6;
		for(int i=0; i<6; i++){
			*(simplex_offset_pos++) = base + i;
		}
	}
	int index_offset[24][3][3];
	for(int i=0; i<24; i++){
		for(int j=0; j<3; j++){
			for(int k=0; k<3; k++){
				index_offset[i][j][k] = coordinates[i][j][k] - coordinates[i][3][k];
			}
		}
	}
	ptrdiff_t offset[24][3];
	for(int i=0; i<24; i++){
		for(int x=0; x<3; x++){
			// offset[i][x] = (coordinates[i][x][0] - coordinates[i][3][0]) * dim0_offset + (coordinates[i][x][1] - coordinates[i][3][1]) * dim1_offset + (coordinates[i][x][2] - coordinates[i][3][2]);
			offset[i][x] = (coordinates[i][x][0] - coordinates[i][3][0]) + (coordinates[i][x][1] - coordinates[i][3][1]) * dim1_offset + (coordinates[i][x][2] - coordinates[i][3][2]) * dim0_offset;
		}
	}
	T * cur_log_U_pos = log_U;
	T * cur_log_V_pos = log_V;
	T * cur_log_W_pos = log_W;
	T * cur_U_pos = decompressed_U;
	T * cur_V_pos = decompressed_V;
	T * cur_W_pos = decompressed_W;
	unpred_vec<T> eb_zero_data = unpred_vec<T>();
	ptrdiff_t max_pointer_pos = num_elements;
	double threshold = std::numeric_limits<float>::epsilon();
	int eb_quant_index_max = (int) (log2(1.0 / threshold)/log_of_base) + 1;

	// double * eb = (double *) malloc(sizeof(double)*num_elements);
	// int eb_index = 0;
	// int est_outrange = num_elements * 0.1;
	// unsigned char * outrange_sign = (unsigned char *) malloc(est_outrange);
	// int * outrange_exp = (int *) malloc(est_outrange*sizeof(int));
	// unsigned char * outrange_residue = (unsigned char *) malloc(est_outrange*sizeof(T));
	// unsigned char * outrange_sign_pos = outrange_sign;
	// int * outrange_exp_pos = outrange_exp; 
	// unsigned char * outrange_residue_pos = outrange_residue;
	// int outrange_pos = 0;
	// unpred_vec<float> outrange_data = unpred_vec<float>();
	// record flags
	for(int i=0; i<r1; i++){
		// printf("start %d row\n", i);
		for(int j=0; j<r2; j++){
			for(int k=0; k<r3; k++){
				double required_eb = max_pwr_eb;
				// derive eb given 24 adjacent simplex
				for(int n=0; n<24; n++){
					bool in_mesh = true;
					for(int p=0; p<3; p++){
						// reversed order!
						if(!(in_range(i + index_offset[n][p][2], (int)r1) && in_range(j + index_offset[n][p][1], (int)r2) && in_range(k + index_offset[n][p][0], (int)r3))){
							in_mesh = false;
							break;
						}
					}
					if(in_mesh){
						int index = simplex_offset[n] + i*6*dim0_offset + j*6*dim1_offset + k*6; // TODO: define index for each simplex
						required_eb = MIN(required_eb, max_eb_to_keep_position_and_type_3d_online(
							cur_U_pos[offset[n][0]], cur_U_pos[offset[n][1]], cur_U_pos[offset[n][2]], *cur_U_pos,
							cur_V_pos[offset[n][0]], cur_V_pos[offset[n][1]], cur_V_pos[offset[n][2]], *cur_V_pos,
							cur_W_pos[offset[n][0]], cur_W_pos[offset[n][1]], cur_W_pos[offset[n][2]], *cur_W_pos));
					}
				}
				// eb[eb_index ++] = required_eb;
				if(required_eb < 1e-6) required_eb = 0;
				if(required_eb > 0){
					bool unpred_flag = false;
					T decompressed[3];
					double abs_eb = log2(1 + required_eb);
					*eb_quant_index_pos = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
					if(*eb_quant_index_pos > 0){
						// compress vector fields
						T * log_data_pos[3] = {cur_log_U_pos, cur_log_V_pos, cur_log_W_pos};
						T * data_pos[3] = {cur_U_pos, cur_V_pos, cur_W_pos};
						for(int p=0; p<3; p++){
							T * cur_log_data_pos = log_data_pos[p];
							T cur_data = *cur_log_data_pos;
							// get adjacent data and perform Lorenzo
							/*
								d6	X
								d4	d5
								d2	d3
								d0	d1
							*/
							T d0 = (i && j && k) ? cur_log_data_pos[- dim0_offset - dim1_offset - 1] : 0;
							T d1 = (i && j) ? cur_log_data_pos[- dim0_offset - dim1_offset] : 0;
							T d2 = (i && k) ? cur_log_data_pos[- dim0_offset - 1] : 0;
							T d3 = (i) ? cur_log_data_pos[- dim0_offset] : 0;
							T d4 = (j && k) ? cur_log_data_pos[- dim1_offset - 1] : 0;
							T d5 = (j) ? cur_log_data_pos[- dim1_offset] : 0;
							T d6 = (k) ? cur_log_data_pos[- 1] : 0;
							T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
							double diff = cur_data - pred;
							double quant_diff = fabs(diff) / abs_eb + 1;
							if(quant_diff < capacity){
								quant_diff = (diff > 0) ? quant_diff : -quant_diff;
								int quant_index = (int)(quant_diff/2) + intv_radius;
								data_quant_index_pos[p] = quant_index;
								decompressed[p] = pred + 2 * (quant_index - intv_radius) * abs_eb; 
								// check original data
								if(fabs(decompressed[p] - cur_data) >= abs_eb){
									unpred_flag = true;
									break;
								}
							}
							else{
								unpred_flag = true;
								break;
							}
						}
					}
					else unpred_flag = true;
					if(unpred_flag){
						*(eb_quant_index_pos ++) = eb_quant_index_max;
						// int outrange_residue_len = exp_offset<double>() - getExponent(required_eb) + 2;
						// *cur_U_pos = out_of_range_data_encode(*cur_U_pos, outrange_residue_len, outrange_sign_pos, outrange_exp_pos, outrange_residue_pos, outrange_pos);
						// *cur_V_pos = out_of_range_data_encode(*cur_V_pos, outrange_residue_len, outrange_sign_pos, outrange_exp_pos, outrange_residue_pos, outrange_pos);
						// *cur_W_pos = out_of_range_data_encode(*cur_W_pos, outrange_residue_len, outrange_sign_pos, outrange_exp_pos, outrange_residue_pos, outrange_pos);
						// *cur_log_U_pos = log2(fabs(*cur_U_pos));
						// *cur_log_V_pos = log2(fabs(*cur_V_pos));
						// *cur_log_W_pos = log2(fabs(*cur_W_pos));
						// printf("outrange_residue_len = %d\n", outrange_residue_pos - outrange_residue);

						eb_zero_data.push_back(*cur_U_pos);
						eb_zero_data.push_back(*cur_V_pos);
						eb_zero_data.push_back(*cur_W_pos);
					}
					else{
						eb_quant_index_pos ++;
						data_quant_index_pos += 3;
						*cur_log_U_pos = decompressed[0];
						*cur_log_V_pos = decompressed[1];
						*cur_log_W_pos = decompressed[2];
						*cur_U_pos = (*cur_U_pos > 0) ? exp2(*cur_log_U_pos) : -exp2(*cur_log_U_pos);
						*cur_V_pos = (*cur_V_pos > 0) ? exp2(*cur_log_V_pos) : -exp2(*cur_log_V_pos);
						*cur_W_pos = (*cur_W_pos > 0) ? exp2(*cur_log_W_pos) : -exp2(*cur_log_W_pos);
					}
				}
				else{
					// record as unpredictable data
					*(eb_quant_index_pos ++) = 0;
					eb_zero_data.push_back(*cur_U_pos);
					eb_zero_data.push_back(*cur_V_pos);
					eb_zero_data.push_back(*cur_W_pos);
				}
				cur_log_U_pos ++, cur_log_V_pos ++, cur_log_W_pos ++;
				cur_U_pos ++, cur_V_pos ++, cur_W_pos ++;
			}
		}
	}
	// printf("%d %d\n", eb_index, num_elements);
	// writefile("eb_3d.dat", eb, num_elements);
	// free(eb);
	// if(outrange_pos) outrange_residue_pos ++;
	free(log_U);
	free(log_V);
	free(log_W);
	free(decompressed_U);
	free(decompressed_V);
	free(decompressed_W);
	//printf("offset eb_q, data_q, unpred: %ld %ld %ld\n", eb_quant_index_pos - eb_quant_index, data_quant_index_pos - data_quant_index, eb_zero_data.size());
	unsigned char * compressed = (unsigned char *) malloc(3*num_elements*sizeof(T));
	unsigned char * compressed_pos = compressed;
	//开始写入
	//先写bitmap
	convertIntArray2ByteArray_fast_1b_to_result_sz(bitmap, num_elements, compressed_pos);
	//printf("bitmap size = %ld\n", num_bytes);
	//再写入lossless数据的大小
	write_variable_to_dst(compressed_pos, index_need_to_lossless.size());
	//再写入lossless数据
	for (auto it = index_need_to_lossless.begin(); it != index_need_to_lossless.end(); it++){
		write_variable_to_dst(compressed_pos, U[*it]);
	}
	for (auto it = index_need_to_lossless.begin(); it != index_need_to_lossless.end(); it++){
		write_variable_to_dst(compressed_pos, V[*it]);
	}
	for (auto it = index_need_to_lossless.begin(); it != index_need_to_lossless.end(); it++){
		write_variable_to_dst(compressed_pos, W[*it]);
	}
	//printf("index_need_to_lossless lossless data pos = %ld\n", compressed_pos - compressed);

	write_variable_to_dst(compressed_pos, base);
	write_variable_to_dst(compressed_pos, intv_radius);
	write_array_to_dst(compressed_pos, sign_map_compressed, 3*sign_map_size);
	free(sign_map_compressed);
	size_t unpredictable_count = eb_zero_data.size();
	write_variable_to_dst(compressed_pos, unpredictable_count);
	write_array_to_dst(compressed_pos, (T *)&eb_zero_data[0], unpredictable_count);	
	//printf("eb_zero_data size = %ld\n", unpredictable_count*sizeof(T));
	// store out range information
	unsigned char * tmp = compressed_pos;
	// size_t outrange_count = outrange_sign_pos - outrange_sign;
	// write_variable_to_dst(compressed_pos, outrange_count);
	// convertIntArray2ByteArray_fast_1b_to_result_sz(outrange_sign, outrange_count, compressed_pos);
	// unsigned char * tmp2 = compressed_pos;
	// Huffman_encode_tree_and_data(2*(exp_offset<T>() + 1), outrange_exp, outrange_count, compressed_pos);
	// unsigned char * tmp3 = compressed_pos;
	// write_array_to_dst(compressed_pos, outrange_residue, outrange_residue_pos - outrange_residue);
	// printf("outrange count = %ld, outrange_exp_size = %ld, outrange_residue_size = %ld\n", outrange_count, tmp3 - tmp2, outrange_residue_pos - outrange_residue);
	// printf("outrange size = %ld\n", compressed_pos - tmp);
	// free(outrange_sign);
	// free(outrange_exp);
	// free(outrange_residue);
	tmp = compressed_pos;
	size_t eb_quant_num = eb_quant_index_pos - eb_quant_index;
	write_variable_to_dst(compressed_pos, eb_quant_num);
	Huffman_encode_tree_and_data(2*256, eb_quant_index, num_elements, compressed_pos);
	//printf("eb_quant_index size = %ld\n", compressed_pos - tmp);
	free(eb_quant_index);
	tmp = compressed_pos;
	size_t data_quant_num = data_quant_index_pos - data_quant_index;
	write_variable_to_dst(compressed_pos, data_quant_num);
	Huffman_encode_tree_and_data(2*capacity, data_quant_index, data_quant_num, compressed_pos);
	//printf("data_quant_index size = %ld\n", compressed_pos - tmp);
	free(data_quant_index);
	compressed_size = compressed_pos - compressed;
	return compressed;	
}


template
unsigned char *
sz_compress_cp_preserve_3d_record_vertex(const float * U, const float * V, const float * W, size_t r1, size_t r2, size_t r3, size_t& compressed_size, bool transpose, double max_pwr_eb,const std::set<size_t>& index_need_to_lossless);

/* for 3d abs error bound*/

// compute offsets for simplex, index, and positions
static const int coordinates[24][4][3] = {
	// offset = 0, 0, 0
	{
		{0, 0, 1},
		{0, 1, 1},
		{1, 1, 1},
		{0, 0, 0}
	},
	{
		{0, 1, 0},
		{0, 1, 1},
		{1, 1, 1},
		{0, 0, 0}
	},
	{
		{0, 0, 1},
		{1, 0, 1},
		{1, 1, 1},
		{0, 0, 0}
	},
	{
		{1, 0, 0},
		{1, 0, 1},
		{1, 1, 1},
		{0, 0, 0}
	},
	{
		{0, 1, 0},
		{1, 1, 0},
		{1, 1, 1},
		{0, 0, 0}
	},
	{
		{1, 0, 0},
		{1, 1, 0},
		{1, 1, 1},
		{0, 0, 0}
	},
	// offset = -1, 0, 0
	{
		{0, 0, 0},
		{1, 0, 1},
		{1, 1, 1},
		{1, 0, 0}
	},
	{
		{0, 0, 0},
		{1, 1, 0},
		{1, 1, 1},
		{1, 0, 0}
	},
	// offset = 0, -1, 0
	{
		{0, 0, 0},
		{0, 1, 1},
		{1, 1, 1},
		{0, 1, 0}
	},
	{
		{0, 0, 0},
		{1, 1, 0},
		{1, 1, 1},
		{0, 1, 0}
	},
	// offset = -1, -1, 0
	{
		{0, 0, 0},
		{0, 1, 0},
		{1, 1, 1},
		{1, 1, 0}
	},
	{
		{0, 0, 0},
		{1, 0, 0},
		{1, 1, 1},
		{1, 1, 0}
	},
	// offset = 0, 0, -1
	{
		{0, 0, 0},
		{0, 1, 1},
		{1, 1, 1},
		{0, 0, 1}
	},
	{
		{0, 0, 0},
		{1, 0, 1},
		{1, 1, 1},
		{0, 0, 1}
	},
	// offset = -1, 0, -1
	{
		{0, 0, 0},
		{0, 0, 1},
		{1, 1, 1},
		{1, 0, 1}
	},
	{
		{0, 0, 0},
		{1, 0, 0},
		{1, 1, 1},
		{1, 0, 1}
	},
	// offset = 0, -1, -1
	{
		{0, 0, 0},
		{0, 0, 1},
		{1, 1, 1},
		{0, 1, 1}
	},
	{
		{0, 0, 0},
		{0, 1, 0},
		{1, 1, 1},
		{0, 1, 1}
	},
	// offset = -1, -1, -1
	{
		{0, 0, 0},
		{0, 0, 1},
		{0, 1, 1},
		{1, 1, 1}
	},
	{
		{0, 0, 0},
		{0, 1, 0},
		{0, 1, 1},
		{1, 1, 1}
	},
	{
		{0, 0, 0},
		{0, 0, 1},
		{1, 0, 1},
		{1, 1, 1}
	},
	{
		{0, 0, 0},
		{1, 0, 0},
		{1, 0, 1},
		{1, 1, 1}
	},
	{
		{0, 0, 0},
		{0, 1, 0},
		{1, 1, 0},
		{1, 1, 1}
	},
	{
		{0, 0, 0},
		{1, 0, 0},
		{1, 1, 0},
		{1, 1, 1}
	}
};

// default coordinates for tets in a cell
static const double default_coords[6][4][3] = {
  {
    {0, 0, 0},
    {0, 0, 1},
    {0, 1, 1},
    {1, 1, 1}
  },
  {
    {0, 0, 0},
    {0, 1, 0},
    {0, 1, 1},
    {1, 1, 1}
  },
  {
    {0, 0, 0},
    {0, 0, 1},
    {1, 0, 1},
    {1, 1, 1}
  },
  {
    {0, 0, 0},
    {1, 0, 0},
    {1, 0, 1},
    {1, 1, 1}
  },
  {
    {0, 0, 0},
    {0, 1, 0},
    {1, 1, 0},
    {1, 1, 1}
  },
  {
    {0, 0, 0},
    {1, 0, 0},
    {1, 1, 0},
    {1, 1, 1}
  },
};

static void 
compute_offset(ptrdiff_t dim0_offset, ptrdiff_t dim1_offset, ptrdiff_t cell_dim0_offset, ptrdiff_t cell_dim1_offset,
				int simplex_offset[24], int index_offset[24][3][3], int offset[24][3]){
	int * simplex_offset_pos = simplex_offset;
	ptrdiff_t base = 0;
	// offset = 0, 0, 0
	for(int i=0; i<6; i++){
		*(simplex_offset_pos++) = i;
	}
	// offset = -1, 0, 0
	base = -6;
	*(simplex_offset_pos++) = base + 3;
	*(simplex_offset_pos++) = base + 5;
	// offset = 0, -1, 0
	base = -6*cell_dim1_offset;
	*(simplex_offset_pos++) = base + 1;
	*(simplex_offset_pos++) = base + 4;
	// offset = -1, -1, 0
	base = -6 - 6*cell_dim1_offset;
	*(simplex_offset_pos++) = base + 4;
	*(simplex_offset_pos++) = base + 5;
	// offset = 0, 0, -1
	base = -6*cell_dim0_offset;
	*(simplex_offset_pos++) = base + 0;
	*(simplex_offset_pos++) = base + 2;
	// offset = -1, 0, -1
	base = -6*cell_dim0_offset - 6;
	*(simplex_offset_pos++) = base + 2;
	*(simplex_offset_pos++) = base + 3;
	// offset = 0, -1, -1
	base = -6*cell_dim1_offset - 6*cell_dim0_offset;
	*(simplex_offset_pos++) = base + 0;
	*(simplex_offset_pos++) = base + 1;
	// offset = -1, -1, -1
	base = -6*cell_dim0_offset - 6*cell_dim1_offset - 6;
	for(int i=0; i<6; i++){
		*(simplex_offset_pos++) = base + i;
	}
	for(int i=0; i<24; i++){
		for(int j=0; j<3; j++){
			for(int k=0; k<3; k++){
				index_offset[i][j][k] = coordinates[i][j][k] - coordinates[i][3][k];
			}
		}
	}
	for(int i=0; i<24; i++){
		for(int x=0; x<3; x++){
			offset[i][x] = (coordinates[i][x][0] - coordinates[i][3][0]) + (coordinates[i][x][1] - coordinates[i][3][1]) * dim1_offset + (coordinates[i][x][2] - coordinates[i][3][2]) * dim0_offset;
		}
	}	
}

// maximal absolute error bound to keep the sign of A*e_1 + B*e_2 + C*e_3 + D
template<typename T>
inline double max_eb_to_keep_sign_3d_online_abs(const T A, const T B, const T C, const T D=0){
	if(fabs(A) + fabs(B) + fabs(C) == 0){
		if(D == 0) return 0;
		return 1e9;
	}
	return fabs(D) / (fabs(A) + fabs(B) + fabs(C));
}
template<typename T>
double 
max_eb_to_keep_position_and_type_3d_online_abs(const T u0, const T u1, const T u2, const T u3, const T v0, const T v1, const T v2, const T v3,
	const T w0, const T w1, const T w2, const T w3){
	// double u3_0 = - u3*v1*w2 + u3*v2*w1, u3_1 = - u3*v2*w0 + u3*v0*w2, u3_2 = - u3*v0*w1 + u3*v1*w0;
	// double v3_0 = u1*v3*w2 - u2*v3*w1, v3_1 = u2*v3*w0 - u0*v3*w2, v3_2 = u0*v3*w1 - u1*v3*w0;
	// double w3_0 = - u1*v2*w3 + u2*v1*w3, w3_1 = u0*v2*w3 - u2*v0*w3, w3_2 = - u0*v1*w3 + u1*v0*w3;
	double u3_0 = - v1*w2 + v2*w1, u3_1 = - v2*w0 + v0*w2, u3_2 = - v0*w1 + v1*w0;
	double v3_0 = u1*w2 - u2*w1, v3_1 = u2*w0 - u0*w2, v3_2 = u0*w1 - u1*w0;
	double w3_0 = - u1*v2 + u2*v1, w3_1 = u0*v2 - u2*v0, w3_2 = - u0*v1 + u1*v0;
	double c_4 = u0*v1*w2 - u0*v2*w1 + u1*v2*w0 - u1*v0*w2 + u2*v0*w1 - u2*v1*w0;
	double M0 = u3_0*u3 + v3_0*v3 + w3_0*w3;
	double M1 = u3_1*u3 + v3_1*v3 + w3_1*w3;
	double M2 = u3_2*u3 + v3_2*v3 + w3_2*w3;
	double M3 = c_4;
	double M = M0 + M1 + M2 + M3;
	if(M == 0){
		return 0;
	}
	bool flag[4];
	flag[0] = (M0 == 0) || (M / M0 > 1);
	flag[1] = (M1 == 0) || (M / M1 > 1);
	flag[2] = (M2 == 0) || (M / M2 > 1);
	flag[3] = (M3 == 0) || (M / M3 > 1);
	if(flag[0] && flag[1] && flag[2] && flag[3]){
		// cp found
		return 0;
	}
	else{
		double eb = 0;
		double cur_eb = 0;
		if(!flag[0]){
			cur_eb = MIN(max_eb_to_keep_sign_3d_online_abs(u3_0, v3_0, w3_0, M0), 
					max_eb_to_keep_sign_3d_online_abs(u3_1 + u3_2, v3_1 + v3_2, w3_1 + w3_2, M1 + M2 + M3));
			eb = MAX(eb, cur_eb);
		}
		if(!flag[1]){
			cur_eb = MIN(max_eb_to_keep_sign_3d_online_abs(u3_1, v3_1, w3_1, M1), 
					max_eb_to_keep_sign_3d_online_abs(u3_0 + u3_2, v3_0 + v3_2, w3_0 + w3_2, M0 + M2 + M3));
			eb = MAX(eb, cur_eb);
		}
		if(!flag[2]){
			cur_eb = MIN(max_eb_to_keep_sign_3d_online_abs(u3_2, v3_2, w3_2, M2), 
					max_eb_to_keep_sign_3d_online_abs(u3_0 + u3_1, v3_0 + v3_1, w3_0 + w3_1, M0 + M1 + M3));
			eb = MAX(eb, cur_eb);
		}
		if(!flag[3]){
			cur_eb = max_eb_to_keep_sign_3d_online_abs(u3_0 + u3_1 + u3_2, v3_0 + v3_1 + v3_2, w3_0 + w3_1 + w3_2, M0 + M1 + M2);
			eb = MAX(eb, cur_eb);
		}
		return eb;
	}
}

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
template static int check_cp(double v[4][3]);

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
	for(int i=1; i<r1-2; i++){
		if(i%10==0) std::cout << i << " / " << r1-1 << std::endl;
		for(int j=1; j<r2-2; j++){
			for(int k=1; k<r3-2; k++){
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

template static vector<bool> compute_cp(const float * U, const float * V, const float * W, int r1, int r2, int r3);
template static vector<bool> compute_cp(const double * U, const double * V, const double * W, int r1, int r2, int r3);

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
    for(int i = 1; i < r1 - 2; i++){
        for(int j = 1; j < r2 - 2; j++){
            for(int k = 1; k < r3 - 2; k++){
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

template std::vector<bool> omp_compute_cp(const float * U, const float * V, const float * W, int r1, int r2, int r3);
template std::vector<bool> omp_compute_cp(const double * U, const double * V, const double * W, int r1, int r2, int r3);

template<typename T>
inline bool in_local_range(T pos, T n){
	return (pos >= 0) && (pos < n);
}

template<typename T>
unsigned char *
sz_compress_cp_preserve_3d_online_abs_record_vertex(const T * U, const T * V, const T * W, size_t r1, size_t r2, size_t r3, size_t& compressed_size, double max_abs_eb,const std::set<size_t>& index_need_to_lossless){
	size_t num_elements = r1 * r2 * r3;
	size_t intArrayLength = num_elements;
	size_t num_bytes = (intArrayLength % 8 == 0) ? intArrayLength / 8 : intArrayLength / 8 + 1;
	unsigned char * bitmap;
	if(index_need_to_lossless.size() != 0){
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
		//set index_need_to_lossless to 1
		for(auto it = index_need_to_lossless.begin(); it != index_need_to_lossless.end(); ++it){
			assert(*it < num_elements);
			bitmap[*it] = 1;
		}
		//准备bitmap#####################
	}
	T * decompressed_U = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_U, U, num_elements*sizeof(T));
	T * decompressed_V = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_V, V, num_elements*sizeof(T));
	T * decompressed_W = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_W, W, num_elements*sizeof(T));
	int * eb_quant_index = (int *) malloc(num_elements*sizeof(int));
	int * data_quant_index = (int *) malloc(3*num_elements*sizeof(int));
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	// next, row by row
	const int base = 4;
	const double log_of_base = log2(base);
	const int capacity = 65536;
	const int intv_radius = (capacity >> 1);

	unpred_vec<T> unpred_data = unpred_vec<T>();
	T * cur_U_pos = decompressed_U;
	T * cur_V_pos = decompressed_V;
	T * cur_W_pos = decompressed_W;

	ptrdiff_t dim0_offset = r2 * r3;
	ptrdiff_t dim1_offset = r3;
	ptrdiff_t cell_dim0_offset = (r2-1) * (r3-1);
	ptrdiff_t cell_dim1_offset = r3-1;
	// offsets to get 24 adjacent simplex indices
	// x -> z, high -> low
	// current data would always be the last index, i.e. x[i][3]
	int simplex_offset[24];
	int index_offset[24][3][3];
	int offset[24][3];
	compute_offset(dim0_offset, dim1_offset, cell_dim0_offset, cell_dim1_offset, simplex_offset, index_offset, offset);

	std::cout << "start cp checking\n";
	std::cout << "max_abs_eb = " << max_abs_eb << std::endl;
	vector<bool> cp_exist = compute_cp(U, V, W, r1, r2, r3);

	double threshold = std::numeric_limits<double>::epsilon();
	for(int i=0; i<r1; i++){
		// printf("start %d row\n", i);
		for(int j=0; j<r2; j++){
			for(int k=0; k<r3; k++){
				double required_eb = max_abs_eb;
				if((*cur_U_pos == 0) || (*cur_V_pos == 0) || (*cur_W_pos == 0)) required_eb = 0;
				if(required_eb){
					// derive eb given 24 adjacent simplex
					for(int n=0; n<24; n++){
						bool in_mesh = true;
						for(int p=0; p<3; p++){
							// reversed order!
							if(!(in_range(i + index_offset[n][p][2], (int)r1) && in_range(j + index_offset[n][p][1], (int)r2) && in_range(k + index_offset[n][p][0], (int)r3))){
								in_mesh = false;
								break;
							}
						}
						if(in_mesh){
							int index = simplex_offset[n] + 6*(i*(r2-1)*(r3-1) + j*(r3-1) + k);
							if(cp_exist[index]){
								required_eb = 0;
								break;
							}
							else{
								// std::cout << required_eb << " " << max_abs_eb << std::endl;
								required_eb = MIN(required_eb, max_eb_to_keep_position_and_type_3d_online_abs(
									cur_U_pos[offset[n][0]], cur_U_pos[offset[n][1]], cur_U_pos[offset[n][2]], *cur_U_pos,
									cur_V_pos[offset[n][0]], cur_V_pos[offset[n][1]], cur_V_pos[offset[n][2]], *cur_V_pos,
									cur_W_pos[offset[n][0]], cur_W_pos[offset[n][1]], cur_W_pos[offset[n][2]], *cur_W_pos));
							}
						}
					}					
				}
				if(required_eb){
					bool unpred_flag = false;
					T decompressed[3];
					double abs_eb = required_eb;
					*eb_quant_index_pos = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
					if(*eb_quant_index_pos > 0){
						// compress vector fields
						T * data_pos[3] = {cur_U_pos, cur_V_pos, cur_W_pos};
						for(int p=0; p<3; p++){
							T * cur_data_pos = data_pos[p];
							T cur_data = *cur_data_pos;
							// get adjacent data and perform Lorenzo
							/*
								d6	X
								d4	d5
								d2	d3
								d0	d1
							*/
							T d0 = (i && j && k) ? cur_data_pos[- dim0_offset - dim1_offset - 1] : 0;
							T d1 = (i && j) ? cur_data_pos[- dim0_offset - dim1_offset] : 0;
							T d2 = (i && k) ? cur_data_pos[- dim0_offset - 1] : 0;
							T d3 = (i) ? cur_data_pos[- dim0_offset] : 0;
							T d4 = (j && k) ? cur_data_pos[- dim1_offset - 1] : 0;
							T d5 = (j) ? cur_data_pos[- dim1_offset] : 0;
							T d6 = (k) ? cur_data_pos[- 1] : 0;
							T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
							double diff = cur_data - pred;
							double quant_diff = fabs(diff) / abs_eb + 1;
							if(quant_diff < capacity){
								quant_diff = (diff > 0) ? quant_diff : -quant_diff;
								int quant_index = (int)(quant_diff/2) + intv_radius;
								data_quant_index_pos[p] = quant_index;
								decompressed[p] = pred + 2 * (quant_index - intv_radius) * abs_eb; 
								// check original data
								if(fabs(decompressed[p] - cur_data) >= abs_eb){
									unpred_flag = true;
									break;
								}
							}
							else{
								unpred_flag = true;
								break;
							}
						}
					}
					else unpred_flag = true;
					if(unpred_flag){
						*(eb_quant_index_pos ++) = 0;
						unpred_data.push_back(*cur_U_pos);
						unpred_data.push_back(*cur_V_pos);
						unpred_data.push_back(*cur_W_pos);
					}
					else{
						eb_quant_index_pos ++;
						data_quant_index_pos += 3;
						*cur_U_pos = decompressed[0];
						*cur_V_pos = decompressed[1];
						*cur_W_pos = decompressed[2];
					}
				}
				else{
					// record as unpredictable data
					*(eb_quant_index_pos ++) = 0;
					unpred_data.push_back(*cur_U_pos);
					unpred_data.push_back(*cur_V_pos);
					unpred_data.push_back(*cur_W_pos);
				}
				cur_U_pos ++, cur_V_pos ++, cur_W_pos ++;
			}
		}
	}	
	free(decompressed_U);
	free(decompressed_V);
	free(decompressed_W);
	printf("offsets eb_q, data_q, unpred: %ld %ld %ld\n", eb_quant_index_pos - eb_quant_index, data_quant_index_pos - data_quant_index, unpred_data.size());
	unsigned char * compressed = (unsigned char *) malloc(3*num_elements*sizeof(T));
	unsigned char * compressed_pos = compressed;
	//先写index_need_to_lossless的大小,不管怎样
	write_variable_to_dst(compressed_pos, index_need_to_lossless.size()); //size_t, index_need_to_lossless的大小
	printf("index_need_to_lossless pos = %ld\n", compressed_pos - compressed);

	//如果index_need_to_lossless.size() != 0，那么写bitmap
	if(index_need_to_lossless.size() != 0){
		//修改：再写bitmap
		// write_variable_to_dst(compressed_pos,num_elements); // 处理后的bitmap的长度 size_t
		//write_array_to_dst(compressed_pos, compressedArray, num_bytes);
		convertIntArray2ByteArray_fast_1b_to_result_sz(bitmap, num_elements, compressed_pos);
		printf("bitmap pos = %ld\n", compressed_pos - compressed);

		//再写index_need_to_lossless对应U和V的数据
		for (auto it = index_need_to_lossless.begin(); it != index_need_to_lossless.end(); it++){
			write_variable_to_dst(compressed_pos, U[*it]); //T, index_need_to_lossless对应的U的值
		}
		for (auto it = index_need_to_lossless.begin(); it != index_need_to_lossless.end(); it++){
			write_variable_to_dst(compressed_pos, V[*it]); //T, index_need_to_lossless对应的V的值
		}
		for (auto it = index_need_to_lossless.begin(); it != index_need_to_lossless.end(); it++){
			write_variable_to_dst(compressed_pos, W[*it]); //T, index_need_to_lossless对应的W的值
		}
		printf("index_need_to_lossless data pos = %ld\n", compressed_pos - compressed);
	}
	//如果index_need_to_lossless.size() = 0 ，那么直接写开始写base
	write_variable_to_dst(compressed_pos, base);
	write_variable_to_dst(compressed_pos, threshold);
	write_variable_to_dst(compressed_pos, intv_radius);
	size_t data_quant_num = data_quant_index_pos - data_quant_index;
	write_variable_to_dst(compressed_pos, data_quant_num);
	size_t unpredictable_count = unpred_data.size();
	write_variable_to_dst(compressed_pos, unpredictable_count);
	printf("start of unpred data: pos = %ld\n", compressed_pos - compressed);
	write_array_to_dst(compressed_pos, (T *)&unpred_data[0], unpredictable_count);	
	printf("start eb decoding: pos = %ld\n", compressed_pos - compressed);
	Huffman_encode_tree_and_data(2*1024, eb_quant_index, num_elements, compressed_pos);
	free(eb_quant_index);
	printf("start data decoding: pos = %ld\n", compressed_pos - compressed);
	Huffman_encode_tree_and_data(2*capacity, data_quant_index, data_quant_num, compressed_pos);
	printf("pos = %ld\n", compressed_pos - compressed);
	free(data_quant_index);
	compressed_size = compressed_pos - compressed;
	return compressed;	
}

template
unsigned char *
sz_compress_cp_preserve_3d_online_abs_record_vertex(const float * U, const float * V, const float * W, size_t r1, size_t r2, size_t r3, size_t& compressed_size, double max_abs_eb,const std::set<size_t>& index_need_to_lossless);

template
unsigned char *
sz_compress_cp_preserve_3d_online_abs_record_vertex(const double * U, const double * V, const double * W, size_t r1, size_t r2, size_t r3, size_t& compressed_size, double max_abs_eb,const std::set<size_t>& index_need_to_lossless);


template <typename T>
unsigned char * omp_sz_compress_cp_preserve_3d_online_abs_record_vertex(
    const T * U, const T * V, const T * W, size_t r1, size_t r2, size_t r3, 
    size_t& compressed_size, double max_eb, const std::set<size_t>& index_need_to_lossless, 
    int n_threads, T* &decompressed_U_ptr, T* &decompressed_V_ptr, T* &decompressed_W_ptr,std::vector<bool> &cp_exist)
{
	auto non_parallel_memcpy_etc = std::chrono::high_resolution_clock::now();
	size_t num_elements = r1 * r2 * r3;
	size_t intArrayLength = num_elements;
	size_t num_bytes = (intArrayLength % 8 == 0) ? intArrayLength / 8 : intArrayLength / 8 + 1;
	unsigned char * bitmap;
	if(index_need_to_lossless.size() != 0){
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
		//set index_need_to_lossless to 1
		for(auto it = index_need_to_lossless.begin(); it != index_need_to_lossless.end(); ++it){
			assert(*it < num_elements);
			bitmap[*it] = 1;
		}
		//准备bitmap#####################
	}
	//确定线程是不是立方数
	int num_threads = n_threads;
	int cube_root = std::round(std::cbrt(num_threads));
    if (cube_root * cube_root * cube_root != num_threads) {
        printf("The number of threads must be a cube of an integer!\n");
        printf("num thread :%d\n", num_threads);
        exit(0);
    }
	T * decompressed_U = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_U, U, num_elements*sizeof(T));
	T * decompressed_V = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_V, V, num_elements*sizeof(T));
	T * decompressed_W = (T *) malloc(num_elements*sizeof(T));
	memcpy(decompressed_W, W, num_elements*sizeof(T));

	int * eb_quant_index = (int *) malloc(num_elements*sizeof(int));
	int * data_quant_index = (int *) malloc(3*num_elements*sizeof(int));
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	// next, row by row
	const int base = 4;
	const double log_of_base = log2(base);
	const int capacity = 65536;
	const int intv_radius = (capacity >> 1);
	unpred_vec<T> unpred_data;
	unpred_vec<T> unpred_data_dividing;
	std::vector<unpred_vec<T>> unpred_data_thread(num_threads);
    // std::vector<T> corner_points;
	auto non_parallel_memcpy_etc_end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> non_parallel_memcpy_etc_duration = non_parallel_memcpy_etc_end - non_parallel_memcpy_etc;
	std::cout << "Non-parallel memcpy and etc time: " << non_parallel_memcpy_etc_duration.count() << "s" << std::endl;

	auto non_parallel_start = std::chrono::high_resolution_clock::now();
	
	//变化最快的是r3（最内层），最慢的是r1（最外层）
	ptrdiff_t dim0_offset = r2 * r3;
	ptrdiff_t dim1_offset = r3;
	ptrdiff_t cell_dim0_offset = (r2-1) * (r3-1);
	ptrdiff_t cell_dim1_offset = r3-1;

	int simplex_offset[24];
	int index_offset[24][3][3];
	int offset[24][3];
	compute_offset(dim0_offset, dim1_offset, cell_dim0_offset, cell_dim1_offset, simplex_offset, index_offset, offset);
	//此时改变了simplex_offset，index_offset，offset

	int t = cube_root;
	// 计算每个块的大小
	int block_r1 = r1 / t;
	int block_r2 = r2 / t;
	int block_r3 = r3 / t;

	// 处理余数情况（如果 r1、r2 或 r3 不能被 t 整除）
	int remaining_r1 = r1 % t;
	int remaining_r2 = r2 % t;
	int remaining_r3 = r3 % t;

	//储存划分平面的位置
	std::vector<int> dividing_r1;
    std::vector<int> dividing_r2;
    std::vector<int> dividing_r3;
    for (int i = 1; i < t; ++i) {
        dividing_r1.push_back(i * block_r1);
        dividing_r2.push_back(i * block_r2);
        dividing_r3.push_back(i * block_r3);
    }
	//print dividing_r1, dividing_r2, dividing_r3
	// for (int i = 0; i < dividing_r1.size(); i++) {
	// 	printf("comp dividing_r1[%d] = %d\n", i, dividing_r1[i]);
	// }

	// 创建一个一维标记数组，标记哪些数据点位于划分线上
	std::vector<bool> on_dividing_line(num_elements, false);
	/*
	for (int i = 0; i < r1; ++i) {
		for (int j = 0; j < r2; ++j) {
			for (int k = 0; k < r3; ++k) {
				if (std::find(dividing_r1.begin(), dividing_r1.end(), i) != dividing_r1.end() ||
					std::find(dividing_r2.begin(), dividing_r2.end(), j) != dividing_r2.end() ||
					std::find(dividing_r3.begin(), dividing_r3.end(), k) != dividing_r3.end()) {
					on_dividing_line[i * r2 * r3 + j * r3 + k] = true;
				}
			}
		}
	}
	*/
	//优化后
	std::vector<bool> is_dividing_r1(r1, false);
	std::vector<bool> is_dividing_r2(r2, false);
	std::vector<bool> is_dividing_r3(r3, false);
	for (int i = 0; i < dividing_r1.size(); ++i) {
		is_dividing_r1[dividing_r1[i]] = true;
	}
	for (int i = 0; i < dividing_r2.size(); ++i) {
		is_dividing_r2[dividing_r2[i]] = true;
	}
	for (int i = 0; i < dividing_r3.size(); ++i) {
		is_dividing_r3[dividing_r3[i]] = true;
	}
	#pragma omp parallel for collapse(3)
	for (int i = 0; i < r1; ++i) {
		for (int j = 0; j < r2; ++j) {
			for (int k = 0; k < r3; ++k) {
				if (is_dividing_r1[i] || is_dividing_r2[j] || is_dividing_r3[k]) {
					on_dividing_line[i * r2 * r3 + j * r3 + k] = true;
				}
			}
		}
	}

	//总数据块数
	int total_blocks = t * t * t;

	auto non_parallel_end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> non_parallel_duration = non_parallel_end - non_parallel_start;
	std::cout << "Non-parallel preparation time: " << non_parallel_duration.count() << "s" << std::endl;

	//以下用来检查是不是切分方法能够遍历所有数据点
	/*
	//统计在划分线上的数据点数
	int num_dividing_points = 0;
	for (int i = 0; i < r1; ++i) {
		for (int j = 0; j < r2; ++j) {
			for (int k = 0; k < r3; ++k) {
				if (on_dividing_line[i * r2 * r3 + j * r3 + k]) {
					num_dividing_points++;
				}
			}
		}
	}
	// 统计块内的数据数量
	int num_points_in_block = 0;
    #pragma omp parallel for num_threads(num_threads) reduction(+:num_points_in_block)
	for (int block_id = 0; block_id < total_blocks; ++block_id){
		int block_i = block_id / (t * t);
		int block_j = (block_id % (t * t)) / t;
		int block_k = block_id % t;
		// 计算块的起始位置
		int start_i = block_i * block_r1;
		int start_j = block_j * block_r2;
		int start_k = block_k * block_r3;

		int end_i = start_i + block_r1;
		int end_j = start_j + block_r2;
		int end_k = start_k + block_r3;
		if (block_i == t - 1) {
			end_i += remaining_r1;
		}
		if (block_j == t - 1) {
			end_j += remaining_r2;
		}
		if (block_k == t - 1) {
			end_k += remaining_r3;
		}
		for (int i = start_i; i < end_i; ++i) {
			if (std::find(dividing_r1.begin(), dividing_r1.end(), i) != dividing_r1.end()) {
				continue;
			}
			for (int j = start_j; j < end_j; ++j) {
				if (std::find(dividing_r2.begin(), dividing_r2.end(), j) != dividing_r2.end()) {
					continue;
				}
				for (int k = start_k; k < end_k; ++k) {
					if (std::find(dividing_r3.begin(), dividing_r3.end(), k) != dividing_r3.end()) {
						continue;
					}
					num_points_in_block++;
				}
			}
		}
	}
	printf("num_points_in_block %d, num_dividing_points %d, sum %d\n", num_points_in_block, num_dividing_points, num_points_in_block + num_dividing_points);
	printf("total points %d\n", r1 * r2 * r3);
	*/

	// pre-compute cp use omp
	// vector<bool> cp_exist = omp_compute_cp(U, V, W, r1, r2, r3);
	double threshold = std::numeric_limits<double>::epsilon(); 

	size_t processed_data_count = 0;
	std::vector<bool> processed_data_flag(num_elements, false);

	

	auto block_start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for num_threads(num_threads) reduction(+:processed_data_count)
	for (int block_id = 0; block_id < total_blocks; ++block_id){
		int block_i = block_id / (t * t);
		int block_j = (block_id % (t * t)) / t;
		int block_k = block_id % t;
		// 计算块的起始位置
		int start_i = block_i * block_r1;
		int start_j = block_j * block_r2;
		int start_k = block_k * block_r3;

		int end_i = start_i + block_r1;
		int end_j = start_j + block_r2;
		int end_k = start_k + block_r3;
		if (block_i == t - 1) {
			end_i += remaining_r1;
		}
		if (block_j == t - 1) {
			end_j += remaining_r2;
		}
		if (block_k == t - 1) {
			end_k += remaining_r3;
		}
		for (int i = start_i; i < end_i; ++i) {
			if (std::find(dividing_r1.begin(), dividing_r1.end(), i) != dividing_r1.end()) { //dividing_r1 = [128,256,384...]
				continue;
			}
			for (int j = start_j; j < end_j; ++j) {
				if (std::find(dividing_r2.begin(), dividing_r2.end(), j) != dividing_r2.end()) {
					continue;
				}
				for (int k = start_k; k < end_k; ++k) {
					if (std::find(dividing_r3.begin(), dividing_r3.end(), k) != dividing_r3.end()) {
						continue;
					}
					
					// 开始处理块内数据
					processed_data_count++;
					size_t position_idx = i * r2 * r3 + j * r3 + k;
					double required_eb = max_eb;
					if ((decompressed_U[position_idx] == 0) || (decompressed_V[position_idx] == 0) || (decompressed_W[position_idx] == 0)) {
						required_eb = 0;
					}
					if (required_eb) {
						// derive eb given 24 adjacent simplex
						for (int n=0; n<24;n++){
							bool in_mesh = true;
							for (int p=0; p<3; p++){
								// reversed order!
								if (!(in_range(i + index_offset[n][p][2], (int)r1) && in_range(j + index_offset[n][p][1], (int)r2) && in_range(k + index_offset[n][p][0], (int)r3))) {
									in_mesh = false;
									break;
								}
							}
							if (in_mesh) {
								int index = simplex_offset[n] + 6 * (i * (r2 - 1) * (r3 - 1) + j * (r3 - 1) + k);
								if (cp_exist[index]) {
									required_eb = 0;
									break;
								}
								else {
									required_eb = MIN(required_eb, max_eb_to_keep_position_and_type_3d_online_abs(
										decompressed_U[position_idx +offset[n][0]], decompressed_U[position_idx +offset[n][1]], decompressed_U[position_idx +offset[n][2]], decompressed_U[position_idx],
										decompressed_V[position_idx +offset[n][0]], decompressed_V[position_idx +offset[n][1]], decompressed_V[position_idx +offset[n][2]], decompressed_V[position_idx],
										decompressed_W[position_idx +offset[n][0]], decompressed_W[position_idx +offset[n][1]], decompressed_W[position_idx +offset[n][2]], decompressed_W[position_idx]));
								}
							}
						}
					}
					
					if(required_eb){
						bool unpred_flag = false;
						T decompressed[3];
						double abs_eb = required_eb;
						eb_quant_index[position_idx] = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
						if(eb_quant_index[position_idx] > 0){
							// compress vector fields
							
							for(int p=0; p<3; p++){
								T * cur_data_field = (p == 0) ? decompressed_U : (p == 1) ? decompressed_V : decompressed_W;
								T d0 = ((i && j && k) && (i - start_i != 1 && j - start_j != 1 && k - start_k != 1)) ? cur_data_field[position_idx - dim0_offset - dim1_offset - 1] : 0;
								T d1 = ((i && j) && (i - start_i != 1 && j - start_j != 1)) ? cur_data_field[position_idx - dim0_offset - dim1_offset] : 0;
								T d2 = ((i && k) && (i - start_i != 1 && k - start_k != 1)) ? cur_data_field[position_idx - dim0_offset - 1] : 0;
								T d3 = (i && (i - start_i != 1)) ? cur_data_field[position_idx - dim0_offset] : 0;
								T d4 = ((j && k) && (j - start_j != 1 && k - start_k != 1)) ? cur_data_field[position_idx - dim1_offset - 1] : 0;
								T d5 = (j && (j - start_j != 1)) ? cur_data_field[position_idx - dim1_offset] : 0;
								T d6 = (k && (k - start_k != 1)) ? cur_data_field[position_idx - 1] : 0;
								T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
								double diff = cur_data_field[position_idx] - pred;
								double quant_diff = fabs(diff) / abs_eb + 1;
								if(quant_diff < capacity){
									quant_diff = (diff > 0) ? quant_diff : -quant_diff;
									int quant_index = (int)(quant_diff/2) + intv_radius;
									data_quant_index[3*position_idx + p] = quant_index;
									decompressed[p] = pred + 2 * (quant_index - intv_radius) * abs_eb;
									// check original data
									if(fabs(decompressed[p] - cur_data_field[position_idx]) >= abs_eb){
										unpred_flag = true;
										break;
									}
								}
								else{
									unpred_flag = true;
									break; 
								}
							}
						}						
						else{
							unpred_flag = true;
						}

						if (unpred_flag){
							eb_quant_index[position_idx] = 0;
							unpred_data_thread[omp_get_thread_num()].push_back(decompressed_U[position_idx]);
							unpred_data_thread[omp_get_thread_num()].push_back(decompressed_V[position_idx]);
							unpred_data_thread[omp_get_thread_num()].push_back(decompressed_W[position_idx]);
						}
						else{
							//predictable data
							decompressed_U[position_idx] = decompressed[0];
							decompressed_V[position_idx] = decompressed[1];
							decompressed_W[position_idx] = decompressed[2];
						}
					}
				
					else{
						//record as unpredictable data
						eb_quant_index[position_idx] = 0;
						unpred_data_thread[omp_get_thread_num()].push_back(decompressed_U[position_idx]);
						unpred_data_thread[omp_get_thread_num()].push_back(decompressed_V[position_idx]);
						unpred_data_thread[omp_get_thread_num()].push_back(decompressed_W[position_idx]);
					}
				}
			}
		}
	}

	auto block_end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> block_duration = block_end - block_start;
	printf("Block processing time: %f s\n", block_duration.count());
	// //merge unpred_data_thread
	// for (int i = 0; i < num_threads; ++i){
	// 	unpred_data.insert(unpred_data.end(), unpred_data_thread[i].begin(), unpred_data_thread[i].end());
	// }

	//目前已经处理完了每个块的数据，现在要特殊处理划分线上的数据
	//串行处理划分线上的数据
/*
	for(int i = 0; i < r1; i++){
		for (int j = 0 ; j < r2; j++){
			for (int k = 0; k < r3; k++){
				if (on_dividing_line[i * r2 * r3 + j * r3 + k]){
					size_t position_idx = i * r2 * r3 + j * r3 + k;
					processed_data_count++;
					processed_data_flag[position_idx] = true;
					double required_eb;
					required_eb = max_eb;
					// if ((decompressed_U[position_idx] == 0) || (decompressed_V[position_idx] == 0) || (decompressed_W[position_idx] == 0)) {
					// 	required_eb = 0;
					// } //这里需要处理吗？
					if(required_eb){
						// derive eb given 24 adjacent simplex
						for (int n = 0; n < 24; n++){
							bool in_mesh = true;
							for (int p = 0; p < 3; p++){
								//reversed order!
								if (!(in_range(i + index_offset[n][p][2], (int)r1) && in_range(j + index_offset[n][p][1], (int)r2) && in_range(k + index_offset[n][p][0], (int)r3))) {
									in_mesh = false;
									break;
								}
							}
							if (in_mesh){
								int index = simplex_offset[n] + 6 * (i * (r2 - 1) * (r3 - 1) + j * (r3 - 1) + k);
								if (cp_exist[index]) {
									required_eb = 0;
									break;
								}
								else {
									required_eb = MIN(required_eb, max_eb_to_keep_position_and_type_3d_online_abs(
										decompressed_U[position_idx +offset[n][0]], decompressed_U[position_idx +offset[n][1]], decompressed_U[position_idx +offset[n][2]], decompressed_U[position_idx],
										decompressed_V[position_idx +offset[n][0]], decompressed_V[position_idx +offset[n][1]], decompressed_V[position_idx +offset[n][2]], decompressed_V[position_idx],
										decompressed_W[position_idx +offset[n][0]], decompressed_W[position_idx +offset[n][1]], decompressed_W[position_idx +offset[n][2]], decompressed_W[position_idx]));
								}
							}
						}
					}

					if(required_eb){
						bool unpred_flag = false;
						T decompressed[3];
						double abs_eb = required_eb;
						eb_quant_index[position_idx] = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
						if(eb_quant_index[position_idx] > 0){
							// compress vector fields
							for(int p = 0; p < 3; p++){
								T * cur_data_field = (p == 0) ? decompressed_U : (p == 1) ? decompressed_V : decompressed_W;

								T d0 = (i && j && k) ? cur_data_field[position_idx - dim0_offset - dim1_offset - 1] : 0;
								T d1 = (i && j) ? cur_data_field[position_idx - dim0_offset - dim1_offset] : 0;
								T d2 = (i && k) ? cur_data_field[position_idx - dim0_offset - 1] : 0;
								T d3 = (i) ? cur_data_field[position_idx - dim0_offset] : 0;
								T d4 = (j && k) ? cur_data_field[position_idx - dim1_offset - 1] : 0;
								T d5 = (j) ? cur_data_field[position_idx - dim1_offset] : 0;
								T d6 = (k) ? cur_data_field[position_idx - 1] : 0;
								T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
								double diff = cur_data_field[position_idx] - pred;
								double quant_diff = fabs(diff) / abs_eb + 1;
								if (quant_diff < capacity){
									quant_diff = (diff > 0) ? quant_diff : -quant_diff;
									int quant_index = (int)(quant_diff/2) + intv_radius;
									data_quant_index[3*position_idx + p] = quant_index;
									decompressed[p] = pred + 2 * (quant_index - intv_radius) * abs_eb;
									if(fabs(decompressed[p] - cur_data_field[position_idx]) >= abs_eb){
										unpred_flag = true;
										break;
									}
								}
								else{
									unpred_flag = true;
									break;
								}
							}
						}

						else{
							unpred_flag = true;
						}
					
						if (unpred_flag){
							//recover quant index
							eb_quant_index[position_idx] = 0;
							unpred_data_dividing.push_back(decompressed_U[position_idx]);
							unpred_data_dividing.push_back(decompressed_V[position_idx]);
							unpred_data_dividing.push_back(decompressed_W[position_idx]);
						}
						else{
							decompressed_U[position_idx] = decompressed[0];
							decompressed_V[position_idx] = decompressed[1];
							decompressed_W[position_idx] = decompressed[2];
						}
					}
					else {
						// record as unpredictable data
						eb_quant_index[position_idx] = 0;
						unpred_data_dividing.push_back(decompressed_U[position_idx]);
						unpred_data_dividing.push_back(decompressed_V[position_idx]);
						unpred_data_dividing.push_back(decompressed_W[position_idx]);
					}
				}
			}
		}
	}
*/

	//优化
	auto face_edge_corner_start = std::chrono::high_resolution_clock::now();
	std::vector<std::vector<T>> unpred_data_faces_x(dividing_r1.size());
	std::vector<std::vector<T>> unpred_data_faces_y(dividing_r2.size());
	std::vector<std::vector<T>> unpred_data_faces_z(dividing_r3.size());

    std::vector<std::vector<T>> unpred_data_edges_x(dividing_r1.size());
	std::vector<std::vector<T>> unpred_data_edges_y(dividing_r2.size());
	std::vector<std::vector<T>> unpred_data_edges_z(dividing_r3.size());
	// Process faces perpendicular to the X-axis(kj-plane)

	#pragma omp parallel for num_threads(dividing_r1.size()) reduction(+:processed_data_count)
    for (size_t idx = 0; idx < dividing_r1.size(); ++idx) {
        int i = dividing_r1[idx]; //dividing_ri = [256,512,...]
        if (i >= r1) continue;
        std::vector<T>& thread_unpred_data = unpred_data_faces_x[idx];
        for (int j = 0; j < r2; ++j) {
            if (is_dividing_r2[j]) continue;
            for (int k = 0; k < r3; ++k) {
                if (is_dividing_r3[k]) continue;
				// 开始处理面内数据
				processed_data_count++;
				size_t position_idx = i * r2 * r3 + j * r3 + k;
				//计算block start 
				int block_i = i / block_r1;
				int block_j = j / block_r2;
				int block_k = k / block_r3;
				// 计算块的起始位置
				int start_i = block_i * block_r1;
				int start_j = block_j * block_r2;
				int start_k = block_k * block_r3;
				//计算block end
				int end_i = start_i + block_r1;
				int end_j = start_j + block_r2;
				int end_k = start_k + block_r3;
				if (block_i == t - 1) {
					end_i += remaining_r1;
				}
				if (block_j == t - 1) {
					end_j += remaining_r2;
				}
				if (block_k == t - 1) {
					end_k += remaining_r3;
				}


				double required_eb = max_eb;
				if ((decompressed_U[position_idx] == 0) || (decompressed_V[position_idx] == 0) || (decompressed_W[position_idx] == 0)) {
					required_eb = 0;
				}
				if (required_eb) {
					// derive eb given 24 adjacent simplex
					for (int n=0; n<24;n++){
						bool in_mesh = true;
						for (int p=0; p<3; p++){
							// reversed order!
							if (!(in_range(i + index_offset[n][p][2], (int)r1) && in_range(j + index_offset[n][p][1], (int)r2) && in_range(k + index_offset[n][p][0], (int)r3))) {
								in_mesh = false;
								break;
							}
						}
						if (in_mesh) {
							int index = simplex_offset[n] + 6 * (i * (r2 - 1) * (r3 - 1) + j * (r3 - 1) + k);
							if (cp_exist[index]) {
								required_eb = 0;
								break;
							}
							else {
								required_eb = MIN(required_eb, max_eb_to_keep_position_and_type_3d_online_abs(
									decompressed_U[position_idx +offset[n][0]], decompressed_U[position_idx +offset[n][1]], decompressed_U[position_idx +offset[n][2]], decompressed_U[position_idx],
									decompressed_V[position_idx +offset[n][0]], decompressed_V[position_idx +offset[n][1]], decompressed_V[position_idx +offset[n][2]], decompressed_V[position_idx],
									decompressed_W[position_idx +offset[n][0]], decompressed_W[position_idx +offset[n][1]], decompressed_W[position_idx +offset[n][2]], decompressed_W[position_idx]));
							}
						}

					}
				}
				
				if(required_eb){
					bool unpred_flag = false;
					T decompressed[3];
					double abs_eb = required_eb;
					eb_quant_index[position_idx] = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
					if(eb_quant_index[position_idx] > 0){
						// compress vector fields
						for(int p=0; p<3; p++){
							T * cur_data_field = (p == 0) ? decompressed_U : (p == 1) ? decompressed_V : decompressed_W;
							T d0 = ((j && k) && (j - start_j != 1 && k-start_k != 1)) ? cur_data_field[position_idx - dim1_offset - 1] : 0;
							T d1 = (j && (j - start_j != 1)) ? cur_data_field[position_idx - dim1_offset] : 0;
							T d2 = (k && (k - start_k != 1)) ? cur_data_field[position_idx - 1] : 0;
							T pred = d1 + d2 - d0;
							double diff = cur_data_field[position_idx] - pred;
							double quant_diff = fabs(diff) / abs_eb + 1;
							if(quant_diff < capacity){
								quant_diff = (diff > 0) ? quant_diff : -quant_diff;
								int quant_index = (int)(quant_diff/2) + intv_radius;
								data_quant_index[3*position_idx + p] = quant_index;
								decompressed[p] = pred + 2 * (quant_index - intv_radius) * abs_eb;
								// check original data
								if(fabs(decompressed[p] - cur_data_field[position_idx]) >= abs_eb){
									unpred_flag = true;
									break;
								}
							}
							else{
								unpred_flag = true;
								break; 
							}
						}
					}
					else{
						unpred_flag = true;
					}

					if (unpred_flag){
						eb_quant_index[position_idx] = 0;
						thread_unpred_data.push_back(decompressed_U[position_idx]);
						thread_unpred_data.push_back(decompressed_V[position_idx]);
						thread_unpred_data.push_back(decompressed_W[position_idx]);
					}
					else{
						//predictable data
						decompressed_U[position_idx] = decompressed[0];
						decompressed_V[position_idx] = decompressed[1];
						decompressed_W[position_idx] = decompressed[2];
					}
				}
			
				else{
					//record as unpredictable data
					eb_quant_index[position_idx] = 0;
					thread_unpred_data.push_back(decompressed_U[position_idx]);
					thread_unpred_data.push_back(decompressed_V[position_idx]);
					thread_unpred_data.push_back(decompressed_W[position_idx]);
				}
            }
        }
    }
	// Process faces perpendicular to the Y-axis(ki-plane)
    #pragma omp parallel for num_threads(dividing_r2.size()) reduction(+:processed_data_count)
    for (size_t idx = 0; idx < dividing_r2.size(); ++idx) {
        int j = dividing_r2[idx];
        if (j >= r2) continue;
        unpred_vec<T>& thread_unpred_data = unpred_data_faces_y[idx];
        for (int i = 0; i < r1; ++i) {
            if (is_dividing_r1[i]) continue;
            for (int k = 0; k < r3; ++k) {
                if (is_dividing_r3[k]) continue;
				// 开始处理块内数据
				processed_data_count++;
				size_t position_idx = i * r2 * r3 + j * r3 + k;
				//计算block start 
				int block_i = i / block_r1;
				int block_j = j / block_r2;
				int block_k = k / block_r3;
				// 计算块的起始位置
				int start_i = block_i * block_r1;
				int start_j = block_j * block_r2;
				int start_k = block_k * block_r3;
				//计算block end
				int end_i = start_i + block_r1;
				int end_j = start_j + block_r2;
				int end_k = start_k + block_r3;
				if (block_i == t - 1) {
					end_i += remaining_r1;
				}
				if (block_j == t - 1) {
					end_j += remaining_r2;
				}
				if (block_k == t - 1) {
					end_k += remaining_r3;
				}
				double required_eb = max_eb;
				if ((decompressed_U[position_idx] == 0) || (decompressed_V[position_idx] == 0) || (decompressed_W[position_idx] == 0)) {
					required_eb = 0;
				}
				if (required_eb) {
					// derive eb given 24 adjacent simplex
					for (int n=0; n<24;n++){
						bool in_mesh = true;
						for (int p=0; p<3; p++){
							// reversed order!
							if (!(in_range(i + index_offset[n][p][2], (int)r1) && in_range(j + index_offset[n][p][1], (int)r2) && in_range(k + index_offset[n][p][0], (int)r3))) {
								in_mesh = false;
								break;
							}
						}
						if (in_mesh) {
							int index = simplex_offset[n] + 6 * (i * (r2 - 1) * (r3 - 1) + j * (r3 - 1) + k);
							if (cp_exist[index]) {
								required_eb = 0;
								break;
							}
							else {
								required_eb = MIN(required_eb, max_eb_to_keep_position_and_type_3d_online_abs(
									decompressed_U[position_idx +offset[n][0]], decompressed_U[position_idx +offset[n][1]], decompressed_U[position_idx +offset[n][2]], decompressed_U[position_idx],
									decompressed_V[position_idx +offset[n][0]], decompressed_V[position_idx +offset[n][1]], decompressed_V[position_idx +offset[n][2]], decompressed_V[position_idx],
									decompressed_W[position_idx +offset[n][0]], decompressed_W[position_idx +offset[n][1]], decompressed_W[position_idx +offset[n][2]], decompressed_W[position_idx]));
							}
						}

					}
				}
				
				if(required_eb){
					bool unpred_flag = false;
					T decompressed[3];
					double abs_eb = required_eb;
					eb_quant_index[position_idx] = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
					if(eb_quant_index[position_idx] > 0){
						// compress vector fields
						for(int p=0; p<3; p++){
							T * cur_data_field = (p == 0) ? decompressed_U : (p == 1) ? decompressed_V : decompressed_W;
							T d0 = ((i && k) && (i - start_i != 1 && k - start_k != 1)) ? cur_data_field[position_idx -1 - dim0_offset] : 0;
							T d1 = (i && (i - start_i != 1)) ? cur_data_field[position_idx - dim0_offset] : 0;
							T d2 = (k && (k - start_k != 1)) ? cur_data_field[position_idx - 1] : 0;
							T pred = d1 + d2 - d0;
							double diff = cur_data_field[position_idx] - pred;
							double quant_diff = fabs(diff) / abs_eb + 1;
							if(quant_diff < capacity){
								quant_diff = (diff > 0) ? quant_diff : -quant_diff;
								int quant_index = (int)(quant_diff/2) + intv_radius;
								data_quant_index[3*position_idx + p] = quant_index;
								decompressed[p] = pred + 2 * (quant_index - intv_radius) * abs_eb;
								// check original data
								if(fabs(decompressed[p] - cur_data_field[position_idx]) >= abs_eb){
									unpred_flag = true;
									break;
								}
							}
							else{
								unpred_flag = true;
								break; 
							}
						}
					}
					else{
						unpred_flag = true;
					}

					if (unpred_flag){
						eb_quant_index[position_idx] = 0;
						thread_unpred_data.push_back(decompressed_U[position_idx]);
						thread_unpred_data.push_back(decompressed_V[position_idx]);
						thread_unpred_data.push_back(decompressed_W[position_idx]);
					}
					else{
						//predictable data
						decompressed_U[position_idx] = decompressed[0];
						decompressed_V[position_idx] = decompressed[1];
						decompressed_W[position_idx] = decompressed[2];
					}
				}
			
				else{
					//record as unpredictable data
					eb_quant_index[position_idx] = 0;
					thread_unpred_data.push_back(decompressed_U[position_idx]);
					thread_unpred_data.push_back(decompressed_V[position_idx]);
					thread_unpred_data.push_back(decompressed_W[position_idx]);
				}
            }
        }
    }

	// Process faces perpendicular to the Z-axis(ij-plane)
    #pragma omp parallel for num_threads(dividing_r3.size()) reduction(+:processed_data_count)
    for (size_t idx = 0; idx < dividing_r3.size(); ++idx) {
        int k = dividing_r3[idx];
        if (k >= r3) continue;
        unpred_vec<T>& thread_unpred_data = unpred_data_faces_z[idx];
        for (int i = 0; i < r1; ++i) {
            if (is_dividing_r1[i]) continue;
            for (int j = 0; j < r2; ++j) {
                if (is_dividing_r2[j]) continue;
				// 开始处理块内数据
				processed_data_count++;
				size_t position_idx = i * r2 * r3 + j * r3 + k;
				//计算block start 
				int block_i = i / block_r1;
				int block_j = j / block_r2;
				int block_k = k / block_r3;
				// 计算块的起始位置
				int start_i = block_i * block_r1;
				int start_j = block_j * block_r2;
				int start_k = block_k * block_r3;
				//计算block end
				int end_i = start_i + block_r1;
				int end_j = start_j + block_r2;
				int end_k = start_k + block_r3;
				if (block_i == t - 1) {
					end_i += remaining_r1;
				}
				if (block_j == t - 1) {
					end_j += remaining_r2;
				}
				if (block_k == t - 1) {
					end_k += remaining_r3;
				}
				double required_eb = max_eb;
				if ((decompressed_U[position_idx] == 0) || (decompressed_V[position_idx] == 0) || (decompressed_W[position_idx] == 0)) {
					required_eb = 0;
				}
				if (required_eb) {
					// derive eb given 24 adjacent simplex
					for (int n=0; n<24;n++){
						bool in_mesh = true;
						for (int p=0; p<3; p++){
							// reversed order!
							if (!(in_range(i + index_offset[n][p][2], (int)r1) && in_range(j + index_offset[n][p][1], (int)r2) && in_range(k + index_offset[n][p][0], (int)r3))) {
								in_mesh = false;
								break;
							}
						}
						if (in_mesh) {
							int index = simplex_offset[n] + 6 * (i * (r2 - 1) * (r3 - 1) + j * (r3 - 1) + k);
							if (cp_exist[index]) {
								required_eb = 0;
								break;
							}
							else {
								required_eb = MIN(required_eb, max_eb_to_keep_position_and_type_3d_online_abs(
									decompressed_U[position_idx +offset[n][0]], decompressed_U[position_idx +offset[n][1]], decompressed_U[position_idx +offset[n][2]], decompressed_U[position_idx],
									decompressed_V[position_idx +offset[n][0]], decompressed_V[position_idx +offset[n][1]], decompressed_V[position_idx +offset[n][2]], decompressed_V[position_idx],
									decompressed_W[position_idx +offset[n][0]], decompressed_W[position_idx +offset[n][1]], decompressed_W[position_idx +offset[n][2]], decompressed_W[position_idx]));
							}
						}

					}
				}
				
				if(required_eb){
					bool unpred_flag = false;
					T decompressed[3];
					double abs_eb = required_eb;
					eb_quant_index[position_idx] = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
					if(eb_quant_index[position_idx] > 0){
						// compress vector fields
						for(int p=0; p<3; p++){
							T * cur_data_field = (p == 0) ? decompressed_U : (p == 1) ? decompressed_V : decompressed_W;
							T d0 = ((i && j) && (i - start_i != 1 && j - start_j != 1)) ? cur_data_field[position_idx - dim0_offset - dim1_offset] : 0;
							T d1 = (i && (i - start_i != 1)) ? cur_data_field[position_idx - dim0_offset] : 0;
							T d2 = (j && (j - start_j != 1)) ? cur_data_field[position_idx - dim1_offset] : 0;
							T pred = d1 + d2 - d0;
							double diff = cur_data_field[position_idx] - pred;
							double quant_diff = fabs(diff) / abs_eb + 1;
							if(quant_diff < capacity){
								quant_diff = (diff > 0) ? quant_diff : -quant_diff;
								int quant_index = (int)(quant_diff/2) + intv_radius;
								data_quant_index[3*position_idx + p] = quant_index;
								decompressed[p] = pred + 2 * (quant_index - intv_radius) * abs_eb;
								// check original data
								if(fabs(decompressed[p] - cur_data_field[position_idx]) >= abs_eb){
									unpred_flag = true;
									break;
								}
							}
							else{
								unpred_flag = true;
								break; 
							}
						}
					}
					else{
						unpred_flag = true;
					}

					if (unpred_flag){
						eb_quant_index[position_idx] = 0;
						thread_unpred_data.push_back(decompressed_U[position_idx]);
						thread_unpred_data.push_back(decompressed_V[position_idx]);
						thread_unpred_data.push_back(decompressed_W[position_idx]);
					}
					else{
						//predictable data
						decompressed_U[position_idx] = decompressed[0];
						decompressed_V[position_idx] = decompressed[1];
						decompressed_W[position_idx] = decompressed[2];
					}
				}
			
				else{
					//record as unpredictable data
					eb_quant_index[position_idx] = 0;
					thread_unpred_data.push_back(decompressed_U[position_idx]);
					thread_unpred_data.push_back(decompressed_V[position_idx]);
					thread_unpred_data.push_back(decompressed_W[position_idx]);
				}
            }
        }
    }

/*
	// Process Edges (Points on the intersection of two dividing planes but not corners)
    // Process edges along the X-axis(垂直于jk平面形成的边)
    #pragma omp parallel for num_threads(num_threads) schedule(static) reduction(+:processed_data_count)
    for (size_t idx_j = 0; idx_j < dividing_r2.size(); ++idx_j) {
        int j = dividing_r2[idx_j];
        if (j >= r2) continue;
        for (size_t idx_k = 0; idx_k < dividing_r3.size(); ++idx_k) {
            int k = dividing_r3[idx_k];
            if (k >= r3) continue;
            unpred_vec<T>& thread_unpred_data = unpred_data_edges_x[omp_get_thread_num()];
            for (int i = 0; i < r1; ++i) {
                if (is_dividing_r1[i]) continue; // Skip corners
				// 开始处理块内数据
				processed_data_count++;
				size_t position_idx = i * r2 * r3 + j * r3 + k;
				//计算block start 
				int block_i = i / block_r1;
				int block_j = j / block_r2;
				int block_k = k / block_r3;
				// 计算块的起始位置
				int start_i = block_i * block_r1;
				int start_j = block_j * block_r2;
				int start_k = block_k * block_r3;
				//计算block end
				int end_i = start_i + block_r1;
				int end_j = start_j + block_r2;
				int end_k = start_k + block_r3;
				if (block_i == t - 1) {
					end_i += remaining_r1;
				}
				if (block_j == t - 1) {
					end_j += remaining_r2;
				}
				if (block_k == t - 1) {
					end_k += remaining_r3;
				}
				double required_eb = max_eb;
				if ((decompressed_U[position_idx] == 0) || (decompressed_V[position_idx] == 0) || (decompressed_W[position_idx] == 0)) {
					required_eb = 0;
				}
				if (required_eb) {
					// derive eb given 24 adjacent simplex
					for (int n=0; n<24;n++){
						bool in_mesh = true;
						for (int p=0; p<3; p++){
							// reversed order!
							if (!(in_range(i + index_offset[n][p][2], (int)r1) && in_range(j + index_offset[n][p][1], (int)r2) && in_range(k + index_offset[n][p][0], (int)r3))) {
								in_mesh = false;
								break;
							}
						}
						if (in_mesh) {
							int index = simplex_offset[n] + 6 * (i * (r2 - 1) * (r3 - 1) + j * (r3 - 1) + k);
							if (cp_exist[index]) {
								required_eb = 0;
								break;
							}
							else {
								required_eb = MIN(required_eb, max_eb_to_keep_position_and_type_3d_online_abs(
									decompressed_U[position_idx +offset[n][0]], decompressed_U[position_idx +offset[n][1]], decompressed_U[position_idx +offset[n][2]], decompressed_U[position_idx],
									decompressed_V[position_idx +offset[n][0]], decompressed_V[position_idx +offset[n][1]], decompressed_V[position_idx +offset[n][2]], decompressed_V[position_idx],
									decompressed_W[position_idx +offset[n][0]], decompressed_W[position_idx +offset[n][1]], decompressed_W[position_idx +offset[n][2]], decompressed_W[position_idx]));
							}
						}

					}
				}
				
				if(required_eb){
					bool unpred_flag = false;
					T decompressed[3];
					double abs_eb = required_eb;
					eb_quant_index[position_idx] = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
					if(eb_quant_index[position_idx] > 0){
						// compress vector fields
						for(int p=0; p<3; p++){
							T * cur_data_field = (p == 0) ? decompressed_U : (p == 1) ? decompressed_V : decompressed_W;
							T d0 = ((i) && (i - start_i >= 1)) ? cur_data_field[position_idx - dim0_offset] : 0;
							T pred = d0;
							double diff = cur_data_field[position_idx] - pred;
							double quant_diff = fabs(diff) / abs_eb + 1;
							if(quant_diff < capacity){
								quant_diff = (diff > 0) ? quant_diff : -quant_diff;
								int quant_index = (int)(quant_diff/2) + intv_radius;
								data_quant_index[3*position_idx + p] = quant_index;
								decompressed[p] = pred + 2 * (quant_index - intv_radius) * abs_eb;
								// check original data
								if(fabs(decompressed[p] - cur_data_field[position_idx]) >= abs_eb){
									unpred_flag = true;
									break;
								}
							}
							else{
								unpred_flag = true;
								break; 
							}
						}
					}
					else{
						unpred_flag = true;
					}

					if (unpred_flag){
						eb_quant_index[position_idx] = 0;
						thread_unpred_data.push_back(decompressed_U[position_idx]);
						thread_unpred_data.push_back(decompressed_V[position_idx]);
						thread_unpred_data.push_back(decompressed_W[position_idx]);
					}
					else{
						//predictable data
						decompressed_U[position_idx] = decompressed[0];
						decompressed_V[position_idx] = decompressed[1];
						decompressed_W[position_idx] = decompressed[2];
					}
				}
			
				else{
					//record as unpredictable data
					eb_quant_index[position_idx] = 0;
					thread_unpred_data.push_back(decompressed_U[position_idx]);
					thread_unpred_data.push_back(decompressed_V[position_idx]);
					thread_unpred_data.push_back(decompressed_W[position_idx]);
            	}
        	}
    	}
	}
    // Process edges along the Y-axis（垂直于ik平面）
    #pragma omp parallel for num_threads(num_threads) schedule(static) reduction(+:processed_data_count)
    for (size_t idx_i = 0; idx_i < dividing_r1.size(); ++idx_i) {
        int i = dividing_r1[idx_i];
        if (i >= r1) continue;
        for (size_t idx_k = 0; idx_k < dividing_r3.size(); ++idx_k) {
            int k = dividing_r3[idx_k];
            if (k >= r3) continue;
            unpred_vec<T>& thread_unpred_data = unpred_data_edges_y[omp_get_thread_num()];
            for (int j = 0; j < r2; ++j) {
                if (is_dividing_r2[j]) continue; // Skip corners
				// 开始处理块内数据
				processed_data_count++;
				size_t position_idx = i * r2 * r3 + j * r3 + k;
				//计算block start 
				int block_i = i / block_r1;
				int block_j = j / block_r2;
				int block_k = k / block_r3;
				// 计算块的起始位置
				int start_i = block_i * block_r1;
				int start_j = block_j * block_r2;
				int start_k = block_k * block_r3;
				//计算block end
				int end_i = start_i + block_r1;
				int end_j = start_j + block_r2;
				int end_k = start_k + block_r3;
				if (block_i == t - 1) {
					end_i += remaining_r1;
				}
				if (block_j == t - 1) {
					end_j += remaining_r2;
				}
				if (block_k == t - 1) {
					end_k += remaining_r3;
				}
				double required_eb = max_eb;
				if ((decompressed_U[position_idx] == 0) || (decompressed_V[position_idx] == 0) || (decompressed_W[position_idx] == 0)) {
					required_eb = 0;
				}
				if (required_eb) {
					// derive eb given 24 adjacent simplex
					for (int n=0; n<24;n++){
						bool in_mesh = true;
						for (int p=0; p<3; p++){
							// reversed order!
							if (!(in_range(i + index_offset[n][p][2], (int)r1) && in_range(j + index_offset[n][p][1], (int)r2) && in_range(k + index_offset[n][p][0], (int)r3))) {
								in_mesh = false;
								break;
							}
						}
						if (in_mesh) {
							int index = simplex_offset[n] + 6 * (i * (r2 - 1) * (r3 - 1) + j * (r3 - 1) + k);
							if (cp_exist[index]) {
								required_eb = 0;
								break;
							}
							else {
								required_eb = MIN(required_eb, max_eb_to_keep_position_and_type_3d_online_abs(
									decompressed_U[position_idx +offset[n][0]], decompressed_U[position_idx +offset[n][1]], decompressed_U[position_idx +offset[n][2]], decompressed_U[position_idx],
									decompressed_V[position_idx +offset[n][0]], decompressed_V[position_idx +offset[n][1]], decompressed_V[position_idx +offset[n][2]], decompressed_V[position_idx],
									decompressed_W[position_idx +offset[n][0]], decompressed_W[position_idx +offset[n][1]], decompressed_W[position_idx +offset[n][2]], decompressed_W[position_idx]));
							}
						}

					}
				}
				
				if(required_eb){
					bool unpred_flag = false;
					T decompressed[3];
					double abs_eb = required_eb;
					eb_quant_index[position_idx] = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
					if(eb_quant_index[position_idx] > 0){
						// compress vector fields
						for(int p=0; p<3; p++){
							T * cur_data_field = (p == 0) ? decompressed_U : (p == 1) ? decompressed_V : decompressed_W;
							// T d0 = ((i && j && k)) ? cur_data_field[position_idx - dim0_offset - dim1_offset - 1] : 0;
							// T d1 = ((i && j)) ? cur_data_field[position_idx - dim0_offset - dim1_offset] : 0;
							// T d2 = ((i && k)) ? cur_data_field[position_idx - dim0_offset - 1] : 0;
							// T d3 = (i) ? cur_data_field[position_idx - dim0_offset] : 0;
							// T d4 = ((j && k)) ? cur_data_field[position_idx - dim1_offset - 1] : 0;
							// T d5 = (j) ? cur_data_field[position_idx - dim1_offset] : 0;
							// T d6 = (k) ? cur_data_field[position_idx - 1] : 0;
							// T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
							T d0 = ((j) && (j - start_j >= 1)) ? cur_data_field[position_idx - dim1_offset] : 0;
							T pred = d0;
							double diff = cur_data_field[position_idx] - pred;
							double quant_diff = fabs(diff) / abs_eb + 1;
							if(quant_diff < capacity){
								quant_diff = (diff > 0) ? quant_diff : -quant_diff;
								int quant_index = (int)(quant_diff/2) + intv_radius;
								data_quant_index[3*position_idx + p] = quant_index;
								decompressed[p] = pred + 2 * (quant_index - intv_radius) * abs_eb;
								// check original data
								if(fabs(decompressed[p] - cur_data_field[position_idx]) >= abs_eb){
									unpred_flag = true;
									break;
								}
							}
							else{
								unpred_flag = true;
								break; 
							}
						}
					}
					else{
						unpred_flag = true;
					}

					if (unpred_flag){
						eb_quant_index[position_idx] = 0;
						thread_unpred_data.push_back(decompressed_U[position_idx]);
						thread_unpred_data.push_back(decompressed_V[position_idx]);
						thread_unpred_data.push_back(decompressed_W[position_idx]);
					}
					else{
						//predictable data
						decompressed_U[position_idx] = decompressed[0];
						decompressed_V[position_idx] = decompressed[1];
						decompressed_W[position_idx] = decompressed[2];
					}
				}
			
				else{
					//record as unpredictable data
					eb_quant_index[position_idx] = 0;
					thread_unpred_data.push_back(decompressed_U[position_idx]);
					thread_unpred_data.push_back(decompressed_V[position_idx]);
					thread_unpred_data.push_back(decompressed_W[position_idx]);
				}
            }
        }
    }

    // Process edges along the Z-axis（垂直于ij）
    #pragma omp parallel for num_threads(num_threads) schedule(static) reduction(+:processed_data_count)
    for (size_t idx_i = 0; idx_i < dividing_r1.size(); ++idx_i) {
        int i = dividing_r1[idx_i];
        if (i >= r1) continue;
        for (size_t idx_j = 0; idx_j < dividing_r2.size(); ++idx_j) {
            int j = dividing_r2[idx_j];
            if (j >= r2) continue;
            unpred_vec<T>& thread_unpred_data = unpred_data_edges_z[omp_get_thread_num()];
            for (int k = 0; k < r3; ++k) {
                if (is_dividing_r3[k]) continue; // Skip corners
				// 开始处理块内数据
				processed_data_count++;
				size_t position_idx = i * r2 * r3 + j * r3 + k;
				//计算block start 
				int block_i = i / block_r1;
				int block_j = j / block_r2;
				int block_k = k / block_r3;
				// 计算块的起始位置
				int start_i = block_i * block_r1;
				int start_j = block_j * block_r2;
				int start_k = block_k * block_r3;
				//计算block end
				int end_i = start_i + block_r1;
				int end_j = start_j + block_r2;
				int end_k = start_k + block_r3;
				if (block_i == t - 1) {
					end_i += remaining_r1;
				}
				if (block_j == t - 1) {
					end_j += remaining_r2;
				}
				if (block_k == t - 1) {
					end_k += remaining_r3;
				}
				double required_eb = max_eb;
				if ((decompressed_U[position_idx] == 0) || (decompressed_V[position_idx] == 0) || (decompressed_W[position_idx] == 0)) {
					required_eb = 0;
				}
				if (required_eb) {
					// derive eb given 24 adjacent simplex
					for (int n=0; n<24;n++){
						bool in_mesh = true;
						for (int p=0; p<3; p++){
							// reversed order!
							if (!(in_range(i + index_offset[n][p][2], (int)r1) && in_range(j + index_offset[n][p][1], (int)r2) && in_range(k + index_offset[n][p][0], (int)r3))) {
								in_mesh = false;
								break;
							}
						}
						if (in_mesh) {
							int index = simplex_offset[n] + 6 * (i * (r2 - 1) * (r3 - 1) + j * (r3 - 1) + k);
							if (cp_exist[index]) {
								required_eb = 0;
								break;
							}
							else {
								required_eb = MIN(required_eb, max_eb_to_keep_position_and_type_3d_online_abs(
									decompressed_U[position_idx +offset[n][0]], decompressed_U[position_idx +offset[n][1]], decompressed_U[position_idx +offset[n][2]], decompressed_U[position_idx],
									decompressed_V[position_idx +offset[n][0]], decompressed_V[position_idx +offset[n][1]], decompressed_V[position_idx +offset[n][2]], decompressed_V[position_idx],
									decompressed_W[position_idx +offset[n][0]], decompressed_W[position_idx +offset[n][1]], decompressed_W[position_idx +offset[n][2]], decompressed_W[position_idx]));
							}
						}

					}
				}
				
				if(required_eb){
					bool unpred_flag = false;
					T decompressed[3];
					double abs_eb = required_eb;
					eb_quant_index[position_idx] = eb_exponential_quantize(abs_eb, base, log_of_base, threshold);
					if(eb_quant_index[position_idx] > 0){
						// compress vector fields
						for(int p=0; p<3; p++){
							T * cur_data_field = (p == 0) ? decompressed_U : (p == 1) ? decompressed_V : decompressed_W;
							T d0 = (k && (k - start_k >= 1)) ? cur_data_field[position_idx - 1] : 0;
							T pred = d0;
							double diff = cur_data_field[position_idx] - pred;
							double quant_diff = fabs(diff) / abs_eb + 1;
							if(quant_diff < capacity){
								quant_diff = (diff > 0) ? quant_diff : -quant_diff;
								int quant_index = (int)(quant_diff/2) + intv_radius;
								data_quant_index[3*position_idx + p] = quant_index;
								decompressed[p] = pred + 2 * (quant_index - intv_radius) * abs_eb;
								// check original data
								if(fabs(decompressed[p] - cur_data_field[position_idx]) >= abs_eb){
									unpred_flag = true;
									break;
								}
							}
							else{
								unpred_flag = true;
								break; 
							}
						}
					}
					else{
						unpred_flag = true;
					}

					if (unpred_flag){
						eb_quant_index[position_idx] = 0;
						thread_unpred_data.push_back(decompressed_U[position_idx]);
						thread_unpred_data.push_back(decompressed_V[position_idx]);
						thread_unpred_data.push_back(decompressed_W[position_idx]);
					}
					else{
						//predictable data
						decompressed_U[position_idx] = decompressed[0];
						decompressed_V[position_idx] = decompressed[1];
						decompressed_W[position_idx] = decompressed[2];
					}
				}
			
				else{
					//record as unpredictable data
					eb_quant_index[position_idx] = 0;
					thread_unpred_data.push_back(decompressed_U[position_idx]);
					thread_unpred_data.push_back(decompressed_V[position_idx]);
					thread_unpred_data.push_back(decompressed_W[position_idx]);
				}
            }
        }
    }
*/

/*
	// Process Edges just fucking lossless
	// Process edges along the X-axis(垂直于jk平面形成的边)
    #pragma omp parallel for num_threads(dividing_r1.size()) reduction(+:processed_data_count)
    for (size_t idx_j = 0; idx_j < dividing_r2.size(); ++idx_j) {
        int j = dividing_r2[idx_j];
        if (j >= r2) continue;
        for (size_t idx_k = 0; idx_k < dividing_r3.size(); ++idx_k) {
            int k = dividing_r3[idx_k];
            if (k >= r3) continue;
            unpred_vec<T>& thread_unpred_data = unpred_data_edges_x[idx_j];
            for (int i = 0; i < r1; ++i) {
                if (is_dividing_r1[i]) continue; // Skip corners
				// 开始处理块内数据
				processed_data_count++;
				size_t position_idx = i * r2 * r3 + j * r3 + k;
				eb_quant_index[position_idx] = 0;
				thread_unpred_data.push_back(decompressed_U[position_idx]);
				thread_unpred_data.push_back(decompressed_V[position_idx]);
				thread_unpred_data.push_back(decompressed_W[position_idx]);
			}
		}	
	}
    // Process edges along the Y-axis（垂直于ik平面）
    #pragma omp parallel for num_threads(dividing_r2.size()) reduction(+:processed_data_count)
    for (size_t idx_i = 0; idx_i < dividing_r1.size(); ++idx_i) {
        int i = dividing_r1[idx_i];
        if (i >= r1) continue;
        for (size_t idx_k = 0; idx_k < dividing_r3.size(); ++idx_k) {
            int k = dividing_r3[idx_k];
            if (k >= r3) continue;
            unpred_vec<T>& thread_unpred_data = unpred_data_edges_y[idx_i];
            for (int j = 0; j < r2; ++j) {
                if (is_dividing_r2[j]) continue; // Skip corners
				// 开始处理块内数据
				processed_data_count++;
				size_t position_idx = i * r2 * r3 + j * r3 + k;
				eb_quant_index[position_idx] = 0;
				thread_unpred_data.push_back(decompressed_U[position_idx]);
				thread_unpred_data.push_back(decompressed_V[position_idx]);
				thread_unpred_data.push_back(decompressed_W[position_idx]);
			}
        }
    }

    // Process edges along the Z-axis（垂直于ij）
    #pragma omp parallel for num_threads(dividing_r3.size()) reduction(+:processed_data_count)
    for (size_t idx_i = 0; idx_i < dividing_r1.size(); ++idx_i) {
        int i = dividing_r1[idx_i];
        if (i >= r1) continue;
        for (size_t idx_j = 0; idx_j < dividing_r2.size(); ++idx_j) {
            int j = dividing_r2[idx_j];
            if (j >= r2) continue;
            unpred_vec<T>& thread_unpred_data = unpred_data_edges_z[idx_i];
            for (int k = 0; k < r3; ++k) {
                if (is_dividing_r3[k]) continue; // Skip corners
				// 开始处理块内数据
				processed_data_count++;
				size_t position_idx = i * r2 * r3 + j * r3 + k;
				eb_quant_index[position_idx] = 0;
				thread_unpred_data.push_back(decompressed_U[position_idx]);
				thread_unpred_data.push_back(decompressed_V[position_idx]);
				thread_unpred_data.push_back(decompressed_W[position_idx]);
			}
        }
    }
*/

/*
	//直接串行处理棱 & corner
	for (int i = 0; i < r1; ++i) {
		for (int j = 0; j < r2; ++j) {
			for (int k = 0; k < r3; ++k) {
				if ((is_dividing_r1[i] && is_dividing_r2[j]) || (is_dividing_r1[i] && is_dividing_r3[k]) || (is_dividing_r2[j] && is_dividing_r3[k]) || (is_dividing_r1[i] && is_dividing_r2[j] && is_dividing_r3[k])) {
					eb_quant_index[i * r2 * r3 + j * r3 + k] = 0;
					corner_points.push_back(decompressed_U[i * r2 * r3 + j * r3 + k]);
					corner_points.push_back(decompressed_V[i * r2 * r3 + j * r3 + k]);
					corner_points.push_back(decompressed_W[i * r2 * r3 + j * r3 + k]);
				}
			}
		}
	}
*/

    // 使用线程局部的 corner_points 变量
	std::vector<std::vector<T>> corner_points(num_threads);
    #pragma omp parallel for num_threads(num_threads) reduction(+:processed_data_count)
	for (int block_id = 0; block_id < total_blocks; ++block_id){
		int block_i = block_id / (t * t);
		int block_j = (block_id % (t * t)) / t;
		int block_k = block_id % t;
		// 计算块的起始位置
		int start_i = block_i * block_r1;
		int start_j = block_j * block_r2;
		int start_k = block_k * block_r3;

		int end_i = start_i + block_r1;
		int end_j = start_j + block_r2;
		int end_k = start_k + block_r3;
		if (block_i == t - 1) {
			end_i += remaining_r1;
		}
		if (block_j == t - 1) {
			end_j += remaining_r2;
		}
		if (block_k == t - 1) {
			end_k += remaining_r3;
		}

		for (int i = start_i; i < end_i; ++i) {
			for (int j = start_j; j < end_j; ++j) {
				for (int k = start_k; k < end_k; ++k) {
					size_t position_idx = i * r2 * r3 + j * r3 + k;
					if ((is_dividing_r1[i] && is_dividing_r2[j]) || (is_dividing_r1[i] && is_dividing_r3[k]) || (is_dividing_r2[j] && is_dividing_r3[k]) || (is_dividing_r1[i] && is_dividing_r2[j] && is_dividing_r3[k])) {
						processed_data_count++;
						corner_points[omp_get_thread_num()].push_back(decompressed_U[position_idx]);
						corner_points[omp_get_thread_num()].push_back(decompressed_V[position_idx]);
						corner_points[omp_get_thread_num()].push_back(decompressed_W[position_idx]);
					}
				}
			}
		}
	}


	// Collect all corner points 串行处理,lossless
    // for (int i : dividing_r1) {
    //     if (i >= r1) continue;
    //     for (int j : dividing_r2) {
    //         if (j >= r2) continue;
    //         for (int k : dividing_r3) {
    //             if (k >= r3) continue;
    //             size_t position_idx = i * r2 * r3 + j * r3 + k;
	// 			processed_data_count++;
	// 			// record as unpredictable data
	// 			eb_quant_index[position_idx] = 0;
	// 			corner_points.push_back(decompressed_U[position_idx]);
	// 			corner_points.push_back(decompressed_V[position_idx]);
	// 			corner_points.push_back(decompressed_W[position_idx]);
	// 			// printf("u:%f,v:%f,w:%f\n", decompressed_U[position_idx], decompressed_V[position_idx], decompressed_W[position_idx]);
				
    //         }
    //     }
    // }

	auto face_edge_corner_end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> face_edge_corner_time = face_edge_corner_end - face_edge_corner_start;
	printf("face&edge&corner processing time = %f\n", face_edge_corner_time.count());
	printf("processed_data_count = %ld, total_data_count = %ld\n", processed_data_count, num_elements);
	// for (size_t i = 0; i < num_elements; i++) {
	// 	if (!processed_data_flag[i]) {
	// 		printf("Error: data point %ld is not processed\n", i);
	// 		exit(0);
	// 	}
	// }
	decompressed_U_ptr = decompressed_U;
	decompressed_V_ptr = decompressed_V;
	decompressed_W_ptr = decompressed_W;

	unsigned char * compressed = (unsigned char *) malloc(3*num_elements*sizeof(T));
	unsigned char * compressed_pos = compressed;
	//先写index_need_to_lossless的大小,不管怎样
	write_variable_to_dst(compressed_pos, index_need_to_lossless.size()); //size_t, index_need_to_lossless的大小
	printf("index_need_to_lossless pos = %ld\n", compressed_pos - compressed);
	//如果index_need_to_lossless.size() != 0，那么写bitmap
	if (index_need_to_lossless.size() != 0){
		//write bitmap
		convertIntArray2ByteArray_fast_1b_to_result_sz(bitmap, num_elements, compressed_pos);
		printf("bitmap pos = %ld\n", compressed_pos - compressed);
		//再写index_need_to_lossless对应U,V,W的数据
		for (auto it = index_need_to_lossless.begin(); it != index_need_to_lossless.end(); it++){
			write_variable_to_dst(compressed_pos, U[*it]); //T, index_need_to_lossless对应的U的值
		}
		for (auto it = index_need_to_lossless.begin(); it != index_need_to_lossless.end(); it++){
			write_variable_to_dst(compressed_pos, V[*it]); //T, index_need_to_lossless对应的V的值
		}
		for (auto it = index_need_to_lossless.begin(); it != index_need_to_lossless.end(); it++){
			write_variable_to_dst(compressed_pos, W[*it]); //T, index_need_to_lossless对应的W的值
		}
		printf("index_need_to_lossless data pos = %ld\n", compressed_pos - compressed);
	}
	// write number of threads
	write_variable_to_dst(compressed_pos, num_threads);
	//printf("num_threads = %d,pos = %ld\n", num_threads, compressed_pos - compressed);
	//写block的unpred_data
	// write number of unpredictable data for each thread for block data
	for (int i = 0; i < num_threads; ++i){
		write_variable_to_dst(compressed_pos, unpred_data_thread[i].size());
		//printf("thread %d, unpred_data size = %ld,maxvalue = %f\n", threadID, unpred_data_thread[threadID].size(), *std::max_element(unpred_data_thread[threadID].begin(), unpred_data_thread[threadID].end()));
	}
	size_t sum_unpred_count = 0;
	for (int i = 0; i < num_threads; ++i){
		sum_unpred_count += unpred_data_thread[i].size();
	}
	printf("comp sum_unpred_count_block = %ld\n", sum_unpred_count);
	//写face的unpred_data

	// write number of unpredictable data for each thread for face_x data
	size_t sum_unpred_count_faces_x = 0;
	for (int i = 0; i < dividing_r1.size(); ++i){
		write_variable_to_dst(compressed_pos, unpred_data_faces_x[i].size());
		sum_unpred_count_faces_x += unpred_data_faces_x[i].size();
	}
	printf("comp sum_unpred_count_faces_x = %ld\n", sum_unpred_count_faces_x);
	for (int i = 0; i < dividing_r1.size(); ++i){
		write_array_to_dst(compressed_pos, (T *)&unpred_data_faces_x[i][0], unpred_data_faces_x[i].size());
	}
	// write number of unpredictable data for each thread for face_y data
	for (int i = 0; i < dividing_r2.size(); ++i){
		write_variable_to_dst(compressed_pos, unpred_data_faces_y[i].size());
	}
	for (int i = 0; i < dividing_r2.size(); ++i){
		write_array_to_dst(compressed_pos, (T *)&unpred_data_faces_y[i][0], unpred_data_faces_y[i].size());
	}
	// write number of unpredictable data for each thread for face_z data
	for (int i = 0; i < dividing_r3.size(); ++i){
		write_variable_to_dst(compressed_pos, unpred_data_faces_z[i].size());
	}
	for (int i = 0; i < dividing_r3.size(); ++i){
		write_array_to_dst(compressed_pos, (T *)&unpred_data_faces_z[i][0], unpred_data_faces_z[i].size());
	}
	printf("comp pos after face = %ld\n", compressed_pos - compressed);

	/*
	// 写edge的unpred_data
	// write number of unpredictable data for each thread for edge_x data
	for (int i = 0; i < dividing_r1.size(); ++i){
		write_variable_to_dst(compressed_pos, unpred_data_edges_x[i].size());
	}
	for (int i = 0; i < dividing_r1.size(); ++i){
		write_array_to_dst(compressed_pos, (T *)&unpred_data_edges_x[i][0], unpred_data_edges_x[i].size());
	}
	// write number of unpredictable data for each thread for edge_y data
	for (int i = 0; i < dividing_r2.size(); ++i){
		write_variable_to_dst(compressed_pos, unpred_data_edges_y[i].size());
	}
	for (int i = 0; i < dividing_r2.size(); ++i){
		write_array_to_dst(compressed_pos, (T *)&unpred_data_edges_y[i][0], unpred_data_edges_y[i].size());
	}
	// write number of unpredictable data for each thread for edge_z data
	for (int i = 0; i < dividing_r3.size(); ++i){
		write_variable_to_dst(compressed_pos, unpred_data_edges_z[i].size());
	}
	for (int i = 0; i < dividing_r3.size(); ++i){
		write_array_to_dst(compressed_pos, (T *)&unpred_data_edges_z[i][0], unpred_data_edges_z[i].size());
	}
	printf("comp pos after edge = %ld\n", compressed_pos - compressed);
	*/
	
	for (int threadID = 0; threadID < dividing_r1.size(); threadID++){
		if (unpred_data_edges_x[threadID].size() != 0)
			printf("comp_edge_x thread %d, unpred_data size = %ld,maxvalue = %f\n", threadID, unpred_data_edges_x[threadID].size(), *std::max_element(unpred_data_edges_x[threadID].begin(), unpred_data_edges_x[threadID].end()));
		if (unpred_data_edges_y[threadID].size() != 0)
			printf("comp_edge_y thread %d, unpred_data size = %ld,maxvalue = %f\n", threadID, unpred_data_edges_y[threadID].size(), *std::max_element(unpred_data_edges_y[threadID].begin(), unpred_data_edges_y[threadID].end()));
		if (unpred_data_edges_z[threadID].size() != 0)
			printf("comp_edge_z thread %d, unpred_data size = %ld,maxvalue = %f\n", threadID, unpred_data_edges_z[threadID].size(), *std::max_element(unpred_data_edges_z[threadID].begin(), unpred_data_edges_z[threadID].end()));
	}
	//写corner的unpred_data
	for (int i = 0; i < num_threads; ++i){
		write_variable_to_dst(compressed_pos, corner_points[i].size());
	}
	for (int i = 0; i < num_threads; ++i){
		write_array_to_dst(compressed_pos, (T *)&corner_points[i][0], corner_points[i].size());
	}
	printf("comp pos after corner = %ld\n", compressed_pos - compressed);
	
	//now write base
	write_variable_to_dst(compressed_pos, base);
	write_variable_to_dst(compressed_pos, threshold);
	write_variable_to_dst(compressed_pos, intv_radius);
	// size_t unpred_data_size = unpred_data.size();
	size_t unpred_data_size = 0;
	for (int i = 0; i < num_threads; ++i){
		unpred_data_size += unpred_data_thread[i].size();
	}
	write_variable_to_dst(compressed_pos, unpred_data_size);
	printf("total unpred data size = %ld, pos = %ld\n", unpred_data_size, compressed_pos - compressed);
	
	//now write unpred_data for each thread
	for (int i = 0; i < num_threads; ++i){
		write_array_to_dst(compressed_pos, (T *)&unpred_data_thread[i][0], unpred_data_thread[i].size());
	}

	//write dividing line data size
	write_variable_to_dst(compressed_pos, unpred_data_dividing.size());
	//write dividing line data
	write_array_to_dst(compressed_pos, (T *)&unpred_data_dividing[0], unpred_data_dividing.size());

	//huffman for eb_quant(now its 1* #num element) and data_quant(still 3* #num element)
	printf("start eb decoding: pos = %ld\n", compressed_pos - compressed);
	size_t * freq =  (size_t *) calloc(num_threads * 4 * capacity, sizeof(size_t));
	omp_Huffman_encode_tree_and_data(2*capacity, eb_quant_index, num_elements, compressed_pos, freq, num_threads);


	free(freq);
	freq = (size_t *) calloc(num_threads * 4 * capacity, sizeof(size_t));
	omp_Huffman_encode_tree_and_data(2*capacity, data_quant_index, 3*num_elements, compressed_pos, freq, num_threads);
	free(freq);
	printf("pos after huffmans = %ld\n", compressed_pos - compressed);
	free(eb_quant_index);
	free(data_quant_index);
	if (index_need_to_lossless.size() != 0){
		free(bitmap);
	}
	compressed_size = compressed_pos - compressed;
	return compressed;
}

template unsigned char* omp_sz_compress_cp_preserve_3d_online_abs_record_vertex<float>(
    const float * U, const float * V, const float * W, size_t r1, size_t r2, size_t r3, 
    size_t& compressed_size, double max_eb, const std::set<size_t>& index_need_to_lossless, 
    int n_threads, float* &decompressed_U_ptr, float* &decompressed_V_ptr, float* &decompressed_W_ptr, std::vector<bool>& cp_exist);

template unsigned char* omp_sz_compress_cp_preserve_3d_online_abs_record_vertex<double>(
    const double * U, const double * V, const double * W, size_t r1, size_t r2, size_t r3, 
    size_t& compressed_size, double max_eb, const std::set<size_t>& index_need_to_lossless, 
    int n_threads, double* &decompressed_U_ptr, double* &decompressed_V_ptr, double* &decompressed_W_ptr,std::vector<bool>& cp_exist);
