#include "sz_decompress_3d.hpp"
#include "sz_decompress_cp_preserve_2d.hpp"
#include "sz_decompress_block_processing.hpp"
#include <limits>
#include <unordered_set>
#include <algorithm>
#include <chrono>

inline void factor2D(int n, int &A, int &B) {
    int start = (int)floor(sqrt(n));
    while (start > 0 && n % start != 0) {
        start--;
    }
    A = start;
    B = n / A;
}

inline void factor3D(int n, int &M, int &K, int &L) {
    // 先从立方根附近寻找因子M
    int start = (int)floor(cbrt(n));
    while (start > 0 && n % start != 0) {
        start--;
    }
    M = start;
    int remain = n / M;

    // 对remain进行2D因子分解
    factor2D(remain, K, L);
}
template<typename T>
void
sz_decompress_cp_preserve_3d_online_log(const unsigned char * compressed, size_t r1, size_t r2, size_t r3, T *& U, T *& V, T *& W){
	if(U) free(U);
	if(V) free(V);
	if(W) free(W);
	size_t num_elements = r1 * r2 * r3;
	const unsigned char * compressed_pos = compressed;
	int base = 0;
	read_variable_from_src(compressed_pos, base);
	printf("base = %d\n", base);
	int intv_radius = 0;
	read_variable_from_src(compressed_pos, intv_radius);
	size_t sign_map_size = (num_elements - 1)/8 + 1;
	unsigned char * sign_map_u = convertByteArray2IntArray_fast_1b_sz(num_elements, compressed_pos, sign_map_size);	
	unsigned char * sign_map_v = convertByteArray2IntArray_fast_1b_sz(num_elements, compressed_pos, sign_map_size);	
	unsigned char * sign_map_w = convertByteArray2IntArray_fast_1b_sz(num_elements, compressed_pos, sign_map_size);	
	const int capacity = (intv_radius << 1);
	size_t unpred_data_count = 0;
	read_variable_from_src(compressed_pos, unpred_data_count);
	const T * eb_zero_data = (T *) compressed_pos;
	const T * eb_zero_data_pos = eb_zero_data;
	compressed_pos += unpred_data_count*sizeof(T);
	size_t eb_quant_num = 0;
	read_variable_from_src(compressed_pos, eb_quant_num);
	int * eb_quant_index = Huffman_decode_tree_and_data(2*256, eb_quant_num, compressed_pos);
	size_t data_quant_num = 0;
	read_variable_from_src(compressed_pos, data_quant_num);
	int * data_quant_index = Huffman_decode_tree_and_data(2*capacity, data_quant_num, compressed_pos);
	U = (T *) malloc(num_elements*sizeof(T));
	V = (T *) malloc(num_elements*sizeof(T));
	W = (T *) malloc(num_elements*sizeof(T));
	T * U_pos = U;
	T * V_pos = V;
	T * W_pos = W;
	size_t dim0_offset = r2 * r3;
	size_t dim1_offset = r3;
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	const double threshold=std::numeric_limits<float>::epsilon();
	double log_of_base = log2(base);
	int eb_quant_index_max = (int) (log2(1.0 / threshold)/log_of_base) + 1;
	std::unordered_set<int> unpred_data_indices;
	for(int i=0; i<r1; i++){
		for(int j=0; j<r2; j++){
			for(int k=0; k<r3; k++){
				// printf("%ld %ld %ld\n", i, j, k);
				T * data_pos[3] = {U_pos, V_pos, W_pos};
				int index = i*dim0_offset + j*dim1_offset + k;
				// get eb
				if(*eb_quant_index_pos == 0 || *eb_quant_index_pos == eb_quant_index_max){
					unpred_data_indices.insert(index);
					for(int p=0; p<3; p++){
						T cur_data = *(eb_zero_data_pos ++);
						*(data_pos[p]) = (cur_data == 0) ? -100 : log2f(fabs(cur_data));
					}
					eb_quant_index_pos ++;
				}
				else{
					double eb = (*eb_quant_index_pos == 0) ? 0 : pow(base, *eb_quant_index_pos) * threshold;
					eb_quant_index_pos ++;
					for(int p=0; p<3; p++){
						T * cur_log_data_pos = data_pos[p];					
						T d0 = (i && j && k) ? cur_log_data_pos[- dim0_offset - dim1_offset - 1] : 0;
						T d1 = (i && j) ? cur_log_data_pos[- dim0_offset - dim1_offset] : 0;
						T d2 = (i && k) ? cur_log_data_pos[- dim0_offset - 1] : 0;
						T d3 = (i) ? cur_log_data_pos[- dim0_offset] : 0;
						T d4 = (j && k) ? cur_log_data_pos[- dim1_offset - 1] : 0;
						T d5 = (j) ? cur_log_data_pos[- dim1_offset] : 0;
						T d6 = (k) ? cur_log_data_pos[- 1] : 0;
						T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
						*cur_log_data_pos = pred + 2 * (data_quant_index_pos[p] - intv_radius) * eb;
					}
					data_quant_index_pos += 3;
				}
				U_pos ++;
				V_pos ++;
				W_pos ++;
			}
		}
	}
	printf("recover data done\n");
	eb_zero_data_pos = eb_zero_data;
	for(int i=0; i<num_elements; i++){
		if(unpred_data_indices.count(i)){
			U[i] = *(eb_zero_data_pos++);
			V[i] = *(eb_zero_data_pos++);
			W[i] = *(eb_zero_data_pos++);
		}
		else{
			if(U[i] < -99) U[i] = 0;
			else U[i] = sign_map_u[i] ? exp2(U[i]) : -exp2(U[i]);
			if(V[i] < -99) V[i] = 0;
			else V[i] = sign_map_v[i] ? exp2(V[i]) : -exp2(V[i]);
			if(W[i] < -99) W[i] = 0;
			else W[i] = sign_map_w[i] ? exp2(W[i]) : -exp2(W[i]);
		}
	}
	free(sign_map_u);
	free(sign_map_v);
	free(sign_map_w);
	free(eb_quant_index);
	free(data_quant_index);
}

template
void
sz_decompress_cp_preserve_3d_online_log<float>(const unsigned char * compressed, size_t r1, size_t r2, size_t r3, float *& U, float *& V, float *& W);

template
void
sz_decompress_cp_preserve_3d_online_log<double>(const unsigned char * compressed, size_t r1, size_t r2, size_t r3, double *& U, double *& V, double *& W);

template<typename T>
void
sz_decompress_cp_preserve_3d_unstructured(const unsigned char * compressed, int n, const T * points, int m, const int * tets_ind, T *& data){

	if(data) free(data);
	const unsigned char * compressed_pos = compressed;
	int base = 0;
	read_variable_from_src(compressed_pos, base);
	printf("base = %d\n", base);
	int intv_radius = 0;
	read_variable_from_src(compressed_pos, intv_radius);
	size_t sign_map_size = (3*n - 1)/8 + 1;
	unsigned char * sign_map = convertByteArray2IntArray_fast_1b_sz(3*n, compressed_pos, sign_map_size);	
	const int capacity = (intv_radius << 1);
	size_t unpred_data_count = 0;
	read_variable_from_src(compressed_pos, unpred_data_count);
	const T * eb_zero_data = (T *) compressed_pos;
	const T * eb_zero_data_pos = eb_zero_data;
	compressed_pos += unpred_data_count*sizeof(T);
	size_t eb_quant_num = 0;
	read_variable_from_src(compressed_pos, eb_quant_num);
	int * eb_quant_index = Huffman_decode_tree_and_data(2*256, eb_quant_num, compressed_pos);
	size_t data_quant_num = 0;
	read_variable_from_src(compressed_pos, data_quant_num);
	int * data_quant_index = Huffman_decode_tree_and_data(2*capacity, data_quant_num, compressed_pos);
	data = (T *) malloc(3*n*sizeof(T));
	T * data_pos = data;
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	const double threshold=std::numeric_limits<float>::epsilon();
	double log_of_base = log2(base);
	int eb_quant_index_max = (int) (log2(1.0 / threshold)/log_of_base) + 1;
	std::unordered_set<int> unpred_data_indices;
	for(int i=0; i<n; i++){
		// printf("%ld %ld %ld\n", i, j, k);
		// get eb
		if(*eb_quant_index_pos == 0 || *eb_quant_index_pos == eb_quant_index_max){
			unpred_data_indices.insert(i);
			for(int p=0; p<3; p++){
				T cur_data = *(eb_zero_data_pos ++);
				data_pos[p] = (cur_data == 0) ? -100 : log2f(fabs(cur_data));
			}
			eb_quant_index_pos ++;
		}
		else{
			double eb = (*eb_quant_index_pos == 0) ? 0 : pow(base, *eb_quant_index_pos) * threshold;
			eb_quant_index_pos ++;
			for(int p=0; p<3; p++){
				T * cur_log_data_pos = data_pos + p;					
				T pred = (i) ? cur_log_data_pos[- 3] : 0;
				*cur_log_data_pos = pred + 2 * (data_quant_index_pos[p] - intv_radius) * eb;
			}
			data_quant_index_pos += 3;
		}
		data_pos += 3;
	}
	printf("recover data done\n");
	eb_zero_data_pos = eb_zero_data;
	unsigned char * sign_pos = sign_map;
	for(int i=0; i<n; i++){
		if(unpred_data_indices.count(i)){
			for(int p=0; p<3; p++){
				data[3*i + p] = *(eb_zero_data_pos++);
			}
			sign_pos += 3;
		}
		else{
			for(int p=0; p<3; p++){
				if(data[3*i + p] < -99) data[3*i + p] = 0;
				else data[3*i + p] = *(sign_pos ++) ? exp2(data[3*i + p]) : -exp2(data[3*i + p]);
			}
		}
	}
	free(sign_map);
	free(eb_quant_index);
	free(data_quant_index);
}
template
void
sz_decompress_cp_preserve_3d_unstructured<float>(const unsigned char * compressed, int n, const float * points, int m, const int * tets_ind, float *& data);

template
void
sz_decompress_cp_preserve_3d_unstructured<double>(const unsigned char * compressed, int n, const double * points, int m, const int * tets_ind, double *& data);

template<typename T>
void
sz_decompress_cp_preserve_3d_record_vertex(const unsigned char * compressed, size_t r1, size_t r2, size_t r3, T *& U, T *& V, T *& W){
	if(U) free(U);
	if(V) free(V);
	if(W) free(W);
	size_t num_elements = r1 * r2 * r3;
	const unsigned char * compressed_pos = compressed;
	int base = 0;

	//先读需要无损的大小
	size_t lossless_count = 0;
	read_variable_from_src(compressed_pos, lossless_count);
	printf("lossless_count = %ld\n", lossless_count);
	unsigned char * bitmap;
	T * lossless_data_U;
	T * lossless_data_V;
	T * lossless_data_W;
	if (lossless_count != 0){
		bitmap = (unsigned char *) malloc(num_elements * sizeof(unsigned char));
		size_t num_bytes = (num_elements % 8 == 0) ? num_elements / 8 : num_elements / 8 + 1;
		convertByteArray2IntArray_fast_1b_sz(num_elements, compressed_pos, num_bytes, bitmap);
		lossless_data_U = read_array_from_src<T>(compressed_pos,lossless_count);
		lossless_data_V = read_array_from_src<T>(compressed_pos,lossless_count);
		lossless_data_W = read_array_from_src<T>(compressed_pos,lossless_count);

	}
	//先搞出来bitmap
	// allocate memory for bitmap

	read_variable_from_src(compressed_pos, base);
	printf("base = %d\n", base);
	int intv_radius = 0;
	read_variable_from_src(compressed_pos, intv_radius);
	size_t sign_map_size = (num_elements - 1)/8 + 1;
	unsigned char * sign_map_u = convertByteArray2IntArray_fast_1b_sz(num_elements, compressed_pos, sign_map_size);	
	unsigned char * sign_map_v = convertByteArray2IntArray_fast_1b_sz(num_elements, compressed_pos, sign_map_size);	
	unsigned char * sign_map_w = convertByteArray2IntArray_fast_1b_sz(num_elements, compressed_pos, sign_map_size);	
	const int capacity = (intv_radius << 1);
	size_t unpred_data_count = 0;
	read_variable_from_src(compressed_pos, unpred_data_count);
	const T * eb_zero_data = (T *) compressed_pos;
	const T * eb_zero_data_pos = eb_zero_data;
	compressed_pos += unpred_data_count*sizeof(T);
	size_t eb_quant_num = 0;
	read_variable_from_src(compressed_pos, eb_quant_num);
	int * eb_quant_index = Huffman_decode_tree_and_data(2*256, eb_quant_num, compressed_pos);
	size_t data_quant_num = 0;
	read_variable_from_src(compressed_pos, data_quant_num);
	int * data_quant_index = Huffman_decode_tree_and_data(2*capacity, data_quant_num, compressed_pos);
	U = (T *) malloc(num_elements*sizeof(T));
	V = (T *) malloc(num_elements*sizeof(T));
	W = (T *) malloc(num_elements*sizeof(T));
	T * U_pos = U;
	T * V_pos = V;
	T * W_pos = W;
	size_t dim0_offset = r2 * r3;
	size_t dim1_offset = r3;
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	const double threshold=std::numeric_limits<float>::epsilon();
	double log_of_base = log2(base);
	int eb_quant_index_max = (int) (log2(1.0 / threshold)/log_of_base) + 1;
	std::unordered_set<int> unpred_data_indices;
	for(int i=0; i<r1; i++){
		for(int j=0; j<r2; j++){
			for(int k=0; k<r3; k++){
				// printf("%ld %ld %ld\n", i, j, k);
				T * data_pos[3] = {U_pos, V_pos, W_pos};
				int index = i*dim0_offset + j*dim1_offset + k;
				// get eb
				if(*eb_quant_index_pos == 0 || *eb_quant_index_pos == eb_quant_index_max){
					unpred_data_indices.insert(index);
					for(int p=0; p<3; p++){
						T cur_data = *(eb_zero_data_pos ++);
						*(data_pos[p]) = (cur_data == 0) ? -100 : log2f(fabs(cur_data));
					}
					eb_quant_index_pos ++;
				}
				else{
					double eb = (*eb_quant_index_pos == 0) ? 0 : pow(base, *eb_quant_index_pos) * threshold;
					eb_quant_index_pos ++;
					for(int p=0; p<3; p++){
						T * cur_log_data_pos = data_pos[p];					
						T d0 = (i && j && k) ? cur_log_data_pos[- dim0_offset - dim1_offset - 1] : 0;
						T d1 = (i && j) ? cur_log_data_pos[- dim0_offset - dim1_offset] : 0;
						T d2 = (i && k) ? cur_log_data_pos[- dim0_offset - 1] : 0;
						T d3 = (i) ? cur_log_data_pos[- dim0_offset] : 0;
						T d4 = (j && k) ? cur_log_data_pos[- dim1_offset - 1] : 0;
						T d5 = (j) ? cur_log_data_pos[- dim1_offset] : 0;
						T d6 = (k) ? cur_log_data_pos[- 1] : 0;
						T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
						*cur_log_data_pos = pred + 2 * (data_quant_index_pos[p] - intv_radius) * eb;
					}
					data_quant_index_pos += 3;
				}
				U_pos ++;
				V_pos ++;
				W_pos ++;
			}
		}
	}
	printf("recover data done\n");
	eb_zero_data_pos = eb_zero_data;
	for(int i=0; i<num_elements; i++){
		if(unpred_data_indices.count(i)){
			U[i] = *(eb_zero_data_pos++);
			V[i] = *(eb_zero_data_pos++);
			W[i] = *(eb_zero_data_pos++);
		}
		else{
			if(U[i] < -99) U[i] = 0;
			else U[i] = sign_map_u[i] ? exp2(U[i]) : -exp2(U[i]);
			if(V[i] < -99) V[i] = 0;
			else V[i] = sign_map_v[i] ? exp2(V[i]) : -exp2(V[i]);
			if(W[i] < -99) W[i] = 0;
			else W[i] = sign_map_w[i] ? exp2(W[i]) : -exp2(W[i]);
		}
	}
	if (lossless_count != 0){
		T * lossless_data_U_pos = lossless_data_U;
		T * lossless_data_V_pos = lossless_data_V;
		T * lossless_data_W_pos = lossless_data_W;
		//最后再根据bitmap来处理lossless的数据
		for(int i=0; i< r1; i++){
			for(int j=0; j< r2; j++){
				for(int k=0; k< r3; k++){
					if(static_cast<int>(bitmap[i*r2*r3 + j*r3 + k])){
						U[i*r2*r3 + j*r3 + k] = *(lossless_data_U_pos++);
						V[i*r2*r3 + j*r3 + k] = *(lossless_data_V_pos++);
						W[i*r2*r3 + j*r3 + k] = *(lossless_data_W_pos++);
						
					}
				}
			}
		}
		free(bitmap);
		free(lossless_data_U);
		free(lossless_data_V);
		free(lossless_data_W);
	}
	free(sign_map_u);
	free(sign_map_v);
	free(sign_map_w);
	free(eb_quant_index);
	free(data_quant_index);

}

template
void
sz_decompress_cp_preserve_3d_record_vertex<float>(const unsigned char * compressed, size_t r1, size_t r2, size_t r3, float *& U, float *& V, float *& W);

/* decomp for 3d abs error bound */

template<typename T>
void
sz_decompress_cp_preserve_3d_online_abs_record_vertex(const unsigned char * compressed, size_t r1, size_t r2, size_t r3, T *& U, T *& V, T *& W){
	if(U) free(U);
	if(V) free(V);
	if(W) free(W);
	size_t num_elements = r1 * r2 * r3;
	const unsigned char * compressed_pos = compressed;
	int base = 0;
	unsigned char * bitmap;
	T * lossless_data_U = NULL;
	T * lossless_data_V = NULL;
	T * lossless_data_W = NULL;
	T * lossless_data_U_pos = NULL;
	T * lossless_data_V_pos = NULL;
	T * lossless_data_W_pos = NULL;
	//先搞出来需要无损的大小
	size_t lossless_count = 0;
	read_variable_from_src(compressed_pos, lossless_count);
	printf("lossless_count = %ld\n", lossless_count);

	//先搞出来bitmap,如果lossless_count不为0
	if (lossless_count != 0){
		// allocate memory for bitmap
		bitmap = (unsigned char *) malloc(num_elements * sizeof(unsigned char));
		memset(bitmap, 0, num_elements * sizeof(unsigned char));
		size_t num_bytes = (num_elements % 8 == 0) ? num_elements / 8 : num_elements / 8 + 1;
		convertByteArray2IntArray_fast_1b_sz(num_elements, compressed_pos, num_bytes, bitmap);
		printf("bitmap done\n");
		lossless_data_U = read_array_from_src<T>(compressed_pos,lossless_count);
		lossless_data_V = read_array_from_src<T>(compressed_pos,lossless_count);
		lossless_data_W = read_array_from_src<T>(compressed_pos,lossless_count);
		lossless_data_U_pos = lossless_data_U;
		lossless_data_V_pos = lossless_data_V;
		lossless_data_W_pos = lossless_data_W;
	}



	read_variable_from_src(compressed_pos, base);
	printf("base = %d\n", base);
	double threshold = 0;
	read_variable_from_src(compressed_pos, threshold);
	int intv_radius = 0;
	read_variable_from_src(compressed_pos, intv_radius);
	const int capacity = (intv_radius << 1);
	size_t data_quant_num = 0;
	read_variable_from_src(compressed_pos, data_quant_num);
	size_t unpred_data_count = 0;
	read_variable_from_src(compressed_pos, unpred_data_count);
	const T * unpred_data_pos = (T *) compressed_pos;
	compressed_pos += unpred_data_count*sizeof(T);	
	int * eb_quant_index = Huffman_decode_tree_and_data(2*1024, num_elements, compressed_pos);
	int * data_quant_index = Huffman_decode_tree_and_data(2*capacity, data_quant_num, compressed_pos);
	U = (T *) malloc(num_elements*sizeof(T));
	V = (T *) malloc(num_elements*sizeof(T));
	W = (T *) malloc(num_elements*sizeof(T));
	T * U_pos = U;
	T * V_pos = V;
	T * W_pos = W;
	size_t dim0_offset = r2 * r3;
	size_t dim1_offset = r3;
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	double log_of_base = log2(base);
	int eb_quant_index_max = (int) (log2(1.0 / threshold)/log_of_base) + 1;
	std::unordered_set<int> unpred_data_indices;
	for(int i=0; i<r1; i++){
		for(int j=0; j<r2; j++){
			for(int k=0; k<r3; k++){
				// printf("%ld %ld %ld\n", i, j, k);
				T * data_pos[3] = {U_pos, V_pos, W_pos};
				int index = i*dim0_offset + j*dim1_offset + k;
				// get eb
				if(*eb_quant_index_pos == 0){
					for(int p=0; p<3; p++){
						*(data_pos[p]) = *(unpred_data_pos ++);
					}
					eb_quant_index_pos ++;
				}
				else{
					double eb = pow(base, *eb_quant_index_pos) * threshold;
					eb_quant_index_pos ++;
					for(int p=0; p<3; p++){
						T * cur_log_data_pos = data_pos[p];					
						T d0 = (i && j && k) ? cur_log_data_pos[- dim0_offset - dim1_offset - 1] : 0;
						T d1 = (i && j) ? cur_log_data_pos[- dim0_offset - dim1_offset] : 0;
						T d2 = (i && k) ? cur_log_data_pos[- dim0_offset - 1] : 0;
						T d3 = (i) ? cur_log_data_pos[- dim0_offset] : 0;
						T d4 = (j && k) ? cur_log_data_pos[- dim1_offset - 1] : 0;
						T d5 = (j) ? cur_log_data_pos[- dim1_offset] : 0;
						T d6 = (k) ? cur_log_data_pos[- 1] : 0;
						T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
						*cur_log_data_pos = pred + 2 * (data_quant_index_pos[p] - intv_radius) * eb;
					}
					data_quant_index_pos += 3;
				}
				U_pos ++;
				V_pos ++;
				W_pos ++;
			}
		}
	}
	//最后再根据bitmap来处理lossless的数据
	if(lossless_count != 0){
		for(int i=0; i< r1; i++){
			for(int j=0; j< r2; j++){
				for(int k=0; k< r3; k++){
					if(static_cast<int>(bitmap[i*r2*r3 + j*r3 + k])){
						U[i*r2*r3 + j*r3 + k] = *(lossless_data_U_pos++);
						V[i*r2*r3 + j*r3 + k] = *(lossless_data_V_pos++);
						W[i*r2*r3 + j*r3 + k] = *(lossless_data_W_pos++);
						
					}
				}
			}
		}
		free(lossless_data_U);
		free(lossless_data_V);
		free(lossless_data_W);
		free(bitmap);
	}
	free(eb_quant_index);
	free(data_quant_index);
}

template
void
sz_decompress_cp_preserve_3d_online_abs_record_vertex<float>(const unsigned char * compressed, size_t r1, size_t r2, size_t r3, float *& U, float *& V, float *& W);

template
void
sz_decompress_cp_preserve_3d_online_abs_record_vertex<double>(const unsigned char * compressed, size_t r1, size_t r2, size_t r3, double *& U, double *& V, double *& W);

template<typename T>
void
omp_sz_decompress_cp_preserve_3d_online_abs_record_vertex(const unsigned char * compressed, size_t r1, size_t r2, size_t r3, T *& U, T *& V, T *& W){
	if(U) free(U);
	if(V) free(V);
	if(W) free(W);

	size_t num_elements = r1 * r2 * r3;
	const unsigned char * compressed_pos = compressed;
	unsigned char * bitmap;
	T * lossless_data_U = NULL;
	T * lossless_data_V = NULL;
	T * lossless_data_W = NULL;
	T * lossless_data_U_pos = NULL;
	T * lossless_data_V_pos = NULL;
	T * lossless_data_W_pos = NULL;
	//first read index_need_to_fix_size
	size_t index_need_to_fix_size = 0;
	read_variable_from_src(compressed_pos, index_need_to_fix_size);
	printf("decomp: index_need_to_fix_size = %ld\n", index_need_to_fix_size);
	//if not 0, then read bitmap and lossless data
	if (index_need_to_fix_size != 0){
		// allocate memory for bitmap
		bitmap = (unsigned char *) malloc(num_elements * sizeof(unsigned char));
		memset(bitmap, 0, num_elements * sizeof(unsigned char));
		size_t num_bytes = (num_elements % 8 == 0) ? num_elements / 8 : num_elements / 8 + 1;
		convertByteArray2IntArray_fast_1b_sz(num_elements, compressed_pos, num_bytes, bitmap);
		//再搞出来需要无损的大小
		size_t lossless_count = index_need_to_fix_size;
		// read_variable_from_src(compressed_pos, lossless_count);
		// printf("lossless_count = %ld\n", lossless_count);
		// allocate memory for lossless data
		lossless_data_U = read_array_from_src<T>(compressed_pos,lossless_count);
		lossless_data_V = read_array_from_src<T>(compressed_pos,lossless_count);
		lossless_data_W = read_array_from_src<T>(compressed_pos,lossless_count);
		lossless_data_U_pos = lossless_data_U;
		lossless_data_V_pos = lossless_data_V;
		lossless_data_W_pos = lossless_data_W;
	}

	int num_threads = 0;
	read_variable_from_src(compressed_pos, num_threads);
	printf("dec num_threads = %d\n", num_threads);

	int M,K,L;
	factor3D(num_threads, M, K, L);
	printf("decomp M = %d, K = %d, L = %d\n", M, K, L);
	// int num_faces_x = (M-1)*K*L;
	// int num_faces_y = M*(K-1)*L;
	// int num_faces_z = M*K*(L-1);
	// int num_edges_x = (M-1)*(K-1)*L;
	// int num_edges_y = K * (M-1)*(L-1);
	// int num_edges_z = M * (K - 1) * (L - 1);
	int num_faces_x = M*K*L;
	int num_faces_y = M*K*L;
	int num_faces_z = M*K*L;
	int num_edges_x = M*K*L;
	int num_edges_y = M*K*L;
	int num_edges_z = M*K*L;

	auto read_variables_time_start = std::chrono::high_resolution_clock::now();

	int sum_unpred_size = 0;
	// 读每一个block的unpred_data size
	std::vector<size_t> unpred_data_count(num_threads);
	for (int i = 0; i < num_threads; i++){
		read_variable_from_src(compressed_pos, unpred_data_count[i]);
		sum_unpred_size += unpred_data_count[i];
	}
	printf("dec sum_unpred_size = %d\n", sum_unpred_size);
	
	//读每个block的unpred_data
	std::vector<std::vector<T>> unpred_data_each_thread(num_threads);
	for (int i = 0; i < num_threads; i++){
		T * raw_data = read_array_from_src<T>(compressed_pos, unpred_data_count[i]);
		unpred_data_each_thread[i] = std::vector<T>(raw_data, raw_data + unpred_data_count[i]);
		free(raw_data);
		//compressed_pos is the sum of previous unpred_data_count
		//read_array_from_src_to_stdarray<T>(compressed_pos, unpred_data_each_thread[i], unpred_data_count[i]);
	}

	// int t = std::round(std::cbrt(num_threads));
	// int num_faces = (t-1) * t * t;
	// int num_edges = t*(t-1)*(t-1);
	//读每个face_x的unpred_data size
	std::vector<size_t> face_x_data_count(num_faces_x);
	for (int i = 0; i < num_faces_x; i++){
		read_variable_from_src(compressed_pos, face_x_data_count[i]);
	}
	//读每个face_x的unpred_data
	std::vector<std::vector<T>> face_x_data_each_thread(num_faces_x);
	for (int i = 0; i < num_faces_x; i++){
		T * raw_data = read_array_from_src<T>(compressed_pos, face_x_data_count[i]);
		face_x_data_each_thread[i] = std::vector<T>(raw_data, raw_data + face_x_data_count[i]);
		free(raw_data);
	}
	//读每个face_y的unpred_data size
	std::vector<size_t> face_y_data_count(num_faces_y);
	for (int i = 0; i < num_faces_y; i++){
		read_variable_from_src(compressed_pos, face_y_data_count[i]);
	}
	//读每个face_y的unpred_data
	std::vector<std::vector<T>> face_y_data_each_thread(num_faces_y);
	for (int i = 0; i < num_faces_y; i++){
		T * raw_data = read_array_from_src<T>(compressed_pos, face_y_data_count[i]);
		face_y_data_each_thread[i] = std::vector<T>(raw_data, raw_data + face_y_data_count[i]);
		free(raw_data);
	}
	//读每个face_z的unpred_data size
	std::vector<size_t> face_z_data_count(num_faces_z);
	for (int i = 0; i < num_faces_z; i++){
		read_variable_from_src(compressed_pos, face_z_data_count[i]);
	}
	//读每个face_z的unpred_data
	std::vector<std::vector<T>> face_z_data_each_thread(num_faces_z);
	for (int i = 0; i < num_faces_z; i++){
		T * raw_data = read_array_from_src<T>(compressed_pos, face_z_data_count[i]);
		face_z_data_each_thread[i] = std::vector<T>(raw_data, raw_data + face_z_data_count[i]);
		free(raw_data);
	}
	//读每个edge_x的unpred_data size
	std::vector<size_t> edge_x_data_count(num_edges_x);
	for (int i = 0; i < num_edges_x; i++){
		read_variable_from_src(compressed_pos, edge_x_data_count[i]);
	}
	//读每个edge_x的unpred_data
	std::vector<std::vector<T>> edge_x_data_each_thread(num_edges_x);
	for (int i = 0; i < num_edges_x; i++){
		T * raw_data = read_array_from_src<T>(compressed_pos, edge_x_data_count[i]);
		edge_x_data_each_thread[i] = std::vector<T>(raw_data, raw_data + edge_x_data_count[i]);
		free(raw_data);
	}
	int sum_unpred_data_edge_y = 0;
	double total_sum= 0;
	//读每个edge_y的unpred_data size
	std::vector<size_t> edge_y_data_count(num_edges_y);
	for (int i = 0; i < num_edges_y; i++){
		read_variable_from_src(compressed_pos, edge_y_data_count[i]);
		sum_unpred_data_edge_y += edge_y_data_count[i];
	}
	//读每个edge_y的unpred_data
	std::vector<std::vector<T>> edge_y_data_each_thread(num_edges_y);
	for (int i = 0; i < num_edges_y; i++){
		T * raw_data = read_array_from_src<T>(compressed_pos, edge_y_data_count[i]);
		edge_y_data_each_thread[i] = std::vector<T>(raw_data, raw_data + edge_y_data_count[i]);
		free(raw_data);
		for (int j = 0; j < edge_y_data_count[i]; j++){
			total_sum += edge_y_data_each_thread[i][j];
		}
	}
	printf("decomp: sum_unpred_data_edges_y = %d\n", sum_unpred_data_edge_y);
	printf("decomp: total_sum = %f\n", total_sum);
	for (int i = 0; i < edge_y_data_each_thread[1].size(); i++){
		printf("decomp: edge_y_data_each_thread[1][%d] = %f\n", i, edge_y_data_each_thread[1][i]);
	}
	printf("====================================\n");
	//读每个edge_z的unpred_data size
	std::vector<size_t> edge_z_data_count(num_edges_z);
	for (int i = 0; i < num_edges_z; i++){
		read_variable_from_src(compressed_pos, edge_z_data_count[i]);
	}
	//读每个edge_z的unpred_data
	std::vector<std::vector<T>> edge_z_data_each_thread(num_edges_z);
	for (int i = 0; i < num_edges_z; i++){
		T * raw_data = read_array_from_src<T>(compressed_pos, edge_z_data_count[i]);
		edge_z_data_each_thread[i] = std::vector<T>(raw_data, raw_data + edge_z_data_count[i]);
		free(raw_data);
	}
	// 读 dot的unpred_data size
	size_t dot_data_count = 0;
	read_variable_from_src(compressed_pos, dot_data_count);
	// 读dot的unpred_data
	T * dot_data = read_array_from_src<T>(compressed_pos, dot_data_count);
	//print sum of dot_data
	double sum_dot_data = 0;
	for (int i = 0; i < dot_data_count; i++){
		sum_dot_data += dot_data[i];
	}
	printf("sum of dot_data = %f\n", sum_dot_data);
	printf("count of dot_data = %ld\n", dot_data_count);
	auto read_variables_time_end = std::chrono::high_resolution_clock::now();
	printf("time for get other variables in second: %f\n", std::chrono::duration<double>(read_variables_time_end - read_variables_time_start).count());
	
	int base = 0;
	read_variable_from_src(compressed_pos, base);
	printf("decomp base = %d\n", base);
	double threshold = 0;
	read_variable_from_src(compressed_pos, threshold);
	int intv_radius = 0;
	read_variable_from_src(compressed_pos, intv_radius);
	const int capacity = (intv_radius << 1);

	auto readAndDecode_EbQuant_start = std::chrono::high_resolution_clock::now();
	// now read eb_quant_index
	//int * eb_quant_index = Huffman_decode_tree_and_data(2*1024, num_elements, compressed_pos);
	// 并行
	// read each compressed_size for eb_quant_index first
	std::vector<size_t> compressed_size_eb(num_threads);
	for (int i = 0; i < num_threads; i++){
		read_variable_from_src(compressed_pos, compressed_size_eb[i]);
	}
	std::vector<const unsigned char *> compressed_chunk_eb_start(num_threads);
	for (int i = 0; i < num_threads; i++){
		compressed_chunk_eb_start[i] = compressed_pos;
		compressed_pos += compressed_size_eb[i];
	}
	int * eb_quant_index = (int *) malloc(num_elements * sizeof(int));
	#pragma omp parallel for num_threads(num_threads)
	for (int i = 0; i < num_threads; i++){
		size_t start_pos = i*num_elements/num_threads;
		size_t end_pos = (i ==num_threads - 1) ? num_elements : (i+1)*num_elements/num_threads;
		const unsigned char * local_compressed_pos = compressed_chunk_eb_start[i];
		size_t local_num_elements = end_pos - start_pos;
		int * local_eb_quant_index = Huffman_decode_tree_and_data(2*1024, local_num_elements, local_compressed_pos);
		std::copy(local_eb_quant_index, local_eb_quant_index + local_num_elements, eb_quant_index + start_pos);
		free(local_eb_quant_index);
	}
	
	auto readAndDecode_EbQuant_end = std::chrono::high_resolution_clock::now();
	// now read data_quant_index
	//int * data_quant_index = Huffman_decode_tree_and_data(2*capacity, 3*num_elements, compressed_pos);
	
	// 并行
	// read each compressed_size for data_quant_index first
	auto readAndDecode_DataQuant_start = std::chrono::high_resolution_clock::now();
	auto readOnly_DataQuant_start = std::chrono::high_resolution_clock::now();
	std::vector<size_t> compressed_size_data(num_threads);
	for (int i = 0; i < num_threads; i++){
		read_variable_from_src(compressed_pos, compressed_size_data[i]);
	}
	std::vector<const unsigned char *> compressed_chunk_data_start(num_threads);
	for (int i = 0; i < num_threads; i++){
		compressed_chunk_data_start[i] = compressed_pos;
		compressed_pos += compressed_size_data[i];
	}
	auto readOnly_DataQuant_end = std::chrono::high_resolution_clock::now();
	int * data_quant_index = (int *) malloc(3*num_elements * sizeof(int));
	#pragma omp parallel for num_threads(num_threads)
	for (int i = 0; i < num_threads; i++){
		size_t start_pos = i*num_elements/num_threads;
		size_t end_pos = (i ==num_threads - 1) ? num_elements : (i+1)*num_elements/num_threads;
		const unsigned char * local_compressed_pos = compressed_chunk_data_start[i];
		size_t local_num_elements = 3*(end_pos - start_pos);
		int * local_data_quant_index = Huffman_decode_tree_and_data(capacity, local_num_elements, local_compressed_pos);
		//auto copy_start = std::chrono::high_resolution_clock::now();
		std::copy(local_data_quant_index, local_data_quant_index + local_num_elements, data_quant_index + 3*start_pos);
		free(local_data_quant_index);
		//auto copy_end = std::chrono::high_resolution_clock::now();
		//total_time_copy += std::chrono::duration<double>(copy_end - copy_start).count();
	}
	auto readAndDecode_DataQuant_end = std::chrono::high_resolution_clock::now();
	printf("abs mode\n");
	printf("time for read and decode eb_quant_index in second: %f\n", std::chrono::duration<double>(readAndDecode_EbQuant_end - readAndDecode_EbQuant_start).count());
	printf("time for read and decode data_quant_index in second: %f(readonly: %f)\n", std::chrono::duration<double>(readAndDecode_DataQuant_end - readAndDecode_DataQuant_start).count(), std::chrono::duration<double>(readOnly_DataQuant_end - readOnly_DataQuant_start).count());
	printf("decomp pos after huffmans = %ld\n", compressed_pos - compressed);


	auto pred_time_start = std::chrono::high_resolution_clock::now();
	U = (T *) calloc(num_elements*sizeof(T),sizeof(T));
	V = (T *) calloc(num_elements*sizeof(T),sizeof(T));
	W = (T *) calloc(num_elements*sizeof(T),sizeof(T));
	T * U_pos = U;
	T * V_pos = V;
	T * W_pos = W;
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;

	double log_of_base = log2(base);
	int eb_quant_index_max = (int) (log2(1.0 / threshold)/log_of_base) + 1;


	// 计算每个块的大小
	// int block_r1 = r1 / t;
	// int block_r2 = r2 / t;
	// int block_r3 = r3 / t;
	int block_r1 = r1 / M;
	int block_r2 = r2 / K;
	int block_r3 = r3 / L;

	//处理余数的情况
	// int remaining_r1 = r1 % t;
	// int remaining_r2 = r2 % t;
	// int remaining_r3 = r3 % t;
	int remaining_r1 = r1 % M;
	int remaining_r2 = r2 % K;
	int remaining_r3 = r3 % L;
	//储存划分平面的位置
	std::vector<int> dividing_r1;
    std::vector<int> dividing_r2;
    std::vector<int> dividing_r3;
    // for (int i = 1; i < t; ++i) {
    //     dividing_r1.push_back(i * block_r1);
    //     dividing_r2.push_back(i * block_r2);
    //     dividing_r3.push_back(i * block_r3);
    // }
	for (int i = 1; i < M; ++i) {
		dividing_r1.push_back(i * block_r1);
	}
	for (int i = 1; i < K; ++i) {
		dividing_r2.push_back(i * block_r2);
	}
	for (int i = 1; i < L; ++i) {
		dividing_r3.push_back(i * block_r3);
	}
	for (int i = 0; i < dividing_r1.size(); i++) {
		printf("decomp dividing_r1[%d] = %d\n", i, dividing_r1[i]);
	}
	for (int i = 0; i < dividing_r2.size(); i++) {
		printf("decomp dividing_r2[%d] = %d\n", i, dividing_r2[i]);
	}
	for (int i = 0; i < dividing_r3.size(); i++) {
		printf("decomp dividing_r3[%d] = %d\n", i, dividing_r3[i]);
	}
	// for (int i = 0; i < dividing_r1.size(); i++) {
	// 	printf("decomp dividing_r1[%d] = %d\n", i, dividing_r1[i]);
	// }

	// 创建一个一维标记数组，标记哪些数据点位于划分线上
	//std::vector<bool> on_dividing_line(num_elements, false);
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
	// #pragma omp parallel for collapse(3)
	// for (int i = 0; i < r1; ++i) {
	// 	for (int j = 0; j < r2; ++j) {
	// 		for (int k = 0; k < r3; ++k) {
	// 			if (is_dividing_r1[i] || is_dividing_r2[j] || is_dividing_r3[k]) {
	// 				on_dividing_line[i * r2 * r3 + j * r3 + k] = true;
	// 			}
	// 		}
	// 	}
	// }
	
	size_t processed_data_dec_count = 0;
	// 处理块
	int total_block = M * K * L;
	omp_set_num_threads(total_block);
	//omp_set_proc_bind(omp_proc_bind_true);
	#pragma omp parallel for num_threads(num_threads) 
	for (int block_id = 0; block_id < total_block; ++block_id){
		// int block_i = block_id / (t * t);
		// int block_j = (block_id % (t * t)) / t;
		// int block_k = block_id % t;
		int block_i = block_id / (K * L);
		int block_j = (block_id % (K * L)) / L;
		int block_k = block_id % L;
		// 计算块的起始位置
		int start_i = block_i * block_r1;
		int start_j = block_j * block_r2;
		int start_k = block_k * block_r3;

		int end_i = start_i + block_r1;
		int end_j = start_j + block_r2;
		int end_k = start_k + block_r3;
		if (block_i == M - 1) {
			end_i += remaining_r1;
		}
		if (block_j == K - 1) {
			end_j += remaining_r2;
		}
		if (block_k == L - 1) {
			end_k += remaining_r3;
		}
		//printf("Thread %d process block %d, start_i = %d, end_i = %d, start_j = %d, end_j = %d, start_k = %d, end_k = %d\n", omp_get_thread_num(), block_id, start_i, end_i, start_j, end_j, start_k, end_k);
		T *unpred_data_pos = &unpred_data_each_thread[omp_get_thread_num()][0];

		//处理块内部的数据
		for (int i = start_i; i < end_i; ++i) {
			// if (std::find(dividing_r1.begin(), dividing_r1.end(), i) != dividing_r1.end()) {
			// 	continue;
			// }
			if (is_dividing_r1[i]) {
				continue;
			}
			for (int j = start_j; j < end_j; ++j) {
				// if (std::find(dividing_r2.begin(), dividing_r2.end(), j) != dividing_r2.end()) {
				// 	continue;
				// }
				if (is_dividing_r2[j]) {
					continue;
				}
				for (int k = start_k; k < end_k; ++k) {
					// if (std::find(dividing_r3.begin(), dividing_r3.end(), k) != dividing_r3.end()) {
					// 	continue;
					// }
					if (is_dividing_r3[k]) {
						continue;
					}
					//开始处理数据
					//processed_data_dec_count++;
					size_t position_idx = i * r2 * r3 + j * r3 + k;
					//get eb
					if (eb_quant_index[position_idx] == 0){
						U[position_idx] = *(unpred_data_pos++);
						V[position_idx] = *(unpred_data_pos++);
						W[position_idx] = *(unpred_data_pos++);
					}
					else{
						double eb = pow(base, eb_quant_index[position_idx]) * threshold;
						for (int p = 0; p < 3; ++p){
							T *cur_data_field = (p == 0) ? U : (p == 1) ? V : W;
							T d0 = ((i && j && k) && (i - start_i != 1 && j - start_j != 1 && k - start_k != 1)) ? cur_data_field[position_idx - r2 * r3 - r3 - 1] : 0;
							T d1 = ((i && j) && (i - start_i != 1 && j - start_j != 1)) ? cur_data_field[position_idx - r2 * r3 - r3] : 0;
							T d2 = ((i && k) && (i - start_i != 1 && k - start_k != 1)) ? cur_data_field[position_idx - r2 * r3 - 1] : 0;
							T d3 = (i && (i - start_i != 1)) ? cur_data_field[position_idx - r2 * r3] : 0;
							T d4 = ((j && k) && (j - start_j != 1 && k - start_k != 1)) ? cur_data_field[position_idx - r3 - 1] : 0;
							T d5 = (j && (j - start_j != 1)) ? cur_data_field[position_idx - r3] : 0;
							T d6 = (k && (k - start_k != 1)) ? cur_data_field[position_idx - 1] : 0;
							T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
							cur_data_field[position_idx] = pred + 2 * (data_quant_index[3 * position_idx + p] - intv_radius) * eb;
						}
					}
				}
			}
		}
		
	}




	//优化: 并行解压TODO
	ptrdiff_t dim0_offset = r2 * r3;
	ptrdiff_t dim1_offset = r3;
	ptrdiff_t cell_dim0_offset = (r2-1) * (r3-1);
	ptrdiff_t cell_dim1_offset = r3-1;



	//并行处理face_x
	omp_set_num_threads(num_faces_x);
	// Process faces perpendicular to the X-axis(jk-plane)
	#pragma omp parallel for collapse(3) 
	for (int i : dividing_r1){
		for (int j = -1; j < (int)dividing_r2.size(); j++){
			for (int k = -1; k < (int)dividing_r3.size(); k++){
				int thread_id = omp_get_thread_num();
				int x_layer = i;
				int start_j = (j == -1) ? 0 : dividing_r2[j];
				int end_j = (j == (int)dividing_r2.size() - 1) ? r2 : dividing_r2[j + 1];
				int start_k = (k == -1) ? 0 : dividing_r3[k];
				int end_k = (k == (int)dividing_r3.size() - 1) ? r3 : dividing_r3[k + 1];
				T *unpred_data_pos = &face_x_data_each_thread[thread_id][0];
				for (int jj = start_j; jj < end_j; ++jj){
					// if (std::find(dividing_r2.begin(), dividing_r2.end(), j) != dividing_r2.end()){
					// 	continue;
					// }
					if (is_dividing_r2[jj]){
						continue;
					}
					for (int kk = start_k; kk < end_k; ++kk){
						// if (std::find(dividing_r3.begin(), dividing_r3.end(), k) != dividing_r3.end()){
						// 	continue;
						// }
						if (is_dividing_r3[kk]){
							continue;
						}
						//开始处理数据
						size_t position_idx = i * r2 * r3 + jj * r3 + kk;
						//processed_data_dec_count++;
						if (eb_quant_index[position_idx] == 0){
							U[position_idx] = *(unpred_data_pos++);
							V[position_idx] = *(unpred_data_pos++);
							W[position_idx] = *(unpred_data_pos++);
						}
						else{
							double eb = pow(base, eb_quant_index[position_idx]) * threshold;
							for (int p = 0; p < 3; ++p){
								//degrade to 2D
								T *cur_data_field = (p == 0) ? U : (p == 1) ? V : W;
								T d0 = ((jj && kk) && (jj - start_j != 1 && kk - start_k != 1)) ? cur_data_field[position_idx - r2 - 1] : 0;
								T d1 = (jj && (jj - start_j != 1)) ? cur_data_field[position_idx - r2] : 0;
								T d2 = (kk && (kk - start_k != 1)) ? cur_data_field[position_idx - 1] : 0;
								T pred = d1 + d2 - d0;
								cur_data_field[position_idx] = pred + 2 * (data_quant_index[3 * position_idx + p] - intv_radius) * eb;
							}
						}
					}
				}
			}
		}
	}

	//并行处理face_y
	omp_set_num_threads(num_faces_y);
	#pragma omp parallel for collapse(3) 
	for (int j : dividing_r2){
		for (int i = -1; i < (int)dividing_r1.size(); i++){
			for (int k = -1; k < (int)dividing_r3.size(); k++){
				int thread_id = omp_get_thread_num();
				int y_layer = j;
				int start_i = (i == -1) ? 0 : dividing_r1[i];
				int end_i = (i == (int)dividing_r1.size() - 1) ? r1 : dividing_r1[i + 1];
				int start_k = (k == -1) ? 0 : dividing_r3[k];
				int end_k = (k == (int)dividing_r3.size() - 1) ? r3 : dividing_r3[k + 1];
				T *unpred_data_pos = &face_y_data_each_thread[thread_id][0];
				for (int ii = start_i; ii < end_i; ++ii){
					// if (std::find(dividing_r1.begin(), dividing_r1.end(), i) != dividing_r1.end()){
					// 	continue;
					// }
					if (is_dividing_r1[ii]){
						continue;
					}
					for (int kk = start_k; kk < end_k; ++kk){
						// if (std::find(dividing_r3.begin(), dividing_r3.end(), k) != dividing_r3.end()){
						// 	continue;
						// }
						if (is_dividing_r3[kk]){
							continue;
						}
						//开始处理数据
						size_t position_idx = ii * r2 * r3 + j * r3 + kk;
						//processed_data_dec_count++;
						if (eb_quant_index[position_idx] == 0){
							U[position_idx] = *(unpred_data_pos++);
							V[position_idx] = *(unpred_data_pos++);
							W[position_idx] = *(unpred_data_pos++);
						}
						else{
							double eb = pow(base, eb_quant_index[position_idx]) * threshold;
							for (int p = 0; p < 3; ++p){
								//degrade to 2D
								T *cur_data_field = (p == 0) ? U : (p == 1) ? V : W;
									T d0 = ((ii && kk) && (ii - start_i != 1 && kk - start_k != 1)) ? cur_data_field[position_idx - r2*r3 - 1] : 0;
									T d1 = (ii && (ii - start_i != 1)) ? cur_data_field[position_idx - r2*r3] : 0;
									T d2 = (kk && (kk - start_k != 1)) ? cur_data_field[position_idx - 1] : 0;
									T pred = d1 + d2 - d0;
									cur_data_field[position_idx] = pred + 2 * (data_quant_index[3 * position_idx + p] - intv_radius) * eb;
							}
						}
					}
				}
			}
		}
	}

	//并行处理face_z
	omp_set_num_threads(num_faces_z);
	#pragma omp parallel for collapse(3) 
	for (int k : dividing_r3){
		for (int i = -1; i < (int)dividing_r1.size(); i++){
			for (int j = -1; j < (int)dividing_r2.size(); j++){
				int thread_id = omp_get_thread_num();
				int z_layer = k;
				int start_i = (i == -1) ? 0 : dividing_r1[i];
				int end_i = (i == (int)dividing_r1.size() - 1) ? r1 : dividing_r1[i + 1];
				int start_j = (j == -1) ? 0 : dividing_r2[j];
				int end_j = (j == (int)dividing_r2.size() - 1) ? r2 : dividing_r2[j + 1];
				T *unpred_data_pos = &face_z_data_each_thread[thread_id][0];
				for (int ii = start_i; ii < end_i; ++ii){
					// if (std::find(dividing_r1.begin(), dividing_r1.end(), i) != dividing_r1.end()){
					// 	continue;
					// }
					if (is_dividing_r1[ii]){
						continue;
					}
					for (int jj = start_j; jj < end_j; ++jj){
						// if (std::find(dividing_r2.begin(), dividing_r2.end(), j) != dividing_r2.end()){
						// 	continue;
						// }
						if (is_dividing_r2[jj]){
							continue;
						}
						//开始处理数据
						size_t position_idx = ii * r2 * r3 + jj * r3 + k;
						//processed_data_dec_count++;
						if (eb_quant_index[position_idx] == 0){
							U[position_idx] = *(unpred_data_pos++);
							V[position_idx] = *(unpred_data_pos++);
							W[position_idx] = *(unpred_data_pos++);
						}
						else{
							double eb = pow(base, eb_quant_index[position_idx]) * threshold;
							for (int p = 0; p < 3; ++p){
								//degrade to 2D
								T *cur_data_field = (p == 0) ? U : (p == 1) ? V : W;
								T d0 = ((ii && jj) && (ii - start_i != 1 && jj - start_j != 1)) ? cur_data_field[position_idx - r2*r3 - r3] : 0;
								T d1 = (ii && (ii - start_i != 1)) ? cur_data_field[position_idx - r2*r3] : 0;
								T d2 = (jj && (jj - start_j != 1)) ? cur_data_field[position_idx - r3] : 0;
								T pred = d1 + d2 - d0;
								cur_data_field[position_idx] = pred + 2 * (data_quant_index[3 * position_idx + p] - intv_radius) * eb;
							}
						}
					}
				}
			}
		}
	}
	printf("done face...\n");


	//并行处理edge_x
	omp_set_num_threads(num_edges_x);
	#pragma omp parallel for collapse(3) 
	for (int i : dividing_r1){
		for (int j :dividing_r2){
			for (int k = -1; k < (int)dividing_r3.size(); k++){
				int thread_id = omp_get_thread_num();
				int x_layer = i;
				int y_layer = j;
				int start_k = (k == -1) ? 0 : dividing_r3[k];
				int end_k = (k == (int)dividing_r3.size() - 1) ? r3 : dividing_r3[k + 1];
				T *unpred_data_pos = &edge_x_data_each_thread[thread_id][0];
				for (int kk = start_k; kk < end_k; ++kk){
					// if (std::find(dividing_r3.begin(), dividing_r3.end(), k) != dividing_r3.end()){
					// 	continue;
					// }
					if (is_dividing_r3[kk]){
						continue;
					}
					//开始处理数据
					size_t position_idx = i * r2 * r3 + j * r3 + kk;
					//processed_data_dec_count++;
					if (eb_quant_index[position_idx] == 0){
						U[position_idx] = *(unpred_data_pos++);
						V[position_idx] = *(unpred_data_pos++);
						W[position_idx] = *(unpred_data_pos++);
					}
					else{
						double eb = pow(base, eb_quant_index[position_idx]) * threshold;
						for (int p = 0; p < 3; ++p){
							//degrade to 1D
							T *cur_data_field = (p == 0) ? U : (p == 1) ? V : W;
							T d0 = (kk && (kk - start_k != 1)) ? cur_data_field[position_idx - 1] : 0;
							T pred = d0;
							cur_data_field[position_idx] = pred + 2 * (data_quant_index[3 * position_idx + p] - intv_radius) * eb;
						}
					}
				}
			}
		}
	}
	printf("done edge_x...\n");

	//并行处理edge_y
	omp_set_num_threads(num_edges_y);
	#pragma omp parallel for collapse(3) 
	for (int j : dividing_r2){
		for (int k : dividing_r3){
			for (int i = -1; i < (int)dividing_r1.size(); i++){
				int thread_id = omp_get_thread_num();
				int y_layer = j;
				int z_layer = k;
				int start_i = (i == -1) ? 0 : dividing_r1[i];
				int end_i = (i == (int)dividing_r1.size() - 1) ? r1 : dividing_r1[i + 1];
				T *unpred_data_pos = &edge_y_data_each_thread[thread_id][0];
				for (int ii = start_i; ii < end_i; ++ii){
					// if (std::find(dividing_r1.begin(), dividing_r1.end(), i) != dividing_r1.end()){
					// 	continue;
					// }
					if (is_dividing_r1[ii]){
						continue;
					}
					//开始处理数据
					size_t position_idx = ii * r2 * r3 + j * r3 + k;
					//processed_data_dec_count++;
					if (eb_quant_index[position_idx] == 0){
						U[position_idx] = *(unpred_data_pos++);
						V[position_idx] = *(unpred_data_pos++);
						W[position_idx] = *(unpred_data_pos++);
					}
					else{
						double eb = pow(base, eb_quant_index[position_idx]) * threshold;
						for (int p = 0; p < 3; ++p){
							//degrade to 1D
							T *cur_data_field = (p == 0) ? U : (p == 1) ? V : W;
							//degrade to 1D
							T d0 = (ii && (ii - start_i != 1)) ? cur_data_field[position_idx - r2*r3] : 0;
							T pred = d0;
							cur_data_field[position_idx] = pred + 2 * (data_quant_index[3 * position_idx + p] - intv_radius) * eb;
						}
					}
				}
			}
		}
	}
	printf("done edge_y...\n");

	//并行处理edge_z
	omp_set_num_threads(num_edges_z);
	#pragma omp parallel for collapse(3) 
	for (int k : dividing_r3){
		for (int i : dividing_r1){
			for (int j = -1; j < (int)dividing_r2.size(); j++){
				int thread_id = omp_get_thread_num();
				int z_layer = k;
				int x_layer = i;
				int start_j = (j == -1) ? 0 : dividing_r2[j];
				int end_j = (j == (int)dividing_r2.size() - 1) ? r2 : dividing_r2[j + 1];
				T *unpred_data_pos = &edge_z_data_each_thread[thread_id][0];
				for (int jj = start_j; jj < end_j; ++jj){
					// if (std::find(dividing_r2.begin(), dividing_r2.end(), j) != dividing_r2.end()){
					// 	continue;
					// }
					if (is_dividing_r2[jj]){
						continue;
					}
					//processed_data_dec_count++;
					size_t position_idx = i * r2 * r3 + jj * r3 + k;
					if (eb_quant_index[position_idx] == 0){
						U[position_idx] = *(unpred_data_pos++);
						V[position_idx] = *(unpred_data_pos++);
						W[position_idx] = *(unpred_data_pos++);
					}
					else{
						double eb = pow(base, eb_quant_index[position_idx]) * threshold;
						for (int p = 0; p < 3; ++p){
							//degrade to 1D
							T *cur_data_field = (p == 0) ? U : (p == 1) ? V : W;
							T d0 = (jj && (jj - start_j != 1)) ? cur_data_field[position_idx - r3] : 0;
							T pred = d0;
							cur_data_field[position_idx] = pred + 2 * (data_quant_index[3 * position_idx + p] - intv_radius) * eb;
						}
					}
				}
			}
		}
	}
	printf("done edge...\n");



	auto pred_dot_time_start = std::chrono::high_resolution_clock::now();
	//处理dot, 串行
	T *unpred_data_pos = dot_data;
	for (int i : dividing_r1){
		for (int j : dividing_r2){
			for (int k : dividing_r3){
				size_t position_idx = i * r2 * r3 + j * r3 + k;
				//processed_data_dec_count++;
				U[position_idx] = *(unpred_data_pos++);
				V[position_idx] = *(unpred_data_pos++);
				W[position_idx] = *(unpred_data_pos++);
			}
		}
	}
	auto pred_dot_time_end = std::chrono::high_resolution_clock::now();
	printf("time for pred dot in second: %f\n", std::chrono::duration<double>(pred_dot_time_end - pred_dot_time_start).count());

	printf("decomp: check if all data points are processed: %ld, %ld\n", processed_data_dec_count, num_elements);
	//最后再根据bitmap来处理lossless的数据

	if (index_need_to_fix_size != 0){
		/*
		//tmd 这段写的好像有问题？有空再优化，先用串行
		// 计算每个线程的起始位置和数据范围
		std::vector<size_t> thread_start_pos(num_threads + 1, 0);
		std::vector<size_t> data_pos_start(num_threads, 0);
		size_t current_pos = 0;
		// 预计算所有需要填充的点数，存储每个线程需要处理的起始索引
		for (int i = 0; i < num_threads; ++i) {
			size_t count = (num_elements / num_threads) + (i < num_elements % num_threads ? 1 : 0);
			thread_start_pos[i] = current_pos;
			current_pos += count;
		}
		thread_start_pos[num_threads] = current_pos; // 最后一个线程的结束位置
		
		#pragma omp parallel for
		for (int t = 0; t < num_threads; ++t) {
			size_t start = thread_start_pos[t];  // 当前线程的 index 起始位置
			size_t end = thread_start_pos[t + 1]; // 当前线程的 index 结束位置
			for (size_t i = start; i < end; ++i) {
				data_pos_start[t] += bitmap[i]; // 当前线程的 lossless_data 起始位置
			}
		}

		// 使用 OpenMP 并行化遍历 bitmap，并根据线程位置填充 lossless_data
		#pragma omp parallel for
		for (int t = 0; t < num_threads; ++t) {
			size_t start = thread_start_pos[t];  // 当前线程的 lossless_data_u 起始位置
			size_t end = thread_start_pos[t + 1]; // 当前线程的 lossless_data_u 结束位置
			size_t pos = data_pos_start[t]; // 当前线程的 lossless_data_u 填充位置
			for (int i = start; i < end; ++i) {
				if (bitmap[i] == 1) { // 需要替换的位置
					U[i] = lossless_data_U[pos]; // 线程独立的 pos
					V[i] = lossless_data_V[pos];
					W[i] = lossless_data_W[pos];
					pos++;
				}
			}
		}
		*/
		// for(int i=0; i< r1; i++){
		// 	for(int j=0; j< r2; j++){
		// 		for(int k=0; k< r3; k++){
		// 			if(static_cast<int>(bitmap[i*r2*r3 + j*r3 + k])){
		// 				U[i*r2*r3 + j*r3 + k] = *(lossless_data_U_pos++);
		// 				V[i*r2*r3 + j*r3 + k] = *(lossless_data_V_pos++);
		// 				W[i*r2*r3 + j*r3 + k] = *(lossless_data_W_pos++);
						
		// 			}
		// 		}
		// 	}
		// }
		std::vector<size_t> indices;
		for (size_t i = 0; i < r1 * r2 * r3; i++) {
			if (bitmap[i]) indices.push_back(i);
		}

		#pragma omp parallel for
		for (size_t i = 0; i < indices.size(); i++) {
			size_t idx = indices[i];
			U[idx] = lossless_data_U[i];
			V[idx] = lossless_data_V[i];
			W[idx] = lossless_data_W[i];
		}
	free(lossless_data_U);
	free(lossless_data_V);
	free(lossless_data_W);
	free(bitmap);
	}
	free(eb_quant_index);
	free(data_quant_index);
	free(dot_data);

	auto pred_time_end = std::chrono::high_resolution_clock::now();
	printf("time for pred in second: %f\n", std::chrono::duration<double>(pred_time_end - pred_time_start).count());



}

template
void
omp_sz_decompress_cp_preserve_3d_online_abs_record_vertex(const unsigned char * compressed, size_t r1, size_t r2, size_t r3, float *& U, float *& V, float *& W);

template
void
omp_sz_decompress_cp_preserve_3d_online_abs_record_vertex(const unsigned char * compressed, size_t r1, size_t r2, size_t r3, double *& U, double *& V, double *& W);

template<typename T>
void
omp_sz_decompress_cp_preserve_3d_record_vertex(const unsigned char * compressed, size_t r1, size_t r2, size_t r3, T *& U, T *& V, T *& W){
	if(U) free(U);
	if(V) free(V);
	if(W) free(W);

	size_t num_elements = r1 * r2 * r3;
	const unsigned char * compressed_pos = compressed;
	unsigned char * bitmap;
	T * lossless_data_U = NULL;
	T * lossless_data_V = NULL;
	T * lossless_data_W = NULL;
	T * lossless_data_U_pos = NULL;
	T * lossless_data_V_pos = NULL;
	T * lossless_data_W_pos = NULL;
	//first read index_need_to_fix_size
	size_t index_need_to_fix_size = 0;
	read_variable_from_src(compressed_pos, index_need_to_fix_size);
	//if not 0, then read bitmap and lossless data
	if (index_need_to_fix_size != 0){
		// allocate memory for bitmap
		bitmap = (unsigned char *) malloc(num_elements * sizeof(unsigned char));
		memset(bitmap, 0, num_elements * sizeof(unsigned char));
		size_t num_bytes = (num_elements % 8 == 0) ? num_elements / 8 : num_elements / 8 + 1;
		convertByteArray2IntArray_fast_1b_sz(num_elements, compressed_pos, num_bytes, bitmap);
		//再搞出来需要无损的大小
		size_t lossless_count = index_need_to_fix_size;
		// allocate memory for lossless data
		lossless_data_U = read_array_from_src<T>(compressed_pos,lossless_count);
		lossless_data_V = read_array_from_src<T>(compressed_pos,lossless_count);
		lossless_data_W = read_array_from_src<T>(compressed_pos,lossless_count);
		lossless_data_U_pos = lossless_data_U;
		lossless_data_V_pos = lossless_data_V;
		lossless_data_W_pos = lossless_data_W;
	}

	int base = 0;
	read_variable_from_src(compressed_pos, base);
	printf("decomp base = %d\n", base);
	int intv_radius = 0;
	read_variable_from_src(compressed_pos, intv_radius);
	printf("decomp intv_radius = %d\n", intv_radius);
	const int capacity = (intv_radius << 1);
	size_t sign_map_size =(num_elements - 1)/8 + 1;
	unsigned char * sign_map_u = convertByteArray2IntArray_fast_1b_sz(num_elements, compressed_pos, sign_map_size);
	unsigned char * sign_map_v = convertByteArray2IntArray_fast_1b_sz(num_elements, compressed_pos, sign_map_size);
	unsigned char * sign_map_w = convertByteArray2IntArray_fast_1b_sz(num_elements, compressed_pos, sign_map_size);
	
	int num_threads = 0;
	read_variable_from_src(compressed_pos, num_threads);
	printf("decomp num_threads = %d\n", num_threads);
	int M,K,L;
	factor3D(num_threads, M, K, L);
	printf("decomp M = %d, K = %d, L = %d\n", M, K, L);

	int num_faces_x = M*K*L;
	int num_faces_y = M*K*L;
	int num_faces_z = M*K*L;
	int num_edges_x = M*K*L;
	int num_edges_y = M*K*L;
	int num_edges_z = M*K*L;

	auto read_variables_time_start = std::chrono::high_resolution_clock::now();
	// 读每一个block的unpred_data size
	std::vector<size_t> unpred_data_count(num_threads);
	for (int i = 0; i < num_threads; i++){
		read_variable_from_src(compressed_pos, unpred_data_count[i]);
	}
	printf("decomp last thread block has size: %ld\n", unpred_data_count[num_threads-1]);
	//读每个block的unpred_data
	std::vector<std::vector<T>> unpred_data_each_thread(num_threads);
	for (int i = 0; i < num_threads; i++){
		T * raw_data = read_array_from_src<T>(compressed_pos, unpred_data_count[i]);
		unpred_data_each_thread[i] = std::vector<T>(raw_data, raw_data + unpred_data_count[i]);
		free(raw_data);
		//compressed_pos is the sum of previous unpred_data_count
		//read_array_from_src_to_stdarray<T>(compressed_pos, unpred_data_each_thread[i], unpred_data_count[i]);
	}
	printf("decomp: compressed_pos after block = %ld\n",compressed_pos - compressed);

	// int t = std::round(std::cbrt(num_threads));
	// printf("t = %d\n", t);
	// int num_faces = (t-1) * t * t;
	// int num_edges = t*(t-1)*(t-1);
	//读每个face_x的unpred_data size
	std::vector<size_t> face_x_data_count(num_faces_x);
	for (int i = 0; i < num_faces_x; i++){
		read_variable_from_src(compressed_pos, face_x_data_count[i]);
	}
	//读每个face_x的unpred_data
	std::vector<std::vector<T>> face_x_data_each_thread(num_faces_x);
	for (int i = 0; i < num_faces_x; i++){
		T * raw_data = read_array_from_src<T>(compressed_pos, face_x_data_count[i]);
		face_x_data_each_thread[i] = std::vector<T>(raw_data, raw_data + face_x_data_count[i]);
		free(raw_data);
	}
	//读每个face_y的unpred_data size
	std::vector<size_t> face_y_data_count(num_faces_y);
	for (int i = 0; i < num_faces_y; i++){
		read_variable_from_src(compressed_pos, face_y_data_count[i]);
	}
	//读每个face_y的unpred_data
	std::vector<std::vector<T>> face_y_data_each_thread(num_faces_y);
	for (int i = 0; i < num_faces_y; i++){
		T * raw_data = read_array_from_src<T>(compressed_pos, face_y_data_count[i]);
		face_y_data_each_thread[i] = std::vector<T>(raw_data, raw_data + face_y_data_count[i]);
		free(raw_data);
	}
	//读每个face_z的unpred_data size
	std::vector<size_t> face_z_data_count(num_faces_z);
	for (int i = 0; i < num_faces_z; i++){
		read_variable_from_src(compressed_pos, face_z_data_count[i]);
	}
	//读每个face_z的unpred_data
	std::vector<std::vector<T>> face_z_data_each_thread(num_faces_z);
	for (int i = 0; i < num_faces_z; i++){
		T * raw_data = read_array_from_src<T>(compressed_pos, face_z_data_count[i]);
		face_z_data_each_thread[i] = std::vector<T>(raw_data, raw_data + face_z_data_count[i]);
		free(raw_data);
	}
	//printf("decomp last face_z size: %ld\n", face_z_data_count[num_faces-1]);

	//读每个edge_x的unpred_data size
	std::vector<size_t> edge_x_data_count(num_edges_x);
	for (int i = 0; i < num_edges_x; i++){
		read_variable_from_src(compressed_pos, edge_x_data_count[i]);
	}
	//读每个edge_x的unpred_data
	std::vector<std::vector<T>> edge_x_data_each_thread(num_edges_x);
	for (int i = 0; i < num_edges_x; i++){
		T * raw_data = read_array_from_src<T>(compressed_pos, edge_x_data_count[i]);
		edge_x_data_each_thread[i] = std::vector<T>(raw_data, raw_data + edge_x_data_count[i]);
		free(raw_data);
	}
	//读每个edge_y的unpred_data size
	std::vector<size_t> edge_y_data_count(num_edges_y);
	for (int i = 0; i < num_edges_y; i++){
		read_variable_from_src(compressed_pos, edge_y_data_count[i]);
	}
	//读每个edge_y的unpred_data
	std::vector<std::vector<T>> edge_y_data_each_thread(num_edges_y);
	for (int i = 0; i < num_edges_y; i++){
		T * raw_data = read_array_from_src<T>(compressed_pos, edge_y_data_count[i]);
		edge_y_data_each_thread[i] = std::vector<T>(raw_data, raw_data + edge_y_data_count[i]);
		free(raw_data);
	}
	//读每个edge_z的unpred_data size
	std::vector<size_t> edge_z_data_count(num_edges_z);
	for (int i = 0; i < num_edges_z; i++){
		read_variable_from_src(compressed_pos, edge_z_data_count[i]);
	}
	//读每个edge_z的unpred_data
	std::vector<std::vector<T>> edge_z_data_each_thread(num_edges_z);
	for (int i = 0; i < num_edges_z; i++){
		T * raw_data = read_array_from_src<T>(compressed_pos, edge_z_data_count[i]);
		edge_z_data_each_thread[i] = std::vector<T>(raw_data, raw_data + edge_z_data_count[i]);
		free(raw_data);
	}
	
	//printf("decomp last edge_z size: %ld\n", edge_z_data_count[num_edges-1]);

	// 读 dot的unpred_data size
	size_t dot_data_count = 0;
	read_variable_from_src(compressed_pos, dot_data_count);
	printf("decomp dot size = %ld\n", dot_data_count);
	// 读dot的unpred_data
	T * dot_data = read_array_from_src<T>(compressed_pos, dot_data_count);
	//printf("decomp: dot maxval = %f, minval = %f\n", *std::max_element(dot_data, dot_data + dot_data_count), *std::min_element(dot_data, dot_data + dot_data_count));

	auto read_variables_time_end = std::chrono::high_resolution_clock::now();
	printf("time for get other variables in second: %f\n", std::chrono::duration<double>(read_variables_time_end - read_variables_time_start).count());
	
	//std::vector<int> check_list = {26739302,26739404,26739506,26739608,26999910,27000012,27000114,80477490,106955570,107215974,107216280};

	auto readAndDecode_EbQuant_start = std::chrono::high_resolution_clock::now();
	// now read eb_quant_index
	//int * eb_quant_index = Huffman_decode_tree_and_data(2*1024, num_elements, compressed_pos);
	// 并行
	// read each compressed_size for eb_quant_index first
	std::vector<size_t> compressed_size_eb(num_threads);
	for (int i = 0; i < num_threads; i++){
		read_variable_from_src(compressed_pos, compressed_size_eb[i]);
	}
	std::vector<const unsigned char *> compressed_chunk_eb_start(num_threads);
	for (int i = 0; i < num_threads; i++){
		compressed_chunk_eb_start[i] = compressed_pos;
		compressed_pos += compressed_size_eb[i];
	}
	int * eb_quant_index = (int *) malloc(num_elements * sizeof(int));
	#pragma omp parallel for num_threads(num_threads)
	for (int i = 0; i < num_threads; i++){
		size_t start_pos = i*num_elements/num_threads;
		size_t end_pos = (i ==num_threads - 1) ? num_elements : (i+1)*num_elements/num_threads;
		const unsigned char * local_compressed_pos = compressed_chunk_eb_start[i];
		size_t local_num_elements = end_pos - start_pos;
		int * local_eb_quant_index = Huffman_decode_tree_and_data(2*256, local_num_elements, local_compressed_pos);
		std::copy(local_eb_quant_index, local_eb_quant_index + local_num_elements, eb_quant_index + start_pos);
		free(local_eb_quant_index);
	}
	
	auto readAndDecode_EbQuant_end = std::chrono::high_resolution_clock::now();
	// now read data_quant_index
	//int * data_quant_index = Huffman_decode_tree_and_data(2*capacity, 3*num_elements, compressed_pos);
	
	// 并行
	// read each compressed_size for data_quant_index first
	auto readAndDecode_DataQuant_start = std::chrono::high_resolution_clock::now();
	auto readOnly_DataQuant_start = std::chrono::high_resolution_clock::now();
	std::vector<size_t> compressed_size_data(num_threads);
	for (int i = 0; i < num_threads; i++){
		read_variable_from_src(compressed_pos, compressed_size_data[i]);
	}
	std::vector<const unsigned char *> compressed_chunk_data_start(num_threads);
	for (int i = 0; i < num_threads; i++){
		compressed_chunk_data_start[i] = compressed_pos;
		compressed_pos += compressed_size_data[i];
	}
	auto readOnly_DataQuant_end = std::chrono::high_resolution_clock::now();
	int * data_quant_index = (int *) malloc(3*num_elements * sizeof(int));
	#pragma omp parallel for num_threads(num_threads)
	for (int i = 0; i < num_threads; i++){
		size_t start_pos = i*num_elements/num_threads;
		size_t end_pos = (i ==num_threads - 1) ? num_elements : (i+1)*num_elements/num_threads;
		const unsigned char * local_compressed_pos = compressed_chunk_data_start[i];
		size_t local_num_elements = 3*(end_pos - start_pos);
		int * local_data_quant_index = Huffman_decode_tree_and_data(capacity, local_num_elements, local_compressed_pos);
		//auto copy_start = std::chrono::high_resolution_clock::now();
		std::copy(local_data_quant_index, local_data_quant_index + local_num_elements, data_quant_index + 3*start_pos);
		free(local_data_quant_index);
		//auto copy_end = std::chrono::high_resolution_clock::now();
		//total_time_copy += std::chrono::duration<double>(copy_end - copy_start).count();
	}
	auto readAndDecode_DataQuant_end = std::chrono::high_resolution_clock::now();
	printf("rel mode\n");
	printf("time for read and decode eb_quant_index in second: %f\n", std::chrono::duration<double>(readAndDecode_EbQuant_end - readAndDecode_EbQuant_start).count());
	printf("time for read and decode data_quant_index in second: %f(readonly: %f)\n", std::chrono::duration<double>(readAndDecode_DataQuant_end - readAndDecode_DataQuant_start).count(), std::chrono::duration<double>(readOnly_DataQuant_end - readOnly_DataQuant_start).count());
	printf("decomp pos after huffmans = %ld\n", compressed_pos - compressed);


	auto pred_time_start = std::chrono::high_resolution_clock::now();
	U = (T *) calloc(num_elements*sizeof(T),sizeof(T));
	V = (T *) calloc(num_elements*sizeof(T),sizeof(T));
	W = (T *) calloc(num_elements*sizeof(T),sizeof(T));
	T * U_pos = U;
	T * V_pos = V;
	T * W_pos = W;
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	const double threshold=std::numeric_limits<float>::epsilon();
	double log_of_base = log2(base);
	int eb_quant_index_max = (int) (log2(1.0 / threshold)/log_of_base) + 1;


	// 计算每个块的大小
	// int block_r1 = r1 / t;
	// int block_r2 = r2 / t;
	// int block_r3 = r3 / t;
	int block_r1 = r1 / M;
	int block_r2 = r2 / K;
	int block_r3 = r3 / L;

	//处理余数的情况
	// int remaining_r1 = r1 % t;
	// int remaining_r2 = r2 % t;
	// int remaining_r3 = r3 % t;
	int remaining_r1 = r1 % M;
	int remaining_r2 = r2 % K;
	int remaining_r3 = r3 % L;

	//储存划分平面的位置
	std::vector<int> dividing_r1;
    std::vector<int> dividing_r2;
    std::vector<int> dividing_r3;
    // for (int i = 1; i < t; ++i) {
    //     dividing_r1.push_back(i * block_r1);
    //     dividing_r2.push_back(i * block_r2);
    //     dividing_r3.push_back(i * block_r3);
    // }
	for (int i = 1; i < M; ++i) {
		dividing_r1.push_back(i * block_r1);
	}
	for (int i = 1; i < K; ++i) {
		dividing_r2.push_back(i * block_r2);
	}
	for (int i = 1; i < L; ++i) {
		dividing_r3.push_back(i * block_r3);
	}
	for (int i = 0; i < dividing_r1.size(); i++) {
		printf("decomp dividing_r1[%d] = %d\n", i, dividing_r1[i]);
	}
	for (int i = 0; i < dividing_r2.size(); i++) {
		printf("decomp dividing_r2[%d] = %d\n", i, dividing_r2[i]);
	}
	for (int i = 0; i < dividing_r3.size(); i++) {
		printf("decomp dividing_r3[%d] = %d\n", i, dividing_r3[i]);
	}
	// for (int i = 0; i < dividing_r1.size(); i++) {
	// 	printf("decomp dividing_r1[%d] = %d\n", i, dividing_r1[i]);
	// }

	// 创建一个一维标记数组，标记哪些数据点位于划分线上
	//std::vector<bool> on_dividing_line(num_elements, false);
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
	// #pragma omp parallel for collapse(3)
	// for (int i = 0; i < r1; ++i) {
	// 	for (int j = 0; j < r2; ++j) {
	// 		for (int k = 0; k < r3; ++k) {
	// 			if (is_dividing_r1[i] || is_dividing_r2[j] || is_dividing_r3[k]) {
	// 				on_dividing_line[i * r2 * r3 + j * r3 + k] = true;
	// 			}
	// 		}
	// 	}
	// }
	
	size_t processed_data_dec_count = 0;
	std::vector<std::unordered_set<int>> unpred_data_indices(num_threads);
	// 处理块
	int total_block = M * K * L;
	omp_set_num_threads(total_block);
	//omp_set_proc_bind(omp_proc_bind_true);
	#pragma omp parallel for num_threads(num_threads) 
	for (int block_id = 0; block_id < total_block; ++block_id){
		// int block_i = block_id / (t * t);
		// int block_j = (block_id % (t * t)) / t;
		// int block_k = block_id % t;
		int block_i = block_id / (K * L);
		int block_j = (block_id % (K * L)) / L;
		int block_k = block_id % L;
		// 计算块的起始位置
		int start_i = block_i * block_r1;
		int start_j = block_j * block_r2;
		int start_k = block_k * block_r3;

		int end_i = start_i + block_r1;
		int end_j = start_j + block_r2;
		int end_k = start_k + block_r3;
		if (block_i == M - 1) {
			end_i += remaining_r1;
		}
		if (block_j == K - 1) {
			end_j += remaining_r2;
		}
		if (block_k == L - 1) {
			end_k += remaining_r3;
		}
		//printf("Thread %d process block %d, start_i = %d, end_i = %d, start_j = %d, end_j = %d, start_k = %d, end_k = %d\n", omp_get_thread_num(), block_id, start_i, end_i, start_j, end_j, start_k, end_k);
		T *unpred_data_pos = &unpred_data_each_thread[omp_get_thread_num()][0];

		//处理块内部的数据
		for (int i = start_i; i < end_i; ++i) {
			// if (std::find(dividing_r1.begin(), dividing_r1.end(), i) != dividing_r1.end()) {
			// 	continue;
			// }
			if (is_dividing_r1[i]) {
				continue;
			}
			for (int j = start_j; j < end_j; ++j) {
				// if (std::find(dividing_r2.begin(), dividing_r2.end(), j) != dividing_r2.end()) {
				// 	continue;
				// }
				if (is_dividing_r2[j]) {
					continue;
				}
				for (int k = start_k; k < end_k; ++k) {
					// if (std::find(dividing_r3.begin(), dividing_r3.end(), k) != dividing_r3.end()) {
					// 	continue;
					// }
					if (is_dividing_r3[k]) {
						continue;
					}
					//开始处理数据
					//processed_data_dec_count++;
					size_t position_idx = i * r2 * r3 + j * r3 + k;
					// if (std::count(check_list.begin(), check_list.end(), position_idx)){
					// 	printf("id %ld in block %d\n", position_idx, block_id);
					// }
					//get eb
					if (eb_quant_index[position_idx] == 0 || eb_quant_index[position_idx] == eb_quant_index_max){
						unpred_data_indices[omp_get_thread_num()].insert(position_idx);
						for (int p = 0; p < 3; ++p){
							T *cur_data_field = (p == 0) ? U : (p == 1) ? V : W;
							T cur_data = *(unpred_data_pos++);
							cur_data_field[position_idx] = (cur_data == 0) ? -100 : log2f(fabs(cur_data));
						}
						// U[position_idx] = *(unpred_data_pos++);
						// V[position_idx] = *(unpred_data_pos++);
						// W[position_idx] = *(unpred_data_pos++);
					}
					else{
						// double eb = pow(base, eb_quant_index[position_idx]) * threshold;
						double eb = (eb_quant_index[position_idx] == 0) ? 0 : pow(base, eb_quant_index[position_idx]) * threshold;
						for (int p = 0; p < 3; ++p){
							T *cur_data_field = (p == 0) ? U : (p == 1) ? V : W;
							T d0 = ((i && j && k) && (i - start_i != 1 && j - start_j != 1 && k - start_k != 1)) ? cur_data_field[position_idx - r2 * r3 - r3 - 1] : 0;
							T d1 = ((i && j) && (i - start_i != 1 && j - start_j != 1)) ? cur_data_field[position_idx - r2 * r3 - r3] : 0;
							T d2 = ((i && k) && (i - start_i != 1 && k - start_k != 1)) ? cur_data_field[position_idx - r2 * r3 - 1] : 0;
							T d3 = (i && (i - start_i != 1)) ? cur_data_field[position_idx - r2 * r3] : 0;
							T d4 = ((j && k) && (j - start_j != 1 && k - start_k != 1)) ? cur_data_field[position_idx - r3 - 1] : 0;
							T d5 = (j && (j - start_j != 1)) ? cur_data_field[position_idx - r3] : 0;
							T d6 = (k && (k - start_k != 1)) ? cur_data_field[position_idx - 1] : 0;
							T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
							cur_data_field[position_idx] = pred + 2 * (data_quant_index[3 * position_idx + p] - intv_radius) * eb;
						}
					}
				}
			}
		}
		
		//recover data done
		unpred_data_pos = &unpred_data_each_thread[omp_get_thread_num()][0];
		for (int i = start_i; i < end_i; ++i) {
			// if (std::find(dividing_r1.begin(), dividing_r1.end(), i) != dividing_r1.end()) {
			// 	continue;
			// }
			if (is_dividing_r1[i]) {
				continue;
			}
			for (int j = start_j; j < end_j; ++j) {
				// if (std::find(dividing_r2.begin(), dividing_r2.end(), j) != dividing_r2.end()) {
				// 	continue;
				// }
				if (is_dividing_r2[j]) {
					continue;
				}
				for (int k = start_k; k < end_k; ++k) {
					// if (std::find(dividing_r3.begin(), dividing_r3.end(), k) != dividing_r3.end()) {
					// 	continue;
					// }
					if (is_dividing_r3[k]) {
						continue;
					}
					size_t position_idx = i * r2 * r3 + j * r3 + k;
					if (unpred_data_indices[omp_get_thread_num()].count(position_idx)){
						U[position_idx] = *(unpred_data_pos++);
						V[position_idx] = *(unpred_data_pos++);
						W[position_idx] = *(unpred_data_pos++);
					}
					else{
						if (U[position_idx] < -99) U[i] = 0;
						else U[position_idx] = sign_map_u[position_idx] ? exp2(U[position_idx]) : -exp2(U[position_idx]);
						if (V[position_idx] < -99) V[i] = 0;
						else V[position_idx] = sign_map_v[position_idx] ? exp2(V[position_idx]) : -exp2(V[position_idx]);
						if (W[position_idx] < -99) W[i] = 0;
						else W[position_idx] = sign_map_w[position_idx] ? exp2(W[position_idx]) : -exp2(W[position_idx]);
					}
				}
			}
		}
	}
	printf("done block...\n");


	//优化: 并行解压TODO
	ptrdiff_t dim0_offset = r2 * r3;
	ptrdiff_t dim1_offset = r3;
	ptrdiff_t cell_dim0_offset = (r2-1) * (r3-1);
	ptrdiff_t cell_dim1_offset = r3-1;

	std::vector<std::unordered_set<int>> face_x_unpred_data_indices(num_faces_x);
	std::vector<std::unordered_set<int>> face_y_unpred_data_indices(num_faces_y);
	std::vector<std::unordered_set<int>> face_z_unpred_data_indices(num_faces_z);
	std::vector<std::unordered_set<int>> edge_x_unpred_data_indices(num_edges_x);
	std::vector<std::unordered_set<int>> edge_y_unpred_data_indices(num_edges_y);
	std::vector<std::unordered_set<int>> edge_z_unpred_data_indices(num_edges_z);



	//并行处理face_x
	omp_set_num_threads(num_faces_x);
	// Process faces perpendicular to the X-axis(jk-plane)
	#pragma omp parallel for collapse(3) 
	for (int i : dividing_r1){
		for (int j = -1; j < (int)dividing_r2.size(); j++){
			for (int k = -1; k < (int)dividing_r3.size(); k++){
				int thread_id = omp_get_thread_num();
				int x_layer = i;
				int start_j = (j == -1) ? 0 : dividing_r2[j];
				int end_j = (j == (int)dividing_r2.size() - 1) ? r2 : dividing_r2[j + 1];
				int start_k = (k == -1) ? 0 : dividing_r3[k];
				int end_k = (k == (int)dividing_r3.size() - 1) ? r3 : dividing_r3[k + 1];
				T *unpred_data_pos = &face_x_data_each_thread[thread_id][0];
				for (int jj = start_j; jj < end_j; ++jj){
					// if (std::find(dividing_r2.begin(), dividing_r2.end(), j) != dividing_r2.end()){
					// 	continue;
					// }
					if (is_dividing_r2[jj]){
						continue;
					}
					for (int kk = start_k; kk < end_k; ++kk){
						// if (std::find(dividing_r3.begin(), dividing_r3.end(), k) != dividing_r3.end()){
						// 	continue;
						// }
						if (is_dividing_r3[kk]){
							continue;
						}
						//开始处理数据
						size_t position_idx = i * r2 * r3 + jj * r3 + kk;
						// if (std::count(check_list.begin(), check_list.end(), position_idx)){
						// 	printf("id %ld in face_x %d\n", position_idx,thread_id);
						// }
						//processed_data_dec_count++;
						if (eb_quant_index[position_idx] == 0 || eb_quant_index[position_idx] == eb_quant_index_max){
							face_x_unpred_data_indices[thread_id].insert(position_idx);
							for (int p = 0; p < 3; ++p){
								T *cur_data_field = (p == 0) ? U : (p == 1) ? V : W;
								T cur_data = *(unpred_data_pos++);
								cur_data_field[position_idx] = (cur_data == 0) ? -100 : log2f(fabs(cur_data));
							}
							// U[position_idx] = *(unpred_data_pos++);
							// V[position_idx] = *(unpred_data_pos++);
							// W[position_idx] = *(unpred_data_pos++);
						}
						else{
							double eb = (eb_quant_index[position_idx] == 0) ? 0 : pow(base, eb_quant_index[position_idx]) * threshold;
							for (int p = 0; p < 3; ++p){
								//degrade to 2D
								T *cur_data_field = (p == 0) ? U : (p == 1) ? V : W;
								T d0 = ((jj && kk) && (jj - start_j != 1 && kk - start_k != 1)) ? cur_data_field[position_idx - r2 - 1] : 0;
								T d1 = (jj && (jj - start_j != 1)) ? cur_data_field[position_idx - r2] : 0;
								T d2 = (kk && (kk - start_k != 1)) ? cur_data_field[position_idx - 1] : 0;
								T pred = d1 + d2 - d0;
								cur_data_field[position_idx] = pred + 2 * (data_quant_index[3 * position_idx + p] - intv_radius) * eb;
							}
						}
					}
				}
			
				// recover data done
				unpred_data_pos = &face_x_data_each_thread[thread_id][0];
				for (int jj = start_j; jj < end_j; ++jj){
					// if (std::find(dividing_r2.begin(), dividing_r2.end(), j) != dividing_r2.end()){
					// 	continue;
					// }
					if (is_dividing_r2[jj]){
						continue;
					}
					for (int kk = start_k; kk < end_k; ++kk){
						// if (std::find(dividing_r3.begin(), dividing_r3.end(), k) != dividing_r3.end()){
						// 	continue;
						// }
						if (is_dividing_r3[kk]){
							continue;
						}
						size_t position_idx = i * r2 * r3 + jj * r3 + kk;
						if (face_x_unpred_data_indices[thread_id].count(position_idx)){
							U[position_idx] = *(unpred_data_pos++);
							V[position_idx] = *(unpred_data_pos++);
							W[position_idx] = *(unpred_data_pos++);
						}
						else{
							if (U[position_idx] < -99) U[i] = 0;
							else U[position_idx] = sign_map_u[position_idx] ? exp2(U[position_idx]) : -exp2(U[position_idx]);
							if (V[position_idx] < -99) V[i] = 0;
							else V[position_idx] = sign_map_v[position_idx] ? exp2(V[position_idx]) : -exp2(V[position_idx]);
							if (W[position_idx] < -99) W[i] = 0;
							else W[position_idx] = sign_map_w[position_idx] ? exp2(W[position_idx]) : -exp2(W[position_idx]);
						}
					}
				}
			}
		}
	}


	//并行处理face_y
	omp_set_num_threads(num_faces_y);
	#pragma omp parallel for collapse(3) 
	for (int j : dividing_r2){
		for (int i = -1; i < (int)dividing_r1.size(); i++){
			for (int k = -1; k < (int)dividing_r3.size(); k++){
				int thread_id = omp_get_thread_num();
				int y_layer = j;
				int start_i = (i == -1) ? 0 : dividing_r1[i];
				int end_i = (i == (int)dividing_r1.size() - 1) ? r1 : dividing_r1[i + 1];
				int start_k = (k == -1) ? 0 : dividing_r3[k];
				int end_k = (k == (int)dividing_r3.size() - 1) ? r3 : dividing_r3[k + 1];
				T *unpred_data_pos = &face_y_data_each_thread[thread_id][0];
				for (int ii = start_i; ii < end_i; ++ii){
					// if (std::find(dividing_r1.begin(), dividing_r1.end(), i) != dividing_r1.end()){
					// 	continue;
					// }
					if (is_dividing_r1[ii]){
						continue;
					}
					for (int kk = start_k; kk < end_k; ++kk){
						// if (std::find(dividing_r3.begin(), dividing_r3.end(), k) != dividing_r3.end()){
						// 	continue;
						// }
						if (is_dividing_r3[kk]){
							continue;
						}
						//开始处理数据
						size_t position_idx = ii * r2 * r3 + j * r3 + kk;
						// if (std::count(check_list.begin(), check_list.end(), position_idx)){
						// 	printf("id %ld in face_y %d\n",position_idx,thread_id);
						// }
						//processed_data_dec_count++;
						if (eb_quant_index[position_idx] == 0 || eb_quant_index[position_idx] == eb_quant_index_max){
							face_y_unpred_data_indices[thread_id].insert(position_idx);
							for (int p = 0; p < 3; ++p){
								T *cur_data_field = (p == 0) ? U : (p == 1) ? V : W;
								T cur_data = *(unpred_data_pos++);
								cur_data_field[position_idx] = (cur_data == 0) ? -100 : log2f(fabs(cur_data));
							}
							// U[position_idx] = *(unpred_data_pos++);
							// V[position_idx] = *(unpred_data_pos++);
							// W[position_idx] = *(unpred_data_pos++);
						}
						else{
							double eb = (eb_quant_index[position_idx] == 0) ? 0 : pow(base, eb_quant_index[position_idx]) * threshold;
							for (int p = 0; p < 3; ++p){
								//degrade to 2D
								T *cur_data_field = (p == 0) ? U : (p == 1) ? V : W;
									T d0 = ((ii && kk) && (ii - start_i != 1 && kk - start_k != 1)) ? cur_data_field[position_idx - r2*r3 - 1] : 0;
									T d1 = (ii && (ii - start_i != 1)) ? cur_data_field[position_idx - r2*r3] : 0;
									T d2 = (kk && (kk - start_k != 1)) ? cur_data_field[position_idx - 1] : 0;
									T pred = d1 + d2 - d0;
									cur_data_field[position_idx] = pred + 2 * (data_quant_index[3 * position_idx + p] - intv_radius) * eb;
							}
						}
					}
				}

				// recover data done
				unpred_data_pos = &face_y_data_each_thread[thread_id][0];
				for (int ii = start_i; ii < end_i; ++ii){
					// if (std::find(dividing_r1.begin(), dividing_r1.end(), i) != dividing_r1.end()){
					// 	continue;
					// }
					if (is_dividing_r1[ii]){
						continue;
					}
					for (int kk = start_k; kk < end_k; ++kk){
						// if (std::find(dividing_r3.begin(), dividing_r3.end(), k) != dividing_r3.end()){
						// 	continue;
						// }
						if (is_dividing_r3[kk]){
							continue;
						}
						size_t position_idx = ii * r2 * r3 + j * r3 + kk;
						if (face_y_unpred_data_indices[thread_id].count(position_idx)){
							U[position_idx] = *(unpred_data_pos++);
							V[position_idx] = *(unpred_data_pos++);
							W[position_idx] = *(unpred_data_pos++);
						}
						else{
							if (U[position_idx] < -99) U[i] = 0;
							else U[position_idx] = sign_map_u[position_idx] ? exp2(U[position_idx]) : -exp2(U[position_idx]);
							if (V[position_idx] < -99) V[i] = 0;
							else V[position_idx] = sign_map_v[position_idx] ? exp2(V[position_idx]) : -exp2(V[position_idx]);
							if (W[position_idx] < -99) W[i] = 0;
							else W[position_idx] = sign_map_w[position_idx] ? exp2(W[position_idx]) : -exp2(W[position_idx]);
						}
					}
				}
			}
		}
	}

	//并行处理face_z
	omp_set_num_threads(num_faces_z);
	#pragma omp parallel for collapse(3) 
	for (int k : dividing_r3){
		for (int i = -1; i < (int)dividing_r1.size(); i++){
			for (int j = -1; j < (int)dividing_r2.size(); j++){
				int thread_id = omp_get_thread_num();
				int z_layer = k;
				int start_i = (i == -1) ? 0 : dividing_r1[i];
				int end_i = (i == (int)dividing_r1.size() - 1) ? r1 : dividing_r1[i + 1];
				int start_j = (j == -1) ? 0 : dividing_r2[j];
				int end_j = (j == (int)dividing_r2.size() - 1) ? r2 : dividing_r2[j + 1];
				T *unpred_data_pos = &face_z_data_each_thread[thread_id][0];
				for (int ii = start_i; ii < end_i; ++ii){
					// if (std::find(dividing_r1.begin(), dividing_r1.end(), i) != dividing_r1.end()){
					// 	continue;
					// }
					if (is_dividing_r1[ii]){
						continue;
					}
					for (int jj = start_j; jj < end_j; ++jj){
						// if (std::find(dividing_r2.begin(), dividing_r2.end(), j) != dividing_r2.end()){
						// 	continue;
						// }
						if (is_dividing_r2[jj]){
							continue;
						}
						//开始处理数据
						size_t position_idx = ii * r2 * r3 + jj * r3 + k;
						// if (std::count(check_list.begin(), check_list.end(), position_idx)){
						// 	printf("id %ld in face_z %d\n", position_idx, thread_id);
						// }
						//processed_data_dec_count++;
						if (eb_quant_index[position_idx] == 0 || eb_quant_index[position_idx] == eb_quant_index_max){
							face_z_unpred_data_indices[thread_id].insert(position_idx);
							for (int p = 0; p < 3; ++p){
								T *cur_data_field = (p == 0) ? U : (p == 1) ? V : W;
								T cur_data = *(unpred_data_pos++);
								cur_data_field[position_idx] = (cur_data == 0) ? -100 : log2f(fabs(cur_data));
							}
							// U[position_idx] = *(unpred_data_pos++);
							// V[position_idx] = *(unpred_data_pos++);
							// W[position_idx] = *(unpred_data_pos++);
						}
						else{
							double eb = (eb_quant_index[position_idx] == 0) ? 0 : pow(base, eb_quant_index[position_idx]) * threshold;
							for (int p = 0; p < 3; ++p){
								//degrade to 2D
								T *cur_data_field = (p == 0) ? U : (p == 1) ? V : W;
								T d0 = ((ii && jj) && (ii - start_i != 1 && jj - start_j != 1)) ? cur_data_field[position_idx - r2*r3 - r3] : 0;
								T d1 = (ii && (ii - start_i != 1)) ? cur_data_field[position_idx - r2*r3] : 0;
								T d2 = (jj && (jj - start_j != 1)) ? cur_data_field[position_idx - r3] : 0;
								T pred = d1 + d2 - d0;
								cur_data_field[position_idx] = pred + 2 * (data_quant_index[3 * position_idx + p] - intv_radius) * eb;
							}
						}
					}
				}
			
				// recover data done
				unpred_data_pos = &face_z_data_each_thread[thread_id][0];
				for (int ii = start_i; ii < end_i; ++ii){
					// if (std::find(dividing_r1.begin(), dividing_r1.end(), i) != dividing_r1.end()){
					// 	continue;
					// }
					if (is_dividing_r1[ii]){
						continue;
					}
					for (int jj = start_j; jj < end_j; ++jj){
						// if (std::find(dividing_r2.begin(), dividing_r2.end(), j) != dividing_r2.end()){
						// 	continue;
						// }
						if (is_dividing_r2[jj]){
							continue;
						}
						size_t position_idx = ii * r2 * r3 + jj * r3 + k;
						if (face_z_unpred_data_indices[thread_id].count(position_idx)){
							U[position_idx] = *(unpred_data_pos++);
							V[position_idx] = *(unpred_data_pos++);
							W[position_idx] = *(unpred_data_pos++);
						}
						else{
							if (U[position_idx] < -99) U[i] = 0;
							else U[position_idx] = sign_map_u[position_idx] ? exp2(U[position_idx]) : -exp2(U[position_idx]);
							if (V[position_idx] < -99) V[i] = 0;
							else V[position_idx] = sign_map_v[position_idx] ? exp2(V[position_idx]) : -exp2(V[position_idx]);
							if (W[position_idx] < -99) W[i] = 0;
							else W[position_idx] = sign_map_w[position_idx] ? exp2(W[position_idx]) : -exp2(W[position_idx]);
						}
					}
				}
			}
		}
	}

	printf("done face...\n");

	//并行处理edge_x
	omp_set_num_threads(num_edges_x);
	#pragma omp parallel for collapse(3) 
	for (int i : dividing_r1){
		for (int j :dividing_r2){
			for (int k = -1; k < (int)dividing_r3.size(); k++){
				int thread_id = omp_get_thread_num();
				int x_layer = i;
				int y_layer = j;
				int start_k = (k == -1) ? 0 : dividing_r3[k];
				int end_k = (k == (int)dividing_r3.size() - 1) ? r3 : dividing_r3[k + 1];
				T *unpred_data_pos = &edge_x_data_each_thread[thread_id][0];
				for (int kk = start_k; kk < end_k; ++kk){
					// if (std::find(dividing_r3.begin(), dividing_r3.end(), k) != dividing_r3.end()){
					// 	continue;
					// }
					if (is_dividing_r3[kk]){
						continue;
					}
					//开始处理数据
					size_t position_idx = i * r2 * r3 + j * r3 + kk;
					// if (std::count(check_list.begin(), check_list.end(), position_idx)){
					// 	printf("id %ld in edge_x %d\n", position_idx, thread_id);
					// }
					//processed_data_dec_count++;
					if (eb_quant_index[position_idx] == 0 || eb_quant_index[position_idx] == eb_quant_index_max){
						edge_x_unpred_data_indices[thread_id].insert(position_idx);
						for (int p = 0; p < 3; ++p){
							T *cur_data_field = (p == 0) ? U : (p == 1) ? V : W;
							T cur_data = *(unpred_data_pos++);
							cur_data_field[position_idx] = (cur_data == 0) ? -100 : log2f(fabs(cur_data));
						}
						// U[position_idx] = *(unpred_data_pos++);
						// V[position_idx] = *(unpred_data_pos++);
						// W[position_idx] = *(unpred_data_pos++);
					}
					else{
						double eb = (eb_quant_index[position_idx] == 0) ? 0 : pow(base, eb_quant_index[position_idx]) * threshold;
						for (int p = 0; p < 3; ++p){
							//degrade to 1D
							T *cur_data_field = (p == 0) ? U : (p == 1) ? V : W;
							T d0 = (kk && (kk - start_k != 1)) ? cur_data_field[position_idx - 1] : 0;
							T pred = d0;
							cur_data_field[position_idx] = pred + 2 * (data_quant_index[3 * position_idx + p] - intv_radius) * eb;
						}
					}
				}
			
				// recover data done
				unpred_data_pos = &edge_x_data_each_thread[thread_id][0];
				for (int kk = start_k; kk < end_k; ++kk){
					// if (std::find(dividing_r3.begin(), dividing_r3.end(), k) != dividing_r3.end()){
					// 	continue;
					// }
					if (is_dividing_r3[kk]){
						continue;
					}
					size_t position_idx = i * r2 * r3 + j * r3 + kk;
					if (edge_x_unpred_data_indices[thread_id].count(position_idx)){
						U[position_idx] = *(unpred_data_pos++);
						V[position_idx] = *(unpred_data_pos++);
						W[position_idx] = *(unpred_data_pos++);
					}
					else{
						if (U[position_idx] < -99) U[i] = 0;
						else U[position_idx] = sign_map_u[position_idx] ? exp2(U[position_idx]) : -exp2(U[position_idx]);
						if (V[position_idx] < -99) V[i] = 0;
						else V[position_idx] = sign_map_v[position_idx] ? exp2(V[position_idx]) : -exp2(V[position_idx]);
						if (W[position_idx] < -99) W[i] = 0;
						else W[position_idx] = sign_map_w[position_idx] ? exp2(W[position_idx]) : -exp2(W[position_idx]);
					}
				}
			}
		}
	}
	printf("done edge_x...\n");

	//并行处理edge_y
	omp_set_num_threads(num_edges_y);
	#pragma omp parallel for collapse(3) 
	for (int j : dividing_r2){
		for (int k : dividing_r3){
			for (int i = -1; i < (int)dividing_r1.size(); i++){
				int thread_id = omp_get_thread_num();
				int y_layer = j;
				int z_layer = k;
				int start_i = (i == -1) ? 0 : dividing_r1[i];
				int end_i = (i == (int)dividing_r1.size() - 1) ? r1 : dividing_r1[i + 1];
				T *unpred_data_pos = &edge_y_data_each_thread[thread_id][0];
				for (int ii = start_i; ii < end_i; ++ii){
					// if (std::find(dividing_r1.begin(), dividing_r1.end(), i) != dividing_r1.end()){
					// 	continue;
					// }
					if (is_dividing_r1[ii]){
						continue;
					}
					//开始处理数据
					size_t position_idx = ii * r2 * r3 + j * r3 + k;
					// if (std::count(check_list.begin(), check_list.end(), position_idx)){
					// 	printf("id %ld in edge_y %d\n", position_idx, thread_id);
					// }
					//processed_data_dec_count++;
					if (eb_quant_index[position_idx] == 0 || eb_quant_index[position_idx] == eb_quant_index_max){
						edge_y_unpred_data_indices[thread_id].insert(position_idx);
						for (int p = 0; p < 3; ++p){
							T *cur_data_field = (p == 0) ? U : (p == 1) ? V : W;
							T cur_data = *(unpred_data_pos++);
							cur_data_field[position_idx] = (cur_data == 0) ? -100 : log2f(fabs(cur_data));
						}
						// U[position_idx] = *(unpred_data_pos++);
						// V[position_idx] = *(unpred_data_pos++);
						// W[position_idx] = *(unpred_data_pos++);
					}
					else{
						double eb = (eb_quant_index[position_idx] == 0) ? 0 : pow(base, eb_quant_index[position_idx]) * threshold;
						for (int p = 0; p < 3; ++p){
							//degrade to 1D
							T *cur_data_field = (p == 0) ? U : (p == 1) ? V : W;
							//degrade to 1D
							T d0 = (ii && (ii - start_i != 1)) ? cur_data_field[position_idx - r2*r3] : 0;
							T pred = d0;
							cur_data_field[position_idx] = pred + 2 * (data_quant_index[3 * position_idx + p] - intv_radius) * eb;
						}
					}
				}
			
				// recover data done
				unpred_data_pos = &edge_y_data_each_thread[thread_id][0];
				for (int ii = start_i; ii < end_i; ++ii){
					// if (std::find(dividing_r1.begin(), dividing_r1.end(), i) != dividing_r1.end()){
					// 	continue;
					// }
					if (is_dividing_r1[ii]){
						continue;
					}
					size_t position_idx = ii * r2 * r3 + j * r3 + k;
					if (edge_y_unpred_data_indices[thread_id].count(position_idx)){
						U[position_idx] = *(unpred_data_pos++);
						V[position_idx] = *(unpred_data_pos++);
						W[position_idx] = *(unpred_data_pos++);
					}
					else{
						if (U[position_idx] < -99) U[i] = 0;
						else U[position_idx] = sign_map_u[position_idx] ? exp2(U[position_idx]) : -exp2(U[position_idx]);
						if (V[position_idx] < -99) V[i] = 0;
						else V[position_idx] = sign_map_v[position_idx] ? exp2(V[position_idx]) : -exp2(V[position_idx]);
						if (W[position_idx] < -99) W[i] = 0;
						else W[position_idx] = sign_map_w[position_idx] ? exp2(W[position_idx]) : -exp2(W[position_idx]);
					}
				}
			}
		}
	}
	printf("done edge_y...\n");

	//并行处理edge_z
	omp_set_num_threads(num_edges_z);
	#pragma omp parallel for collapse(3) 
	for (int k : dividing_r3){
		for (int i : dividing_r1){
			for (int j = -1; j < (int)dividing_r2.size(); j++){
				int thread_id = omp_get_thread_num();
				int z_layer = k;
				int x_layer = i;
				int start_j = (j == -1) ? 0 : dividing_r2[j];
				int end_j = (j == (int)dividing_r2.size() - 1) ? r2 : dividing_r2[j + 1];
				T *unpred_data_pos = &edge_z_data_each_thread[thread_id][0];
				for (int jj = start_j; jj < end_j; ++jj){
					// if (std::find(dividing_r2.begin(), dividing_r2.end(), j) != dividing_r2.end()){
					// 	continue;
					// }
					if (is_dividing_r2[jj]){
						continue;
					}
					//processed_data_dec_count++;
					size_t position_idx = i * r2 * r3 + jj * r3 + k;
					// if (std::count(check_list.begin(), check_list.end(), position_idx)){
					// 	printf("id %ld in edge_z %d\n", position_idx, thread_id);
					// }
					if (eb_quant_index[position_idx] == 0 || eb_quant_index[position_idx] == eb_quant_index_max){
						edge_z_unpred_data_indices[thread_id].insert(position_idx);
						for (int p = 0; p < 3; ++p){
							T *cur_data_field = (p == 0) ? U : (p == 1) ? V : W;
							T cur_data = *(unpred_data_pos++);
							cur_data_field[position_idx] = (cur_data == 0) ? -100 : log2f(fabs(cur_data));
						}
						// U[position_idx] = *(unpred_data_pos++);
						// V[position_idx] = *(unpred_data_pos++);
						// W[position_idx] = *(unpred_data_pos++);
					}
					else{
						double eb = (eb_quant_index[position_idx] == 0) ? 0 : pow(base, eb_quant_index[position_idx]) * threshold;
						for (int p = 0; p < 3; ++p){
							//degrade to 1D
							T *cur_data_field = (p == 0) ? U : (p == 1) ? V : W;
							T d0 = (jj && (jj - start_j != 1)) ? cur_data_field[position_idx - r3] : 0;
							T pred = d0;
							cur_data_field[position_idx] = pred + 2 * (data_quant_index[3 * position_idx + p] - intv_radius) * eb;
						}
					}
				}
			

				// recover data done
				unpred_data_pos = &edge_z_data_each_thread[thread_id][0];
				for (int jj = start_j; jj < end_j; ++jj){
					// if (std::find(dividing_r2.begin(), dividing_r2.end(), j) != dividing_r2.end()){
					// 	continue;
					// }
					if (is_dividing_r2[jj]){
						continue;
					}
					size_t position_idx = i * r2 * r3 + jj * r3 + k;
					if (edge_z_unpred_data_indices[thread_id].count(position_idx)){
						U[position_idx] = *(unpred_data_pos++);
						V[position_idx] = *(unpred_data_pos++);
						W[position_idx] = *(unpred_data_pos++);
					}
					else{
						if (U[position_idx] < -99) U[i] = 0;
						else U[position_idx] = sign_map_u[position_idx] ? exp2(U[position_idx]) : -exp2(U[position_idx]);
						if (V[position_idx] < -99) V[i] = 0;
						else V[position_idx] = sign_map_v[position_idx] ? exp2(V[position_idx]) : -exp2(V[position_idx]);
						if (W[position_idx] < -99) W[i] = 0;
						else W[position_idx] = sign_map_w[position_idx] ? exp2(W[position_idx]) : -exp2(W[position_idx]);
					}
				}
			}
		}
	}
	printf("done edge_z...\n");

	printf("done edge...\n");

	auto pred_dot_time_start = std::chrono::high_resolution_clock::now();
	//处理dot, 串行
	T *unpred_data_pos = dot_data;
	for (int i : dividing_r1){
		for (int j : dividing_r2){
			for (int k : dividing_r3){
				size_t position_idx = i * r2 * r3 + j * r3 + k;
				// if (std::count(check_list.begin(), check_list.end(), position_idx)){
				// 	printf("id %ld in dot \n", position_idx);
				// }
				//printf("dot coord: x: %d, y: %d, z: %d\n", i, j, k);
				//processed_data_dec_count++;
				if (eb_quant_index[position_idx] == 0 || eb_quant_index[position_idx] == eb_quant_index_max){
					for (int p = 0; p < 3; ++p){
						T *cur_data_field = (p == 0) ? U : (p == 1) ? V : W;
						T cur_data = *(unpred_data_pos++);
						cur_data_field[position_idx] = (cur_data == 0) ? -100 : log2f(fabs(cur_data));
					}
					// U[position_idx] = *(unpred_data_pos++);
					// V[position_idx] = *(unpred_data_pos++);
					// W[position_idx] = *(unpred_data_pos++);
				}
				else{
					printf("dot data should not predict\n");
				}
				// U[position_idx] = *(unpred_data_pos++);
				// V[position_idx] = *(unpred_data_pos++);
				// W[position_idx] = *(unpred_data_pos++);
			}
		}
	}
	
	// recover data done
	unpred_data_pos = dot_data;
	for (int i : dividing_r1){
		for (int j : dividing_r2){
			for (int k : dividing_r3){
				size_t position_idx = i * r2 * r3 + j * r3 + k;
				if (U[position_idx] < -99) U[i] = 0;
				else U[position_idx] = sign_map_u[position_idx] ? exp2(U[position_idx]) : -exp2(U[position_idx]);
				if (V[position_idx] < -99) V[i] = 0;
				else V[position_idx] = sign_map_v[position_idx] ? exp2(V[position_idx]) : -exp2(V[position_idx]);
				if (W[position_idx] < -99) W[i] = 0;
				else W[position_idx] = sign_map_w[position_idx] ? exp2(W[position_idx]) : -exp2(W[position_idx]);
			}
		}
	}
	
	
	auto pred_dot_time_end = std::chrono::high_resolution_clock::now();
	printf("time for pred dot in second: %f\n", std::chrono::duration<double>(pred_dot_time_end - pred_dot_time_start).count());

	printf("decomp: check if all data points are processed: %ld, %ld\n", processed_data_dec_count, num_elements);
	//最后再根据bitmap来处理lossless的数据
	if (index_need_to_fix_size != 0){
		//擦 这里也要注释掉。。
		// 计算每个线程的起始位置和数据范围
		// std::vector<size_t> thread_start_pos(num_threads + 1, 0);
		// std::vector<size_t> data_pos_start(num_threads, 0);
		// size_t current_pos = 0;
		// // 预计算所有需要填充的点数，存储每个线程需要处理的起始索引
		// for (int i = 0; i < num_threads; ++i) {
		// 	size_t count = (num_elements / num_threads) + (i < num_elements % num_threads ? 1 : 0);
		// 	thread_start_pos[i] = current_pos;
		// 	current_pos += count;
		// }
		// thread_start_pos[num_threads] = current_pos; // 最后一个线程的结束位置
		
		// #pragma omp parallel for
		// for (int t = 0; t < num_threads; ++t) {
		// 	size_t start = thread_start_pos[t];  // 当前线程的 index 起始位置
		// 	size_t end = thread_start_pos[t + 1]; // 当前线程的 index 结束位置
		// 	for (size_t i = start; i < end; ++i) {
		// 		data_pos_start[t] += bitmap[i]; // 当前线程的 lossless_data 起始位置
		// 	}
		// }

		// // 使用 OpenMP 并行化遍历 bitmap，并根据线程位置填充 lossless_data
		// #pragma omp parallel for
		// for (int t = 0; t < num_threads; ++t) {
		// 	size_t start = thread_start_pos[t];  // 当前线程的 lossless_data_u 起始位置
		// 	size_t end = thread_start_pos[t + 1]; // 当前线程的 lossless_data_u 结束位置
		// 	size_t pos = data_pos_start[t]; // 当前线程的 lossless_data_u 填充位置
		// 	for (int i = start; i < end; ++i) {
		// 		if (bitmap[i] == 1) { // 需要替换的位置
		// 			U[i] = lossless_data_U[pos]; // 线程独立的 pos
		// 			V[i] = lossless_data_V[pos];
		// 			W[i] = lossless_data_W[pos];
		// 			pos++;
		// 		}
		// 	}
		// }

		// for(int i=0; i< r1; i++){
		// 	for(int j=0; j< r2; j++){
		// 		for(int k=0; k< r3; k++){
		// 			if(static_cast<int>(bitmap[i*r2*r3 + j*r3 + k])){
		// 				U[i*r2*r3 + j*r3 + k] = *(lossless_data_U_pos++);
		// 				V[i*r2*r3 + j*r3 + k] = *(lossless_data_V_pos++);
		// 				W[i*r2*r3 + j*r3 + k] = *(lossless_data_W_pos++);
						
		// 			}
		// 		}
		// 	}
		// }

	std::vector<size_t> indices;
	for (size_t i = 0; i < r1 * r2 * r3; i++) {
		if (bitmap[i]) indices.push_back(i);
	}

	#pragma omp parallel for
	for (size_t i = 0; i < indices.size(); i++) {
		size_t idx = indices[i];
		U[idx] = lossless_data_U[i];
		V[idx] = lossless_data_V[i];
		W[idx] = lossless_data_W[i];
	}
	free(lossless_data_U);
	free(lossless_data_V);
	free(lossless_data_W);
	free(bitmap);
	}
	free(eb_quant_index);
	free(data_quant_index);
	free(dot_data);

	auto pred_time_end = std::chrono::high_resolution_clock::now();
	printf("time for pred in second: %f\n", std::chrono::duration<double>(pred_time_end - pred_time_start).count());



}

template
void
omp_sz_decompress_cp_preserve_3d_record_vertex(const unsigned char * compressed, size_t r1, size_t r2, size_t r3, float *& U, float *& V, float *& W);

template
void
omp_sz_decompress_cp_preserve_3d_record_vertex(const unsigned char * compressed, size_t r1, size_t r2, size_t r3, double *& U, double *& V, double *& W);
