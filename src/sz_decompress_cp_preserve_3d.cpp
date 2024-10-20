#include "sz_decompress_3d.hpp"
#include "sz_decompress_cp_preserve_2d.hpp"
#include "sz_decompress_block_processing.hpp"
#include <limits>
#include <unordered_set>
#include <algorithm>
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
	//先搞出来bitmap
	// allocate memory for bitmap
	unsigned char * bitmap = (unsigned char *) malloc(num_elements * sizeof(unsigned char));
	size_t num_bytes = (num_elements % 8 == 0) ? num_elements / 8 : num_elements / 8 + 1;
	convertByteArray2IntArray_fast_1b_sz(num_elements, compressed_pos, num_bytes, bitmap);
	//再搞出来需要无损的大小
	size_t lossless_count = 0;
	read_variable_from_src(compressed_pos, lossless_count);
	printf("lossless_count = %ld\n", lossless_count);
	// allocate memory for lossless data
	T * lossless_data_U;
	T * lossless_data_V;
	T * lossless_data_W;
	lossless_data_U = read_array_from_src<T>(compressed_pos,lossless_count);
	lossless_data_V = read_array_from_src<T>(compressed_pos,lossless_count);
	lossless_data_W = read_array_from_src<T>(compressed_pos,lossless_count);
	T * lossless_data_U_pos = lossless_data_U;
	T * lossless_data_V_pos = lossless_data_V;
	T * lossless_data_W_pos = lossless_data_W;


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
	free(sign_map_u);
	free(sign_map_v);
	free(sign_map_w);
	free(eb_quant_index);
	free(data_quant_index);
	free(lossless_data_U);
	free(lossless_data_V);
	free(lossless_data_W);
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

	}


	// allocate memory for lossless data
	T * lossless_data_U;
	T * lossless_data_V;
	T * lossless_data_W;
	if (lossless_count != 0){
		lossless_data_U = read_array_from_src<T>(compressed_pos,lossless_count);
		lossless_data_V = read_array_from_src<T>(compressed_pos,lossless_count);
		lossless_data_W = read_array_from_src<T>(compressed_pos,lossless_count);
	}
	T * lossless_data_U_pos = lossless_data_U;
	T * lossless_data_V_pos = lossless_data_V;
	T * lossless_data_W_pos = lossless_data_W;

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
	//if not 0, then read bitmap and lossless data
	if (index_need_to_fix_size != 0){
		// allocate memory for bitmap
		bitmap = (unsigned char *) malloc(num_elements * sizeof(unsigned char));
		memset(bitmap, 0, num_elements * sizeof(unsigned char));
		size_t num_bytes = (num_elements % 8 == 0) ? num_elements / 8 : num_elements / 8 + 1;
		convertByteArray2IntArray_fast_1b_sz(num_elements, compressed_pos, num_bytes, bitmap);
		//再搞出来需要无损的大小
		size_t lossless_count = 0;
		read_variable_from_src(compressed_pos, lossless_count);
		printf("lossless_count = %ld\n", lossless_count);
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

	// 读 block data
	size_t sum_unpred_count_block = 0;
	std::vector<size_t> unpred_data_count(num_threads);
	for (int i = 0; i < num_threads; i++){
		read_variable_from_src(compressed_pos, unpred_data_count[i]);
		sum_unpred_count_block += unpred_data_count[i];
	}
	printf("decomp sum_unpred_count_block = %ld\n", sum_unpred_count_block);

	// 读 face_x data size for each thread
	int croot = std::round(std::cbrt(num_threads));
	int split_line = croot -1;
	size_t sum_unpred_count_face_x = 0;
	std::vector<size_t> face_x_data_count(num_threads);
	for (int i = 0; i < split_line; i++){
		read_variable_from_src(compressed_pos, face_x_data_count[i]);
		sum_unpred_count_face_x += face_x_data_count[i];
	}
	printf("decomp sum_face_x_count_block = %ld\n", sum_unpred_count_face_x);
	// 读face_x data
	std::vector<std::vector<T>> face_x_data_each_thread(split_line);
	for (int i = 0; i < split_line; i++){
		T * raw_data = read_array_from_src<T>(compressed_pos, face_x_data_count[i]);
		face_x_data_each_thread[i] = std::vector<T>(raw_data, raw_data + face_x_data_count[i]);
	}
	// 读 face_y data size for each thread
	std::vector<size_t> face_y_data_count(split_line);
	for (int i = 0; i < split_line; i++){
		read_variable_from_src(compressed_pos, face_y_data_count[i]);
	}
	// 读face_y data
	std::vector<std::vector<T>> face_y_data_each_thread(split_line);
	for (int i = 0; i < split_line; i++){
		T * raw_data = read_array_from_src<T>(compressed_pos, face_y_data_count[i]);
		face_y_data_each_thread[i] = std::vector<T>(raw_data, raw_data + face_y_data_count[i]);
	}

	// 读 face_z data size for each thread
	std::vector<size_t> face_z_data_count(split_line);
	for (int i = 0; i < split_line; i++){
		read_variable_from_src(compressed_pos, face_z_data_count[i]);
	}
	// 读face_z data
	std::vector<std::vector<T>> face_z_data_each_thread(split_line);
	for (int i = 0; i < split_line; i++){
		T * raw_data = read_array_from_src<T>(compressed_pos, face_z_data_count[i]);
		face_z_data_each_thread[i] = std::vector<T>(raw_data, raw_data + face_z_data_count[i]);
	}
	printf("decomp pos after face = %ld\n", compressed_pos - compressed);

	/*
	// 读 edge_x data size for each thread
	std::vector<size_t> edge_x_data_count(split_line);
	for (int i = 0; i < split_line; i++){
		read_variable_from_src(compressed_pos, edge_x_data_count[i]);
	}
	// 读edge_x data
	std::vector<std::vector<T>> edge_x_data_each_thread(split_line);
	for (int i = 0; i < split_line; i++){
		T * raw_data = read_array_from_src<T>(compressed_pos, edge_x_data_count[i]);
		edge_x_data_each_thread[i] = std::vector<T>(raw_data, raw_data + edge_x_data_count[i]);
	}

	// 读 edge_y data size for each thread
	std::vector<size_t> edge_y_data_count(split_line);
	for (int i = 0; i < split_line; i++){
		read_variable_from_src(compressed_pos, edge_y_data_count[i]);
	}
	// 读edge_y data
	std::vector<std::vector<T>> edge_y_data_each_thread(split_line);
	for (int i = 0; i < split_line; i++){
		T * raw_data = read_array_from_src<T>(compressed_pos, edge_y_data_count[i]);
		edge_y_data_each_thread[i] = std::vector<T>(raw_data, raw_data + edge_y_data_count[i]);
	}

	// 读 edge_z data size for each thread
	std::vector<size_t> edge_z_data_count(split_line);
	for (int i = 0; i < split_line; i++){
		read_variable_from_src(compressed_pos, edge_z_data_count[i]);
	}
	// 读edge_z data
	std::vector<std::vector<T>> edge_z_data_each_thread(split_line);
	for (int i = 0; i < split_line; i++){
		T * raw_data = read_array_from_src<T>(compressed_pos, edge_z_data_count[i]);
		edge_z_data_each_thread[i] = std::vector<T>(raw_data, raw_data + edge_z_data_count[i]);
	}
	for (int threadID = 0; threadID < split_line; threadID++){
		if (edge_x_data_each_thread[threadID].size() > 0)
			printf("decomp_edge_x thread %d, unpred_data size = %ld,maxvalue = %f,minvalue = %f\n", threadID, edge_x_data_each_thread[threadID].size(), *std::max_element(edge_x_data_each_thread[threadID].begin(), edge_x_data_each_thread[threadID].end()), *std::min_element(edge_x_data_each_thread[threadID].begin(), edge_x_data_each_thread[threadID].end()));
		if (edge_y_data_each_thread[threadID].size() > 0)
			printf("decomp_edge_y thread %d, unpred_data size = %ld,maxvalue = %f,minvalue = %f\n", threadID, edge_y_data_each_thread[threadID].size(), *std::max_element(edge_y_data_each_thread[threadID].begin(), edge_y_data_each_thread[threadID].end()), *std::min_element(edge_y_data_each_thread[threadID].begin(), edge_y_data_each_thread[threadID].end()));
		if(edge_z_data_each_thread[threadID].size() > 0)
			printf("decomp_edge_z thread %d, unpred_data size = %ld,maxvalue = %f,minvalue = %f\n", threadID, edge_z_data_each_thread[threadID].size(), *std::max_element(edge_z_data_each_thread[threadID].begin(), edge_z_data_each_thread[threadID].end()), *std::min_element(edge_z_data_each_thread[threadID].begin(), edge_z_data_each_thread[threadID].end()));
		}
	printf("decomp pos after edge = %ld\n", compressed_pos - compressed);
	*/
	
	//corner data
	std::vector<size_t> corner_data_count(num_threads);
	std::vector<std::vector<T>> corner_data_each_thread(num_threads);

	for (int i = 0; i < num_threads; i++){
		read_variable_from_src(compressed_pos, corner_data_count[i]);
	}
	for (int i = 0; i < num_threads; i++){
		T * raw_data = read_array_from_src<T>(compressed_pos, corner_data_count[i]);
		corner_data_each_thread[i] = std::vector<T>(raw_data, raw_data + corner_data_count[i]);
	}

	int base = 0;
	read_variable_from_src(compressed_pos, base);
	printf("decomp base = %d\n", base);
	double threshold = 0;
	read_variable_from_src(compressed_pos, threshold);
	int intv_radius = 0;
	read_variable_from_src(compressed_pos, intv_radius);
	const int capacity = (intv_radius << 1);
	size_t total_unpred_data_count = 0;
	read_variable_from_src(compressed_pos, total_unpred_data_count);
	printf("unpred_data_count = %ld\n", total_unpred_data_count);
	//now read unpred_data for each thread
	std::vector<std::vector<T>> unpred_data_each_thread(num_threads);
	for (int i = 0; i < num_threads; i++){
		T * raw_data = read_array_from_src<T>(compressed_pos, unpred_data_count[i]);
		unpred_data_each_thread[i] = std::vector<T>(raw_data, raw_data + unpred_data_count[i]);
	}
	//now read dividing line data size
	size_t unpred_data_dividing_count = 0;
	read_variable_from_src(compressed_pos, unpred_data_dividing_count);
	//now read dividing line data
	T * raw_data_dividing = read_array_from_src<T>(compressed_pos, unpred_data_dividing_count);
	std::vector<T> unpred_data_dividing;
	unpred_data_dividing = std::vector<T>(raw_data_dividing, raw_data_dividing + unpred_data_dividing_count);
	// now read eb_quant_index
	int * eb_quant_index = Huffman_decode_tree_and_data(2*capacity, num_elements, compressed_pos);
	// now read data_quant_index
	int * data_quant_index = Huffman_decode_tree_and_data(2*capacity, 3*num_elements, compressed_pos);
	printf("decomp pos after huffmans = %ld\n", compressed_pos - compressed);
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

	int t;
	int cube_root = std::round(std::cbrt(num_threads));
	if (cube_root * cube_root * cube_root  == num_threads){
		t = cube_root;
	}
	else{
		printf("num_threads is not a cube number\n");
		exit(0);
	}
	// 计算每个块的大小
	int block_r1 = r1 / t;
	int block_r2 = r2 / t;
	int block_r3 = r3 / t;

	//处理余数的情况
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
	for (int i = 0; i < dividing_r1.size(); i++) {
		printf("decomp dividing_r1[%d] = %d\n", i, dividing_r1[i]);
	}

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
	
	size_t processed_data_dec_count = 0;
	// 处理块
	int total_block = t * t * t;
	omp_set_num_threads(total_block);
	#pragma omp parallel for num_threads(num_threads) reduction(+:processed_data_dec_count)
	for (int block_id = 0; block_id < total_block; ++block_id){
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
		//printf("Thread %d process block %d, start_i = %d, end_i = %d, start_j = %d, end_j = %d, start_k = %d, end_k = %d\n", omp_get_thread_num(), block_id, start_i, end_i, start_j, end_j, start_k, end_k);
		T *unpred_data_pos = &unpred_data_each_thread[omp_get_thread_num()][0];

		//处理块内部的数据
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
					//开始处理数据
					processed_data_dec_count++;
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


	//串行
	/*
	//现在特殊处理划分线上的数据
	T *unpred_data_dividing_pos = &unpred_data_dividing[0];
	for (int i = 0; i < r1; ++i) {
		for (int j = 0; j < r2; ++j) {
			for (int k = 0; k < r3; ++k) {
				if (on_dividing_line[i * r2 * r3 + j * r3 + k]) {
					//开始处理数据
					size_t position_idx = i * r2 * r3 + j * r3 + k;
					if (eb_quant_index[position_idx] == 0){
						U[position_idx] = *(unpred_data_dividing_pos++);
						V[position_idx] = *(unpred_data_dividing_pos++);
						W[position_idx] = *(unpred_data_dividing_pos++);
					}
					else{
						double eb = pow(base, eb_quant_index[position_idx]) * threshold;
						for (int p = 0; p < 3; ++p){
							T *cur_data_field = (p == 0) ? U : (p == 1) ? V : W;
							T d0 = (i && j && k) ? cur_data_field[position_idx - r2 * r3 - r3 - 1] : 0;
							T d1 = (i && j) ? cur_data_field[position_idx - r2 * r3 - r3] : 0;
							T d2 = (i && k) ? cur_data_field[position_idx - r2 * r3 - 1] : 0;
							T d3 = (i) ? cur_data_field[position_idx - r2 * r3] : 0;
							T d4 = (j && k) ? cur_data_field[position_idx - r3 - 1] : 0;
							T d5 = (j) ? cur_data_field[position_idx - r3] : 0;
							T d6 = (k) ? cur_data_field[position_idx - 1] : 0;
							T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
							cur_data_field[position_idx] = pred + 2 * (data_quant_index[3 * position_idx + p] - intv_radius) * eb;
						}
					}
				}
			}
		}
	}
	*/

	//优化: 并行解压TODO
	ptrdiff_t dim0_offset = r2 * r3;
	ptrdiff_t dim1_offset = r3;
	ptrdiff_t cell_dim0_offset = (r2-1) * (r3-1);
	ptrdiff_t cell_dim1_offset = r3-1;

	omp_set_num_threads(split_line);
	// Process faces perpendicular to the X-axis(jk-plane)
	#pragma omp parallel for num_threads(split_line) reduction(+:processed_data_dec_count)
	for (size_t idx = 0; idx < dividing_r1.size(); ++idx) {
		int i = dividing_r1[idx];
        if (i >= r1) continue;
		T *unpred_data_pos = &face_x_data_each_thread[idx][0];
        for (int j = 0; j < r2; ++j) {
            if (is_dividing_r2[j]) continue;
            for (int k = 0; k < r3; ++k) {
                if (is_dividing_r3[k]) continue;
                //开始处理数据
				processed_data_dec_count++;
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
                        // T d0 = ((i && j && k)) ? cur_data_field[position_idx - r2 * r3 - r3 - 1] : 0;
                        // T d1 = ((i && j)) ? cur_data_field[position_idx - r2 * r3 - r3] : 0;
                        // T d2 = ((i && k) ) ? cur_data_field[position_idx - r2 * r3 - 1] : 0;
                        // T d3 = (i) ? cur_data_field[position_idx - r2 * r3] : 0;
                        // T d4 = ((j && k)) ? cur_data_field[position_idx - r3 - 1] : 0;
                        // T d5 = (j) ? cur_data_field[position_idx - r3] : 0;
                        // T d6 = (k) ? cur_data_field[position_idx - 1] : 0;
                        // T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;

						T d0 = ((j && k) && (j - start_j != 1 && k-start_k != 1)) ? cur_data_field[position_idx - dim1_offset - 1] : 0;
						T d1 = (j && (j - start_j != 1)) ? cur_data_field[position_idx - dim1_offset] : 0;
						T d2 = (k && (k - start_k != 1)) ? cur_data_field[position_idx - 1] : 0;
						T pred = d1 + d2 - d0;
                        cur_data_field[position_idx] = pred + 2 * (data_quant_index[3 * position_idx + p] - intv_radius) * eb;
                    }
                }

			}
		}
	}


	// Process faces perpendicular to the Y-axis(ik-plane)
	#pragma omp parallel for num_threads(split_line) reduction(+:processed_data_dec_count)
	for (size_t idx = 0; idx < dividing_r2.size(); ++idx) {
		int j = dividing_r2[idx];
		if (j >= r2) continue;
		T *unpred_data_pos = &face_y_data_each_thread[idx][0];
		for (int i = 0; i < r1; ++i) {
			if (is_dividing_r1[i]) continue;
			for (int k = 0; k < r3; ++k) {
				if (is_dividing_r3[k]) continue;
				//开始处理数据
				processed_data_dec_count++;
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
                        // T d0 = ((i && j && k) ) ? cur_data_field[position_idx - r2 * r3 - r3 - 1] : 0;
                        // T d1 = ((i && j) ) ? cur_data_field[position_idx - r2 * r3 - r3] : 0;
                        // T d2 = ((i && k)) ? cur_data_field[position_idx - r2 * r3 - 1] : 0;
                        // T d3 = (i) ? cur_data_field[position_idx - r2 * r3] : 0;
                        // T d4 = ((j && k)) ? cur_data_field[position_idx - r3 - 1] : 0;
                        // T d5 = (j) ? cur_data_field[position_idx - r3] : 0;
                        // T d6 = (k) ? cur_data_field[position_idx - 1] : 0;
                        // T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;

						T d0 = ((i && k) && (i - start_i != 1 && k - start_k != 1)) ? cur_data_field[position_idx -1 - dim0_offset] : 0;
						T d1 = (i && (i - start_i != 1)) ? cur_data_field[position_idx - dim0_offset] : 0;
						T d2 = (k && (k - start_k != 1)) ? cur_data_field[position_idx - 1] : 0;
						T pred = d1 + d2 - d0;
                        cur_data_field[position_idx] = pred + 2 * (data_quant_index[3 * position_idx + p] - intv_radius) * eb;
                    }
                }
			}
		}
	}
	
	// Process faces perpendicular to the Z-axis(ij-plane)
    #pragma omp parallel for num_threads(split_line) reduction(+:processed_data_dec_count)
    for (size_t idx = 0; idx < dividing_r3.size(); ++idx) {
        int k = dividing_r3[idx];
        if (k >= r3) continue;
        T *unpred_data_pos = &face_z_data_each_thread[idx][0];
        for (int i = 0; i < r1; ++i) {
            if (is_dividing_r1[i]) continue;
            for (int j = 0; j < r2; ++j) {
                if (is_dividing_r2[j]) continue;
                //开始处理数据
				processed_data_dec_count++;
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
                        // T d0 = ((i && j && k) ) ? cur_data_field[position_idx - r2 * r3 - r3 - 1] : 0;
                        // T d1 = ((i && j) ) ? cur_data_field[position_idx - r2 * r3 - r3] : 0;
                        // T d2 = ((i && k)) ? cur_data_field[position_idx - r2 * r3 - 1] : 0;
                        // T d3 = (i) ? cur_data_field[position_idx - r2 * r3] : 0;
                        // T d4 = ((j && k)) ? cur_data_field[position_idx - r3 - 1] : 0;
                        // T d5 = (j) ? cur_data_field[position_idx - r3] : 0;
                        // T d6 = (k) ? cur_data_field[position_idx - 1] : 0;
                        // T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
						T d0 = ((i && j) && (i - start_i != 1 && j - start_j != 1)) ? cur_data_field[position_idx - dim0_offset - dim1_offset] : 0;
						T d1 = (i && (i - start_i != 1)) ? cur_data_field[position_idx - dim0_offset] : 0;
						T d2 = (j && (j - start_j != 1)) ? cur_data_field[position_idx - dim1_offset] : 0;
						T pred = d1 + d2 - d0;
                        cur_data_field[position_idx] = pred + 2 * (data_quant_index[3 * position_idx + p] - intv_radius) * eb;
                    }
                }
			}
		}
	}

/*
	// Process Edges (Points on the intersection of two dividing planes but not corners)

	// Process edges along the X-axis(垂直于jk平面)
    #pragma omp parallel for num_threads(num_threads) schedule(static) reduction(+:processed_data_dec_count)
    for (size_t idx_j = 0; idx_j < dividing_r2.size(); ++idx_j) {
        int j = dividing_r2[idx_j];
        if (j >= r2) continue;
        for (size_t idx_k = 0; idx_k < dividing_r3.size(); ++idx_k) {
            int k = dividing_r3[idx_k];
            if (k >= r3) continue;
			T *unpred_data_pos = &edge_x_data_each_thread[omp_get_thread_num()][0];
            for (int i = 0; i < r1; ++i) {
                if (is_dividing_r1[i]) continue; // Skip corners
                //开始处理数据
				processed_data_dec_count++;
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
                        // T d0 = ((i && j && k) ) ? cur_data_field[position_idx - r2 * r3 - r3 - 1] : 0;
                        // T d1 = ((i && j) ) ? cur_data_field[position_idx - r2 * r3 - r3] : 0;
                        // T d2 = ((i && k)) ? cur_data_field[position_idx - r2 * r3 - 1] : 0;
                        // T d3 = (i) ? cur_data_field[position_idx - r2 * r3] : 0;
                        // T d4 = ((j && k)) ? cur_data_field[position_idx - r3 - 1] : 0;
                        // T d5 = (j) ? cur_data_field[position_idx - r3] : 0;
                        // T d6 = (k) ? cur_data_field[position_idx - 1] : 0;
                        // T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
						T d0 = ((i) && (i - start_i >= 1)) ? cur_data_field[position_idx - dim0_offset] : 0;
						T pred = d0;
                        cur_data_field[position_idx] = pred + 2 * (data_quant_index[3 * position_idx + p] - intv_radius) * eb;
                    }
                }
			}
		}
	}

	// Process edges along the Y-axis(orthogonal to ik-plane)
    #pragma omp parallel for num_threads(num_threads) schedule(static) reduction(+:processed_data_dec_count)
    for (size_t idx_i = 0; idx_i < dividing_r1.size(); ++idx_i) {
        int i = dividing_r1[idx_i];
        if (i >= r1) continue;
        for (size_t idx_k = 0; idx_k < dividing_r3.size(); ++idx_k) {
            int k = dividing_r3[idx_k];
            if (k >= r3) continue;
            T *unpred_data_pos = &edge_y_data_each_thread[omp_get_thread_num()][0];
            for (int j = 0; j < r2; ++j) {
                if (is_dividing_r2[j]) continue; // Skip corners
                //开始处理数据
				processed_data_dec_count++;
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
                        // T d0 = ((i && j && k) ) ? cur_data_field[position_idx - r2 * r3 - r3 - 1] : 0;
                        // T d1 = ((i && j) ) ? cur_data_field[position_idx - r2 * r3 - r3] : 0;
                        // T d2 = ((i && k)) ? cur_data_field[position_idx - r2 * r3 - 1] : 0;
                        // T d3 = (i) ? cur_data_field[position_idx - r2 * r3] : 0;
                        // T d4 = ((j && k)) ? cur_data_field[position_idx - r3 - 1] : 0;
                        // T d5 = (j) ? cur_data_field[position_idx - r3] : 0;
                        // T d6 = (k) ? cur_data_field[position_idx - 1] : 0;
                        // T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
						T d0 = ((j) && (j - start_j != 1)) ? cur_data_field[position_idx - dim1_offset] : 0;
						T pred = d0;
                        cur_data_field[position_idx] = pred + 2 * (data_quant_index[3 * position_idx + p] - intv_radius) * eb;
                    }
                }
			}
		}
	}

	// Process edges along the Z-axis
    #pragma omp parallel for num_threads(num_threads) schedule(static) reduction(+:processed_data_dec_count)
    for (size_t idx_i = 0; idx_i < dividing_r1.size(); ++idx_i) {
        int i = dividing_r1[idx_i];
        if (i >= r1) continue;
        for (size_t idx_j = 0; idx_j < dividing_r2.size(); ++idx_j) {
            int j = dividing_r2[idx_j];
            if (j >= r2) continue;
			T *unpred_data_pos = &edge_z_data_each_thread[omp_get_thread_num()][0];
            for (int k = 0; k < r3; ++k) {
                if (is_dividing_r3[k]) continue; // Skip corners
                //开始处理数据
				processed_data_dec_count++;
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
                        // T d0 = ((i && j && k) ) ? cur_data_field[position_idx - r2 * r3 - r3 - 1] : 0;
                        // T d1 = ((i && j) ) ? cur_data_field[position_idx - r2 * r3 - r3] : 0;
                        // T d2 = ((i && k)) ? cur_data_field[position_idx - r2 * r3 - 1] : 0;
                        // T d3 = (i) ? cur_data_field[position_idx - r2 * r3] : 0;
                        // T d4 = ((j && k)) ? cur_data_field[position_idx - r3 - 1] : 0;
                        // T d5 = (j) ? cur_data_field[position_idx - r3] : 0;
                        // T d6 = (k) ? cur_data_field[position_idx - 1] : 0;
                        // T pred = d0 + d3 + d5 + d6 - d1 - d2 - d4;
						T d0 = (k && (k - start_k != 1)) ? cur_data_field[position_idx - 1] : 0;
						T pred = d0;
                        cur_data_field[position_idx] = pred + 2 * (data_quant_index[3 * position_idx + p] - intv_radius) * eb;
                    }
                }
			}
		}
	}
*/

/*
	// Process Edges : just fucking decompress losslessly
	
	// Process edges along the X-axis(垂直于jk平面)
    #pragma omp parallel for num_threads(split_line)  reduction(+:processed_data_dec_count)
    for (size_t idx_j = 0; idx_j < dividing_r2.size(); ++idx_j) {
        int j = dividing_r2[idx_j];
        if (j >= r2) continue;
        for (size_t idx_k = 0; idx_k < dividing_r3.size(); ++idx_k) {
            int k = dividing_r3[idx_k];
            if (k >= r3) continue;
			T *unpred_data_pos = &edge_x_data_each_thread[idx_j][0];
            for (int i = 0; i < r1; ++i) {
                if (is_dividing_r1[i]) continue; // Skip corners
                //开始处理数据
				processed_data_dec_count++;
                size_t position_idx = i * r2 * r3 + j * r3 + k;
				U[position_idx] = *(unpred_data_pos++);
                V[position_idx] = *(unpred_data_pos++);
                W[position_idx] = *(unpred_data_pos++);
			}
		}
	}

	// Process edges along the Y-axis(orthogonal to ik-plane)
    #pragma omp parallel for num_threads(split_line)reduction(+:processed_data_dec_count)
    for (size_t idx_i = 0; idx_i < dividing_r1.size(); ++idx_i) {
        int i = dividing_r1[idx_i];
        if (i >= r1) continue;
        for (size_t idx_k = 0; idx_k < dividing_r3.size(); ++idx_k) {
            int k = dividing_r3[idx_k];
            if (k >= r3) continue;
            T *unpred_data_pos = &edge_y_data_each_thread[idx_i][0];
            for (int j = 0; j < r2; ++j) {
                if (is_dividing_r2[j]) continue; // Skip corners
                //开始处理数据
				processed_data_dec_count++;
                size_t position_idx = i * r2 * r3 + j * r3 + k;
				U[position_idx] = *(unpred_data_pos++);
                V[position_idx] = *(unpred_data_pos++);
                W[position_idx] = *(unpred_data_pos++);
			}
		}
	}

	// Process edges along the Z-axis
    #pragma omp parallel for num_threads(split_line) reduction(+:processed_data_dec_count)
    for (size_t idx_i = 0; idx_i < dividing_r1.size(); ++idx_i) {
        int i = dividing_r1[idx_i];
        if (i >= r1) continue;
        for (size_t idx_j = 0; idx_j < dividing_r2.size(); ++idx_j) {
            int j = dividing_r2[idx_j];
            if (j >= r2) continue;
			T *unpred_data_pos = &edge_z_data_each_thread[idx_i][0];
            for (int k = 0; k < r3; ++k) {
                if (is_dividing_r3[k]) continue; // Skip corners
                //开始处理数据
				processed_data_dec_count++;
                size_t position_idx = i * r2 * r3 + j * r3 + k;
				U[position_idx] = *(unpred_data_pos++);
                V[position_idx] = *(unpred_data_pos++);
				W[position_idx] = *(unpred_data_pos++);
			}
		}
	}

*/

/*
	T * corner_data_pos = corner_data;
	// Collect all corner points 串行处理
    for (int i : dividing_r1) {
        if (i >= r1) continue;
        for (int j : dividing_r2) {
            if (j >= r2) continue;
            for (int k : dividing_r3) {
                if (k >= r3) continue;
				//开始处理数据
				processed_data_dec_count++;
				size_t position_idx = i * r2 * r3 + j * r3 + k;
				U[position_idx] = *(corner_data_pos++);
				V[position_idx] = *(corner_data_pos++);
				W[position_idx] = *(corner_data_pos++);
			}
		}
	}
*/
	//直接处理棱和corner
	#pragma omp parallel for num_threads(num_threads) schedule(static) reduction(+:processed_data_dec_count)
	for (int block_id = 0; block_id < num_threads; ++block_id){
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

	T * corner_pos = corner_data_each_thread[block_id].data();

	for (int i = start_i; i < end_i; ++i) {
		for (int j = start_j; j < end_j; ++j) {
			for (int k = start_k; k < end_k; ++k) {
				size_t position_idx = i * r2 * r3 + j * r3 + k;
				if ((is_dividing_r1[i] && is_dividing_r2[j]) || (is_dividing_r1[i] && is_dividing_r3[k]) || (is_dividing_r2[j] && is_dividing_r3[k]) || (is_dividing_r1[i] && is_dividing_r2[j] && is_dividing_r3[k])) {
					//开始处理数据
					processed_data_dec_count++;
					U[position_idx] = *(corner_pos++);
					V[position_idx] = *(corner_pos++);
					W[position_idx] = *(corner_pos++);
				}
			}
		}
	}
	}

	

	printf("decomp: check if all data points are processed: %ld, %ld\n", processed_data_dec_count, num_elements);
	//最后再根据bitmap来处理lossless的数据
	if (index_need_to_fix_size != 0){
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
omp_sz_decompress_cp_preserve_3d_online_abs_record_vertex(const unsigned char * compressed, size_t r1, size_t r2, size_t r3, float *& U, float *& V, float *& W);

template
void
omp_sz_decompress_cp_preserve_3d_online_abs_record_vertex(const unsigned char * compressed, size_t r1, size_t r2, size_t r3, double *& U, double *& V, double *& W);