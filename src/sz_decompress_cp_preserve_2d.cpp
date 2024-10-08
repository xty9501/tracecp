#include "sz_decompress_3d.hpp"
#include "sz_decompress_cp_preserve_2d.hpp"
#include "sz_decompress_block_processing.hpp"
#include <limits>
#include <unordered_set>
#include "utilsIO.h"



template<typename T>
void
sz_decompress_cp_preserve_2d_online(const unsigned char * compressed, size_t r1, size_t r2, T *& U, T *& V){
	if(U) free(U);
	if(V) free(V);
	size_t num_elements = r1 * r2;
	const unsigned char * compressed_pos = compressed;
	int base = 0;
	read_variable_from_src(compressed_pos, base);
	//printf("decomp base = %d\n", base);
	double threshold = 0;
	read_variable_from_src(compressed_pos, threshold);
	//printf("decomp threshold = %f\n", threshold);
	int intv_radius = 0;
	read_variable_from_src(compressed_pos, intv_radius);
	//printf("decomp intv_radius = %d\n", intv_radius);
	const int capacity = (intv_radius << 1);
	size_t unpred_data_count = 0;
	read_variable_from_src(compressed_pos, unpred_data_count);
	//printf("decomp unpred_data_count = %ld\n", unpred_data_count);
	const T * unpred_data_pos = (T *) compressed_pos;
	compressed_pos += unpred_data_count*sizeof(T);
	int * eb_quant_index = Huffman_decode_tree_and_data(2*capacity, 2*num_elements, compressed_pos);
	//writefile("decomp_eb_quant_index.txt",eb_quant_index, 2*num_elements);
	int * data_quant_index = Huffman_decode_tree_and_data(2*capacity, 2*num_elements, compressed_pos);
	//writefile("decomp_data_quant_index.txt",data_quant_index, 2*num_elements);
	//printf("pos = %ld\n", compressed_pos - compressed);
	U = (T *) malloc(num_elements*sizeof(T));
	V = (T *) malloc(num_elements*sizeof(T));
	T * U_pos = U;
	T * V_pos = V;
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	// const double threshold=std::numeric_limits<float>::epsilon();
	for(int i=0; i<r1; i++){
		for(int j=0; j<r2; j++){
			// get eb
			if(*eb_quant_index_pos == 0){
				*U_pos = *(unpred_data_pos ++);
				*V_pos = *(unpred_data_pos ++);
				eb_quant_index_pos += 2;
			}
			else{
				for(int k=0; k<2; k++){
					T * cur_data_pos = (k == 0) ? U_pos : V_pos;					
					double eb = pow(base, *eb_quant_index_pos ++) * threshold;
					// double eb = *(eb_quant_index_pos ++) * 1e-3;
					T d0 = (i && j) ? cur_data_pos[-1 - r2] : 0;
					T d1 = (i) ? cur_data_pos[-r2] : 0;
					T d2 = (j) ? cur_data_pos[-1] : 0;
					T pred = d1 + d2 - d0;
					*cur_data_pos = pred + 2 * (data_quant_index_pos[k] - intv_radius) * eb;
				}
			}
			U_pos ++;
			V_pos ++;
			data_quant_index_pos += 2;
		}
	}
	free(eb_quant_index);
	free(data_quant_index);
}

template
void
sz_decompress_cp_preserve_2d_online<float>(const unsigned char * compressed, size_t r1, size_t r2, float *& U, float *& V);

template
void
sz_decompress_cp_preserve_2d_online<double>(const unsigned char * compressed, size_t r1, size_t r2, double *& U, double *& V);

template<typename T>
void
sz_decompress_cp_preserve_2d_online_record_vertex(const unsigned char * compressed, size_t r1, size_t r2, T *& U, T *& V){
	if(U) free(U);
	if(V) free(V);
	size_t num_elements = r1 * r2;
	const unsigned char * compressed_pos = compressed;
	int base = 0;
	//先搞出来bitmap
	// allocate memory for bitmap
	unsigned char * bitmap = (unsigned char *) malloc(num_elements * sizeof(unsigned char));
	size_t num_bytes = (num_elements % 8 == 0) ? num_elements / 8 : num_elements / 8 + 1;
	convertByteArray2IntArray_fast_1b_sz(num_elements, compressed_pos, num_bytes, bitmap);
	//bitmap[3]  == 1

	//再搞出来需要无损的大小
	size_t lossless_count = 0;
	read_variable_from_src(compressed_pos, lossless_count);
	printf("lossless_count = %ld\n", lossless_count);
	// allocate memory for lossless data
	T * lossless_data_U;
	T * lossless_data_V;
	lossless_data_U = read_array_from_src<T>(compressed_pos,lossless_count);
	lossless_data_V = read_array_from_src<T>(compressed_pos,lossless_count);
	T * lossless_data_U_pos = lossless_data_U;
	T * lossless_data_V_pos = lossless_data_V;

	read_variable_from_src(compressed_pos, base);
	printf("base = %d\n", base);
	double threshold = 0;
	read_variable_from_src(compressed_pos, threshold);
	int intv_radius = 0;
	read_variable_from_src(compressed_pos, intv_radius);
	const int capacity = (intv_radius << 1);
	size_t unpred_data_count = 0;
	read_variable_from_src(compressed_pos, unpred_data_count);
	const T * unpred_data_pos = (T *) compressed_pos;
	compressed_pos += unpred_data_count*sizeof(T);
	int * eb_quant_index = Huffman_decode_tree_and_data(2*1024, 2*num_elements, compressed_pos);
	int * data_quant_index = Huffman_decode_tree_and_data(2*capacity, 2*num_elements, compressed_pos);
	printf("pos = %ld\n", compressed_pos - compressed);
	U = (T *) malloc(num_elements*sizeof(T));
	V = (T *) malloc(num_elements*sizeof(T));
	T * U_pos = U;
	T * V_pos = V;
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	double lossless_sum_u = 0;
	// const double threshold=std::numeric_limits<float>::epsilon();
	for(int i=0; i<r1; i++){
		for(int j=0; j<r2; j++){
			// get eb
			if(*eb_quant_index_pos == 0){
				*U_pos = *(unpred_data_pos ++);
				*V_pos = *(unpred_data_pos ++);
				eb_quant_index_pos += 2;
			}
			else{
				for(int k=0; k<2; k++){
					T * cur_data_pos = (k == 0) ? U_pos : V_pos;					
					double eb = pow(base, *eb_quant_index_pos ++) * threshold;
					// double eb = *(eb_quant_index_pos ++) * 1e-3;
					T d0 = (i && j) ? cur_data_pos[-1 - r2] : 0;
					T d1 = (i) ? cur_data_pos[-r2] : 0;
					T d2 = (j) ? cur_data_pos[-1] : 0;
					T pred = d1 + d2 - d0;
					*cur_data_pos = pred + 2 * (data_quant_index_pos[k] - intv_radius) * eb;
				}
			}
			U_pos ++;
			V_pos ++;
			data_quant_index_pos += 2;
		}
	}

	//最后在根据bitmap更新
	for(int i=0; i<r1; i++){
		for(int j=0; j<r2; j++){
		//check bitmap
			if(static_cast<int>(bitmap[i*r2+j]) == 1){
				U[i*r2+j] = *(lossless_data_U_pos ++);
				V[i*r2+j] = *(lossless_data_V_pos ++);
				lossless_sum_u += U[i*r2+j];
				continue;
			}
		}
	}

	free(eb_quant_index);
	free(data_quant_index);
	free(bitmap);
	free(lossless_data_U);
	free(lossless_data_V);
	printf("lossless_count = %ld\n", lossless_data_U_pos - lossless_data_U);
	printf("lossless_sum_u_when_decomp = %lf\n", lossless_sum_u);
	// //loop through bitmap, if bitmap[i] == 1, then replace U[i] and V[i] with lossless_data_U[i] and lossless_data_V[i]
	// size_t count_lossless = 0;
	// for(size_t i = 0; i < num_elements; i++){
	// 	if(static_cast<int>(bitmap[i]) == 1){
	// 		U[i] = lossless_data_U[i];
	// 		V[i] = lossless_data_V[i];
	// 		count_lossless++;
	// 	}
	// }
	// printf("lossless_count = %ld\n", lossless_count);

}

template
void
sz_decompress_cp_preserve_2d_online_record_vertex<float>(const unsigned char * compressed, size_t r1, size_t r2, float *& U, float *& V);

template
void
sz_decompress_cp_preserve_2d_online_record_vertex<double>(const unsigned char * compressed, size_t r1, size_t r2, double *& U, double *& V);

template<typename T>
void
omp_sz_decompress_cp_preserve_2d_online(const unsigned char * compressed, size_t r1, size_t r2, T *& U, T *& V){
	if(U) free(U);
	if(V) free(V);
	size_t num_elements = r1 * r2;
	const unsigned char * compressed_pos = compressed;
	unsigned char * bitmap;
	T * lossless_data_U = NULL;
	T * lossless_data_V = NULL;
	T * lossless_data_U_pos = NULL;
	T * lossless_data_V_pos = NULL;
	//first read index_need_to_fix
	size_t index_need_to_fix_size = 0;
	read_variable_from_src(compressed_pos, index_need_to_fix_size);
	// if not 0, then read bitmap and lossless data
	if (index_need_to_fix_size > 0){
		//allocate memory for bitmap
		bitmap = (unsigned char *) malloc(num_elements * sizeof(unsigned char));
		size_t num_bytes = (num_elements % 8 == 0) ? num_elements / 8 : num_elements / 8 + 1;
		convertByteArray2IntArray_fast_1b_sz(num_elements, compressed_pos, num_bytes, bitmap);
		lossless_data_U = read_array_from_src<T>(compressed_pos, index_need_to_fix_size);
		lossless_data_V = read_array_from_src<T>(compressed_pos, index_need_to_fix_size);
		lossless_data_U_pos = lossless_data_U;
		lossless_data_V_pos = lossless_data_V;
	}

	int num_threads = 0;
	read_variable_from_src(compressed_pos, num_threads);
	//printf("decomp num_threads = %d,pos = %ld\n", num_threads, compressed_pos - compressed);
	//读每个线程的unpred_count
	std::vector<size_t> unpredictable_count(num_threads);
	for (int i = 0; i < num_threads; i++){
		//每次读取一个int
		read_variable_from_src(compressed_pos, unpredictable_count[i]);
		//printf("decomp thread %d unpredictable_count = %d\n", i, unpredictable_count[i]);
	}
	int base = 0;
	read_variable_from_src(compressed_pos, base);
	//printf("decomp base = %d ,pos = %ld\n", base, compressed_pos - compressed);
	double threshold = 0;
	read_variable_from_src(compressed_pos, threshold);
	//printf("decomp threshold = %f, pos = %ld\n", threshold, compressed_pos - compressed);
	int intv_radius = 0;
	read_variable_from_src(compressed_pos, intv_radius);
	//printf("decomp intv_radius = %d,pos = %ld\n", intv_radius, compressed_pos - compressed);
	const int capacity = (intv_radius << 1);
	size_t unpred_data_count = 0;
	read_variable_from_src(compressed_pos, unpred_data_count);
	//printf("decomp total unpred_data_count = %ld ,pos = %ld\n", unpred_data_count, compressed_pos - compressed);

	// const T * unpred_data_pos = (T *) compressed_pos;
	// compressed_pos += unpred_data_count*sizeof(T);
	std::vector<std::vector<T>> unpred_data_each_thread(num_threads);
	//read unpredictable data for each thread
	for (int i = 0; i < num_threads; i++){
		T* raw_data  = read_array_from_src<T>(compressed_pos, unpredictable_count[i]);
		unpred_data_each_thread[i] = std::vector<T>(raw_data, raw_data + unpredictable_count[i]);
	}
	//check num of unpredictable data
	
	// for (int i = 0; i < num_threads; i++){
	// 	printf("decomp thread %d unpredictable data size = %ld, maxval = %f \n", i, unpred_data_each_thread[i].size(), *std::max_element(unpred_data_each_thread[i].begin(), unpred_data_each_thread[i].end()));
	// }
	//printf("decomp pos after write all upredict = %ld\n", compressed_pos - compressed);

	//read upredict_dividing
	size_t unpred_data_dividing_count = 0;
	read_variable_from_src(compressed_pos, unpred_data_dividing_count);
	//printf("decomp unpred_data_dividing_count = %ld\n", unpred_data_dividing_count);
	T * raw_data_dividing = read_array_from_src<T>(compressed_pos, unpred_data_dividing_count);
	std::vector<T> unpred_data_dividing;
	unpred_data_dividing = std::vector<T>(raw_data_dividing, raw_data_dividing + unpred_data_dividing_count);
	//printf("decomp pos after all upredict_dividing = %ld\n", compressed_pos - compressed);
	int * eb_quant_index = Huffman_decode_tree_and_data(2*capacity, 2*num_elements, compressed_pos);
	//printf("decomp pos after huffman_eb_quant_index = %ld\n", compressed_pos - compressed);
	//writefile("decomp_eb_quant_index.txt",eb_quant_index, 2*num_elements);
	//printf("decomp huffman data_quant_index..\n");
	int * data_quant_index = Huffman_decode_tree_and_data(2*capacity, 2*num_elements, compressed_pos);
	//writefile("decomp_data_quant_index.txt",data_quant_index, 2*num_elements);
	printf("decomp pos after huffman_data_quant_index = %ld\n", compressed_pos - compressed);
	// U = (T *) malloc(num_elements*sizeof(T));
	// V = (T *) malloc(num_elements*sizeof(T));
	U = (T *) calloc(num_elements*sizeof(T),sizeof(T));
	V = (T *) calloc(num_elements*sizeof(T),sizeof(T));
	T * U_pos = U;
	T * V_pos = V;
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	// const double threshold=std::numeric_limits<float>::epsilon();


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
	// 创建一个一维标记数组，标记哪些数据点位于划分线上
	std::vector<bool> is_dividing_line(n * m, false);
	// 标记划分线上的数据点
	for (int i = 0; i < n; ++i) {
        for (int col : dividing_cols) {
            if (col < m) {
                is_dividing_line[i * m + col] = true;
            }
        }
    }
    for (int j = 0; j < m; ++j) {
        for (int row : dividing_rows) {
            if (row < n) {
                is_dividing_line[row * m + j] = true;
            }
        }
    }
	int total_blocks = num_threads;
	omp_set_num_threads(num_threads);
	#pragma omp parallel for
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
		T * unpred_data_pos = &unpred_data_each_thread[omp_get_thread_num()][0];
		//print out each thread process which block
		//printf("Thread %d process block %d, start_row = %d, end_row = %d, start_col = %d, end_col = %d\n", omp_get_thread_num(), block_id, start_row, end_row, start_col, end_col);
		for(int i=start_row; i<end_row; ++i){
			if (std::find(dividing_rows.begin(), dividing_rows.end(), i) != dividing_rows.end()) {
				continue;
			}
			for (int j = start_col; j<end_col; ++j){
				if (is_dividing_line[i * m + j]) {
					continue;
				}
				//get eb
				//get position index
				size_t position_idx = (i * r2 + j);
				//get eb
				if(eb_quant_index[2*position_idx] == 0){
					U[position_idx] = *(unpred_data_pos ++);
					V[position_idx] = *(unpred_data_pos ++);
				}
				else{
					for(int k=0; k<2; k++){
						T * cur_data_pos = (k == 0) ? U : V;
						double eb = pow(base, eb_quant_index[2*position_idx + k]) * threshold;
						T d0 = ((i != 0 && j != 0) && (i - start_row != 1 && j - start_col != 1)) ? cur_data_pos[position_idx - 1 - r2] : 0;
						T d1 = (i != 0 && i - start_row != 1) ? cur_data_pos[position_idx - r2] : 0;
						T d2 = (j != 0 && j - start_col != 1) ? cur_data_pos[position_idx - 1] : 0;
						T pred = d1 + d2 - d0;
						cur_data_pos[position_idx] = pred + 2 * (data_quant_index[2*position_idx + k] - intv_radius) * eb;

						// if (position_idx == 2506800){
						// 	printf("=======\n");
						// 	printf("i = %d, j = %d, k = %d, d0 = %f, d1 = %f, d2 = %f, pred = %f, eb = %f, data_quant_index = %d, decompressed = %f\n", i, j, k, d0, d1, d2, pred, eb, data_quant_index[2*position_idx + k], cur_data_pos[position_idx]);
						// 	printf("decomp U[%d] = %f, V[%d] = %f\n", position_idx, U[position_idx], position_idx, V[position_idx]);
						// }
					}
				}
			}
		}
	}

	//目前已经处理完了每个块的数据，现在要特殊处理划分线上的数据
	//串行处理划分线上的数据

	T * unpred_data_dividing_pos = &unpred_data_dividing[0];
	for(int i = 0; i < r1; i++){
		for (int j = 0 ; j < r2; j++){
			if (is_dividing_line[i * m + j]){
				// 处理线上的点
				size_t position_idx = (i * r2 + j);
				if(eb_quant_index[2*position_idx] == 0){
					U[position_idx] = *(unpred_data_dividing_pos ++);
					V[position_idx] = *(unpred_data_dividing_pos ++);
				}
				else{
					for(int k=0; k<2; k++){
						T * cur_data_pos = (k == 0) ? U : V;
						double eb = pow(base, eb_quant_index[2*position_idx + k]) * threshold;
						T d0 = ((i != 0 && j != 0)) ? cur_data_pos[position_idx - 1 - r2] : 0;
						T d1 = (i != 0) ? cur_data_pos[position_idx - r2] : 0;
						T d2 = (j != 0) ? cur_data_pos[position_idx - 1] : 0;
						T pred = d1 + d2 - d0;
						cur_data_pos[position_idx] = pred + 2 * (data_quant_index[2*position_idx + k] - intv_radius) * eb;
					}
				}
			}
		}
	}

	//最后在根据bitmap更新
	if (index_need_to_fix_size != 0){
		for(int i=0; i<r1; i++){
			for(int j=0; j<r2; j++){
			//check bitmap
				if(static_cast<int>(bitmap[i*r2+j]) == 1){
					U[i*r2+j] = *(lossless_data_U_pos ++);
					V[i*r2+j] = *(lossless_data_V_pos ++);
				}
			}
		}
	}

	free(eb_quant_index);
	free(data_quant_index);
	free(bitmap);
	free(lossless_data_U);
	free(lossless_data_V);

/*
	for(int i=0; i<r1; i++){
		for(int j=0; j<r2; j++){
			// get eb
			if(*eb_quant_index_pos == 0){
				*U_pos = *(unpred_data_pos ++);
				*V_pos = *(unpred_data_pos ++);
				eb_quant_index_pos += 2;
			}
			else{
				for(int k=0; k<2; k++){
					T * cur_data_pos = (k == 0) ? U_pos : V_pos;					
					double eb = pow(base, *eb_quant_index_pos ++) * threshold;
					// double eb = *(eb_quant_index_pos ++) * 1e-3;
					T d0 = (i && j) ? cur_data_pos[-1 - r2] : 0;
					T d1 = (i) ? cur_data_pos[-r2] : 0;
					T d2 = (j) ? cur_data_pos[-1] : 0;
					T pred = d1 + d2 - d0;
					*cur_data_pos = pred + 2 * (data_quant_index_pos[k] - intv_radius) * eb;
				}
			}
			U_pos ++;
			V_pos ++;
			data_quant_index_pos += 2;
		}
	}
	free(eb_quant_index);
	free(data_quant_index);
*/
}

template
void
omp_sz_decompress_cp_preserve_2d_online<float>(const unsigned char * compressed, size_t r1, size_t r2, float *& U, float *& V);
