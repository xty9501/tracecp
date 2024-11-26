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
	int * eb_quant_index = Huffman_decode_tree_and_data(2*1024, 2*num_elements, compressed_pos);
	//writefile("decomp_eb_quant_index.txt",eb_quant_index, 2*num_elements);
	int * data_quant_index = Huffman_decode_tree_and_data(capacity, 2*num_elements, compressed_pos);
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

	// if (WRITE_OUT_EB == 1){
	// 	std::vector<float> result_eb(num_elements);
	// 	for (int i = 0; i < 2*num_elements; i+2){
	// 		result_eb[i] = pow(base, eb_quant_index[i]) * threshold;
	// 	}
	// 	writefile(("~/data/ocean_eb_rel.bin"), result_eb.data(), result_eb.size());
	// }
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
	unsigned char * bitmap;
	T * lossless_data_U;
	T * lossless_data_V;
	//搞出来需要无损的大小
	size_t lossless_count = 0;
	read_variable_from_src(compressed_pos, lossless_count);
	printf("lossless_count = %ld\n", lossless_count);

	if (lossless_count != 0){
		bitmap = (unsigned char *) malloc(num_elements * sizeof(unsigned char));
		size_t num_bytes = (num_elements % 8 == 0) ? num_elements / 8 : num_elements / 8 + 1;
		convertByteArray2IntArray_fast_1b_sz(num_elements, compressed_pos, num_bytes, bitmap);
		lossless_data_U = read_array_from_src<T>(compressed_pos,lossless_count);
		lossless_data_V = read_array_from_src<T>(compressed_pos,lossless_count);
	}


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

	// if(WRITE_OUT_EB == 1){
	// 	std::vector<float> result_eb(num_elements);
	// 	for (int i = 0; i < 2*num_elements; i+2){
	// 		result_eb[i] = pow(base, eb_quant_index[i]) * threshold;
	// 	}
	// 	writefile(("~/data/ocean_eb_abs.bin"), result_eb.data(), result_eb.size());
	// }

	if (lossless_count != 0){
		T * lossless_data_U_pos = lossless_data_U;
		T * lossless_data_V_pos = lossless_data_V;
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
		free(bitmap);
		free(lossless_data_U);
		free(lossless_data_V);
		printf("lossless_count = %ld\n", lossless_data_U_pos - lossless_data_U);
	}

	free(eb_quant_index);
	free(data_quant_index);

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
	printf("decomp num_threads = %d,pos = %ld\n", num_threads, compressed_pos - compressed);
	
	//读每个block的unpred_count
	std::vector<size_t> unpredictable_count(num_threads);
	for (int i = 0; i < num_threads; i++){
		//每次读取一个int
		read_variable_from_src(compressed_pos, unpredictable_count[i]);
		//printf("decomp thread %d unpredictable_count = %d\n", i, unpredictable_count[i]);
	}
	// 读每个block的unpred_data
	std::vector<std::vector<T>> unpred_data_each_thread(num_threads);
	for (int i = 0; i < num_threads; i++){
		T* raw_data  = read_array_from_src<T>(compressed_pos, unpredictable_count[i]);
		unpred_data_each_thread[i] = std::vector<T>(raw_data, raw_data + unpredictable_count[i]);
	}
	//print the detail for each thread
	// for (int i = 0; i < num_threads; i++){
	// 	printf("decomp thread %d unpredictable data size = %ld, maxval = %f \n", i, unpred_data_each_thread[i].size(), *std::max_element(unpred_data_each_thread[i].begin(), unpred_data_each_thread[i].end()));
	// }

	int t = std::sqrt(num_threads);
	//读每个row的unpred_count
	std::vector<size_t> unpredictable_count_row((t-1)*t);
	for (int i = 0; i < (t-1)*t ; i++){
		read_variable_from_src(compressed_pos, unpredictable_count_row[i]);
	}
	//读每个row的unpred_data
	std::vector<std::vector<T>> unpred_data_row((t-1)*t);
	for (int i = 0; i < (t-1)*t ; i++){
		T* raw_data = read_array_from_src<T>(compressed_pos,unpredictable_count_row[i]);
		unpred_data_row[i] = std::vector<T>(raw_data, raw_data + unpredictable_count_row[i]);
	}

	//print detail for each thread for row
	// for (int i = 0; i < (t-1)*t; i++){
	// 	printf("decomp thread %d unpredictable row size = %ld, maxval = %f \n", i, unpred_data_row[i].size(), (unpred_data_row[i].size() == 0) ? 0: *std::max_element(unpred_data_row[i].begin(), unpred_data_row[i].end()));
	// }


	//读每个col的unpred_count
	std::vector<size_t> unpredictable_count_col((t-1)*t);
	for (int i = 0; i < (t-1)*t ; i++){
		read_variable_from_src(compressed_pos, unpredictable_count_col[i]);
	}
	//读每个col的unpred_data
	std::vector<std::vector<T>> unpred_data_col((t-1)*t);
	for (int i = 0; i < (t-1)*t ; i++){
		T* raw_data = read_array_from_src<T>(compressed_pos,unpredictable_count_col[i]);
		unpred_data_col[i] = std::vector<T>(raw_data, raw_data + unpredictable_count_col[i]);
	}

	//读串行的dot的unpred_count
	size_t dot_size;
	read_variable_from_src(compressed_pos, dot_size);
	T* unpred_data_dot = read_array_from_src<T>(compressed_pos, dot_size);


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

	// const T * unpred_data_pos = (T *) compressed_pos;
	// compressed_pos += unpred_data_count*sizeof(T);

	//check num of unpredictable data
	
	// for (int i = 0; i < num_threads; i++){
	// 	printf("decomp thread %d unpredictable data size = %ld, maxval = %f \n", i, unpred_data_each_thread[i].size(), *std::max_element(unpred_data_each_thread[i].begin(), unpred_data_each_thread[i].end()));
	// }
	//printf("decomp pos after write all upredict = %ld\n", compressed_pos - compressed);

	/*
	// //read upredict_dividing
	// size_t unpred_data_dividing_count = 0;
	// read_variable_from_src(compressed_pos, unpred_data_dividing_count);
	// //printf("decomp unpred_data_dividing_count = %ld\n", unpred_data_dividing_count);
	// T * raw_data_dividing = read_array_from_src<T>(compressed_pos, unpred_data_dividing_count);
	// std::vector<T> unpred_data_dividing;
	// unpred_data_dividing = std::vector<T>(raw_data_dividing, raw_data_dividing + unpred_data_dividing_count);
	// //printf("decomp pos after all upredict_dividing = %ld\n", compressed_pos - compressed);
	*/

	// serial huffman decode for eb quant ****************************************
	// int * eb_quant_index = Huffman_decode_tree_and_data(2*1024, 2*num_elements, compressed_pos);
	// serial huffman decode for eb quant ****************************************
	
	// parallel huffman decode for eb quant ****************************************
	// read each compressed_size for eb_quant_index first
	std::vector<size_t> compressed_size_eb(num_threads);
	for (int i = 0; i < num_threads; i++){
		read_variable_from_src(compressed_pos, compressed_size_eb[i]);
		//printf("decomp thread %d eb_quant_index size = %ld\n", i, compressed_size_eb[i]);
	}
	std::vector<const unsigned char *> compressed_chunk_eb_start(num_threads);
	for (int i = 0; i < num_threads; i++){
		compressed_chunk_eb_start[i] = compressed_pos;
		compressed_pos += compressed_size_eb[i];
	}
	int * eb_quant_index = (int *) malloc(2*num_elements*sizeof(int));

	#pragma omp parallel for num_threads(num_threads)
	for (int i = 0; i < num_threads; i++){
		size_t start_pos = i*num_elements/num_threads;
		size_t end_pos = (i ==num_threads - 1) ? num_elements : (i+1)*num_elements/num_threads;
		const unsigned char * local_compressed_pos = compressed_chunk_eb_start[i];
		size_t local_num_elements = 2 * (end_pos - start_pos);
		int * local_eb_quant_index = Huffman_decode_tree_and_data(2*1024, local_num_elements, local_compressed_pos);
		std::copy(local_eb_quant_index, local_eb_quant_index + local_num_elements, eb_quant_index + 2*start_pos);
		free(local_eb_quant_index);
	}

	//printf("decomp eb max = %d\n", *std::max_element(eb_quant_index, eb_quant_index + 2*num_elements));
	//printf("decomp eb min = %d\n", *std::min_element(eb_quant_index, eb_quant_index + 2*num_elements));
	// parallel huffman decode for eb quant ****************************************

	/*	
	// serial huffman decode for data quant ****************************************
	int * data_quant_index = Huffman_decode_tree_and_data(capacity, 2*num_elements, compressed_pos);
	//writefile("decomp_data_quant_index.txt",data_quant_index, 2*num_elements);
	printf("decomp data max = %d\n", *std::max_element(data_quant_index, data_quant_index + 2*num_elements));
	printf("decomp data min = %d\n", *std::min_element(data_quant_index, data_quant_index + 2*num_elements));
	printf("decomp pos after huffman_data_quant_index = %ld\n", compressed_pos - compressed);
	// serial huffman decode for data quant ****************************************
	*/

	// parallel huffman decode for data quant ****************************************
	// read each compressed_size for data_quant_index first
	std::vector<size_t> compressed_size_data(num_threads);
	for (int i = 0; i < num_threads; i++){
		read_variable_from_src(compressed_pos, compressed_size_data[i]);
		//printf("decomp thread %d data_quant_index size = %ld\n", i, compressed_size_data[i]);
	}
	std::vector<const unsigned char *> compressed_chunk_data_start(num_threads);
	for (int i = 0; i < num_threads; i++){
		compressed_chunk_data_start[i] = compressed_pos;
		compressed_pos += compressed_size_data[i];
	}
	int * data_quant_index = (int *) malloc(2*num_elements*sizeof(int));
	#pragma omp parallel for num_threads(num_threads)
	for (int i = 0; i < num_threads; i++){
		size_t start_pos = i*num_elements/num_threads;
		size_t end_pos = (i ==num_threads - 1) ? num_elements : (i+1)*num_elements/num_threads;
		const unsigned char * local_compressed_pos = compressed_chunk_data_start[i];
		size_t local_num_elements = 2 * (end_pos - start_pos);
		int * local_data_quant_index = Huffman_decode_tree_and_data(capacity, local_num_elements, local_compressed_pos);
		std::copy(local_data_quant_index, local_data_quant_index + local_num_elements, data_quant_index + 2*start_pos);
		free(local_data_quant_index);
	}
	//printf("decomp parallel data max = %d\n", *std::max_element(data_quant_index, data_quant_index + 2*num_elements));
	//printf("decomp parallel data min = %d\n", *std::min_element(data_quant_index, data_quant_index + 2*num_elements));



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
	// int t = sqrt(num_threads);
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
	size_t total_processed = 0;	
	// printf("use %d threads for block processing\n", num_threads);
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
		//printf("dec-threadID = %d, block_id = %d, start_row = %d, end_row = %d, start_col = %d, end_col = %d\n", omp_get_thread_num(), block_id, start_row, end_row, start_col, end_col);
		T * unpred_data_pos = &unpred_data_each_thread[omp_get_thread_num()][0];
		//print out each thread process which block
		//printf("Thread %d process block %d, start_row = %d, end_row = %d, start_col = %d, end_col = %d\n", omp_get_thread_num(), block_id, start_row, end_row, start_col, end_col);
		for(int i=start_row; i<end_row; ++i){
			if (std::find(dividing_rows.begin(), dividing_rows.end(), i) != dividing_rows.end()) {
				continue;
			}
			for (int j = start_col; j<end_col; ++j){
				// if (is_dividing_line[i * m + j]) {
				// 	continue;
				// }
				if (std::find(dividing_cols.begin(), dividing_cols.end(), j) != dividing_cols.end()) {
					continue;
				}
				//total_processed++;
				//get eb
				//get position index
				size_t position_idx = (i * r2 + j);
				// if (position_idx == 877*3600+1799){
				// 	printf("qqqqqq\n");
				// 	printf("eb_quant_index[2*position_idx] = %d, eb_quant_index[2*position_idx+1] = %d\n", eb_quant_index[2*position_idx], eb_quant_index[2*position_idx+1]);
				// 	printf("data_quant_index[2*position_idx] = %d, data_quant_index[2*position_idx+1] = %d\n", data_quant_index[2*position_idx], data_quant_index[2*position_idx+1]);
				// }
				//get eb
				if(eb_quant_index[2*position_idx] == 0){
					U[position_idx] = *(unpred_data_pos ++);
					V[position_idx] = *(unpred_data_pos ++);
					// if(position_idx == 877*3600+1799){
					// 	printf("Unpredictable..\n");
					// 	printf("use U=%f, V=%f\n", U[position_idx], V[position_idx]);
					// }
				}
				else{
					for(int k=0; k<2; k++){
						T * cur_data_pos = (k == 0) ? U : V;
						double eb = pow(base, eb_quant_index[2*position_idx + k]) * threshold;
						//T d0 = (i && j) ? cur_data_pos[position_idx - 1 - r2] : 0;
						T d0 = ((i != 0 && j != 0) && (i - start_row > 1 && j - start_col > 1)) ? cur_data_pos[position_idx - 1 - r2] : 0;
						//T d1 = (i) ? cur_data_pos[position_idx - r2] : 0;
						T d1 = (i != 0 && i - start_row > 1) ? cur_data_pos[position_idx - r2] : 0;
						//T d2 = (j) ? cur_data_pos[position_idx - 1] : 0;
						T d2 = (j != 0 && j - start_col > 1) ? cur_data_pos[position_idx - 1] : 0;
						T pred = d1 + d2 - d0;
						cur_data_pos[position_idx] = pred + 2 * (data_quant_index[2*position_idx + k] - intv_radius) * eb;

						// if (position_idx == 877*3600+1799){ //7900187, 1100*3600+1802
						// 	printf("dec=======\n");
						// 	printf("i = %d, j = %d, k = %d, d0 = %f, d1 = %f, d2 = %f, pred = %f, eb = %f, data_quant_index = %d, decompressed = %f\n", i, j, k, d0, d1, d2, pred, eb, data_quant_index[2*position_idx + k], cur_data_pos[position_idx]);
						// 	printf("decomp U[%d] = %f, V[%d] = %f\n", position_idx, U[position_idx], position_idx, V[position_idx]);
						// }
					}
				}
			}
		}
	}


	//目前已经处理完了每个块的数据，现在要特殊处理划分线上的数据
	//优化：并行处理
	//先处理横着的线（行）
	omp_set_num_threads((t-1)*t);
	// printf("use %d threads for row processing\n", (t-1)*t);
	#pragma omp parallel for collapse(2)
	for (int i : dividing_rows){
		for (int j = -1; j < (int)dividing_cols.size(); j++){
			int thread_id = omp_get_thread_num();
			int start_col = (j == -1) ? 0 : dividing_cols[j];
			int end_col = (j == dividing_cols.size() - 1) ? m : dividing_cols[j+1];
			// printf("threadID %d get row=%d, start_col=%d,end_col=%d\n",thread_id,i,start_col,end_col);
			T * unpred_data_pos = &unpred_data_row[thread_id][0];
			//处理线上的点
			for (int c = start_col; c < end_col; ++c){
				if (std::find(dividing_cols.begin(), dividing_cols.end(), c) != dividing_cols.end()) {
					//k is a dividing point
					continue;
				}
				//total_processed++;
				size_t position_idx = (i * r2 + c);
				//get eb
				if(eb_quant_index[2*position_idx] == 0){
					U[position_idx] = *(unpred_data_pos ++);
					V[position_idx] = *(unpred_data_pos ++);
				}
				else{
					for(int k=0; k<2; k++){
						T * cur_data_pos = (k == 0) ? U : V;
						double eb = pow(base, eb_quant_index[2*position_idx + k]) * threshold;
						//T d0 = (c) ? cur_data_pos[position_idx - 1] : 0;
						T d0 = (c && (c - start_col > 1)) ? cur_data_pos[position_idx - 1] : 0;
						T pred = d0;
						cur_data_pos[position_idx] = pred + 2 * (data_quant_index[2*position_idx + k] - intv_radius) * eb;
					}
				}
			}
		}
	}

	//再处理竖着的线（列）
	omp_set_num_threads((t-1)*t);
	// printf("use %d threads for col processing\n", (t-1)*t);
	#pragma omp parallel for collapse(2) 
	for (int j : dividing_cols){
		for (int i = -1; i < (int)dividing_rows.size(); i++){
			int thread_id = omp_get_thread_num();
			int start_row = (i == -1) ? 0 : dividing_rows[i];
			int end_row = (i == dividing_rows.size() - 1) ? n : dividing_rows[i+1];
			T * unpred_data_pos = &unpred_data_col[thread_id][0];
			//处理线上的点
			for (int r = start_row; r < end_row; ++r){
				if (std::find(dividing_rows.begin(), dividing_rows.end(), r) != dividing_rows.end()) {
					//k is a dividing point
					continue;
				}
				size_t position_idx = (r * r2 + j);
				//total_processed++;
				//get eb
				if(eb_quant_index[2*position_idx] == 0){
					U[position_idx] = *(unpred_data_pos ++);
					V[position_idx] = *(unpred_data_pos ++);
				}
				else{
					for(int k=0; k<2; k++){
						T * cur_data_pos = (k == 0) ? U : V;
						double eb = pow(base, eb_quant_index[2*position_idx + k]) * threshold;
						//T d0 = (r) ? cur_data_pos[position_idx - r2] : 0;
						T d0 = (r && (r - start_row > 1)) ? cur_data_pos[position_idx - r2] : 0;
						T pred = d0;
						cur_data_pos[position_idx] = pred + 2 * (data_quant_index[2*position_idx + k] - intv_radius) * eb;
					}
				}
			}
		}
	}

	//最后处理点,串行
	T * unpred_data_pos = unpred_data_dot;
	for (int i : dividing_rows){
		for (int j : dividing_cols){
			size_t position_idx = (i * r2 + j);
			U[position_idx] = *(unpred_data_pos ++);
			V[position_idx] = *(unpred_data_pos ++);
			//total_processed++;
		}
	}
	//printf("total_processed = %ld,total_elements = %ld\n", total_processed, num_elements);

	/*
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
	*/

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
		free(bitmap);
	}

	free(eb_quant_index);
	free(data_quant_index);
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
