#include "sz_decompress_3d.hpp"
#include "sz_decompress_cp_preserve_2d.hpp"
#include "sz_decompress_block_processing.hpp"
#include <limits>
#include <unordered_set>
#include "utilsIO.h"
#include "sz3_utils.hpp"



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
	printf("decomp index_need_to_fix_size = %ld\n", index_need_to_fix_size);
	// if not 0, then read bitmap and lossless data
	void * data_dec;
	char * buff;
	if (index_need_to_fix_size > 0){
		//allocate memory for bitmap
		bitmap = (unsigned char *) malloc(num_elements * sizeof(unsigned char));
		size_t num_bytes = (num_elements % 8 == 0) ? num_elements / 8 : num_elements / 8 + 1;
		convertByteArray2IntArray_fast_1b_sz(num_elements, compressed_pos, num_bytes, bitmap);
		if(FPZIP_FLAG){
			//read outbytes for fpzip
			size_t fpzip_outbytes = 0;
			read_variable_from_src(compressed_pos, fpzip_outbytes);
			//decompress using fpzip
			buff = (char *) malloc(fpzip_outbytes);
			memcpy(buff, compressed_pos, fpzip_outbytes);
			compressed_pos += fpzip_outbytes;
			FPZ* fpz_dec = fpzip_read_from_buffer(buff);
			fpz_dec->type = FPZIP_TYPE_FLOAT;
			fpz_dec->prec = 0;
			fpz_dec->nx = index_need_to_fix_size;
			fpz_dec->ny = 2;
			fpz_dec->nz = 1;
			fpz_dec->nf = 1;
			data_dec = (fpz_dec->type == FPZIP_TYPE_FLOAT ? static_cast<void*>(new float[index_need_to_fix_size * 2]) : static_cast<void*>(new double[index_need_to_fix_size * 2]));
			size_t outbytes_dec = fpzip_read(fpz_dec, data_dec);
			lossless_data_U = (T *) data_dec;
			lossless_data_V = (T *) data_dec + index_need_to_fix_size;
			lossless_data_U_pos = lossless_data_U;
			lossless_data_V_pos = lossless_data_V;
			//print first U value
			printf("lossless_data_U[0] = %f\n", lossless_data_U[0]);
			//print first V value
			printf("lossless_data_V[0] = %f\n", lossless_data_V[0]);
			//print sum of data_dec
			double sum = 0;
			for (size_t i = 0; i < index_need_to_fix_size; i++){
				sum += lossless_data_U[i];
				sum += lossless_data_V[i];
			}
			printf("decomp sum data = %f\n", sum);
			fpzip_read_close(fpz_dec);
		}
		else{
			lossless_data_U = read_array_from_src<T>(compressed_pos, index_need_to_fix_size);
			lossless_data_V = read_array_from_src<T>(compressed_pos, index_need_to_fix_size);
			lossless_data_U_pos = lossless_data_U;
			lossless_data_V_pos = lossless_data_V;
		}

		


	}

	int num_threads = 0;
	read_variable_from_src(compressed_pos, num_threads);
	printf("decomp num_threads = %d,pos = %ld\n", num_threads, compressed_pos - compressed);
	int M = (int)floor(sqrt(num_threads));
	int K;

	// 从 M 往下找，直到找到一个能整除 n 的因子
	while (M > 0 && num_threads % M != 0) {
		M--;
	}

	// 找到因子后，K = n / M
	K = num_threads / M;

	int num_edge_x = M*K;
	int num_edge_y = M*K;
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

	//读每个row的unpred_count
	std::vector<size_t> unpredictable_count_row(num_edge_x);
	for (int i = 0; i < num_edge_x ; i++){
		read_variable_from_src(compressed_pos, unpredictable_count_row[i]);
	}
	//读每个row的unpred_data
	std::vector<std::vector<T>> unpred_data_row(num_edge_x);
	for (int i = 0; i < num_edge_x ; i++){
		T* raw_data = read_array_from_src<T>(compressed_pos,unpredictable_count_row[i]);
		unpred_data_row[i] = std::vector<T>(raw_data, raw_data + unpredictable_count_row[i]);
	}

	//print detail for each thread for row
	// for (int i = 0; i < (t-1)*t; i++){
	// 	printf("decomp thread %d unpredictable row size = %ld, maxval = %f \n", i, unpred_data_row[i].size(), (unpred_data_row[i].size() == 0) ? 0: *std::max_element(unpred_data_row[i].begin(), unpred_data_row[i].end()));
	// }


	//读每个col的unpred_count
	std::vector<size_t> unpredictable_count_col(num_edge_y);
	for (int i = 0; i < num_edge_y ; i++){
		read_variable_from_src(compressed_pos, unpredictable_count_col[i]);
	}
	//读每个col的unpred_data
	std::vector<std::vector<T>> unpred_data_col(num_edge_y);
	for (int i = 0; i < num_edge_y ; i++){
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
    int block_height = n / M;
    int block_width = m / K;

    // 处理余数情况（如果 n 或 m 不能被 t 整除）
    int remaining_rows = n % M;
    int remaining_cols = m % K;
	// 存储划分线的位置
    std::vector<int> dividing_rows;
    std::vector<int> dividing_cols;
    for (int i = 1; i < M; ++i) {
    	dividing_rows.push_back(i * block_height);
	}
	for (int j = 1; j < K; ++j) {
		dividing_cols.push_back(j * block_width);
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
	int total_blocks = M*K;
	omp_set_num_threads(num_threads);
	size_t total_processed = 0;	
	// printf("use %d threads for block processing\n", num_threads);
	#pragma omp parallel for
	for (int block_id = 0; block_id < total_blocks; ++block_id){
		int block_row = block_id / K;
		int block_col = block_id % K;

		// 计算块的起始和结束行列索引
		int start_row = block_row * block_height;
		int end_row = (block_row + 1) * block_height;
		if (block_row == K - 1) {
			end_row += remaining_rows;
		}
		int start_col = block_col * block_width;
		int end_col = (block_col + 1) * block_width;
		if (block_col == K - 1) {
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
	omp_set_num_threads(num_edge_x);
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
	omp_set_num_threads(num_edge_y);
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
	//如果是用vector+unorder_set的方法这里就不是按照顺序的了，需要额外处理
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
		if(FPZIP_FLAG){
			delete[] static_cast<float*>(data_dec);
		}
		
	}

	free(eb_quant_index);
	free(data_quant_index);
	if(!FPZIP_FLAG){
		free(lossless_data_U);
		free(lossless_data_V);
	}


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

template <class T>
void recover(T * U_pos, T * V_pos, size_t n, int * quantization, int& quant_count, size_t stride, VariableEBLinearQuantizer<T, T>& quantizer, int *& eb_quant_index_pos, int base, double threshold){
	if(n <= 1){
		return;
	}
	if(n < 5){
		// all linear
        for (size_t i = 1; i + 1 < n; i += 2) {
            T *dU = U_pos + i * stride;
            T *dV = V_pos + i * stride;
			double eb = pow(base, *eb_quant_index_pos ++) * threshold;
            *dU = quantizer.recover(interp_linear(*(dU - stride), *(dU + stride)), quantization[quant_count ++], eb);
            *dV = quantizer.recover(interp_linear(*(dV - stride), *(dV + stride)), quantization[quant_count ++], eb);
        }
        if (n % 2 == 0) {
            T *dU = U_pos + (n - 1) * stride;
            T *dV = V_pos + (n - 1) * stride;
			double eb = pow(base, *eb_quant_index_pos ++) * threshold;
            *dU = quantizer.recover(*(dU - stride), quantization[quant_count ++], eb);
            *dV = quantizer.recover(*(dV - stride), quantization[quant_count ++], eb);
        }

	}
	else{
		// cubic
	    size_t stride3x = 3 * stride;
	    size_t stride5x = 5 * stride;

        T *dU = U_pos + stride;
        T *dV = V_pos + stride;
		double eb = pow(base, *eb_quant_index_pos ++) * threshold;
        *dU = quantizer.recover(interp_quad_1(*(dU - stride), *(dU + stride), *(dU + stride3x)), quantization[quant_count ++], eb);
        *dV = quantizer.recover(interp_quad_1(*(dV - stride), *(dV + stride), *(dV + stride3x)), quantization[quant_count ++], eb);

        size_t i;
        for (i = 3; i + 3 < n; i += 2) {
            dU = U_pos + i * stride;
            dV = V_pos + i * stride;
            eb = pow(base, *eb_quant_index_pos ++) * threshold;
            *dU = quantizer.recover(interp_cubic(*(dU - stride3x), *(dU - stride), *(dU + stride), *(dU + stride3x)), quantization[quant_count ++], eb);
            *dV = quantizer.recover(interp_cubic(*(dV - stride3x), *(dV - stride), *(dV + stride), *(dV + stride3x)), quantization[quant_count ++], eb);
        }

        dU = U_pos + i * stride;
        dV = V_pos + i * stride;
        eb = pow(base, *eb_quant_index_pos ++) * threshold;
        *dU = quantizer.recover(interp_quad_2(*(dU - stride3x), *(dU - stride), *(dU + stride)), quantization[quant_count ++], eb);
        *dV = quantizer.recover(interp_quad_2(*(dV - stride3x), *(dV - stride), *(dV + stride)), quantization[quant_count ++], eb);
        if (n % 2 == 0) {
            dU = U_pos + (n - 1) * stride;
            dV = V_pos + (n - 1) * stride;
	        eb = pow(base, *eb_quant_index_pos ++) * threshold;
            *dU = quantizer.recover(*(dU - stride), quantization[quant_count ++], eb);
            *dV = quantizer.recover(*(dV - stride), quantization[quant_count ++], eb);
        }
	}
}

template <class T>
void omp_recover(T * U_pos, T * V_pos, size_t n, int * global_data_quant_index, size_t stride, VariableEBLinearQuantizer<T, T>& local_quantizer, int *eb_quant_index, T * U_start, int base, double threshold){
	ptrdiff_t global_offset;
	if(n <= 1){
		return;
	}
	if(n < 5){
		// all linear
        for (size_t i = 1; i + 1 < n; i += 2) {
            T *dU = U_pos + i * stride;
            T *dV = V_pos + i * stride;
			global_offset = dU - U_start;
			double eb = pow(base, eb_quant_index[global_offset]) * threshold;
            *dU = local_quantizer.recover(interp_linear(*(dU - stride), *(dU + stride)), global_data_quant_index[2*global_offset], eb);
            *dV = local_quantizer.recover(interp_linear(*(dV - stride), *(dV + stride)), global_data_quant_index[2*global_offset+1], eb);
        }
        if (n % 2 == 0) {
            T *dU = U_pos + (n - 1) * stride;
            T *dV = V_pos + (n - 1) * stride;
			global_offset = dU - U_start;
			double eb = pow(base, eb_quant_index[global_offset]) * threshold;
            *dU = local_quantizer.recover(*(dU - stride), global_data_quant_index[2*global_offset], eb);
            *dV = local_quantizer.recover(*(dV - stride), global_data_quant_index[2*global_offset+1], eb);
        }

	}
	else{
		// cubic
	    size_t stride3x = 3 * stride;
	    size_t stride5x = 5 * stride;

        T *dU = U_pos + stride;
        T *dV = V_pos + stride;
		global_offset = dU - U_start;
		double eb = pow(base, eb_quant_index[global_offset]) * threshold;
        *dU = local_quantizer.recover(interp_quad_1(*(dU - stride), *(dU + stride), *(dU + stride3x)), global_data_quant_index[2*global_offset], eb);
        *dV = local_quantizer.recover(interp_quad_1(*(dV - stride), *(dV + stride), *(dV + stride3x)), global_data_quant_index[2*global_offset+1], eb);

        size_t i;
        for (i = 3; i + 3 < n; i += 2) {
            dU = U_pos + i * stride;
            dV = V_pos + i * stride;
			global_offset = dU - U_start;
            eb = pow(base, eb_quant_index[global_offset]) * threshold;
            *dU = local_quantizer.recover(interp_cubic(*(dU - stride3x), *(dU - stride), *(dU + stride), *(dU + stride3x)), global_data_quant_index[2*global_offset], eb);
            *dV = local_quantizer.recover(interp_cubic(*(dV - stride3x), *(dV - stride), *(dV + stride), *(dV + stride3x)), global_data_quant_index[2*global_offset+1], eb);
        }

        dU = U_pos + i * stride;
        dV = V_pos + i * stride;
		global_offset = dU - U_start;
        eb = pow(base, eb_quant_index[global_offset]) * threshold;
        *dU = local_quantizer.recover(interp_quad_2(*(dU - stride3x), *(dU - stride), *(dU + stride)), global_data_quant_index[2*global_offset], eb);
        *dV = local_quantizer.recover(interp_quad_2(*(dV - stride3x), *(dV - stride), *(dV + stride)), global_data_quant_index[2*global_offset+1], eb);
        if (n % 2 == 0) {
            dU = U_pos + (n - 1) * stride;
            dV = V_pos + (n - 1) * stride;
			global_offset = dU - U_start;
	        eb = pow(base, eb_quant_index[global_offset]) * threshold;
            *dU = local_quantizer.recover(*(dU - stride), global_data_quant_index[2*global_offset], eb);
            *dV = local_quantizer.recover(*(dV - stride), global_data_quant_index[2*global_offset+1], eb);
        }
	}
}


template<typename T>
void
sz3_decompress_cp_preserve_2d_online_record_vertex(const unsigned char * compressed, size_t r1, size_t r2, T *& U, T *& V){
	printf("sz3...");
	if(U) free(U);
	if(V) free(V);
	size_t num_elements = r1 * r2;
	const unsigned char * compressed_pos = compressed;
	int base = 0;
	//搞出来需要无损的大小
	size_t lossless_count = 0;
	unsigned char * bitmap;
	T * lossless_data_U;
	T * lossless_data_V;
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
	int capacity = 0;
	read_variable_from_src(compressed_pos, capacity);
	int interpolation_level = (uint) ceil(log2(max(r1, r2)));
	auto quantizer = VariableEBLinearQuantizer<T, T>(capacity>>1);
	size_t remaining_length = num_elements*sizeof(T);//placeholder
	quantizer.load(compressed_pos, remaining_length);

	int * eb_quant_index = Huffman_decode_tree_and_data(2*1024, num_elements, compressed_pos);
	int * quantization = Huffman_decode_tree_and_data(2*capacity, 2*num_elements, compressed_pos);
	printf("pos = %ld\n", compressed_pos - compressed);
	U = (T *) malloc(num_elements*sizeof(T));
	V = (T *) malloc(num_elements*sizeof(T));
	T * U_pos = U;
	T * V_pos = V;
	int * eb_quant_index_pos = eb_quant_index;
	int quant_index = 0;
	double eb = pow(base, *eb_quant_index_pos ++) * threshold;
	U_pos[0] = quantizer.recover(0, quantization[quant_index ++], eb);
	V_pos[0] = quantizer.recover(0, quantization[quant_index ++], eb);
	for (uint level = interpolation_level; level > 0 && level <= interpolation_level; level--) {
		size_t stride = 1U << (level - 1);
		int n1 = (r1 - 1) / stride + 1;
		int n2 = (r2 - 1) / stride + 1;
		// std::cout << "level = " << level << ", stride = " << stride << ", n1 = " << n1 << ", n2 = " << n2 << ", quant_index_before = " << quant_index;//  << std::endl;
		// predict along r1
		for(int j=0; j<r2; j+=stride*2){
			recover(U_pos + j, V_pos + j, n1, quantization, quant_index, stride*r2, quantizer, eb_quant_index_pos, base, threshold);
		}
		// std::cout << ", quant_index_middle = " << quant_index;
		// predict along r2
		for(int i=0; i<r1; i+=stride){
			recover(U_pos + i*r2, V_pos + i*r2, n2, quantization, quant_index, stride, quantizer, eb_quant_index_pos, base, threshold);
		}
		// std::cout << ", quant_index_after = " << quant_index << std::endl;
	}

	//recover lossless data
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
	free(quantization);
}

template
void
sz3_decompress_cp_preserve_2d_online_record_vertex<float>(const unsigned char * compressed, size_t r1, size_t r2, float *& U, float *& V);

template
void
sz3_decompress_cp_preserve_2d_online_record_vertex<double>(const unsigned char * compressed, size_t r1, size_t r2, double *& U, double *& V);


template<typename T>
void
omp_sz3_decompress_cp_preserve_2d_online(const unsigned char * compressed, size_t r1, size_t r2, T *& U, T *& V){
	if(U) free(U);
	if(V) free(V);
	printf("omp-sz3-decomp\n");
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
	printf("decomp index_need_to_fix_size = %ld\n", index_need_to_fix_size);
	int base = 0;
	read_variable_from_src(compressed_pos, base);
	printf("decomp base = %d ,pos = %ld\n", base, compressed_pos - compressed);
	double threshold = 0;
	read_variable_from_src(compressed_pos, threshold);
	//printf("decomp threshold = %f, pos = %ld\n", threshold, compressed_pos - compressed);
	int intv_radius = 0;
	read_variable_from_src(compressed_pos, intv_radius);
	printf("decomp intv_radius = %d\n",intv_radius);
	//printf("decomp intv_radius = %d,pos = %ld\n", intv_radius, compressed_pos - compressed);
	const int capacity = (intv_radius << 1);

	// if not 0, then read bitmap and lossless data
	void * data_dec;
	char * buff;
	if (index_need_to_fix_size > 0){
		//allocate memory for bitmap
		bitmap = (unsigned char *) malloc(num_elements * sizeof(unsigned char));
		size_t num_bytes = (num_elements % 8 == 0) ? num_elements / 8 : num_elements / 8 + 1;
		convertByteArray2IntArray_fast_1b_sz(num_elements, compressed_pos, num_bytes, bitmap);
		if(FPZIP_FLAG){
			//read outbytes for fpzip
			size_t fpzip_outbytes = 0;
			read_variable_from_src(compressed_pos, fpzip_outbytes);
			//decompress using fpzip
			buff = (char *) malloc(fpzip_outbytes);
			memcpy(buff, compressed_pos, fpzip_outbytes);
			compressed_pos += fpzip_outbytes;
			FPZ* fpz_dec = fpzip_read_from_buffer(buff);
			fpz_dec->type = FPZIP_TYPE_FLOAT;
			fpz_dec->prec = 0;
			fpz_dec->nx = index_need_to_fix_size;
			fpz_dec->ny = 2;
			fpz_dec->nz = 1;
			fpz_dec->nf = 1;
			data_dec = (fpz_dec->type == FPZIP_TYPE_FLOAT ? static_cast<void*>(new float[index_need_to_fix_size * 2]) : static_cast<void*>(new double[index_need_to_fix_size * 2]));
			size_t outbytes_dec = fpzip_read(fpz_dec, data_dec);
			lossless_data_U = (T *) data_dec;
			lossless_data_V = (T *) data_dec + index_need_to_fix_size;
			lossless_data_U_pos = lossless_data_U;
			lossless_data_V_pos = lossless_data_V;
			//print first U value
			printf("lossless_data_U[0] = %f\n", lossless_data_U[0]);
			//print first V value
			printf("lossless_data_V[0] = %f\n", lossless_data_V[0]);
			//print sum of data_dec
			double sum = 0;
			for (size_t i = 0; i < index_need_to_fix_size; i++){
				sum += lossless_data_U[i];
				sum += lossless_data_V[i];
			}
			printf("decomp sum data = %f\n", sum);
			fpzip_read_close(fpz_dec);
		}
		else{
			lossless_data_U = read_array_from_src<T>(compressed_pos, index_need_to_fix_size);
			lossless_data_V = read_array_from_src<T>(compressed_pos, index_need_to_fix_size);
			lossless_data_U_pos = lossless_data_U;
			lossless_data_V_pos = lossless_data_V;
		}
		
		


	}

	int num_threads = 0;
	read_variable_from_src(compressed_pos, num_threads);
	printf("decomp num_threads = %d,pos = %ld\n", num_threads, compressed_pos - compressed);
	int M = (int)floor(sqrt(num_threads));
	int K;

	// 从 M 往下找，直到找到一个能整除 n 的因子
	while (M > 0 && num_threads % M != 0) {
		M--;
	}

	// 找到因子后，K = n / M
	K = num_threads / M;

	int num_edge_x = M*K;
	int num_edge_y = M*K;
	//读每个block的unpred_count，如果quantier就不需要
	// std::vector<size_t> unpredictable_count(num_threads);
	// for (int i = 0; i < num_threads; i++){
	// 	//每次读取一个int
	// 	read_variable_from_src(compressed_pos, unpredictable_count[i]);
	// 	//printf("decomp thread %d unpredictable_count = %d\n", i, unpredictable_count[i]);
	// }

	// 读每个block的unpred_data
	// std::vector<std::vector<T>> unpred_data_each_thread(num_threads);
	std::vector<VariableEBLinearQuantizer<T, T>> local_quant_vec(num_threads, VariableEBLinearQuantizer<T, T>(capacity/2)); //use quantizer
	for (int i = 0; i < num_threads; i++){
		// T* raw_data  = read_array_from_src<T>(compressed_pos, unpredictable_count[i]);
		// unpred_data_each_thread[i] = std::vector<T>(raw_data, raw_data + unpredictable_count[i]);
		auto quantizer = VariableEBLinearQuantizer<T, T>(capacity>>1);
		size_t remaining_length = num_elements*sizeof(T);//placeholder
		// printf("add: %ld\n",compressed_pos);
		local_quant_vec[i].load(compressed_pos,remaining_length);
	}
	//print the detail for each thread
	// for (int i = 0; i < num_threads; i++){
	// 	printf("decomp thread %d unpredictable data size = %ld, maxval = %f \n", i, unpred_data_each_thread[i].size(), *std::max_element(unpred_data_each_thread[i].begin(), unpred_data_each_thread[i].end()));
	// }

	//读每个row的unpred_count
	std::vector<size_t> unpredictable_count_row(num_edge_x);
	for (int i = 0; i < num_edge_x ; i++){
		read_variable_from_src(compressed_pos, unpredictable_count_row[i]);
	}
	//读每个row的unpred_data
	std::vector<std::vector<T>> unpred_data_row(num_edge_x);
	for (int i = 0; i < num_edge_x ; i++){
		T* raw_data = read_array_from_src<T>(compressed_pos,unpredictable_count_row[i]);
		unpred_data_row[i] = std::vector<T>(raw_data, raw_data + unpredictable_count_row[i]);
	}

	//print detail for each thread for row
	// for (int i = 0; i < (t-1)*t; i++){
	// 	printf("decomp thread %d unpredictable row size = %ld, maxval = %f \n", i, unpred_data_row[i].size(), (unpred_data_row[i].size() == 0) ? 0: *std::max_element(unpred_data_row[i].begin(), unpred_data_row[i].end()));
	// }


	//读每个col的unpred_count
	std::vector<size_t> unpredictable_count_col(num_edge_y);
	for (int i = 0; i < num_edge_y ; i++){
		read_variable_from_src(compressed_pos, unpredictable_count_col[i]);
	}
	//读每个col的unpred_data
	std::vector<std::vector<T>> unpred_data_col(num_edge_y);
	for (int i = 0; i < num_edge_y ; i++){
		T* raw_data = read_array_from_src<T>(compressed_pos,unpredictable_count_col[i]);
		unpred_data_col[i] = std::vector<T>(raw_data, raw_data + unpredictable_count_col[i]);
	}

	//读串行的dot的unpred_count
	size_t dot_size;
	read_variable_from_src(compressed_pos, dot_size);
	T* unpred_data_dot = read_array_from_src<T>(compressed_pos, dot_size);


	

	// const T * unpred_data_pos = (T *) compressed_pos;
	// compressed_pos += unpred_data_count*sizeof(T);

	//check num of unpredictable data
	
	// for (int i = 0; i < num_threads; i++){
	// 	printf("decomp thread %d unpredictable data size = %ld, maxval = %f \n", i, unpred_data_each_thread[i].size(), *std::max_element(unpred_data_each_thread[i].begin(), unpred_data_each_thread[i].end()));
	// }
	//printf("decomp pos after write all upredict = %ld\n", compressed_pos - compressed);



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
	int * eb_quant_index = (int *) malloc(num_elements*sizeof(int));

	#pragma omp parallel for num_threads(num_threads)
	for (int i = 0; i < num_threads; i++){
		size_t start_pos = i*num_elements/num_threads;
		size_t end_pos = (i ==num_threads - 1) ? num_elements : (i+1)*num_elements/num_threads;
		const unsigned char * local_compressed_pos = compressed_chunk_eb_start[i];
		size_t local_num_elements = (end_pos - start_pos);
		int * local_eb_quant_index = Huffman_decode_tree_and_data(2*1024, local_num_elements, local_compressed_pos);
		std::copy(local_eb_quant_index, local_eb_quant_index + local_num_elements, eb_quant_index + start_pos);
		free(local_eb_quant_index);
	}


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
	printf("decomp parallel data max = %d\n", *std::max_element(data_quant_index, data_quant_index + 2*num_elements));
	printf("decomp parallel data min = %d\n", *std::min_element(data_quant_index, data_quant_index + 2*num_elements));



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
    int block_height = n / M;
    int block_width = m / K;

    // 处理余数情况（如果 n 或 m 不能被 t 整除）
    int remaining_rows = n % M;
    int remaining_cols = m % K;
	// 存储划分线的位置
    std::vector<int> dividing_rows;
    std::vector<int> dividing_cols;
    for (int i = 1; i < M; ++i) {
    	dividing_rows.push_back(i * block_height);
	}
	for (int j = 1; j < K; ++j) {
		dividing_cols.push_back(j * block_width);
	}
	// 创建一个一维标记数组，标记哪些数据点位于划分线上
	std::vector<bool> is_dividing_row(n, false);
	std::vector<bool> is_dividing_col(m, false);
	for (int i = 0; i < dividing_rows.size(); ++i) {
		is_dividing_row[dividing_rows[i]] = true;
	}
	for (int i = 0; i < dividing_cols.size(); ++i) {
		is_dividing_col[dividing_cols[i]] = true;
	}
	int total_blocks = M*K;
	omp_set_num_threads(num_threads);
	size_t total_processed = 0;	
	// printf("use %d threads for block processing\n", num_threads);
	#pragma omp parallel for
	for (int block_id = 0; block_id < total_blocks; ++block_id){
		int block_row = block_id / K;
		int block_col = block_id % K;

		// 计算块的起始和结束行列索引
		int start_row = block_row * block_height;
		int end_row = (block_row + 1) * block_height;
		if (block_row == K - 1) {
			end_row += remaining_rows;
		}
		int start_col = block_col * block_width;
		int end_col = (block_col + 1) * block_width;
		if (block_col == K - 1) {
			end_col += remaining_cols;
		}

		int true_r1 = 0, true_r2 = 0;
		for(int i=start_row; i<end_row; ++i){
			if (is_dividing_row[i]) {
				continue;
			}
			true_r1 ++;
		}
		for(int j=start_col; j<end_col; ++j){
			if (is_dividing_col[j]) {
				continue;
			}
			true_r2 ++;
		}
		int true_start_row = start_row;
		int true_end_row = start_row + true_r1;
		int true_start_col = start_col;
		int true_end_col = start_col + true_r2;
		
		// 跳过划分线上的数据点
		true_start_col = (std::find(dividing_cols.begin(), dividing_cols.end(), start_col) != dividing_cols.end()) ? start_col + 1 : start_col;
		true_start_row = (std::find(dividing_rows.begin(), dividing_rows.end(), start_row) != dividing_rows.end()) ? start_row + 1 : start_row;
		printf("threadID = %d, true_start_row = %d, true_end_row = %d, true_start_col = %d, true_end_col = %d\n", omp_get_thread_num(),true_start_row,true_end_row,true_start_col,true_end_col);

		int interpolation_level = (uint) ceil(log2(max(true_r1, true_r2)));
		//auto quantizer = VariableEBLinearQuantizer<T, T>(capacity>>1);
		auto quantizer = local_quant_vec[block_id];
		printf("decomp: threadid = %d, num_unpred for this quantizer = %d\n", block_id,quantizer.num_unpred());
		size_t first_data_pos = true_start_row * r2 + true_start_col;
		if(first_data_pos == 4323600){
			printf("hsq thread %d\n",block_id);
			exit(0);
		}
		//process first data
		double eb = pow(base,eb_quant_index[first_data_pos]) * threshold;
		U[first_data_pos] = quantizer.recover(0,data_quant_index[2 * first_data_pos],eb);
		V[first_data_pos] = quantizer.recover(0,data_quant_index[2 * first_data_pos + 1],eb);

		for (uint level = interpolation_level; level > 0 && level <= interpolation_level; level--) {
			size_t stride = 1U << (level - 1); // stride = 2^(level-1)
			int n1 = (true_r1 - 1) / stride + 1; // number of blocks along r1, each block size is stride
			int n2 = (true_r2 - 1) / stride + 1;
			// predict along r1
			for(int j=0; j<true_r2; j+=stride*2){
				//do something
				omp_recover(U_pos + first_data_pos + j, V_pos + first_data_pos + j, n1, data_quant_index,stride*r2,quantizer,eb_quant_index,U_pos,base,threshold);
			}
			for(int i=0; i<true_r1; i+=stride){
				omp_recover(U_pos + first_data_pos + i*r2, V_pos + first_data_pos + i*r2,n2,data_quant_index,stride,quantizer,eb_quant_index,U_pos,base,threshold);
			}
				
				
		}
	}


	//目前已经处理完了每个块的数据，现在要特殊处理划分线上的数据
	//优化：并行处理
	//先处理横着的线（行）
	omp_set_num_threads(num_edge_x);
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
				if(eb_quant_index[position_idx] == 0){
					U[position_idx] = *(unpred_data_pos ++);
					V[position_idx] = *(unpred_data_pos ++);
				}
				else{
					for(int k=0; k<2; k++){
						T * cur_data_pos = (k == 0) ? U : V;
						double eb = pow(base, eb_quant_index[position_idx]) * threshold;
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
	omp_set_num_threads(num_edge_y);
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
				if(eb_quant_index[position_idx] == 0){
					U[position_idx] = *(unpred_data_pos ++);
					V[position_idx] = *(unpred_data_pos ++);
				}
				else{
					for(int k=0; k<2; k++){
						T * cur_data_pos = (k == 0) ? U : V;
						double eb = pow(base, eb_quant_index[position_idx]) * threshold;
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


	//最后在根据bitmap更新
	//如果是用vector+unorder_set的方法这里就不是按照顺序的了，需要额外处理
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
		if(FPZIP_FLAG){
			delete[] static_cast<float*>(data_dec);
		}
		
	}

	free(eb_quant_index);
	free(data_quant_index);
	if(!FPZIP_FLAG){
		free(lossless_data_U);
		free(lossless_data_V);
	}

}

template
void
omp_sz3_decompress_cp_preserve_2d_online<float>(const unsigned char * compressed, size_t r1, size_t r2, float *& U, float *& V);

//sos-method

template<typename T, typename T_fp>
static void 
convert_to_floating_point(const T_fp * U_fp, const T_fp * V_fp, size_t num_elements, T * U, T * V, int64_t vector_field_scaling_factor){
	for(int i=0; i<num_elements; i++){
		U[i] = U_fp[i] * (T)1.0 / vector_field_scaling_factor;
		V[i] = V_fp[i] * (T)1.0 / vector_field_scaling_factor;
	}
}

//cpsz-sos
template<typename T_data>
void
sz_decompress_cp_preserve_sos_2d_online_fp(const unsigned char * compressed, size_t r1, size_t r2, T_data *& U, T_data *& V){
	if(U) free(U);
	if(V) free(V);
	using T = int64_t;
	size_t num_elements = r1 * r2;
	const unsigned char * compressed_pos = compressed;
	T vector_field_scaling_factor = 0;
	read_variable_from_src(compressed_pos, vector_field_scaling_factor);
	int base = 0;
	read_variable_from_src(compressed_pos, base);
	printf("base = %d\n", base);
	T threshold = 0;
	read_variable_from_src(compressed_pos, threshold);
	int intv_radius = 0;
	read_variable_from_src(compressed_pos, intv_radius);
	const int capacity = (intv_radius << 1);
	size_t unpred_data_count = 0;
	read_variable_from_src(compressed_pos, unpred_data_count);
	const T_data * unpred_data = (T_data *) compressed_pos;
	const T_data * unpred_data_pos = unpred_data;
	compressed_pos += unpred_data_count*sizeof(T_data);
	size_t eb_quant_num = 0;
	read_variable_from_src(compressed_pos, eb_quant_num);
	int * eb_quant_index = Huffman_decode_tree_and_data(2*1024, eb_quant_num, compressed_pos);
	size_t data_quant_num = 0;
	read_variable_from_src(compressed_pos, data_quant_num);
	int * data_quant_index = Huffman_decode_tree_and_data(2*capacity, data_quant_num, compressed_pos);
	printf("pos = %ld\n", compressed_pos - compressed);
	T * U_fp = (T *) malloc(num_elements*sizeof(T));
	T * V_fp = (T *) malloc(num_elements*sizeof(T));
	T * U_pos = U_fp;
	T * V_pos = V_fp;
	int * eb_quant_index_pos = eb_quant_index;
	int * data_quant_index_pos = data_quant_index;
	std::vector<int> unpred_data_indices;
	for(int i=0; i<r1; i++){
		for(int j=0; j<r2; j++){
			// get eb
			if(*eb_quant_index_pos == 0){
				size_t offset = U_pos - U_fp;
				unpred_data_indices.push_back(offset);
				*U_pos = *(unpred_data_pos ++) * vector_field_scaling_factor;
				*V_pos = *(unpred_data_pos ++) * vector_field_scaling_factor;
				eb_quant_index_pos ++;
			}
			else{
				T eb = pow(base, *eb_quant_index_pos ++) * threshold;
				for(int k=0; k<2; k++){
					T * cur_data_pos = (k == 0) ? U_pos : V_pos;					
					// double eb = *(eb_quant_index_pos ++) * 1e-3;
					T d0 = (i && j) ? cur_data_pos[-1 - r2] : 0;
					T d1 = (i) ? cur_data_pos[-r2] : 0;
					T d2 = (j) ? cur_data_pos[-1] : 0;
					T pred = d1 + d2 - d0;
					*cur_data_pos = pred + 2 * (data_quant_index_pos[k] - intv_radius) * eb;
				}
				data_quant_index_pos += 2;
			}
			U_pos ++;
			V_pos ++;
		}
	}
	free(eb_quant_index);
	free(data_quant_index);
	U = (T_data *) malloc(num_elements*sizeof(T_data));
	V = (T_data *) malloc(num_elements*sizeof(T_data));
	convert_to_floating_point(U_fp, V_fp, num_elements, U, V, vector_field_scaling_factor);
	unpred_data_pos = unpred_data;
	for(const auto& index:unpred_data_indices){
		U[index] = *(unpred_data_pos++);
		V[index] = *(unpred_data_pos++);
	}
	free(U_fp);
	free(V_fp);
}

template
void
sz_decompress_cp_preserve_sos_2d_online_fp<float>(const unsigned char * compressed, size_t r1, size_t r2, float *& U, float *& V);

template
void
sz_decompress_cp_preserve_sos_2d_online_fp<double>(const unsigned char * compressed, size_t r1, size_t r2, double *& U, double *& V);
