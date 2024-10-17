#ifndef _sz_compress_cp_preserve_3d_hpp
#define _sz_compress_cp_preserve_3d_hpp

#include <cstddef>
#include <set>
#include <string>

template<typename T>
unsigned char *
sz_compress_cp_preserve_3d_offline_log(const T * U, const T * V, const T * W, size_t r1, size_t r2, size_t r3, size_t& compressed_size, bool transpose=false, double max_pwr_eb=0.1);

template<typename T>
unsigned char *
sz_compress_cp_preserve_3d_online_log(const T * U, const T * V, const T * W, size_t r1, size_t r2, size_t r3, size_t& compressed_size, bool transpose=false, double max_pwr_eb=0.1);

template<typename T>
unsigned char *
sz_compress_cp_preserve_3d_unstructured(int n, const T * points, const T * data, int m, const int * tets_ind, size_t& compressed_size, double max_pwr_eb=0.1);

template<typename T>
unsigned char *
sz_compress_cp_preserve_3d_record_vertex(const T * U, const T * V, const T * W, size_t r1, size_t r2, size_t r3, size_t& compressed_size, bool transpose=false, double max_pwr_eb=0.1, const std::set<size_t> &index_need_to_fix = {});

template<typename T>
unsigned char *
sz_compress_cp_preserve_3d_online_abs_record_vertex(const T * U, const T * V, const T * W, size_t r1, size_t r2, size_t r3, size_t& compressed_size, double max_abs_eb=0.1,const std::set<size_t> &index_need_to_fix = {});

// template<typename T>
// unsigned char *
// omp_sz_compress_cp_preserve_3d_online_record_vertex(
//  const T * U, const T * V, const T * W, size_t r1, size_t r2, size_t r3,
// size_t& compressed_size, double max_abs_eb=0.1,const std::set<size_t> &index_need_to_fix = {}, int threads = 64, T *& decompressed_U_ptr= NULL, T *& decompressed_V_ptr=NULL, T *& decompressed_W_ptr=NULL, std::string eb_type = "rel");

template <typename T>
unsigned char * omp_sz_compress_cp_preserve_3d_online_abs_record_vertex(
    const T * U, const T * V, const T * W, size_t r1, size_t r2, size_t r3, 
    size_t& compressed_size, double max_eb, const std::set<size_t>& index_need_to_lossless, 
    int n_threads, T* &decompressed_U_ptr, T* &decompressed_V_ptr, T* &decompressed_W_ptr, std::vector<bool> &cp_exist_vec) ;

#endif