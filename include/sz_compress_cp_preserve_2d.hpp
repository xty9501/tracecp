#ifndef _sz_compress_cp_preserve_2d_hpp
#define _sz_compress_cp_preserve_2d_hpp
#define WRITE_OUT_EB 0
#include <cstddef>
#include <unordered_map>
#include <set>
#include "cp.hpp"

// #define DEFAULT_EB 0.1

template<typename T>
unsigned char *
sz_compress_cp_preserve_2d_offline(const T * U, const T * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose=false, double max_pwr_eb=0.1);

template<typename T>
unsigned char *
sz_compress_cp_preserve_2d_offline_log(const T * U, const T * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose=false, double max_pwr_eb=0.1);

template<typename T>
unsigned char *
sz_compress_cp_preserve_2d_online(const T * U, const T * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose=false, double max_pwr_eb=0.1,const std::unordered_map <size_t, size_t> &lossless_index = {},const std::unordered_map <size_t, size_t> &index_need_to_fix = {});

// template<typename T>
// unsigned char *
// sz_compress_cp_preserve_sos_2d_online(const T * U, const T * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose=false, double max_pwr_eb=0.1);

template<typename T>
unsigned char *
sz_compress_cp_preserve_sos_2d_online_fp(const T * U, const T * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose=false, double max_pwr_eb=0.1);

template<typename T_data>
unsigned char *
compress_lossless_index(const T_data * U, const T_data * V, const std::unordered_map <size_t, size_t> &lossless_index, size_t r1, size_t r2, size_t& compressed_size, bool transpose, double max_pwr_eb);

template<typename T>
unsigned char *
sz_compress_cp_preserve_2d_fix(const T * U, const T * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose=false, double max_pwr_eb = 0.01, double modified_eb = 0,const std::set<size_t> &index_need_to_fix = {});

template<typename T>
unsigned char *
sz_compress_cp_preserve_2d_record_vertex(const T * U, const T * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose=false, double max_pwr_eb = 0.01,const std::set<size_t> &index_need_to_fix = {});

template<typename T>
unsigned char *
sz_compress_cp_preserve_2d_st2_fix(const T * U, const T * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose=false, double max_pwr_eb = 0.01, double modified_eb = 0 ,std::unordered_map<size_t, critical_point_t> & critical_points = {},const std::set<size_t> &index_need_to_fix = {});

template<typename T>
unsigned char *
sz_compress_cp_preserve_2d_online_abs_record_vertex(const T * U, const T * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose=false, double max_pwr_eb = 0.01,const std::set<size_t> &index_need_to_fix = {});

template<typename T>
unsigned char *
omp_sz_compress_cp_preserve_2d_record_vertex(const T * U, const T * V, size_t r1, size_t r2, size_t& compressed_size, bool transpose=false, double max_pwr_eb = 0.01,const std::set<size_t> &index_need_to_fix = {}, int threads = 64, T *&decompressed_U= NULL, T *&decompressed_V=NULL,std::string eb_type = "rel");
#endif