#ifndef _sz_decompress_cp_preserve_2d_offline_hpp
#define _sz_decompress_cp_preserve_2d_offline_hpp

#include "sz_decompression_utils.hpp"
#include "sz_def.hpp"
#include "sz_prediction.hpp"
#include <vector>
#include "cp.hpp"
#define FPZIP_FLAG 0

// template<typename T>
// void
// sz_decompress_cp_preserve_2d_offline(const unsigned char * compressed, size_t r1, size_t r2, T *& U, T *& V);
// template<typename T>
// void
// sz_decompress_cp_preserve_2d_offline_log(const unsigned char * compressed, size_t r1, size_t r2, T *& U, T *& V);
// template<typename T>
// void
// sz_decompress_cp_preserve_2d_online_fp(const unsigned char * compressed, size_t r1, size_t r2, T *& U, T *& V);
// template<typename T>
// void
// sz_decompress_cp_preserve_2d_online_log(const unsigned char * compressed, size_t r1, size_t r2, T *& U, T *& V);

template<typename T>
void
sz_decompress_cp_preserve_2d_online(const unsigned char * compressed, size_t r1, size_t r2, T *& U, T *& V);

template<typename T>
void
sz_decompress_cp_preserve_2d_online_record_vertex(const unsigned char * compressed, size_t r1, size_t r2, T *& U, T *& V);

template<typename T>
void
omp_sz_decompress_cp_preserve_2d_online(const unsigned char * compressed, size_t r1, size_t r2, T *& U, T *& V);

template<typename T>
void
sz3_decompress_cp_preserve_2d_online_record_vertex(const unsigned char * compressed, size_t r1, size_t r2, T *& U, T *& V);

template<typename T>
void
omp_sz3_decompress_cp_preserve_2d_online(const unsigned char * compressed, size_t r1, size_t r2, T *& U, T *& V);

//cpsz-sos option 0
template<typename T>
void
sz_decompress_cp_preserve_sos_2d_online_fp(const unsigned char * compressed, size_t r1, size_t r2, T *& U, T *& V);
#endif