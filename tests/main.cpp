#include "utils.hpp"
#include <interp.h>
#include <utilsIO.h>
#include <ftk/numeric/print.hh>
#include <ftk/numeric/cross_product.hh>
#include <ftk/numeric/vector_norm.hh>
#include <ftk/numeric/linear_interpolation.hh>
#include <ftk/numeric/bilinear_interpolation.hh>
#include <ftk/numeric/inverse_linear_interpolation_solver.hh>
#include <ftk/numeric/inverse_bilinear_interpolation_solver.hh>
#include <ftk/numeric/gradient.hh>
#include <ftk/numeric/matrix_multiplication.hh>
#include <ftk/numeric/matrix_inverse.hh>
#include <ftk/numeric/clamp.hh>
#include <ftk/numeric/eigen_solver2.hh>
#include <ftk/algorithms/cca.hh>
#include <ftk/geometry/cc2curves.hh>
#include <ftk/geometry/curve2tube.hh>
#include "ftk/ndarray.hh"
#include "ftk/numeric/critical_point_type.hh"
#include "ftk/numeric/critical_point_test.hh"
#include "cp.hpp"
#include <chrono>
#include "advect.hpp"
#include <math.h>


// #include <hypermesh/ndarray.hh>
// #include <hypermesh/regular_simplex_mesh.hh>
#include <mutex>
// #include <hypermesh/ndarray.hh>
// #include <hypermesh/regular_simplex_mesh.hh>

#include <ftk/ndarray/ndarray_base.hh>
#include <unordered_map>
#include <queue>
#include <fstream>
#include <utility>
#include <set>


#include "sz_cp_preserve_utils.hpp"
#include "sz_compress_cp_preserve_2d.hpp"
#include "sz_decompress_cp_preserve_2d.hpp"
#include "sz_lossless.hpp"
#include <iostream> 
#include <algorithm>

double euclideanDistance(const std::array<double, 2>& p1, const std::array<double, 2>& p2) {
    return std::sqrt(std::pow(p1[0] - p2[0], 2) + std::pow(p1[1] - p2[1], 2));
}
double calculateEDR2D(const std::vector<std::array<double, 2>>& seq1, const std::vector<std::array<double, 2>>& seq2, double threshold) {
    size_t len1 = seq1.size();
    size_t len2 = seq2.size();

    // Create a 2D vector to store distances, initialize with zeros
    std::vector<std::vector<double>> dp(len1 + 1, std::vector<double>(len2 + 1, 0));

    // Initialize the first column and first row
    for (size_t i = 0; i <= len1; ++i) {
        dp[i][0] = i;
    }
    for (size_t j = 0; j <= len2; ++j) {
        dp[0][j] = j;
    }

    // Fill the dp matrix
    for (size_t i = 1; i <= len1; ++i) {
        for (size_t j = 1; j <= len2; ++j) {
            // Use Euclidean distance and compare with threshold
            double cost = euclideanDistance(seq1[i - 1], seq2[j - 1]) <= threshold ? 0 : 1;
            dp[i][j] = std::min({
                dp[i - 1][j] + 1,    // Deletion
                dp[i][j - 1] + 1,    // Insertion
                dp[i - 1][j - 1] + cost  // Substitution or match
            });
        }
    }

    return dp[len1][len2];
}

// void refill_gradient(int id,const int DH,const int DW, const float* grad_tmp, ftk::ndarray<float>& grad){
//   const float * grad_tmp_pos = grad_tmp;
//   for (int i = 0; i < DH; i ++) {
//     for (int j = 0; j < DW; j ++) {
//       grad(id, j, i) = *(grad_tmp_pos ++);
//     }
//   }
// }

typedef struct traj_config{
  double h;
  double eps;
  int max_length; 
} traj_config;

traj_config t_config = {0.05, 0.01, 2000};

template<typename Type>
void verify(Type * ori_data, Type * data, size_t num_elements){
    size_t i = 0;
    Type Max = 0, Min = 0, diffMax = 0;
    Max = ori_data[0];
    Min = ori_data[0];
    diffMax = fabs(data[0] - ori_data[0]);
    size_t k = 0;
    double sum1 = 0, sum2 = 0;
    for (i = 0; i < num_elements; i++){
        sum1 += ori_data[i];
        sum2 += data[i];
    }
    double mean1 = sum1/num_elements;
    double mean2 = sum2/num_elements;

    double sum3 = 0, sum4 = 0;
    double sum = 0, prodSum = 0, relerr = 0;

    double maxpw_relerr = 0; 
    for (i = 0; i < num_elements; i++){
        if (Max < ori_data[i]) Max = ori_data[i];
        if (Min > ori_data[i]) Min = ori_data[i];
        
        Type err = fabs(data[i] - ori_data[i]);
        if(ori_data[i]!=0 && fabs(ori_data[i])>1)
        {
            relerr = err/fabs(ori_data[i]);
            if(maxpw_relerr<relerr)
                maxpw_relerr = relerr;
        }

        if (diffMax < err)
            diffMax = err;
        prodSum += (ori_data[i]-mean1)*(data[i]-mean2);
        sum3 += (ori_data[i] - mean1)*(ori_data[i]-mean1);
        sum4 += (data[i] - mean2)*(data[i]-mean2);
        sum += err*err; 
    }
    double std1 = sqrt(sum3/num_elements);
    double std2 = sqrt(sum4/num_elements);
    double ee = prodSum/num_elements;
    double acEff = ee/std1/std2;

    double mse = sum/num_elements;
    double range = Max - Min;
    double psnr = 20*log10(range)-10*log10(mse);
    double nrmse = sqrt(mse)/range;

    printf ("Min=%.20G, Max=%.20G, range=%.20G\n", Min, Max, range);
    printf ("Max absolute error = %.10f\n", diffMax);
    printf ("Max relative error = %f\n", diffMax/(Max-Min));
    printf ("Max pw relative error = %f\n", maxpw_relerr);
    printf ("PSNR = %f, NRMSE= %.20G\n", psnr,nrmse);
    printf ("acEff=%f\n", acEff);   
}

namespace std {
    template<> struct hash<array<double, 2>> {
        size_t operator()(const array<double, 2>& arr) const noexcept {
            // 可以使用任何你认为合适的哈希组合方式
            // 这里仅为示例，实际应用时可能需要根据实际需求调整
            size_t h1 = hash<double>()(arr[0]);
            size_t h2 = hash<double>()(arr[1]);
            return h1 ^ (h2 << 1); // 简单的组合哈希值
        }
    };
}


double vector_field_resolution = std::numeric_limits<double>::max();
uint64_t vector_field_scaling_factor = 1;
// int DW = 128, DH = 128;// the dimensionality of the data is DW*DH

ftk::simplicial_regular_mesh m(2);
std::mutex mutex;

size_t global_count = 0;


// typedef struct critical_point_t{
//   double x[2];
//   double eig_vec[2][2];
//   double V[3][2];
//   double X[3][2];
//   double Jac[2][2];
//   std::complex<double> eig[2];
//   // double mu[3];
//   int type;
//   size_t simplex_id;
//   critical_point_t(double* x_, double eig_[2], double eig_v[2][2], double Jac_[2][2], double V_[3][2],double X_[3][2], int t_, size_t simplex_id_){
//     x[0] = x_[0];
//     x[1] = x_[1];
//     eig[0] = eig_[0];
//     eig[1] = eig_[1];
//     eig_vec[0][0] = eig_v[0][0];
//     eig_vec[0][1] = eig_v[0][1];
//     eig_vec[1][0] = eig_v[1][0];
//     eig_vec[1][1] = eig_v[1][1];
//     for (int i = 0; i < 3; i++) {
//       for (int j = 0; j < 2; j++) {
//         V[i][j] = V_[i][j];
//       }
//     }
//     for (int i = 0; i < 3; i++) {
//       for (int j = 0; j < 2; j++) {
//         X[i][j] = X_[i][j];
//       }
//     }
//     for (int i = 0; i < 2; i++) {
//       for (int j = 0; j < 2; j++) {
//         Jac[i][j] = Jac_[i][j];
//       }
//     }
//     // for (int i = 0; i < 3; i++) {
//     //   mu[i] = mu_[i];
//     // }
//     type = t_;
//     simplex_id = simplex_id_;
//   }
//   critical_point_t(){}
// }critical_point_t;



// std::unordered_map<int, critical_point_t> critical_points;
 
// #define DEFAULT_EB 1
// #define SINGULAR 0
// #define ATTRACTING 1 // 2 real negative eigenvalues
// #define REPELLING 2 // 2 real positive eigenvalues
// #define SADDLE 3// 1 real negative and 1 real positive
// #define ATTRACTING_FOCUS 4 // complex with negative real
// #define REPELLING_FOCUS 5 // complex with positive real
// #define CENTER 6 // complex with 0 real

// std::string get_critical_point_type_string(int type){
//   switch(type){
//     case 0:
//       return "SINGULAR";
//     case 1:
//       return "ATTRACTING";
//     case 2:
//       return "REPELLING";
//     case 3:
//       return "SADDLE";
//     case 4:
//       return "ATTRACTING_FOCUS";
//     case 5:
//       return "REPELLING_FOCUS";
//     case 6:
//       return "CENTER";
//     default:
//       return "INVALID";
//   }
// }





// template<typename T>
// static void 
// check_simplex_seq_saddle(const T v[3][2], const double X[3][2], const int indices[3], int i, int j, int simplex_id, std::unordered_map<int, critical_point_t>& critical_points){
//   int sos = 0;
//   double mu[3]; // check intersection
//   double cond;
//   if (sos == 1){
//     // for(int i=0; i<3; i++){ //skip if any of the vertex is 0 //
//     // if((v[i][0] == 0) && (v[i][1] == 0)){ //
//     //   return; //
//     //   } //
//     // } //
//     // // robust critical point test
//     // bool succ = ftk::robust_critical_point_in_simplex2(vf, indices);
//     // if (!succ) return;
//     // bool succ2 = ftk::inverse_lerp_s2v2(v, mu, &cond);
//     // if (!succ2) ftk::clamp_barycentric<3>(mu);
//     printf("sos is not supported\n");
//   }
//   else{
//     for(int i=0; i<3; i++){ //skip if any of the vertex is 0 //
//     if((v[i][0] == 0) && (v[i][1] == 0)){ //
//       return; //
//       } //
//     } //
//     bool succ2 = ftk::inverse_lerp_s2v2(v, mu, &cond);
//     if (!succ2) return;
//   }
//   critical_point_t cp;
//   double eig_vec[2][2]={0};
//   double eig_r[2];
//   std::complex<double> eig[2];
//   double x[2]; // position
//   ftk::lerp_s2v2(X, mu, x);
//   cp.x[0] = j + x[0]; cp.x[1] = i + x[1];
//   double J[2][2]; // jacobian
//   ftk::jacobian_2dsimplex2(X, v, J);  
//   int cp_type = 0;
//   double delta = ftk::solve_eigenvalues2x2(J, eig);
//   // if(fabs(delta) < std::numeric_limits<double>::epsilon())
//   if (delta >= 0) { // two real roots
//     if (eig[0].real() * eig[1].real() < 0) {
//       cp.eig[0] = eig[0], cp.eig[1] = eig[1];
//       cp.Jac[0][0] = J[0][0]; cp.Jac[0][1] = J[0][1];
//       cp.Jac[1][0] = J[1][0]; cp.Jac[1][1] = J[1][1];
//       cp_type = SADDLE;
//       double eig_r[2];
//       eig_r[0] = eig[0].real(), eig_r[1] = eig[1].real();
//       ftk::solve_eigenvectors2x2(J, 2, eig_r, eig_vec);
//     } else if (eig[0].real() < 0) {
//       cp_type = ATTRACTING;
//     }
//     else if (eig[0].real() > 0){
//       cp_type = REPELLING;
//     }
//     else cp_type = SINGULAR;
//   } else { // two conjugate roots
//     if (eig[0].real() < 0) {
//       cp_type = ATTRACTING_FOCUS;
//     } else if (eig[0].real() > 0) {
//       cp_type = REPELLING_FOCUS;
//     } else 
//       cp_type = CENTER;
//   }
//   // critical_point_t cp(x, eig_vec,v,X, cp_type);
//   //ftk::transpose2x2(J);
//   cp.eig_vec[0][0] = eig_vec[0][0]; cp.eig_vec[0][1] = eig_vec[0][1];
//   cp.eig_vec[1][0] = eig_vec[1][0]; cp.eig_vec[1][1] = eig_vec[1][1];
//   // 这里eig_vec由（x,y）变成了（-y,x）
//   // cp.eig_vec[0][0] = -eig_vec[0][1]; cp.eig_vec[0][1] = eig_vec[0][0];
//   // cp.eig_vec[1][0] = -eig_vec[1][1]; cp.eig_vec[1][1] = eig_vec[1][0];
//   cp.V[0][0] = v[0][0]; cp.V[0][1] = v[0][1];
//   cp.V[1][0] = v[1][0]; cp.V[1][1] = v[1][1];
//   cp.V[2][0] = v[2][0]; cp.V[2][1] = v[2][1];
//   cp.X[0][0] = X[0][0]; cp.X[0][1] = X[0][1];
//   cp.X[1][0] = X[1][0]; cp.X[1][1] = X[1][1];
//   cp.X[2][0] = X[2][0]; cp.X[2][1] = X[2][1];
//   cp.type = cp_type;
//   cp.simplex_id = simplex_id;
//   // cp.v = mu[0]*v[0][0] + mu[1]*v[1][0] + mu[2]*v[2][0];
//   // cp.u = mu[0]*v[0][1] + mu[1]*v[1][1] + mu[2]*v[2][1];
//   critical_points[simplex_id] = cp;
// }



// template<typename T>
// std::unordered_map<int, critical_point_t>
// compute_critical_points(const T * U, const T * V, int r1, int r2){
//   size_t num_elements = r1*r2;

//   int indices[3] = {0};
// 	double X1[3][2] = {
// 		{0, 0},
// 		{0, 1},
// 		{1, 1}
// 	};
// 	double X2[3][2] = {
// 		{0, 0},
// 		{1, 0},
// 		{1, 1}
// 	};
//   double v[3][2] = {0};
//   std::unordered_map<int, critical_point_t> critical_points;
// 	for(int i=0; i<r1-1; i++){ //坑
//     // if(i%100==0) std::cout << i << " / " << r1-1 << std::endl;
// 		for(int j=0; j<r2-1; j++){
//       ptrdiff_t cell_offset = 2*(i * (r2-1) + j);
// 			indices[0] = i*r2 + j;
// 			indices[1] = (i+1)*r2 + j;
// 			indices[2] = (i+1)*r2 + (j+1); 
// 			// cell index 0
// 			for(int p=0; p<3; p++){
// 				v[p][0] = U[indices[p]];
// 				v[p][1] = V[indices[p]];
// 			}
      
//       check_simplex_seq_saddle(v, X1, indices, i, j, cell_offset, critical_points);
// 			// cell index 1
// 			indices[1] = i*r2 + (j+1);
// 			v[1][0] = U[indices[1]], v[1][1] = V[indices[1]];			
//       check_simplex_seq_saddle(v, X2, indices, i, j, cell_offset + 1, critical_points);
// 		}
// 	}
//   return critical_points; 
// }


inline bool file_exists(const std::string& filename) {
    std::ifstream f(filename.c_str());
    return f.good();
}


// template<typename Container>
// bool inside(const Container& x, int DH, int DW) {
//   if (x[0] <=0 || x[0] > DW-1 || x[1] <= 0 || x[1] > DH-1) return false;
//   return true;
// }

void record_criticalpoints(const std::string& prefix, const std::unordered_map<int, critical_point_t>& cps, bool write_sid=false){
  std::vector<double> singular;
  std::vector<double> attracting;
  std::vector<double> repelling;
  std::vector<double> saddle;
  std::vector<double> attracting_focus;
  std::vector<double> repelling_focus;
  std::vector<double> center;
  for (const auto& cpd:cps){
    auto cp = cpd.second;
    if (cp.type == SINGULAR){
      singular.push_back(cp.x[0]);
      singular.push_back(cp.x[1]);
    }
    else if (cp.type == ATTRACTING){
      attracting.push_back(cp.x[0]);
      attracting.push_back(cp.x[1]);
    }
    else if (cp.type == REPELLING){
      repelling.push_back(cp.x[0]);
      repelling.push_back(cp.x[1]);
    }
    else if (cp.type == SADDLE){
      saddle.push_back(cp.x[0]);
      saddle.push_back(cp.x[1]);
    }
    else if (cp.type == ATTRACTING_FOCUS){
      attracting_focus.push_back(cp.x[0]);
      attracting_focus.push_back(cp.x[1]);
    }
    else if (cp.type == REPELLING_FOCUS){
      repelling_focus.push_back(cp.x[0]);
      repelling_focus.push_back(cp.x[1]);
    }
    else if (cp.type == CENTER){
      center.push_back(cp.x[0]);
      center.push_back(cp.x[1]);
    }
    else {
      continue;
    }
  }
  // std::string prefix = "../data/position";
  writefile((prefix + "_singular.dat").c_str(), singular.data(), singular.size());
  writefile((prefix + "_attracting.dat").c_str(), attracting.data(), attracting.size());
  writefile((prefix + "_repelling.dat").c_str(), repelling.data(), repelling.size());
  writefile((prefix + "_saddle.dat").c_str(), saddle.data(), saddle.size());
  writefile((prefix + "_attracting_focus.dat").c_str(), attracting_focus.data(), attracting_focus.size());
  writefile((prefix + "_repelling_focus.dat").c_str(), repelling_focus.data(), repelling_focus.size());
  writefile((prefix + "_center.dat").c_str(), center.data(), center.size());
  printf("Successfully write critical points to file %s\n", prefix.c_str());
  
}

// inline bool is_upper(const std::array<double, 2> x){
//   double x_ex = x[0] - floor(x[0]);
//   double y_ex = x[1] - floor(x[1]);
//   if (y_ex >= x_ex){
//     return true;
//   }
//   else{
//     return false;
//   }
// }

// inline int get_cell_offset(const double *x, const int DW, const int DH){
//   int x0 = floor(x[0]);
//   int y0 = floor(x[1]);
//   int cell_offset = 2*(y0 * (DW-1) + x0);
//   if (!is_upper({x[0], x[1]})){
//     cell_offset += 1;
//   }
//   return cell_offset;
// }

// inline bool vaild_offset(const int offset, const int DW, const int DH){
//   if (offset < 0 || offset >= 2*(DW-1)*(DH-1)){
//     return false;
//   }
//   else{
//     return true;
//   }
// }
// inline bool vaild_offset(const std::array<double,2>& x, const int DW, const int DH){
//   if(x[0] < 0 || x[0] > DW-1 || x[1] < 1 || x[1] > DH-1){
//     return false;
//   }
//   else{
//     return true;
//   }

// }

// inline std::vector<int> get_surrounding_cell(const int cell_offset,const std::array<double,2>& x, const int DW, const int DH){
//   std::vector<int> surrounding_cell;
//   // 修改了这里
//   if (vaild_offset(cell_offset,DW,DH)){
//       surrounding_cell.push_back(cell_offset);
//     }
//   return surrounding_cell;
// }

//   if(floor(x[0]) == 0 || floor(x[0]) == DW || floor(x[1]) == 0 || floor(x[1]) == DH-2){
//     surrounding_cell.push_back(cell_offset);
//     return surrounding_cell;
//   }
//   if (cell_offset %2 == 0){
//     //upper cell
//     if (vaild_offset(cell_offset,DW,DH)){
//       surrounding_cell.push_back(cell_offset);
//     }
//     if (vaild_offset(cell_offset+1,DW,DH)){
//       surrounding_cell.push_back(cell_offset+1);
//     }
//     if (vaild_offset(cell_offset+2*(DW-1)+1,DW,DH)){
//       surrounding_cell.push_back(cell_offset+2*(DW-1)+1);
//     }
//     if (vaild_offset(cell_offset-1,DW,DH)){
//       surrounding_cell.push_back(cell_offset-1);
//     }
//     // if (vaild_offset(cell_offset-2*(DW-1),DW,DH)){
//     //   surrounding_cell.push_back(cell_offset-2*(DW-1));
//     // }
//   }
//   else {
//     if (vaild_offset(cell_offset,DW,DH)){
//       surrounding_cell.push_back(cell_offset);
//     }
//     if (vaild_offset(cell_offset-1,DW,DH)){
//       surrounding_cell.push_back(cell_offset-1);
//     }
//     if (vaild_offset(cell_offset+1,DW,DH)){
//       surrounding_cell.push_back(cell_offset+1);
//     }
//     if (vaild_offset(cell_offset-2*(DW-1)-1,DW,DH)){
//       surrounding_cell.push_back(cell_offset-2*(DW-1)-1);
//     }
//   }
//   return surrounding_cell;
// }








//overload trajectory function

bool areTrajsEqual(const std::vector<std::vector<std::array<double, 2>>>& vector1,
                     const std::vector<std::vector<std::array<double, 2>>>& vector2) {
    // 首先比较外层 std::vector 的大小
    bool result = true;
    if (vector1.size() != vector2.size()) {
        return false;
    }

    // 逐个比较中间层的 std::vector
    for (size_t i = 0; i < vector1.size(); ++i) {
        const std::vector<std::array<double, 2>>& innerVector1 = vector1[i];
        const std::vector<std::array<double, 2>>& innerVector2 = vector2[i];

        // 检查中间层 std::vector 的大小
        if (innerVector1.size() != innerVector2.size()) {
            printf("traj_length diff: %ld, %ld\n", innerVector1.size(), innerVector2.size());
            result = false;
        }

        // 逐个比较内部的 std::array
        for (size_t j = 0; j < innerVector1.size(); ++j) {
            const std::array<double, 2>& array1 = innerVector1[j];
            const std::array<double, 2>& array2 = innerVector2[j];

            // 比较内部的 std::array 是否相等
            if (array1 != array2) {
                //printf("traj diff: (%f, %f), (%f, %f)\n", array1[0], array1[1], array2[0], array2[1]);
                result = false;
            }
        }
    }

    // 如果所有元素都相等，则返回 true，表示两个数据结构相等
    return result;
}

bool areVectorsEqual(const std::vector<int>& vector1, const std::vector<int>& vector2) {
    // 首先比较向量的大小
    if (vector1.size() != vector2.size()) {
        return false;
    }
    // 逐个比较向量的元素
    for (size_t i = 0; i < vector1.size(); ++i) {
        if (vector1[i] != vector2[i]) {
            return false;
        }
    }
    // 如果所有元素都相等，则返回 true，表示两个向量相等
    return true;
}

// inline std::set<std::array<size_t,2>> get_three_coords(const std::array<double, 2>& x, const int DW, const int DH){
//   size_t x0 = floor(x[0]); 
//   size_t y0 = floor(x[1]);
//   std::set<std::array<size_t,2>> result;
//   if (is_upper({x[0], x[1]})){
//     std::array<size_t, 2> left_low= {x0, y0};
//     std::array<size_t, 2> left_up = {x0, y0+1};
//     std::array<size_t, 2> right_up = {x0+1, y0+1};
//     result.insert(left_low);
//     result.insert(left_up);
//     result.insert(right_up);
//   }
//   else{
//     std::array<size_t, 2> left_low= {x0, y0};
//     std::array<size_t, 2> right_low = {x0+1, y0};
//     std::array<size_t, 2> right_up = {x0+1, y0+1};
//     result.insert(left_low);
//     result.insert(right_low);
//     result.insert(right_up);
//   }
//   return result;
// }



inline std::set<size_t> convert_simplexID_to_offset(const std::set<size_t>& simplex, const int DW, const int DH){
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



inline std::set<size_t>findIntersectingTriangle(std::array<double, 2> p, std::array<double, 2> q, const int DW, const int DH){
  std::set<size_t> result;
  //find the cell that contains p and q
  std::array<size_t, 2> p_cell = {static_cast<size_t>(floor(p[0])), static_cast<size_t>(floor(p[1]))};
  std::array<size_t, 2> q_cell = {static_cast<size_t>(floor(q[0])), static_cast<size_t>(floor(q[1]))};
  if (p_cell == q_cell){
    //p and q are in the same rectangle
    if (is_upper(p) != is_upper(q)){
    //diff triangle
      result.insert(is_upper(q) ? 2*(q_cell[1] * (DW-1) + q_cell[0]) : 2*(q_cell[1] * (DW-1) + q_cell[0]) + 1);
      result.insert(is_upper(p) ? 2*(p_cell[1] * (DW-1) + p_cell[0]) : 2*(p_cell[1] * (DW-1) + p_cell[0]) + 1);
    }
    else{
      //same triangle
      result.insert(is_upper(p) ? 2*(p_cell[1] * (DW-1) + p_cell[0]) : 2*(p_cell[1] * (DW-1) + p_cell[0]) + 1);
    }
    return result;
  }
  else{
  //printf("start: (%f, %f), end: (%f, %f)\n", p[0], p[1], q[0], q[1]);
  //find the intersecting cell
  int startX = std::floor(p[0]);
  int startY = std::floor(p[1]);
  int endX = std::floor(q[0]);
  int endY = std::floor(q[1]);

  double deltaX = q[0] - p[0];
  double deltaY = q[1] - p[1];
  // 计算水平方向上需要经过的网格数量
  int stepsX = std::abs(endX - startX);
  // 计算垂直方向上需要经过的网格数量
  int stepsY = std::abs(endY - startY);

  for (int i = 0; i <= stepsX; ++i) {
        for (int j = 0; j <= stepsY; ++j) {
            size_t x = startX + i * ((deltaX >= 0) ? 1 : -1);
            size_t y = startY + j * ((deltaY >= 0) ? 1 : -1);
            result.insert(static_cast<size_t>(2*(y * (DW-1) + x)-1)); //可能有坑这块
            result.insert(static_cast<size_t>(2*(y * (DW-1) + x)));
            result.insert(static_cast<size_t>(2*(y * (DW-1) + x) +1));
            //printf("insert simplex: %ld, %ld\n", 2*(y * (DW-1) + x), 2*(y * (DW-1) + x)+1);
        }
    }
  }
  return result;
}

inline std::set<size_t>get_all_simplex_ID_new(const std::vector<std::array<double, 2>>& tracepoints, const int DW, const int DH){
  std::set<size_t> result;
  result.insert(get_cell_offset(tracepoints[0].data(), DW, DH)); //insert the first point
  //遍历前后两个点，找到所有的simplex
  for (size_t i = 0; i < tracepoints.size()-1; ++i){
    auto& point1 = tracepoints[i];
    auto& point2 = tracepoints[i+1];
    //printf("point1: (%f, %f), point2: (%f, %f)\n", point1[0], point1[1], point2[0], point2[1]);
    // if (point1[0] <= 1.0 || point1[0] >= DW-2.0 || point1[1] <= 1.0 || point1[1] >= DH-2.0) continue;
    auto intersecting_simplex_ids = findIntersectingTriangle(point1, point2, DW, DH);
    //printf("intersecting_cells size: %ld\n", intersecting_cells.size());
    result.insert(intersecting_simplex_ids.begin(), intersecting_simplex_ids.end());
  }
  return result;
  
}


void difftrajectory(const std::vector<std::vector<std::array<double, 2>>>& tracepoints1,const std::vector<std::vector<std::array<double, 2>>>& tracepoints2, const int DW, const int DH, std::set<size_t>& diff_offset_index,std::string test_flag,std::vector<size_t>& diff_traj_index) {
  // std::set<int> diff_offset;
  // std::set<int> diff_coords;
  int same_count = 0;
  for (size_t i =0 ; i < tracepoints1.size(); ++i){
    const auto& t1 = tracepoints1[i]; // trajectory 1,orginal
    const auto& t2 = tracepoints2[i]; // trajectory 2,decompressed
    int diff_flag = 0;
    //check if two trajectories has same simplexes set
    auto t1_simplexs = get_all_simplex_ID_new(t1, DW, DH);
    auto t2_simplexs = get_all_simplex_ID_new(t2, DW, DH);
    // need to remove 0, since t2 may shorter than t1
    t1_simplexs.erase(0);
    // if (!std::includes(t2_simplexs.begin(), t2_simplexs.end(),t1_simplexs.begin(), t1_simplexs.end())){
    if (t1_simplexs != t2_simplexs){
      diff_traj_index.push_back(i);
      // std::set<size_t> diff;
      diff_flag = 1;
      // std::set_symmetric_difference(t1_simplexs.begin(), t1_simplexs.end(), t2_simplexs.begin(), t2_simplexs.end(), std::inserter(diff, diff.begin()));
    }

    if (diff_flag == 1){
        // not equal set
        auto offests = convert_simplexID_to_offset(t1_simplexs, DW, DH);
        diff_offset_index.insert(offests.begin(), offests.end());
    }

    
    if (diff_flag == 0){
      //printf("trajectory %ld is the same\n", i);
      same_count ++;
    }
  }
  printf("same trajectory count: %d / %zu\n", same_count, tracepoints1.size());

}



void check_two_traj_start_end_cell(const std::vector<std::vector<std::array<double, 2>>>& tracepoints1,const std::vector<std::vector<std::array<double, 2>>>& tracepoints2,std::unordered_map<int, critical_point_t>& critical_points_ori, std::unordered_map<int, critical_point_t>& critical_points_out, const int DW, const int DH){
  //check if start and end point is the same
  std::vector<std::array<double,4>> traj1;
  std::vector<std::array<double,4>> traj2;
  for (auto t1:tracepoints1){
    std::array<double,4> result;
    result[0] = t1[0][0]; //start x
    result[1] = t1[0][1]; //start y
    result[2] = t1.back()[0]; //end x
    result[3] = t1.back()[1]; //end y
    traj1.push_back(result);
  }
  for (auto t2:tracepoints2){
    std::array<double,4> result;
    result[0] = t2[0][0];
    result[1] = t2[0][1];
    result[2] = t2.back()[0];
    result[3] = t2.back()[1];
    traj2.push_back(result);
  }
  // compare two sets
  //calculate how many start and end points are the same
  int count_same_cell = 0;
  for (auto t1 : traj1) {
    //double tmp1[2] = {t1[0],t1[1]};
    double tmp2[2] = {t1[2],t1[3]};
    //size_t cell_start = get_cell_offset(tmp1, DW, DH);
    size_t cell_end = get_cell_offset(tmp2, DW, DH);
    for (auto t2 : traj2) {
      //double tmp3[2] = {t2[0],t2[1]};
      double tmp4[2] = {t2[2],t2[3]};
      //size_t cell_start2 = get_cell_offset(tmp3, DW, DH);
      size_t cell_end2 = get_cell_offset(tmp4, DW, DH);
      if (t1[0] == t2[0] && t1[1] == t2[1] && cell_end == cell_end2){
        count_same_cell ++;
        break;
      }
    }

  }
  printf("start and end point are the in the same cell(or outbound): %d / %ld\n", count_same_cell, traj1.size());

  // 检查两个轨迹的交集
  
  std::unordered_map<std::array<double,2>,std::array<double,2>> traj1_map; //key: start point, value: end critical point
  std::unordered_map<std::array<double,2>,std::array<double,2>> traj2_map;
  for (auto t1:tracepoints1){
    if(t1.size() < 2000){ //not reach limit
      std::array<double,2> end_point = {t1.back()[0],t1.back()[1]};
      if (end_point[0] != -1 && end_point[1] != -1){ //not out bound
        //check if end point in critical point
        for (auto cp:critical_points_ori){
          if (fabs(end_point[0] - cp.second.x[0]) < 1e-3 && fabs(end_point[1] - cp.second.x[1]) < 1e-3){
            std::array<double,2> start_point = {t1[0][0],t1[0][1]};
            traj1_map[start_point] = end_point;
            break;
          }
        }
      }
    }
  }

  for (auto t2:tracepoints2){
    if(t2.size() < 2000){ //not reach limit
      std::array<double,2> end_point = {t2.back()[0],t2.back()[1]};
      if (end_point[0] != -1 && end_point[1] != -1){ //not out bound
        //check if end point in critical point
        for (auto cp:critical_points_out){
          if (fabs(end_point[0] - cp.second.x[0]) < 1e-3 && fabs(end_point[1] - cp.second.x[1]) < 1e-3){
            std::array<double,2> start_point = {t2[0][0],t2[0][1]};
            traj2_map[start_point] = end_point;
            break;
          }
        }
      }
    }
  }

  //now we have two maps, check the intersection
  int count_intersection = 0;
  for (auto t1:traj1_map){
    for (auto t2:traj2_map){
      if (t1.first[0] == t2.first[0] && t1.first[1] == t2.first[1]){
        //start point is the same
        if (t1.second[0] == t2.second[0] && t1.second[1] == t2.second[1]){
          //end point is the same
          count_intersection ++;
          break;
        }
      }
    }
  }
  printf("intersection of found cp in two trajs: %d\n", count_intersection);
      
    
  

  // std::set<std::array<double,4>> intersection;
  // std::set_intersection(traj1.begin(),traj1.end(),traj2.begin(),traj2.end(),std::inserter(intersection,intersection.begin()));
  // printf("start and end point are the same: %ld\n", intersection.size());

  // analyze the distance between original and decompressed end point
  double total_distance = 0;
  std::vector<double> distance_vector;
  for (size_t i = 0; i < tracepoints1.size(); ++i){
    if (tracepoints1[i].size() < 2000 && tracepoints1[i].back()[0] != -1 && tracepoints1[i].back()[1] != -1){
    // not reach max iteration for original data, and not out bound
      if (tracepoints2[i].size() >= 2000){
        // reach max iteration for decompressed data, but not out bound
      } 
    }
  }

}

void check_two_traj (std::vector<std::vector<std::array<double, 2>>>& tracepoints1,const std::vector<std::vector<std::array<double, 2>>>& tracepoints2, std::unordered_map<int, critical_point_t>& critical_points_ori, std::unordered_map<int, critical_point_t>& critical_points_out, const int DW, const int DH){
  // define reaching cp if the distance between critical point is less than 1e-3
  int count_reach_limit = 0;
  int count_found = 0;
  int count_out_bound = 0;
  for (auto t1:tracepoints1){
    if (t1.size() == 2000){
      count_reach_limit ++;
    }
    //check last element
    else if (t1.back()[0] <= 0 || t1.back()[0] >= DW-1 || t1.back()[1] <= 0 || t1.back()[1] >= DH-1){
      count_out_bound ++; //should be 0 since we have checked the boundary, if hit  boundary, then return the last point inside the boundary
    }
    else{
      for (auto cp:critical_points_ori){
        if (fabs(t1.back()[0] - cp.second.x[0]) < 1e-3 && fabs(t1.back()[1] - cp.second.x[1]) < 1e-3){
          count_found ++;
          break;
        }
      }
    }
  }
  printf("*******original data:*********\n");
  printf("total trajectory: %ld\n", tracepoints1.size());
  printf("reach limit(2000): %d\n", count_reach_limit);
  printf("found cp: %d\n", count_found);
  printf("out of bound: %d\n", count_out_bound);

  count_reach_limit = 0;
  count_found = 0;
  count_out_bound = 0;
  for (auto t2:tracepoints2){
    if (t2.size() == 2000){
      count_reach_limit ++;
    }
    //check last element
    else if (t2.back()[0] <= 0 || t2.back()[0] >= DW-1 || t2.back()[1] <= 0 || t2.back()[1] >= DH-1){
      count_out_bound ++;
    }
    else{
      for (auto cp:critical_points_out){
        if (fabs(t2.back()[0] - cp.second.x[0]) < 1e-3 && fabs(t2.back()[1] - cp.second.x[1]) < 1e-3){
          count_found ++;
          break;
        }
      }
    }
  }
  printf("*******decompressed data:*********\n");
  printf("total trajectory: %ld\n", tracepoints2.size());
  printf("reach limit(2000): %d\n", count_reach_limit);
  printf("found cp: %d\n", count_found);
  printf("out of bound: %d\n", count_out_bound);
}
 
void check_and_write_two_traj_detail(std::vector<std::vector<std::array<double, 2>>>& tracepoints1,const std::vector<std::vector<std::array<double, 2>>>& tracepoints2,std::vector<int> &index_1, std::vector<int> &index_2,
 std::unordered_map<int, critical_point_t>& critical_points_ori, std::unordered_map<int, critical_point_t>& critical_points_out, 
 const int DW, const int DH,
 std::vector<std::vector<std::array<double, 2>>> &ori_found_dec_found_diff_ORI,
 std::vector<std::vector<std::array<double, 2>>> &ori_found_dec_found_diff_DEC,
 std::vector<std::vector<std::array<double, 2>>> &ori_found_dec_not_found_ORI,
 std::vector<std::vector<std::array<double, 2>>> &ori_found_dec_not_found_DEC,
 std::vector<int> &index_ori_found_dec_found_diff_ORI,
 std::vector<int> &index_ori_found_dec_found_diff_DEC,
 std::vector<int> &index_ori_found_dec_not_found_ORI,
 std::vector<int> &index_ori_found_dec_not_found_DEC,traj_config t_config){
  // given two trajectories,(original and decompressed,without lossless), chek diff
  // if start and end point are in the same cell(but not reach limit), ignore
  printf("orginal trajectory size: %ld, decompressed trajectory size: %ld\n", tracepoints1.size(), tracepoints2.size());
  int both_found_cp = 0;
  int ori_found_cp_dec_not_found = 0;
  int ori_found_cp_dec_found_diff = 0;

  for(int i = 0; i<tracepoints1.size();i++){
    auto original = tracepoints1[i];
    auto decompressed = tracepoints2[i];
    int original_index = index_1[i];
    int decompressed_index = index_2[i];
    if(original.back()[0] != -1){
      if (original.size() < t_config.max_length){
        bool ori_found_cp = false;
        for (auto cp:critical_points_ori){
          if (fabs(original.back()[0] - cp.second.x[0]) < 1e-3 && fabs(original.back()[1] - cp.second.x[1]) < 1e-3){
            if(get_cell_offset(original.back().data(), DW, DH) == get_cell_offset(cp.second.x, DW, DH)){
              //end point is the same cell
              ori_found_cp = true;
              break;
            }
          }
        }
        if (ori_found_cp == false){
          printf("end point is not cp\n");
        }
        else{
          //ori end point is cp 
          if (decompressed.back()[0] == -1 || decompressed.size() == t_config.max_length){
            // ori found cp, dec not found cp(out or not found)
            ori_found_dec_not_found_ORI.push_back(original);
            index_ori_found_dec_not_found_ORI.push_back(original_index);
            ori_found_dec_not_found_DEC.push_back(decompressed);
            index_ori_found_dec_not_found_DEC.push_back(decompressed_index);
            ori_found_cp_dec_not_found ++;
          }
          else if(get_cell_offset(decompressed.back().data(), DW, DH) != get_cell_offset(original.back().data(), DW, DH)){
            // ori found cp, dec found cp, but not in the same cell
            ori_found_dec_found_diff_ORI.push_back(original);
            index_ori_found_dec_found_diff_ORI.push_back(original_index);
            ori_found_dec_found_diff_DEC.push_back(decompressed);
            index_ori_found_dec_found_diff_DEC.push_back(decompressed_index);
            ori_found_cp_dec_found_diff ++;
          }
          else{
            // ori found cp, dec found cp, in the same cell
            both_found_cp ++;
          }
          

        }
      }
    }
  }

  // std::vector<int> index_ori_found_dec_not_ORI;
  // std::vector<int> index_ori_found_dec_not_DEC;
  // int count_diff_cell = 0;
  // int count_diff_both_inbound = 0;
  // int count_intersection_both_cp = 0;
  // for(int i=0; i<tracepoints1.size(); i++){
  //   auto original = tracepoints1[i];
  //   auto decompressed = tracepoints2[i];
  //   int original_index = index_1[i];
  //   int decompressed_index = index_2[i];
  //   if (original.back()[0] != -1 && decompressed.back()[0] != -1){
  //     //both trajs are not out bound
  //     if(get_cell_offset(original.back().data(), DW, DH) != get_cell_offset(decompressed.back().data(), DW, DH)){
  //       //end point is diff
  //       count_diff_both_inbound ++;
  //     }
  //   }
  //   if(original.back()[0] != -1 && decompressed.back()[0] != -1){
  //     //both trajs are not out bound
  //     if (original.size() < 2000 && decompressed.size() < 2000){
  //       //both trajs are not reach limit
  //       //check if the end points has cp
  //       for (auto cp:critical_points_ori){
  //         if (fabs(original.back()[0] - cp.second.x[0]) < 1e-3 && fabs(original.back()[1] - cp.second.x[1]) < 1e-3){
  //           bool found = false;
  //           for (auto cp2:critical_points_out){
  //             if (fabs(decompressed.back()[0] - cp2.second.x[0]) < 1e-3 && fabs(decompressed.back()[1] - cp2.second.x[1]) < 1e-3){ //decompressed data found cp
  //               //both end points are cp
  //               if(get_cell_offset(original.back().data(), DW, DH) == get_cell_offset(decompressed.back().data(), DW, DH)){
  //                 //end point is the same cell
  //                 count_intersection_both_cp ++;
  //                 found = true;
  //                 break;
  //               }
  //             }
  //           }
  //           if (found == false){
  //               ori_found_dec_not_ORI.push_back(original);
  //               index_ori_found_dec_not_ORI.push_back(original_index);
  //               ori_found_dec_not_DEC.push_back(decompressed);
  //               index_ori_found_dec_not_DEC.push_back(decompressed_index);
  //           }
  //           break;
  //         }
  //       }
  //     }
  //   }
  //   // 
  //   // if (get_cell_offset(original[0].data(), DW, DH) == get_cell_offset(decompressed[0].data(), DW, DH)){
  //   //   //start point is the same
  //   //   //now get the non (-1,-1) end point
  //   //   for (int j = original.size()-1; j >= 0; j--){
  //   //     if (original[j][0] != -1 && original[j][1] != -1){
  //   //       last_inbound_original = {original[j][0], original[j][1]};
  //   //       break;
  //   //     }
  //   //   }
  //   //   for (int j = decompressed.size()-1; j >= 0; j--){
  //   //     if (decompressed[j][0] != -1 && decompressed[j][1] != -1){
  //   //       last_inbound_decompressed = {decompressed[j][0], decompressed[j][1]};
  //   //       break;
  //   //     }
  //   //   }
  //   //   if (last_inbound_original[0] == -1 || last_inbound_decompressed[0] == -1){
  //   //     printf("last inbound point not found!!!\n");
  //   //   }
  //   //   // if the cell is the same, then ignore
  //   //   if (get_cell_offset(last_inbound_original.data(), DW, DH) == get_cell_offset(last_inbound_decompressed.data(), DW, DH)){
  //   //     //end point is the same
  //   //     continue;
  //   //   }
  //   //   else{
  //   //     count_diff_cell ++;
  //   //     //end point is different
  //   //     //printf("trajectory %d , end point is different\n", i);
  //   //     printf("original: (%f, %f), decompressed: (%f, %f)\n", last_inbound_original[0], last_inbound_original[1], last_inbound_decompressed[0], last_inbound_decompressed[1]);
  //   //     out_traj1.push_back(original);
  //   //     out_traj2.push_back(decompressed);
  //   //     out_index1.push_back(original_index);
  //   //     out_index2.push_back(decompressed_index);
  //   //   }
  //   // }
  //   // else printf("start point is different!!! (original: (%f, %f), decompressed: (%f, %f))\n", original[0][0], original[0][1], decompressed[0][0], decompressed[0][1]);
  //   // printf("start point (%f, %f), last inbound point: (%f, %f), last point:(%f, %f)\n", original[0][0], original[0][1], last_inbound_original[0], last_inbound_original[1], original.back()[0], original.back()[1]);
  // }
  // printf("different end point count(no matter what kind): %d\n", count_diff_cell);
  // printf("different end point count(both in bound): %d\n", count_diff_both_inbound);
  // printf("intersection of both cp: %d\n", count_intersection_both_cp);
  // //write to file
  // write_trajectory(ori_found_dec_not_ORI, "/Users/mingzexia/Documents/Github/tracecp/small_data/ori_found_dec_not_ORI.bin");
  // write_trajectory(ori_found_dec_not_DEC, "/Users/mingzexia/Documents/Github/tracecp/small_data/ori_found_dec_not_DEC.bin");
  // writefile("/Users/mingzexia/Documents/Github/tracecp/small_data/index_ori_found_dec_not_ORI.bin", index_ori_found_dec_not_ORI.data(), index_ori_found_dec_not_ORI.size());
  // writefile("/Users/mingzexia/Documents/Github/tracecp/small_data/index_ori_found_dec_not_DEC.bin", index_ori_found_dec_not_DEC.data(), index_ori_found_dec_not_DEC.size());

}


// template<typename T>
// void
// fix_traj(const T * U, const T * V, T *&dec_U, T *&dec_V,size_t r1, size_t r2, double max_pwr_eb,uint64_t vector_field_scaling_factor,uint64_t vector_field_scaling_factor_dec, traj_config t_config){
//   //r2 is DW, r1 is DH
//   int DW = r2;
//   int DH = r1;
//   bool stop = false;
//   int NUM_ITER = 0;
//   int MAX_ITER = 10;
//   double current_pwr_eb = max_pwr_eb;
//   while(!stop){
//     //current_pwr_eb /= 2;
//     current_pwr_eb = 0;
//     printf("CURRENT ITERATION: %d, CURRENT_EB: %f\n", NUM_ITER, current_pwr_eb);
//     //get grad for original data
//     ftk::ndarray<float> grad_ori;
//     grad_ori.reshape({2, static_cast<unsigned long>(r2), static_cast<unsigned long>(r1)});
//     refill_gradient(0, r1, r2, U, grad_ori);
//     refill_gradient(1, r1, r2, V, grad_ori);
//     //get grad for decompressed data
//     ftk::ndarray<float> grad_dec;
//     grad_dec.reshape({2, static_cast<unsigned long>(r2), static_cast<unsigned long>(r1)});
//     refill_gradient(0, r1, r2, dec_U, grad_dec);
//     refill_gradient(1, r1, r2, dec_V, grad_dec);
//     printf("999 ori_u: %f, 999 dec_u: %f\n", grad_ori(0, 999, 999), grad_dec(0, 999, 999));
//     //get cp for dec_U and dec_V
//     auto critical_points_dec = compute_critical_points(dec_U, dec_V, r1, r2,vector_field_scaling_factor);
//     //get cp for U and V
//     auto critical_points_ori = compute_critical_points(U, V, r1, r2,vector_field_scaling_factor_dec);
//     std::vector<std::vector<std::array<double, 2>>> trajs_ori;
//     std::vector<std::vector<std::array<double, 2>>> trajs_dec;
//     std::vector<int> index_ori;
//     std::vector<int> index_dec;
//     //get trajectory for original data
//     for(const auto& p:critical_points_ori){
//       auto cp = p.second;
//       if (cp.type == SADDLE){
//         global_count ++;
//         std::vector<std::array<double,2>> X_all_direction;  
//         X_all_direction.push_back({cp.x[0] + t_config.eps*cp.eig_vec[0][0], cp.x[1] + t_config.eps*cp.eig_vec[0][1]});
//         X_all_direction.push_back({cp.x[0] - t_config.eps*cp.eig_vec[0][0], cp.x[1] - t_config.eps*cp.eig_vec[0][1]});
//         X_all_direction.push_back({cp.x[0] + t_config.eps*cp.eig_vec[1][0], cp.x[1] + t_config.eps*cp.eig_vec[1][1]});
//         X_all_direction.push_back({cp.x[0] - t_config.eps*cp.eig_vec[1][0], cp.x[1] - t_config.eps*cp.eig_vec[1][1]});                              
//         double lambda[3];
//         double values[2];
//         std::vector<std::vector<double>> config;
//         config.push_back({cp.eig_vec[0][0], cp.eig_vec[0][1],1});
//         config.push_back({cp.eig_vec[0][0], cp.eig_vec[0][1],1});
//         config.push_back({cp.eig_vec[1][0], cp.eig_vec[1][1],-1});
//         config.push_back({cp.eig_vec[1][0], cp.eig_vec[1][1],-1});
//         for (int i = 0; i < 4; i ++) {
//           std::array<double, 2> X_start;
//           std::vector<std::array<double, 2>> result_return;
//           X_start = X_all_direction[i];
//           //check if inside
//           if (inside(X_start,DH, DW)){
//             //printf("processing (%f,%f)\n", X_start[0], X_start[1]);
//             if(i == 0 || i ==1){
//               result_return = trajectory(cp.x,X_start, t_config.h,t_config.max_length,DH,DW, critical_points_ori,grad_ori,index_ori);
//               trajs_ori.push_back(result_return);
//             }
//             else{
//               result_return = trajectory(cp.x,X_start, -t_config.h,t_config.max_length,DH,DW, critical_points_ori,grad_ori,index_ori);
//               trajs_ori.push_back(result_return);
//             }
//           }
//         }
//       }
//     }   
//     printf("original trajectory calculation done,size: %ld\n", trajs_ori.size());
//     //get trajectory for decompressed data
//     for(const auto& p:critical_points_dec){
//       auto cp = p.second;
//       if (cp.type == SADDLE){
//         global_count ++;
//         std::vector<std::array<double,2>> X_all_direction;  
//         X_all_direction.push_back({cp.x[0] + t_config.eps*cp.eig_vec[0][0], cp.x[1] + t_config.eps*cp.eig_vec[0][1]});
//         X_all_direction.push_back({cp.x[0] - t_config.eps*cp.eig_vec[0][0], cp.x[1] - t_config.eps*cp.eig_vec[0][1]});
//         X_all_direction.push_back({cp.x[0] + t_config.eps*cp.eig_vec[1][0], cp.x[1] + t_config.eps*cp.eig_vec[1][1]});
//         X_all_direction.push_back({cp.x[0] - t_config.eps*cp.eig_vec[1][0], cp.x[1] - t_config.eps*cp.eig_vec[1][1]});                              
//         double lambda[3];
//         double values[2];
//         std::vector<std::vector<double>> config;
//         config.push_back({cp.eig_vec[0][0], cp.eig_vec[0][1],1});
//         config.push_back({cp.eig_vec[0][0], cp.eig_vec[0][1],1});
//         config.push_back({cp.eig_vec[1][0], cp.eig_vec[1][1],-1});
//         config.push_back({cp.eig_vec[1][0], cp.eig_vec[1][1],-1});
//         for (int i = 0; i < 4; i ++) {
//           std::array<double, 2> X_start;
//           std::vector<std::array<double, 2>> result_return;
//           X_start = X_all_direction[i];
//           //check if inside
//           if (inside(X_start,DH, DW)){
//             //printf("processing (%f,%f)\n", X_start[0], X_start[1]);
//             if(i == 0 || i ==1){
//               result_return = trajectory(cp.x,X_start, t_config.h,t_config.max_length,DH,DW, critical_points_dec,grad_dec,index_dec);
//               trajs_dec.push_back(result_return);
//             }
//             else{
//               result_return = trajectory(cp.x,X_start, -t_config.h,t_config.max_length,DH,DW, critical_points_dec,grad_dec,index_dec);
//               trajs_dec.push_back(result_return);   
//             }
//           }
//         }
//       }
//     }
//     printf("decompressed trajectory calculation done, size: %ld\n", trajs_dec.size());
//     std::set<size_t> diff_vertex_index;
//     std::set<size_t> all_vertex_for_all_diff_traj;
//     int diff_flag_0_count = 0; // same
//     int diff_flag_1_count = 0; // decompressed not found cp
//     int diff_flag_2_count = 0; // decompressed found wrong cp
//     for (size_t i = 0; i < trajs_ori.size(); ++i){
//       auto t1 = trajs_ori[i];
//       auto t2 = trajs_dec[i];
//       int diff_flag = 0; // 0: same, 1: diff(decomp not found), 2: diff(decomp found wrong cp)
//       if (t1.size() != t_config.max_length && t1.back()[0] != -1){
//         // original data found cp
//         if (t2.size() >= t_config.max_length-1 || t2.back()[0] == -1){
//           // decompressed data not found cp
//           diff_flag = 1;
//         }
//         else{
//           if (get_cell_offset(t1.back().data(), DW, DH) != get_cell_offset(t2.back().data(), DW, DH)){
//             // both found cp but different
//             diff_flag = 2;
//           }
//         }
//       }
//       if(diff_flag == 0) diff_flag_0_count ++;
//       if(diff_flag == 1) diff_flag_1_count ++;
//       if(diff_flag == 2) diff_flag_2_count ++;
//       if (diff_flag != 0){
//         std::set<size_t> ori_vertex_index;
//         std::set<size_t> dec_vertex_index;
//         //
//         double current_v[2] = {0};
//         interp2d(t1[1].data(), current_v,grad_dec); 
//         std::array<double, 2> temp_result = newRK4(t1[1].data(),current_v,grad_dec,t_config.h, DH, DW,all_vertex_for_all_diff_traj);
//         for (auto p : t1){
//           auto offset = get_three_offsets(p.data(), DW, DH);
//           for (auto o:offset){
//             ori_vertex_index.insert(o);
//           }
//         }
//         for (auto p : t2){
//           // dec_vertex_index.insert(get_three_offsets(p.data(), DW, DH));
//           auto offset = get_three_offsets(p.data(), DW, DH);
//           for (auto o:offset){
//             dec_vertex_index.insert(o);
//           }
//         }
//         //std::set_symmetric_difference(ori_vertex_index.begin(), ori_vertex_index.end(), dec_vertex_index.begin(), dec_vertex_index.end(), std::inserter(diff_vertex_index, diff_vertex_index.begin()));
//         std::set_difference(ori_vertex_index.begin(), ori_vertex_index.end(), dec_vertex_index.begin(), dec_vertex_index.end(), std::inserter(diff_vertex_index, diff_vertex_index.begin()));
//         // 看起来这里用差集然后去减小eb似乎不能有效的减少错误的trajectory， 也许应该加上从出错的地方的前一个点开始的计算差集
//         //printf("trajectory %ld, diff_vertex_index size: %ld\n", i, diff_vertex_index.size());
//       }
//       // else{
//       //   printf("trajectory %ld, same\n", i);
//       // }
//     }
//     printf("diff_flag_0_count: %d, diff_flag_1_count: %d, diff_flag_2_count: %d\n", diff_flag_0_count, diff_flag_1_count, diff_flag_2_count);
//     if (diff_vertex_index.size() == 0){
//       stop = true;
//     }
//     else{
//       //fix the trajectory
//       size_t result_size = 0;
//       unsigned char * result =NULL;
//       // result = sz_compress_cp_preserve_2d_fix(U, V, DH,DW, result_size, false, max_pwr_eb, current_pwr_eb, diff_vertex_index);
//       printf("all_vertex_for_all_diff_traj size: %ld\n", all_vertex_for_all_diff_traj.size());
//       result = sz_compress_cp_preserve_2d_fix(U, V, DH,DW, result_size, false, max_pwr_eb, current_pwr_eb, all_vertex_for_all_diff_traj);
//       unsigned char * result_after_lossless = NULL;
//       size_t lossless_outsize = sz_lossless_compress(ZSTD_COMPRESSOR, 3, result, result_size, &result_after_lossless);
//       size_t lossless_output = sz_lossless_decompress(ZSTD_COMPRESSOR, result_after_lossless, lossless_outsize, &result, result_size);
//       //这里最好打印一下压缩率
//       printf("Compressed size(original) = %zu, ratio = %f\n", lossless_outsize, (2*r1*r2*sizeof(float)) * 1.0/lossless_outsize);
//       sz_decompress_cp_preserve_2d_online<float>(result, DH,DW, dec_U, dec_V);
//       free(result);
//       free(result_after_lossless);
//     }
//     NUM_ITER ++;
//   }
// }

template<typename T>
void write_current_state_data(std::string file_path, const T * U, const T * V, T *dec_U, T *dec_V, std::vector<std::vector<std::array<double, 2>>> &trajs_dec,std::vector<int> &index_dec, size_t r1, size_t r2, int NUM_ITER){
  
  // write decompress traj
  std::string dec_traj_file = file_path + "dec_traj_iteration_" + std::to_string(NUM_ITER) + ".bin" + ".out";
  write_trajectory(trajs_dec, dec_traj_file.c_str());
  printf("Successfully write decompressed trajectory to file, total trajectory: %ld\n",trajs_dec.size());
  //write decompress index
  std::string index_file = file_path + "index_iteration_" + std::to_string(NUM_ITER) + ".bin" + ".out";
  writefile(index_file.c_str(), index_dec.data(), index_dec.size());
  printf("Successfully write orginal index to file, total index: %ld\n",index_dec.size());

  //write dec data
  std::string decU_file = file_path + "dec_u_iteration_" + std::to_string(NUM_ITER) + ".bin" + ".out";
  writefile(decU_file.c_str(), dec_U, r1*r2);
  std::string decV_file = file_path + "dec_v_iteration_" + std::to_string(NUM_ITER) + ".bin" + ".out";
  writefile(decV_file.c_str(), dec_V, r1*r2);
}

void find_fix_index_end(int &end_fix_index,std::vector<std::array<double, 2>> t1,std::vector<std::array<double, 2>> t2,ftk::ndarray<float> &grad_ori, int DW, int DH, traj_config t_config, std::set<size_t> &all_vertex_for_all_diff_traj, size_t i, std::unordered_map<int, critical_point_t>& critical_points_ori){
    int start_fix_index = 0;
    if(t1[0][0] != t2[0][0] || t1[0][1] != t2[0][1] || t1[1][0] != t2[1][0] || t1[1][1] != t2[1][1]){
      printf("start point(cp) or seed is different!!!\n");
      exit(0);
    }
    for (size_t j = 0; j < t1.size(); ++j){
      auto p1 = t1[j];
      auto p2 = t2[j];
      if (p1[0] > 0 && p2[0] > 0){
        double dist = sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2));
        if (dist > 1.4142){
          start_fix_index = 0;
          end_fix_index = j;
          //end_fix_index = t1.size() - 1;
          break;
        }
      }
    }
    if (end_fix_index != 0){
      printf("trajectory %ld goes wrong, start_fix_at_index: %d, end_fix_at_index: %d\n", i, start_fix_index, end_fix_index);
      printf("traj go different ori: %f, %f, dec: %f, %f, dist: %f\n", t1[end_fix_index][0], t1[end_fix_index][1], t2[end_fix_index][0], t2[end_fix_index][1], sqrt(pow(t1[end_fix_index][0] - t2[end_fix_index][0], 2) + pow(t1[end_fix_index][1] - t2[end_fix_index][1], 2)));
      printf("orginal start point: (%f, %f),seed point:(%f,%f), original end point: (%f, %f),length: %zu\n", t1[0][0], t1[0][1],t1[1][0],t1[1][1], t1.back()[0], t1.back()[1], t1.size());
      printf("decomp start point: (%f, %f), seed point(%f,%f), decomp end point: (%f, %f),length: %zu\n", t2[0][0], t2[0][1], t2[1][0],t2[1][1],t2.back()[0], t2.back()[1], t2.size());
      printf("add vertexs from coord (%f, %f) to (%f, %f)\n", t1[start_fix_index][0], t1[start_fix_index][1], t1[end_fix_index][0], t1[end_fix_index][1]);
      // add vertexs at start_fix_index, end at end_fix_index

      /*
      for (int j = 1; j <= end_fix_index; ++j){
        auto p = t1[j];
        double current_v[2] = {0};
        interp2d(p.data(),current_v,grad_ori);
        auto offsets = vertex_for_each_RK4(p.data(), current_v,grad_ori, t_config.h,DH, DW);
        //auto offsets = get_three_offsets(p.data(), DW, DH);
        for (auto o:offsets){
          all_vertex_for_all_diff_traj.insert(o);
          // if (i == 9311){
          //   int temp_x = o % DW;
          //   int temp_y = o / DW;
          //   printf("add vertex: (%d, %d)\n", temp_x, temp_y);
          // }

        }
      }
      */

      int num = 0;
      auto cp_coord = t1[0]; //cp
      auto current_X = t1[1]; //seed
      int flag = 0;
      //check t1[1] is cp or not
      for(auto p:critical_points_ori){
        if (p.second.type == SADDLE && p.second.x[0] == cp_coord[0] && p.second.x[1] == cp_coord[1]){
          //cp_coord is cp
          flag = 1;
          break;
        }
      }
      if (flag == 0){
        printf("second element of original trajectory is not cp!!!\n");
        exit(0);  
      }

      auto offsets_cp = get_three_offsets(cp_coord.data(), DW, DH);
      auto offsets_seed = get_three_offsets(current_X.data(), DW, DH);
      for (auto o:offsets_cp){
        all_vertex_for_all_diff_traj.insert(o);
      }
      for (auto o:offsets_seed){
        all_vertex_for_all_diff_traj.insert(o);
      }

      std::set<size_t> temp_vertexs;
      std::vector<int> temp_index;
      
      int direction;
      if((i %4) == 0 || (i % 4) == 1){
        direction = 1;
      }
      else{
        direction = -1;
      }
      auto temp_trajs_pos = trajectory(cp_coord.data(), current_X, direction * t_config.h, end_fix_index, DH, DW, critical_points_ori, grad_ori,temp_index, temp_vertexs);
      //这里如果用trajectory需要知道他的方向，这里暂时两个方向都加上
      //auto temp_traj_neg = trajectory(cp_coord.data(), current_X, -t_config.h, end_fix_index, DH, DW, critical_points_ori, grad_ori,temp_index, temp_vertexs);
      for (auto o:temp_vertexs){
        all_vertex_for_all_diff_traj.insert(o);
      }
      
      // while(num < end_fix_index){
      //   std::set<size_t> temp_vertex;
      //   double current_v[2] = {0};
      //   interp2d(current_X.data(), current_v,grad_ori);
      //   //std::array<Type, 2> newRK4(const Type * x, const Type * v, const ftk::ndarray<float> &data,  Type h, const int DH, const int DW,std::set<size_t>& lossless_index)
      //   std::array<double, 2> temp_result = newRK4(current_X.data(),current_v,grad_ori,t_config.h, DH, DW, temp_vertex);
      //   current_X = temp_result;
      //   num++;
      //   for (auto o:temp_vertex){
      //     all_vertex_for_all_diff_traj.insert(o);
      //   }
      // }
    }
}


template<typename T>
void
fix_traj(const T * U, const T * V,size_t r1, size_t r2, double max_pwr_eb,traj_config t_config){
  int DW = r2;
  int DH = r1;
  bool stop = false;
  std::set<size_t> all_vertex_for_all_diff_traj;
  int NUM_ITER = 0;
  const int MAX_ITER = 10;
  //std::set<size_t> current_diff_traj_id;
  std::unordered_map<size_t,int> current_diff_traj_id;
  //std::set<size_t> last_diff_traj_id;
  std::unordered_map<size_t,int> last_diff_traj_id;
  
  while (!stop)
  {

  if (NUM_ITER >= MAX_ITER) break;
  
  
  //double current_pwr_eb = max_pwr_eb / pow(2, NUM_ITER);
  double current_pwr_eb = 0;
  printf("CURRENT ITERATION: %d, current_pwr_eb: %f\n", NUM_ITER, current_pwr_eb);
  // first get decompressed data
  size_t result_size = 0;
  unsigned char * result = NULL;
  result = sz_compress_cp_preserve_2d_fix(U, V, r1, r2, result_size, false, max_pwr_eb, current_pwr_eb, all_vertex_for_all_diff_traj);
  unsigned char * result_after_lossless = NULL;
  size_t lossless_outsize = sz_lossless_compress(ZSTD_COMPRESSOR, 3, result, result_size, &result_after_lossless);
  size_t lossless_output = sz_lossless_decompress(ZSTD_COMPRESSOR, result_after_lossless, lossless_outsize, &result, result_size);
  float * dec_U = NULL;
  float * dec_V = NULL;
  sz_decompress_cp_preserve_2d_online<float>(result, r1,r2, dec_U, dec_V); // use cpsz

  //把dec_U, dec_V里对应all_vertex_for_all_diff_traj的点的值都设为相应的U,V的值
  // for (auto o:all_vertex_for_all_diff_traj){
  //   dec_U[o] = U[o];
  //   dec_V[o] = V[o];
  // }

  // print compression ratio
  printf("Compressed size(original) = %zu, ratio = %f, all_vertex_need_record_size = %zu\n", lossless_outsize, (2*r1*r2*sizeof(float)) * 1.0/lossless_outsize, all_vertex_for_all_diff_traj.size());

  //get grad for original data
  ftk::ndarray<float> grad_ori;
  grad_ori.reshape({2, static_cast<unsigned long>(r2), static_cast<unsigned long>(r1)});
  refill_gradient(0, r1, r2, U, grad_ori);
  refill_gradient(1, r1, r2, V, grad_ori);
  //get grad for decompressed data
  ftk::ndarray<float> grad_dec;
  grad_dec.reshape({2, static_cast<unsigned long>(r2), static_cast<unsigned long>(r1)});
  refill_gradient(0, r1, r2, dec_U, grad_dec);
  refill_gradient(1, r1, r2, dec_V, grad_dec);
  //get cp for original data
  auto critical_points_ori = compute_critical_points(U, V, r1, r2);
  //get cp for decompressed data
  auto critical_points_dec = compute_critical_points(dec_U, dec_V, r1, r2);

  //check if dec_U and grad_dec are correct
  //for loop for r1 and for loop for r2
  for (int i = 0; i < r1; i++){
    for (int j = 0; j < r2; j++){
      //get dec_U and grad_dec(0, i, j)
      double dec_U_val = dec_U[i*r2+j];
      double grad_dec_U_val = grad_dec(0, j, i);
      double dec_V_val = dec_V[i*r2+j];
      double grad_dec_V_val = grad_dec(1, j, i);
      if (dec_U_val != grad_dec_U_val || dec_V_val != grad_dec_V_val){
        printf("Dec_U/Dec_V and grad_dec are not correct!!!\n");
        printf("may need check refill_gradient function\n");
        printf("dec_U: %f, grad_dec_U: %f, dec_V: %f, grad_dec_V: %f\n", dec_U_val, grad_dec_U_val, dec_V_val, grad_dec_V_val);
        exit(0);
      }
    }
  }



  // print out cp size and saddle size
  printf("critical_points_ori size: %ld, critical_points_dec size: %ld\n", critical_points_ori.size(), critical_points_dec.size());
  size_t saddle_ori_count = 0;
  size_t saddle_dec_count = 0;
  for (auto p:critical_points_ori){
    if (p.second.type == SADDLE){
      saddle_ori_count ++;
    }
  }
  for (auto p:critical_points_dec){
    if (p.second.type == SADDLE){
      saddle_dec_count ++;
    }
  }
  printf("saddle_ori_count: %ld, saddle_dec_count: %ld\n", saddle_ori_count, saddle_dec_count);

  // check if critical_points_ori and critical_points_dec are same
  for (auto p:critical_points_ori){
    auto cp_ori = p.second;
    auto cp_dec = critical_points_dec[p.first];
    if (cp_ori.type != cp_dec.type || cp_ori.x[0] != cp_dec.x[0] || cp_ori.x[1] != cp_dec.x[1] || cp_ori.eig_vec[0][0] != cp_dec.eig_vec[0][0] || cp_ori.eig_vec[0][1] != cp_dec.eig_vec[0][1] || cp_ori.eig_vec[1][0] != cp_dec.eig_vec[1][0] || cp_ori.eig_vec[1][1] != cp_dec.eig_vec[1][1]){
      printf("critical_points_ori and critical_points_dec are not same!!!\n");
      //print diff
      printf("ori: type: %d, x: (%f, %f), eig_vec: (%f, %f), (%f, %f)\n", cp_ori.type, cp_ori.x[0], cp_ori.x[1], cp_ori.eig_vec[0][0], cp_ori.eig_vec[0][1], cp_ori.eig_vec[1][0], cp_ori.eig_vec[1][1]);
      printf("dec: type: %d, x: (%f, %f), eig_vec: (%f, %f), (%f, %f)\n", cp_dec.type, cp_dec.x[0], cp_dec.x[1], cp_dec.eig_vec[0][0], cp_dec.eig_vec[0][1], cp_dec.eig_vec[1][0], cp_dec.eig_vec[1][1]);
      exit(0);
    }
  }


  //get trajectory for original data
  std::vector<std::vector<std::array<double, 2>>> trajs_ori;
  std::vector<int> index_ori;
  std::set<size_t> vertex_ori;
  // std::unordered_map<size_t,std::vector<size_t>> vertex_ori_map;
  for(const auto& p:critical_points_ori){
    auto cp = p.second;
    if (cp.type == SADDLE){
      global_count ++;
      std::vector<std::array<double,2>> X_all_direction;  
      X_all_direction.push_back({cp.x[0] + t_config.eps*cp.eig_vec[0][0], cp.x[1] + t_config.eps*cp.eig_vec[0][1]});
      X_all_direction.push_back({cp.x[0] - t_config.eps*cp.eig_vec[0][0], cp.x[1] - t_config.eps*cp.eig_vec[0][1]});
      X_all_direction.push_back({cp.x[0] + t_config.eps*cp.eig_vec[1][0], cp.x[1] + t_config.eps*cp.eig_vec[1][1]});
      X_all_direction.push_back({cp.x[0] - t_config.eps*cp.eig_vec[1][0], cp.x[1] - t_config.eps*cp.eig_vec[1][1]});                              
      double lambda[3];
      double values[2];
      std::vector<std::vector<double>> config;
      config.push_back({cp.eig_vec[0][0], cp.eig_vec[0][1],1});
      config.push_back({cp.eig_vec[0][0], cp.eig_vec[0][1],1});
      config.push_back({cp.eig_vec[1][0], cp.eig_vec[1][1],-1});
      config.push_back({cp.eig_vec[1][0], cp.eig_vec[1][1],-1});
      for (int i = 0; i < 4; i ++) {
        std::array<double, 2> X_start;
        std::vector<std::array<double, 2>> result_return;
        X_start = X_all_direction[i];
        //check if inside
        if (inside(X_start,DH, DW)){
          //printf("processing (%f,%f)\n", X_start[0], X_start[1]);
          double pos[2] = {cp.x[0], cp.x[1]};
          if(i == 0 || i ==1){
            result_return = trajectory(pos, X_start, t_config.h,t_config.max_length,DH,DW, critical_points_ori,grad_ori,index_ori,vertex_ori);
            trajs_ori.push_back(result_return);
          }
          else{
            result_return = trajectory(pos,X_start, -t_config.h,t_config.max_length,DH,DW, critical_points_ori,grad_ori,index_ori,vertex_ori);
            trajs_ori.push_back(result_return);
          }

        }
      }
    }
  }   
  
  //get trajectory for decompressed data
  std::vector<std::vector<std::array<double, 2>>> trajs_dec;
  std::vector<int> index_dec;
  std::set<size_t> vertex_dec;
  for(const auto& p:critical_points_dec){
    auto cp = p.second;
    if (cp.type == SADDLE){
      global_count ++;
      std::vector<std::array<double,2>> X_all_direction;  
      X_all_direction.push_back({cp.x[0] + t_config.eps*cp.eig_vec[0][0], cp.x[1] + t_config.eps*cp.eig_vec[0][1]});
      X_all_direction.push_back({cp.x[0] - t_config.eps*cp.eig_vec[0][0], cp.x[1] - t_config.eps*cp.eig_vec[0][1]});
      X_all_direction.push_back({cp.x[0] + t_config.eps*cp.eig_vec[1][0], cp.x[1] + t_config.eps*cp.eig_vec[1][1]});
      X_all_direction.push_back({cp.x[0] - t_config.eps*cp.eig_vec[1][0], cp.x[1] - t_config.eps*cp.eig_vec[1][1]});                              
      double lambda[3];
      double values[2];
      std::vector<std::vector<double>> config;
      config.push_back({cp.eig_vec[0][0], cp.eig_vec[0][1],1});
      config.push_back({cp.eig_vec[0][0], cp.eig_vec[0][1],1});
      config.push_back({cp.eig_vec[1][0], cp.eig_vec[1][1],-1});
      config.push_back({cp.eig_vec[1][0], cp.eig_vec[1][1],-1});
      for (int i = 0; i < 4; i ++) {
        std::array<double, 2> X_start;
        std::vector<std::array<double, 2>> result_return;
        X_start = X_all_direction[i];
        //check if inside
        if (inside(X_start,DH, DW)){
          double pos[2] = {cp.x[0], cp.x[1]};
          //printf("processing (%f,%f)\n", X_start[0], X_start[1]);
          if(i == 0 || i ==1){
            result_return = trajectory(pos,X_start, t_config.h,t_config.max_length,DH,DW, critical_points_dec,grad_dec,index_dec,vertex_dec);
            trajs_dec.push_back(result_return);
          }
          else{
            result_return = trajectory(pos,X_start, -t_config.h,t_config.max_length,DH,DW, critical_points_dec,grad_dec,index_dec,vertex_dec);
            trajs_dec.push_back(result_return);   
          }
        }
      }
    }
  }




  //compare the trajectory, and get all vertex for all different trajectories
  int ori_out_reach_limit = 0; // same
  int dec_not_found_count = 0; // decompressed not found cp
  int dec_found_wrong_cp_count = 0; // decompressed found wrong cp

  /*
  check if the trajectory is the same
  */
  for(size_t i=0;i < trajs_ori.size(); ++i){
    auto t1 = trajs_ori[i];
    auto t2 = trajs_dec[i];
    if (t1.size() != t_config.max_length && t1.back()[0] != -1){
      // original data found cp
      if (t2.size() == t_config.max_length || t2.back()[0] == -1){
        // decompressed data not found cp
        dec_not_found_count ++;

      }
      else{
        if (get_cell_offset(t1.back().data(), DW, DH) != get_cell_offset(t2.back().data(), DW, DH)){
          // both found cp but different
          dec_found_wrong_cp_count ++;
        }
      }
    }
    else{
      ori_out_reach_limit ++;
    }
  }

  if (dec_not_found_count == 0 && dec_found_wrong_cp_count ==0){
    printf("all trajectories are the same...stop\n");
    printf("ori_out_or_reach_limit: %d, dec_not_found_count: %d, dec_found_wrong_cp_count: %d\n", ori_out_reach_limit, dec_not_found_count, dec_found_wrong_cp_count);
    printf("current_pwr_eb: %f\n", current_pwr_eb);
    stop = true;
    write_current_state_data("/Users/mingzexia/Documents/Github/tracecp/data/", U, V, dec_U, dec_V, trajs_dec, index_dec, r1, r2, NUM_ITER);

  }
  write_current_state_data("/Users/mingzexia/Documents/Github/tracecp/data/", U, V, dec_U, dec_V, trajs_dec, index_dec, r1, r2, NUM_ITER);
  
  if (NUM_ITER == 0){
    // wrtie original traj when first iteration
    std::string file_path = "/Users/mingzexia/Documents/Github/tracecp/data/";
    std::string ori_traj_file = file_path + "ori_traj.bin";
    write_trajectory(trajs_ori, ori_traj_file.c_str());
    printf("Successfully write original trajs");
    //write original index
    std::string index_file = file_path + "index_ori_traj.bin";
    writefile(index_file.c_str(), index_ori.data(), index_ori.size());
    printf("Successfully write orginal index to file, total index: %ld\n",index_ori.size());
  }

  //clean up
  {
  //all_vertex_for_all_diff_traj.clear();// clear all_vertex_for_all_diff_traj
  current_diff_traj_id.clear();
  // free(dec_U);
  // free(dec_V);
  free(result);
  }

  //尝试按照比较距离来判断是不是跑偏了，然后添加到all_vertex_for_all_diff_traj
  for(size_t i=0;i < trajs_ori.size(); ++i){
  //for(size_t i=9310;i < 9312; ++i){
    auto t1 = trajs_ori[i];
    auto t2 = trajs_dec[i];
    if (t1.size() != t_config.max_length && t1.back()[0] != -1){
      // original data found cp
      if (t2.size() == t_config.max_length || t2.back()[0] == -1){
        // decompressed data not found cp
        printf("decomp not found cp\n");
        // int end_fix_index = 0;
        int end_fix_index = t1.size() - 1;
        find_fix_index_end(end_fix_index,t1,t2,grad_ori,DW,DH,t_config,all_vertex_for_all_diff_traj,i,critical_points_ori);
        current_diff_traj_id[i] = end_fix_index;
            
      }
      else{
        if (get_cell_offset(t1.back().data(), DW, DH) != get_cell_offset(t2.back().data(), DW, DH)){
          // both found cp but different
          printf("decomp found wrong cp\n");
          // int end_fix_index = 0;
          int end_fix_index = t1.size() - 1;
          find_fix_index_end(end_fix_index,t1,t2,grad_ori,DW,DH,t_config,all_vertex_for_all_diff_traj,i,critical_points_ori);
          current_diff_traj_id[i] = end_fix_index;

          
        }
      }
    }
  }
  
  //if current_diff_traj_id is the same as last_diff_traj_id, then add all vertex for all different trajectories
  // if(current_diff_traj_id == last_diff_traj_id && current_diff_traj_id.size() != 0){
  //   printf("current_diff_traj_id is the same as last_diff_traj_id\n");
  //   for (auto i:current_diff_traj_id){
  //     auto t1 = trajs_ori[i];
  //     for (size_t j = 1; j < t1.size(); ++j){ //start from 1, since 0 is cp
  //       auto p = t1[j];
  //       double current_v[2] = {0};
  //       interp2d(p.data(),current_v,grad_ori);
  //       auto offsets = vertex_for_each_RK4(p.data(), current_v,grad_ori, t_config.h,DH, DW);
  //       for (auto o:offsets){
  //         all_vertex_for_all_diff_traj.insert(o);
  //       }
  //     }
  //   }
  // }

  //if(current_diff_traj_id == last_diff_traj_id && current_diff_traj_id.size() != 0){
  if(1==1){
    //if last_diff_traj_id has 9311
    for (auto i:last_diff_traj_id){
      auto traj_ind = i.first;
      auto end_fix_index = i.second;
      printf("last_diff_traj_id: %ld, end_fix_index %d\n", traj_ind, end_fix_index);
      auto t1 = trajs_ori[traj_ind];
      auto t2 = trajs_dec[traj_ind];
      for (size_t j = 0; j < end_fix_index-1; ++j){
        auto p1 = t1[j];
        auto p2 = t2[j];
        if (p1 != p2){
          printf("point diff at index: %ld, ori: (%f, %f), dec: (%f, %f)\n", j, p1[0], p1[1], p2[0], p2[1]);
          //print the V[3][2] from dec_u and u
          if (is_upper(p1)){
            int x = floor(p1[0]);
            int y = floor(p1[1]);
            printf("ori: %f, dec: %f\n", U[y*DW+x], dec_U[y*DW+x]);
            printf("ori: %f, dec: %f\n", grad_ori(0, x, y), grad_dec(0, x, y));

            printf("ori: %f, dec: %f\n", U[(y+1)*DW+x], dec_U[(y+1)*DW+x]);
            printf("ori: %f, dec: %f\n", grad_ori(0, x, y+1), grad_dec(0, x, y+1));

            printf("ori: %f, dec: %f\n", U[(y+1)*DW+x+1], dec_U[(y+1)*DW+x+1]);
            printf("ori: %f, dec: %f\n", grad_ori(0, x+1, y+1), grad_dec(0, x+1, y+1));

          }
          else{
            int x = floor(p1[0]);
            int y = floor(p1[1]);
            printf("ori: %f, dec: %f\n", U[y*DW+x], dec_U[y*DW+x]);
            printf("ori: %f, dec: %f\n", grad_ori(0, x, y), grad_dec(0, x, y));

            printf("ori: %f, dec: %f\n", U[y*DW+x+1], dec_U[y*DW+x+1]);
            printf("ori: %f, dec: %f\n", grad_ori(0, x+1, y), grad_dec(0, x+1, y));

            printf("ori: %f, dec: %f\n", U[(y+1)*DW+x+1], dec_U[(y+1)*DW+x+1]);
            printf("ori: %f, dec: %f\n", grad_ori(0, x+1, y+1), grad_dec(0, x+1, y+1));
          }
        }
        
        // else{
        //   printf("point same at index: %ld, ori: (%f, %f), dec: (%f, %f)\n", j, p1[0], p1[1], p2[0], p2[1]);
        // }
      }
    }
    //printf("CONVERGE\n");
    //exit(0);
  }


  //std::set_difference(vertex_ori.begin(), vertex_ori.end(), vertex_dec.begin(), vertex_dec.end(), std::inserter(all_vertex_for_all_diff_traj, all_vertex_for_all_diff_traj.begin()));

  // 把所有ori的vertex加入到all_vertex_for_all_diff_traj ,即lossless all original trajectories's vertex
  // for (auto p:vertex_ori){
  //   all_vertex_for_all_diff_traj.insert(p);
  // }

  //加上saddle所在的周围3个cell的offset
  // for(auto p:critical_points_ori){
  //   auto cp = p.second;
  //   if (cp.type == SADDLE){
  //     auto surrounding_vertexs = get_surrounding_3_cells_vertexs(cp.x, DW, DH);
  //     for (auto v:surrounding_vertexs){
  //       all_vertex_for_all_diff_traj.insert(v);
  //     }
  //   }
  // }


  // for (size_t i = 0; i < trajs_ori.size(); ++i){
  //   auto cp_ori = trajs_ori[i][0];
  //   auto cp_dec = trajs_dec[i][0];
  //   auto offsets_cp_ori = get_three_offsets(cp_ori.data(), DW, DH);
  //   for (auto o:offsets_cp_ori){
  //     all_vertex_for_all_diff_traj.insert(o);
  //   }
  //   std::vector<int> index_ori;
  //   std::vector<int> index_dec;
    
  //   std::set<size_t> lossless_index_ori;
  //   std::set<size_t> lossless_index_dec;
    
  //   //这里有问题
  //   auto t1 = trajectory(cp_ori.data(), trajs_ori[i][1], t_config.h, t_config.max_length, DH, DW, critical_points_ori, grad_ori, index_ori,lossless_index_ori);
  //   auto t2 = trajectory(cp_dec.data(), trajs_dec[i][1], t_config.h, t_config.max_length, DH, DW, critical_points_dec, grad_dec, index_dec,lossless_index_dec);
  //   std::set_difference(lossless_index_ori.begin(), lossless_index_ori.end(), lossless_index_dec.begin(), lossless_index_dec.end(), std::inserter(all_vertex_for_all_diff_traj, all_vertex_for_all_diff_traj.begin()));
  
  // }
  printf("current iteration: %d, all_vertex_for_all_diff_traj size: %ld,ori_out_or_reach_limit: %d, dec_not_found_count: %d, dec_found_wrong_cp_count: %d\n", NUM_ITER, all_vertex_for_all_diff_traj.size(), ori_out_reach_limit, dec_not_found_count, dec_found_wrong_cp_count);
 
  NUM_ITER ++;

  printf("last_diff_traj_id size: %ld, current_diff_traj_id size: %ld\n", last_diff_traj_id.size(), current_diff_traj_id.size());
  //print current_diff_traj_id
  for (auto i:current_diff_traj_id){
    printf("current_diff_traj_id: %ld, end_fix_index %d\n", i.first, i.second);
  }
  last_diff_traj_id.clear();
  last_diff_traj_id = current_diff_traj_id;
  
  grad_dec.reset();
  grad_ori.reset();
  free(dec_U);
  free(dec_V);
  printf("######################\n\n");

  }
}



int main(int argc, char **argv){
  ftk::ndarray<float> grad; //grad是三纬，第一个纬度是2，代表着u或者v，第二个纬度是DH，第三个纬度是DW
  ftk::ndarray<float> grad_out;
  size_t num = 0;
  float * u = readfile<float>(argv[1], num);
  float * v = readfile<float>(argv[2], num);
  int DW = atoi(argv[3]); //1800
  int DH = atoi(argv[4]); //1200
  double max_eb = 0.01;

  fix_traj(u, v,DH, DW, max_eb, t_config);
  exit(0);

  grad.reshape({2, static_cast<unsigned long>(DW), static_cast<unsigned long>(DH)});
  refill_gradient(0, DH, DW, u, grad);
  refill_gradient(1, DH, DW, v, grad);

  //get decompressed data


    size_t result_size = 0;
    unsigned char * result = NULL;
    //result = sz_compress_cp_preserve_sos_2d_online_fp(U, V, DH,DW, result_size, false, max_eb); // use cpsz-sos
    result = sz_compress_cp_preserve_2d_online(u, v, DH,DW, result_size, false, max_eb); // use cpsz
    unsigned char * result_after_lossless = NULL;
    size_t lossless_outsize = sz_lossless_compress(ZSTD_COMPRESSOR, 3, result, result_size, &result_after_lossless);
    cout << "Compressed size(original) = " << lossless_outsize << ", ratio = " << (2*num*sizeof(float)) * 1.0/lossless_outsize << endl;
    
    //正常压缩后的数据解压
    free(result);
    size_t lossless_output = sz_lossless_decompress(ZSTD_COMPRESSOR, result_after_lossless, lossless_outsize, &result, result_size);
    float * dec_u = NULL;
    float * dec_v = NULL;
    // sz_decompress_cp_preserve_2d_online_fp<float>(result, DH,DW, dec_U, dec_V); // use cpsz-sos
    sz_decompress_cp_preserve_2d_online<float>(result, DH,DW, dec_u, dec_v); // use cpsz
    printf("verifying...\n");
    verify(u, dec_u, num);

    writefile((string(argv[1]) + ".out").c_str(), dec_u, num);
    writefile((string(argv[2]) + ".out").c_str(), dec_v, num);
    printf("written to %s.out and %s.out\n", argv[1], argv[2]);
    free(result);


 
  std::string test_flag = argv[11];
  std::string file_name;
  if (test_flag == "baseline" || test_flag == "out"){
    file_name = "out";
  }
  else{
    file_name = test_flag;
  }
  printf("flag : %s\n", test_flag.c_str());
  printf("file_name : %s\n", file_name.c_str());
  std::string outfile_u = std::string(argv[1]) + "." + file_name;
  std::string outfile_v = std::string(argv[2]) + "." + file_name;
  printf("outfile_u: %s\n", outfile_u.c_str());
  printf("outfile_v: %s\n", outfile_v.c_str());


  float * u_out = readfile<float>(outfile_u.c_str(), num);
  float * v_out = readfile<float>(outfile_v.c_str(), num);
  grad_out.reshape({2, static_cast<unsigned long>(DW), static_cast<unsigned long>(DH)});
  refill_gradient(0, DH, DW, u_out, grad_out);
  refill_gradient(1, DH, DW, v_out, grad_out);


  auto critical_points_0 =compute_critical_points(u, v, DH, DW);

  auto critical_points_out =compute_critical_points(u_out, v_out, DH, DW);

  std::vector<record_t> record;
  std::vector<record_t> record_out;

  



  	

  printf("orginal critical points size: %ld\n", critical_points_0.size());
  printf("decompressed critical points size: %ld\n", critical_points_out.size());
  //print number of saddle points
  int global_count_ori = 0;
  for (auto p:critical_points_0){
    auto cp = p.second;
    if (cp.type == SADDLE){
      global_count ++;
    }
  }
  printf("original saddle points size: %zu\n", global_count);
  int global_count_out = 0;
  for (auto p:critical_points_out){
    auto cp = p.second;
    if (cp.type == SADDLE){
      global_count_out ++;
    }
  }
  printf("decompressed saddle points size: %d\n", global_count_out);



  // fix_traj(u, v,DH, DW, max_eb, t_config);
  // exit(0);

  // need to free dec_u,dec_v,u_out,v_out somewhere

  auto it = critical_points_0.begin();
  int sad_count = 0;
  while (it != critical_points_0.end()) {
    auto cp = it->second;
    if (cp.type == SADDLE) sad_count ++;
    it ++;
  
  }
  printf("original saddle points size: %d\n", sad_count);
  it = critical_points_out.begin();
  sad_count = 0;
  while (it != critical_points_out.end()) {
    auto cp = it->second;
    if (cp.type == SADDLE) sad_count ++;
    it ++;
  
  }
  printf("decompressed saddle points size: %d\n", sad_count);

  free(u);
  free(v);
  free(u_out);
  free(v_out);

  // now do rk4

  int num_steps = 2000;
  double h = atof(argv[5]);
  // double h = 0.1;
  double eps = atof(argv[6]);
  // double eps = 0.01;
  int count_reach_limit = 0;
  int count_found = 0;
  int count_not_found = 0;
  int count_out_bound = 0;
  // int *index = (int *) malloc(sizeof(int)*sad_count*4);
  // int *index_pos = index;
  std::vector<int> myindex;
  std::vector<int> myindex_out;
  printf("current setting: num_steps: %d, h: %f, eps: %f\n", num_steps, h, eps);
  // need record each size of tracepoint
  // printf("creating tracepoints size: %ld,%d \n", saddle_points_0.size(), num_steps);
  std::vector<std::vector<std::array<double, 2>>> tracepoints;
  std::vector<std::vector<std::array<double, 2>>> tracepoints_out;
  std::vector<std::vector<size_t>> traj_cells; //list of list
  std::vector<std::vector<size_t>> traj_cells_out;
  // tracepoints.reserve(critical_points_0.size()*4);
  // tracepoints_out.reserve(critical_points_out.size()*4);
  std::set<size_t> lossless_index;
  std::set<size_t> lossless_index_out;

 auto start = std::chrono::high_resolution_clock::now();
  for(const auto& p:critical_points_0){
    auto cp = p.second;
    if (cp.type == SADDLE){
      global_count ++;
      std::vector<std::array<double,2>> X_all_direction;  
      X_all_direction.push_back({cp.x[0] + eps*cp.eig_vec[0][0], cp.x[1] + eps*cp.eig_vec[0][1]});
      X_all_direction.push_back({cp.x[0] - eps*cp.eig_vec[0][0], cp.x[1] - eps*cp.eig_vec[0][1]});
      X_all_direction.push_back({cp.x[0] + eps*cp.eig_vec[1][0], cp.x[1] + eps*cp.eig_vec[1][1]});
      X_all_direction.push_back({cp.x[0] - eps*cp.eig_vec[1][0], cp.x[1] - eps*cp.eig_vec[1][1]});                              
      double lambda[3];
      double values[2];
      std::vector<std::vector<double>> config;
      config.push_back({cp.eig_vec[0][0], cp.eig_vec[0][1],1});
      config.push_back({cp.eig_vec[0][0], cp.eig_vec[0][1],1});
      config.push_back({cp.eig_vec[1][0], cp.eig_vec[1][1],-1});
      config.push_back({cp.eig_vec[1][0], cp.eig_vec[1][1],-1});
      for (int i = 0; i < 4; i ++) {
        std::array<double, 2> X_start;
        std::vector<std::array<double, 2>> result_return;
        X_start = X_all_direction[i];
        //check if inside
        if (inside(X_start,DH, DW)){
          //printf("processing (%f,%f)\n", X_start[0], X_start[1]);
          if(i == 0 || i ==1){ 
            result_return = trajectory(cp.x,X_start, h,DH,DW, critical_points_0,grad,myindex,config[i],record,lossless_index);
            //坑:边界
            //check if result_return contains boundary
            tracepoints.push_back(result_return);
            std::vector<size_t> cells;
            for (auto r:result_return){
              //每一个坐标转换成cell index（-1，-1）表示出界
              if (r[0] != -1 && r[1] != -1){
                cells.push_back(get_cell_offset(r.data(), DW, DH));
              }
              else break;
            }
            traj_cells.push_back(cells);
          }
          else{
            result_return = trajectory(cp.x,X_start, -h,DH,DW, critical_points_0,grad,myindex,config[i],record,lossless_index);
            tracepoints.push_back(result_return);
            std::vector<size_t> cells;
            for (auto r:result_return){
              //每一个坐标转换成cell index（-1，-1）表示出界
              if (r[0] != -1 && r[1] != -1){
                cells.push_back(get_cell_offset(r.data(), DW, DH));
              }
              else break;
            }
            traj_cells.push_back(cells);
        //printf("tracepoints size: %ld\n", result_return.size());
          }
        
        {
        if (floor(cp.x[0]) == 1417 && floor(cp.x[1]) == 117){
          printf("original cp: (%f, %f)\n", cp.x[0], cp.x[1]);
          printf("eig_vec: (%f, %f), (%f, %f)\n", cp.eig_vec[0][0], cp.eig_vec[0][1], cp.eig_vec[1][0], cp.eig_vec[1][1]);
          printf("eig_val: %f + %fi, %f + %fi\n", cp.eig[0].real(), cp.eig[0].imag(), cp.eig[1].real(), cp.eig[1].imag());
          printf("Jacobi: (%f, %f),\n (%f, %f)\n", cp.Jac[0][0], cp.Jac[0][1], cp.Jac[1][0], cp.Jac[1][1]);
          printf("X: (%f, %f)\n (%f,%f)\n (%f,%f)\n", cp.X[0][0], cp.X[0][1], cp.X[1][0], cp.X[1][1], cp.X[2][0], cp.X[2][1]);
          printf("V: (%f, %f)\n (%f,%f)\n (%f,%f)\n", cp.V[0][0], cp.V[0][1], cp.V[1][0], cp.V[1][1], cp.V[2][0], cp.V[2][1]);


          printf("X_start: (%f, %f)\n", X_start[0], X_start[1]);
          }
        }

        }
      }
    }
  }

  printf("orginal Done..\n");

  // for decompressed data  

  if (test_flag == "test"){
    for(const auto& p:critical_points_out){
      auto cp = p.second;
      if (cp.type == SADDLE){
        
        global_count ++;
        std::vector<std::array<double,2>> X_all_direction;  
        X_all_direction.push_back({cp.x[0] + eps*cp.eig_vec[0][0], cp.x[1] + eps*cp.eig_vec[0][1]});
        X_all_direction.push_back({cp.x[0] - eps*cp.eig_vec[0][0], cp.x[1] - eps*cp.eig_vec[0][1]});
        X_all_direction.push_back({cp.x[0] + eps*cp.eig_vec[1][0], cp.x[1] + eps*cp.eig_vec[1][1]});
        X_all_direction.push_back({cp.x[0] - eps*cp.eig_vec[1][0], cp.x[1] - eps*cp.eig_vec[1][1]});                              
        double lambda[3];
        double values[2];
        std::vector<std::vector<double>> config;
        config.push_back({cp.eig_vec[0][0], cp.eig_vec[0][1],1});
        config.push_back({cp.eig_vec[0][0], cp.eig_vec[0][1],1});
        config.push_back({cp.eig_vec[1][0], cp.eig_vec[1][1],-1});
        config.push_back({cp.eig_vec[1][0], cp.eig_vec[1][1],-1});
        for (int i = 0; i < 4; i ++) {
          std::array<double, 2> X_start;
          std::vector<std::array<double, 2>> result_return;
          X_start = X_all_direction[i];
          //check if inside
          if (inside(X_start,DH, DW)){
            if(i == 0 || i ==1){
              result_return = trajectory(cp.x,X_start, h,DH,DW, critical_points_out,grad_out,myindex_out,config[i],record_out,lossless_index_out);
              tracepoints_out.push_back(result_return);
              std::vector<size_t> cells;
              for (auto r:result_return){
                //每一个坐标转换成cell index（-1，-1）表示出界
                if (r[0] != -1 && r[1] != -1){
                  cells.push_back(get_cell_offset(r.data(), DW, DH));
                }
                else break;
              }
              traj_cells_out.push_back(cells);
            }
            else{
              result_return = trajectory(cp.x,X_start, -h,DH,DW, critical_points_out,grad_out,myindex_out,config[i],record_out,lossless_index_out);
              tracepoints_out.push_back(result_return);
              std::vector<size_t> cells;
              for (auto r:result_return){
                //每一个坐标转换成cell index（-1，-1）表示出界
                if (r[0] != -1 && r[1] != -1){
                  cells.push_back(get_cell_offset(r.data(), DW, DH));
                }
                else break;
              }
              traj_cells_out.push_back(cells);
            }
          //printf("tracepoints size: %ld\n", result_return.size());
          }
        }
      }
    }
    printf("tracepoints size: %ld\n", tracepoints.size());
    printf("tracepoints_out size: %ld\n", tracepoints_out.size());
    //TODO
    //.....
    //write files
    std::string filename = argv[7]; // trajectory filename
    std::string filename2 = argv[8]; // index corresponding to trajectory
    std::string filename3 = argv[9]; //record start & end filename
    std::string filename4 = argv[10]; //record cp filename

    // if no filename provided, no write file
    if (filename.empty() && filename2.empty() && filename3.empty()){
        printf("missing parameter\n");
        exit(0);
    }

    write_trajectory(tracepoints, filename);
    printf("Successfully write orginal trajectory to file, total trajectory: %ld\n",tracepoints.size());
    write_trajectory(tracepoints_out, filename + ".test");
    printf("Successfully write decompressed trajectory to file, total trajectory: %ld\n",tracepoints_out.size());

    //write index to file
    writefile(filename2.c_str(), myindex.data(), myindex.size());
    printf("Successfully write orginal index to file, total index: %ld\n",myindex.size());
    writefile((filename2 + ".test").c_str(), myindex_out.data(), myindex_out.size());
    printf("Successfully write decompressed index to file, total index: %ld\n",myindex_out.size());

    //write record to file
    writeRecordsToBinaryFile(record, filename3);

    //write critical points to file
    record_criticalpoints(filename4, critical_points_0);

  }
  else if (test_flag == "out"){ // .test: only cpsz, no modify
    for(const auto& p:critical_points_out){
      auto cp = p.second;
      if (cp.type == SADDLE){
        global_count ++;
        std::vector<std::array<double,2>> X_all_direction;  
        X_all_direction.push_back({cp.x[0] + eps*cp.eig_vec[0][0], cp.x[1] + eps*cp.eig_vec[0][1]});
        X_all_direction.push_back({cp.x[0] - eps*cp.eig_vec[0][0], cp.x[1] - eps*cp.eig_vec[0][1]});
        X_all_direction.push_back({cp.x[0] + eps*cp.eig_vec[1][0], cp.x[1] + eps*cp.eig_vec[1][1]});
        X_all_direction.push_back({cp.x[0] - eps*cp.eig_vec[1][0], cp.x[1] - eps*cp.eig_vec[1][1]});                              
        double lambda[3];
        double values[2];
        std::vector<std::vector<double>> config;
        config.push_back({cp.eig_vec[0][0], cp.eig_vec[0][1],1});
        config.push_back({cp.eig_vec[0][0], cp.eig_vec[0][1],1});
        config.push_back({cp.eig_vec[1][0], cp.eig_vec[1][1],-1});
        config.push_back({cp.eig_vec[1][0], cp.eig_vec[1][1],-1});
        for (int i = 0; i < 4; i ++) {
          std::array<double, 2> X_start;
          std::vector<std::array<double, 2>> result_return;
          X_start = X_all_direction[i];
          //check if inside
          if (inside(X_start,DH, DW)){
            if(i == 0 || i ==1){
              result_return = trajectory(cp.x,X_start, h,DH,DW, critical_points_out,grad_out,myindex_out,config[i],record_out,lossless_index_out);
              tracepoints_out.push_back(result_return);
              std::vector<size_t> cells;
              for (auto r:result_return){
                //每一个坐标转换成cell index（-1，-1）表示出界
                if (r[0] != -1 && r[1] != -1){
                  cells.push_back(get_cell_offset(r.data(), DW, DH));
                }
                else break;
              }
              traj_cells_out.push_back(cells);
            }
            else{
              result_return = trajectory(cp.x,X_start, -h,DH,DW, critical_points_out,grad_out,myindex_out,config[i],record_out,lossless_index_out);
              tracepoints_out.push_back(result_return);
              std::vector<size_t> cells;
              for (auto r:result_return){
                //每一个坐标转换成cell index（-1，-1）表示出界
                if (r[0] != -1 && r[1] != -1){
                  cells.push_back(get_cell_offset(r.data(), DW, DH));
                }
                else break;
              }
              traj_cells_out.push_back(cells);
            }
          //printf("tracepoints size: %ld\n", result_return.size());

          {
          if (floor(cp.x[0]) == 1417 && floor(cp.x[1]) == 117){
            printf("decompressed cp: (%f, %f)\n", cp.x[0], cp.x[1]);
            printf("eig_vec: (%f, %f), (%f, %f)\n", cp.eig_vec[0][0], cp.eig_vec[0][1], cp.eig_vec[1][0], cp.eig_vec[1][1]);
            printf("eig_val: %f + %fi, %f + %fi\n", cp.eig[0].real(), cp.eig[0].imag(), cp.eig[1].real(), cp.eig[1].imag());
            printf("Jacobi: (%f, %f),\n (%f, %f)\n", cp.Jac[0][0], cp.Jac[0][1], cp.Jac[1][0], cp.Jac[1][1]);
            printf("X_start: (%f, %f)\n", X_start[0], X_start[1]);  
            printf("X: (%f, %f)\n (%f,%f)\n (%f,%f)\n", cp.X[0][0], cp.X[0][1], cp.X[1][0], cp.X[1][1], cp.X[2][0], cp.X[2][1]);
            printf("V: (%f, %f)\n (%f,%f)\n (%f,%f)\n", cp.V[0][0], cp.V[0][1], cp.V[1][0], cp.V[1][1], cp.V[2][0], cp.V[2][1]);
            }
          }

          }
        }
      }
    }
    printf("tracepoints size: %ld\n", tracepoints.size());
    printf("tracepoints_out size: %ld\n", tracepoints_out.size());
    
    // //test if first and second point are the same
    // for (size_t i = 0; i < tracepoints.size(); i++){
    //   auto t1 = tracepoints[i];
    //   auto t2 = tracepoints_out[i];
    //   if (t1[0][0] != t2[0][0] || t1[0][1] != t2[0][1]){
    //     printf("first point not the same\n");
        
    //   }
    //   if (t1[1][0] != t2[1][0] || t1[1][1] != t2[1][1]){
    //     printf("second point not the same\n");
    //   }
    // }
    // exit(0);

    // write files
    std::string filename = argv[7]; // trajectory filename
    std::string filename2 = argv[8]; // index corresponding to trajectory
    std::string filename3 = argv[9]; //record start & end filename
    std::string filename4 = argv[10]; //record cp filename

    // if no filename provided, no write file
    if (filename.empty() && filename2.empty() && filename3.empty()){
        printf("missing parameter\n");
        exit(0);
    }

    write_trajectory(tracepoints, filename);
    printf("Successfully write orginal trajectory to file, total trajectory: %ld\n",tracepoints.size());
    write_trajectory(tracepoints_out, filename + ".out");
    printf("Successfully write decompressed trajectory to file, total trajectory: %ld\n",tracepoints_out.size());

    //write index to file
    writefile(filename2.c_str(), myindex.data(), myindex.size());
    printf("Successfully write orginal index to file, total index: %ld\n",myindex.size());
    writefile((filename2 + ".out").c_str(), myindex_out.data(), myindex_out.size());
    printf("Successfully write decompressed index to file, total index: %ld\n",myindex_out.size());

    // //write record to file
    // writeRecordsToBinaryFile(record, filename3);

    // //write critical points to file
    // record_criticalpoints(filename4, critical_points_0);

  }
  else if (test_flag == "baseline"){
    std::set<size_t> diff_simplics;
    std::set<std::array<size_t,2>> diff_coords;
    std::set<size_t> diff_offset_index;
    std::vector<size_t> diff_traj_index;
    size_t * index_ptr = (size_t *) malloc(sizeof(size_t)*lossless_index.size());
    if (index_ptr) {
      // 复制 set 中的数据到动态分配的数组中
      size_t i = 0;
      for (const size_t& element : lossless_index) {
          index_ptr[i++] = element;
      }
    }
    writefile("../small_data/index_need_lossless.bin", index_ptr, lossless_index.size());
    printf("index_need_lossless.bin written, size: %ld\n", lossless_index.size());
    free(index_ptr);
    writeVectorOfVector(traj_cells, "../small_data/traj_cells.bin");

  }
  else {
    printf("wrong flag\n");
    exit(0);
  }

  // do some tests:
  if(test_flag == "test" || test_flag == "out"){
    std::vector<std::vector<std::array<double, 2>>> ori_found_dec_found_diff_ORI;
    std::vector<std::vector<std::array<double, 2>>> ori_found_dec_found_diff_DEC;
    std::vector<std::vector<std::array<double, 2>>> ori_found_dec_not_found_ORI;
    std::vector<std::vector<std::array<double, 2>>> ori_found_dec_not_found_DEC;
    std::vector<int> index_ori_found_dec_found_diff_ORI;
    std::vector<int> index_ori_found_dec_found_diff_DEC;
    std::vector<int> index_ori_found_dec_not_found_ORI;
    std::vector<int> index_ori_found_dec_not_found_DEC;
    check_and_write_two_traj_detail(tracepoints, tracepoints_out,myindex,myindex_out, critical_points_0, critical_points_out, DW, DH,
    ori_found_dec_found_diff_ORI, ori_found_dec_found_diff_DEC, ori_found_dec_not_found_ORI, ori_found_dec_not_found_DEC,
    index_ori_found_dec_found_diff_ORI, index_ori_found_dec_found_diff_DEC, index_ori_found_dec_not_found_ORI, index_ori_found_dec_not_found_DEC,t_config);
    //write to file
    write_trajectory(ori_found_dec_found_diff_ORI, "../small_data/ori_found_dec_found_diff_ORI.bin");
    write_trajectory(ori_found_dec_found_diff_DEC, "../small_data/ori_found_dec_found_diff_DEC.bin");
    write_trajectory(ori_found_dec_not_found_ORI, "../small_data/ori_found_dec_not_found_ORI.bin");
    write_trajectory(ori_found_dec_not_found_DEC, "../small_data/ori_found_dec_not_found_DEC.bin");
    writefile("../small_data/index_ori_found_dec_found_diff_ORI.bin", index_ori_found_dec_found_diff_ORI.data(), index_ori_found_dec_found_diff_ORI.size());
    writefile("../small_data/index_ori_found_dec_found_diff_DEC.bin", index_ori_found_dec_found_diff_DEC.data(), index_ori_found_dec_found_diff_DEC.size());
    writefile("../small_data/index_ori_found_dec_not_found_ORI.bin", index_ori_found_dec_not_found_ORI.data(), index_ori_found_dec_not_found_ORI.size());
    writefile("../small_data/index_ori_found_dec_not_found_DEC.bin", index_ori_found_dec_not_found_DEC.data(), index_ori_found_dec_not_found_DEC.size());
    // write_trajectory(ori_found_dec_not_ORI, "/Users/mingzexia/Documents/Github/tracecp/small_data/ori_found_dec_not_ORI.bin");
    // write_trajectory(ori_found_dec_not_DEC, "/Users/mingzexia/Documents/Github/tracecp/small_data/ori_found_dec_not_DEC.bin");
    // writefile("/Users/mingzexia/Documents/Github/tracecp/small_data/index_ori_found_dec_not_ORI.bin", index_ori_found_dec_not_ORI.data(), index_ori_found_dec_not_ORI.size());
    // writefile("/Users/mingzexia/Documents/Github/tracecp/small_data/index_ori_found_dec_not_DEC.bin", index_ori_found_dec_not_DEC.data(), index_ori_found_dec_not_DEC.size());

    // now the difference between two trajectories are saved in two files
    check_two_traj(tracepoints, tracepoints_out, critical_points_0, critical_points_out, DW, DH);
    printf("saved diff trajectories, 2 trajs and 2 index files\n");



    // {
    // //EDR distance
    // double threshold = 1;
    // double totoal_distance = 0;
    // for (int i = 0; i < tracepoints.size(); i++){
    //   auto traj1 = tracepoints[i];
    //   auto traj2 = tracepoints_out[i];
    //   // remove (-1,-1) points
    //   traj1.erase(std::remove_if(traj1.begin(), traj1.end(), [](std::array<double, 2> p){return p[0] == -1 && p[1] == -1;}), traj1.end());
    //   traj2.erase(std::remove_if(traj2.begin(), traj2.end(), [](std::array<double, 2> p){return p[0] == -1 && p[1] == -1;}), traj2.end());
    //   double current_distance = calculateEDR2D(traj1, traj2, threshold);
    //   printf("trajectory %d, EDR: %f\n", i, current_distance);
    //   totoal_distance += current_distance;
      
    // }
    // printf("total EDR: %f\n", totoal_distance);
    // }

    

  }

/*

  if (test_flag == "test" || test_flag == "out"){
    for(const auto& p:critical_points_out){
      auto cp = p.second;
      if (cp.type == SADDLE){
        global_count ++;
        std::vector<std::array<double,2>> X_all_direction;  
        X_all_direction.push_back({cp.x[0] + eps*cp.eig_vec[0][0], cp.x[1] + eps*cp.eig_vec[0][1]});
        X_all_direction.push_back({cp.x[0] - eps*cp.eig_vec[0][0], cp.x[1] - eps*cp.eig_vec[0][1]});
        X_all_direction.push_back({cp.x[0] + eps*cp.eig_vec[1][0], cp.x[1] + eps*cp.eig_vec[1][1]});
        X_all_direction.push_back({cp.x[0] - eps*cp.eig_vec[1][0], cp.x[1] - eps*cp.eig_vec[1][1]});                              
        double lambda[3];
        double values[2];
        std::vector<std::vector<double>> config;
        config.push_back({cp.eig_vec[0][0], cp.eig_vec[0][1],1});
        config.push_back({cp.eig_vec[0][0], cp.eig_vec[0][1],1});
        config.push_back({cp.eig_vec[1][0], cp.eig_vec[1][1],-1});
        config.push_back({cp.eig_vec[1][0], cp.eig_vec[1][1],-1});
        for (int i = 0; i < 4; i ++) {
          std::array<double, 2> X_start;
          std::vector<std::array<double, 2>> result_return;
          X_start = X_all_direction[i];
          //check if inside
          if (inside(X_start,DH, DW)){
            if(i == 0 || i ==1){
              result_return = trajectory(cp.x,X_start, h,DH,DW, critical_points_out,grad_out,myindex_out,config[i],record_out,lossless_index_out);
              tracepoints_out.push_back(result_return);
              std::vector<size_t> cells;
              for (auto r:result_return){
                //每一个坐标转换成cell index（-1，-1）表示出界
                if (r[0] != -1 && r[1] != -1){
                  cells.push_back(get_cell_offset(r.data(), DW, DH));
                }
                else break;
              }
              traj_cells_out.push_back(cells);
            }
            else{
              result_return = trajectory(cp.x,X_start, -h,DH,DW, critical_points_out,grad_out,myindex_out,config[i],record_out,lossless_index_out);
              tracepoints_out.push_back(result_return);
              std::vector<size_t> cells;
              for (auto r:result_return){
                //每一个坐标转换成cell index（-1，-1）表示出界
                if (r[0] != -1 && r[1] != -1){
                  cells.push_back(get_cell_offset(r.data(), DW, DH));
                }
                else break;
              }
              traj_cells_out.push_back(cells);
            }
          //printf("tracepoints size: %ld\n", result_return.size());
          }
        }
      }
    }

    check_and_write_two_traj_detail(tracepoints, tracepoints_out,myindex,myindex_out, critical_points_0, critical_points_out, DW, DH);
    printf("saved diff trajectories, 2 files\n");

  }

  printf("tracepoints size: %ld\n", tracepoints.size());
  printf("tracepoints_out size: %ld\n", tracepoints_out.size());


  std::set<size_t> diff_simplics;
  std::set<std::array<size_t,2>> diff_coords;
  std::set<size_t> diff_offset_index;
  std::vector<size_t> diff_traj_index;
  if (test_flag == "baseline"){
    size_t * index_ptr = (size_t *) malloc(sizeof(size_t)*lossless_index.size());
    if (index_ptr) {
      // 复制 set 中的数据到动态分配的数组中
      size_t i = 0;
      for (const size_t& element : lossless_index) {
          index_ptr[i++] = element;
      }
    }
    writefile("../small_data/index_need_lossless.bin", index_ptr, lossless_index.size());
    printf("index_need_lossless.bin written, size: %ld\n", lossless_index.size());
    free(index_ptr);

    writeVectorOfVector(traj_cells, "../small_data/traj_cells.bin");
    
  }
  else if (test_flag == "test" || test_flag == "out"){

    //把.out文件的trajectory经过的cell index写入文件
    size_t * index_ptr = (size_t *) malloc(sizeof(size_t)*lossless_index_out.size());
    if (index_ptr) {
      // 复制 set 中的数据到动态分配的数组中
      size_t i = 0;
      for (const size_t& element : lossless_index_out) {
          index_ptr[i++] = element;
      }
    }
    writefile("../small_data/index_need_lossless.bin.out", index_ptr, lossless_index_out.size());
    printf("index_need_lossless.bin.out written, size: %ld\n", lossless_index_out.size());
    free(index_ptr);

    writeVectorOfVector(traj_cells_out, "../small_data/traj_cells_out.bin");
    
    if (areTrajsEqual(tracepoints, tracepoints_out)){
      printf("Two Trajs are equal\n");
    }
    else{
      printf("Two Trajs are not equal\n");
      printf("lossless_index size: %ld, lossless_index_out size: %ld\n", lossless_index.size(), lossless_index_out.size());
    }
    //check_start_end(tracepoints, tracepoints_out, DW, DH);
    check_two_traj(tracepoints, tracepoints_out, critical_points_0, critical_points_out, DW, DH);
    check_two_traj_start_end_cell(tracepoints, tracepoints_out, critical_points_0, critical_points_out,DW, DH);
    printf("test done\n");
  
  }

  //exit(0);
  
  //writing some files
  if (test_flag == "out" || test_flag == "test"){
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

  printf("record size: %ld\n", record.size());
  

  std::string filename = argv[7];
  std::string filename2 = argv[8];
  std::string filename3 = argv[9];
  std::string filename4 = argv[10];

  // if no filename provided, no write file
  if (filename.empty() && filename2.empty() && filename3.empty()){
      printf("missing parameter\n");
      exit(0);
  }

  //write tracepoints to file
  // std::ofstream file(filename, std::ios::out | std::ios::binary);
  //   if (!file.is_open()) {
  //       std::cerr << "Failed to open file for writing." << std::endl;
  //       return 1;
  //   }
  //   int count = 0;
  //   std::cout << tracepoints.size() <<std::endl;
  //   for (const auto& row : tracepoints) {
  //       for (const auto& point : row) {
  //           file.write(reinterpret_cast<const char*>(point.data()), point.size() * sizeof(double));
  //           count ++;
  //       }
  //   }
  //   printf("Successfully write trajectory to file, total points: %d\n",count);
  //   file.close();
    write_trajectory(tracepoints, filename);
    printf("Successfully write orginal trajectory to file, total trajectory: %ld\n",tracepoints.size());
    write_trajectory(tracepoints_out, filename + ".out");
    printf("Successfully write decompressed trajectory to file, total trajectory: %ld\n",tracepoints_out.size());

    //write index to file
    writefile(filename2.c_str(), myindex.data(), myindex.size());
    printf("Successfully write orginal index to file, total index: %ld\n",myindex.size());
    writefile((filename2 + ".out").c_str(), myindex_out.data(), myindex_out.size());
    printf("Successfully write decompressed index to file, total index: %ld\n",myindex_out.size());

    //write record to file
    writeRecordsToBinaryFile(record, filename3);

    //write critical points to file
    //std::string cp_prefix = "../data/position";
    record_criticalpoints(filename4, critical_points_0);

    // write all the pos that need to lossless compress
  }
  */
}





