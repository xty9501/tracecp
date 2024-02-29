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
#include <chrono>


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

double vector_field_resolution = std::numeric_limits<double>::max();
uint64_t vector_field_scaling_factor = 1;
// int DW = 128, DH = 128;// the dimensionality of the data is DW*DH
ftk::ndarray<double> grad; //grad是三纬，第一个纬度是2，代表着u或者v，第二个纬度是DH，第三个纬度是DW
ftk::ndarray<double> grad_out;
ftk::simplicial_regular_mesh m(2);
std::mutex mutex;

size_t global_count = 0;


typedef struct critical_point_t{
  double x[2];
  double eig_vec[2][2];
  double V[3][2];
  double X[3][2];
  // double mu[3];
  int type;
  size_t simplex_id;
  critical_point_t(double* x_, double eig_v[2][2], double V_[3][2],double X_[3][2], int t_, size_t simplex_id_){
    x[0] = x_[0];
    x[1] = x_[1];
    eig_vec[0][0] = eig_v[0][0];
    eig_vec[0][1] = eig_v[0][1];
    eig_vec[1][0] = eig_v[1][0];
    eig_vec[1][1] = eig_v[1][1];
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 2; j++) {
        V[i][j] = V_[i][j];
      }
    }
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 2; j++) {
        X[i][j] = X_[i][j];
      }
    }
    // for (int i = 0; i < 3; i++) {
    //   mu[i] = mu_[i];
    // }
    type = t_;
    simplex_id = simplex_id_;
  }
  critical_point_t(){}
}critical_point_t;



// std::unordered_map<int, critical_point_t> critical_points;
 
#define DEFAULT_EB 1
#define SINGULAR 0
#define ATTRACTING 1 // 2 real negative eigenvalues
#define REPELLING 2 // 2 real positive eigenvalues
#define SADDLE 3// 1 real negative and 1 real positive
#define ATTRACTING_FOCUS 4 // complex with negative real
#define REPELLING_FOCUS 5 // complex with positive real
#define CENTER 6 // complex with 0 real

std::string get_critical_point_type_string(int type){
  switch(type){
    case 0:
      return "SINGULAR";
    case 1:
      return "ATTRACTING";
    case 2:
      return "REPELLING";
    case 3:
      return "SADDLE";
    case 4:
      return "ATTRACTING_FOCUS";
    case 5:
      return "REPELLING_FOCUS";
    case 6:
      return "CENTER";
    default:
      return "INVALID";
  }
}


void refill_gradient(int id,const int DH,const int DW, const float* grad_tmp, ftk::ndarray<double>& grad){
  const float * grad_tmp_pos = grad_tmp;
  for (int i = 0; i < DH; i ++) {
    for (int j = 0; j < DW; j ++) {
      grad(id, j, i) = *(grad_tmp_pos ++);
    }
  }
}



template<typename T_fp>
static void 
check_simplex_seq_saddle(const T_fp vf[3][2], const double v[3][2], const double X[3][2], const int indices[3], int i, int j, int simplex_id, std::unordered_map<int, critical_point_t>& critical_points){
  int sos = 1;
  double mu[3]; // check intersection
  double cond;
  if (sos == 1){
    for(int i=0; i<3; i++){ //skip if any of the vertex is 0 //
    if((v[i][0] == 0) && (v[i][1] == 0)){ //
      return; //
      } //
    } //
    // robust critical point test
    bool succ = ftk::robust_critical_point_in_simplex2(vf, indices);
    if (!succ) return;

    bool succ2 = ftk::inverse_lerp_s2v2(v, mu, &cond);
    if (!succ2) ftk::clamp_barycentric<3>(mu);
  }
  else{
    for(int i=0; i<3; i++){ //skip if any of the vertex is 0 //
    if((v[i][0] == 0) && (v[i][1] == 0)){ //
      return; //
      } //
    } //
    bool succ2 = ftk::inverse_lerp_s2v2(v, mu, &cond);
    if (!succ2) return;
  }
  double x[2]; // position
  ftk::lerp_s2v2(X, mu, x);
  double J[2][2]; // jacobian
  ftk::jacobian_2dsimplex2(X, v, J);  
  int cp_type = 0;
  std::complex<double> eig[2];
  double delta = ftk::solve_eigenvalues2x2(J, eig);
  // if(fabs(delta) < std::numeric_limits<double>::epsilon())
  if (delta >= 0) { // two real roots
    if (eig[0].real() * eig[1].real() < 0) {
      cp_type = SADDLE;
    } else if (eig[0].real() < 0) {
      cp_type = ATTRACTING;
    }
    else if (eig[0].real() > 0){
      cp_type = REPELLING;
    }
    else cp_type = SINGULAR;
  } else { // two conjugate roots
    if (eig[0].real() < 0) {
      cp_type = ATTRACTING_FOCUS;
    } else if (eig[0].real() > 0) {
      cp_type = REPELLING_FOCUS;
    } else 
      cp_type = CENTER;
  }
  critical_point_t cp;
  // critical_point_t cp(x, eig_vec,v,X, cp_type);
  cp.x[0] = j + x[0]; cp.x[1] = i + x[1];
  double eig_vec[2][2]={0};
  double eig_r[2];
  eig_r[0] = eig[0].real(), eig_r[1] = eig[1].real();
  ftk::solve_eigenvectors2x2(J, 2, eig_r, eig_vec);
  cp.eig_vec[0][0] = eig_vec[0][0]; cp.eig_vec[0][1] = eig_vec[0][1];
  cp.eig_vec[1][0] = eig_vec[1][0]; cp.eig_vec[1][1] = eig_vec[1][1];
  cp.V[0][0] = v[0][0]; cp.V[0][1] = v[0][1];
  cp.V[1][0] = v[1][0]; cp.V[1][1] = v[1][1];
  cp.V[2][0] = v[2][0]; cp.V[2][1] = v[2][1];
  cp.X[0][0] = X[0][0]; cp.X[0][1] = X[0][1];
  cp.X[1][0] = X[1][0]; cp.X[1][1] = X[1][1];
  cp.X[2][0] = X[2][0]; cp.X[2][1] = X[2][1];
  cp.type = cp_type;
  cp.simplex_id = simplex_id;
  // cp.v = mu[0]*v[0][0] + mu[1]*v[1][0] + mu[2]*v[2][0];
  // cp.u = mu[0]*v[0][1] + mu[1]*v[1][1] + mu[2]*v[2][1];
  critical_points[simplex_id] = cp;


  
  
}

template<typename T>
std::unordered_map<int, critical_point_t>
compute_critical_points(const T * U, const T * V, int r1, int r2, uint64_t vector_field_scaling_factor){
  // check cp for all cells
  using T_fp = int64_t;
  size_t num_elements = r1*r2;
  T_fp * U_fp = (T_fp *) malloc(num_elements * sizeof(T_fp));
  T_fp * V_fp = (T_fp *) malloc(num_elements * sizeof(T_fp));
  for(int i=0; i<num_elements; i++){
    U_fp[i] = U[i]*vector_field_scaling_factor;
    V_fp[i] = V[i]*vector_field_scaling_factor;
  }
  int indices[3] = {0};
  // __int128 vf[4][3] = {0};
	double X1[3][2] = {
		{0, 0},
		{0, 1},
		{1, 1}
	};
	double X2[3][2] = {
		{0, 0},
		{1, 0},
		{1, 1}
	};
  int64_t vf[3][2] = {0};
  double v[3][2] = {0};
  std::unordered_map<int, critical_point_t> critical_points;
	for(int i=0; i<r1-1; i++){ //坑
    if(i%100==0) std::cout << i << " / " << r1-1 << std::endl;
		for(int j=0; j<r2-1; j++){
      ptrdiff_t cell_offset = 2*(i * (r2-1) + j);
			indices[0] = i*r2 + j;
			indices[1] = (i+1)*r2 + j;
			indices[2] = (i+1)*r2 + (j+1); 
			// cell index 0
			for(int p=0; p<3; p++){
				vf[p][0] = U_fp[indices[p]];
				vf[p][1] = V_fp[indices[p]];
				v[p][0] = U[indices[p]];
				v[p][1] = V[indices[p]];
			}
      
      check_simplex_seq_saddle(vf, v, X1, indices, i, j, cell_offset, critical_points);
			// cell index 1
			indices[1] = i*r2 + (j+1);
			vf[1][0] = U_fp[indices[1]], vf[1][1] = V_fp[indices[1]];
			v[1][0] = U[indices[1]], v[1][1] = V[indices[1]];			
      check_simplex_seq_saddle(vf, v, X2, indices, i, j, cell_offset + 1, critical_points);
		}
	}
  free(U_fp);
  free(V_fp);
  return critical_points; 
}



template<typename Type>
std::array<Type, 2> RK4(const Type * x, const Type * v, const Type h){
  // x is the position, v is the velocity, h is the step size
  Type k1_u = h*v[0];
  Type k1_v = h*v[1];
  Type k2_u = v[0] + h * k1_u / 2.0;
  Type k2_v = v[1] + h * k1_v / 2.0;
  Type k3_u = v[0] + h * k2_u / 2.0;
  Type k3_v = v[1] + h * k2_v / 2.0;
  Type k4_u = v[0] + h * k3_u;
  Type k4_v = v[1] + h * k3_v;
  Type u = x[0] + h * (k1_u + 2 * k2_u + 2 * k3_u + k4_u) / 6.0;
  Type v_result = x[1] + h * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0;
  std::array<Type, 2> result = {u, v_result};
  return result;
}


inline bool file_exists(const std::string& filename) {
    std::ifstream f(filename.c_str());
    return f.good();
}


template<typename Container>
bool inside(const Container& x, int DH, int DW) {
  if (x[0] <=0 || x[0] > DW-1 || x[1] <= 0 || x[1] > DH-1) return false;
  return true;
}

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

inline bool is_upper(const std::array<double, 2> x){
  double x_ex = x[0] - floor(x[0]);
  double y_ex = x[1] - floor(x[1]);
  if (y_ex >= x_ex){
    return true;
  }
  else{
    return false;
  }
}

inline int get_cell_offset(const double *x, const int DW, const int DH){
  int x0 = floor(x[0]);
  int y0 = floor(x[1]);
  int cell_offset = 2*(y0 * (DW-1) + x0);
  if (!is_upper({x[0], x[1]})){
    cell_offset += 1;
  }
  return cell_offset;
}

inline bool vaild_offset(const int offset, const int DW, const int DH){
  if (offset < 0 || offset >= 2*(DW-1)*(DH-1)){
    return false;
  }
  else{
    return true;
  }
}
inline bool vaild_offset(const std::array<double,2>& x, const int DW, const int DH){
  if(x[0] < 0 || x[0] > DW-1 || x[1] < 1 || x[1] > DH-1){
    return false;
  }
  else{
    return true;
  }

}

inline std::vector<int> get_surrounding_cell(const int cell_offset,const std::array<double,2>& x, const int DW, const int DH){
  std::vector<int> surrounding_cell;
  // 修改了这里
  if (vaild_offset(cell_offset,DW,DH)){
      surrounding_cell.push_back(cell_offset);
    }
  return surrounding_cell;
}
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

inline bool check_result(const double error, const double *v){
  if (fabs(v[0]) < error && fabs(v[1]) < error){
    //printf("v[0]: %f, v[1]: %f\n", v[0], v[1]);
    return true;
  }
  else{
    return false;
  }
}


inline int printsign(double value){
  if (value > 0){
    return 1;
  }
  else if (value < 0){
    return -1;
  }
  else{
    return 0;
  }
} 

template<typename T>
std::array<size_t, 3> get_three_offsets(const T& x, const int DW, const int DH){
  // vertex offset
  size_t x0 = floor(x[0]); 
  size_t y0 = floor(x[1]);
  std::array<size_t, 3> result;
  if (is_upper({x[0], x[1]})){
    result[0] = y0 * DW + x0;
    result[1] = y0 * DW + x0 + DW;
    result[2] = y0 * DW + x0 + DW + 1;
  }
  else{
    result[0] = y0 * DW + x0;
    result[1] = y0 * DW + x0 + 1;
    result[2] = y0 * DW + x0 + DW + 1;
  }
  return result;

}

template std::array<size_t, 3> get_three_offsets(const std::array<double, 2>& x, const int DW, const int DH);


template<typename Type>
std::array<Type, 2> newRK4(const Type * x, const Type * v, const ftk::ndarray<double> &data,  Type h, const int DH, const int DW,std::set<size_t>& lossless_index) {
  // x and y are positions, and h is the step size
  double rk1[2] = {0};
  const double p1[] = {x[0], x[1]};

  auto coords = get_three_offsets(x, DW, DH);
  for (auto offset:coords){
    lossless_index.insert(offset);
  }

  if(!inside(p1, DH, DW)){
    //return std::array<Type, 2>{x[0], x[1]};
    return std::array<Type, 2>{-1, -1};
  }
  interp2d(p1, rk1,data);
  coords = get_three_offsets(p1, DW, DH);
  for (auto offset:coords){
    lossless_index.insert(offset);
  }
  
  double rk2[2] = {0};
  const double p2[] = {x[0] + 0.5 * h * rk1[0], x[1] + 0.5 * h * rk1[1]};
  if (!inside(p2, DH, DW)){
    //return std::array<Type, 2>{p1[0], p1[1]};
    return std::array<Type, 2>{-1, -1};
  }
  interp2d(p2, rk2,data);
  coords = get_three_offsets(p2, DW, DH);
  for (auto offset:coords){
    lossless_index.insert(offset);
  }
  
  double rk3[2] = {0};
  const double p3[] = {x[0] + 0.5 * h * rk2[0], x[1] + 0.5 * h * rk2[1]};
  if (!inside(p3, DH, DW)){
    //return std::array<Type, 2>{p2[0], p2[1]};
    return std::array<Type, 2>{-1, -1};
  }
  interp2d(p3, rk3,data);
  coords = get_three_offsets(p3, DW, DH);
  for (auto offset:coords){
    lossless_index.insert(offset);
  }
  
  double rk4[2] = {0};
  const double p4[] = {x[0] + h * rk3[0], x[1] + h * rk3[1]};
  if (!inside(p4, DH, DW)){
    //return std::array<Type, 2>{p3[0], p3[1]};
    return std::array<Type, 2>{-1, -1};
  }
  interp2d(p4, rk4,data);
  coords = get_three_offsets(p4, DW, DH);
  for (auto offset:coords){
    lossless_index.insert(offset);
  }
  
  Type next_x = x[0] + h * (rk1[0] + 2 * rk2[0] + 2 * rk3[0] + rk4[0]) / 6;
  Type next_y = x[1] + h * (rk1[1] + 2 * rk2[1] + 2 * rk3[1] + rk4[1]) / 6;
  // printf("shift: (%f, %f)\n", next_x - x[0], next_y - x[1]);
  // printf("coefficients: (%f,%f)\n",(rk1[0] + 2 * rk2[0] + 2 * rk3[0] + rk4[0]) / 6, (rk1[1] + 2 * rk2[1] + 2 * rk3[1] + rk4[1]) / 6);
  // printf("current h sign: %d\n", printsign(h));
  // printf("sign of coefficients x (%d,%d,%d,%d)\n", printsign(rk1[0]), printsign(rk2[0]), printsign(rk3[0]), printsign(rk4[0]));
  // printf("sign of coefficients y (%d,%d,%d,%d)\n", printsign(rk1[1]), printsign(rk2[1]), printsign(rk3[1]), printsign(rk4[1]));
  if (!inside(std::array<Type, 2>{next_x, next_y}, DH, DW)){
    //return std::array<Type, 2>{p4[0], p4[1]};
    return std::array<Type, 2>{-1, -1};
  }
  std::array<Type, 2> result = {next_x, next_y};
  coords = get_three_offsets(result, DW, DH);
  for (auto offset:coords){
    lossless_index.insert(offset);
  }
  return result;
}


std::vector<std::array<double, 2>> trajectory(double *X_original,const std::array<double, 2>& initial_x, const double time_step, const int DH,const int DW, const std::unordered_map<int, critical_point_t>& critical_points, ftk::ndarray<double>& data  ,std::vector<int>& index, std::vector<double>& config, std::vector<record_t>& record,std::set<size_t>& lossless_index){
  std::vector<std::array<double, 2>> result;
  int flag = 0; // 1 means found, -1 means out of bound， 0 means reach max length
  int length = 0;
  result.push_back({X_original[0], X_original[1]}); //add original true position
  length ++;
  int orginal_offset = get_cell_offset(X_original, DW, DH);

  std::array<double, 2> current_x = initial_x;

  //add original and initial_x position's offset
  auto ori_offset = get_three_offsets(X_original, DW, DH);
  for (auto offset:ori_offset){
    lossless_index.insert(offset);
  }
  auto ini_offset = get_three_offsets(initial_x, DW, DH);
  for (auto offset:ini_offset){
    lossless_index.insert(offset);
  }
  

  if(!inside(current_x, DH, DW)){
    //count_out_bound ++;
    flag = -1;
    result.push_back({-1, -1});
    length ++;
    index.push_back(length);
    return result;
  }
  else{
    result.push_back(current_x); //add initial position(seed)
    length ++;
  }

  while (flag == 0){
    if(!inside(current_x, DH, DW)){
      //count_out_bound ++;
      flag = -1;
      result.push_back({-1, -1});
      length ++;
      break;
    }
    if (length == 2000) {
      //printf("reach max length!\n");
      //count_limit ++;
      // flag = 1;
      break;
    }

    double current_v[2] = {0};

    interp2d(current_x.data(), current_v,data); 

    //int current_offset = get_cell_offset(current_x.data(), DW, DH);    

    std::array<double, 2> RK4result = newRK4(current_x.data(), current_v, data, time_step, DH, DW,lossless_index);
    if (RK4result[0] == -1 && RK4result[1] == -1){
      //count_out_bound ++;
      flag = -1;
      result.push_back({-1, -1});
      length ++;
      break;
    }

    int current_offset = get_cell_offset(RK4result.data(), DW, DH);

    if (current_offset != orginal_offset){
      //moved to another cell
        auto surrounding_cell = get_surrounding_cell(current_offset,RK4result, DW, DH);
        for (auto cell_offset:surrounding_cell){
          try{
              auto cp = critical_points.at(cell_offset);
              if (cp.type == SADDLE) break;
              //check if distance between current_x and cp.x is small enough
              double error = 1e-3;
              if (fabs(RK4result[0] - cp.x[0]) < error && fabs(RK4result[1] - cp.x[1]) < error){
                // if interpolated location is close to cp location, then find cp
                flag = 1;
                //count_found ++;
                //printf("found cp after %d iteration, type: %s\n",length, get_critical_point_type_string(cp.type).c_str());
                //printf("distance: %f\n",sqrt((initial_x[0]-current_x[0])*(initial_x[0]-current_x[0]) + (initial_x[1]-current_x[1])*(initial_x[1]-current_x[1])));
                //printf("start_id: %d, current_id: %d\n", orginal_offset,get_cell_offset(current_x.data(), DW, DH));
                //printf("start_values: (%f, %f), current_values: (%f, %f)\n", temp_v[0],temp_v[1],current_v[0],current_v[1]);
                //printf("start_position: (%f, %f), current_position: (%f, %f)\n", initial_x[0],initial_x[1],current_x[0],current_x[1]);

                //add to record
                int cp_offset = get_cell_offset(cp.x, DW, DH);
                record_t r(static_cast<double>(orginal_offset), static_cast<double>(cp_offset), config[2], config[0], config[1]);
                record.push_back(r);

                // first add rk4 position
                result.push_back({RK4result[0], RK4result[1]});
                length++;
                // then add cp position
                std::array<double, 2> true_cp = {cp.x[0], cp.x[1]};
                result.push_back(true_cp);
                length++;
                index.push_back(length);
                return result;
              }
              else{
                //not found cp in this cell

              }
          }
          catch(const std::out_of_range& e){
            // 键不存在，继续查找下一个键
          }
        }
    }
    current_x = RK4result;
    // printf("current_x: (%f, %f)\n", current_x[0], current_x[1]);
    result.push_back(current_x);
    length++;
  }
  // printf("length: %d\n", length);
  // printf("trajectory length: %ld\n", result.size());
  // printf("current_x: (%f, %f)\n", current_x[0], current_x[1]);
  
  // if (flag == 0){
  //   // printf("not found after %d iteration\n",length);
  //   // printf("start_id: %d, current_id: %d\n", orginal_offset,get_cell_offset(current_x.data(), DW, DH));
  //   // double temp_v[2] = {0};
  //   // interp2d(initial_x.data(), temp_v);
  //   // double current_v[2] = {0};
  //   // interp2d(current_x.data(), current_v);
  //   // printf("start_values: (%f, %f), current_values: (%f, %f)\n", temp_v[0],temp_v[1],current_v[0],current_v[1]);
  //   // printf("start_position: (%f, %f), current_position: (%f, %f)\n", initial_x[0],initial_x[1],current_x[0],current_x[1]);
  // }

  index.push_back(length);

  return result;
}

bool areVectorsEqual(const std::vector<std::vector<std::array<double, 2>>>& vector1,
                     const std::vector<std::vector<std::array<double, 2>>>& vector2) {
    // 首先比较外层 std::vector 的大小
    if (vector1.size() != vector2.size()) {
        return false;
    }

    // 逐个比较中间层的 std::vector
    for (size_t i = 0; i < vector1.size(); ++i) {
        const std::vector<std::array<double, 2>>& innerVector1 = vector1[i];
        const std::vector<std::array<double, 2>>& innerVector2 = vector2[i];

        // 检查中间层 std::vector 的大小
        if (innerVector1.size() != innerVector2.size()) {
            return false;
        }

        // 逐个比较内部的 std::array
        for (size_t j = 0; j < innerVector1.size(); ++j) {
            const std::array<double, 2>& array1 = innerVector1[j];
            const std::array<double, 2>& array2 = innerVector2[j];

            // 比较内部的 std::array 是否相等
            if (array1 != array2) {
                return false;
            }
        }
    }

    // 如果所有元素都相等，则返回 true，表示两个数据结构相等
    return true;
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

inline std::set<std::array<size_t,2>> get_three_coords(const std::array<double, 2>& x, const int DW, const int DH){
  size_t x0 = floor(x[0]); 
  size_t y0 = floor(x[1]);
  std::set<std::array<size_t,2>> result;
  if (is_upper({x[0], x[1]})){
    std::array<size_t, 2> left_low= {x0, y0};
    std::array<size_t, 2> left_up = {x0, y0+1};
    std::array<size_t, 2> right_up = {x0+1, y0+1};
    result.insert(left_low);
    result.insert(left_up);
    result.insert(right_up);
  }
  else{
    std::array<size_t, 2> left_low= {x0, y0};
    std::array<size_t, 2> right_low = {x0+1, y0};
    std::array<size_t, 2> right_up = {x0+1, y0+1};
    result.insert(left_low);
    result.insert(right_low);
    result.insert(right_up);
  }
  return result;
}



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

inline std::set<size_t> get_all_simplex_ID(const std::vector<std::array<double, 2>>& tracepoints, const int DW, const int DH){
  //潜在的bug: 如果velcoity很大，可能会跳过cell
  std::set<size_t> result;
  for (const auto& coord:tracepoints){
    //ignore the points on boundary
    //if (coord[0] <= 1.0 || coord[0] >= DW-2.0 || coord[1] <= 1.0 || coord[1] >= DH-2.0) continue; //这是之前的坑，晚点补上
    auto cell_offset = get_cell_offset(coord.data(), DW, DH); 
    result.insert(cell_offset);
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

void baseline(const std::vector<std::vector<std::array<double, 2>>>& true_traj,const std::vector<std::vector<std::array<double, 2>>>& tracepoints2, const int DW, const int DH, std::set<size_t>& diff_offset_index){
  for (size_t i =0 ; i < true_traj.size(); ++i){
    const auto& t1 = true_traj[i]; 
    auto t1_simplexs = get_all_simplex_ID(t1, DW, DH);
    auto offests = convert_simplexID_to_offset(t1_simplexs, DW, DH);
    diff_offset_index.insert(offests.begin(), offests.end());
  }
}

void check_start_end(const std::vector<std::vector<std::array<double, 2>>>& tracepoints1,const std::vector<std::vector<std::array<double, 2>>>& tracepoints2, const int DW, const int DH){
  //check if start and end point is the same
  int count_start_diff = 0;
  int count_end_diff = 0;
  for (size_t i =0 ; i < tracepoints1.size(); ++i){
    const auto& t1 = tracepoints1[i]; // trajectory 1,orginal
    const auto& t2 = tracepoints2[i]; // trajectory 2,decompressed
    if (get_cell_offset(t1[0].data(), DW, DH) != get_cell_offset(t2[0].data(), DW, DH)){
      count_start_diff ++;
    }
    if (get_cell_offset(t1[t1.size()-1].data(), DW, DH) != get_cell_offset(t2[t2.size()-1].data(), DW, DH)){
      // printf("t1 start:(%f,%f), t2 start:(%f,%f)\n", t1[0][0], t1[0][1], t2[0][0], t2[0][1]);
      // printf("t1 end:(%f,%f), t2 end:(%f,%f)\n", t1[t1.size()-1][0], t1[t1.size()-1][1], t2[t2.size()-1][0], t2[t2.size()-1][1]);
      // printf("t1[:-2]:(%f,%f), t2[:-2]:(%f,%f)\n", t1[t1.size()-2][0], t1[t1.size()-2][1], t2[t2.size()-2][0], t2[t2.size()-2][1]);
      count_end_diff ++;
    }
  }
  printf("start point different count: %d / %zu\n", count_start_diff, tracepoints1.size());
  printf("end point different count: %d / %zu\n", count_end_diff, tracepoints1.size());

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
 


int main(int argc, char **argv){

  size_t num = 0;
  float * u = readfile<float>(argv[1], num);
  float * v = readfile<float>(argv[2], num);
  int DW = atoi(argv[3]); //1800
  int DH = atoi(argv[4]); //1200
  int sos = 1;
  grad.reshape({2, static_cast<unsigned long>(DW), static_cast<unsigned long>(DH)});
  refill_gradient(0, DH, DW, u, grad);
  refill_gradient(1, DH, DW, v, grad);

  //decompressed data
 
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

   // compute vector_field_resolution, orginal data
  const int type_bits = 63;
  double vector_field_resolution = 0;
  uint64_t vector_field_scaling_factor = 1;
  for (int i=0; i<num; i++){
    double min_val = std::max(fabs(u[i]), fabs(v[i]));
    vector_field_resolution = std::max(vector_field_resolution, min_val);
  }
  int vbits = std::ceil(std::log2(vector_field_resolution));
  int nbits = (type_bits - 3) / 2;
  vector_field_scaling_factor = 1 << (nbits - vbits);
  std::cerr << "resolution=" << vector_field_resolution 
  << ", factor=" << vector_field_scaling_factor 
  << ", nbits=" << nbits << ", vbits=" << vbits << ", shift_bits=" << nbits - vbits << std::endl;
  auto critical_points_0 =compute_critical_points(u, v, DH, DW, vector_field_scaling_factor);

  // compute vector_field_resolution,decompressed data
  vector_field_resolution = 0;
  for (int i=0; i<num; i++){
    double min_val = std::max(fabs(u_out[i]), fabs(v_out[i]));
    vector_field_resolution = std::max(vector_field_resolution, min_val);
  }
  vbits = std::ceil(std::log2(vector_field_resolution));
  nbits = (type_bits - 3) / 2;
  vector_field_scaling_factor = 1 << (nbits - vbits);
  std::cerr << "resolution=" << vector_field_resolution 
  << ", factor=" << vector_field_scaling_factor 
  << ", nbits=" << nbits << ", vbits=" << vbits << ", shift_bits=" << nbits - vbits << std::endl;
  auto critical_points_out =compute_critical_points(u_out, v_out, DH, DW, vector_field_scaling_factor);

  std::vector<record_t> record;
  std::vector<record_t> record_out;

  	
  // std::ofstream outputFile("../debug/cp_index_test_ori.txt");
	// for (int i = 0; i < critical_points_0.size(); i++) {
	// 	outputFile << i << std::endl;
	// }
	// outputFile.close();	

  printf("orginal critical points size: %ld\n", critical_points_0.size());
  printf("decompressed critical points size: %ld\n", critical_points_out.size());
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

  //exit(0);

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
  tracepoints.reserve(critical_points_0.size()*4);
  tracepoints_out.reserve(critical_points_out.size()*4);
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
          }
          else{
            result_return = trajectory(cp.x,X_start, -h,DH,DW, critical_points_0,grad,myindex,config[i],record,lossless_index);
            tracepoints.push_back(result_return);
          }
        //printf("tracepoints size: %ld\n", result_return.size());
        
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
            }
            else{
              result_return = trajectory(cp.x,X_start, -h,DH,DW, critical_points_out,grad_out,myindex_out,config[i],record_out,lossless_index_out);
              tracepoints_out.push_back(result_return);
            }
          //printf("tracepoints size: %ld\n", result_return.size());
          
          }
        }
      }
    }
  }

  printf("tracepoints size: %ld\n", tracepoints.size());
  printf("tracepoints_out size: %ld\n", tracepoints_out.size());


  std::set<size_t> diff_simplics;
  std::set<std::array<size_t,2>> diff_coords;
  std::set<size_t> diff_offset_index;
  std::vector<size_t> diff_traj_index;
  if (test_flag == "baseline"){
    // boundary
    // for (int i = 0; i < DW; ++i) {
    //   lossless_index.insert(i);                    // 第一行
    //   lossless_index.insert((DH-1)*DW+i);        // 最后一行
    // }
    // for (int i = 0; i < DH; ++i) {
    //   lossless_index.insert(i*DW);                    // 第一列
    //   lossless_index.insert(i*DW+DW-1);        // 最后一列
    // }
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
    exit(0);
  }
  else if (test_flag == "test"){
    int diff_flag = 0;
    //check if two trajectories has same simplexes set
    for (int i =0; i < tracepoints.size(); ++i){
      const auto& t1 = tracepoints[i]; // trajectory 1,orginal
      const auto& t2 = tracepoints_out[i]; // trajectory 2,decompressed
      for (int j = 0; j < t1.size(); ++j){
        if (tracepoints[i][j] != tracepoints_out[i][j]){
          printf("diff: trajectory %d, point %d, t1: (%f, %f), t2: (%f, %f)\n", i, j, t1[j][0], t1[j][1], t2[j][0], t2[j][1]);
          diff_flag = 1;
        }
      }
    }
    if (diff_flag == 0){
      printf("test passed\n");
    }
    else{
      printf("test failed\n");
      printf("lossless_index size: %ld, lossless_index_out size: %ld\n", lossless_index.size(), lossless_index_out.size());
      // size_t *gt_lossless_index = NULL;
      // size_t gt_lossless_index_size = 0;
      // gt_lossless_index = readfile<size_t>("../small_data/index_need_lossless.bin", gt_lossless_index_size);
      // //build hashmap
      // std::unordered_map<size_t, size_t> gt_lossless_index_map;
      // for (size_t i = 0; i < gt_lossless_index_size; i++){
      //   gt_lossless_index_map[gt_lossless_index[i]] = i;
      // }

      // printf("check first 4 different trajectory\n");
      // int count = 0;
      // for (auto index : diff_traj_index){
      //   if (count == 4) break;
      //   std::vector<std::array<double, 2>> t1 = tracepoints[index];
      //   std::vector<std::array<double, 2>> t2 = tracepoints_out[index];
      //   //print each point

      //   for (int i = 0; i < t1.size(); i++){
      //     if(get_cell_offset(t1[i].data(), DW, DH)!= get_cell_offset(t2[i].data(), DW, DH))
      //       printf("trajectory %ld, point %d, t1: (%f, %f), t2: (%f, %f),id:(%d,%d) DIFF!\n", index, i, t1[i][0], t1[i][1], t2[i][0], t2[i][1],get_cell_offset(t1[i].data(), DW, DH), get_cell_offset(t2[i].data(), DW, DH));
      //     else
      //       printf("trajectory %ld, point %d, t1: (%f, %f), t2: (%f, %f),id:(%d,%d)\n", index, i, t1[i][0], t1[i][1], t2[i][0], t2[i][1],get_cell_offset(t1[i].data(), DW, DH), get_cell_offset(t2[i].data(), DW, DH));


      //     //check get_three_offsets(t1[i], DW, DH) is in gt_lossless_index_map
      //     auto offsets = get_three_offsets(t1[i], DW, DH);
      //     for (auto offset : offsets){
      //       if (gt_lossless_index_map.find(offset) == gt_lossless_index_map.end()){
      //         printf("offset %ld not found in gt_lossless_index\n", offset);
      //       }
      //     }  
      //   }
      //   count ++;
      // }
    }
    //check_start_end(tracepoints, tracepoints_out, DW, DH);
    check_two_traj(tracepoints, tracepoints_out, critical_points_0, critical_points_out, DW, DH);
    printf("test done\n");
    exit(0);
  
  }


  
  if (test_flag == "out" || test_flag == "baseline"){
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
  std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing." << std::endl;
        return 1;
    }
    int count = 0;
    std::cout << tracepoints.size() <<std::endl;
    for (const auto& row : tracepoints) {
        for (const auto& point : row) {
            file.write(reinterpret_cast<const char*>(point.data()), point.size() * sizeof(double));
            count ++;
        }
    }
    printf("Successfully write trajectory to file, total points: %d\n",count);
    file.close();

    //write index to file
    writefile(filename2.c_str(), myindex.data(), myindex.size());
    printf("Successfully write index to file, total index: %ld\n",myindex.size());

    //write record to file
    writeRecordsToBinaryFile(record, filename3);

    //write critical points to file
    //std::string cp_prefix = "../data/position";
    record_criticalpoints(filename4, critical_points_0);

    // write all the pos that need to lossless compress
  }
}





