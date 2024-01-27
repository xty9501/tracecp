#include <ftk/numeric/print.hh>
#include <ftk/numeric/cross_product.hh>
#include <ftk/numeric/vector_norm.hh>
#include <ftk/numeric/linear_interpolation.hh>
#include <ftk/numeric/bilinear_interpolation.hh>
#include <ftk/numeric/inverse_linear_interpolation_solver.hh>
#include <ftk/numeric/inverse_bilinear_interpolation_solver.hh>
#include <ftk/numeric/gradient.hh>
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

double vector_field_resolution = std::numeric_limits<double>::max();
uint64_t vector_field_scaling_factor = 1;
// int DW = 128, DH = 128;// the dimensionality of the data is DW*DH
ftk::ndarray<double> grad; //grad是三纬，第一个纬度是2，代表着u或者v，第二个纬度是DH，第三个纬度是DW
ftk::simplicial_regular_mesh m(2);
std::mutex mutex;

size_t global_count = 0;




// struct critical_point_t {
// double mu[3]; // interpolation coefficients
// double x[2]; // the coordinates of the critical points
// double v;
// int type;
// size_t simplex_id;
// critical_point_t(){}
// critical_point_t(const double * mu_, const double * x_, const double v_, const int t_){
//     for(int i=0; i<3; i++) mu[i] = mu_[i];
//     for(int i=0; i<2; i++) x[i] = x_[i];
//     v = v_;
//     type = t_;
// }
// };

double triarea(double a, double b, double c)

{

    double s = (a + b + c)/2.0;

    double area=sqrt(fabs(s*(s-a)*(s-b)*(s-c)));

    return area;     

}

double dist(double x0, double y0, double z0, double x1, double y1, double z1)

{

    double a = x1 - x0;	  

    double b = y1 - y0;

    double c = z1 - z0;

    return sqrt(a*a + b*b + c*c);

}

void barycent2d(double *p0, double *p1, double *p2, const double *v, double *lambda )
{

	double x0 = p0[0], y0 = p0[1], z0 = 0;
	double x1 = p1[0], y1 = p1[1], z1 = 0;
	double x2 = p2[0], y2 = p2[1], z2 = 0;
	double vx = v[0], vy = v[1], vz = 0;

    // compute the area of the big triangle

    double a = dist(x0, y0, z0, x1, y1, z1);
    double b = dist(x1, y1, z1, x2, y2, z2);
    double c = dist(x2, y2, z2, x0, y0, z0);

    double totalarea = triarea(a, b, c);

	

    // compute the distances from the outer vertices to the inner vertex

    double length0 = dist(x0, y0, z0, vx, vy, vz);	  

    double length1 = dist(x1, y1, z1, vx, vy, vz);	  

    double length2 = dist(x2, y2, z2, vx, vy, vz);	  

    

    // divide the area of each small triangle by the area of the big triangle

    lambda[0] = triarea(b, length1, length2)/totalarea;

    lambda[1] = triarea(c, length0, length2)/totalarea;

    lambda[2] = triarea(a, length0, length1)/totalarea;	  

}

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

typedef struct record_t{
  double sid_start;
  double sid_end;
  double dir;
  double eig_vector_x;
  double eig_vector_y;
  record_t(double sid_start_, double sid_end_, double dir_, double eig_vector_x_, double eig_vector_y_){
    sid_start = sid_start_;
    sid_end = sid_end_;
    dir = dir_;
    eig_vector_x = eig_vector_x_;
    eig_vector_y = eig_vector_y_;
  }
  record_t(){}
}record_t;

struct PairHash {
    size_t operator()(const std::pair<int, int>& p) const {
        auto hash1 = std::hash<int>{}(p.first);
        auto hash2 = std::hash<int>{}(p.second);
        return hash1 ^ hash2; // XOR 两个哈希值，也可以用其他方法组合
    }
};

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


std::unordered_map<int, bool> flags;
bool original = true;





void refill_gradient(int id,const int DH,const int DW, const float* grad_tmp){
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
  // robust critical point test
  bool succ = ftk::robust_critical_point_in_simplex2(vf, indices);
  if (!succ) return;
  double mu[3]; // check intersection
  double cond;
  bool succ2 = ftk::inverse_lerp_s2v2(v, mu, &cond);
  if (!succ2) ftk::clamp_barycentric<3>(mu);
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
  // if (cp_type == SINGULAR) return;
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
	for(int i=1; i<r1-2; i++){
    if(i%100==0) std::cout << i << " / " << r1-1 << std::endl;
		for(int j=1; j<r2-2; j++){
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
Type * readfile(const char * file, size_t& num){
  std::ifstream fin(file, std::ios::binary);
  if(!fin){
        std::cout << " Error, Couldn't find the file" << "\n";
        return 0;
    }
    fin.seekg(0, std::ios::end);
    const size_t num_elements = fin.tellg() / sizeof(Type);
    fin.seekg(0, std::ios::beg);
    Type * data = (Type *) malloc(num_elements*sizeof(Type));
  fin.read(reinterpret_cast<char*>(&data[0]), num_elements*sizeof(Type));
  fin.close();
  num = num_elements;
  return data;
}

template<typename Type>
void writefile(const char * file, Type * data, size_t num_elements){
  std::ofstream fout(file, std::ios::binary);
  fout.write(reinterpret_cast<const char*>(&data[0]), num_elements*sizeof(Type));
  fout.close();
}

void writeRecordsToBinaryFile(const std::vector<record_t>& records, const std::string& filename) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "Unable to open file for writing." << std::endl;
        return;
    }

    for (const auto& record : records) {
        outfile.write(reinterpret_cast<const char*>(&record.sid_start), sizeof(record.sid_start));
        outfile.write(reinterpret_cast<const char*>(&record.sid_end), sizeof(record.sid_end));
        outfile.write(reinterpret_cast<const char*>(&record.dir), sizeof(record.dir));
        outfile.write(reinterpret_cast<const char*>(&record.eig_vector_x), sizeof(record.eig_vector_x));
        outfile.write(reinterpret_cast<const char*>(&record.eig_vector_y), sizeof(record.eig_vector_y));
    }

    outfile.close();
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
  if (x[0] < 0 || x[0] >= DW-1 || x[1] < 0 || x[1] >= DH-1) return false;
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
  
}

inline bool is_upper(const std::array<double, 2> x){
  double x_ex = x[0] - floor(x[0]);
  double y_ex = x[1] - floor(x[1]);
  if (y_ex > x_ex){
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

inline std::vector<int> get_surrounding_cell(const int cell_offset, const int DW, const int DH){
  std::vector<int> surrounding_cell;
  if (cell_offset %2 == 0){
    //upper cell
    if (vaild_offset(cell_offset,DW,DH)){
      surrounding_cell.push_back(cell_offset);
    }
    if (vaild_offset(cell_offset+1,DW,DH)){
      surrounding_cell.push_back(cell_offset+1);
    }
    if (vaild_offset(cell_offset-2*(DW-1)+1,DW,DH)){
      surrounding_cell.push_back(cell_offset-2*(DW-1)+1);
    }
    // if (vaild_offset(cell_offset-2*(DW-1),DW,DH)){
    //   surrounding_cell.push_back(cell_offset-2*(DW-1));
    // }


  }
  else {
    if (vaild_offset(cell_offset,DW,DH)){
      surrounding_cell.push_back(cell_offset);
    }
    if (vaild_offset(cell_offset-1,DW,DH)){
      surrounding_cell.push_back(cell_offset-1);
    }
    if (vaild_offset(cell_offset+2*(DW-1)-1,DW,DH)){
      surrounding_cell.push_back(cell_offset+2*(DW-1)-1);
    }
  }
  return surrounding_cell;
}

inline bool check_result(const double error, const double *v){
  if (fabs(v[0]) < error && fabs(v[1]) < error){
    printf("v[0]: %f, v[1]: %f\n", v[0], v[1]);
    return true;
  }
  else{
    return false;
  }
}



void interp2d(const double p[2], double v[2]){
  double X[3][2];
  double V[3][2];
  int x0 = floor(p[0]);
  int y0 = floor(p[1]);
  float x_ex = p[0] - x0;
  float y_ex = p[1] - y0;
  int upper =1;
  if (y_ex > x_ex){
    upper = 1;
  }
  else{
    upper = 0;
  }
  if (upper == 1){
      X[0][0] = x0;
      X[0][1] = y0;
      X[1][0] = x0;
      X[1][1] = y0+1;
      X[2][0] = x0+1;
      X[2][1] = y0+1;
      for (int i =0;i <2;i++){
        V[0][i] = grad(i, x0, y0);
        V[1][i] = grad(i, x0, y0+1);
        V[2][i] = grad(i, x0+1, y0+1);
      }
    }
  else{
    X[0][0] = x0;
    X[0][1] = y0;
    X[1][0] = x0+1;
    X[1][1] = y0;
    X[2][0] = x0+1;
    X[2][1] = y0+1;
    for (int i =0;i <2;i++){
      V[0][i] = grad(i, x0, y0);
      V[1][i] = grad(i, x0+1, y0);
      V[2][i] = grad(i, x0+1, y0+1);
    }
  }
  double lambda[3];
  barycent2d(X[0], X[1], X[2], p, lambda);
  v[0] = lambda[0]*V[0][0] + lambda[1]*V[1][0] + lambda[2]*V[2][0];
  v[1] = lambda[0]*V[0][1] + lambda[1]*V[1][1] + lambda[2]*V[2][1];

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

template<typename Type>
std::array<Type, 2> newRK4(const Type * x, const Type * v,  Type h, const int DH, const int DW) {
  // x and y are positions, and h is the step size
  double rk1[2] = {0};
  const double p1[] = {x[0], x[1]};
  if(!inside(p1, DH, DW)){
    return std::array<Type, 2>{x[0], x[1]};
  }
  interp2d(p1, rk1);
  
  double rk2[2] = {0};
  const double p2[] = {x[0] + 0.5 * h * rk1[0], x[1] + 0.5 * h * rk1[1]};
  if (!inside(p2, DH, DW)){
    return std::array<Type, 2>{p1[0], p1[1]};
  }
  interp2d(p2, rk2);
  
  double rk3[2] = {0};
  const double p3[] = {x[0] + 0.5 * h * rk2[0], x[1] + 0.5 * h * rk2[1]};
  if (!inside(p3, DH, DW)){
    return std::array<Type, 2>{p2[0], p2[1]};
  }
  interp2d(p3, rk3);
  
  double rk4[2] = {0};
  const double p4[] = {x[0] + h * rk3[0], x[1] + h * rk3[1]};
  if (!inside(p4, DH, DW)){
    return std::array<Type, 2>{p3[0], p3[1]};
  }
  interp2d(p4, rk4);
  
  Type next_x = x[0] + h * (rk1[0] + 2 * rk2[0] + 2 * rk3[0] + rk4[0]) / 6;
  Type next_y = x[1] + h * (rk1[1] + 2 * rk2[1] + 2 * rk3[1] + rk4[1]) / 6;
  // printf("shift: (%f, %f)\n", next_x - x[0], next_y - x[1]);
  // printf("coefficients: (%f,%f)\n",(rk1[0] + 2 * rk2[0] + 2 * rk3[0] + rk4[0]) / 6, (rk1[1] + 2 * rk2[1] + 2 * rk3[1] + rk4[1]) / 6);
  // printf("current h sign: %d\n", printsign(h));
  // printf("sign of coefficients x (%d,%d,%d,%d)\n", printsign(rk1[0]), printsign(rk2[0]), printsign(rk3[0]), printsign(rk4[0]));
  // printf("sign of coefficients y (%d,%d,%d,%d)\n", printsign(rk1[1]), printsign(rk2[1]), printsign(rk3[1]), printsign(rk4[1]));

  std::array<Type, 2> result = {next_x, next_y};
  return result;
}


std::vector<std::array<double, 2>> trajectory(double *X_original,const std::array<double, 2>& initial_x, const double time_step, const int DH,const int DW, const std::unordered_map<int, critical_point_t>& critical_points_0, int &count_limit,int &count_found, int &count_not_found, int &count_out_bound, std::vector<int>& index, std::vector<double>& config, std::vector<record_t>& record){
  std::vector<std::array<double, 2>> result;
  int flag = 0;
  int length = 0;
  result.push_back({X_original[0], X_original[1]}); //add original true position
  length ++;
  int orginal_offset = get_cell_offset(X_original, DW, DH);

  std::array<double, 2> current_x = initial_x;

  if(!inside(current_x, DH, DW)){
    count_out_bound ++;
    return result;
  }
  else{
    result.push_back(current_x); //add initial position
    length ++;
  }

  while (flag == 0){
    if (length == 1000) {
      //printf("reach max length!\n");
      count_limit ++;
      flag = 1;
      break;
    }
    if(!inside(current_x, DH, DW)){
      count_out_bound ++;
      flag = 1;
      break;
    }
    double current_v[2] = {0};
    //printf("current_x: (%f, %f)\n", current_x[0], current_x[1]);
    interp2d(current_x.data(), current_v);
    //printf("current_v: (%f, %f)\n", current_v[0], current_v[1]);
    int current_offset = get_cell_offset(current_x.data(), DW, DH);
    //std::array<double, 2> RK4result = RK4(current_x.data(), current_v, time_step);
    std::array<double, 2> RK4result = newRK4(current_x.data(), current_v, time_step, DH, DW);

    //printf("RK4result: (%f, %f)\n", RK4result[0], RK4result[1]);
    // double temp_v[2] = {0};
    // interp2d(RK4result.data(), temp_v);
    // printf("current time_step: %f\n", time_step);
    // printf("temp_v: (%f, %f)\n", temp_v[0], temp_v[1]);

    if (current_offset != orginal_offset){
      //moved to another cell
        auto surrounding_cell = get_surrounding_cell(current_offset, DW, DH);
        for (auto cell_offset:surrounding_cell){
          try{
              auto cp = critical_points_0.at(cell_offset);
              if (cp.type == SADDLE) break;
              //check if distance between current_x and cp.x is small enough
              double error = 1e-3;
              if (fabs(RK4result[0] - cp.x[0]) < error && fabs(RK4result[1] - cp.x[1]) < error){
                // if interpolated location is close to cp location, then find cp
                flag = 1;
                count_found ++;
                //printf("found cp after %d iteration, type: %s\n",length, get_critical_point_type_string(cp.type).c_str());
                //printf("distance: %f\n",sqrt((initial_x[0]-current_x[0])*(initial_x[0]-current_x[0]) + (initial_x[1]-current_x[1])*(initial_x[1]-current_x[1])));
                //printf("start_id: %d, current_id: %d\n", orginal_offset,get_cell_offset(current_x.data(), DW, DH));
                //printf("start_values: (%f, %f), current_values: (%f, %f)\n", temp_v[0],temp_v[1],current_v[0],current_v[1]);
                //printf("start_position: (%f, %f), current_position: (%f, %f)\n", initial_x[0],initial_x[1],current_x[0],current_x[1]);

                //add to record
                record_t r(static_cast<double>(orginal_offset), static_cast<double>(cell_offset), config[2], config[0], config[1]);
                //printf("start_id: %d, current_id: %d, dir: %f, eig_vector: (%f, %f)\n", orginal_offset,cell_offset, config[2], config[0], config[1]);
                record.push_back(r);

                // break;
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
    result.push_back(current_x);
    length++;
  }
  // printf("length: %d\n", length);
  // printf("result size: %ld\n", result.size());
  
  if (flag == 0){
    count_not_found ++;
    printf("not found after %d iteration\n",length);
    printf("start_id: %d, current_id: %d\n", orginal_offset,get_cell_offset(current_x.data(), DW, DH));
    // double temp_v[2] = {0};
    // interp2d(initial_x.data(), temp_v);
    // double current_v[2] = {0};
    // interp2d(current_x.data(), current_v);
    // printf("start_values: (%f, %f), current_values: (%f, %f)\n", temp_v[0],temp_v[1],current_v[0],current_v[1]);
    // printf("start_position: (%f, %f), current_position: (%f, %f)\n", initial_x[0],initial_x[1],current_x[0],current_x[1]);
  }
  index.push_back(length);

  return result;
}
/*
std::vector<std::array<double, 2>> simulate_motion(const std::array<double, 2>& initial_x, const double time_step, const int num_steps,const int DH,const int DW, const std::unordered_map<int, critical_point_t>& critical_points_0) {
  //x is the position, v is the velocity, h is the step size, n is the number of iterations
  std::vector<std::array<double, 2>> result;
  // result.push_back(initial_x);
  std::array<double, 2> current_x = initial_x;
  // std::array<double, 2> current_v;
  // current_v[0] = grad(0, initial_x[0], initial_x[1]);
  // current_v[1] = grad(1, initial_x[0], initial_x[1]);
  int flag = 0;
  for(int i=0; i<num_steps; i++){
    result.push_back(current_x);
    if (inside(current_x, DH, DW)){
      double current_v[2] = {0};
      interp2d(current_x.data(), current_v);
      std::array<double, 2> RK4result = RK4(current_x.data(), current_v, time_step);
      
      // for(const auto& p:critical_points_0){
      //   // need check each RK4result is non-saddle cp
      //   auto cp = p.second;
      //   if (cp.type != SADDLE) {
      //     double error = 1e-3;
      //     if (fabs(RK4result[0] - cp.x[0]) < error && fabs(RK4result[1] - cp.x[1]) < error){
      //       //printf("reach critical point!\n");
      //       flag = 1;
      //       break;
      //     }
      //   }
      // }
      current_x = RK4result;
    }
    else{
      // printf("out of bound!\n");
      break; // dont need break
      }
    if (flag == 1){
      break;
    }
  }
  
  if (flag == 0){
    //printf("not reach critical point when hit max iteration\n");
  }
  if (result.size() < num_steps){
    for (int i = result.size(); i < num_steps; i ++) {
      result.push_back(result.back());
    }
  }

  return result;
}

*/


int main(int argc, char **argv){
  size_t num = 0;
  float * u = readfile<float>(argv[1], num);
  float * v = readfile<float>(argv[2], num);
  int DW = atoi(argv[3]); //2400
  int DH = atoi(argv[4]); //3600
  int sos = 1;
  grad.reshape({2, static_cast<unsigned long>(DW), static_cast<unsigned long>(DH)});
  refill_gradient(0, DH, DW, u);
  refill_gradient(1, DH, DW, v);
   // compute vector_field_resolution
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

  // std::string cp_prefix = "origin_M_" + std::to_string(nbits - vbits) + "_bits";
  // if(sos) cp_prefix += "_sos";
  // bool cp_file = false; //file_exists(cp_prefix + "_sid.dat");
  // cp_file ? printf("Critical point file found!\n") : printf("Critical point Not found, recomputing\n");

  // auto saddle_points_0 = cp_file ? read_saddlepoints(cp_prefix) : compute_saddle_critical_points(u, v, DH, DW, vector_field_scaling_factor);
  // auto saddle_points_0 = get_saddle_points(DH, DW);

  // std::unordered_map<std::pair<int, int>, record_t, PairHash> record;
  std::vector<record_t> record;

  auto critical_points_0 =compute_critical_points(u, v, DH, DW, vector_field_scaling_factor);

  printf("critical points size: %ld\n", critical_points_0.size());
  auto it = critical_points_0.begin();
  int sad_count = 0;
  while (it != critical_points_0.end()) {
    auto cp = it->second;
    if (cp.type == SADDLE) sad_count ++;
    it ++;
    //print if coordinates is around 1587,314

    // if (cp.x[0] > 1586 && cp.x[0] < 1588 && cp.x[1] > 313 && cp.x[1] < 315){
    //   int count_reach_limit = 0;
    //   int count_found = 0;
    //   int count_not_found = 0;
    //   int count_out_bound = 0;
    //   std::vector<int> myindex;
    //   printf("cp: (%f, %f), type: %s\n", cp.x[0], cp.x[1], get_critical_point_type_string(cp.type).c_str());
    //   //cp: (1587.301997, 314.571846), type: SADDLE
    //   printf("eig_vec: (%f, %f), (%f, %f)\n", cp.eig_vec[0][0], cp.eig_vec[0][1], cp.eig_vec[1][0], cp.eig_vec[1][1]);
    //   std::vector<std::array<double,2>> X_all_direction;  
    //   double eps = 0.1;
    //   double h = 0.01;
    //   X_all_direction.push_back({cp.x[0] + eps*cp.eig_vec[0][0], cp.x[1] + eps*cp.eig_vec[0][1]}); //+
    //   X_all_direction.push_back({cp.x[0] - eps*cp.eig_vec[0][0], cp.x[1] - eps*cp.eig_vec[0][1]}); //+
    //   X_all_direction.push_back({cp.x[0] + eps*cp.eig_vec[1][0], cp.x[1] + eps*cp.eig_vec[1][1]}); //-
    //   X_all_direction.push_back({cp.x[0] - eps*cp.eig_vec[1][0], cp.x[1] - eps*cp.eig_vec[1][1]});  //-
    //   double lambda[3];
    //   double values[2];
    //   std::vector<std::vector<double>> config;
    //   config.push_back({cp.eig_vec[0][0], cp.eig_vec[0][1],1});
    //   config.push_back({cp.eig_vec[0][0], cp.eig_vec[0][1],-1});
    //   config.push_back({cp.eig_vec[1][0], cp.eig_vec[1][1],1});
    //   config.push_back({cp.eig_vec[1][0], cp.eig_vec[1][1],-1});
    //   for (int i = 0; i < 4; i ++) {
    //     std::array<double, 2> X_start;
        
    //     std::vector<std::array<double, 2>> result_return;
    //     X_start = X_all_direction[i];
    //     //check if inside
    //     if (inside(X_start,DH, DW)){
    //      for (int j = 0; j < 2; j ++){
    //         printf("i/j (%d,%d)\n", i,j);
    //         std::vector<std::array<double, 2>> result_return;
    //         if (j == 0){
    //           result_return = trajectory(cp.x,X_start, h,DH,DW, critical_points_0,count_reach_limit,count_found,count_not_found,count_out_bound,myindex,config[i],record);
    //         }
    //         else{
    //         result_return = trajectory(cp.x,X_start, -h,DH,DW, critical_points_0,count_reach_limit,count_found,count_not_found,count_out_bound,myindex,config[i],record);
    //         }
    //         for (int k = 0; k < result_return.size(); k ++){
    //           printf("result_return: (%f, %f)\n", result_return[k][0], result_return[k][1]);
    //         }
    //      }
    //     }
    //     else{
    //       printf("X_start: (%f, %f) is out of bound\n", X_start[0], X_start[1]);
    //     }
    //   }
      
    // }
  
  }
  printf("saddle points size: %d\n", sad_count);
  free(u);
  free(v);

  //exit(0);


  // double test[3][2] = {
  //   {1587, 314},
  //   {1587, 315},
  //   {1588, 315}};
  // double test_coord[2] = {1587.3,314.572};
  // double test_inter[2] = {0};
  // interp2d(test_coord, test_inter);
  // printf("interpolated value at cp: (%f, %f)\n", test_inter[0], test_inter[1]);
  // double test_coord2[2] = {1587.35,314.484};
  // double test_inter2[2] = {0};
  // interp2d(test_coord2, test_inter2);
  // printf("interpolated value at cp+eps*eig: (%f, %f)\n", test_inter2[0], test_inter2[1]);
  // printf("grad at (1587,314): (%f, %f)\n", grad(0, 1587, 314), grad(1, 1587, 314));

 

  int num_steps = 10;
  double h = atof(argv[5]);
  // double h = 0.1;
  double eps = atof(argv[6]);
  // double eps = 0.01;
  int count_reach_limit = 0;
  int count_found = 0;
  int count_not_found = 0;
  int count_out_bound = 0;
  int *index = (int *) malloc(sizeof(int)*sad_count*4);
  int *index_pos = index;
  std::vector<int> myindex;
  printf("current setting: num_steps: %d, h: %f, eps: %f\n", num_steps, h, eps);
  // need record each size of tracepoint
  // printf("creating tracepoints size: %ld,%d \n", saddle_points_0.size(), num_steps);
  std::vector<std::vector<std::array<double, 2>>> tracepoints;
  tracepoints.reserve(critical_points_0.size()*4);

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
          //result_return = simulate_motion(X_start, h, num_steps,DH,DW, critical_points_0);
          
          if(i == 0 || i ==1){
            //result_return = trajectory(X_start, h,DH,DW, critical_points_0);
            //result_return = simulate_motion(X_start, h, num_steps,DH,DW, critical_points_0);
            //printf("config: %f, %f, %f\n", config[i][0],config[i][1],config[i][2]);
            result_return = trajectory(cp.x,X_start, h,DH,DW, critical_points_0,count_reach_limit,count_found,count_not_found,count_out_bound,myindex,config[i],record);
            tracepoints.push_back(result_return);
          }
          else{
            //result_return = trajectory(X_start, -h,DH,DW, critical_points_0);
            //result_return = simulate_motion(X_start, -h, num_steps,DH,DW, critical_points_0);
            //printf("config: %f, %f, %f\n", config[i][0],config[i][1],config[i][2]);
            result_return = trajectory(cp.x,X_start, -h,DH,DW, critical_points_0,count_reach_limit,count_found,count_not_found,count_out_bound,myindex,config[i],record);
            tracepoints.push_back(result_return);
          }
        //printf("tracepoints size: %ld\n", result_return.size());
        
        }
      }
    }
  }
  printf("myindex size: %ld\n", myindex.size());
  printf("total saddle points: %ld\n", global_count);
  printf("number of reach max length: %d\n", count_reach_limit);
  printf("number of found cp: %d\n", count_found);
  printf("number of not found cp: %d\n", count_not_found);
  printf("number of out of bound: %d\n", count_out_bound);
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
    printf("write %d points\n", count);
    file.close();
    std::cout << "Data written to " << filename << std::endl;

    //write index to file
    writefile(filename2.c_str(), myindex.data(), myindex.size());
    std::cout << "index written to " << filename2 << std::endl;

    //write record to file
    writeRecordsToBinaryFile(record, filename3);

    //write critical points to file
    std::string cp_prefix = "../data/position";
    record_criticalpoints(filename4, critical_points_0);





}





