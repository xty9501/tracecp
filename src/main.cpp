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

// #include <hypermesh/ndarray.hh>
// #include <hypermesh/regular_simplex_mesh.hh>
#include <mutex>
// #include <hypermesh/ndarray.hh>
// #include <hypermesh/regular_simplex_mesh.hh>

#include <ftk/ndarray/ndarray_base.hh>
#include <unordered_map>
#include <queue>
#include <fstream>

double vector_field_resolution = std::numeric_limits<double>::max();
uint64_t vector_field_scaling_factor = 1;
// int DW = 128, DH = 128;// the dimensionality of the data is DW*DH
ftk::ndarray<double> grad; //grad是三纬，第一个纬度是2，代表着u或者v，第二个纬度是DH，第三个纬度是DW
ftk::simplicial_regular_mesh m(2);
std::mutex mutex;




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
  critical_point_t(double* x_, double eig_v[2][2], double V_[3][2],double X_[3][2], int t_){
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
  }
  critical_point_t(){}
}critical_point_t;

std::unordered_map<int, critical_point_t> saddles;

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



bool check_simplex(const ftk::simplicial_regular_mesh_element &s)
{
  // printf("check_simplex...\n");
  if (!s.valid(m)) return false; // check if the 3-simplex is valid
  // printf("valid...\n");
  const auto &vertices = s.vertices(m);

  //print vertices
  // for (int i = 0; i < 3; i ++) {
  //   printf("vertices[%d][0]=%f, vertices[%d][1]=%f\n", i, vertices[i][0], i, vertices[i][1]);
  // }

  // simplex_vectors(vertices, v);
  double X[3][2], v[3][2]; // X为3x2矩阵， v为3x2矩阵（存uv的值），X存坐标（x，y 分别3组）
  for (int i = 0; i < 3; i ++) {
    for (int j = 0; j < 2; j ++)
      v[i][j] = grad(j, vertices[i][0], vertices[i][1]);
    for (int j = 0; j < 2; j ++)
      X[i][j] = vertices[i][j];     
  }

// #if FTK_HAVE_GMP //通常直接跳转到else
//   typedef mpf_class fp_t;
//   // typedef double fp_t;

//   fp_t vf[3][2];
//   for (int i = 0; i < 3; i ++) 
//     for (int j = 0; j < 2; j ++) {
//       const double x = v[i][j];
//       if (std::isnan(x) || std::isinf(x)) return false;
//       else vf[i][j] = v[i][j];
//     }
// #else
  // typedef fixed_point<> fp_t;
  int64_t vf[3][2];
  for (int i = 0; i < 3; i ++)
    for (int j = 0; j < 2; j ++) {
      const double x = v[i][j];
      if (std::isnan(x) || std::isinf(x)) return false;
      else vf[i][j] = v[i][j] * vector_field_scaling_factor;
    }
// #endif

  // robust critical point test
  int indices[3];
  for (int i = 0; i < vertices.size(); i ++)
    indices[i] = m.get_lattice().to_integer(vertices[i]);
  bool succ = ftk::robust_critical_point_in_simplex2(vf, indices); //robust_critical_point_in_simplex2 是sos
  //succ 返回如果uv在原点组成的三角形包含原点？
  if (!succ) return false;

  double mu[3]; // check intersection
  double cond;
  bool succ2 = ftk::inverse_lerp_s2v2(v, mu, &cond); //mu if in 0-1
  // succ2这个是干嘛的？

  // if (!succ2) return false;
  // if (std::isnan(mu[0]) || std::isnan(mu[1]) || std::isnan(mu[2])) return false;
  // fprintf(stderr, "mu=%f, %f, %f\n", mu[0], mu[1], mu[2]);
  if (!succ2) ftk::clamp_barycentric<3>(mu); //把mu 标准化到0-1区间且sum=1
  double x[2]; // position
  // simplex_coordinates(vertices, X);
  ftk::lerp_s2v2(X, mu, x);
  //fprintf(stdout, "simplex_id, corner=%d, %d, type=%d, mu=%f, %f, %f, x=%f, %f\n", s.corner[0], s.corner[1], s.type, mu[0], mu[1], mu[2], x[0], x[1]);
  //fprintf(stdout, "simplex_id=%d, corner=%d, %d, type=%d, mu=%f, %f, %f, x=%f, %f\n", s.to_integer(), s.corner[0], s.corner[1], s.type, mu[0], mu[1], mu[2], x[0], x[1]);
  double J[2][2]; // jacobian
  ftk::jacobian_2dsimplex2(X, v, J);  
  int cp_type = 0;
  std::complex<double> eig[2];
  double delta = ftk::solve_eigenvalues2x2(J, eig);
  double eig_vec[2][2]={0};
  // if(fabs(delta) < std::numeric_limits<double>::epsilon())
  if (delta >= 0) { // two real roots
    if (eig[0].real() * eig[1].real() < 0) { //different sign
      cp_type = SADDLE;
      double eig_r[2];
      eig_r[0] = eig[0].real(), eig_r[1] = eig[1].real();
      ftk::solve_eigenvectors2x2(J, 2, eig_r, eig_vec);
      critical_point_t cp(x, eig_vec,v,X, cp_type);

      // std::cout << cp.x[0] << " " << cp.x[1] << ": " << eig[0] << " " << eig[1] << std::endl;
      std::lock_guard<std::mutex> guard(mutex);
      //printf(std::to_string(s.to_integer(m)).c_str());
      saddles.insert(std::make_pair(s.to_integer(m), cp));
      
    } 
  } 
//   critical_point_t cp(mu, x, 0, cp_type);
  
  // cp.simplex_id = s.to_integer(m);
//   {
//     std::lock_guard<std::mutex> guard(mutex);
//     critical_points.insert(std::make_pair(s.to_integer(m), cp));
//   }
//   //fprintf(stdout, "simplex_id=%d, corner=%d, %d, type=%d, x=%f, %f, my_x=%f,%f\n", cp.simplex_id,s.corner[0], s.corner[1], s.type, x[0], x[1],cp.x[0],cp.x[1]);
//   return true;  

}

void refill_gradient(int id,const int DH,const int DW, const float* grad_tmp){
  const float * grad_tmp_pos = grad_tmp;
  for (int i = 0; i < DH; i ++) {
    for (int j = 0; j < DW; j ++) {
      grad(id, j, i) = *(grad_tmp_pos ++);
    }
  }
}


void extract_saddles(const int DH,const int DW)
{
  m.set_lb_ub({1, 1}, {DW-2, DH-2}); // set the lower and upper bounds of the mesh
  // printf("DW=%d, DH=%d\n", DW, DH);
  m.element_for(2, check_simplex); // iterate over all 3-simplices #这里是不是2-simplex？
}

std::unordered_map<int, critical_point_t> get_saddle_points(const int DH,const int DW){
  saddles.clear();
  // derive_gradients();
  extract_saddles(DH, DW);
  std::unordered_map<int, critical_point_t> result(saddles);
  return result;
}


// template<typename T_fp>
// static void 
// check_simplex_seq_saddle(const T_fp vf[3][2], const double v[3][2], const double X[3][2], const int indices[3], int i, int j, int simplex_id, std::unordered_map<int, critical_point_t>& critical_points){
//   // robust critical point test
//   bool succ = ftk::robust_critical_point_in_simplex2(vf, indices);
//   if (!succ) return;
//   double mu[3]; // check intersection
//   double cond;
//   bool succ2 = ftk::inverse_lerp_s2v2(v, mu, &cond);
//   if (!succ2) ftk::clamp_barycentric<3>(mu);
//   double x[2]; // position
//   ftk::lerp_s2v2(X, mu, x);
//   double J[2][2]; // jacobian
//   ftk::jacobian_2dsimplex2(X, v, J);  
//   int cp_type = 0;
//   std::complex<double> eig[2];
//   double delta = ftk::solve_eigenvalues2x2(J, eig);
//   // if(fabs(delta) < std::numeric_limits<double>::epsilon())
//   if (delta >= 0) { // two real roots
//     if (eig[0].real() * eig[1].real() < 0) {
//         cp_type = SADDLE;
//         critical_point_t cp;
//         cp.x[0] = j + x[0]; cp.x[1] = i + x[1];
//         cp.simplex_id = simplex_id;
//         cp.type = cp_type;
//         // cp.v = mu[0]*v[0][0] + mu[1]*v[1][0] + mu[2]*v[2][0];
//         // cp.u = mu[0]*v[0][1] + mu[1]*v[1][1] + mu[2]*v[2][1];
//         critical_points[simplex_id] = cp;
//     } 
//   }
  
// }

// template<typename T>
// std::unordered_map<int, critical_point_t>
// compute_saddle_critical_points(const T * U, const T * V, int r1, int r2, uint64_t vector_field_scaling_factor){
//   // check cp for all cells
//   using T_fp = int64_t;
//   size_t num_elements = r1*r2;
//   T_fp * U_fp = (T_fp *) malloc(num_elements * sizeof(T_fp));
//   T_fp * V_fp = (T_fp *) malloc(num_elements * sizeof(T_fp));
//   for(int i=0; i<num_elements; i++){
//     U_fp[i] = U[i]*vector_field_scaling_factor;
//     V_fp[i] = V[i]*vector_field_scaling_factor;
//   }
//   int indices[3] = {0};
//   // __int128 vf[4][3] = {0};
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
//   int64_t vf[3][2] = {0};
//   double v[3][2] = {0};
//   std::unordered_map<int, critical_point_t> critical_points;
// 	for(int i=1; i<r1-2; i++){
//     if(i%100==0) std::cout << i << " / " << r1-1 << std::endl;
// 		for(int j=1; j<r2-2; j++){
//       ptrdiff_t cell_offset = 2*(i * (r2-1) + j);
// 			indices[0] = i*r2 + j;
// 			indices[1] = (i+1)*r2 + j;
// 			indices[2] = (i+1)*r2 + (j+1); 
// 			// cell index 0
// 			for(int p=0; p<3; p++){
// 				vf[p][0] = U_fp[indices[p]];
// 				vf[p][1] = V_fp[indices[p]];
// 				v[p][0] = U[indices[p]];
// 				v[p][1] = V[indices[p]];
// 			}
      
//       check_simplex_seq_saddle(vf, v, X1, indices, i, j, cell_offset, critical_points);
// 			// cell index 1
// 			indices[1] = i*r2 + (j+1);
// 			vf[1][0] = U_fp[indices[1]], vf[1][1] = V_fp[indices[1]];
// 			v[1][0] = U[indices[1]], v[1][1] = V[indices[1]];			
//       check_simplex_seq_saddle(vf, v, X2, indices, i, j, cell_offset + 1, critical_points);
// 		}
// 	}
//   free(U_fp);
//   free(V_fp);
//   return critical_points; 
// }

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

template<typename Type>
std::array<Type, 2> RK4(const Type * x, const Type * v, const Type h){
  // x is the position, v is the velocity, h is the step size
  Type k1_u = v[0];
  Type k1_v = v[1];
  Type k2_u = v[0] + h * k1_u / 2;
  Type k2_v = v[1] + h * k1_v / 2;
  Type k3_u = v[0] + h * k2_u / 2;
  Type k3_v = v[1] + h * k2_v / 2;
  Type k4_u = v[0] + h * k3_u;
  Type k4_v = v[1] + h * k3_v;
  Type u = x[0] + h * (k1_u + 2 * k2_u + 2 * k3_u + k4_u) / 6;
  Type v_result = x[1] + h * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6;
  std::array<Type, 2> result = {u, v_result};
  return result;
}

// template<typename Type>
// std::array<Type, 2> RK4(const Type * X, const Type * V, const Type h){
//   // x is the position, v is the velocity, h is the step size
//   std::array<Type, 2> result;
//   memcpy(result, X, 2*sizeof(Type));

//   double v[2];

//  //1st
//  interp2d(X, v);
//   Type k1_u;
//   Type k1_v;
//   k1_u = h*v[0];
//   k1_v = h*v[1];
//   result[0] = result[0] + k1_u/2;
        

//   Type k2_u = v[0] + h * k1_u / 2;
//   Type k2_v = v[1] + h * k1_v / 2;
//   Type k3_u = v[0] + h * k2_u / 2;
//   Type k3_v = v[1] + h * k2_v / 2;
//   Type k4_u = v[0] + h * k3_u;
//   Type k4_v = v[1] + h * k3_v;
//   Type u = x[0] + h * (k1_u + 2 * k2_u + 2 * k3_u + k4_u) / 6;
//   Type v_result = x[1] + h * (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6;
//   std::array<Type, 2> result = {u, v_result};
//   return result;
// }

// void record_criticalpoints(std::string prefix, const std::vector<critical_point_t>& cps){
//   double * positions = (double *) malloc(cps.size()*2*sizeof(double));
//   int * type = (int *) malloc(cps.size()*sizeof(int));
//   int i = 0;
//   for(const auto& cp:cps){
//     positions[2*i] = cp.x[0];
//     positions[2*i+1] = cp.x[1];
//     type[i] = cp.type;
//     i ++;
//   }
//   writefile((prefix + "_pos.dat").c_str(), positions, cps.size()*2);
//   writefile((prefix + "_type.dat").c_str(), type, cps.size());
//   free(positions);
//   free(type);
// }

inline bool file_exists(const std::string& filename) {
    std::ifstream f(filename.c_str());
    return f.good();
}


inline bool inside(const std::array<double, 2> x,const int DH, const int DW){
  if (x[0] < 0 || x[0] >= DW || x[1] < 0 || x[1] >= DH) return false;
  else return true;

}

// std::unordered_map<int, critical_point_t> read_saddlepoints(const std::string& prefix){
//   std::unordered_map<int, critical_point_t> cps;
//   size_t num = 0;
//   double * positions = readfile<double>((prefix + "_pos.dat").c_str(), num);
//   int * type = readfile<int>((prefix + "_type.dat").c_str(), num);
//   size_t * sid = readfile<size_t>((prefix + "_sid.dat").c_str(), num);
//   printf("Read %ld critical points\n", num);
//   for(int i=0; i<num; i++){
//     critical_point_t p;
//     p.x[0] = positions[2*i]; p.x[1] = positions[2*i+1];
//     p.type = type[i];
//     p.simplex_id = sid[i]; 

//     cps.insert(std::make_pair(sid[i], p));
//   }
//   return cps;
// }

// void record_criticalpoints(const std::string& prefix, const std::vector<critical_point_t>& cps, bool write_sid=false){
//   double * positions = (double *) malloc(cps.size()*2*sizeof(double));
//   int * type = (int *) malloc(cps.size()*sizeof(int));
//   int i = 0;
//   for(const auto& cp:cps){
//     positions[2*i] = cp.x[0];
//     positions[2*i+1] = cp.x[1];
//     type[i] = cp.type;
//     i ++;
//   }
//   writefile((prefix + "_pos.dat").c_str(), positions, cps.size()*2);
//   writefile((prefix + "_type.dat").c_str(), type, cps.size());
//   if(write_sid){
//     size_t * sid = (size_t *) malloc(cps.size()*sizeof(size_t));
//     int i = 0;
//     for(const auto& cp:cps){
//       sid[i ++] = cp.simplex_id;
//     }
//     writefile((prefix + "_sid.dat").c_str(), sid, cps.size());
//     free(sid);
//   }
//   free(positions);
//   free(type);
// }

// void init_grad(const Data& data){
//   grad.reshape({2, data.nv, data.nu});
//   for (int i = 0; i < data.nu; i ++) {
//     for (int j = 0; j < data.nv; j ++) {
//       grad(0, j, i) = data.u[i*data.nu + j];
//       grad(1, j, i) = data.v[i*data.nu + j];
//     }
//   }
// }

// void extract_critical_points()
// {
//   fprintf(stderr, "extracting critical points...\n");
//   // ftk::simplicial_regular_mesh m(2);
//   m.set_lb_ub({1, 1}, {DW-2, DH-2}); // set the lower and upper bounds of the mesh
//   m.element_for(2, check_simplex_seq,1); // iterate over all 3-simplices
// }

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

std::vector<std::array<double, 2>> simulate_motion(const std::array<double, 2>& initial_x, const double time_step, const int num_steps,const int DH,const int DW) {
  //x is the position, v is the velocity, h is the step size, n is the number of iterations
  std::vector<std::array<double, 2>> result;
  // result.push_back(initial_x);
  std::array<double, 2> current_x = initial_x;
  // std::array<double, 2> current_v;
  // current_v[0] = grad(0, initial_x[0], initial_x[1]);
  // current_v[1] = grad(1, initial_x[0], initial_x[1]);

  for(int i=0; i<num_steps; i++){
    result.push_back(current_x);
    if (inside(current_x, DH, DW)){
      double current_v[2] = {0};
      interp2d(current_x.data(), current_v);
      std::array<double, 2> RK4result = RK4(current_x.data(), current_v, time_step);
      current_x = RK4result;
    }
    else{
      // printf("out of bound!\n");
      break; // dont need break
      }
  }
  return result;
}


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

  std::string cp_prefix = "origin_M_" + std::to_string(nbits - vbits) + "_bits";
  if(sos) cp_prefix += "_sos";
  bool cp_file = false; //file_exists(cp_prefix + "_sid.dat");
  cp_file ? printf("Critical point file found!\n") : printf("Critical point Not found, recomputing\n");
  // auto saddle_points_0 = cp_file ? read_saddlepoints(cp_prefix) : compute_saddle_critical_points(u, v, DH, DW, vector_field_scaling_factor);
  auto saddle_points_0 = get_saddle_points(DH, DW);
  printf("Saddle points size: %ld\n", saddle_points_0.size());


//   free(u);
//   free(v);

  int num_steps = 1000;
  double h = 0.1;
  // need record each size of tracepoint
  // printf("creating tracepoints size: %ld,%d \n", saddle_points_0.size(), num_steps);
  std::vector<std::vector<std::array<double, 2>>> tracepoints;
  tracepoints.reserve(saddle_points_0.size());
  
  // (saddle_points_0.size(), std::vector<std::array<double, 2>>(num_steps));
  // Initialize all elements to zero
  // for (auto& row : tracepoints) {
  //   row.resize(num_steps);
  //     for (auto& point : row) {
  //         point = {0.0, 0.0}; // Set both elements to zero
  //     }
  // }

  //std::vector<std::vector<std::array<double, 2>>> tracepoints;
 
  for(const auto& p:saddle_points_0){
    auto cp = p.second;
    // std::cout << "SADDLE POINT FOUND AT " << cp.x[0] << ", " << cp.x[1] << std::endl;
    // std::cout << "V: " << cp.V[0][0] << ", " << cp.V[0][1] << std::endl;
    // std::cout << "V: " << cp.V[1][0] << ", " << cp.V[1][1] << std::endl;
    // std::cout << "V: " << cp.V[2][0] << ", " << cp.V[2][1] << std::endl;
    // std::cout << "X: " << cp.X[0][0] << ", " << cp.X[0][1] << std::endl;
    // std::cout << "X: " << cp.X[1][0] << ", " << cp.X[1][1] << std::endl;
    // std::cout << "X: " << cp.X[2][0] << ", " << cp.X[2][1] << std::endl;

    // printf("Saddle point found at %f, %f\n", cp.x[0], cp.x[1]);
    // get Jacobian
    //目前每个struct结构只包括x（坐标）,eig_vec（特征向量，由J算出来的）,type（类型）
    //需要修改成包括simplexid？
    // print eigenvector

    // std::cout << "Eigenvector1: " << cp.eig_vec[0][0] << ", " << cp.eig_vec[0][1] << std::endl;

    //std::cout << "Eigenvector2: " << cp.eig_vec[1][0] << ", " << cp.eig_vec[1][1] << std::endl;
    //add perturbation
    double eps = 0.01;
    //double X0[2] = {cp.x[0] + eps*cp.eig_vec[0][0], cp.x[1] + eps*cp.eig_vec[0][1]}; //direction1 positive
    // double X0[2] = {cp.x[0] - eps*cp.eig_vec[0][0], cp.x[1] - eps*cp.eig_vec[0][1]}; //direction1 negative
    // double X0[2] = {cp.x[0] + eps*cp.eig_vec[1][0], cp.x[1] + eps*cp.eig_vec[1][1]}; //direction2 positive
    // double X0[2] = {cp.x[0] - eps*cp.eig_vec[1][0], cp.x[1] - eps*cp.eig_vec[1][1]}; //direction2 negative
    double X_all_direction[4][2] = {{cp.x[0] + eps*cp.eig_vec[0][0], cp.x[1] + eps*cp.eig_vec[0][1]},
                                    {cp.x[0] - eps*cp.eig_vec[0][0], cp.x[1] - eps*cp.eig_vec[0][1]},
                                    {cp.x[0] + eps*cp.eig_vec[1][0], cp.x[1] + eps*cp.eig_vec[1][1]},
                                    {cp.x[0] - eps*cp.eig_vec[1][0], cp.x[1] - eps*cp.eig_vec[1][1]}};
    double lambda[3];
    double values[2];

    for (int i = 0; i < 4; i ++) {
      std::array<double, 2> X_start;
      std::vector<std::array<double, 2>> result_return;
      X_start[0] = X_all_direction[i][0];
      X_start[1] = X_all_direction[i][1];
      //check if inside
      if (inside(X_start,DH, DW))
        result_return = simulate_motion(X_start, h, num_steps,DH,DW);
      if(result_return.size() < num_steps){
        // fill the rest with the last element
        for (int i = result_return.size(); i < num_steps; i ++) {
          result_return.push_back(result_return.back());
        }
      }
      tracepoints.push_back(result_return);
    }
  }

    // X_start[0] = X0[0];
    // X_start[1] = X0[1];
    // std::vector<std::array<double, 2>> result_return = simulate_motion(X_start, h, num_steps,DH,DW);

    // if(result_return.size() < num_steps){
    //   // fill the rest with the last element
    //   for (int i = result_return.size(); i < num_steps; i ++) {
    //     result_return.push_back(result_return.back());
    //   }
    // }

      
  //write tracepoints to file
  std::string filename = argv[5];
  // std::string filename = "/home/mxi235/data/traceview/tracepoints.bin";
  std::ofstream file(filename, std::ios::out | std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing." << std::endl;
        return 1;
    }

    // Write the tracepoints data as binary
    int count = 0;
    std::cout << tracepoints.size() <<std::endl;
    for (const auto& row : tracepoints) {
        for (const auto& point : row) {
            file.write(reinterpret_cast<const char*>(point.data()), point.size() * sizeof(double));
            count ++;
        }
    }
    printf("write %d points\n", count);
    // Close the file
    file.close();
    std::cout << "Data written to " << filename << std::endl;



}





