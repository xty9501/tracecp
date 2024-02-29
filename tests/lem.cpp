#include <interp.h>
#include <utilsIO.h>
#include <ftk/numeric/print.hh>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

typedef struct lem_struct{
  double eig_vec1[2];
  double eig_vec2[2];
  // std::complex<double> eigvalue1;
  // std::complex<double> eigvalue2;
  double eigvalue1;
  double eigvalue2;
  int type;
  lem_struct(double* eig_vec1_, double* eig_vec2_, double eigvalue1_, double eigvalue2_, int type_){
    eig_vec1[0] = eig_vec1_[0];
    eig_vec1[1] = eig_vec1_[1];
    eig_vec2[0] = eig_vec2_[0];
    eig_vec2[1] = eig_vec2_[1];
    eigvalue1 = eigvalue1_;
    eigvalue2 = eigvalue2_;
    type = type_;
  }
  lem_struct(){}
}lem_struct;

inline bool computeDeviation(const double *v1, const double *v2, const double *v3,const double error){
  if (fabs(v1[0] - v2[0]) < error && fabs(v1[1] - v2[1]) < error && fabs(v2[0] - v3[0]) < error && fabs(v2[1] - v3[1]) < error){
    return true;
  }
  else{
    return false;
  }
  
}

std::array<std::array<size_t, 2>, 3> get_X_matrix(const double *x, const int DW, const int DH){
  size_t x0 = floor(x[0]); 
  size_t y0 = floor(x[1]);
  std::array<std::array<size_t, 2>, 3> X_matrix;
  if (is_upper({x[0], x[1]})){
    // return {{x0, y0}, {x0, y0 + 1}, {x0 + 1, y0 + 1}};
    X_matrix[0] = {x0, y0};
    X_matrix[1] = {x0, y0 + 1};
    X_matrix[2] = {x0 + 1, y0 + 1};
  }
  else{
    // return {{x0, y0}, {x0 + 1, y0}, {x0 + 1, y0 + 1}};
    X_matrix[0] = {x0, y0};
    X_matrix[1] = {x0 + 1, y0};
    X_matrix[2] = {x0 + 1, y0 + 1};
  }
  return X_matrix;
}
template<typename T>
inline int isInside(const T& x, const double X[3][2]){
  //check if x is inside the triangle
  double x0 = x[0];
  double y0 = x[1];
  double x1 = X[0][0];
  double y1 = X[0][1];
  double x2 = X[1][0];
  double y2 = X[1][1];
  double x3 = X[2][0];
  double y3 = X[2][1];
  double a = (x1 - x0) * (y2 - y1) - (x2 - x1) * (y1 - y0);
  double b = (x2 - x0) * (y3 - y2) - (x3 - x2) * (y2 - y0);
  double c = (x3 - x0) * (y1 - y3) - (x1 - x3) * (y3 - y0);
  if ((a >= 0 && b >= 0 && c >= 0) || (a <= 0 && b <= 0 && c <= 0)){
    return 1;
  }
  else{
    return 0;
  }
}

// template int swap<int>(int& a, int& b);

bool converged(double old_t, double new_t, double tolerance) {
    // 检查t的相对变化是否小于容忍度
    return std::abs(new_t - old_t) < tolerance;
}

std::unordered_map<int, lem_struct> lem_store;

std::array<double,2> NewtonIter(const double pin[2], const size_t cell_offset, const ftk::ndarray<double>& data,const int DW, const int DH){
  lem_struct lem = lem_store[cell_offset];
  // double S[2][2] = {{lem.eigvectors[0][0], lem.eigvectors[1][0]}, {lem.eigvectors[0][1], lem.eigvectors[1][1]}};
  //convert S to eigen vector matrix
  Eigen::Matrix2d S;
  S << lem.eig_vec1[0], lem.eig_vec2[0], lem.eig_vec1[1], lem.eig_vec2[1];
  Eigen::Vector2d A;
  A = S.inverse() * Eigen::Vector2d(pin[0], pin[1]);
  Eigen::Matrix2d S_transpose = S.transpose();
  
  double V[3][2];
  Eigen::MatrixXd V_(3, 2);
  Eigen::Vector2d v1, v2, v3;
  if (cell_offset % 2 == 0){
    //upper triangle
    V[0][0] = data(0,cell_offset/DW,cell_offset%DW);
    V[0][1] = data(1,cell_offset/DW,cell_offset%DW);
    V[1][0] = data(0,cell_offset/DW,cell_offset%DW+1);
    V[1][1] = data(1,cell_offset/DW,cell_offset%DW+1);
    V[2][0] = data(0,cell_offset/DW+1,cell_offset%DW+1);
    V[2][1] = data(1,cell_offset/DW+1,cell_offset%DW+1);
  }
  else{
    //lower triangle
    V[0][0] = data(0,cell_offset/DW,cell_offset%DW);
    V[0][1] = data(1,cell_offset/DW,cell_offset%DW);
    V[1][0] = data(0,cell_offset/DW,cell_offset%DW+1);
    V[1][1] = data(1,cell_offset/DW,cell_offset%DW+1);
    V[2][0] = data(0,cell_offset/DW+1,cell_offset%DW);
    V[2][1] = data(1,cell_offset/DW+1,cell_offset%DW);
  }
  V_ << V[0][0], V[0][1], V[1][0], V[1][1], V[2][0], V[2][1];
  v1 << V[0][0], V[0][1];
  v2 << V[1][0], V[1][1];
  v3 << V[2][0], V[2][1];
  

  Eigen::Vector2d B = S_transpose * ((v1 - v2).cross(v2));
  Eigen::Vector2d C = S.inverse() * B;

  double a = v2.dot((v1 - v2).cross(v2));
  std::complex<double> t = 0; // could be complex or real
  std::complex<double>  f = A.dot(B); // could be complex or real
  Eigen::Vector2cd lambda;
  lambda << lem.eigvalue1, lem.eigvalue2;
  std::complex<double> f_prime = (lambda.array() * A.array() * B.array()).sum() + (C.array() * B.array()).sum(); // could be complex or real

  while(1){
    t = t - f/f_prime;
    if (std::abs(f) < 1e-6 || std::abs(f_prime) < 1e-6){
      break;
    }
    f = A.dot(B) + (lambda.array() * A.array() * B.array()).sum() * t + (C.array() * B.array()).sum() * t;
    f_prime = (lambda.array() * A.array() * B.array()).sum() + (C.array() * B.array()).sum();
  }
  Eigen::Vector2d result = A + t * B;
  return {result[0], result[1]};


}

void lem(const double v[3][2],const double X[3][2], const int DW, const int DH, const ftk::ndarray<double>& data){
  double V[3][2];
  int mark = 0; //0:parallel,1:real,2:complex, 3:extraordinary
  double J[2][2]; // jacobian
	double v_d[3][2];
  double eigvectors[2][2];
  double eigvalues[2];
	for(int i=0; i<3; i++){
		for(int j=0; j<2; j++){
			v_d[i][j] = v[i][j];
		}
	}
  //check if 3 vectors are the same
  if(computeDeviation(v[0], v[1], v[2], 1e-6)){
    //same vectors
    mark = 0; //parallel
  }
  else{
    double b_u[3] ={v_d[0][0], v_d[1][0], v_d[2][0]};
    double A_u[3][3] = {{X[0][0], X[0][1], 1}, {X[1][0], X[1][1], 1}, {X[2][0], X[2][1], 1}};
    double x_u[3];
    ftk::solve_linear3x3(A_u, b_u, x_u);
    double b_v[3] ={v_d[0][1], v_d[1][1], v_d[2][1]};
    double A_v[3][3] = {{X[0][0], X[0][1], 1}, {X[1][0], X[1][1], 1}, {X[2][0], X[2][1], 1}};
    double x_v[3];
    ftk::solve_linear3x3(A_v, b_v, x_v);
    double A[2][2] = {{x_u[0], x_u[1]}, {x_v[0], x_v[1]}};
    double B[2] = {x_u[2], x_v[2]};
    
    double delta = ftk::solve_eigenvalues2x2(A, eigvalues);
    //check for singularities(same eigenvalues)
    if (fabs(eigvalues[0] - eigvalues[1]) < 1e-6){
    //singularities(EXTRAORDINARY)
    mark = 3;
    }
    else if(delta >=0){
      // two real roots
      mark = 1;
    }
    else{
      mark = 2;
    }

  
  ftk::solve_eigenvectors2x2(A,2, eigvalues, eigvectors);
  //transpose eigvectors
  double S[2][2];
  for (int i = 0; i < 2; i++){
    for (int j = 0; j < 2; j++){
      S[j][i] = eigvectors[i][j];
    }
  }
  //check location of criticcal point
  double cp[2];
  double neg_S_inv[2][2];
  //get -A
  double neg_A[2][2] = {{-A[0][0], -A[0][1]}, {-A[1][0], -A[1][1]}};
  ftk::matrix_inverse2x2(neg_A, neg_S_inv);
  ftk::matrix2x2_vector2_multiplication(neg_S_inv, B, cp);
  //check if cp is in the cell using baricentric coordinates
  if(isInside(cp, X)){
    //cp is in the cell
    mark = 3;
  }
}

  size_t cell_offset;
  cell_offset = X[0][1] * DW + X[0][0];
  if(X[0][1] == X[1][1]){
    // lower triangle,offset+1
    cell_offset += 1;
  }
  lem_struct tmp = {eigvectors[0], eigvectors[1], eigvalues[0], eigvalues[1], mark};
  lem_store[cell_offset] = tmp;
}

int lem_exit(const double Pin[2], const size_t cell_offset, ftk::ndarray<double>& data, const int DW, const int DH){
  int mark = lem_store[cell_offset].type;
  std::array<std::array<size_t, 2>, 3> X_tmp = get_X_matrix(Pin, DW, DH);
  //convert Xtmp to double[3][2]
  double X[3][2];
  for (int i = 0; i < 3; i++){
    for (int j = 0; j < 2; j++){
      X[i][j] = X_tmp[i][j];
    }
  }
  int id; // -1: not found, 0: {upper:left cell, lower: down cell}, 1: {upper: upper cell, lower: right cell} ,2: {upper: right cell, lower: left cell}
  //0:parallel,1: extraodinary, 2:real,3:complex
  if(mark == 0){
    double direction[2] = {data(0,floor(Pin[1]),floor(Pin[0])), data(1,floor(Pin[1]),floor(Pin[0]))}; 
    double t = INFINITY;
    // 没写完
    id = -1;
  }

  if(mark == 1 || mark == 2){
    std::array<double,2> pout ={INFINITY,INFINITY};
    for (int i = 0; i < 2; i++){
      std::array<double,2> tmp = NewtonIter(Pin,cell_offset,data,DW,DH);
      if(tmp < pout){ //这里vector的比较重载了，按照每一个元素的大小进行比较，可能有问题
        pout = tmp;
        id = i;
      }
    }
  }
  if(mark == 3){
    double current_v[2] = {0};
    interp2d(Pin, current_v,data); 
    std::set<size_t> tmp_set;
    std::array<double, 2> Pout = newRK4(Pin, current_v, data, 0.0001, DH, DW,tmp_set);
    int isUpper = is_upper(Pout);
    int num_steps = 0;
    const int max_steps = 1000;

    while(isInside(Pout, X)){
      if (num_steps == max_steps){
      // could not find exit
      id = -1;
      break;
      }
      interp2d(Pout.data(), current_v,data);
      Pout = newRK4(Pout.data(), current_v, data, 0.0001, DH, DW,tmp_set);
      num_steps ++;
    }
    size_t Pout_offset = get_cell_offset(Pout.data(), DW, DH);
    if(isUpper){
      //upper triangle
      if (Pout_offset == cell_offset + 1){
        id = 2;
      }
      else if (Pout_offset == cell_offset + 2*(DW-1)+1){
        id = 1;
      }
      else if (Pout_offset == cell_offset-1){
        id = 0;
      }
      else{
        id = -1;
      }
        
    }
    else{
      //lower triangle
      if (Pout_offset == cell_offset - 1){
        id = 2;
      }
      else if (Pout_offset == cell_offset - 2*(DW-1) - 1){
        id = 0;
      }
      else if (Pout_offset == cell_offset+1){
        id = 1;
      }
      else{
        id = -1;
      }
    }
  }
  return id;

}