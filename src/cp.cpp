#include "cp.hpp"
#include <iostream>
#include <complex>

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

void refill_gradient(int id,const int DH,const int DW, const float* grad_tmp, ftk::ndarray<float>& grad){
  const float * grad_tmp_pos = grad_tmp;
  for (int i = 0; i < DH; i ++) {
    for (int j = 0; j < DW; j ++) {
      grad(id, j, i) = *(grad_tmp_pos ++);
    }
  }
}

void refill_gradient_3d(int id, const int DD, const int DH, const int DW, const float* grad_tmp, ftk::ndarray<float>& grad){
  const float * grad_tmp_pos = grad_tmp;
  for (int i = 0; i < DD; i ++) {
    for (int j = 0; j < DH; j ++) {
      for (int k = 0; k < DW; k ++) {
        grad(id, k, j, i) = *(grad_tmp_pos ++);
      }
    }
  }
}

template<typename T>
static void 
check_simplex_seq_saddle(const T v[3][2], const double X[3][2], const int indices[3], int i, int j, int simplex_id, std::unordered_map<size_t, critical_point_t>& critical_points){
  int sos = 0;
  double mu[3]; // check intersection
  double cond;
  if (sos == 1){
    // for(int i=0; i<3; i++){ //skip if any of the vertex is 0 //
    // if((v[i][0] == 0) && (v[i][1] == 0)){ //
    //   return; //
    //   } //
    // } //
    // // robust critical point test
    // bool succ = ftk::robust_critical_point_in_simplex2(vf, indices);
    // if (!succ) return;

    // bool succ2 = ftk::inverse_lerp_s2v2(v, mu, &cond);
    // if (!succ2) ftk::clamp_barycentric<3>(mu);
    printf("sos is not supported\n");
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
  critical_point_t cp;
  double eig_vec[2][2]={0};
  double eig_r[2];
  std::complex<double> eig[2];
  double x[2]; // position
  ftk::lerp_s2v2(X, mu, x);
  cp.x[0] = j + x[0]; cp.x[1] = i + x[1];
  double J[2][2]; // jacobian
  ftk::jacobian_2dsimplex2(X, v, J);  
  int cp_type = 0;
  double delta = ftk::solve_eigenvalues2x2(J, eig);

  // if(fabs(delta) < std::numeric_limits<double>::epsilon())
  if (delta >= 0) { // two real roots
    if (eig[0].real() * eig[1].real() < 0) {
      cp.eig[0] = eig[0], cp.eig[1] = eig[1];
      cp.Jac[0][0] = J[0][0]; cp.Jac[0][1] = J[0][1];
      cp.Jac[1][0] = J[1][0]; cp.Jac[1][1] = J[1][1];
      cp_type = SADDLE;
      double eig_r[2];
      eig_r[0] = eig[0].real(), eig_r[1] = eig[1].real();
      ftk::solve_eigenvectors2x2(J, 2, eig_r, eig_vec);
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

  // critical_point_t cp(x, eig_vec,v,X, cp_type);


  //ftk::transpose2x2(J);
  cp.eig_vec[0][0] = eig_vec[0][0]; cp.eig_vec[0][1] = eig_vec[0][1];
  cp.eig_vec[1][0] = eig_vec[1][0]; cp.eig_vec[1][1] = eig_vec[1][1];
  
  // 这里eig_vec由（x,y）变成了（-y,x）
  // cp.eig_vec[0][0] = -eig_vec[0][1]; cp.eig_vec[0][1] = eig_vec[0][0];
  // cp.eig_vec[1][0] = -eig_vec[1][1]; cp.eig_vec[1][1] = eig_vec[1][0];

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
std::unordered_map<size_t, critical_point_t>
compute_critical_points(const T * U, const T * V, int r1, int r2){
  size_t num_elements = r1*r2;

  int indices[3] = {0};
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
  double v[3][2] = {0};
  std::unordered_map<size_t, critical_point_t> critical_points;
	for(int i=0; i<r1-1; i++){ //坑
    // if(i%100==0) std::cout << i << " / " << r1-1 << std::endl;
		for(int j=0; j<r2-1; j++){
      ptrdiff_t cell_offset = 2*(i * (r2-1) + j);
			indices[0] = i*r2 + j;
			indices[1] = (i+1)*r2 + j;
			indices[2] = (i+1)*r2 + (j+1); 
			// cell index 0
			for(int p=0; p<3; p++){
				v[p][0] = U[indices[p]];
				v[p][1] = V[indices[p]];
			}
      
      check_simplex_seq_saddle(v, X1, indices, i, j, cell_offset, critical_points);
			// cell index 1
			indices[1] = i*r2 + (j+1);
			v[1][0] = U[indices[1]], v[1][1] = V[indices[1]];			
      check_simplex_seq_saddle(v, X2, indices, i, j, cell_offset + 1, critical_points);
		}
	}
  return critical_points; 
}

// This should follow immediately after the template definition
template std::unordered_map<size_t, critical_point_t> compute_critical_points<float>(float const*, float const*, int, int);



int check_cp(double v[3][2]){
  // numeric check cp
  double cond;
  double mu[3];
  for(int i=0; i<3; i++){ //skip if any of the vertex is 0 //
    if((v[i][0] == 0) && (v[i][1] == 0)){ 
      return -1; 
      } 
    } 
  bool succ =  ftk::inverse_lerp_s2v2(v, mu, &cond);
  if (!succ) return -1;
  return 1;
}

