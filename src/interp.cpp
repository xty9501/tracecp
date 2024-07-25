#include "math.h"
#include "interp.h"
#include "ftk/ndarray.hh"
#include <cmath>


/*
Point V0(x0, y0, z0); //（0，0，0）
Point V1(x0 + 1, y0, z0); //（1，0，0）
Point V2(x0, y0 + 1, z0); //（0，1，0）
Point V3(x0, y0, z0 + 1); //（0，0，1）
Point V4(x0 + 1, y0 + 1, z0); //（1，1，0）
Point V5(x0 + 1, y0, z0 + 1); //（1，0，1）
Point V6(x0, y0 + 1, z0 + 1); //（0，1，1）
Point V7(x0 + 1, y0 + 1, z0 + 1); //（1，1，1）
std::vector<Tetrahedron> tetrahedrons = {
        {V0,V3,V6,V7},
        {V0,V2,V6,V7},
        {V0,V3,V5,V7},
        {V0,V1,V5,V7},
        {V0,V2,V4,V7},
        {V0,V1,V4,V7},
    };
*/

// 重心坐标计算函数
void barycent3d(const Point& X0, const Point& X1, const Point& X2, const Point& X3, const Point& p, Eigen::Vector4d& lambda) {
    Eigen::Matrix4d A;
    A << X0[0], X1[0], X2[0], X3[0],
         X0[1], X1[1], X2[1], X3[1],
         X0[2], X1[2], X2[2], X3[2],
         1.0,   1.0,   1.0,   1.0;

    Eigen::Vector4d B;
    B << p[0], p[1], p[2], 1.0;

    lambda = A.inverse() * B;
}

// 判断点是否在四面体内并返回重心坐标
// bool isPointInTetrahedron(const Point& p, const Tetrahedron& tetra, Eigen::Vector4d& bary_coords) {
//     Eigen::Matrix3d mat;
//     mat.col(0) = tetra[1] - tetra[0];
//     mat.col(1) = tetra[2] - tetra[0];
//     mat.col(2) = tetra[3] - tetra[0];
//     double det = mat.determinant();
//     if (det == 0) return false;
//     Eigen::Vector3d b = mat.colPivHouseholderQr().solve(p - tetra[0]);
//     double sum_b = b.sum();
//     if (b[0] >= -1e-6 && b[1] >= -1e-6 && b[2] >= -1e-6 && sum_b <= 1 + 1e-6) {
//         bary_coords = Eigen::Vector4d(1 - sum_b, b[0], b[1], b[2]);
//         return true;
//     }
//     return false;
// }
bool isPointInTetrahedron(const Point& p, const Tetrahedron& tetra, Eigen::Vector4d& bary_coords) {
    // 预计算不变的向量差
    const Eigen::Vector3d& v0 = tetra[0];
    const Eigen::Vector3d d1 = tetra[1] - v0;
    const Eigen::Vector3d d2 = tetra[2] - v0;
    const Eigen::Vector3d d3 = tetra[3] - v0;

    // 构建矩阵并计算行列式
    Eigen::Matrix3d mat;
    mat.col(0) = d1;
    mat.col(1) = d2;
    mat.col(2) = d3;
    double det = mat.determinant();
    if (det == 0) return false;

    // 计算重心坐标
    Eigen::Vector3d b = mat.colPivHouseholderQr().solve(p - v0);
    double sum_b = b.sum();
    if ((b.array() >= -1e-6).all() && sum_b <= 1 + 1e-6) {
        bary_coords = Eigen::Vector4d(1 - sum_b, b[0], b[1], b[2]);
        return true;
    }

    return false;
}

// 找到给定点所在的四面体并返回插值坐标
std::pair<Tetrahedron, Eigen::Vector4d> findTetrahedronAndInterpolate(double x, double y, double z, const ftk::ndarray<float> &grad) {
    int x0 = std::floor(x);
    int y0 = std::floor(y);
    int z0 = std::floor(z);

    Point V0(x0, y0, z0); //（0，0，0）
    Point V1(x0 + 1, y0, z0); //（1，0，0）
    Point V2(x0, y0 + 1, z0); //（0，1，0）
    Point V3(x0, y0, z0 + 1); //（0，0，1）
    Point V4(x0 + 1, y0 + 1, z0); //（1，1，0）
    Point V5(x0 + 1, y0, z0 + 1); //（1，0，1）
    Point V6(x0, y0 + 1, z0 + 1); //（0，1，1）
    Point V7(x0 + 1, y0 + 1, z0 + 1); //（1，1，1）
    std::vector<Tetrahedron> tetrahedrons = {
        {V0,V3,V6,V7},
        {V0,V2,V6,V7},
        {V0,V3,V5,V7},
        {V0,V1,V5,V7},
        {V0,V2,V4,V7},
        {V0,V1,V4,V7},
    };

    Point point(x, y, z);
    Eigen::Vector4d bary_coords;
    for (const auto& tetra : tetrahedrons) {
        if (isPointInTetrahedron(point, tetra, bary_coords)) {
            return std::make_pair(tetra, bary_coords);
        }
    }

    throw std::runtime_error("Point is not within any tetrahedron.");
}

// 3D插值函数
void interp3d(const double p[3], double *v, const ftk::ndarray<float> &grad) {
    try {
        auto result = findTetrahedronAndInterpolate(p[0], p[1], p[2], grad);
        const Tetrahedron& tetra = result.first;
        const Eigen::Vector4d& bary_coords = result.second;

        double V[4][3];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 3; ++j) {
                V[i][j] = grad(j, static_cast<int>(tetra[i][0]), static_cast<int>(tetra[i][1]), static_cast<int>(tetra[i][2]));
            }
        }

        for (int i = 0; i < 3; ++i) {
            v[i] = bary_coords[0] * V[0][i] + bary_coords[1] * V[1][i] + bary_coords[2] * V[2][i] + bary_coords[3] * V[3][i];
        }
    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        v[0] = v[1] = v[2] = 0;  // 设置默认值或采取其他错误处理措施
    }
}

bool solve_linear3x3(const double A[3][3], const double b[3], double x[3]) {
    double det = A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1]) -
                 A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0]) +
                 A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]);

    if (std::abs(det) < std::numeric_limits<double>::epsilon()) {
        return false; // 矩阵不可逆
    }

    x[0] = (b[0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1]) -
            b[1]*(A[0][1]*A[2][2] - A[0][2]*A[2][1]) +
            b[2]*(A[0][1]*A[1][2] - A[0][2]*A[1][1])) / det;
    
    x[1] = (A[0][0]*(b[1]*A[2][2] - b[2]*A[2][1]) -
            A[1][0]*(b[0]*A[2][2] - b[2]*A[0][2]) +
            A[2][0]*(b[0]*A[1][2] - b[1]*A[0][2])) / det;
    
    x[2] = (A[0][0]*(A[1][1]*b[2] - A[1][2]*b[1]) -
            A[0][1]*(A[1][0]*b[2] - A[1][2]*b[0]) +
            A[0][2]*(A[1][0]*b[1] - A[1][1]*b[0])) / det;

    return true;
}

// 计算重心坐标
bool compute_barycentric_coordinates(const double V[4][3], const double P[3], double lambda[4]) {
    double A[3][3] = {
        {V[0][0] - V[3][0], V[1][0] - V[3][0], V[2][0] - V[3][0]},
        {V[0][1] - V[3][1], V[1][1] - V[3][1], V[2][1] - V[3][1]},
        {V[0][2] - V[3][2], V[1][2] - V[3][2], V[2][2] - V[3][2]}
    };

    double b[3] = {P[0] - V[3][0], P[1] - V[3][1], P[2] - V[3][2]};
    double x[3];

    if (!solve_linear3x3(A, b, x)) {
        return false; // 线性方程组无解
    }

    lambda[0] = x[0];
    lambda[1] = x[1];
    lambda[2] = x[2];
    lambda[3] = 1.0 - x[0] - x[1] - x[2];

    return lambda[0] >= 0.0 && lambda[0] <= 1.0 &&
           lambda[1] >= 0.0 && lambda[1] <= 1.0 &&
           lambda[2] >= 0.0 && lambda[2] <= 1.0 &&
           lambda[3] >= 0.0 && lambda[3] <= 1.0;
}

static const int tet_coords[6][4][3] = {
  {
    {0, 0, 0},
    {0, 0, 1},
    {0, 1, 1},
    {1, 1, 1}
  },
  {
    {0, 0, 0},
    {0, 1, 0},
    {0, 1, 1},
    {1, 1, 1}
  },
  {
    {0, 0, 0},
    {0, 0, 1},
    {1, 0, 1},
    {1, 1, 1}
  },
  {
    {0, 0, 0},
    {1, 0, 0},
    {1, 0, 1},
    {1, 1, 1}
  },
  {
    {0, 0, 0},
    {0, 1, 0},
    {1, 1, 0},
    {1, 1, 1}
  },
  {
    {0, 0, 0},
    {1, 0, 0},
    {1, 1, 0},
    {1, 1, 1}
  },
};

void interp2d(const double p[2], double *v,const ftk::ndarray<float> &grad){
  double X[3][2];
  double V[3][2];
  int x0 = floor(p[0]);
  int y0 = floor(p[1]);
  double x_ex = p[0] - x0;
  double y_ex = p[1] - y0;
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

//3d case:
// use ftk::inverse_lerp_s3v3 to get lambda[4] （ all lambda should be 0～1）
// then use lerp_s3v3(const T V[4][3], const T mu[4], T v[3]) to get v
// void interp3d(const double p[3], double *v, const ftk::ndarray<float> &grad) {
//   //给定一个点p，根据该点所在的四面体的顶点的值，返回该点的插值，
//   int x0 = std::floor(p[0]);
//   int y0 = std::floor(p[1]);
//   int z0 = std::floor(p[2]);
//   printf("x0: %d, y0: %d, z0: %d\n", x0, y0, z0);

//   for (int i = 0; i < 6; i++) {
//     // printf("######\n");
//     double X[4][3];
//     double V[4][3];

//     // 拷贝四面体顶点坐标
//     for (int j = 0; j < 4; j++) {
//       for (int k = 0; k < 3; k++) {
//         X[j][k] = tet_coords[i][j][k];
//       }
//     }

//     // 打印初始四面体顶点坐标
//     for (int j = 0; j < 4; j++) {
//       printf("Initial X[%d]: %f, %f, %f\n", j, X[j][0], X[j][1], X[j][2]);
//     }

//     // 将四面体顶点坐标平移到实际坐标系中
//     for (int j = 0; j < 4; j++) {
//       X[j][0] += x0;
//       X[j][1] += y0;
//       X[j][2] += z0;
//     }

//     // 打印平移后的四面体顶点坐标
//     for (int j = 0; j < 4; j++) {
//       printf("Translated X[%d]: %f, %f, %f\n", j, X[j][0], X[j][1], X[j][2]);
//     }

//     // 获取 V 的值
//     for (int j = 0; j < 4; j++) {
//       for (int k = 0; k < 3; k++) {
//         V[j][k] = grad(k, X[j][0], X[j][1], X[j][2]);
//       }
//     }

//     // 打印 V 的值
//     for (int j = 0; j < 4; j++) {
//       printf("V[%d]: %f, %f, %f\n", j, V[j][0], V[j][1], V[j][2]);
//     }

//     double lambda[4];
//     double cond;
//     bool is_inside = compute_barycentric_coordinates(X, p, lambda);
//     // printf("is_inside: %d\n", is_inside);
//     printf("lambda[0]: %f, lambda[1]: %f, lambda[2]: %f, lambda[3]: %f\n", lambda[0], lambda[1], lambda[2], lambda[3]);

//     if (is_inside) {
//       ftk::lerp_s3v3(V, lambda, v);
//       printf("v[0]: %f, v[1]: %f, v[2]: %f\n", v[0], v[1], v[2]);
//       return; // 找到包含点 p 的四面体后立即返回
//     }
//   }
// }

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

