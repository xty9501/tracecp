#include <array>
#include <set>
#include <vector>
#include "interp.h"
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


template<typename Container>
bool inside(const Container& x, int DH, int DW) {
  if (x[0] <=0 || x[0] > DW-1 || x[1] <= 0 || x[1] > DH-1) return false;
  return true;
}



inline bool is_upper(const std::array<double, 2> x){
  double x_ex = x[0] - floor(x[0]);
  double y_ex = x[1] - floor(x[1]);
  if (y_ex >x_ex){
    return true;
  }
  else{
    return false;
  }
}

inline size_t get_cell_offset(const double *x, const int DW, const int DH){
  int x0 = floor(x[0]);
  int y0 = floor(x[1]);
  size_t cell_offset = 2*(y0 * (DW-1) + x0);
  if (!is_upper({x[0], x[1]})){
    cell_offset += 1;
  }
  return cell_offset;
}

// template<typename T>
// size_t get_cell_offset_3d(const T *x, const int DW, const int DH, const int DD){
//   int x0 = floor(x[0]);
//   int y0 = floor(x[1]);
//   int z0 = floor(x[2]);
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
//     // 将四面体顶点坐标平移到实际坐标系中
//     for (int j = 0; j < 4; j++) {
//       X[j][0] += x0;
//       X[j][1] += y0;
//       X[j][2] += z0;
//     }
//     double lambda[4];
//     double cond;
//     bool is_inside = compute_barycentric_coordinates(X, x, lambda);
//     if (is_inside) {
//       switch (i)
//       {
//       case 0:
//         return 0+ (x0 + y0 * DW + z0 * DW * DH)*6;
//       case 1:
//         return 2+ (x0 + y0 * DW + z0 * DW * DH)*6;
//       case 2:
//         return 1+ (x0 + y0 * DW + z0 * DW * DH)*6;
//       case 3:
//         return 4+ (x0 + y0 * DW + z0 * DW * DH)*6;
//       case 4:
//         return 3+ (x0 + y0 * DW + z0 * DW * DH)*6;
//       case 5:
//         return 5+ (x0 + y0 * DW + z0 * DW * DH)*6;        
//       }

//     }
//   }
//   printf("x: %f, %f, %f\n", x[0], x[1], x[2]);
//   throw std::runtime_error("Could not get_cell_offset_3d");
  
// }

template<typename T>
size_t get_cell_offset_3d(const T *x_, const int DW, const int DH, const int DD){
    double x = x_[0];
    double y = x_[1];
    double z = x_[2];
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
    int index = 0;
    for (const auto& tetra : tetrahedrons) {
        if (isPointInTetrahedron(point, tetra, bary_coords)) {
            return index + (x0 + y0 * DW + z0 * DW * DH) * 6;
        }
        index++;
    }
    printf("x: %f, %f, %f\n", x, y, z);
    throw std::runtime_error("Could not get_cell_offset_3d");
}



template size_t get_cell_offset_3d(const double *x, const int DW, const int DH, const int DD);

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

// template<typename T>
// std::array<size_t, 4> get_four_offsets(const T& x, const int DW, const int DH, const int DD){
//   // vertex offset
//   size_t x0 = floor(x[0]); 
//   size_t y0 = floor(x[1]);
//   size_t z0 = floor(x[2]);
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
//     // 将四面体顶点坐标平移到实际坐标系中
//     for (int j = 0; j < 4; j++) {
//       X[j][0] += x0;
//       X[j][1] += y0;
//       X[j][2] += z0;
//     }
//     double lambda[4];
//     double cond;
//     double x_temp[3] = {x[0], x[1], x[2]};
//     bool is_inside = compute_barycentric_coordinates(X, x_temp, lambda);
//     if (is_inside) {
//       std::array<size_t, 4> _Result;
//       for (int j = 0; j < 4; j++) {
//         _Result[j] = X[j][0] + X[j][1] * DW + X[j][2] * DW * DH;
//       }
//       return _Result;
//     }
//   }
//   throw std::runtime_error("could not determine the cell offset");
// }

template<typename T>
std::array<size_t,4> get_four_offsets(const T& x_, const int DW, const int DH,const int DD){
  double x = x_[0];
  double y = x_[1];
  double z = x_[2];
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
  int index = 0;
  for (const auto& tetra : tetrahedrons) {
      if (isPointInTetrahedron(point, tetra, bary_coords)) {
          //get the four vertexs
          std::array<size_t, 4> result;
          Tetrahedron temp = tetrahedrons[index];
          for (int i = 0; i < 4; i++){
            result[i] = temp[i][0] + temp[i][1] * DW + temp[i][2] * DW * DH;
          }
          return result;
      }
      index++;
  }
  printf("x: %f, %f, %f\n", x, y, z);
  printf("x0: %d, y0: %d, z0: %d\n", x0, y0, z0);

  throw std::runtime_error("Could not get_four_offsets");
}

template std::array<size_t, 4> get_four_offsets(const std::array<double, 3>& x, const int DW, const int DH, const int DD);

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

inline std::set<size_t> get_surrounding_3_cells_vertexs(double *x, const int DW, const int DH){
  std::set<size_t> surrounding_cells;
  size_t x0 = floor(x[0]);
  size_t y0 = floor(x[1]);
  if (is_upper({x[0], x[1]})){
    //current
    surrounding_cells.insert(y0 * DW + x0); //left_low
    surrounding_cells.insert(y0 * DW + x0 + DW); //top left
    surrounding_cells.insert(y0 * DW + x0 + DW + 1); //top right
    //top
    surrounding_cells.insert(y0 * DW + x0 + DW + 1+DW);
    //left
    surrounding_cells.insert(y0 * DW + x0-1);
    //right
    surrounding_cells.insert(y0 * DW + x0 + 1);
  }
  else{
    //current
    surrounding_cells.insert(y0 * DW + x0);
    surrounding_cells.insert(y0 * DW + x0 + 1);
    surrounding_cells.insert(y0 * DW + x0 + DW + 1);
    //top
    surrounding_cells.insert(y0 * DW + x0 + DW);
    //bottom
    surrounding_cells.insert(y0 * DW + x0 - DW);
    //right
    surrounding_cells.insert(y0 * DW + x0 + DW + 1 +1);

  }
  return surrounding_cells;
  
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

