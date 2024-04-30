#include <array>
#include <set>
#include <vector>




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

