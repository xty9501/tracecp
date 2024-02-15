#include <iostream>
#include <vector>
#include <array>
#include <set>
#include <unordered_map>
#include <cmath>

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

inline std::array<size_t, 3> get_three_offsets(const std::array<double, 2>& x, const int DW, const int DH){
  size_t x0 = floor(x[0]); 
  size_t y0 = floor(x[1]);
  std::array<size_t, 3> result;
  if (is_upper({x[0], x[1]})){
    result[0] = y0 * DW + x0;
    result[1] = (y0+1) * DW + x0;
    result[2] = (y0+1) * DW + x0 + 1;
  }
  else{
    result[0] = y0 * DW + x0;
    result[1] = y0 * DW + x0 + 1;
    result[2] = (y0+1) * DW + x0 + 1;
  }
  return result;

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

void difftrajectory(const std::vector<std::vector<std::array<double, 2>>>& tracepoints1,const std::vector<std::vector<std::array<double, 2>>>& tracepoints2, const int DW, const int DH, std::set<size_t>& diff_offset_index) {
  // std::set<int> diff_offset;
  // std::set<int> diff_coords;
  for (size_t i =0 ; i < tracepoints1.size(); ++i){
    const auto& t1 = tracepoints1[i]; // trajectory 1,orginal
    const auto& t2 = tracepoints2[i]; // trajectory 2,decompressed
    int diff_flag = 0;
    for (int j = 0; j < t1.size(); j++){
      if (get_cell_offset(t1[j].data(),DW,DH) != get_cell_offset(t2[j].data(),DW,DH)){
        printf("trajectory %ld 's %dth point is different, from (%f, %f) to (%f, %f)\n", i, j, t1[j][0], t1[j][1], t2[j][0], t2[j][1]);
        diff_flag = 1;
        // not equal offset
        std::array<size_t, 3> offsets = get_three_offsets(t1[j], DW, DH);
        for (auto offset:offsets){
          diff_offset_index.insert(offset);
        }
        //diff_simplex.insert(adding_simplics);
      }
    }
    if (diff_flag == 0){
      printf("trajectory %ld is the same\n", i);
    }
  
  }

}

int main(int argc, char **argv){
    
}