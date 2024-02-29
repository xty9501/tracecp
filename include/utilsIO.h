#include <cstddef>
#include <vector>   // 对于 std::vector
#include <string>   // 对于 std::string
#include <array>
#include <fstream>

using Array2D = std::array<double, 2>;
using Vector2D = std::vector<std::vector<Array2D>>;

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

template<typename Type>
void writefile(const char * file, Type * data, size_t num_elements);

template<typename Type>
Type * readfile(const char * file, size_t& num);

void writeRecordsToBinaryFile(const std::vector<record_t>& records, const std::string& filename);

void write_tracepoints(const Vector2D& data, const std::string& filename);


Vector2D read_tracepoints(const std::string& filename);