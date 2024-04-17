#include <cstddef>
#include <vector>   // 对于 std::vector
#include <string>   // 对于 std::string
#include <array>
#include <fstream>
#include "cp.hpp"

using Array2D = std::array<double, 2>;
using Vector2D = std::vector<std::vector<Array2D>>;


template<typename Type>
void writefile(const char * file, Type * data, size_t num_elements);

int write_trajectory(const std::vector<std::vector<std::array<double, 2>>>& traj, const std::string& filename);


template<typename Type>
Type * readfile(const char * file, size_t& num);

void writeRecordsToBinaryFile(const std::vector<record_t>& records, const std::string& filename);

void write_tracepoints(const Vector2D& data, const std::string& filename);


Vector2D read_tracepoints(const std::string& filename);

// void writeVectorOfVector(std::vector<std::vector<size_t>> vec, const std::string& filename);

// void readVectorOfVector(const std::string& filename);


template<typename Type>
void writeVectorOfVector(const std::vector<std::vector<Type>>& vec, const std::string& filename);

template<typename Type>
std::vector<std::vector<Type>> readVectorOfVector(const std::string& filename);