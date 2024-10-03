
#include <iostream>
#include <fstream>
#include "utilsIO.h"
#include <cstddef> 



// 函数用于将数据写入文件
void write_tracepoints(const Vector2D& data, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);

    if (!outFile) {
        std::cerr << "Failed to open the file for writing." << std::endl;
        return;
    }

    // 写入数据
    size_t numRows = data.size();
    outFile.write(reinterpret_cast<const char*>(&numRows), sizeof(numRows));

    for (const auto& row : data) {
        size_t numCols = row.size();
        outFile.write(reinterpret_cast<const char*>(&numCols), sizeof(numCols));

        for (const auto& element : row) {
            outFile.write(reinterpret_cast<const char*>(element.data()), sizeof(Array2D));
        }
    }

    outFile.close();
}

// 函数用于从文件中读取数据
Vector2D read_tracepoints(const std::string& filename) {
    Vector2D data;
    std::ifstream inFile(filename, std::ios::binary);

    if (!inFile) {
        std::cerr << "Failed to open the file for reading." << std::endl;
        return data;
    }

    // 读取数据
    size_t numRows;
    inFile.read(reinterpret_cast<char*>(&numRows), sizeof(numRows));

    data.resize(numRows);

    for (size_t i = 0; i < numRows; ++i) {
        size_t numCols;
        inFile.read(reinterpret_cast<char*>(&numCols), sizeof(numCols));

        data[i].resize(numCols);

        for (size_t j = 0; j < numCols; ++j) {
            inFile.read(reinterpret_cast<char*>(data[i][j].data()), sizeof(Array2D));
        }
    }

    inFile.close();
    return data;
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
    if (!fout.is_open()) {
    std::cerr << "Error: Could not open file " << file << " for writing." << std::endl;
    return;
    }
    if (!data) {
    std::cerr << "Error: data pointer is null." << std::endl;
    fout.close();
    return;
    }
    fout.write(reinterpret_cast<const char*>(&data[0]), num_elements*sizeof(Type));
    fout.close();
}

//write trajectory to file  
int write_trajectory(const std::vector<std::vector<std::array<double, 2>>>& traj, const std::string& filename) {
      //write tracepoints to file
  std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing." << std::endl;
        return 1;
    }
    int count = 0;
    std::cout << traj.size() <<std::endl;
    for (const auto& row : traj) {
        for (const auto& point : row) {
            file.write(reinterpret_cast<const char*>(point.data()), point.size() * sizeof(double));
            count ++;
        }
    }
    printf("Successfully write trajectory to file, total points: %d\n",count);
    file.close();
    return 0;
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
    printf("Successfully write %ld records to file %s\n", records.size(), filename.c_str());

    outfile.close();
}

template float* readfile<float>(const char* file, size_t& num);
template int* readfile<int>(const char* file, size_t& num);
template size_t* readfile<size_t>(const char* file, size_t& num);

template void writefile<double>(const char* file, double* data, unsigned long num_elements);
template void writefile<int>(const char* file, int* data, unsigned long num_elements);
template void writefile<size_t>(const char* file, size_t* data, unsigned long num_elements);
template void writefile<float>(const char* file, float* data, unsigned long num_elements);

template<typename Type>
void writeVectorOfVector(const std::vector<std::vector<Type>>& vec, const std::string& filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        throw std::runtime_error("Cannot open file for writing.");
    }

    for (const auto& innerVec : vec) {
        // 首先写入内部vector的大小
        size_t size = innerVec.size();
        outFile.write(reinterpret_cast<const char*>(&size), sizeof(size));
        // 然后写入内部vector的数据
        outFile.write(reinterpret_cast<const char*>(innerVec.data()), size * sizeof(Type));
    }
    outFile.close();
}

template void writeVectorOfVector<size_t>(const std::vector<std::vector<size_t>>& vec, const std::string& filename);

template<typename Type>
std::vector<std::vector<Type>> readVectorOfVector(const std::string& filename) {
    std::vector<std::vector<Type>> vec;
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        throw std::runtime_error("Cannot open file for reading.");
    }

    while (!inFile.eof()) {
        size_t size;
        // 首先读取内部vector的大小
        inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
        if (inFile.eof()) break; // 防止文件末尾的额外读取
        std::vector<Type> innerVec(size);
        // 然后读取内部vector的数据
        inFile.read(reinterpret_cast<char*>(innerVec.data()), size * sizeof(Type));
        vec.push_back(innerVec);
    }
    inFile.close();
    return vec;
}

template std::vector<std::vector<size_t>> readVectorOfVector<size_t>(const std::string& filename);

void save_trajs_to_binary(const std::vector<std::vector<std::array<double, 2>>>& data, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);

    // 记录外部 vector 的长度
    size_t outer_size = data.size();
    file.write(reinterpret_cast<const char*>(&outer_size), sizeof(outer_size));

    for (const auto& inner_vector : data) {
        // 记录每个内部 vector 的长度
        size_t inner_size = inner_vector.size();
        file.write(reinterpret_cast<const char*>(&inner_size), sizeof(inner_size));

        // 记录内部 vector 的数据
        for (const auto& arr : inner_vector) {
            file.write(reinterpret_cast<const char*>(arr.data()), sizeof(arr));
        }
    }

    file.close();
}

void save_trajs_to_binary_3d(const std::vector<std::vector<std::array<double, 3>>>& data, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);

    // 记录外部 vector 的长度
    size_t outer_size = data.size();
    file.write(reinterpret_cast<const char*>(&outer_size), sizeof(outer_size));

    for (const auto& inner_vector : data) {
        // 记录每个内部 vector 的长度
        size_t inner_size = inner_vector.size();
        file.write(reinterpret_cast<const char*>(&inner_size), sizeof(inner_size));

        // 记录内部 vector 的数据
        for (const auto& arr : inner_vector) {
            file.write(reinterpret_cast<const char*>(arr.data()), sizeof(arr));
        }
    }

    file.close();
}