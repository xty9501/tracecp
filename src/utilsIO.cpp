
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
  fout.write(reinterpret_cast<const char*>(&data[0]), num_elements*sizeof(Type));
  fout.close();
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

