
#include <iostream>
#include <fstream>
#include "utilsIO.h"
#include <cstddef> 

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

    outfile.close();
}

template float* readfile<float>(const char* file, size_t& num);
//template int* readfile<int>(const char* file, size_t& num);

template void writefile<double>(const char* file, double* data, unsigned long num_elements);
template void writefile<int>(const char* file, int* data, unsigned long num_elements);

