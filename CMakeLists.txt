
# cmake_minimum_required(VERSION 3.12)  # 指定CMake的最低版本号
# project(traceview)  # 指定你的项目名称
# set(CMAKE_CXX_STANDARD 11)
# # set(CMAKE_BUILD_TYPE "Release")
# # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# set(CMAKE_BUILD_TYPE "Debug")

# find_package(FTK REQUIRED HINTS "/Users/mingzexia/Documents/Github/ftk/install/")
# # find_path(ZSTD_INCLUDE_DIR NAMES zstd.h HINTS "/opt/homebrew/Cellar/zstd/1.5.5/include")
# find_library(ZSTD_LIBRARY NAMES zstd HINTS "/opt/homebrew/Cellar/zstd/1.5.5/lib")

# find_package(Eigen3 REQUIRED)
# include_directories(${EIGEN3_INCLUDE_DIR})
# message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")

# # include_directories (${FTK_INCLUDE_DIR})
# include_directories("/Users/mingzexia/Documents/Github/ftk/install/include/ftk/")
# include_directories ("/Users/mingzexia/Documents/Github/tracecp/include")
# include_directories("/opt/homebrew/Cellar/zstd/1.5.5/include")
# set(EXECUTABLE_OUTPUT_PATH /Users/mingzexia/Documents/Github/tracecp/)


# file(GLOB SOURCES "src/*.cpp")
# add_executable(test ${SOURCES} ../tests/main.cpp src/cp.cpp src/advect.cpp ) 
# add_executable(single_traj ${SOURCES} tests/single_traj.cpp src/interp.cpp src/utilsIO.cpp include/advect.hpp include/utils.hpp include/cp.hpp)
# # 链接FTK库
# target_link_libraries(test FTK::libftk ${ZSTD_LIBRARY})
# target_link_libraries(single_traj FTK::libftk ${ZSTD_LIBRARY})

# add_executable(test3d ${SOURCES} tests/main3d.cpp src/interp.cpp src/utilsIO.cpp include/advect.hpp include/utils.hpp include/cp.hpp)

# target_link_libraries(test3d FTK::libftk ${ZSTD_LIBRARY} Eigen3::Eigen)

cmake_minimum_required(VERSION 3.12)
project(traceview)
set(CMAKE_CXX_STANDARD 11)

# 设置编译类型
set(CMAKE_BUILD_TYPE "Debug")

# 查找 FTK 包
find_package(FTK REQUIRED HINTS "/Users/mingzexia/Documents/Github/ftk/install/")
# 查找 ZSTD 库
find_library(ZSTD_LIBRARY NAMES zstd HINTS "/opt/homebrew/Cellar/zstd/1.5.5/lib")
# 查找 Eigen3 包
find_package(Eigen3 REQUIRED)

# 包含目录
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories("/Users/mingzexia/Documents/Github/ftk/install/include/ftk/")
include_directories("/Users/mingzexia/Documents/Github/tracecp/include")
include_directories("/opt/homebrew/Cellar/zstd/1.5.5/include")

# 设置可执行文件输出路径
set(EXECUTABLE_OUTPUT_PATH /Users/mingzexia/Documents/Github/tracecp/)

# 查找源文件
file(GLOB SOURCES "src/*.cpp")

# 添加可执行文件
add_executable(test ${SOURCES} ../tests/main.cpp src/cp.cpp src/advect.cpp)
add_executable(single_traj ${SOURCES} tests/single_traj.cpp src/interp.cpp src/utilsIO.cpp include/advect.hpp include/utils.hpp include/cp.hpp)
add_executable(test3d ${SOURCES} tests/main3d.cpp src/interp.cpp src/utilsIO.cpp include/advect.hpp include/utils.hpp include/cp.hpp)

# 链接库
target_link_libraries(test PUBLIC FTK::libftk ${ZSTD_LIBRARY})
target_link_libraries(single_traj PUBLIC FTK::libftk ${ZSTD_LIBRARY})
target_link_libraries(test3d PUBLIC FTK::libftk ${ZSTD_LIBRARY} Eigen3::Eigen)

# 设置使用 Clang 编译器（Homebrew 版本）
set(CMAKE_C_COMPILER /opt/homebrew/opt/llvm/bin/clang)
set(CMAKE_CXX_COMPILER /opt/homebrew/opt/llvm/bin/clang++)

# 手动设置 OpenMP 标志
set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
set(OpenMP_C_LIB_NAMES "omp")
set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/llvm/lib/libomp.dylib")
set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
set(OpenMP_CXX_LIB_NAMES "omp")
set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/llvm/lib/libomp.dylib")

# 查找 OpenMP 包
find_package(OpenMP REQUIRED)

# 添加 OpenMP 支持
if (OpenMP_CXX_FOUND)
    target_link_libraries(test PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(single_traj PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(test3d PUBLIC OpenMP::OpenMP_CXX)
endif()


