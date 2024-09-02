cmake_minimum_required(VERSION 3.12)
set(CMAKE_C_COMPILER "/opt/ohpc/pub/compiler/gcc/9.3.0/bin/gcc")
set(CMAKE_CXX_COMPILER "/opt/ohpc/pub/compiler/gcc/9.3.0/bin/g++")
#set(CMAKE_CXX_STANDARD 11)
project(traceview)


# 设置编译类型
set(CMAKE_BUILD_TYPE "Debug")

# 查找 FTK 包
find_package(FTK REQUIRED HINTS "/home/mxi235/CP/ftk/install/lib64/cmake/FTK/")
# find_package(FTK REQUIRED HINTS "/home/mxi235/test_cpsz/cpSZ/external/ftk/install/lib64/cmake/FTK")
# 查找zstd包
find_package(zstd REQUIRED)
# 查找 Eigen3 包
find_package(Eigen3 REQUIRED)

# 包含目录
# include_directories(${EIGEN3_INCLUDE_DIR})
# include_directories("/Users/mingzexia/Documents/Github/ftk/install/include/ftk/")
# /home/mxi235/test_cpsz/cpSZ/external/ftk/include/ftk
include_directories("./include")
# include_directories("/opt/homebrew/include")
# link_directories("/opt/homebrew/lib")

# 设置可执行文件输出路径
set(EXECUTABLE_OUTPUT_PATH ../)

# 查找源文件
file(GLOB SOURCES "src/*.cpp")

# 添加可执行文件
add_executable(test ${SOURCES} ../tests/main.cpp src/cp.cpp src/advect.cpp)
add_executable(single_traj ${SOURCES} tests/single_traj.cpp src/interp.cpp src/utilsIO.cpp include/advect.hpp include/utils.hpp include/cp.hpp)
add_executable(test3d ${SOURCES} tests/main3d.cpp src/interp.cpp src/utilsIO.cpp include/advect.hpp include/utils.hpp include/cp.hpp)

# 链接库
target_link_libraries(test PUBLIC FTK::libftk zstd::libzstd_shared)
target_link_libraries(single_traj PUBLIC FTK::libftk zstd::libzstd_shared)
target_link_libraries(test3d PUBLIC FTK::libftk zstd::libzstd_shared Eigen3::Eigen)

# 查找 OpenMP 包
find_package(OpenMP REQUIRED)


if(OpenMP_CXX_FOUND)
    # 设置 OpenMP 标志
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    
    # 链接 OpenMP 库
    target_link_libraries(test PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(single_traj PUBLIC OpenMP::OpenMP_CXX)
    # target_link_libraries(test3d PUBLIC OpenMP::OpenMP_CXX)
else()
    message(FATAL_ERROR "OpenMP not found, but it is required for this project.")
endif()