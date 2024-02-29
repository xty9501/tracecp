
cmake_minimum_required(VERSION 3.12)  # 指定CMake的最低版本号
project(traceview)  # 指定你的项目名称
set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(CMAKE_BUILD_TYPE "Debug")

find_package(FTK REQUIRED HINTS "/Users/mingzexia/Documents/Github/ftk/install/")
# find_path(ZSTD_INCLUDE_DIR NAMES zstd.h HINTS "/opt/homebrew/Cellar/zstd/1.5.5/include")
find_library(ZSTD_LIBRARY NAMES zstd HINTS "/opt/homebrew/Cellar/zstd/1.5.5/lib")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
message(STATUS "Eigen3 include dir: ${EIGEN3_INCLUDE_DIR}")

# include_directories (${FTK_INCLUDE_DIR})
include_directories("/Users/mingzexia/Documents/Github/ftk/install/include/ftk/")
include_directories ("/Users/mingzexia/Documents/Github/tracecp/include")
include_directories("/opt/homebrew/Cellar/zstd/1.5.5/include")
set(EXECUTABLE_OUTPUT_PATH /Users/mingzexia/Documents/Github/tracecp/)


file(GLOB SOURCES "src/*.cpp")
add_executable(test ../tests/main.cpp src/interp.cpp src/utilsIO.cpp)
# add_executable(test ${SOURCES} tests/main.cpp)
add_executable(test2 ${SOURCES} tests/preserve_skeleton.cpp)
# 链接FTK库
target_link_libraries(test FTK::libftk ${ZSTD_LIBRARY})
target_link_libraries(test2 FTK::libftk ${ZSTD_LIBRARY})
