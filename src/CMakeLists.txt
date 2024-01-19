
cmake_minimum_required(VERSION 3.12)  # 指定CMake的最低版本号
project(traceview)  # 指定你的项目名称
set(CMAKE_CXX_STANDARD 11)

find_package(FTK REQUIRED HINTS "/Users/mingzexia/Documents/Github/ftk/install/")

# include_directories (${FTK_INCLUDE_DIR})
include_directories("/Users/mingzexia/Documents/Github/ftk/install/include/ftk/")

set(EXECUTABLE_OUTPUT_PATH /Users/mingzexia/Documents/Github/tracecp/)

add_executable(test main.cpp)
# 链接FTK库
target_link_libraries(test FTK::libftk)
