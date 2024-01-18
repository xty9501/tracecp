cmake_minimum_required(VERSION 3.12)  # 指定CMake的最低版本号

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(FTK_INCLUDE_DIR /home/mxi235/cpSZ_naive/external/ftk/install/include)

# 等同于cmake -DFTK_DIR=/home/mxi235/CP/ftk/debuginstall/lib64/cmake/FTK .
list(APPEND CMAKE_PREFIX_PATH "/home/mxi235/CP/ftk/debuginstall")

project(traceview)  # 指定你的项目名称

# 查找FTK库
find_package(FTK REQUIRED)

# 添加你的可执行文件或库的源文件
add_executable(test main.cpp)

# 指定可执行文件的输出路径
set(EXECUTABLE_OUTPUT_PATH /home/mxi235/traceview)

# 包含FTK库的头文件路径
include_directories(${FTK_INCLUDE_DIR})

# 链接FTK库
target_link_libraries(test FTK::libftk)
