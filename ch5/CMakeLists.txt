cmake_minimum_required(VERSION 2.8)

set(CMAKE_BUILD_TYPE "Debug")
# 添加c++ 11标准支持
set(CMAKE_CXX_FLAGS "-std=c++11 -O2")

# Eigen
include_directories("/usr/include/eigen3")

# 寻找OpenCV库
set(OpenCV_DIR /home/xwl/opencv-4.5.4/build)
find_package(OpenCV REQUIRED)
# 添加头文件
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(imageBasics)
#add_subdirectory(stereo)
add_subdirectory(rgbd)