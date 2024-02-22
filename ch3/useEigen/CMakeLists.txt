cmake_minimum_required(VERSION 2.8)
project(useEigen)

set(CMAKE_BUILD_TYPE "Debug")
#eigen只有头文件，没有库文件，因此不用find_package
include_directories("/usr/include/eigen3")
add_executable(eigenMatrix eigenMatrix.cpp)
