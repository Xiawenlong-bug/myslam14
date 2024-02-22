cmake_minimum_required(VERSION 2.8)
project(plotTrack)
set(CMAKE_BUILD_TYPE "Debug")
#eigen只有头文件，没有库文件，因此不用find_package
include_directories("/usr/include/eigen3")

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})
add_executable(plotTrajectory plotTrajectory.cpp)
target_link_libraries(plotTrajectory ${Pangolin_LIBRARIES})