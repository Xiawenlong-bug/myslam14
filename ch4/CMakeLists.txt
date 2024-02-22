cmake_minimum_required(VERSION 2.8)

project(useSophus)
find_package(Sophus REQUIRED)


include_directories(${Sophus_INCLUDE_DIRS}) #找到头文件的路径


# Eigen
include_directories("/usr/include/eigen3")

add_executable(useSophus useSophus.cpp)

#采用对象的方式 不用再写include_directories(${xxx_INCLUDE_DIRS})
target_link_libraries(useSophus Sophus::Sophus)  

add_subdirectory(example)