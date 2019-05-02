cmake_minimum_required(VERSION 3.13)
project(Tema3)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(Tema3
        gpu_hashtable.cu
        gpu_hashtable.hpp
        std_hashtable.cpp
        test_map.cpp)
