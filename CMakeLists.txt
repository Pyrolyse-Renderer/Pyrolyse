cmake_minimum_required(VERSION 4.0)
project(Pyrolyse CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)

add_library(PyrolyseCoreLib "src/pyrmaths.cu" "src/pyrexport.cu" "src/pyrrender.cu")
add_executable(Pyrolyse main.cu)
target_link_libraries(Pyrolyse PRIVATE PyrolyseCoreLib)
target_include_directories(PyrolyseCoreLib PUBLIC include)

set_target_properties(Pyrolyse PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(PyrolyseCoreLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
