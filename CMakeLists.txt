cmake_minimum_required(VERSION 4.0)
project(Pyrolyse CUDA CXX)

set(CMAKE_CUDA_STANDARD 20)

add_library(PyrolyseCoreLib "src/pyrmaths.cu" "src/pyrexport.cu" "src/pyrrender.cu" "include/pyrolyse/pyrimport.cuh" "src/pyrimport.cu" "src/pyrutils.cpp")
add_executable(Pyrolyse main.cu)

include(FetchContent)
FetchContent_Declare(nlohmann_json GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG v3.11.3)
FetchContent_MakeAvailable(nlohmann_json)

set_target_properties(Pyrolyse PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(PyrolyseCoreLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(PyrolyseCoreLib PRIVATE nlohmann_json)
target_link_libraries(Pyrolyse PRIVATE PyrolyseCoreLib)
target_include_directories(PyrolyseCoreLib PUBLIC include)
