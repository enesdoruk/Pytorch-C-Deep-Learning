cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(Pytorch_2_2)

find_package(Torch REQUIRED)

add_executable(Pytorch_2_2 Pytorch_2_2.cpp)
target_link_libraries(Pytorch_2_2 "${TORCH_LIBRARIES}")
set_property(TARGET Pytorch_2_2 PROPERTY CXX_STANDARD 14)

