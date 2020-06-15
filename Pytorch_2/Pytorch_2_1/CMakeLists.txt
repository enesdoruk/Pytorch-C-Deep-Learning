cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(Pytorch_2_1)

find_package(Torch REQUIRED)

add_executable(Pytorch_2_1 Pytorch_2_1.cpp)
target_link_libraries(Pytorch_2_1 "${TORCH_LIBRARIES}")
set_property(TARGET Pytorch_2_1 PROPERTY CXX_STANDARD 14)

