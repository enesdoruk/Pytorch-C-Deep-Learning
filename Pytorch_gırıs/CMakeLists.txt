cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(Pytorch_giris)

find_package(Torch REQUIRED)

add_executable(Pytorch_giris Pytorch_giris.cpp)
target_link_libraries(Pytorch_giris "${TORCH_LIBRARIES}")
set_property(TARGET Pytorch_giris PROPERTY CXX_STANDARD 14)

