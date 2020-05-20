cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(libtorch1.0_RefineDet)

SET(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_PREFIX_PATH "/data_1/Yang/project/2019/chejian/3rdparty/libtorch")
find_package(Torch REQUIRED)

set(CMAKE_PREFIX_PATH "/home/yhl/software_install/opencv3.2")
find_package(OpenCV REQUIRED)


#include_directories("/data_1/Anaconda1105/include")
#link_directories("/data_1/Anaconda1105/lib")

LINK_DIRECTORIES("/data_1/Anaconda1105/lib/")

add_executable(example-app example-app.cpp refine_det.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
target_link_libraries(example-app ${OpenCV_LIBS})
set_property(TARGET example-app PROPERTY CXX_STANDARD 11)
