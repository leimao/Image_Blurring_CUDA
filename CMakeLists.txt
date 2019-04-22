cmake_minimum_required( VERSION 3.10.0 )

project( RGBBlur VERSION 1.0.0 )

find_package( OpenCV REQUIRED )
find_package( CUDA REQUIRED )

set ( CMAKE_CXX_FLAGS "-no-pie" )

include_directories( ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} )
cuda_add_executable( ${PROJECT_NAME} main.cpp utils.cpp blur.cu )
target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBRARIES} ${CUDA_LIBRARIES})