cmake_minimum_required(VERSION 2.8)
include(ExternalProject)

project(wrapcpppy)
add_definitions(-std=c++11 -fopenmp -Wpedantic -Wextra -Wfatal-errors -MMD -MP -pthread -fPIC -O3 -Wall)

list(APPEND CMAKE_MODULE_PATH  "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package( OpenCV REQUIRED ) 
find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(PkgConfig REQUIRED)

file(GLOB_RECURSE PyCPP_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")

include(BuildBoost) # Custom module

include_directories(
	${Boost_INCLUDE_DIR} 
	${PYTHON_INCLUDE_DIRS}
	${CMAKE_SOURCE_DIR}/include
        /usr/local/lib/python3.5/dist-packages/numpy/core/include
)
link_directories(${Boost_LIBRARY_DIR})

# Build and link the pylib module
add_library(wrapcpppy SHARED ${PyCPP_SOURCES})
target_link_libraries(wrapcpppy ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} ${LINKER_LIBS} ${OpenCV_LIBS})
add_dependencies(wrapcpppy Boost)

# Tweaks the name of the library to match what Python expects
set_target_properties(wrapcpppy PROPERTIES SUFFIX .so)
set_target_properties(wrapcpppy PROPERTIES PREFIX "")

