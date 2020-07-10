cmake_minimum_required(VERSION 2.8)

#find_package (gflags REQUIRED)
include_directories (${gflags_INCLUDE_DIR})

add_executable (testflags main.cpp)
target_link_libraries (testflags gflags)

set_target_properties(testflags PROPERTIES CXX_STANDARD 14
                      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BUILD_DIR}/bin)

install(TARGETS testflags RUNTIME DESTINATION ${CMAKE_BUILD_DIR}/bin)

