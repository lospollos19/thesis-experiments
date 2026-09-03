# Job A build: native host unit tests only (no CUDA, no CuPBoP).
# The CUDA/kernel path is built by translate_srad.sh, not here.
cmake_minimum_required(VERSION 3.14)
project(srad_host_tests CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GTest REQUIRED)
enable_testing()

add_executable(host_test tests/host_test.cpp)
target_include_directories(host_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(host_test GTest::gtest GTest::gtest_main)

add_test(NAME host_test COMMAND host_test)
