cmake_minimum_required(VERSION 3.18)
include(FetchContent)

enable_testing()

set(CMAKE_CXX_STANDARD 20)

include_directories(
    ${PROJECT_SOURCE_DIR}/..
)

add_executable(bert_test test.cpp)
target_link_libraries(
    bert_test
    PRIVATE
    libtokenizers_cpp
)