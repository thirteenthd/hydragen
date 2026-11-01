cmake_minimum_required(VERSION 3.18)
project(strata_trtllm_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDA REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

add_library(strata_trtllm_plugin SHARED
    strata_paged_moe_plugin.cpp
)

target_link_libraries(strata_trtllm_plugin
    ${CUDA_LIBRARIES}
)
