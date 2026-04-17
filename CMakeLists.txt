cmake_minimum_required(VERSION 3.16)

project(lab1
    VERSION 0.1.0
    LANGUAGES CXX
)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-fopenmp -O3 -march=native -Wall -g -DNDEBUG")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB_RECURSE LIB_SOURCES
    "${PROJECT_SOURCE_DIR}/lib/*.c"
    "${PROJECT_SOURCE_DIR}/lib/*.cpp"
    "${PROJECT_SOURCE_DIR}/lib/*.pp"
)

file(GLOB_RECURSE HEADERS
    "${PROJECT_SOURCE_DIR}/include/*.hpp"
)

set(COMMON_SOURCES
    ${LIB_SOURCES}
    ${HEADERS}
)

add_library(${PROJECT_NAME}_core ${COMMON_SOURCES})

target_include_directories(${PROJECT_NAME}_core
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

find_package(BLAS REQUIRED)
target_link_libraries(${PROJECT_NAME}_core PRIVATE BLAS::BLAS)

add_executable(regression src/regression.cpp)
target_include_directories(regression PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(regression PRIVATE ${PROJECT_NAME}_core BLAS::BLAS)

add_executable(classification src/classification.cpp)
target_include_directories(classification PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(classification PRIVATE ${PROJECT_NAME}_core BLAS::BLAS)

add_executable(cnn src/cnn.cpp)
target_include_directories(cnn PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(cnn PRIVATE ${PROJECT_NAME}_core BLAS::BLAS)

add_executable(split_dataset src/split_dataset.cpp)
target_include_directories(split_dataset PRIVATE ${PROJECT_SOURCE_DIR}/include)


# Discover all test source files under ./tests
file(GLOB_RECURSE TEST_SOURCES
    "${PROJECT_SOURCE_DIR}/tests/*.cpp"
)

add_executable(all_tests ${TEST_SOURCES})

target_include_directories(all_tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/vendor  # for doctest.hpp
)

target_link_libraries(all_tests PRIVATE ${PROJECT_NAME}_core BLAS::BLAS)
