cmake_minimum_required(VERSION 3.16)

project(lab1
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-fsanitize=undefined -Wall")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# === Project sources ===
file(GLOB_RECURSE LIB_SOURCES
    "${PROJECT_SOURCE_DIR}/lib/*.c"
    "${PROJECT_SOURCE_DIR}/lib/*.cpp"
    "${PROJECT_SOURCE_DIR}/lib/*.pp"
)

file(GLOB_RECURSE SRC_SOURCES
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
)

file(GLOB_RECURSE HEADERS
    "${PROJECT_SOURCE_DIR}/include/*.hpp"
)

set(PROJECT_SOURCES
    ${LIB_SOURCES}
    ${SRC_SOURCES}
    ${HEADERS}  # optional
)

# Main executable
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

# === Tests ===
# Discover all test source files under ./tests
file(GLOB_RECURSE TEST_SOURCES
    "${PROJECT_SOURCE_DIR}/tests/*.cpp"
)

# Combine test sources with library sources (so tests can use the implementation)
add_executable(all_tests ${TEST_SOURCES} ${LIB_SOURCES})

target_include_directories(all_tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/include/vendor  # for doctest.hpp
)