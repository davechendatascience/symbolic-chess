cmake_minimum_required(VERSION 3.16)
project(math_engine_cpp
    VERSION 0.1.0
    DESCRIPTION "Symbolic-regression-derived chess engine (UCI)"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Warnings + optimization flags
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        add_compile_options(-O3 -march=native)
    endif()
endif()

# chess-library (header-only single .hpp at third_party/)
add_library(chess_library INTERFACE)
target_include_directories(chess_library INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# Engine library — search, eval, UCI, etc.
add_library(math_engine_core STATIC
    src/uci.cpp
    src/search.cpp
    src/eval.cpp
    src/eval_dyn.cpp
    src/ast.cpp
    src/features.cpp
    src/tt.cpp
)
target_link_libraries(math_engine_core PUBLIC chess_library)
target_include_directories(math_engine_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# Main executable
add_executable(math_engine src/main.cpp)
target_link_libraries(math_engine PRIVATE math_engine_core)

# Tests (optional, built on demand)
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_executable(test_eval tests/test_eval.cpp)
    target_link_libraries(test_eval PRIVATE math_engine_core)
    add_test(NAME test_eval COMMAND test_eval)
endif()
