cmake_minimum_required(VERSION 3.14)
project(regex_test CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generated pattern file - set via cmake -DPATTERN_FILE=...
set(PATTERN_FILE "generated/pattern_split.cpp" CACHE STRING "Path to generated pattern file")

# Test harness executable
add_executable(regex_test
    test-main.cpp
    unicode.cpp
    unicode-data.cpp
    ${PATTERN_FILE}
)

target_include_directories(regex_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor
)
