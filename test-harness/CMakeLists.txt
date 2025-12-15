cmake_minimum_required(VERSION 3.14)
project(pcre_to_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch PCRE2 for benchmarking
include(FetchContent)

FetchContent_Declare(
    pcre2
    GIT_REPOSITORY https://github.com/PCRE2Project/pcre2.git
    GIT_TAG        pcre2-10.44
)

# PCRE2 build options - disable unnecessary components
set(PCRE2_BUILD_PCRE2GREP OFF CACHE BOOL "" FORCE)
set(PCRE2_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PCRE2_SUPPORT_UNICODE ON CACHE BOOL "" FORCE)
set(PCRE2_SUPPORT_JIT ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(pcre2)

# Generated pattern file - set via cmake -DPATTERN_FILE=...
set(PATTERN_FILE "generated/pattern_split.cpp" CACHE STRING "Path to generated pattern file")

# Main executable for test/benchmark harness
add_executable(runner
    main.cpp
    unicode.cpp
    unicode-data.cpp
    ${PATTERN_FILE}
)

target_include_directories(runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor
)

target_link_libraries(runner PRIVATE pcre2-8)
target_compile_definitions(runner PRIVATE PCRE2_CODE_UNIT_WIDTH=8)
