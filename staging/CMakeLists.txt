cmake_minimum_required(VERSION 3.14)
project(unicode_split_test CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(unicode_split_test
    main.cpp
    unicode.cpp
    unicode-data.cpp
    unicode-split.cpp
)

target_include_directories(unicode_split_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
