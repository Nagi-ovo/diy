cmake_minimum_required(VERSION 3.10)

project(OpenGL-lab1 VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${CMAKE_SOURCE_DIR}/dependencies/include)

link_directories(${CMAKE_SOURCE_DIR}/dependencies/library)

set(SOURCES
    ${CMAKE_SOURCE_DIR}/main.cpp
    ${CMAKE_SOURCE_DIR}/glad.c
)

add_executable(app ${SOURCES})

target_link_libraries(app
    ${CMAKE_SOURCE_DIR}/dependencies/library/libglfw.3.4.dylib
    ${CMAKE_SOURCE_DIR}/dependencies/library/libglm.dylib
    "-framework OpenGL"
    "-framework Cocoa"
    "-framework IOKit"
    "-framework CoreVideo"
    "-framework CoreFoundation"
)

target_compile_options(app PRIVATE
    -Wall
    -fdiagnostics-color=always
    -fcolor-diagnostics
    -fansi-escape-codes
    -g
    -Wno-deprecated-declarations
)