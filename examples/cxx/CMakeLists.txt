cmake_minimum_required(VERSION 3.20)

# Please rewrite the following 3 lines.
set(PROJECT_NAME momotransjpn)
set(APP_NAME momotransjpn-app)
set(PROJECT_VERSION 1.0.0)

project(${PROJECT_NAME} VERSION ${PROJECT_VERSION} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


if(MSVC)
    add_compile_options("/utf-8" "/W4")
endif()

# main application
add_executable(${APP_NAME} main.cpp)
target_link_directories(${APP_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(${APP_NAME} PRIVATE momors_ffi.dll.lib)
