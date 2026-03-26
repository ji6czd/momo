# クロスコンパイルツールチェーンを指定
# 使用例: cmake -DCROSS_COMPILE=riscv64 ..
if(DEFINED CROSS_COMPILE)
    set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/toolchain-${CROSS_COMPILE}.cmake)
endif()

cmake_minimum_required(VERSION 3.20)

# Please rewrite the following 3 lines.
set(PROJECT_NAME momo)
set(APP_NAME momo)
set(PROJECT_VERSION 0.5.0)

project(${PROJECT_NAME} VERSION ${PROJECT_VERSION} LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options("/utf-8" "/W4")
endif()

add_subdirectory(src)
add_subdirectory(crfsuite)
add_subdirectory(liblbfgs)

# main application
add_executable(${APP_NAME} main.cpp)
target_link_libraries(${APP_NAME} PRIVATE ${PROJECT_NAME}-lib crfsuite-lib lbfgs-lib)
