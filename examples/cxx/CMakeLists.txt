cmake_minimum_required(VERSION 3.20)

# Please rewrite the following 3 lines.
set(PROJECT_NAME momoapp)
set(APP_NAME momoapp)
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

# SDK 配布からビルドする場合: cmake -DMOMO_SDK_DIR=/path/to/momo-sdk-cpp
# リポジトリ内からビルドする場合: MOMO_SDK_DIR 不要（自動的にリポジトリ内パスを参照）
if(DEFINED MOMO_SDK_DIR)
    set(_MOMO_INCLUDE "${MOMO_SDK_DIR}/include")
    set(_MOMO_LIB     "${MOMO_SDK_DIR}/lib")
    set(_MOMO_BIN     "${MOMO_SDK_DIR}/bin")
else()
    set(_MOMO_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../momors/crates/momors-ffi/include")
    set(_MOMO_LIB     "${CMAKE_CURRENT_SOURCE_DIR}/../../momors/target/release")
    set(_MOMO_BIN     "${CMAKE_CURRENT_SOURCE_DIR}/../../momors/target/release")
endif()

target_include_directories(${APP_NAME} PRIVATE "${_MOMO_INCLUDE}")
target_link_directories(${APP_NAME} PRIVATE "${_MOMO_LIB}")
target_link_libraries(${APP_NAME} PRIVATE momors_ffi.dll.lib)

# DLL をビルド出力ディレクトリへコピーする
add_custom_command(TARGET ${APP_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        "${_MOMO_BIN}/momors_ffi.dll"
        $<TARGET_FILE_DIR:${APP_NAME}>)
