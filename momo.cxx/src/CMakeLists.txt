# ソースを指定してください:
set(SOURCES
    loader.cpp
    momo_features.cpp
    predictor.cpp
    utf8.cpp
)

add_library(${PROJECT_NAME}-lib STATIC ${SOURCES})
target_include_directories(${PROJECT_NAME}-lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
