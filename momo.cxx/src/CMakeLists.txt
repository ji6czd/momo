# ソースを指定してください:
set(SOURCES
    utils.cpp
    crf_features.cpp
    predictor.cpp
)

add_library(${PROJECT_NAME}-lib STATIC ${SOURCES})
target_include_directories(${PROJECT_NAME}-lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../crfsuite/include)
