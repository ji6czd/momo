# ソースを指定してください:
set(SOURCES
    crf_features.cpp
    utils.cpp
    crfsuite_impl.cpp
    predictor.cpp
    trainer.cpp
)

add_library(${PROJECT_NAME}-lib STATIC ${SOURCES})
target_include_directories(${PROJECT_NAME}-lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../crfsuite/include)
