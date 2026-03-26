# liblbfgs ライブラリ
# ソースを指定してください:
set(SOURCES
    lib/lbfgs.c
)

add_library(lbfgs-lib STATIC ${SOURCES})
target_include_directories(lbfgs-lib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/lib)
