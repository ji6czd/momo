# CRFSuiteライブラリ
# ソースを指定してください:
set(SOURCES
    lib/crf/src/crf1d_encode.c
    lib/crf/src/crf1d_context.c
    lib/crf/src/crf1d_feature.c
    lib/crf/src/crf1d_model.c
    lib/crf/src/crf1d_tag.c
    lib/crf/src/crfsuite.c
    lib/crf/src/crfsuite_train.c
    lib/crf/src/dataset.c
    lib/crf/src/dictionary.c
    lib/crf/src/holdout.c
    lib/crf/src/logging.c
    lib/crf/src/params.c
    lib/crf/src/quark.c
    lib/crf/src/rumavl.c
    lib/crf/src/train_arow.c
    lib/crf/src/train_averaged_perceptron.c
    lib/crf/src/train_l2sgd.c
    lib/crf/src/train_lbfgs.c
    lib/crf/src/train_passive_aggressive.c
    lib/cqdb/src/cqdb.c
    lib/cqdb/src/lookup3.c
)

add_library(crfsuite-lib STATIC ${SOURCES})
target_include_directories(crfsuite-lib PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/crf
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/cqdb/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../liblbfgs/include)
