#!/bin/env bash
set -eu
raw=../dataset/basic_raw.tsv
tsv=../dataset/basic_data.tsv
cmd=momo
# 学習用関数
check() {
    echo "Training..."
    uv run ${cmd} createdata --raw=${raw}
    uv run ${cmd} train --dry-run --tsv ${tsv} --window 7
}

train() {
    echo "Training..."
    uv run ${cmd} createdata --raw=${raw}
    uv run ${cmd} train --tsv=${tsv} --window 7
}

# 予測用関数
predict() {
    echo "Predicting..."
    uv run ${cmd} predict
}

translate() {
    echo "Translating..."
    uv run ${cmd} translate
}

label() {
    echo "Labeling..."
    uv run ${cmd} label
}

build_model() {
    echo "Building model..."
    uv run ${cmd} createdata --raw=${raw}
    uv run ${cmd} train --tsv=${tsv} --window 7
    uv run ${cmd} train --tsv ${tsv} --window 5
    uv run ${cmd} train --tsv ${tsv} --window 4
    uv run ${cmd} train --tsv ${tsv} --window 3
}

# メイン処理
case "$1" in
    train)
        train
        ;;
    predict)
        predict
        ;;
    check)
        check
        ;;
    translate)
        translate
        ;;
    label)
        label
        ;;
    build)
        build_model
        ;;
    *)
        echo "Usage: ${0} {train|predict}"
        exit 1
        ;;
esac
