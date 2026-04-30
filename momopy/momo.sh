#!/bin/env bash
set -eu
raw=../dataset/basic_raw.tsv
tsv=../dataset/basic_data.tsv
model=../dataset/basic_data
cmd=momo
# 学習用関数
check() {
    echo "Training..."
    uv run ${cmd} createdata --raw=${raw}
}

train() {
    echo "Training..."
    uv run ${cmd} createdata --raw=${raw}
    uv run ${cmd} train --tsv=${tsv} --window 7
    mv ${model}.zip ${model}_7.zip
    mv ${model}.mbm ${model}_7.mbm
}

# 予測用関数
predict() {
    echo "Predicting..."
    uv run ${cmd} predict --model=${model}_7.zip
}

translate() {
    echo "Translating..."
    uv run ${cmd} translate
}

label() {
    echo "Labeling..."
    uv run ${cmd} label --model=${model}_7.zip
}

build_model() {
    echo "Building model..."
    uv run ${cmd} createdata --raw=${raw}
    uv run ${cmd} train --tsv=${tsv} --window 7 --model ${model}_7
    uv run ${cmd} train --tsv ${tsv} --window 5 --model ${model}_5
    uv run ${cmd} train --tsv ${tsv} --window 3 --model ${model}_3
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
