#!/bin/env bash
set -eu
raw=dataset/training_raw.tsv
tsv=dataset/training_data.tsv
model=dataset/training_data.model
cmd=./src/momobrl/neomomo.py
# 学習用関数
check() {
    echo "Training..."
    uv run $cmd createdata --raw=$raw
}

train() {
    echo "Training..."
    uv run $cmd createdata --raw=$raw
    uv run $cmd train --tsv=$tsv
}

# 予測用関数
predict() {
    echo "Predicting..."
    uv run $cmd predict --model=$model
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
    *)
        echo "Usage: $0 {train|predict}"
        exit 1
        ;;
esac
