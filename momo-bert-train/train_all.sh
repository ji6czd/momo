#!/usr/bin/env bash
# キットの中で、BERT の微調整をまとめて回す（make_kit がキットに入れる）。
#
#   bash train_all.sh            # 6 本: train / full（--all-data）× 読みだけ・同時学習・同時学習＋断片
#   bash train_all.sh unit       # 2 本: 同時学習＋マスあけの単位で切った断片（train_joint_frag_unit・full_joint_frag_unit）
#   bash train_all.sh seed       # 3 本: --seed 1 の取り直しで揺れを見る（train_joint_frag_unit_s1・
#                                #       train_joint_s1・full_joint_frag_unit_s1）
#   bash train_all.sh collapse   # 2 本: 隠れ状態の潰れを確かめる（語の途中でも切る free の断片で、
#                                #       train_joint_frag の別シードと、断片の最後の切れ目も学習する版）
#   bash train_all.sh large      # 9 本: tohoku-nlp/bert-large-japanese-char-v2 を char-v3 と同じデータで比べる
#                                #       （large は学習率 2e-5 を軸に 1e-5・3e-5 とシード 3 本、char-v3 もシード 3 本、
#                                #       配る形の large_full も。重みを持ち帰るのは large_train・large_full・
#                                #       char-v3 の seed 0 だけで、ほかは --no-weights）
#
# 終わったものは work/finetuned/<名前>/meta.json があるので飛ばす。
# プリエンプトされたら、VM を起動し直して同じコマンドをもう一度実行すれば、途中のものは checkpoint から続ける。
# ログは work/logs/<名前>.log。最後に、持ち帰る分を finetuned.tar にまとめる。

set -u -o pipefail
cd "$(dirname "$0")"
mkdir -p work/logs

case "${1:-all}" in
  all)
    smoke="smoke_joint_frag_unit|--smoke --boundary --fragments 1"
    runs=(
      "train_joint_frag_unit|--boundary --fragments 1"
      "full_joint_frag_unit|--all-data --boundary --fragments 1"
      "train|"
      "full|--all-data"
      "train_joint|--boundary"
      "full_joint|--all-data --boundary"
    )
    ;;
  unit)
    smoke="smoke_joint_frag_unit|--smoke --boundary --fragments 1"
    runs=(
      "train_joint_frag_unit|--boundary --fragments 1"
      "full_joint_frag_unit|--all-data --boundary --fragments 1"
    )
    ;;
  seed)
    smoke="smoke_joint_frag_unit|--smoke --boundary --fragments 1"
    runs=(
      "train_joint_frag_unit_s1|--boundary --fragments 1 --seed 1 --name train_joint_frag_unit_s1"
      "train_joint_s1|--boundary --seed 1 --name train_joint_s1"
      "full_joint_frag_unit_s1|--all-data --boundary --fragments 1 --seed 1 --name full_joint_frag_unit_s1"
    )
    ;;
  collapse)
    smoke="smoke_joint_frag_last|--smoke --boundary --fragments 1 --fragment-cut free --fragment-last-gap"
    runs=(
      "train_joint_frag_s1|--boundary --fragments 1 --fragment-cut free --seed 1 --name train_joint_frag_s1"
      "train_joint_frag_last|--boundary --fragments 1 --fragment-cut free --fragment-last-gap"
    )
    ;;
  large)
    # 大きさだけが違う比較（語彙・事前学習のデータは char-v3 と同じ）。設定は本命の train_joint_frag_unit。
    # 役に立つ順に並べる（途中で止めても、先の分だけで比べられる）
    smoke="smoke_joint_frag_unit|--smoke --boundary --fragments 1"
    L="--model tohoku-nlp/bert-large-japanese-char-v2 --boundary --fragments 1"
    V3="--boundary --fragments 1"
    runs=(
      "large_train_joint_frag_unit|$L --lr 2e-5 --name large_train_joint_frag_unit"
      "train_joint_frag_unit|$V3"
      "large_train_joint_frag_unit_lr1e-5|$L --lr 1e-5 --no-weights --name large_train_joint_frag_unit_lr1e-5"
      "large_train_joint_frag_unit_lr3e-5|$L --lr 3e-5 --no-weights --name large_train_joint_frag_unit_lr3e-5"
      "large_train_joint_frag_unit_s1|$L --lr 2e-5 --seed 1 --no-weights --name large_train_joint_frag_unit_s1"
      "large_train_joint_frag_unit_s2|$L --lr 2e-5 --seed 2 --no-weights --name large_train_joint_frag_unit_s2"
      "train_joint_frag_unit_s1|$V3 --seed 1 --no-weights --name train_joint_frag_unit_s1"
      "train_joint_frag_unit_s2|$V3 --seed 2 --no-weights --name train_joint_frag_unit_s2"
      "large_full_joint_frag_unit|$L --lr 2e-5 --all-data --name large_full_joint_frag_unit"
    )
    ;;
  *)
    echo "使い方: bash train_all.sh [all|unit|seed|collapse|large]" >&2
    exit 2
    ;;
esac
runs=("$smoke" "${runs[@]}")

failed=()
done_runs=()
for r in "${runs[@]}"; do
  name=${r%%|*}
  opts=${r#*|}
  [[ $name == smoke* ]] || done_runs+=("finetuned/$name")
  if [ -f "work/finetuned/$name/meta.json" ]; then
    echo "済み: $name"
    continue
  fi
  echo "=== $name: step9_finetune $opts"
  # shellcheck disable=SC2086 # opts は空白区切りの引数列として展開する
  if ! uv run step9_finetune $opts 2>&1 | tee "work/logs/$name.log"; then
    echo "失敗: $name（work/logs/$name.log）"
    [[ $name == smoke* ]] && exit 1 # 動作確認が通らなければ先へ進まない
    failed+=("$name")
  fi
done

# 持ち帰る分（今回の組の分だけ。途中の checkpoint は除く）
existing=()
for d in "${done_runs[@]}"; do
  log="logs/$(basename "$d").log"
  [ -d "work/$d" ] && existing+=("$d")
  [ -f "work/$log" ] && existing+=("$log")
done
tar cf finetuned.tar --exclude='checkpoint.pt' -C work "${existing[@]}"
echo "持ち帰る: $(pwd)/finetuned.tar（展開すると finetuned/ と logs/。finetuned/ の中身を momo-bert-train/work/finetuned/ に置く）"
if [ ${#failed[@]} -gt 0 ]; then
  echo "失敗したもの: ${failed[*]}（もう一度実行すると、それだけ回し直す）"
  exit 1
fi
