#!/usr/bin/env bash
# キットの中で、BERT の微調整をまとめて回す（make_kit がキットに入れる）。
#
#   bash train_all.sh            # 6 本: train / full（--all-data）× 読みだけ・同時学習・同時学習＋断片
#   bash train_all.sh collapse   # 2 本: 隠れ状態の潰れを確かめる（train_joint_frag の別シードと、
#                                #       断片の最後の切れ目も学習する版）
#
# 終わったものは work/finetuned/<名前>/meta.json があるので飛ばす。
# プリエンプトされたら、VM を起動し直して同じコマンドをもう一度実行すれば、途中のものは checkpoint から続ける。
# ログは work/logs/<名前>.log。最後に、持ち帰る分を finetuned.tar にまとめる。

set -u -o pipefail
cd "$(dirname "$0")"
mkdir -p work/logs

case "${1:-all}" in
  all)
    runs=(
      "train_joint_frag|--boundary --fragments 1"
      "full_joint_frag|--all-data --boundary --fragments 1"
      "train|"
      "full|--all-data"
      "train_joint|--boundary"
      "full_joint|--all-data --boundary"
    )
    ;;
  collapse)
    runs=(
      "train_joint_frag_s1|--boundary --fragments 1 --seed 1 --name train_joint_frag_s1"
      "train_joint_frag_last|--boundary --fragments 1 --fragment-last-gap"
    )
    ;;
  *)
    echo "使い方: bash train_all.sh [all|collapse]" >&2
    exit 2
    ;;
esac
runs=("smoke_joint_frag_last|--smoke --boundary --fragments 1 --fragment-last-gap" "${runs[@]}")

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
