#!/usr/bin/env bash
# キットの中で、BERT の微調整を 6 本まとめて回す（make_kit がキットに入れる）。
#
#   bash train_all.sh
#
# train / full（--all-data）× 読みだけ・同時学習（--boundary）・同時学習＋断片（--fragments 1）。
# 新しい断片の 2 本を先に回す。終わったものは work/finetuned/<名前>/meta.json があるので飛ばす。
# プリエンプトされたら、VM を起動し直してもう一度実行すれば、途中のものは checkpoint から続ける。
# ログは work/logs/<名前>.log。最後に、持ち帰る分を finetuned.tar にまとめる。

set -u -o pipefail
cd "$(dirname "$0")"
mkdir -p work/logs

runs=(
  "smoke_joint_frag|--smoke --boundary --fragments 1"
  "train_joint_frag|--boundary --fragments 1"
  "full_joint_frag|--all-data --boundary --fragments 1"
  "train|"
  "full|--all-data"
  "train_joint|--boundary"
  "full_joint|--all-data --boundary"
)

failed=()
for r in "${runs[@]}"; do
  name=${r%%|*}
  opts=${r#*|}
  if [ -f "work/finetuned/$name/meta.json" ]; then
    echo "済み: $name"
    continue
  fi
  echo "=== $name: step9_finetune $opts"
  # shellcheck disable=SC2086 # opts は空白区切りの引数列として展開する
  if ! uv run step9_finetune $opts 2>&1 | tee "work/logs/$name.log"; then
    echo "失敗: $name（work/logs/$name.log）"
    [ "$name" = smoke_joint_frag ] && exit 1 # 動作確認が通らなければ先へ進まない
    failed+=("$name")
  fi
done

# 持ち帰る分（動作確認の smoke と、途中の checkpoint は除く）
tar cf finetuned.tar --exclude='smoke*' --exclude='checkpoint.pt' -C work finetuned logs
echo "持ち帰る: $(pwd)/finetuned.tar（展開すると finetuned/ と logs/。finetuned/ の中身を momo-bert-train/work/finetuned/ に置く）"
if [ ${#failed[@]} -gt 0 ]; then
  echo "失敗したもの: ${failed[*]}（もう一度実行すると、それだけ回し直す）"
  exit 1
fi
