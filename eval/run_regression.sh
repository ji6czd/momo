#!/usr/bin/env bash
#
# 回帰テスト: dataset/eval_regression.tsv の各入力を全ウィンドウ(small/medium/large)で
# 推論し、期待するかな出力（マスあけ込み）と一致するか検証する。
#
# 使い方:
#   bash eval/run_regression.sh            # 全窓を検証
#   bash eval/run_regression.sh medium     # 特定の窓だけ
#
# 期待出力は「壊してはいけない基準」。再学習のたびに走らせ、差分が出たら
# データ追加が既存を壊していないか確認する。全ケース通れば exit 0、
# 1件でも外れたら exit 1（CI/フック向け）。

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOMO="$ROOT/momors/target/release/momo.exe"
DATA="$ROOT/dataset/eval_regression.tsv"

if [ $# -ge 1 ]; then
  MODELS=("$@")
else
  MODELS=(small medium large)
fi

if [ ! -x "$MOMO" ] && [ ! -f "$MOMO" ]; then
  echo "❌ momo.exe が見つかりません: $MOMO" >&2
  echo "   先に 'task rust'（cargo build --release）を実行してください。" >&2
  exit 2
fi
if [ ! -f "$DATA" ]; then
  echo "❌ 回帰データが見つかりません: $DATA" >&2
  exit 2
fi

# --- tsv を読む（# と空行はスキップ・タブ区切り） ---
inputs=()
expects=()
while IFS=$'\t' read -r inp exp || [ -n "${inp:-}" ]; do
  case "${inp:-}" in ''|'#'*) continue ;; esac
  inputs+=("$inp")
  expects+=("$exp")
done < "$DATA"

n=${#inputs[@]}
if [ "$n" -eq 0 ]; then
  echo "⚠️  検証ケースが0件です（$DATA）。" >&2
  exit 2
fi

total_fail=0
for m in "${MODELS[@]}"; do
  # 全入力を一括で推論（1入力=1行出力）
  mapfile -t outs < <(printf '%s\n' "${inputs[@]}" | PYTHONIOENCODING=utf-8 "$MOMO" --model "$m" 2>/dev/null)

  fail=0
  for i in $(seq 0 $((n - 1))); do
    got="${outs[i]:-<出力なし>}"
    if [ "$got" != "${expects[i]}" ]; then
      fail=$((fail + 1))
      total_fail=$((total_fail + 1))
      printf '  ✗ [%s] 入力: %s\n        期待: %s\n        実際: %s\n' \
        "$m" "${inputs[i]}" "${expects[i]}" "$got"
    fi
  done
  printf '%-7s : %d/%d pass\n' "$m" $((n - fail)) "$n"
done

echo "--------------------------------------------------"
if [ "$total_fail" -eq 0 ]; then
  echo "✅ 全 ${n} ケース × ${#MODELS[@]} 窓 すべて pass"
  exit 0
else
  echo "❌ ${total_fail} 件が不一致"
  exit 1
fi
