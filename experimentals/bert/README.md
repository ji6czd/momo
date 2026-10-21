# bert

`tohoku-nlp/bert-base-japanese-char-v3` を、Momo の読み推定に使えるかを調べる実験場。
検証の知見は [docs/bert-char-v3-findings.md](docs/bert-char-v3-findings.md)。

## 実験の流れ（本体固定のベクトル + 軽い分類器）

BERT 本体は公開重みのまま固定し、漢字スロットの文字ベクトルを一度だけ取り出して保存する。
読みの分類器だけを学習するので、設定を変えた実験を数分で繰り返せる。

```bash
uv run step6_extract                                 # 文字ベクトルの抽出（CPU で約5分）
uv run step7_head --vectors L1_w0                    # 分類器の学習と w7 との比較
```

- `step6_extract` の主な指定: `--layers N`（最後の N 層の平均）、`--window N`（スロットごとに前後 N 文字だけ見せる）
- `step7_head` の主な指定: `--hidden N`（1隠れ層）、`--epochs`、`--seeds`
- 出力は `work/`（git 管理外）

`pyproject.toml` の `[project.scripts]` に登録してあるので、`uv run <名前>` で呼べる
（`make_kit`、`step6_extract`、`step7_head`、`step8_predict`、`step9_finetune`、`step10_infer`）。
GPU での微調整と保存、手元での推論は `step9_finetune`、`step10_infer`（`make_kit` でクラウドへ持ち込む）。

## 比較の物差し

分割・ラベル表・辞書候補・漢字スロットの定義は `tools/llm_reading` と共有する
（[src/bert/common.py](src/bert/common.py) が `encoder_core` を取り込む）。
データは `tools/llm_reading/06_export_bundle.py` が書き出す `work/gpu_kit/bundle/` を読むので、
先にそちらを実行しておく。

dev 923 行の漢字スロット一致率で、w7（95.4%）と微調整（97.68%）を基準に見る。

## ファイル

| ファイル | 内容 |
|---|---|
| `src/bert/step2_tokenize.py` 〜 `step5_benchmark.py` | 最初の検証（トークナイザ・文字ベクトル・速度） |
| `src/bert/common.py` | データの場所と `encoder_core` の再利用 |
| `src/bert/step6_extract.py` | 漢字スロットの文字ベクトルを抽出して保存 |
| `src/bert/step7_head.py` | 保存したベクトルで分類器を学習し、w7 と比較 |
