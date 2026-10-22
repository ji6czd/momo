# bert

`tohoku-nlp/bert-base-japanese-char-v3` を、Momo の読み推定に使えるかを調べる実験場。
検証の知見は [docs/bert-char-v3-findings.md](docs/bert-char-v3-findings.md)。

`tools/llm_reading` には依存しない（エンコーダ本体の [src/bert/encoder_core.py](src/bert/encoder_core.py) は
そこから引き取ったもの）。必要なのは `momo-py`（データの展開）と `momors-py`（dev に w7 を添えるときだけ）。

## データ

```bash
uv run export_data                       # dataset/basic_raw.tsv から work/data/ を作り直す
uv run export_data --w7 <train だけで学習した basic_data_7.mbm>   # dev に w7 の読みを添える
```

- 原文のハッシュで train と dev に分ける（dev は約 1 割。足した文も同じ規則で振り分けるので、既存の dev は変わらない）。
- 出力: `work/data/{train,dev}_full.jsonl`（行ごとの原文とスロット）、`{train,dev}_raw.tsv`、`meta.json`。
- `basic_raw.tsv` を更新したら、その都度 `export_data` を実行する（`make_kit` は自動で実行する）。
- モデルの保存先 `work/finetuned/` に入っているモデルは、学習したときのデータのラベル表を持つ。
  データを変えたあと、`step8_predict` と `step11_audit` は現在のデータとラベル表が合うモデルだけを読める
  （`step10_infer` は保存済みのラベル表を使うので、古いモデルも動く）。

### w7 との比較に使う、train だけの w7

dev の w7 比較（`step7_head` ほか）には、train だけで学習した w7 が要る（全データで学習した w7 は dev を学習済みで、比較にならない）。

```bash
cd ../../momo-py
d=../experimentals/bert/work/w7_train && mkdir -p $d && cp ../experimentals/bert/work/data/train_raw.tsv $d/basic_raw.tsv
uv run trainer createdata --raw $d/basic_raw.tsv
uv run makedic $d/basic_data.tsv $d/single_character_dic.tsv
uv run trainer train --tsv $d/basic_data.tsv --window 7 --jobs 16 --boundary-n-estimators 200
cd ../experimentals/bert && uv run export_data --w7 work/w7_train/basic_data_7.mbm
```

（人名辞書を使うときは `--name-dict` を付ける。手順は `tools/llm_reading/02_learning_curve.py` と同じ。）

## 実験の流れ（本体固定のベクトル + 軽い分類器）

BERT 本体は公開重みのまま固定し、漢字スロットの文字ベクトルを一度だけ取り出して保存する。
読みの分類器だけを学習するので、設定を変えた実験を数分で繰り返せる。

```bash
uv run step6_extract                                 # 文字ベクトルの抽出（CPU で約5分）
uv run step7_head --vectors L1_w0                    # 分類器の学習（dev に w7 があれば比較も出る）
```

- `step6_extract` の主な指定: `--layers N`（最後の N 層の平均）、`--window N`（スロットごとに前後 N 文字だけ見せる）
- `step7_head` の主な指定: `--hidden N`（1隠れ層）、`--epochs`、`--seeds`、`--save 名前`
- `step8_predict --head 名前 "文"`: 保存した分類器で漢字の読みを表示
- 出力は `work/`（git 管理外）

## 微調整（GPU）と保存、手元での使用

```bash
uv run make_kit                          # データを作り直し、クラウド用 work/bert_kit.zip を作る
# クラウドで: uv run step9_finetune [--all-data --name full]
uv run step10_infer --name train "文"     # 保存したモデルで読みを表示（手元の CPU）
uv run step11_audit                      # 全データで学習したモデルと注釈の食い違い一覧
```

`pyproject.toml` の `[project.scripts]` に登録してあるので、`uv run <名前>` で呼べる。

## 物差し

dev の漢字スロット一致率。旧データ（dev 923 行）での記録は w7 95.4%、本体固定の分類器 96.9%、
微調整 97.7%。データを足したので、新しい dev では取り直す（w7 も train だけで学習し直したもの）。
