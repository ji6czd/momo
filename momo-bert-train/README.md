# bert

`tohoku-nlp/bert-base-japanese-char-v3` を、Momo の読み推定に使えるかを調べる実験場。
検証の知見は [docs/bert-char-v3-findings.md](docs/bert-char-v3-findings.md)。

`tools/llm_reading` には依存しない（エンコーダ本体の [src/momo_bert_train/encoder_core.py](src/momo_bert_train/encoder_core.py) は
そこから引き取ったもの）。必要なのは `momo-train`（データの展開）と `momors-py`（dev に w7 を添えるときだけ）。
`momors-py` は `uv sync` のときに `../momors/crates/momors-pyo3` から作るので、Rust のツールチェインが要る。

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
cd ../momo-train
d=../momo-bert-train/work/w7_train && mkdir -p $d && cp ../momo-bert-train/work/data/train_raw.tsv $d/basic_raw.tsv
uv run trainer createdata --raw $d/basic_raw.tsv
uv run makedic $d/basic_data.tsv $d/single_character_dic.tsv
uv run trainer train --tsv $d/basic_data.tsv --window 7 --jobs 16 --boundary-n-estimators 200
cd ../momo-bert-train && uv run export_data --w7 work/w7_train/basic_data_7.mbm
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

step9_finetune は、5 分ごと（`--checkpoint-minutes`）と各エポックの終わりに途中の状態を
`work/finetuned/<名前>/checkpoint.pt`（重みと AdamW の状態で 1GB 余り）へ保存する。VM がプリエンプトされたら、VM を起動し直して
同じコマンドを打ち直せば続きから学習する。設定かデータが違えば使わずに最初から学習する。終われば消える。

## momo 本体の BERT 読みモード（Rust）

微調整モデルを ONNX に書き出し、momo 本体（`momors/crates/momors-bert`）で 1 行ずつ推論する。
計画と結果は [docs/bert-plan.md](../docs/bert-plan.md)。

```bash
uv run export_onnx                       # work/finetuned/train → work/onnx/train（dev で PyTorch と突き合わせる）
uv run export_onnx --name full --no-eval # 実用（全データで学習したもの）
```

- 出力は `reading_bert.onnx`（float32）、`reading_bert.int8.onnx`（int8 動的量子化）、`labels.json`、`cands.json`、`vocab.txt`、`reference.jsonl`（Rust との突き合わせ用）、`meta.json`。
- 配布用の `work/onnx/momo-bert-<名前>.zip` も作る（中は `momo-bert/`。int8 版・表・語彙・meta・NOTICE・LICENSE（元のモデルの Apache-2.0 の本文）。`--zip-f32` で float32 版も入る）。本体のリリースのあと `gh release upload <tag> work/onnx/momo-bert-full.zip` で上げる。
- `meta.json` の `basic_raw_sha256` は、学習に使った basic_raw.tsv の SHA-256（改行コードは無視）。`export_data` → `make_kit` → `step9_finetune` の順に引き継がれる。書き出すときに今の basic_raw.tsv と違えば注意が出る（本体の w7 と同じデータから作ったかの確認）。
- int8 版は、出力層と最後の 2 層の FFN 出力を float32 のまま残す。全部 int8 にすると dev が 0.35pt 落ちるため。

momo 本体（`momo` / `momo-inspect` / momors_py）は既定で BERT 読みモードを含む（feature `bert`。ONNX Runtime はビルド時に取得される）。
モデルの既定の場所は、.mbm を探す既定のディレクトリ（`MOMO_DATASET_DIR`、無ければ実行ファイルの場所）の
下の `momo-bert/`。配布用の zip をそこで展開すればよい（`dataset/momo-bert/` は git の管理外）。

```bash
cd ../momors
cargo build --release -p momors-cli
target/release/momo --bert --bert-threads 6 < input.txt          # .mbm と同じ既定のディレクトリの momo-bert/ を読む
target/release/momo --bert-dir ../momo-bert-train/work/onnx/full < input.txt   # 場所を指定する
target/release/momo-inspect trace --bert   # 根拠 BERT が出る
MOMO_BERT_DIR=../momo-bert-train/work/onnx/train cargo test -p momors-bert --release   # 参照 logit と突き合わせる
```

Rust の推論経路で w7 と比べるには、feature `bert` 付き（既定）の momors_py が要る。
`pyproject.toml` は `../momors/crates/momors-pyo3` を直接指していて、`uv sync` のときに wheel を作る
（初回は Rust のビルドが走る）。Rust のソースを変えたら、次の `uv sync`／`uv run` で作り直される。
作り直しの判断に使うファイルは、momors-pyo3 の pyproject.toml の `tool.uv.cache-keys`。

```bash
uv sync
uv run step12_rust_eval                  # w7（work/w7_train）と w7＋BERT（work/onnx/train）を dev で比べる
```

## 物差し

dev の漢字スロット一致率。旧データ（dev 923 行）での記録は w7 95.4%、本体固定の分類器 96.9%、
微調整 97.7%。データを足したので、新しい dev では取り直す（w7 も train だけで学習し直したもの）。
