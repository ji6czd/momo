# momo-bert-train

## 目的

`tohoku-nlp/bert-base-japanese-char-v3` を微調整して、momo の BERT 読みモード（`momors/crates/momors-bert`）が読む
ONNX モデルを作る。検証の実験（step2〜step8）もここにある。手順は README.md。

## ツール軍

- Pythonを使うときは、uvコマンドを使います。
- 事前の仮想環境を作成します。
