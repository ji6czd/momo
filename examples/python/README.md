# MOMO Python サンプル (peach)

MOMO の Python バインディング [`momors-py`](../../momors/crates/momors-pyo3) を使い、
日本語テキストを **かな / 点字** に変換する最小サンプルです。

## セットアップ

```sh
uv add momors-py
```

要するに、`momors-py`をインストールするだけです。pipでもuvでもどちらでも構いません。

## 実行

```sh
uv run main.py
```

標準入力から日本語テキストを受け取り、仮名と点字を表示します。空行を入力すると終了します。

出力例:

```text
原文      : 吾輩は猫である
かな      : ワガハイワ ネコデ アル
点字      : ⠄⠐⠡⠥⠃⠄⠀⠏⠪⠐⠟⠀⠁⠙
```

## 使っている API

- `Predictor.from_bundled(window=7)` — 同梱モデルから予測器を作成
- `predictor.predict(text)` — かな（語間スペース入り）・分かち書き・インデックス対応を返す
- `BrailleTranslator().translate(kana)` — 語間スペース入りのかなを点字に変換（言語自動判定）

型定義は `momors_py/momors_py.pyi` を参照。
