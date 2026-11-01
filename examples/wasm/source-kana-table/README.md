# 原文・かな対応表サンプル (TypeScript)

MOMO の WebAssembly バインディング [`momors-wasm`](../../../momors/crates/momors-wasm) を Node.js から使い、
日本語テキストをかなに変換して、原文の 1 文字ずつがどのかなに読まれたかを表で表示するサンプルです。

## 必要なもの

- Node.js 24 以降（`main.ts` を型除去でそのまま実行し、`using` 宣言を使うため）
- pnpm
- WebAssembly SDK（リポジトリの `dist/sdk/wasm/`）

## セットアップ

リポジトリのルートで WebAssembly SDK をビルドしてから、依存パッケージを入れます。

```sh
task sdk:wasm
cd examples/wasm/source-kana-table
pnpm install
```

`momors-wasm` は `package.json` で `link:../../../dist/sdk/wasm/pkg` として参照しています（シンボリックリンク）。
SDK を作り直しても入れ直しは要りません。
配布 ZIP（`momo-sdk-wasm-*.zip`）を使うときは、このパスを展開先の `pkg/` に書き換え、
モデルは環境変数 `MOMO_MODEL` で `data/` の `.mbm` を指定してください。

## 実行

```sh
pnpm start                 # 既定の文字列を変換
pnpm start 吾輩は猫である  # 引数の文字列を変換
pnpm check                 # 型検査（tsc）
```

既定では `dist/sdk/wasm/data/basic_data_7.mbm` を読みます。別のモデルを使うときは環境変数 `MOMO_MODEL` に `.mbm` のパスを指定します。

出力例:

```text
原文: 今日、東京でＴシャツを2枚買った。
かな: キョー、トーキョーデ Tシャツヲ 2マイ カッタ。

位置  原文  かな位置  かな
0     今    0,1,2     キョー
1     日              （前の字に含む）
2     、    3         、
3     東    4,5       トー
4     京    6,7,8     キョー
5     で    9,10      デ␣
6     Ｔ    11        T
7     シ    12,13     シャ
8     ャ              （前の字に含む）
9     ツ    14        ツ
10    を    15,16     ヲ␣
11    2     17        2
12    枚    18,19,20  マイ␣
13    買    21        カ
14    っ    22        ッ
15    た    23        タ
16    。    24        。

␣ は空白（かなの側ではマスあけ）
```

## 表の見かた

- **位置・かな位置**: 原文・かなそれぞれの文字インデックス。どちらもコードポイント単位なので、`𠮷` のようなサロゲートペアも 1 文字と数えます。
- **（前の字に含む）**: 「今日→キョー」のように何文字かをまとめて読むときは、読みが先頭の文字に付き、残りの文字には何も対応しません。「シャ」の「ャ」も同じです。
- **␣**: 空白。かなの側では語の区切り（マスあけ）で、区切りの直前の文字に付きます。

## 使っている API

- `initSync({ module })` — wasm を同期で初期化する。web ターゲットの既定の初期化は `fetch()` を使うが、Node の `fetch()` は `file:` URL を読めないので、`.wasm` のバイト列を渡す
- `new Predictor(mbmBytes)` — モデル（`.mbm`）のバイト列から予測器を作成
- `predictor.predict(text)` — `PredictionResult` を返す
  - `source` / `kana` — 原文と、マスあけ入りのかな
  - `sourceToKana` — 原文の文字位置 → かなの文字位置の配列（`Uint32Array[]`）
  - `kanaToSource` — その逆（かなの文字位置 → 原文の文字位置）

型定義は `dist/sdk/wasm/pkg/momors_wasm.d.ts` を参照。
