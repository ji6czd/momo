# testdata

loader のテストで使用するモデルファイル。

## ファイル

- `fixture.mbm` — `testdata/gen_fixture_mbm.py` で生成された小さなテスト用モデル。
  3 クラス × 5 特徴量。再生成するには:

  ```
  python3 testdata/gen_fixture_mbm.py
  ```

- `fixture.mbmf` — `testdata/gen_fixture_mbmf.py` で生成された、`fixture.mbm`
  の量子化前 (float32) 版。`gen_fixture_mbm.py` の語彙・ラベル・人名辞書・
  単一文字辞書・量子化スケールをそのまま再利用し、重みを `int8_val * scale`
  で機械的に導出しているため、`fixture.mbm` と厳密に同じ実値を持つ
  （`MomoModel` と `FloatMomoModel` の `predict()` 結果が完全一致することの
  クロスチェックに使う）。再生成するには:

  ```
  python3 testdata/gen_fixture_mbmf.py
  ```

  `fixture.mbm` を変更したときは、こちらも合わせて再生成すること。

- `fixture_delta.mbm` — `gen_fixture_mbm.py` が `fixture.mbm` と同時に書く、同じモデルの
  差分 + varint レイアウト版（version 0x0A、ヘッダ flags bit1）。ローダーが両方を
  同じメモリ構造に展開することのテストに使う。

- `fixture_gbdt.mbm` / `fixture_gbdt.mbmf` / `fixture_gbdt_delta.mbm` —
  `testdata/gen_fixture_mbm_gbdt.py` で生成する GBDT 境界モデル版。
  `fixture_gbdt_delta.mbm` は語彙キー・CSC・GBDT `cats` の 3 種類すべてが
  差分 + varint になる唯一のフィクスチャ。再生成するには:

  ```
  python3 testdata/gen_fixture_mbm_gbdt.py
  ```

実モデル (`basic_data.mbm` 等) を置きたい場合は、サイズが大きいので
`.gitignore` に追加することを推奨。同様に `basic_data.mbmf`
（`momo_py.exporter.export_float()` が書き出す量子化前サイドカー、
`.mbm` の学習ツール `momopy train` を実行すると自動生成される）も
サイズが大きいため `.gitignore` に追加することを推奨。
