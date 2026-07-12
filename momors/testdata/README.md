# testdata

loader のテストで使用するモデルファイル。

## ファイル

- `fixture.mbm` — `testdata/gen_fixture_mbm.py` で生成された小さなテスト用モデル。
  3 クラス × 5 特徴量。再生成するには:

  ```
  python3 testdata/gen_fixture_mbm.py
  ```

実モデル (`basic_data.mbm` 等) を置きたい場合は、サイズが大きいので
`.gitignore` に追加することを推奨。
