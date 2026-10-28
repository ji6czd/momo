# BERT 読みモード 実装計画（第1段階：1 行ずつ推論）

第1段階は 2026-10-27 に実装済み。結果は末尾の「第1段階の結果」。

## Context

- **目的は文脈による読み分け。**
  - Linear 系（w7）でも、データを倍にすれば +2% 程度は見込める。
  - BERT に期待するのは、窓の外を含めて文全体から読みを決めること。
- **momo は 1 行ずつ入力を受ける。** 前後の行が本当に関連しているかは分からないので、行はつながない。
  - 第1段階は**その 1 行の中だけで文全体を見る**。
  - 例：「避ける」をヨケル／サケルで、「市場」をイチバ／シジョーで読み分ける。
- **蒸留はしない。** 微調整済みの `tohoku-nlp/bert-base-japanese-char-v3` をそのまま使う。
  - 評価用は `momo-bert-train/work/finetuned/train`、実用は `full`。
- **対象はデスクトップ。** 実行速度はできる限り速くする。

## 段階の分け方

| 段階 | 範囲 |
|---|---|
| **1（本計画）** | ONNX 書き出し → Rust 推論 → core 組み込み → CLI / momo-inspect / pyo3 → 評価と速度 |
| 2 | FFI・C#・エディタ・MSI での配布 |
| 3（必要なら） | 前後の行の文脈、文脈つきの再学習 |

## 事前に分かったこと（データ）

- **`市場`**：basic_raw の読みは イチバ 18 件、シジョー 26 件。両方あるので、そのまま読み分けを試せる。
- **`避ける`**：basic_raw の読みは サケ 24 件、ヒ 15 件で、**ヨケは 0 件**。
  - BERT は、その字の辞書候補（basic_raw に出た読み）からしか選ばない。
  - なのでヨケルを出すには、ヨケルの例文を basic_raw に足して再学習が要る。
  - **第1段階では扱わない**（ユーザー判断）。評価のときに、ユーザーが微調整モデルを作り直す。実装は、作り直したモデルを `export_onnx` に通せば差し替えられる作りにしておく。

## 依存

- Rust：新クレート `momors-bert` だけに `ort`（ONNX Runtime）を足す。`momors-core` には依存を足さない（trait だけ）。
- 配布物：
  - `onnxruntime.dll`。`ort` の配布済みバイナリを使う。
  - モデル `reading_bert.onnx`（int8 で約 90MB / f32 で約 350MB）、`labels.json`、`cands.json`、`vocab.txt`。
- Python（`momo-bert-train`）：`onnx`、`onnxruntime` を足す。
- トークナイザのクレートは要らない。1 字 1 トークンなので、`vocab.txt` を 1 字ずつ引く。
  - 語彙に無い字は NFKC で 1 字になればそれを、だめなら [UNK]。
  - 規則は `momo-bert-train/src/momo_bert_train/encoder_core.py:95` の `CharVocab.token_id` と同じ。

## Step 1：ONNX 書き出し（Python、`momo-bert-train`）

- `src/momo_bert_train/export_onnx.py` を新規に作る。`pyproject.toml` の scripts に `export_onnx` を足す。
  - 入力：`work/finetuned/<name>/`。`step10_infer.FineTunedReader` の読み込み方をそのまま使う。
  - グラフ：入力は `input_ids [B,T]`、`attention_mask [B,T]`。出力は `logits [B,T,L]`。dynamic axes は B と T。
  - 量子化：f32 版と、`onnxruntime.quantization.quantize_dynamic` で作る int8 版の 2 つを出す。
  - 出力先：`work/onnx/<name>/`。中身は `reading_bert.onnx`、`reading_bert.int8.onnx`、`labels.json`、`cands.json`、`vocab.txt`、`reference.jsonl`。
    - `reference.jsonl` は、数文分の入力 id と、漢字位置の logit。Rust との突き合わせに使う。
- 確認：PyTorch と onnxruntime（f32 / int8）で、dev の漢字スロット一致率と logit の最大差を出す。
  - int8 の低下が 0.1pt 以内なら、int8 を既定にする。

## Step 2：`momors/crates/momors-bert`（新規）

- ワークスペースの `members` に入れる。`default-members` には入れない（CLI が feature `bert` で取り込む）。
- `vocab.rs`：文字 → id（上記の規則）。
- `labels.rs`：
  - `labels.json` を、`MomoModel.read_classes()` の class id へ解決する。継続 `""` は `"---"` に読み替える。
  - 解決できないラベルは警告を出し、その候補は捨てる。
  - `cands.json`（文字 → ラベル番号）は、ロード時に class id の並びへ直す。
- `reader.rs`：`BertReader`。
  - `ort::Session` を 1 つ持つ。設定はグラフ最適化 Level3、スレッド数は物理コア数。
  - `read_line(&self, text: &str) -> Vec<Option<SlotScore>>` を、正規化後の文字位置ごとに返す。
    - `SlotScore` は、候補内 softmax の `(class_id, prob)` の並び（確率の降順）。
    - 対象は、`char_type` が Kanji / Numeric（漢数字）の位置だけ。
    - [UNK] の位置と候補が無い位置は `None` を返す（LR に任せる）。
  - 長い行の扱い：510 字を超える行は 510 字以下に区切って流す。
    - 区切りは「。！？」の後ろを優先し、無ければ機械的に切る。
    - 各区切りに前後 64 字の重なりを付け、重なりの部分の結果は捨てる。
  - 速度の打ち手：
    - 漢字を含まない行（または区切り）は流さない。
    - 推論の前後で作業用の領域を使い回す。
    - 出力は漢字位置の行だけ取り出す。
- テスト：
  - `reference.jsonl` と logit が一致すること（f32 は < 1e-3、int8 は argmax の一致率）。
  - 長い行を区切ったときに、重なりも欠けもなく覆えていること。

## Step 3：`momors-core` への組み込み

- **trait `SlotReader`**（新規 `src/slot_reader.rs`）：
  - `fn read_line(&self, normalized: &[char]) -> Vec<Option<SlotScore>>`。
  - `SlotScore` も core に置き、`momors-bert` がこの trait を実装する。
  - `Predictor` に `slot_reader: Option<Box<dyn SlotReader + Send + Sync>>` と、それを設定する `set_slot_reader(...)` を足す。
- **`prediction.rs`**：
  - `predict`（659 行）で正規化したあと、`predict_with_brackets` の前に、行全体で 1 回 `read_line` を呼ぶ。
    - 括弧の注釈も含めた行全体を、BERT に読ませるため。
  - 結果は、正規化後の位置をキーにした表として、`predict_with_brackets` → `predict_normalized_with_feature_text` に渡す。
    - 注釈 span の再帰（719-756 行）では、span の開始位置のずれ（offset）を足して引く。
  - `read_argmax`（1537-1598 行）を呼ぶ 4 か所（1045、1185-1199、1220、1297）では、その位置に BERT の結果があれば、そちらを使う。
    - 単漢字辞書による絞り込みは、BERT の候補表がすでに担っている。
    - 「々」は連濁 2 択（1555-1579 行）の候補に、BERT の確率を当てて選ぶ。2 択がどちらも BERT の候補に無ければ、従来の規則に任せる。
    - BERT が決めた位置では、LR の `compute_read_scores` を呼ばない。
  - `decided[i]` には BERT のラベルの class id を入れる。後ろの LR の文字が `read_p1-3` でそれを見るようにするため。
  - 自信度：BERT の確率をそのまま使う。
    - 漢数字ルールの閾値（`numeric_confidence_threshold`）と、人名辞書の固定読みのフォールバック（conf < 0.5）は、ひとまずそのまま使い、Step 5 で見直す。
  - カスタム辞書・助数詞パスは BERT より先に効く（順序は変えない）。
- **`DecisionSource::Bert`**：
  - `as_str`（93-106 行）、`crates/momors-cli/src/bin/inspect.rs` の `decision_color` に足す。
  - `phase.rs` には BERT の段を足す（`PHASE_NAMES`、配列長）。

## Step 4：入口（CLI / momo-inspect / pyo3）

- **`momors-cli`**：
  - feature `bert`（`momors-bert` を取り込む）を足す。オプションは `--bert <dir>`（Step 1 の出力ディレクトリ）と `--bert-threads N`。
  - `--bert` を付けたときは、`Predictor::set_slot_reader` を呼ぶ。
  - 入力は今までどおり 1 行ずつ処理する。
  - momo-inspect も同じにする（DecisionSource=Bert が色付きで出る）。
- **`momors-pyo3`**：
  - feature `bert` を足し、`Predictor(..., bert=<dir>)` を受けるようにする。評価スクリプトから呼ぶため。
  - wheel の更新は memory の手順どおり（`uv lock --upgrade-package momors-py --refresh-package momors-py` と `uv sync --reinstall-package`）。

## Step 5：評価と速度

- **`momo-bert-train/src/momo_bert_train/step12_rust_eval.py`**（新規）：
  - pyo3 経由で、w7 単独と w7＋BERT を dev に流す。
  - 漢字スロット一致率（`export_data.w7_readings` と同じ割り付け）と、かな文字列の一致を並べる。
  - 読みが変わったスロットの一覧を出す。w7 が誤り・BERT が正解、およびその逆。
- **読み分けの確認**：
  - `eval_context.tsv` の dev 行について、語ごとに全部の読みを当てたかを出す（「市場」を含む）。
  - 評価には、その時点の微調整モデル（ユーザーが作り直したもの）を使う。
- **閾値の見直し**：漢数字ルール・人名辞書のフォールバックの閾値を、dev の誤りの数で決める。
- **回帰**：`eval/run_regression.sh` を BERT モードで通す（今は外れなしなので、外れたら退行）。
- **速度**：実コーパス（351,728 字）で、BERT モードと w7 の µs/字、ロード時間、常駐メモリを測る。
  - int8 / f32 と、スレッド数 1・4・物理コア数の組み合わせで測る。
  - 見込みは int8・CPU で 150〜250 µs/字（w7 は 14.33 µs/字）。
  - 実測を見て次の打ち手を選ぶ。候補は、漢字位置だけを出力するグラフ・層の削減・DirectML。

## コスト見積もり（第1段階）

| Step | 人日 |
|---|---|
| 1 ONNX 書き出し・量子化・突き合わせ | 1 |
| 2 `momors-bert` | 2 |
| 3 core 組み込み | 2.5〜3 |
| 4 CLI / momo-inspect / pyo3 | 1 |
| 5 評価・閾値・速度 | 1〜1.5 |
| **合計** | **7.5〜8.5 人日**（GPU は使わない。モデルの作り直しはユーザーが行う） |

## 検証

1. `uv run export_onnx --name train`：PyTorch と ONNX で、dev 一致率と logit の差が出ること。
2. `cargo test -p momors-bert`：参照 logit と一致すること、長い行の区切りが正しいこと。
3. `MOMO_DATASET_DIR=... cargo test`：BERT なしの経路の既存テストが全部通ること（挙動が変わっていないこと）。
4. `momo --bert work/onnx/train` と `momo-inspect` で次を確かめる。
   - 「駅前の市場で魚を買った。」→ イチバ
   - 「株式市場が反発した。」→ シジョー
   - 「今日の中東情勢」
   - 々 の例
   - いずれも DecisionSource=Bert が出ること。
5. `step12_rust_eval`：dev で w7 との差が出ること。`eval/run_regression.sh` が BERT モードで通ること。
6. 実装の最初に、この計画を `docs/bert-plan.md`（今は空）に書き写す。

## 第1段階の結果（2026-10-27 実装）

### 実装したもの

| 場所 | 中身 |
|---|---|
| `momo-bert-train/src/momo_bert_train/export_onnx.py` | ONNX（float32 / int8）とラベル表・候補表・語彙・参照 logit の書き出し |
| `momors/crates/momors-core/src/slot_reader.rs` | trait `SlotReader`・`SlotScores`（core の依存は増えていない） |
| `momors/crates/momors-core/src/prediction.rs` | `Predictor::set_slot_reader`・`pick_reading`/`reader_argmax`・`DecisionSource::Bert`・`Phase::SlotReader` |
| `momors/crates/momors-bert/` | `BertReader`（ort 2.0.0-rc.15 = ONNX Runtime 1.31）・語彙・長い行の窓 |
| `momors-cli`（feature `bert`） | `momo` / `momo-inspect` に `--bert`（既定の場所 `<.mbm の既定ディレクトリ>/momo-bert/`、int8 版）`--bert-dir DIR` `--bert-f32` `--bert-threads N` |
| `momors-pyo3`（feature `bert`） | `Predictor(..., bert=DIR, bert_f32=, bert_threads=)`・`PredictionResult.decisions` |
| `momo-bert-train/src/momo_bert_train/step12_rust_eval.py` | Rust の推論経路で w7 と w7＋BERT を dev で比べる |
| `eval/run_regression.sh` | `MOMO_ARGS` で momo に追加の引数を渡せる |

### 計画からの変更

- **int8 は一部を float32 に残す。** 全部を int8 にすると dev が 97.78% → 97.43% に落ちる。出力層と最後の 2 層の FFN 出力（3072→768）を float32 に残すと 97.77% に戻る（106MiB）。前の層を残しても戻らない。
- **長い行の区切りは句点を探さない。** 510 字の窓を前後 64 字ずつ重ねて機械的に切り、重なりは隣の窓の中央側の結果を使う。文脈は重なりで確保できるので、句点を探す処理は省いた。
- **飛ばす文字のラベル。** BERT の学習データは、飛ばす文字（「三百二十」の「十」）の正解をその文字自身で表している。読み込み時に、そのラベルを `_` に読み替える。
- **モデルの既定の場所（2026-10-28）。** `--bert` だけで、.mbm を探す既定のディレクトリ（`MOMO_DATASET_DIR`、無ければ実行ファイルの場所）の下の `momo-bert/` を読む。配布用 zip を展開するとこのディレクトリができる。別の場所は `--bert-dir DIR`。`--model-file` で .mbm を別の場所から読んでも、既定の場所は変わらない。
- **feature `bert` を既定にした（2026-10-28）。** `momors-cli` と `momors-pyo3` は既定で `momors-bert` を取り込む。ONNX Runtime の配布バイナリが無いターゲット（Milk-V Duo の `riscv64gc-unknown-linux-musl` など）は `--no-default-features` で外す。
- **候補表に無い文字（学習データに出なかった漢字）は LR が読む。** Python の評価では全ラベルの argmax にしていたが、Rust では LR の制約なし argmax と同じ扱いにした。

### 精度（dev 957 行・漢字スロット 14,706）

w7 は train だけで学習したもの、BERT は `work/finetuned/train`（int8）。どちらも momo 本体の推論経路（助数詞・漢数字・人名辞書・「々」の規則・境界モデル込み）で測った。

| | 漢字スロット一致率 | 漢字が全部当たった行 |
|---|---|---|
| w7 | 95.48% | 53.4% |
| w7＋BERT | **97.74%** | **72.9%** |

- 読みが変わったスロットは 518。BERT が正解 402、w7 が正解 71、両方誤り 45。
- `eval_context.tsv` の dev 220 行（読み分けのある語 26）：行の一致 86.8% → 92.3%、全部の読みを当てた語 14 → 19。
- `eval/run_regression.sh large` は BERT モード（`work/onnx/full`）でも 53/53 通過。
- 変わったスロットの一覧は `momo-bert-train/work/rust_eval/train/changed.tsv`。

### 速度（i9-13900H、dev 38,004 字、1 行ずつ）

- BERT だけで 280〜500 µs/字（int8）。w7 は全体で約 26 µs/字。
- 測るたびに 2 割前後ぶれる。
- 既定のスレッド数（ONNX Runtime の既定＝物理 14 コア、E コア込み）より、`--bert-threads 6`（P コア数）の方が速いことが多かった。
- float32 は int8 の約 1.6〜2 倍遅い。
- Python の onnxruntime とほぼ同じ速さで、Rust 側の上乗せはほとんど無い。

### 残り

- 速度の次の打ち手：
  - 漢字位置だけを出力するグラフ
  - 文書変換のときに複数行をまとめて 1 回で流す
  - スレッド数の既定値を決める
  - DirectML
- 閾値の見直し：漢数字ルール・人名辞書のフォールバックは LR の sigmoid のときの 0.5 のまま。
- 第2段階（FFI・C#・エディタ・MSI）。
  - ort は ONNX Runtime を静的に組み込んでいて、`onnxruntime.dll` は出ない。
  - ビルド時に `DirectML.dll` が実行ファイルの隣にコピーされるので、配布に要るかどうかを確かめる。
    - CPU で動かすだけなら要らない（2026-10-28：`momo.exe` だけを置いたフォルダで `--bert` が動いた）。DirectML を使うときに改めて確かめる。
  - BERT 入りの `momo.exe` は、`VCRUNTIME140.dll` に加えて `MSVCP140.dll`・`MSVCP140_1.dll`・`VCRUNTIME140_1.dll`（どれも VC++ 再頒布可能パッケージ）を読む。MSI は今も再頒布可能パッケージを同梱していない。
