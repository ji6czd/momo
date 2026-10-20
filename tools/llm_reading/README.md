# LLM 読み推定の前段（フェーズ A）

LLM をファインチューンして読み推定に使う前に、手元だけで「何が律速か」を測る診断スクリプト群。
結果は LLM をやるかどうかに関係なく、`basic_raw.tsv` の育て方の指針になる。

- **学習曲線**：データ量を変えたときの伸び方を見る。まだ伸びていればデータが律速。
- **誤りの分解**：誤りを、辞書の候補内での選び損ねと、辞書にない読みに分ける。
- **分け方の監査**：同じ漢字列の分け方（トー/キョー と トーキョー/---）が揺れていないかを見る。

## 準備

uv プロジェクト。`momo-py`（正解の展開に `process_line_to_tsv` を使う）と `momors-py`
（`../../momors/crates/momors-pyo3/dist/` の wheel。推論と `kana_to_source` に使う）が入る。
Rust 側を変えたら、wheel を作り直してから lock を更新する。

```sh
task pyo3                             # リポジトリ直下で
uv lock --refresh-package momors-py   # このディレクトリで
uv sync
```

## 手順

```sh
uv run python 00_check_harness.py     # 評価の仕組みの確認（回帰セットの通過数が run_regression.sh と一致するか）
uv run python 01_split.py             # A1: train/dev 分割（seed 固定）
uv run python 02_learning_curve.py    # A2: 25/50/75/100% で w7 を学習して dev で評価（学習に時間がかかる）
uv run python 03_error_breakdown.py   # A3: 誤りを (a) 選び損ね / (b) 辞書にない / (c) まとまり に分解
uv run python 04_split_audit.py       # A4: 分け方の揺れの一覧
```

中間ファイルと結果は `work/`（git 管理外）に出る。

dev は原文のハッシュで振り分ける（約1割）。今ある文の振り分けは変わらず、足した文も同じ規則で約1割が dev に入る。ある時点の dev は `01_split.py --snapshot <名前>` で `splits/` に残す（git 管理下）。dev を外すのは実験用のモデル（`work/frac_*`）だけで、momo 本体は全データで学習する。分割を作り直したら、実験用のモデルも `02_learning_curve.py --force` で学習し直すこと。

## LLM に候補から読みを選ばせる実験（フェーズ B の予備）

`llm` グループ（torch・transformers など）を入れて使う（`uv sync --group llm`）。

```sh
uv run --group llm python 05_llm_select.py --text "上手に入るとは…"           # 任意の文（手元の CPU）
uv run --group llm python 05_llm_select.py --dev 100 --dev-offset 100        # dev の短い文で w7 と比較
uv run --group llm python 06_export_bundle.py                                # クラウド用キット work/gpu_kit.zip を作る
```

- `llm_core.py`：選び方の本体（torch と transformers だけで動く。05 と 07 が共有）。
- `07_gpu_eval.py`：キットの中で、複数モデルを順に回して `results/summary.tsv` にまとめる。手順は `gpu_kit/README.md`。
- `probe_context.tsv`：文脈で読み分ける確認用の文（期待値は注釈者が確認済み）。
- `09_eval_context.py`：読み分けの評価データ `dataset/eval_context.tsv` の作成（`bootstrap`、一度だけ）と検査（`check`）。形式・種別・決め手・学習データとの関係は、このスクリプトの冒頭に書いてある。例文は `eval_context.tsv` に直接足し、足したら `check` を通す。
- `10_encoder_train.py` / `encoder_core.py`：エンコーダ（BERT 系）を train で微調整し、漢字スロットの読みを辞書候補から選ばせて w7 と比べる（`--window N` で前後 N 文字だけに絞る切り分け）。GPU キットに同梱。`11_encoder_check.py` は学習前の整合性検査（位置・復号・辞書）。

## 指標

正解は `basic_raw.tsv` の1行を学習データと同じ行（スロット）に展開したもの。予測のかなは
`kana_to_source` で原文位置へ戻し、同じスロットに集めて比べる。

| 指標 | 意味 |
|---|---|
| `kanji_slot_acc` | 漢字スロットの一致率。読みと分け方の両方が合って一致（主指標） |
| `slot_acc` | 全スロットの一致率 |
| `concat_reading_acc` | 文単位で、連結した読みが一致する割合（分け方の違いを無視） |
| `sentence_exact` | 文単位で、全スロットとマスあけが一致する割合 |
| `boundary_p/r/f1` | マスあけの適合率・再現率・F1 |
| `kanji_cont_gold/pred` | 漢字スロットのうち継続（読みが空）の割合。予測が正解より多ければまとめすぎ |
