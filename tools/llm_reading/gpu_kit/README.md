# momo LLM 評価キット（クラウドの GPU 用）

LLM に、1文字ごとの読みの候補から漢字の読みを選ばせて、w7 と比べる。複数のモデルを順に回し、
結果を `results/summary.tsv` にまとめる。手元の `tools/llm_reading/06_export_bundle.py` が作る。

## 中身

| ファイル | 内容 |
|---|---|
| `07_gpu_eval.py` | 評価の実行 |
| `llm_core.py` | 候補から選ぶ本体（torch と transformers だけで動く） |
| `bundle/` | dev の正解・w7 の読みと自信度・辞書・手本・確認用の文 |

## 測るもの

- **short**：dev の短い文。w7 単独・LLM 単独・w7 の自信度 0.5 未満だけ LLM、の一致率。
- **context**：dev の段落の2文目以降。同じ段落の前の文を置いた場合と置かない場合で、LLM の一致率を比べる。**文書の文脈が効くかどうか**の本題。
- **context_train**：同じものを train の段落から（dev は段落が少ないので数を補う）。w7 は学習済みなので w7 系の数字は参考にならない。見るのは LLM の文脈あり 対 なし。
- **probe**：確認用の文。期待値は注釈者が確認済み。

## Compute Engine での手順

1. GPU の割り当て（`GPUS_ALL_REGIONS` と、使う GPU の種類の枠）が承認済みであることを確かめる。
2. Deep Learning VM のイメージ（CUDA とドライバ入り）を探す。

   ```sh
   gcloud compute images list --project deeplearning-platform-release \
     --filter="family~common-cu" --format="value(family)" | sort -u
   ```

3. VM を作る。下の例は A100 80GB の Spot VM（33B まで全部回せる）。L4 で 8B までなら `--machine-type=g2-standard-8`。
   ディスクは、全部のモデルを残すなら 300GB 程度要る。`--purge-cache` で回すなら 100GB で足りる。

   ```sh
   gcloud compute instances create momo-llm \
     --zone=<ゾーン> --machine-type=a2-ultragpu-1g \
     --provisioning-model=SPOT --instance-termination-action=STOP \
     --maintenance-policy=TERMINATE \
     --image-project=deeplearning-platform-release --image-family=<手順2で選んだもの> \
     --boot-disk-size=300GB --metadata=install-nvidia-driver=True
   ```

4. キットを送り、VM に入る。

   ```sh
   gcloud compute scp work/gpu_kit.zip momo-llm:~ --zone=<ゾーン>
   gcloud compute ssh momo-llm --zone=<ゾーン>
   ```

5. VM の中で、uv を入れて準備し、動作確認してから本番を回す。接続が切れても止まらないよう `tmux` の中で回す。

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
   unzip gpu_kit.zip && cd gpu_kit
   nvidia-smi                                   # GPU が見えること
   uv sync
   uv run python 07_gpu_eval.py --smoke         # 0.5B・各数件で数分
   tmux new -s momo
   uv run python 07_gpu_eval.py --preset a100   # L4 なら --preset l4
   ```

   済んだモデルは飛ばすので、Spot VM が止められても、起動し直して同じコマンドで続きから回せる。

6. 結果を手元に持ち帰る（手元のリポジトリ直下で）。

   ```sh
   gcloud compute scp --recurse momo-llm:~/gpu_kit/results tools/llm_reading/work/gpu_results --zone=<ゾーン>
   ```

7. **VM を削除する**（停止だけではディスクの料金がかかり続ける）。

   ```sh
   gcloud compute instances delete momo-llm --zone=<ゾーン>
   ```

## オプション

| オプション | 意味 |
|---|---|
| `--models A B …` | 回すモデルを直接指定（`--preset` と併用可） |
| `--n-short N` / `--n-context N` / `--n-context-train N` | 使う件数（既定 200 / 200 / 300） |
| `--batch N` | 候補を一括採点する行数（既定 16。GPU メモリが足りなければ下げる） |
| `--dtype float32` | 手元の CPU の結果と厳密に比べたいとき |
| `--purge-cache` | モデルごとに Hugging Face のキャッシュを消す（ディスク節約） |
| `--force` | 済んだモデルも回し直す |

## エンコーダの読み推定（10_encoder_train.py）

日本語エンコーダ（既定は `tohoku-nlp/bert-base-japanese-char-v3`、文字単位・Apache 2.0）を train の全行で微調整し、漢字スロットの読みを辞書候補から選ばせて、w7 と比べる。LLM の評価（上の `07_gpu_eval.py`）とは別の実験で、同じキット・同じ VM で回せる。
学習データの割合（train 先頭 N 行）× シードで回し、dev の漢字スロット一致率・w7 の自信度ゲート・文脈（probe と `eval_context`）を測る。

| ファイル | 内容 |
|---|---|
| `10_encoder_train.py` | 学習と評価の実行 |
| `encoder_core.py` | ラベル表・辞書絞り込み・トークン化・分類ヘッド・学習ループ |
| `bundle/train_full.jsonl` `dev_full.jsonl` `eval_context.jsonl` | 学習用の全行・dev 全行（w7 つき）・読み分けの評価データ |

```sh
uv sync
uv run python 10_encoder_train.py --smoke                          # 極小モデル・数十行で動作確認（1分未満）
tmux new -s momo-enc
uv run python 10_encoder_train.py --fracs 25,50,75,100 --seeds 0,1,2   # 本番（8 エポック固定）
```

- 精度は GPU に合わせて自動で選ぶ（bf16 が使えれば bf16、T4 のように使えなければ fp16 の混合精度）。
- 済んだ回（`results_encoder/<モデル>/f<割合>_s<シード>/result.json`）は飛ばすので、Spot VM が止められても同じコマンドで続きから回せる。`--force` で回し直す。
- 集計は `results_encoder/<モデル>/summary.tsv`（回ごと）と `summary_agg.tsv`（割合ごとの平均±標準偏差）。`--aggregate-only` で集計だけやり直せる。
- 結果の持ち帰り（手元のリポジトリ直下で）:

  ```sh
  gcloud compute scp --recurse momo-llm:~/gpu_kit/results_encoder tools/llm_reading/results_encoder --zone=<ゾーン>
  ```

| オプション | 意味 |
|---|---|
| `--model NAME` | 微調整するモデル（文字単位の語彙 `vocab.txt` を持つもの） |
| `--fracs` / `--seeds` | 学習データの割合（%）とシード（既定 25,50,75,100 と 0,1,2） |
| `--epochs N` / `--lr X` / `--batch-size N` | 既定 8 / 5e-5 / 32 |
| `--limit-train N` `--limit-dev N` `--limit-eval N` | 予備実験用に件数を絞る |

### 窓を絞った切り分け（`--window N`）

エンコーダの勝因が「行全体の文脈」かを確かめるため、スロットごとに前後 N 文字だけを見せて学習・評価する（w7 の window=7 は前後 3 文字）。結果は `results_encoder/<モデル>__win<N>/` に分かれる。
1 スロットが 1 件の短い入力になるので、バッチは行ではなくスロット単位。行全体の学習と1ステップあたりのスロット数を合わせるため `--batch-size 256` を使う。

```sh
for w in 3 8 16; do uv run python 10_encoder_train.py --window $w --batch-size 256 --fracs 100 --seeds 0,1,2; done
```
