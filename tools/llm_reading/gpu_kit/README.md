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
