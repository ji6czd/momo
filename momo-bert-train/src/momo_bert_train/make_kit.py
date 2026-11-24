"""クラウドの GPU に持ち込むキット（zip）を作る（手元で実行）。

    uv run make_kit

basic_raw.tsv から学習データ（work/data/）を作り直してから詰めるので、キットは常に最新のデータになる。
中身: pyproject（torch・transformers・numpy）、src/momo_bert_train/（common・encoder_core・step9_finetune）、
work/data/（train_full.jsonl・dev_full.jsonl・meta.json）、train_all.sh（組ごとにまとめて回す）。
出力: work/bert_kit.zip。作ったあとの手順も表示する。
"""

import zipfile

from .common import DATA, PKG_ROOT, WORK
from .export_data import export

PKG = "momo_bert_train"  # キットに入れるパッケージ（src/ の下）

PYPROJECT = """[project]
name = "momo-bert-train"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = ["torch", "transformers", "numpy"]

[project.scripts]
step9_finetune = "momo_bert_train.step9_finetune:main"

[build-system]
requires = ["uv_build>=0.13.0,<0.14.0"]
build-backend = "uv_build"
"""

STEPS = """
クラウドで:
  unzip bert_kit.zip -d bert_kit && cd bert_kit
  curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
  uv sync
  bash train_all.sh large      # 今回: bert-large-japanese-char-v2 を char-v3 と同じデータで比べる 9 本（動作確認のあと。
                               #   L4 で 3〜5 時間の見込み。役に立つ順に回すので、途中で止めても先の分で比べられる。
                               #   途中保存と重みでディスクを 15GB ほど使うので、始める前に df -h で空きを見る）
                               #   large_train_joint_frag_unit         large・学習率 2e-5・seed 0（重みあり）
                               #   train_joint_frag_unit               char-v3・seed 0（今のデータで取り直し。重みあり）
                               #   large_train_joint_frag_unit_lr1e-5  学習率 1e-5（重みなし）
                               #   large_train_joint_frag_unit_lr3e-5  学習率 3e-5（重みなし）
                               #   large_train_joint_frag_unit_s1/_s2  シードの取り直し（重みなし）
                               #   train_joint_frag_unit_s1/_s2        char-v3 のシードの取り直し（重みなし）
                               #   large_full_joint_frag_unit          配る形（全データ・重みあり。回帰と速さを測る）
  bash train_all.sh seed       # --seed 1 の取り直し 3 本で揺れを見る
                               #   train_joint_frag_unit_s1  unit の断片（seed 0 の train_joint_frag_unit と比べる）
                               #   train_joint_s1            断片なし（seed 0 の train_joint と比べる）
                               #   full_joint_frag_unit_s1   全データ（回帰の「AI点訳」などの揺れ）
  bash train_all.sh unit       # マスあけの単位で切った断片の 2 本
                               #   train_joint_frag_unit  dev で比べる用
                               #   full_joint_frag_unit   配る用（全データ）
  bash train_all.sh collapse   # 隠れ状態の潰れを確かめる 2 本（語の途中でも切る free の断片）
                               #   train_joint_frag_s1   --seed 1（前回の train_joint_frag の取り直し）
                               #   train_joint_frag_last 断片の最後の切れ目も学習（--fragment-last-gap）
  bash train_all.sh            # 6 本: train_joint_frag_unit・full_joint_frag_unit・train・full・train_joint・full_joint
1 本ずつなら: uv run step9_finetune [--model 元のモデル] [--lr 学習率] [--no-weights] [--all-data] [--boundary [--fragments 1 [--fragment-cut free [--fragment-last-gap]]]]
プリエンプトされたら: VM を起動し直し、同じコマンドを打ち直すと続きから学習する（済んだものは飛ばす）
手元へ: finetuned.tar を展開し、finetuned/ の中身を momo-bert-train/work/finetuned/ に置く"""


def main() -> None:
    meta = export()
    c = meta["counts"]
    print(f"データを作り直した: train {c['train']} / dev {c['dev']}")
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / "bert_kit.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("pyproject.toml", PYPROJECT)
        for name in (
            "__init__.py",
            "common.py",
            "encoder_core.py",
            "step9_finetune.py",
        ):
            z.write(PKG_ROOT / "src" / PKG / name, f"src/{PKG}/{name}")
        for name in ("train_full.jsonl", "dev_full.jsonl", "meta.json"):
            z.write(DATA / name, f"work/data/{name}")
        # Linux で動かすので、改行は LF にそろえる
        script = (PKG_ROOT / "train_all.sh").read_text(encoding="utf-8")
        z.writestr("train_all.sh", script.replace("\r\n", "\n"))
    print(f"{out}（{out.stat().st_size / 1e6:.1f} MB）")
    print(STEPS)


if __name__ == "__main__":
    main()
