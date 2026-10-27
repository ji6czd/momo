"""クラウドの GPU に持ち込むキット（zip）を作る（手元で実行）。

    uv run make_kit

basic_raw.tsv から学習データ（work/data/）を作り直してから詰めるので、キットは常に最新のデータになる。
中身: pyproject（torch・transformers・numpy）、src/momo_bert_train/（common・encoder_core・step9_finetune）、
work/data/（train_full.jsonl・dev_full.jsonl・meta.json）。出力: work/bert_kit.zip。作ったあとの手順も表示する。
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
  uv run step9_finetune --smoke            # 動作確認
  uv run step9_finetune                    # train で学習・dev で評価
  uv run step9_finetune --all-data --name full
手元へ: work/finetuned/<名前>/ を momo-bert-train/work/finetuned/ に置く"""


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
    print(f"{out}（{out.stat().st_size / 1e6:.1f} MB）")
    print(STEPS)


if __name__ == "__main__":
    main()
