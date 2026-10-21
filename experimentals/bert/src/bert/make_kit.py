"""クラウドの GPU に持ち込むキット（zip）を作る（手元で実行）。

    uv run python -m bert.make_kit

中身: pyproject（torch・transformers・numpy）、src/bert/（common・step9_finetune）、encoder_core.py、
bundle/（train_full.jsonl・dev_full.jsonl）。出力: work/bert_kit.zip。作ったあとの手順も表示する。
"""

import zipfile

from .common import BUNDLE, ENCODER_CORE_DIR, PKG_ROOT, WORK

PYPROJECT = """[project]
name = "bert"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["torch", "transformers", "numpy"]

[project.scripts]
step9_finetune = "bert.step9_finetune:main"

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
手元へ: work/finetuned/<名前>/ を experimentals/bert/work/finetuned/ に置く"""


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / "bert_kit.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("pyproject.toml", PYPROJECT)
        z.write(ENCODER_CORE_DIR / "encoder_core.py", "encoder_core.py")
        for name in ("__init__.py", "common.py", "step9_finetune.py"):
            z.write(PKG_ROOT / "src" / "bert" / name, f"src/bert/{name}")
        for name in ("train_full.jsonl", "dev_full.jsonl"):
            z.write(BUNDLE / name, f"bundle/{name}")
    print(f"{out}（{out.stat().st_size / 1e6:.1f} MB）")
    print(STEPS)


if __name__ == "__main__":
    main()
