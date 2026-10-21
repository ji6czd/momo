"""実験の共通部分: データの場所と、tools/llm_reading の encoder_core の再利用。

比較を公平にするため、学習・dev の分割、読みのラベル表、辞書候補、漢字スロットの定義は
tools/llm_reading（微調整の実験）と同じものを使う。encoder_core は torch と transformers
だけで動くので、パスを足して取り込む（コピーして二重管理にしない）。

データは tools/llm_reading/06_export_bundle.py が書き出した bundle/（train_full.jsonl、
dev_full.jsonl。dev には w7 の読みと自信度が入っている）。
"""

from __future__ import annotations

import sys
from pathlib import Path

PKG_ROOT = (
    Path(__file__).resolve().parents[2]
)  # experimentals/bert（クラウドのキットでは展開先）
WORK = PKG_ROOT / "work"  # git 管理外

DEFAULT_MODEL = "tohoku-nlp/bert-base-japanese-char-v3"

if (PKG_ROOT / "encoder_core.py").exists():
    # クラウドに持ち込むキット（make_kit.py）: encoder_core と bundle が隣にある
    ENCODER_CORE_DIR = PKG_ROOT
    BUNDLE = PKG_ROOT / "bundle"
else:
    ENCODER_CORE_DIR = PKG_ROOT.parents[1] / "tools" / "llm_reading"
    BUNDLE = ENCODER_CORE_DIR / "work" / "gpu_kit" / "bundle"

sys.path.insert(0, str(ENCODER_CORE_DIR))
from encoder_core import (  # noqa: E402
    Candidates,
    CharVocab,
    Labels,
    ReadingModel,
    Rec,
    amp_mode,
    build_model,
    collate,
    make_examples,
    masked_logits,
    predict,
    train,
    is_target,
    load_jsonl,
    make_example,
    verify_records,
)

__all__ = [
    "ReadingModel",
    "build_model",
    "collate",
    "make_examples",
    "masked_logits",
    "predict",
    "train",
    "BUNDLE",
    "Candidates",
    "CharVocab",
    "DEFAULT_MODEL",
    "Labels",
    "Rec",
    "WORK",
    "amp_mode",
    "is_target",
    "load_jsonl",
    "make_example",
    "verify_records",
]
